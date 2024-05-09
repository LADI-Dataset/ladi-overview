import configs
import datetime
import json
import numpy as np
import os
import pandas as pd
import torch

from accelerate import Accelerator
from pathlib import Path
from accelerate.scheduler import AcceleratedScheduler
from hydra.conf import HydraConf, RunDir
from hydra_zen import zen, to_yaml, store, builds
from omegaconf import OmegaConf
from sklearn.metrics import (roc_auc_score,
                             accuracy_score,
                             precision_recall_fscore_support,
                             average_precision_score)
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

HF_MODELS_DIR = os.environ['HF_HUB_CACHE'] if 'HF_HUB_CACHE' in os.environ else '~/.cache/huggingface/hub'
FINETUNED_MODELS_DIR = './finetuned_models/'
LADI_DATA_DIR='./ladi_dataset'
USE_MLFLOW = True

if USE_MLFLOW:
    accelerator = Accelerator(log_with='mlflow')
else:
    accelerator = Accelerator()

def multi_label_metrics(logits, y_true, labels, threshold=0.5):
    '''
    calculates a number of metrics, both mean metrics over all labels, and label-specific
    '''
    # print(labels)
    probs = torch.sigmoid(logits)
    y_pred = probs > threshold
    # print(y_true)
    accuracy = accuracy_score(y_true, y_pred)
    mean_prec, mean_rec, mean_f1, _ = precision_recall_fscore_support(y_true=y_true, y_pred=y_pred, average='weighted', zero_division=np.nan)
    precs, recs, f1s, supports = precision_recall_fscore_support(y_true=y_true, y_pred=y_pred, average=None, zero_division=np.nan)
    mean_ap = average_precision_score(y_true, probs, average='weighted')
    aps = average_precision_score(y_true, probs, average=None)
    # print(f'{precs=}\n{recs=}\n{f1s=}\n{aps=}')
    
    def convert_to_labeled_dict(metric, metric_name):
        '''
        takes metric array and converts it into a dictionary 
        whose key is metric_name.label, 
        and the value is the metric for that label
        '''
        return {f'{metric_name}.{l}':m for l,m in zip(labels, metric)}
    
    labeled_precs = convert_to_labeled_dict(precs, 'precision')
    labeled_recs = convert_to_labeled_dict(recs, 'recall')
    labeled_f1s = convert_to_labeled_dict(f1s, 'f_1')
    labeled_aps = convert_to_labeled_dict(aps, 'AP')

    # return as dictionary
    metrics = {'mean_f1': mean_f1,
            'mean_accuracy': accuracy,
            'mean_precision': mean_prec,
            'mean_recall': mean_rec,
            'mean_AP': mean_ap}\
                | labeled_precs\
                | labeled_recs\
                | labeled_f1s\
                | labeled_aps
    return metrics

def evaluate(model, 
        dataloader, 
        split_name, 
        epoch, 
        completed_steps, 
        labels):
    '''
    computes validation metrics for given split
    '''
    logit_list = []
    ref_list = []
    model.eval()
    for batch in dataloader:
        with torch.no_grad():
            outputs = model(**batch)
        logits = outputs.logits
        logits, references = accelerator.gather_for_metrics((logits, batch["labels"]))
        logit_list.append(logits)
        ref_list.append(references)
    logit_list = torch.vstack(logit_list).cpu()
    ref_list = torch.vstack(ref_list).cpu()
    metrics = multi_label_metrics(logit_list, ref_list, labels)
    metrics = {f'{split_name}.{k}':v for k,v in metrics.items()}
    metrics['epoch'] = epoch
    accelerator.log(metrics, step=completed_steps)

def train(model,
          image_processor,
          model_output_dir,
          labels,
          optimizer,
          loss_func,
          train_dataloader,
          val_dataloader=None,
          test_dataloader=None,
          epochs=100,
          checkpoint_every=5,
          lr_scheduler=None):
    """
    Trains the given model on the data in train_dataloader, and optionally evaluates
    every epoch on the data in val_dataloader and test_dataloader
    """
    completed_steps = 0
    progress_bar = tqdm(range(epochs), disable=not accelerator.is_local_main_process)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_dataloader:
            inputs = batch['pixel_values']
            truth_labels = batch['labels']
            outputs = model(inputs)
            loss = loss_func(outputs.logits, truth_labels) + 0. * sum(p.sum() for p in model.parameters())
            total_loss += loss.detach().float()
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
        if lr_scheduler:
            # using ReduceLROnPlateau requires passing in a metric for reduction
            if isinstance(lr_scheduler, ReduceLROnPlateau) or \
                (isinstance(lr_scheduler, AcceleratedScheduler) and isinstance(lr_scheduler.scheduler, ReduceLROnPlateau)):
                lr_scheduler.step(total_loss)
            else:
                lr_scheduler.step()
        accelerator.log({'total_train_loss': float(total_loss)}, step=completed_steps)

        if val_dataloader is not None:
            evaluate(model, val_dataloader, 'validation', epoch, completed_steps, labels)
        if test_dataloader is not None:
            evaluate(model, test_dataloader, 'test', epoch, completed_steps, labels)
        if (epoch % checkpoint_every == 0) or (epoch == epochs-1):
            output_dir = model_output_dir/f'epoch_{epoch:03}'
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
            )
            if accelerator.is_main_process:
                image_processor.save_pretrained(output_dir)
        if accelerator.sync_gradients:
            progress_bar.update(1)
            completed_steps += 1
                
    if USE_MLFLOW and accelerator.is_main_process:
        import mlflow
        mlflow.transformers.log_model({'model': unwrapped_model,
                                       'image_processor': image_processor},
                                        artifact_path=f'model/epoch_{epoch}',
                                        task='image-classification')

#####################
# Hydrazen task logic
#####################

def get_model_output_dir(model_id, run_id):
    '''
    gives the output path for corresponding model_id and run_id
    '''
    finetuned_models_dir = Path(FINETUNED_MODELS_DIR)
    return finetuned_models_dir/f'model_{model_id}'/f'run_{run_id}'


def get_timestamped_run_id():
    '''
    generates a timestamped run id using current timestamp and run_id from environment
    '''
    return datetime.datetime.now().strftime('%Y%m%d-%H%M%S')+'_'+os.environ['run_id']

def task(dataset,
         model_builder,
         dataloader_builder,
         optimizer_builder,
         scheduler_builder,
         train_builder,
         zen_cfg):
    """
    Main entrypoint - gets the dataset, model, optimizer, etc from their builders
    and begins the training process
    """
    dataset, labels, class_weights = dataset
    image_processor, model, model_id = model_builder(labels=labels)
    dataloaders = dataloader_builder(dataset,
                          image_processor,
                          labels)
    optimizer = optimizer_builder(model)
    
    # flatten the config for logging
    config_dict = pd.json_normalize(OmegaConf.to_container(zen_cfg), sep='.').iloc[0].to_dict()
    # filter out keys we don't care about 
    config_dict = {k:v for k,v in config_dict.items() if not any([q in k for q in ['_target_','_partial_','zen_cfg']])}
    
    train_split = config_dict["dataset.train_split"]
    train_dataloader = dataloaders[train_split]
    val_dataloader = dataloaders['validation']
    test_dataloader = dataloaders['test']

    run_id = store['hydra','config'].run.dir.split('/')[1]
    model_output_dir = get_model_output_dir(model_id, run_id)
    tracker_params={**config_dict, 
                'accelerator.num_processes':accelerator.num_processes,
                'accelerator.gradient_accumulation_steps':accelerator.gradient_accumulation_steps,
                'model.run_id':run_id,
                'model.model_id':model_id,
                'model.output_dir':str(model_output_dir),
                }
    if 'lr' in optimizer.defaults:
        tracker_params['optimizer_builder.lr']  = optimizer.defaults['lr']
    accelerator.init_trackers(
        project_name=os.environ['experiment_name'],
        config=tracker_params,
        init_kwargs={
            'mlflow': {
                'run_name': f'{model_id}/{run_id}'
            }
        }
    )
    
    model, optimizer, train_dataloader, val_dataloader, test_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, test_dataloader
    )
    lr_scheduler = scheduler_builder(optimizer)
    if lr_scheduler:
        lr_scheduler = accelerator.prepare(lr_scheduler)
    loss_func = torch.nn.BCEWithLogitsLoss(pos_weight=torch.from_numpy(class_weights.values).cuda())
    total_batch_size = train_dataloader.batch_size * accelerator.num_processes * accelerator.gradient_accumulation_steps
    
    try:    
        train_builder(model,
                image_processor,
                model_output_dir,
                labels,
                optimizer,
                loss_func,
                train_dataloader,
                val_dataloader=val_dataloader, 
                test_dataloader=test_dataloader,
                lr_scheduler=lr_scheduler)
    except Exception as e:
        accelerator.end_training()
        raise e
    else:
        accelerator.end_training()
    finally:
        accelerator.end_training()


def debug_task(dataset,
         model_builder,
         dataloader_builder,
         optimizer_builder,
         scheduler_builder,
         train_builder,
         zen_cfg):
    print(dataset)
    # print(store)
    if accelerator.is_main_process:
        print(to_yaml(zen_cfg))
    config_dict = pd.json_normalize(OmegaConf.to_container(zen_cfg), sep='.').iloc[0].to_dict()
    config_dict = {k:v for k,v in config_dict.items() if not (any([q in k for q in ['_target_','_partial_','zen_cfg']])
                                                              or (v is None))}
    # print(config_dict)
    
    print(store['hydra','config'].run.dir)
        
if __name__ == '__main__':
    if 'run_id' not in os.environ.keys():
        raise ValueError('the environment variable run_id needs to be set')
    if 'experiment_name' not in os.environ.keys():
        print('the environment variable experiment_name is not set, will be set to default: cap_ai_training_explore')
        os.environ['experiment_name'] = 'cap_ai_training_explore'

    
    store(HydraConf(run=RunDir(f'outputs/{get_timestamped_run_id()}')))
    store.add_to_hydra_store()

    configs.register_all_configs(train, HF_MODELS_DIR, FINETUNED_MODELS_DIR, LADI_DATA_DIR)
    zen(task).hydra_main(config_name='default', config_path=None, version_base='1.3')