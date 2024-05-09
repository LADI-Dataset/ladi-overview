import albumentations as A
import torch
import numpy as np

from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2
from hydra_zen import builds, store, MISSING
from functools import partial

from datasets import Dataset
from transformers import AutoImageProcessor

def get_dataloaders(dataset: Dataset,
                    image_processor: AutoImageProcessor,
                    labels,
                    splits:list[str],
                    per_device_batch_size:int,
                    num_workers_per_process:int):
    """
    This function is internal to this file - it is exposed via global state by
    means of the Hydra store below.

    Returns a dictionary mapping split names to appropriate torch.DataLoaders.
    DataLoaders will properly convert the formatting of labels and apply any
    training augmentations as part of the preprocessing. Parameters from image_processor 
    will be used to properly resize and normalize the images.
    """
    dataloaders = {}
        
    if "shortest_edge" in image_processor.size:
        size = (image_processor.size["shortest_edge"], image_processor.size["shortest_edge"])
    else:
        size = (image_processor.size["height"], image_processor.size["width"])
    normalize = A.Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    
    def preprocess(example_batch, transforms):
        """Apply transforms across a batch, also converts labels from key:bool to a vector of floats"""
        images = example_batch["image"]
        # albumentations expects image as np.array, so we cast it to array first
        example_batch["pixel_values"] = [transforms(image=np.array(image))['image']
                                         for image in example_batch["image"]]
        labels_batch = {k: example_batch[k] for k in example_batch.keys() if k in labels}
        labels_matrix = np.zeros((len(images), len(labels)))
        for idx, label in enumerate(labels):
            labels_matrix[:, idx] = labels_batch[label]
        example_batch["labels"] = labels_matrix.tolist()    
        return example_batch
    
    def collate_fn(examples):
        """
        converts a batch into single set of tensors
        """
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.tensor([example["labels"] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels}
    
    train_transforms = A.Compose([
            A.RandomResizedCrop(*size, scale=(0.8, 1)),
            A.HorizontalFlip(),
            A.ColorJitter(),
            normalize,
            ToTensorV2()
        ])
    
    val_transforms = A.Compose([
            A.Resize(*size),
            A.CenterCrop(*size),
            normalize,
            ToTensorV2(),
        ])
    
    # add augmentations
    if 'train' in splits:
        preprocess_train = partial(preprocess, transforms=train_transforms)
        train_dataset = dataset['train'].map(preprocess_train,
                                             batch_size=1,
                                             batched=True,
                                             remove_columns=dataset['train'].column_names)
        dataloaders['train'] = DataLoader(train_dataset,
                                          collate_fn=collate_fn,
                                          batch_size=per_device_batch_size,
                                          num_workers=num_workers_per_process
                                         )
    if 'all' in splits:
        all_transforms = train_transforms
        preprocess_all = partial(preprocess, transforms=all_transforms)
        all_dataset = dataset['all'].map(preprocess_all,
                                         batch_size=1,
                                         batched=True,
                                         remove_columns=dataset['all'].column_names)
        dataloaders['all'] = DataLoader(all_dataset,
                                          collate_fn=collate_fn,
                                          batch_size=per_device_batch_size,
                                          num_workers=num_workers_per_process
                                         )
    if 'validation' in splits:
        preprocess_val = partial(preprocess, transforms=val_transforms)
        val_dataset = dataset['validation'].map(preprocess_val,
                                                batch_size=1,
                                                batched=True,
                                                remove_columns=dataset['validation'].column_names)
        dataloaders['validation'] = DataLoader(val_dataset,
                                          collate_fn=collate_fn,
                                          batch_size=per_device_batch_size,
                                          num_workers=num_workers_per_process
                                         )
    if 'test' in splits:
        test_transforms = val_transforms
        preprocess_test = partial(preprocess, transforms=test_transforms)
        test_dataset = dataset['test'].map(preprocess_test,
                                           batch_size=1,
                                           batched=True,
                                           remove_columns=dataset['test'].column_names)
        dataloaders['test'] = DataLoader(test_dataset,
                                          collate_fn=collate_fn,
                                          batch_size=per_device_batch_size,
                                          num_workers=num_workers_per_process
                                         )
    return dataloaders

## dataloaders config
def register_configs():
    """
    Registers configurations for dataloaders with the Hydra store
    """
    dataloader_builder = builds(get_dataloaders,
                splits=MISSING,
                per_device_batch_size=16,
                num_workers_per_process=20,
                zen_partial=True)
    default_dataloader_config = dataloader_builder(splits=['train','validation','test', 'all'])
    full_training_dataloader_config = dataloader_builder(splits=['all'])

    store_dl = store(group="dataloader_builder")
            
    store_dl(default_dataloader_config,
        name='default')

    store_dl(full_training_dataloader_config,
        name='full')

    store_dl.add_to_hydra_store()