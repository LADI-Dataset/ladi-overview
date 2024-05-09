import config_dataloader
import config_dataset
import config_model
import config_optimizer
import config_scheduler
import config_train

from dataclasses import dataclass
from hydra_zen import store, make_config
from typing import Callable
from pathlib import Path

def register_all_configs(train_fn: Callable,
                         hf_models_dir: str,
                         finetuned_models_dir: str,
                         data_base_dir: str):
    default_task_conf = make_config(hydra_defaults = ['_self_', 
                                                    {'dataset':'ladi_v2_resized'},
                                                        {'model_builder':'google-vit-base-patch16-224-in21k'},
                                                        {'dataloader_builder':'default'},
                                                        {'optimizer_builder':'adafactor'},
                                                        {'scheduler_builder':'none'},
                                                        {'train_builder':'default'},],
                                    dataset=None,
                                    model_builder=None,
                                    dataloader_builder=None,
                                    optimizer_builder=None,
                                    scheduler_builder=None,
                                    train_builder=None)

    store(default_task_conf, name='default')

    store.add_to_hydra_store()
    
    config_dataloader.register_configs()
    config_dataset.register_configs(data_base_dir)
    config_model.register_configs(hf_models_dir, finetuned_models_dir)
    config_optimizer.register_configs()
    config_scheduler.register_configs()
    config_train.register_configs(train_fn)