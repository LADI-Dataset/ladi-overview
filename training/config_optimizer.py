import transformers
import mlflow
import torch

from typing import Optional, Literal
from hydra_zen import builds, store, MISSING

def get_optimizer(model,
                  optimizer_name: Literal['adafactor', 'adamw'],
                  lr: Optional[float],
                  weight_decay: float,
                  **kwargs
                 ):
    """
    This function is internal to this file - it is exposed via global state by
    means of the Hydra store below.

    Returns an optimizer for the model with the given lr and weight decay. kwargs
    are passed into the optimizer constructor.
    """
    no_decay = ["bias", "LayerNorm.weight"]
    # construct optimizer such that bias and LayerNorm weights are not regularized
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    if optimizer_name.lower() == "adafactor":
        return transformers.Adafactor(optimizer_grouped_parameters, lr=lr, **kwargs)
    elif optimizer_name.lower() == "adamw":
        if lr == None:
            lr = 0.001
        return torch.optim.AdamW(optimizer_grouped_parameters, lr=lr, **kwargs)
    else:
        raise ValueError('only Adafactor and AdamW are currently supported optimizers')

def register_configs():
    """
    Registers configurations for optimizers with the Hydra store
    """
    optimizer_builder = builds(get_optimizer,
                            optimizer_name=MISSING,
                            lr=None,
                            weight_decay=0.01,
                            zen_partial=True)
    adafactor_config = optimizer_builder(optimizer_name='adafactor',)
    adamw_conf = optimizer_builder(optimizer_name='adamw',)

    store(adafactor_config,
        name='adafactor',
        group='optimizer_builder')
            
    store(adamw_conf,
        name='adamw',
        group='optimizer_builder')

    store.add_to_hydra_store()