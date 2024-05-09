import torch

from hydra_zen import builds, store, MISSING, make_config
from typing import Union

def get_scheduler(optimizer,
                  scheduler_name: Union[str,None]=None,
                  **kwargs):
    """
    This function is internal to this file - it is exposed via global state by
    means of the Hydra store below.

    Returns requested LR scheduler, constructed with kwargs
    """
    if not scheduler_name:
        return None
    if str(scheduler_name).lower() == 'exponential':
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, **kwargs)
    elif str(scheduler_name).lower() == 'reducelronplateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)
    elif scheduler_name:
        raise NotImplementedError('only Exponential and ReduceLROnPlateau are currently supported schedulers. Leave lr_scheduler=None for constant LR')

## Scheduler configs
def register_configs():
    """
    Registers configurations for LR schedulers with the Hydra store
    """
    scheduler_builder = builds(get_scheduler,
                            scheduler_name=MISSING,
                            zen_partial=True)
    no_scheduler_config = scheduler_builder(scheduler_name=None)
            
    store(no_scheduler_config,
        name='none',
        group='scheduler_builder')

    plateau_lr_config = scheduler_builder(scheduler_name='reducelronplateau')
            
    store(plateau_lr_config,
        name='reducelronplateau',
        group='scheduler_builder')

    exponential_lr_config = scheduler_builder(scheduler_name='exponential')
            
    store(exponential_lr_config,
        name='exponential',
        group='scheduler_builder')

    store.add_to_hydra_store()