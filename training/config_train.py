from hydra_zen import builds, store, MISSING
from typing import Callable

# Looking for the train function? It's in train.py because it needs access
# to local state defined there.

def register_configs(train_fn: Callable):
    """
    Registers configurations for training with the Hydra store
    """
    ## train config
    train_builder = builds(train_fn,
                        epochs=MISSING,
                        checkpoint_every=MISSING,
                        zen_partial=True)
    debug_train_config = train_builder(
                            epochs=2,
                            checkpoint_every=1,)

    default_train_config = train_builder(
                            epochs=50,
                            checkpoint_every=5,)

    store(debug_train_config,
        name='debug',
        group='train_builder')

    store(default_train_config,
        name='default',
        group='train_builder')

    store.add_to_hydra_store()