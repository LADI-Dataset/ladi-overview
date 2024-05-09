# Brief Tour of Code
Training and evaluation logic and entry point are in `train.py`. Basic code to retrieve a model, dataloader, dataset, optimizer, or scheduler, as well as a few configurations for each, live in their respective `config_{item}.py` files. A few configurations with training options (eg: number of epochs, how often to create checkpoints) can be found in `config_train.py`.

Configurations are accumulated into the Hydra store by `configs.py:register_all_configs()`, and most hardcoded defaults are in `train.py` or `configs.py`.

By default, models will be written to `./finetuned_models` and the dataset will be downloaded to `./ladi_dataset`. You can change this by modifying `FINETUNED_MODELS_DIR` and `LADI_DATA_DIR` in `train.py`.

# Customizing the Code
## Adding a new config
You can add a new config to the Hydra store at any time before `zen(task).hydra_main(...)` runs in `train.py`. We recommend that you follow the examples in `config_{x}.py` to generate working configs. You may select which configs are used in the command line. For example, this command:

```bash
python train.py \
    model_builder=google-bit-50 \
    scheduler_builder=exponential \
    optimizer_builder=adamw \
    optimizer_builder.lr=0.0001 \
    +scheduler_builder.gamma=0.9 \
    dataset=ladi_v2a_resized_all
```

Will train a Google bit-50 model with an exponential LR schedule, an AdamW optimizer, and an lr of 1e-4 on the `all` split of the `ladi_v2a_resized` dataset. The `+scheduler_builder.gamma=0.9` portion will send the keyword argument `gamma=0.9` to `config_scheduler.get_scheduler` (accepts all **kwargs and passes them to the scheduler constructor).

## Logging
By default, this project logs with [MLFlow](https://mlflow.org/), which can log to a remote server (you can set the endpoint using environment variables) or a local directory. If you don't want an MLFlow dependency, you can set `USE_MLFLOW = False` in `train.py` and all logging during the training process will be done using the `accelerate` default logger.