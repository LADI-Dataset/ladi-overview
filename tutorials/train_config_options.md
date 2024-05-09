# Train Configuration Options
This document details all the configuration options for training on the CAP dataset.

A training configuration consists of configs for a dataset, model builder, dataloader builder, optimizer builder, scheduler builder, and train builder. While not required, a set of values for each of these can be added to the Hydra store under the appropriate group name (one of `dataset`, `model_builder`, `dataloader_builder`, `optimizer_builder`, `scheduler_builder`, or `train_builder`), which provides a shorthand for referring to that set.

## Hydra-Zen
Hydra-zen provides a function called `builds(fn, *args)`, which you can think of as being similar to `functools.partial` - it returns either a new class or an instance of a class containing all arguments to a function. Once all arguments have been provided and you have an instance of the class, calling `hydra_zen.instantiate` on that instance will call the function with the given arguments. `builds` is the most important method to know for generating configs.

## Dataset
- dataset_base_dir: str,
- dataset_config_name: str,
- train_split: Literal['train', 'all']

dataset_base_dir: the base directory of the LADI dataset - should contain dirs named 'v1', 'v2', etc
dataset_config_name: which config to use from LadiClassifyDataset.BUILDER_CONFIGS
train_split: which train split to use - 'train' trains on the training set, 'all' trains on all data

## Model
- base_model_path: Union[str, Path],
- labels: list[str],
- hf_models_dir: Path,
- finetuned_models_dir: Path

base_model_path: path to the model directory (if training a new model, this is hf_models_dir/model_name)
hf_models_dir: the huggingface models directory
finetuned_models_dir: directory for user's finetuned models

## Optimizer
- model: torch.nn.Module,
- optimizer_name: Literal['adafactor', 'adamw'],
- lr: Optional[float],
- weight_decay: float,
- is_main_process: bool,
- **kwargs

model: the model being optimized. Can be any Pytorch model, including a Huggingface AutoModel
optimizer_name: the name of the optimizer. Currently only supports adafactor and AdamW
lr: the value of the learning rate
weight_decay: the value of the weight decay
**kwargs: will be passed to the optimizer constructor

## Scheduler
- optimizer,
- scheduler_name: Union[str,None]=None

optimizer: the constructed optimizer object
scheduler_name: one of 'exponential', 'reducelronplateau'

## Train
- epochs=MISSING,
- checkpoint_every=MISSING

epochs: integer number of training epochs
checkpoint_every: integer, how many epochs to wait between saving checkpoints