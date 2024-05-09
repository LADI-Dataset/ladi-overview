from hydra_zen import builds, store, MISSING
from pathlib import Path
from transformers import (AutoConfig,
                          AutoImageProcessor,
                          AutoModelForImageClassification)
from typing import Union, Tuple
from utils import get_random_string

models_list = [
    'facebook/convnextv2-huge-22k-384',
    'facebook/convnextv2-large-22k-224',
    'google/vit-base-patch32-384',
    'google/vit-large-patch16-384',
    'google/vit-base-patch16-224-in21k',
    'google/vit-base-patch32-224-in21k',
    'google/vit-huge-patch14-224-in21k',
    'google/vit-large-patch16-224-in21k',
    'google/bit-50',
    'facebook/dinov2-base',
    'facebook/dinov2-large',
    'facebook/dinov2-giant',
    'microsoft/resnet-50',
    'microsoft/resnet-152',
    'apple/mobilevitv2-1.0-imagenet1k-256',
    'microsoft/focalnet-base',
    'google/efficientnet-b0',
    'google/efficientnet-b7',
    'microsoft/beit-base-patch16-224-pt22k',
    'facebook/deit-base-distilled-patch16-224',
    'facebook/deit-base-patch16-224',
    'google/mobilenet_v2_1.0_224',
    'google/mobilenet_v1_1.0_224',
    'microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft',
    'microsoft/swinv2-large-patch4-window12-192-22k',
    'microsoft/swin-tiny-patch4-window7-224',
    'microsoft/swin-large-patch4-window7-224-in22k'
]

def get_model(base_model_path: Union[str, Path],
              labels: list[str],
              hf_models_dir: Path,
              finetuned_models_dir: Path) -> Tuple[AutoImageProcessor, AutoModelForImageClassification, str]:
    """
    This function is internal to this file - it is exposed via global state by
    means of the Hydra store below.

    base_model_path: path to the model directory (if training a new model, this
        is hf_models_dir/model_name)
    hf_models_dir: the huggingface models directory
    finetuned_models_dir: directory for user's finetuned models

    Returns the image preprocessor for a model, the model itself, and the unique
    ID used to store a finetuned version of the model
    """
    if not isinstance(base_model_path, Path):
        base_model_path = Path(base_model_path)
    if base_model_path.is_relative_to(hf_models_dir):
        # if the base model is foundation model, generate model id
        model_id = get_random_string(8)
    elif base_model_path.is_relative_to(finetuned_models_dir):
        # otherwise we're starting from a finetuned model, get the id from there
        model_id = base_model_path.relative_to(finetuned_models_dir).parts[0].removeprefix('model_')
    else:
        raise ValueError(f'model_path must point to a subdirectory in {hf_models_dir} or {finetuned_models_dir}')
    base_model_path = str(base_model_path)
    label2id = {label: str(i) for i, label in enumerate(labels)}
    id2label = {str(i): label for i, label in enumerate(labels)}
    config = AutoConfig.from_pretrained(
        base_model_path,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        trust_remote_code=False,
        problem_type="multi_label_classification"
    )
    image_processor = AutoImageProcessor.from_pretrained(
        base_model_path,
        trust_remote_code=False
    )
    model = AutoModelForImageClassification.from_pretrained(
        base_model_path,
        from_tf=bool(".ckpt" in base_model_path),
        config=config,
        ignore_mismatched_sizes=True,
        trust_remote_code=False
    )
    return image_processor, model, model_id

## Model configs
def register_configs(hf_models_dir: str | Path, finetuned_models_dir: str | Path):
    """
    Registers configurations for models with the Hydra store

    hf_models_dir: directory containing downloaded auto models from Huggingface
    finetuned_models_dir: directory for storing weights of finetuned models
    """
    hf_models_dir = Path(hf_models_dir)
    
    for model_name in models_list:
        model_config = builds(get_model,
                            hf_models_dir=hf_models_dir,
                            finetuned_models_dir=finetuned_models_dir,
                            base_model_path=str(hf_models_dir/model_name),
                            zen_partial=True)
        store(model_config,
            name=model_name.replace('/','-'),
            group='model_builder')

    store.add_to_hydra_store()