from transformers import (AutoConfig,
                          AutoImageProcessor,
                          AutoModelForImageClassification)
from sys import argv

# start with the top 3 models
models = [
    'microsoft/swinv2-large-patch4-window12-192-22k',
    'microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft',
    'google/bit-50',
]

if __name__ == "__main__":
    cache_dir = None
    if len(argv) == 2:
        cache_dir = argv[1]
    for base_model_path in models:
        config = AutoConfig.from_pretrained(
            base_model_path,
            num_labels=5,
            id2label={a: chr(a+97) for a in range(5)},
            label2id={chr(a+97): a for a in range(5)},
            trust_remote_code=False,
            problem_type="multi_label_classification",
            cache_dir=cache_dir
        )
        image_processor = AutoImageProcessor.from_pretrained(
            base_model_path,
            trust_remote_code=False,
            cache_dir=cache_dir
        )
        model = AutoModelForImageClassification.from_pretrained(
            base_model_path,
            from_tf=False,
            config=config,
            ignore_mismatched_sizes=True,
            trust_remote_code=False,
            cache_dir=cache_dir
        )