import huggingface_hub as hub
import transformers

MODEL_NAME = 'MITLL/LADI-v2-classifier-small'

if __name__ == "__main__":
    hub.snapshot_download(MODEL_NAME)
    transformers.utils.move_cache()

