from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
import requests
from PIL import Image
from io import BytesIO

image_url = "http://s3.amazonaws.com/fema-cap-imagery/Images/CAP_-_VT_Flooding_Jul_2023/Source/23-1-5464/A0001_AerialOblique/_CAP0347.JPG"

img_data = requests.get(image_url).content
img = Image.open(BytesIO(img_data))

feature_extractor = AutoImageProcessor.from_pretrained("/home/gridsan/groups/CAP_shared/finetuned_models/model_DyG9LBaB/run_20240208-110739_8vZVm0y/epoch_049")
model = AutoModelForImageClassification.from_pretrained("/home/gridsan/groups/CAP_shared/finetuned_models/model_DyG9LBaB/run_20240208-110739_8vZVm0y/epoch_049")

inputs = feature_extractor(img, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

# model predicts one of the 1000 ImageNet classes
predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])