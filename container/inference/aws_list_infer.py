import os
import boto3
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from transformers import pipeline
from tqdm import tqdm
from typing import List
from io import BytesIO
from pathlib import Path
from PIL import Image
from metadata_utils import get_metadata_img
from multiprocessing import Manager

# arcane botocore s3 stuff needed for zero-auth reads
from botocore import UNSIGNED
from botocore.config import Config

from dl_models import MODEL_NAME

labels = ['trees_any',
          'water_any',
          'trees_damage',
          'debris_any',
          'roads_any',
          'flooding_any',
          'buildings_any',
          'buildings_affected_or_greater',
          'bridges_any',
          'flooding_structures',
          'roads_damage']

class AWSListDataset(Dataset):
    def __init__(self, urls: List[str]):
        self.s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
        self.urls = urls
        self.manager = Manager()
        self.metadata_map = self.manager.dict()
        
    def __len__(self):
        return len(self.urls)
    
    def __getitem__(self, idx):
        # url is formatted as s3://bucket/key
        url = self.urls[idx]
        url = url.removeprefix("s3://")
        url_parts = url.split('/')
        bucket_name, key = url_parts[0], '/'.join(url_parts[1:])
        
        # trick to read the image directly into memory rather than going
        # to disk twice for no reason
        f = BytesIO()
        self.s3_client.download_fileobj(bucket_name, key, f)
        # need f.getvalue() here to copy the buffer so it can be reshaped
        f.seek(0)
        img = Image.open(f)
        self.metadata_map[self.urls[idx]] = get_metadata_img(img)
        return img

def postprocess_output(infer_output):
    output_dict = {}
    for response in infer_output:
        if response['label'] in labels:
            output_dict[response['label']] = response['score']
    return dict(sorted(output_dict.items()))

if __name__ == "__main__":
    pipe = pipeline(model=MODEL_NAME,
         task='image-classification',
         function_to_apply='sigmoid',
         device=0,
         num_workers=40)
    
    urls = ["s3://fema-cap-imagery/Images/12/20131/IMG_6150_4d5f3c2b-0b7c-4ed8-a1e3-b444f1bde0e0.jpg"]
    ds = AWSListDataset(urls)
    
    outputs = []
    for i, output in tqdm(enumerate(pipe(ds, batch_size=12, top_k=20))):
        classes = postprocess_output(output)
        curr_filename = urls[i]
        img_metadata = ds.metadata_map[curr_filename]

        outputs.append({'file_path': curr_filename, **classes, **img_metadata})
    
    df = pd.DataFrame(data=outputs)
    df.to_csv('outputs.csv', index=False)

