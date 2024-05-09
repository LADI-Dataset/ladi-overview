import requests
import numpy as np
import pathlib
import pandas as pd

from io import BytesIO
from PIL import Image
from torch.utils.data import Dataset
from transformers import pipeline
from tqdm import tqdm
from multiprocessing import Manager

from typing import List
from metadata_utils import get_metadata_img

MODEL_NAME = 'MITLL/LADI-v2-classifier-small'

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

class URLListDataset(Dataset):
    def __init__(self, urls: List[str]):
        self.urls = urls
        self.manager = Manager()
        self.metadata_map = self.manager.dict()
        
    def __len__(self):
        return len(self.urls)
    
    def __getitem__(self, idx):
        url = self.urls[idx]
        
        response = requests.get(url)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            self.metadata_map[url] = get_metadata_img(img)
            return img
        else:
            raise Exception(f"Failed to download image at: {url}")

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
    
    urls = ["http://s3.amazonaws.com/fema-cap-imagery/Images/CAP_-_Spring_Storms_2024/Source/24-1-5100_OKWG/A0003_AerialOblique/2415100A0003_Marietta_Area__310.JPG", "http://s3.amazonaws.com/fema-cap-imagery/Images/CAP_-_Spring_Storms_2024/Source/24-1-5100_OKWG/A0003_AerialOblique/2415100A0003_Marietta_Area__301.JPG"]
    ds = URLListDataset(urls)
    
    outputs = []
    for i, output in tqdm(enumerate(pipe(ds, batch_size=12, top_k=20))):
        classes = postprocess_output(output)
        curr_filename = urls[i]
        img_metadata = ds.metadata_map[curr_filename]

        outputs.append({'file_path': curr_filename, **classes, **img_metadata})
    
    df = pd.DataFrame(data=outputs)
    df.to_csv('outputs.csv', index=False)
        
