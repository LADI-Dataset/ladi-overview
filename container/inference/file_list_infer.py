import numpy as np
import pathlib
import sys
import pickle
import pandas as pd

from PIL import Image
from torch.utils.data import Dataset
from transformers import pipeline
from tqdm import tqdm
from pathlib import Path

from typing import List
from metadata_utils import get_metadata_entry

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

class FileListDataset(Dataset):
    def __init__(self, paths: List[str]):
        self.paths = paths
        self.paths = [Path(x) for x in self.paths if Path(x).exists()]
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        url = Path(self.paths[idx])
        if url.exists():
            img = Image.open(url)
        return img


def postprocess_output(infer_output):
    output_dict = {}
    for response in infer_output:
        if response['label'] in labels:
            output_dict[response['label']] = response['score']
    output_dict
    return dict(sorted(output_dict.items()))

if __name__ == "__main__":
    
    pipe = pipeline(model=MODEL_NAME,
         task='image-classification',
         function_to_apply='sigmoid',
         device=0,
         num_workers=20)
    
    with open(sys.argv[1], 'r') as f:
        files = f.readlines()
        files = [x.strip() for x in files]

    ds = FileListDataset(files)
    
    outputs = []
    for i, output in tqdm(enumerate(pipe(ds, batch_size=12, top_k=20))):
        classes = postprocess_output(output)
        curr_filename = files[i]
        img_metadata = get_metadata_entry(curr_filename)

        outputs.append({'file_path': curr_filename, **classes, **img_metadata})
    
    df = pd.DataFrame(data=outputs)
    df.to_csv('outputs.csv', index=False)
