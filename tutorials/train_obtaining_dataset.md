# Obtaining the LADI Dataset

## Figure out what you want
This project has multiple sets of images, and multiple sets of labels for each set of images. Typically, the difference between label sets is that some sets of labels focus on particularly interesting subsets of the complete labels. When you download an image set through the interface we've provided, all label sets associated with that image set will be downloaded as well (all label sets once uncompressed take up about as much disk space as a single image). 

When it's time to use the data, we provide names for both the data (images) and the labels behind the scenes, which you can view in the file `LADI-v2-dataset.py`. However, the label name should typically be sufficient to uniquely identify the configuration you want, so the default configurations for our dataset are named according to the label names. The available label names are: 'v1_damage', 'v1_infrastructure', 'v2', 'v2_resized', 'v2a', and 'v2a_resized'. The available data names are 'v1', 'v2', and 'v2_resized'. `v2a` datasets have the same images as the `v2` datasets, but only a subset of the labels. So while the naming may imply otherwise, `v2a` is not a data name. All labels are for multi-label classification.

- The `v1` data was labeled by crowdsourced workers and isn't very reliable
- The `v2` data was labeled by Civil Air Patrol volunteers and is much more reliable
- The `v2_resized` data contains resized versions of the `v2` images. Using the `v2_resized` data will make your code run substantially faster and should be preferred if you don't need access to the very high resolution (8k+ resolution) original imagery. The resized images are resized to be at most 1800 x 1200 px (retaining aspect ratio), and are higher resolution than the operating resolutions for most standard computer vision models.

## Get it
While there are many ways to obtain the LADI dataset, we have provided a programmatic way to do it through the Huggingface datasets library. When you pass `download_ladi=True` to the dataset, _even if it's been downloaded before_, it will be downloaded and moved to the `base_dir` that you specify. So, you can simply write:

```python
from datasets import load_dataset
ds = load_dataset("./ladi_classify_dataset", "v2a_resized", 
                    streaming=True, download_ladi=True, 
                    base_dir='./ladi_dataset', trust_remote_code=True)
```

This will download the v2_resized data and all of the v2a labels into `./ladi_dataset`. Although `streaming=True` has been passed here, the entire dataset will be downloaded up-front - it simply won't be converted to an Arrow table. In the future, you can run the same code without `download_ladi=True` (it defaults to `False`) and the code will read from your local copy of the dataset.
