# LADI Overview

The Low Altitude Disaster Imagery (LADI) dataset was created to address the relative lack of annotated post-disaster aerial imagery in the computer vision community. Low altitude post-disaster aerial imagery from small planes and UAVs can provide high-resolution imagery to emergency management agencies to help them prioritize response efforts and perform damage assessments. In order to accelerate their workflow, computer vision can be used to automatically identify images that contain features of interest, including infrastructure such as buildings and roads, damage to such infrastructure, and hazards such as floods or debris. 

Version 1 of the LADI dataset was funded as part of a NIST public safety innovation accelerator effort to create datasets for public safety. It collected over 40k images from the Civil Air Patrol (CAP), which were annotated by crowdsourced workers providing labels from a hierarchical set of classes. The dataset was used in a number of NIST TRECVID challenges ([2020](https://arxiv.org/abs/2104.13473), [2021](https://www.nist.gov/publications/evaluating-multiple-video-understanding-and-retrieval-tasks-trecvid-2021), and [2022](https://arxiv.org/abs/2306.13118)) and provided a broad basis for classification of aerial post-disaster scenes. However, the authors found that the quality of the labels were inconsistent, stemming in part from the untrained nature of the crowdsource workers, as well as the potentially subjective definition of certain labels, such as "damage".

For LADI v2, the authors used CAP volunteers who were trained in the FEMA damage assessment process, and we collected damage labels using the defined [FEMA Preliminary Damage Assessment scale](https://www.fema.gov/disaster/how-declared/preliminary-damage-assessments): unaffected, affected, minor, major, destroyed. These damage levels have specific criteria, helping reduce the subjectivity of identifying whether a structure is damaged. We also provide pretrained classifiers to aid in replication, and serve as a basis for fine-tuning and potential deployments.

## Getting Started
The LADI v2 dataset is available on Hugging Face at [MITLL/LADI-v2-dataset](https://huggingface.co/datasets/MITLL/LADI-v2-dataset). This is the recommended method.

The first time you load the dataset, you should pass `download_ladi=True`, which will download a local copy of the relevant dataset version to your local system at `base_dir`.
```python
from datasets import load_dataset

ds = load_dataset("MITLL/LADI-v2-dataset", "v2a_resized",
                streaming=True, download_ladi=True,
                base_dir='./ladi_dataset', trust_remote_code=True)
```
This only needs to be done once, and subsequent calls can omit the argument, and it will automatically use your local copy.
```python
ds = load_dataset("MITLL/LADI-v2-dataset", "v2a_resized",
                streaming=True, base_dir='./ladi_dataset', trust_remote_code=True)
``` 

You can also manually access the dataset at https://ladi.s3.amazonaws.com/index.html, or via AWS S3 at `s3://ladi`, and the provided dataset class in `training/LADI-v2-dataset`. We recommend using the [LADI_v2_resized](https://ladi.s3.amazonaws.com/ladi_v2_resized.tar.gz) version which resizes the images to 1800x1200, which should be large enough for most applications, but drastically reduces the overall dataset file size.

The LADI v1 and v2 dataset files are hosted as part of the [AWS Open Data](https://registry.opendata.aws/ladi/) program.

### Dataset Details

The LADI-v2 dataset is a set of aerial disaster images captured and labeled by the Civil Air Patrol (CAP). The images are geotagged (in their EXIF metadata). Each image has been labeled in triplicate by CAP volunteers trained in the FEMA damage assessment process for multi-label classification; where volunteers disagreed about the presence of a class, a majority vote was taken. The classes are:

- bridges_any
- bridges_damage
- buildings_affected
- buildings_any
- buildings_destroyed
- buildings_major
- buildings_minor
- debris_any
- flooding_any
- flooding_structures
- roads_any
- roads_damage
- trees_any
- trees_damage
- water_any

The v2 dataset consists of approximately 10k images, split into a train set of 8k images, a validation set of 1k images, and a test test of 1k images. The train and validation sets are drawn from the same distribution (CAP images from federally-declared disasters 2015-2022), whereas the test set is drawn from events in 2023, which has a different distribution of event types and locations. This is done to simulate the distribution shift as new events occur each year.

#### Dataset v2a
The `v2a` variant of the dataset presents the same images with a modified set of labels, where the damage categories for buildings have been compressed into two classes of `buildings_affected_or_greater` and `buildings_minor_or_greater`. We find that this task is easier and of similar practical value for triage purposes. The `bridges_damage` label has also been removed due to the low number of positive examples in the dataset.

- bridges_any
- buildings_any
- buildings_affected_or_greater
- buildings_minor_or_greater
- debris_any
- flooding_any
- flooding_structures
- roads_any
- roads_damage
- trees_any
- trees_damage
- water_any


#### Dataset Summary: v1
The dataset code also supports loading a subset of the LADI v1 dataset, which consists of roughly 25k images, broken into two tasks, 'infrastructure' and 'damage'. The LADI v1 dataset was labeled by crowdsourced workers and the labels shouldn't be considered definitive. The data may be suitable for a pretraining task prior to fine-tuning on LADI v2.

The infrastructure task involves identifying infrastructure in images and has classes `building` and `road`. It is divided into a train set of 8.2k images and a test set of 2k images. 

The damage task involves identifying damage and has classes `flood`, `rubble`, and `misc_damage`. It is divided into a train set of 14.4k images and a test set of 3.6k images. 


#### Supported Tasks
The images are labeled for multi-label classification, as any number of the elements listed above may be present in a single image.

#### Data Structure
A single example in the v2a dataset looks like this:

```
{
'image': <PIL.PngImagePlugin.PngImageFile image mode=RGB size=1800x1200 at ...>,
'bridges_any': False, 
'buildings_any': False, 
'buildings_affected_or_greater': False, 
'buildings_minor_or_greater': False, 
'debris_any': False, 
'flooding_any': False, 
'flooding_structures': False, 
'roads_any': False, 
'roads_damage': False, 
'trees_any': True, 
'trees_damage': True, 
'water_any': True
}
```

Examples in the v1 datasets are analogous, with classes drawn from their respective tasks (infrastructure and damage).


### Pretrained Classifiers 
We provide a set of pretrained classifiers on the LADI v2 dataset for downstream finetuning and deployment purposes. See the associated model cards on Hugging Face for instructions on how to use.
- [LADI-v2-classifier-small](https://huggingface.co/MITLL/LADI-v2-classifier-small) - Recommended for deployment, based on [google/bit-50](https://huggingface.co/google/bit-50), trained on entire LADI v2 dataset
- [LADI-v2-classifier-large](https://huggingface.co/MITLL/LADI-v2-classifier-large) - Based on [microsoft/swinv2](https://huggingface.co/microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft), trained on entire LADI v2 dataset

We also provide "reference" versions of each model which are trained only on the `train` split of the LADI v2 dataset, to aid in reproducing the results presented in the paper.
- [LADI-v2-classifier-small-reference](https://huggingface.co/MITLL/LADI-v2-classifier-small) 
- [LADI-v2-classifier-large-reference](https://huggingface.co/MITLL/LADI-v2-classifier-large-reference)

### Demo Application
![Demonstration ArcGIS Online Application of LADI v2 classifier outputs for select 2024 disaster events](https://github.com/LADI-Dataset/ladi-overview/assets/45951413/0504a73f-3427-4625-9a8f-826ef886bca4)
A demonstration application is [available](https://experience.arcgis.com/experience/12facb85c45a4b7e9c4caef04be6e4d2/?draft=true#data_s=id%3AdataSource_2-18f59fc7c23-layer-2%3A13187) on ArcGIS Online showing a potential use case of the classifier in action. The web app shows locations of Civil Air Patrol images taken after various storms in April and May 2024. LADI-v2-classifier-small is run on each of the images and the annotations are stored in a feature service. This allows for filtering of images based on classified labels, as well as tying symbology to the output of the classifiers to identify regions of interest. 

## Repository Structure
This repository contains reference code for performing inference using the pretrained classifier, as well as the scripts used in training and finetuning the LADI-v2-classifiers.

[`container/`](container/) contains a Dockerfile and instructions to build a docker image for inference purposes.

[`inference/`](inference/) contains code for performing inference with the pretrained classifier.

[`training/`](training/) contains code and [instructions](training/README.md) for training and finetuning the classifier.

[`tutorials/`](tutorials/) provides more instructions and examples of how to [install dependencies](tutorials/set_up_environment.md), [run inference](tutorials/inference_tutorial.md), and [configure the training process](tutorials/train_config_options.md).

[`v1_docs/`](v1_docs/) provides the archived version of the now deprecated LADI v1.

## Point of Contact

We encourage the use of the [GitHub Issues](https://github.com/LADI-Dataset/ladi-overview/issues) to submit questions or issues, but when email is required, please contact the administrators at [ladi-dataset-admin@mit.edu](mailto:ladi-dataset-admin@mit.edu). 

## Citations

### LADI v2
If you use the LADI v2 dataset or classifiers, please cite the following:

🚧PAPER Forthcoming🚧

### LADI v1
If you use the LADI v1 dataset, please cite the following:

```tex
@inproceedings{liuLargeScale2019,
author={J. {Liu} and D. {Strohschein} and S. {Samsi} and A. {Weinert}},
booktitle={2019 IEEE High Performance Extreme Computing Conference (HPEC)},
title={Large Scale Organization and Inference of an Imagery Dataset for Public Safety},
year={2019},
volume={},
number={},
pages={1-6},
keywords={Safety;Metadata;Organizations;Servers;Computer architecture;Program processors;Broadband communication;big data;indexing;inference;public safety;video},
doi={10.1109/HPEC.2019.8916437},
ISSN={2377-6943},
month={Sep.},}
```


## Distribution Information

DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the Department of the Air Force under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Department of the Air Force.

© 2024 Massachusetts Institute of Technology.

The software/firmware is provided to you on an As-Is basis

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.
