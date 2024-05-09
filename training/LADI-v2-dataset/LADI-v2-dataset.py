import cv2
import datasets
import pandas as pd
from datasets.data_files import DataFilesDict, sanitize_patterns
from pathlib import Path
from PIL import Image, ImageFile

from typing import List, Optional

ImageFile.LOAD_TRUNCATED_IMAGES = True

# maps the dataset names to names for the image sets they rely on
DATA_NAME_MAP = {
    'v1_damage': 'v1',
    'v1_infrastructure': 'v1',
    'v2': 'v2',
    'v2_resized': 'v2_resized',
    'v2a': 'v2',
    'v2a_resized': 'v2_resized'
}
# DATA_URLS = {x: None for x in ['v1', 'v2', 'v2_resized']}
DATA_URLS = {'v1': "https://ladi.s3.amazonaws.com/ladi_v1.tar.gz", 
             'v2': 'https://ladi.s3.amazonaws.com/ladi_v2.tar.gz', 
             'v2_resized': 'https://ladi.s3.amazonaws.com/ladi_v2_resized.tar.gz'}

SPLIT_REL_PATHS = {
    # note: the v1 datasets don't have separate 'test' and 'val' splits
    'v1_damage': {'train':'v1/damage_dataset/damage_df_train.csv',
                            'val':'v1/damage_dataset/damage_df_test.csv',
                            'test':'v1/damage_dataset/damage_df_test.csv',
                            'all': 'v1/damage_dataset/damage_df.csv'},
    'v1_infrastructure': {'train':'v1/infra_dataset/infra_df_train.csv',
                            'val':'v1/infra_dataset/infra_df_test.csv',
                            'test':'v1/infra_dataset/infra_df_test.csv',
                            'all':'v1/infra_dataset/infra_df.csv'},
    'v2': {'train':'v2/ladi_v2_labels_train.csv',
                             'val':'v2/ladi_v2_labels_val.csv',
                             'test':'v2/ladi_v2_labels_test.csv',
                             'all':'v2/ladi_v2_labels_train_full.csv'},
    'v2_resized': {'train':'v2/ladi_v2_labels_train_resized.csv',
                             'val':'v2/ladi_v2_labels_val_resized.csv',
                             'test':'v2/ladi_v2_labels_test_resized.csv',
                             'all':'v2/ladi_v2_labels_train_full_resized.csv'},
    'v2a': {'train':'v2/ladi_v2a_labels_train.csv',
                             'val':'v2/ladi_v2a_labels_val.csv',
                             'test':'v2/ladi_v2a_labels_test.csv',
                             'all':'v2/ladi_v2a_labels_train_full.csv'},
    'v2a_resized': {'train':'v2/ladi_v2a_labels_train_resized.csv',
                             'val':'v2/ladi_v2a_labels_val_resized.csv',
                             'test':'v2/ladi_v2a_labels_test_resized.csv',
                             'all':'v2/ladi_v2a_labels_train_full_resized.csv'}
}

class LadiClassifyDatasetConfig(datasets.BuilderConfig):
    def __init__(self, 
                 name: str = 'v2a_resized',
                 base_dir: Optional[str] = None, 
                 split_csvs = None,
                 download_ladi = False,
                 data_name: Optional[str] = None,
                 label_name: Optional[str] = None,
                 **kwargs):
        """
        split_csvs: a dictionary mapping split names to existing csv files containing annotations
            if this arg is set, you MUST already have the dataset
        base_dir: the base directory of the label CSVs and data files.
        data_name: the version of the data you're using. Used to determine what files to download if
            you don't specify split_csvs or url_list. Must be in DATA_URLS.keys().
            
        If split_csvs is None, the requested data will be downloaded from the hub. Please do NOT 
            use this feature with streaming=True, you will perform a large download every time.
        """
        self.download_ladi = download_ladi
        self.data_name = DATA_NAME_MAP[name] if data_name is None else data_name
        self.label_name = name if label_name is None else label_name
        self.base_dir = None if base_dir is None else Path(base_dir)
        self.split_csvs = split_csvs

        if self.data_name not in DATA_URLS.keys():
            raise ValueError(f"Expected data_name to be one of {DATA_URLS.keys()}, got {self.data_name}")
        
        if split_csvs is None and download_ladi == False:
            self.split_csvs = SPLIT_REL_PATHS[self.label_name]

        super(LadiClassifyDatasetConfig, self).__init__(name=name, **kwargs)
            

class LADIClassifyDataset(datasets.GeneratorBasedBuilder):
    """
    Dataset for LADI Classification task
    """
    
    VERSION = datasets.Version("0.2.1")
    BUILDER_CONFIG_CLASS = LadiClassifyDatasetConfig
    DEFAULT_CONFIG_NAME = 'v2a_resized'
    
    BUILDER_CONFIGS = [
        LadiClassifyDatasetConfig(
            name='v1_damage',
            version=VERSION,
            description="Dataset for recognizing damage (flood, rubble, misc) from LADI"
        ),
        LadiClassifyDatasetConfig(
            name="v1_infrastructure",
            version=VERSION,
            description="Dataset for recognizing infrastructure (buildings, roads) from LADI"
        ),
        LadiClassifyDatasetConfig(
            name="v2",
            version=VERSION,
            description="Dataset using the v2 labels for LADI"
        ),
        LadiClassifyDatasetConfig(
            name="v2_resized",
            version=VERSION,
            description="Dataset using the v2 labels for LADI, pointing to the lower resolution source images for speed"
        ),
        LadiClassifyDatasetConfig(
            name="v2a",
            version=VERSION,
            description="Dataset using the v2a labels for LADI"
        ),
        LadiClassifyDatasetConfig(
            name="v2a_resized",
            version=VERSION,
            description="Dataset using the v2a labels for LADI, pointing to the lower resolution source images for speed"
        ),
    ]
    
    def _info(self):
        if self.config.label_name == "v1_damage":
            features = datasets.Features(
                {
                    "image":datasets.Image(),
                    "flood":datasets.Value("bool"),
                    "rubble":datasets.Value("bool"),
                    "misc_damage":datasets.Value("bool")
                }
            )
        elif self.config.label_name == "v1_infrastructure":
            features = datasets.Features(
                 {
                    "image":datasets.Image(),
                    "building":datasets.Value("bool"),
                    "road":datasets.Value("bool")
                }
            )
        elif self.config.label_name in ["v2", "v2_resized"]:
            features = datasets.Features(
                {
                    "image":datasets.Image(),
                    "bridges_any": datasets.Value("bool"),
                    "bridges_damage": datasets.Value("bool"),
                    "buildings_affected": datasets.Value("bool"),
                    "buildings_any": datasets.Value("bool"),
                    "buildings_destroyed": datasets.Value("bool"),
                    "buildings_major": datasets.Value("bool"),
                    "buildings_minor": datasets.Value("bool"),
                    "debris_any": datasets.Value("bool"),
                    "flooding_any": datasets.Value("bool"),
                    "flooding_structures": datasets.Value("bool"),
                    "roads_any": datasets.Value("bool"),
                    "roads_damage": datasets.Value("bool"),
                    "trees_any": datasets.Value("bool"),
                    "trees_damage": datasets.Value("bool"),
                    "water_any": datasets.Value("bool"),
                }
            )
        elif self.config.label_name in ["v2a", "v2a_resized"]:
            features = datasets.Features(
                {
                    "image":datasets.Image(),
                    "bridges_any": datasets.Value("bool"),
                    "buildings_any": datasets.Value("bool"),
                    "buildings_affected_or_greater": datasets.Value("bool"),
                    "buildings_minor_or_greater": datasets.Value("bool"),
                    "debris_any": datasets.Value("bool"),
                    "flooding_any": datasets.Value("bool"),
                    "flooding_structures": datasets.Value("bool"),
                    "roads_any": datasets.Value("bool"),
                    "roads_damage": datasets.Value("bool"),
                    "trees_any": datasets.Value("bool"),
                    "trees_damage": datasets.Value("bool"),
                    "water_any": datasets.Value("bool"),
                }
            )
        else:
            raise NotImplementedError
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=f"LADI Dataset for {self.config.label_name} category",
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            # If there's a common (input, target) tuple from the features, uncomment supervised_keys line below and
            # specify them. They'll be used if as_supervised=True in builder.as_dataset.
            # supervised_keys=("image", "label"),
        )
    
    def read_ann_csv(self, fpath):
        if self.config.data_name == 'v1':
            return pd.read_csv(fpath, sep='\t', index_col=False)
        return pd.read_csv(fpath, sep=',', index_col=False)

    def _split_generators(self, dl_manager):
        generators = []
        data_files = self.config.split_csvs

        if self.config.download_ladi:
            # download data files to config.base_dir
            dl_url = dl_manager.download(DATA_URLS[self.config.data_name])
            base_dir = Path(self.config.base_dir)
            tar_iterator = dl_manager.iter_archive(dl_url)
            base_dir.mkdir(exist_ok=True)
            for filename, file in tar_iterator:
                file_path: Path = base_dir/filename
                file_path.parent.mkdir(parents=True, exist_ok=True)
                with open(base_dir/filename, 'wb') as f:
                    f.write(file.read())

        data_files = DataFilesDict.from_local_or_remote(
            sanitize_patterns(data_files), 
            base_path=self.config.base_dir
        )

        if 'train' in data_files.keys():
            train_df = self.read_ann_csv(data_files['train'][0])
            label_cols = tuple(label for label in train_df.columns if label not in ['url','local_path'])
            train_examples = [x._asdict() for x in train_df.itertuples()]
            generators.append(datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"examples":train_examples,
                            "label_cols":label_cols}
            ))
        if 'val' in data_files.keys():
            val_df = self.read_ann_csv(data_files['val'][0])
            label_cols = tuple(label for label in val_df.columns if label not in ['url','local_path'])
            val_examples = [x._asdict() for x in val_df.itertuples()]
            generators.append(datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"examples":val_examples,
                            "label_cols":label_cols}
            ))
        if 'test' in data_files.keys():
            test_df = self.read_ann_csv(data_files['test'][0])
            label_cols = tuple(label for label in test_df.columns if label not in ['url','local_path'])
            test_examples = [x._asdict() for x in test_df.itertuples()]
            generators.append(datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"examples":test_examples,
                            "label_cols":label_cols}
            ))
        if 'all' in data_files.keys():
            all_df = self.read_ann_csv(data_files['all'][0])
            label_cols = tuple(label for label in all_df.columns if label not in ['url','local_path'])
            all_examples = [x._asdict() for x in all_df.itertuples()]
            generators.append(datasets.SplitGenerator(
                name=datasets.Split.ALL,
                gen_kwargs={"examples":all_examples,
                            "label_cols":label_cols}
            ))

        return generators

    def _generate_examples(self, examples, label_cols, from_url_list=False):
        for ex in examples:
            try:
                image_path = Path(ex['local_path'])
                if not image_path.is_absolute():
                    image_path = str(self.config.base_dir/image_path)
            except:
                print(ex)
                raise
            
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            labels = {k:ex[k] for k in label_cols}
            labels |= {"image":image}
            yield image_path, labels