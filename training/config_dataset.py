import pandas as pd

from hydra_zen import builds, store, MISSING, make_config
from datasets import load_dataset, load_dataset_builder
from datasets.data_files import DataFilesDict, sanitize_patterns
from typing import Optional, Literal
from pathlib import Path

def get_datasets(dataset_base_dir: str,
                 dataset_config_name: str,
                 train_split: Literal['train', 'all']):
    '''
    This function is internal to this file - it is exposed via global state by
    means of the Hydra store below.

    dataset_base_dir: the base directory of the LADI dataset - should contain
      dirs named 'v1', 'v2', etc
    dataset_config_name: which config to use from LadiClassifyDataset.BUILDER_CONFIGS
    train_split: which train split to use - 'train' trains on the training set,
        'all' trains on all data
    
    returns HF dataset, list of labels, and class weights
    class weights are the ratio of negative to positive examples
    '''
    # when dataset is on Hub we can use a non-local version
    dataset = load_dataset('MITLL/LADI-v2-dataset', dataset_config_name, 
                           streaming=True, trust_remote_code=True, 
                           base_dir=dataset_base_dir)
    labels = [x for x in dataset["train"].column_names if x != "image"]

    if train_split not in ['train', 'all']:
        raise ValueError('only train_split=train or train_split=all is supported')
    
    ds_builder_cls = load_dataset_builder('MITLL/LADI-v2-dataset', dataset_config_name, 
                           trust_remote_code=True, base_dir=dataset_base_dir)
    data_files = DataFilesDict.from_local_or_remote(
            sanitize_patterns(ds_builder_cls.config.split_csvs), 
            base_path=ds_builder_cls.config.base_dir
        )
    
    if ds_builder_cls.config.data_name == 'v1':
        train_df = pd.read_csv(data_files[train_split][0], sep='\t')
    else:
        train_df = pd.read_csv(data_files[train_split][0])
    
    pos_examples = train_df[labels].sum()
    class_weights = (len(train_df) - pos_examples)/pos_examples
    return dataset, labels, class_weights

## Dataset configs
def register_configs(data_base_dir: Optional[str]):
      """
      Registers dataset configs with the Hydra store

      data_base_dir: path to the base directory of the LADI dataset
      """
      dataset_builder = builds(get_datasets,
                              dataset_base_dir=data_base_dir,
                              populate_full_signature=True)
      
      ladi_v1_damage_config = dataset_builder(dataset_config_name='v1_damage',
                                       train_split='train')
      
      ladi_v1_infra_config = dataset_builder(dataset_config_name='v1_infrastructure',
                                       train_split='train')
      
      ladi_v2_config = dataset_builder(dataset_config_name='v2',
                                    train_split='train',)

      ladi_v2a_config = dataset_builder(dataset_config_name='v2a',
                                    train_split='train',)

      ladi_v2_resized_config = dataset_builder(dataset_config_name='v2_resized',
                                                train_split='train')

      ladi_v2_resized_all_config = dataset_builder(dataset_config_name='v2_resized',
                                                train_split='all')

      ladi_v2a_resized_config = dataset_builder(dataset_config_name='v2a_resized',
                                                train_split='train')

      ladi_v2a_resized_all_config = dataset_builder(dataset_config_name='v2a_resized',
                                                train_split='all')


      # add them to the store
      store_ds = store(group='dataset')

      store_ds(ladi_v1_damage_config,
            name='ladi_v1_damage')
      
      store_ds(ladi_v1_infra_config,
            name='ladi_v1_infra')

      store_ds(ladi_v2_config,
            name='ladi_v2')

      store_ds(ladi_v2a_config,
            name='ladi_v2a')

      store_ds(ladi_v2_resized_config,
            name='ladi_v2_resized')

      store_ds(ladi_v2a_resized_config,
            name='ladi_v2a_resized')

      store_ds(ladi_v2_resized_all_config,
            name='ladi_v2_resized_all')

      store_ds(ladi_v2a_resized_all_config,
            name='ladi_v2a_resized_all')

      store_ds.add_to_hydra_store()