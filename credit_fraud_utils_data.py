import json
import pandas as pd
import numpy as np
from load_configs import Config
from credit_fraud_utils_helper import load_data
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours
from imblearn.combine import SMOTEENN, SMOTETomek



def feature_construction(df:pd.DataFrame):
    '''
    Extracting new features according what concluded from EDA
    :param df: dataset
    :return: df
    '''
    df = df.copy()
    df['hour'] = ((df['Time'] // 3600) % 48).astype(int)  # hours
    df['minute'] = ((df['Time'] // 60) % 60).astype(int)  # minutes
    df['second'] = ((df['Time']) % 60).astype(int)  # seconds

    # Day time periods
    df['morning'] = df['hour'].between(6, 11).astype(float)
    df['afternoon'] = df['hour'].between(12, 17).astype(float)
    df['evening'] = df['hour'].between(18, 23).astype(float)
    df['night'] = df['hour'].between(0, 5).astype(float)

    # High amount feature
    df['high_amount'] = (df['Amount'] > 2000).astype(float)

    return df


def feature_transformation(df:pd.DataFrame):
    '''
    Apply feature transformation methods for specific features:
        1- log transformation for skewed data ('Amount')
    :param df: data frame
    :return: df
    '''

    # Log transformation for Amount feature
    df = df.copy()
    df['Amount'] = np.log1p(df['Amount'])
    return df


def sample_save_data(data_path:str='', config=Config()):
    '''
    apply all sample techniques for train split data and save for modeling fits
    :param data_path: where data is saved
    :param config: set configurations
    :return: None
    '''

    X, t = load_data(data_path, target_col_name=config.FEATURES['target'])

    # get our configs
    kn = config.SAMPLING['params']['smote_kn']
    nn = config.SAMPLING['params']['enn_nn']
    sampling_strategy = config.SAMPLING['params']['sampling_strategy']
    random_state = config.RANDOM_STATE
    techniques = config.SAMPLING['techniques']


    # map different samplers in dict
    sampler_map = {
        'rus': RandomUnderSampler(random_state=random_state, sampling_strategy=sampling_strategy),
        'enn': EditedNearestNeighbours(sampling_strategy=sampling_strategy, n_neighbors=nn),
        'smote': SMOTE(random_state=random_state, sampling_strategy=sampling_strategy, k_neighbors=kn),
        'smoteenn': SMOTEENN(random_state=random_state,
                             sampling_strategy=sampling_strategy,
                             smote=SMOTE(k_neighbors=kn), enn=EditedNearestNeighbours(n_neighbors=nn)),
        'smotetomek': SMOTETomek(random_state=random_state ,sampling_strategy=sampling_strategy,
                             smote=SMOTE(k_neighbors=kn))
    }

    # dict of samples according to configurations
    samplers = {technique_name: sampler_map[technique_name] for technique_name in techniques}

    # apply all techniques on loaded data and save processed data in files
    for name, sampler in samplers.items():

        x_sampled, y_sampled = sampler.fit_resample(X, t)  # get sampled data

        meta = {
            'cols_names' : X.columns.to_list(),
            'n_features'    : X.shape[0],
            'n_samples'     : X.shape[1]
        }

        # save sampled data in .npz file
        data_save_path = config.DATASET['sampled'][name]['train']['data']
        metadata_save_path = config.DATASET['sampled'][name]['train']['metadata']

        # save data
        np.savez_compressed(data_save_path, x=x_sampled, y=y_sampled)

        with open(metadata_save_path, 'w') as f:
            json.dump(meta,f, indent=4)


if __name__ == '__main__':

    # configs
    config = Config()
    splits_to_sample = ['train']

    # sample train and val data
    for split in splits_to_sample:
        sample_save_data(data_path=config.DATASET['unprocessed'][split],
                         config=config)