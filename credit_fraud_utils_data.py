import json
import pandas as pd
import numpy as np
from credit_fraud_utils_helper import get_processed_data
from load_configs import Config
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours
from imblearn.combine import SMOTEENN, SMOTETomek

def load_data(path: str):
    '''
    Load dataset & split to X, y
    :param path: data path
    :return X: input
    :return y: ground truth
    '''
    df = pd.read_csv(path)
    X = df.drop(config.FEATURES['target'], axis=1)
    y = df[config.FEATURES['target']]
    return X, y


def feature_construction(df:pd.DataFrame):
    '''
    Extracting new features according what concluded from EDA
    :param df: dataset
    :return: None
    '''

    df['hour'] = ((df['Time'] // 3600) % 24).astype(int)  # hours
    df['minute'] = ((df['Time'] // 60) % (24 * 60)).astype(int)  # minutes
    df['second'] = ((df['Time']) % (24 * 60 * 60)).astype(int)  # seconds

    # Day time periods
    df['morning'] = df['hour'].between(6, 11).astype(float)
    df['afternoon'] = df['hour'].between(12, 17).astype(float)
    df['evening'] = df['hour'].between(18, 23).astype(float)
    df['night'] = df['hour'].between(0, 5).astype(float)

    # High amount feature
    df['high_amount'] = (df['Amount'] > 2000).astype(float)


def feature_transformation(df:pd.DataFrame):
    '''
    Apply feature transformation methods for specific features:
        1- log transformation for skewed data ('Amount')
    :param df: data frame
    '''

    for col in config.FEATURES['log_trans']:
        df[col] = np.log1p(df[col])


def prepare_data(load_path:str='', target_preparing_path:str='', preparing_meta_path:str=''):
    '''
    preparing methods (feature extraction - feature transformation)
    :param load_path: data path
    :param target_preparing_path: where prepared data will be saved
    :param preparing_meta_path: where metadata of prepared data will be saved
    :return: None
    '''

    # load data
    X, y =  load_data(load_path)

    # apply feature engineering
    feature_construction(df=X)
    feature_transformation(df=X)

    # convert x & y to numpy array
    x = X.to_numpy()
    y = y.to_numpy()

    # save data in .npz file
    np.savez_compressed(target_preparing_path, x=x, y=y)

    # metadata of prepared data
    metadata= {
        'cols_names' : X.columns.tolist(),
        'n_samples': X.shape[0],
        'n_features': X.shape[1],
    }
    # save in json file
    with open(preparing_meta_path, 'w') as f:
        json.dump(metadata, f, indent=4)


def sample_save_data(load_prepared_path:str='', meta_path:str='',config=Config()):
    '''
    apply all sample techniques for train split data and save for modeling fits
    :param load_prepared_path: where prepared data saved
    :param meta_path: path of metadata that contain cols names
    :param config: set configurations
    :return: None
    '''

    X, t = get_processed_data(data_path=load_prepared_path, dtype='df', meta_path=meta_path)

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
        data_save_path = None
        metadata_save_path = None
        if 'train'  in load_prepared_path:
            data_save_path = config.DATASET['sampled'][name]['train']['data']
            metadata_save_path = config.DATASET['sampled'][name]['train']['metadata']
        elif 'val' in load_prepared_path:
            data_save_path = config.DATASET['sampled'][name]['val']['data']
            metadata_save_path = config.DATASET['sampled'][name]['val']['metadata']

        # save data
        np.savez_compressed(data_save_path, x=x_sampled, y=y_sampled)

        with open(metadata_save_path, 'w') as f:
            json.dump(meta,f, indent=4)


if __name__ == '__main__':

    # configs
    config = Config()

    # prepare all data splits
    splits_to_prepare = ['train', 'val' ,'test']
    splits_to_sample = ['train', 'val']
    for split in splits_to_prepare:
        prepare_data(load_path=config.DATASET['unprocessed'][split],
                     target_preparing_path=config.DATASET['prepared'][split]['data'],
                     preparing_meta_path=config.DATASET['prepared'][split]['metadata'])

    # sample train and val data
    for split in splits_to_sample:
        sample_save_data(load_prepared_path=config.DATASET['prepared'][split]['data'],
                         meta_path=config.DATASET['prepared'][split]['metadata'], config=config)