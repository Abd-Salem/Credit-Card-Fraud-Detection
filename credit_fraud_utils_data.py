import json
import pandas as pd
import numpy as np
from configs import Config
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours
from imblearn.combine import SMOTEENN, SMOTETomek


# create config loader
config = Config()

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

    # dict for saving data
    data = {
        'x_prepared': x,
        'y_prepared': y
    }

    # save data in .npz file
    np.savez_compressed(target_preparing_path, **data)

    # metadata of prepared data
    metadata= {
        'input_cols_names' : X.columns.tolist(),
        'n_samples': X.shape[0],
        'n_features': X.shape[1],
    }
    # save in json file
    with open(preparing_meta_path, 'w') as f:
        json.dump(metadata, f, indent=4)


def sample_save_data(load_prepared_path:str='', load_meta_path:str='',
                     split_name:str = ''):
    '''
    apply all sample techniques for train split data and save for modeling fits
    :param load_prepared_path: where prepared data saved
    :param load_meta_path: where metadata of prepared data saved
    :param split_name: name to get save path
    :return: None
    '''

    # get prepared data from given paths
    data = np.load(load_prepared_path)

    with open(load_meta_path, 'r') as f:
        meta = json.load(f)

    X = data['x_prepared']
    y = data['y_prepared']
    cols_names = meta['input_cols_names']       # columns names

    # change to dataframe type
    X, y = pd.DataFrame(data=X, columns=cols_names), pd.Series(y)

    # get our configs
    kn = config.SAMPLING['params']['smote_kn']
    nn = config.SAMPLING['params']['enn_nn']
    sampling_strategy = config.SAMPLING['params']['sampling_strategy']

    sampler_map = {
        'rus': RandomUnderSampler(random_state=config.RANDOM_STATE, sampling_strategy=sampling_strategy),
        'enn': EditedNearestNeighbours(sampling_strategy=sampling_strategy, n_neighbors=nn),
        'smote': SMOTE(random_state=config.RANDOM_STATE, sampling_strategy=sampling_strategy, k_neighbors=kn),
        'smoteenn': SMOTEENN(random_state=config.RANDOM_STATE,
                             sampling_strategy=sampling_strategy,
                             smote=SMOTE(k_neighbors=kn), enn=EditedNearestNeighbours(n_neighbors=nn)),
        'smotetomek': SMOTETomek(random_state=config.RANDOM_STATE ,sampling_strategy=sampling_strategy,
                             smote=SMOTE(k_neighbors=kn))
    }

    # dict of samples processes
    samplers = {technique_name: sampler_map[technique_name] for technique_name in config.SAMPLING['techniques']}

    # apply all techniques on loaded data and save processed data in files
    for name, sampler in samplers.items():

        x_sampled, y_sampled = sampler.fit_resample(X, y)  # get sampled data

        # save sampled data in .npz file
        sampled = {
            'x_sampled': x_sampled,
            'y_sampled': y_sampled
        }
        np.savez_compressed(f'{config.DATASET['sampled'][name][split_name]}', **sampled)

        # save metadata
        sample_metadata = {
            'input_cols_names': X.columns.tolist(),
            'num_samples': X.shape[0],
            'num_features': X.shape[1]
        }
        with open(f'{config.DATASET['sampled'][name][f'{split_name}_metadata']}', 'w') as f:
            json.dump(sample_metadata, f, indent=4)


if __name__ == '__main__':

    # prepare all data splits
    splits_to_prepare = ['train', 'val', 'train_val', 'test']
    splits_to_sample = ['train', 'train_val']
    for split in splits_to_prepare:
        prepare_data(load_path=config.DATASET['unprocessed'][split],
                     target_preparing_path=config.DATASET['prepared'][split]['data'],
                     preparing_meta_path=config.DATASET['prepared'][split]['metadata'])

    # sample train data
    for split in splits_to_sample:
        sample_save_data(load_prepared_path=config.DATASET['prepared'][split]['data'],
                         load_meta_path=config.DATASET['prepared'][split]['metadata'],
                         split_name=split)