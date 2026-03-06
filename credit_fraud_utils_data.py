import json
import pandas as pd
import numpy as np
from config import config
from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours
from imblearn.combine import SMOTEENN, SMOTETomek

def load_data(path: str):
    '''Load dataset and split to X, y'''
    df = pd.read_csv(path)
    X = df.drop(config.FEATURES['target'], axis=1)
    y = df[config.FEATURES['target']]
    return X, y


def feature_construction(df:pd.DataFrame):
    '''
    Extracting features from existence features

    parameter:
        df (DataFrame): dataset
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



def feature_transformation(preprocessing_type:str ='standard'):
    '''
    Column transformation methods:
        1- log transformation for skewed data ('Amount', 'V8', 'V28')
        2- standard scaling for numeric features
    parameter:
        preprocessing_type (str): standard or minmax
    return:
        col_trans (ColumnTransformation): column transformation method for pipelining
    '''

    # check for scaling type
    scaler = StandardScaler()
    if preprocessing_type == 'minmax':
        scaler = MinMaxScaler()

    # bridge the gap between numpy functions and sklearn's transformer interface
    log_trans = Pipeline([
        ('log_trans', FunctionTransformer(np.log1p)),
        ('scaler', scaler)
    ])
    # column transformation steps
    col_trans = ColumnTransformer([
        ('log_trans', log_trans, config.FEATURES['log_trans']),
        ('scaler', scaler, config.FEATURES['numeric'])
    ], remainder='passthrough')
    return col_trans


def prepare_data(load_path:str='', target_preparing_path:str='', preparing_meta_path:str=''):
    '''
    Data preparing before modeling:
        - feature construction (extraction)
    parameter:
        load_path(str):             where data will be loaded from
        target_preparing_path(str): where preparing data will be stored
        target_meta_path(str):      where metadata will be stored
    '''

    # load data
    X, y =  load_data(load_path)

    # apply feature construction step
    feature_construction(df=X)

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
        'num_samples': X.shape[0],
        'num_features': X.shape[1],
    }
    # save in json file
    with open(preparing_meta_path, 'w') as f:
        json.dump(metadata, f, indent=4)


def sample_save_data(load_prepared_path:str='', load_meta_path:str='',
                     target_sampling_path:str='', sample_meta_path:str=''):
    '''
    - Sample & save data with different techniques: oversampling, undersampling or both
    parameter:
        load_preparing_path(str):   where data will be loaded from
        load_meta_path(str):        where metadat will be loaded from
        target_preparing_path(str): where preparing data will be stored
        sample_meta_path(str):      where metadata will be stored
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

    # list of samples processes
    samplers = {technique_name: sampler_map[technique_name] for technique_name in config.SAMPLING['techniques']}

    # apply all techniques on loaded data and save processed data in files
    for name, sampler in samplers.items():

        x_sampled, y_sampled = sampler.fit_resample(X, y)  # get sampled data

        # save sampled data in .npz file
        sampled = {
            'x_sampled': x_sampled,
            'y_sampled': y_sampled
        }
        np.savez_compressed(f'{target_sampling_path}_{name}.npz', **sampled)

        # save metadata
        sample_metadata = {
            'input_cols_names': X.columns.tolist(),
            'num_samples': X.shape[0],
            'num_features': X.shape[1]
        }
        with open(f'{sample_meta_path}_{name}.json', 'w') as f:
            json.dump(sample_metadata, f, indent=4)


def get_processed_data(sample_technique: (str | None)=None):
    '''
    - get processed data for training & validation
    parameter:
        sample_technique(str | None): get train data according to technique
    return:
        x_train: processed train data
        t_train: ground truth
        x_val: processed val data
        t_val: ground truth
    '''
    if sample_technique is None:
        with open(config.DATASET['prepared']['train_metadata'], 'r') as f:
            metadata = json.load(f)
        cols_names = metadata['input_cols_names']

        data = np.load(config.DATASET['prepared']['train'])
        x_train, t_train = pd.DataFrame(data['x_prepared'], columns=cols_names), pd.Series(data['y_prepared'])

    else:
        # check if technique is passed wrong
        if not sample_technique in config.SAMPLING['techniques']:
            raise ValueError('''sample_technique must be value of these:
             'rus', 'enn', 'smote', 'smoteenn', 'smotetome', 'None' ''')

        with open(f'{config.DATASET['sampled']['train_metadata']}_{sample_technique}.json', 'r') as f:
            metadata = json.load(f)
        cols_names = metadata['input_cols_names']

        data = np.load(f'{config.DATASET['sampled']['train']}_{sample_technique}.npz')
        x_train, t_train = pd.DataFrame(data['x_sampled'], columns=cols_names), pd.Series(data['y_sampled'])

    # load validation data for evaluation
    val_data = np.load(config.DATASET['prepared']['val'])
    x_val, t_val = pd.DataFrame(val_data['x_prepared'], columns=cols_names), pd.Series(val_data['y_prepared'])

    return x_train, x_val, t_train, t_val


if __name__ == '__main__':

    # prepare all data splits
    prepare_data(load_path=config.DATASET['unprocessed']['train'],
                 target_preparing_path=config.DATASET['prepared']['train'],
                 preparing_meta_path=config.DATASET['prepared']['train_metadata'])

    prepare_data(load_path=config.DATASET['unprocessed']['val'],
                 target_preparing_path=config.DATASET['prepared']['val'],
                 preparing_meta_path=config.DATASET['prepared']['val_metadata'])

    prepare_data(load_path=config.DATASET['unprocessed']['train_val'],
                 target_preparing_path=config.DATASET['prepared']['train_val'],
                 preparing_meta_path=config.DATASET['prepared']['train_val_metadata'])

    prepare_data(load_path=config.DATASET['unprocessed']['test'],
                 target_preparing_path=config.DATASET['prepared']['test'],
                 preparing_meta_path=config.DATASET['prepared']['test_metadata'])

    # sample train data
    sample_save_data(load_prepared_path=config.DATASET['prepared']['train'],
                     load_meta_path=config.DATASET['prepared']['train_metadata'],
                     target_sampling_path=config.DATASET['sampled']['train'],
                     sample_meta_path=config.DATASET['sampled']['train_metadata'])

    sample_save_data(load_prepared_path=config.DATASET['prepared']['train_val'],
                     load_meta_path=config.DATASET['prepared']['train_val_metadata'],
                     target_sampling_path=config.DATASET['sampled']['train_val'],
                     sample_meta_path=config.DATASET['sampled']['train_val_metadata'])


