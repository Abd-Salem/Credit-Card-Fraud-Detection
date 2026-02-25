import pandas as pd
import numpy as np
import config
from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN, SMOTETomek
from collections import Counter

def load_data(path: str):
    '''Load dataset and split to X, y'''
    df = pd.read_csv(path)
    X = df.drop(config.DATASET['target_feature'], axis=1)
    y = df[config.DATASET['target_feature']]
    return X, y


def feature_construction(df:pd.DataFrame, * ,inplace:bool =False):
    '''
    Extracting features from existence features

    parameter:
        df (DataFrame): dataset
        inplace (bool): default False
            If False, return a copy. Otherwise, do operation
            in place and return None
    return:
     extracted_features_names: columns names
    '''

    # raise error if inplace isn't bool
    if not isinstance(inplace, bool):
        raise TypeError('Inplace isn\'t boolean')

    if inplace is False:         # check inplace bool type
        new_df = df.copy(deep=True)
        df = new_df         # original dataframe will not change

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

    # add extracted features to configuration
    extracted_numeric_features = ['hour', 'minute', 'second']


    if inplace is False:         # check for different returns
        return df           # new dataframe
    else:
        if config.DATASET['prepared_once'] is False:        # if numeric features and input features is updated once don't do this again
            config.DATASET['numeric_features'].extend(extracted_numeric_features)
            config.DATASET['input_features'] = df.columns.tolist()
        return None             # change in the original dataframe


def feature_transformation(preprocessing_type='standard'):
    '''
    Column transformation methods:
        1- log transformation for skewed data ('Amount', 'V8', 'V28')
        2- standard scaling for numeric features
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
        ('log_trans', log_trans, config.DATASET['log_cols']),
        ('scaler', scaler, config.DATASET['numeric_features'])
    ], remainder='passthrough')
    return col_trans


def prepare_data(df:pd.DataFrame, * , inplace=False , preprocessing_type='standard'):
    '''
    Data preparing before modeling:
        1- feature construction (extraction)
        2- feature transformation
    parameter:
        df (DataFrame): dataset
        numeric_features (list): numeric column names for numeric transformations
        categorical_features (list): categoric column names for categorical transformations
        log_trans_cols (list): columns names for log transformation
    return:
        df (DataFrame): dataset
        input_cols_name (list): names of engineered columns which are ready for modeling step
        col_trans (ColumnTransformer): column transformation method for pipelining
    '''

    # raise error if inplace isn't bool
    if not isinstance(inplace, bool):
        raise TypeError('Inplace isn\'t boolean')

    # feature extraction step
    if inplace is False:
        df = feature_construction(df=df, inplace=inplace)
    else:
        feature_construction(df=df, inplace=inplace)

    col_trans = feature_transformation(preprocessing_type=preprocessing_type)    # column transformation step

    # return depends on inplace bool type
    if inplace is False:
        return df, col_trans
    return col_trans

def sample_data(y, *, technique:str='None', sample_strategy='auto'):
    '''
    - Sampling data with different techniques like oversampling, undersampling or both
    parameter:
        technique (str): type of sampling
        factor (int): factor for scaling sample size
    return:
        None: if keyword string doesn't match any
        x_sampled & y_sampled according to chosen technique
    '''

    count = Counter(y)  # for counting samples number of each class
    sampler = None      # sampler technique

    # over sampling technique using SMOTE
    if technique == 'oversampling':
        sampler = SMOTE(sampling_strategy=sample_strategy, k_neighbors=5, random_state=config.RANDOM_STATE)

    # under sampling technique using RandomUnderSampler
    elif technique == 'undersampling':
        sampler = RandomUnderSampler(sampling_strategy=sample_strategy, random_state=config.RANDOM_STATE)

    # both under and over sampling techniques using SMOTE & RandomUnderSampler
    elif technique == 'smoteenn':
        sampler = SMOTEENN(sampling_strategy=sample_strategy, random_state=config.RANDOM_STATE)

    elif technique == 'smotetomek':
        sampler = SMOTETomek(sampling_strategy=sample_strategy, random_state=config.RANDOM_STATE)

    return sampler     # None if nothing