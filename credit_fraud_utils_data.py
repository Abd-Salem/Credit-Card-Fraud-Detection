import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer
from sklearn.pipeline import  Pipeline
from sklearn.compose import ColumnTransformer


def load_data(path: str):
    '''Load dataset'''
    df = pd.read_csv(path)
    return df


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

    if not inplace:         # check inplace bool type
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

    # columns names
    extracted_numeric_features = ['hour', 'minute', 'second']

    if not inplace:         # check for different returns
        return df, extracted_numeric_features       # change in new dataframe
    return extracted_numeric_features           # change in the original dataframe


def feature_transformation(*, numeric_features:list =None, categorical_features:list =None, log_trans_cols:list =None):
    '''
    Column transformation methods:
        1- log transformation for skewed data ('Amount', 'V8', 'V28')
        2- standard scaling for numeric features
    parameter:
        numeric_features (list): numeric column names for numeric transformations
        categorical_features (list): categoric column names for categorical transformations
        log_trans_cols (list): columns names for log transformation
    return:
        col_trans (ColumnTransformation): column transformation method for pipelining
    '''

    # bridge the gap between numpy functions and sklearn's transformer interface
    log_trans = Pipeline([
        ('log_trans', FunctionTransformer(np.log1p)),
        ('scaler', MinMaxScaler())
    ])

    # column transformation steps
    col_trans = ColumnTransformer([
        ('log_trans', log_trans, log_trans_cols),
        ('scaler', MinMaxScaler(), numeric_features)
    ], remainder='passthrough')
    return col_trans


def prepare_data(df:pd.DataFrame, * , inplace=False ,numeric_features:list =None,
                 categorical_features:list =None, log_trans_cols:list =None):
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

    # feature extraction step
    if not inplace:
        df, extracted_numeric_features = feature_construction(df=df, inplace=inplace)
    else:
        extracted_numeric_features = feature_construction(df=df, inplace=inplace)

    numeric_features.extend(extracted_numeric_features)     # adding new numeric features
    col_trans = feature_transformation(numeric_features=numeric_features,
                                       log_trans_cols=log_trans_cols)    # column transformation step

    # return depends on inplace bool type
    if not inplace:
        return df, col_trans
    return col_trans
