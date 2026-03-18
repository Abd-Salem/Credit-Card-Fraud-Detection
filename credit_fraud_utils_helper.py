import numpy as np
import pandas as pd
import argparse
import json, joblib
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from credit_fraud_utils_eval import avg_pr_fb_score, model_eval_report



def get_processed_train_data(sample_technique:str='none',
                             train_path:str|None=None, train_meta_path:str|None=None):
    '''
    get saved processed data for training
    :param sample_technique: get data according to applied technique
    :param train_path: file path to train dataset
    :param train_meta_path: file path to train metadata
    :return x_train: input
    :return t_train: ground truth
    '''

    if sample_technique == 'none':
        with open(train_meta_path, 'r') as f:
            metadata = json.load(f)
        cols_names = metadata['input_cols_names']

        data = np.load(train_path)
        x_train, t_train = pd.DataFrame(data['x_prepared'], columns=cols_names), pd.Series(data['y_prepared'])

    else:
        with open(train_meta_path, 'r') as f:
            metadata = json.load(f)
        cols_names = metadata['input_cols_names']
        data = np.load(train_path)
        x_train, t_train = pd.DataFrame(data['x_sampled'], columns=cols_names), pd.Series(data['y_sampled'])

    return x_train, t_train


def get_processed_val_data(val_path:str='', metadata_path:str=''):
    '''
    get saved processed data for evaluation
    :param val_path: file path of validation dataset
    :param metadata_path: file path of metadata of validation dataset
    :return x_val: input data
    :return t_val: ground truth
    '''

    # get metadata to get columns names
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    cols_names = metadata['input_cols_names']

    # load validation data for evaluation
    val_data = np.load(val_path)
    x_val, t_val = pd.DataFrame(val_data['x_prepared'], columns=cols_names), pd.Series(val_data['y_prepared'])

    return x_val, t_val


def get_preprocessing_methods(methods:list = ['standard']):
    '''
    get scalers for preprocessing step (tuning on them)
    :return: dict of scalers according to configurations
    '''
    # scalers options
    scalers = {
        'standard' : StandardScaler(),
        'robust'   : RobustScaler(),
        'minmax'   : MinMaxScaler()
    }
    return {scaler:scalers[scaler] for scaler in methods}


def model_eval(model_path:str|None=None, model_eval_path:str|None=None,
               val_path:str|None=None, meta_val_path:str|None=None,
                    show_plot:bool=False,beta:int=2):
    '''
    :param model_path: trained model path
    :param model_eval_path: evaluation path for metrics scores (.json)
    :param val_path: file path of validation dataset
    :param meta_val_path: file path of validation metadata
    :param show_plot: plot precision recall curve with the best threshold
    :param beta: fscore (1, 2, 0.5)
    :return: None
    '''

    # get best model
    model = joblib.load(model_path)

    # get validation dataset
    x_val, t_val = get_processed_val_data(val_path=val_path, metadata_path=meta_val_path)

    # evaluate using avg precision score and f(beta)score and show plot with best threshold
    result = avg_pr_fb_score(model, x_val, t_val, beta=beta, show_plot=show_plot)

    # get classification report by using the best threshold with respect to the highest f(beta)score
    report_1 = model_eval_report(model, x_val, t_val, threshold=result[f'best_threshold(f{beta}-score)'])
    report_2 = model_eval_report(model, x_val, t_val, threshold=0.5)

    # model parameters and eval scores
    metadata = {
        f'results' : result,
        f'classification_report(threshold={result[f'best_threshold(f{beta}-score)']})': report_1,
        f'classification_report(threshold=0.5)' : report_2
    }

    # save model metadata
    with open(model_eval_path, 'w') as f:
        json.dump(metadata, f, indent=4)


def save_best_model(model, metadata, model_path:str|None=None, metadata_path:str|None=None):
    '''
    save best trained model and it's parameters as metadat
    :param model: best model
    :param model_path: where model will be saved
    :param metadata: model parameters
    :param metadata_path: where metadata will be saved
    :return: None
    '''

    # save model
    joblib.dump(model, model_path)

    # save metadata
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)


def parse_arg():
    '''
    argument parser to choose config file, sampling technique, model algorithm, mode(train, eval, full), plotting
    :return None:
    '''

    # create parser
    parser = argparse.ArgumentParser(description='Fraud-Detection-Pipeline')

    # add config argument for path file of configurations
    parser.add_argument('-c','--config', type=str, default='configurations.yaml'
                        ,help='configuration path (.yaml)')

    # add sampling argument for data sampling technique
    parser.add_argument('-s','--sampling', type=str, default='smote',
                        choices =['enn', 'rus', 'smote', 'smoteenn', 'smotetomek', 'none'],
                        help='sampling technique: {'
                             'enn, rus, smote, smoteenn, smotetomek, none'
                             '}')

    # add algorithm argument for model algorithm
    parser.add_argument('-alg', '--algorithm', type=str,
                        choices=['lr', 'rf', 'nn', 'vc'], default='lr',
                        help='model algorithm {'
                             'lr: linear regression / '
                             'rf: random forest / '
                             'nn: neural network / '
                             'vc: voting classifier'
                             '}')

    # add mode argument for processes ( train, eval, full )
    parser.add_argument('-m','--mode', type=str, choices=['train', 'eval', 'full'],
                        default='full', help='pipeline mode (train or eval or full)')

    # add plot argument for plotting precision recall curve show or not
    parser.add_argument('-p', '--plot', type=str, choices=['false','true'],
                        default='false', help='show plot of precision recall curve')

    return parser.parse_args()

