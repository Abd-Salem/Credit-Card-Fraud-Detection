import numpy as np
import argparse
import json, joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from credit_fraud_utils_eval import avg_pr_fb_score, model_eval_report



def get_processed_data(data_path:str|None=None, dtype='np', meta_path:str|None=None):
    '''
    load prepared or prepared and sampled data (train, val, test)
    :param data_path: path of file
    :param dtype: indicate the datatype of returned data (df , np )
    :param meta_path: path to metadata to get columns names saved when data prepared
    :return: X, t ( input & target )
    '''
    data = np.load(data_path)
    X, t = data['x'], data['y']

    if dtype == 'df':
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        cols_names = meta['cols_names']
        X, t = pd.DataFrame(X, columns=cols_names), pd.Series(t)

    return X, t


def get_scaling_method(scalers_names:list=['standard']):
    '''
    return the intended scalers inside configs
    :param scalers_names: scaler names
    '''
    scalers = {
        'standard': StandardScaler(),
        'robust':   RobustScaler(),
        'minmax':   MinMaxScaler(),
    }
    for name in scalers_names:
        if not name in scalers.keys():
            raise ValueError("Invalid scaler name, value must be in list ['standard', 'robust', 'minmax']")

    return [scalers[scaler] for scaler in scalers_names]        # return scalers


def model_eval(val_data_path:str=None, val_meta_path:str=None,model_path:str|None=None ,model_eval_path:str|None=None,
               show_plot:bool=False,beta:int=2):
    '''
    :param val_data_path: path to validation dataset
    :param val_meta_path: path to metadata saved for validation dataset ( columns names )
    :param model_path: trained model path
    :param model_eval_path: evaluation path for metrics scores (.json)
    :param show_plot: plot precision recall curve with the best threshold
    :param beta: fscore (1, 2, 0.5)
    :return: None
    '''

    # get best model
    model = joblib.load(model_path)

    x_val, t_val = get_processed_data(data_path=val_data_path, dtype='df', meta_path=val_meta_path)

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


def save_best_model(model, metadata, model_path:str|None=None,metadata_path:str|None=None):
    '''
    save best trained model and it's parameters as metadat
    :param model: trained model
    :param metadata: model's parameters
    :param model_path: where model will be saved
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
    parser.add_argument('-c','--config', type=str, default='configs.yml'
                        ,help='configuration path (.yaml)')

    # add sampling argument for data sampling technique
    parser.add_argument('-s','--sampling', type=str, default='smote',
                        choices =['enn', 'rus', 'smote', 'smoteenn', 'smotetomek', 'none'],
                        help='sampling technique: {'
                             'enn, rus, smote, smoteenn, smotetomek, none'
                             '}')

    # add algorithm argument for model algorithm
    parser.add_argument('-alg', '--algorithm', type=str,
                        choices=['lr', 'rf', 'nn','nn_fl','knn', 'vc'], default='lr',
                        help='model algorithm {'
                             'lr: linear regression / '
                             'rf: random forest / '
                             'nn: neural network / '
                             'nn_fl: neural network with Focal Loss  /  '
                             'knn: knn classifier       /   '
                             'vc: voting classifier'
                             '}')

    # add mode argument for processes ( train, eval, full )
    parser.add_argument('-m','--mode', type=str, choices=['train', 'eval', 'full'],
                        default='full', help='pipeline mode (train or eval or full)')

    # add plot argument for plotting precision recall curve show or not
    parser.add_argument('-p', '--plot', type=str, choices=['false','true'],
                        default='false', help='show plot of precision recall curve')

    return parser.parse_args()

