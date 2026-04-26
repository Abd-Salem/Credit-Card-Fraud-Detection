import numpy as np
import argparse
import json, joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from credit_fraud_utils_eval import avg_pr_fb_score, model_eval_report
from load_configs import Config

def load_data(path: str, target_col_name='Class'):
    '''
    Load dataset & split to X, y
    :param path: data path
    :param target_col_name: columns name
    :return X: input
    :return y: ground truth
    '''
    df = pd.read_csv(path)
    X = df.drop(target_col_name, axis=1)
    y = df[target_col_name]
    return X, y


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


def model_eval(X, t, model_path:str|None=None ,eval_result_path:str|None=None,
               beta:int=2, show_plot:bool=False, plot_path=None):
    '''
    :param X: input validation
    :param t: ground truth
    :param model_path: trained model path
    :param eval_result_path: evaluation path for metrics scores (.json)
    :param beta: fscore (1, 2, 0.5)
    :param show_plot: plot precision recall curve with the best threshold
    :param plot_path: path of plot picture
    :return: None
    '''

    # get best model
    model = joblib.load(model_path)

    # evaluate using avg precision score and f(beta)score and show plot with best threshold
    result = avg_pr_fb_score(model, X, t, beta=beta, show_plot=show_plot, plot_path=plot_path)

    # get classification report by using the best threshold with respect to the highest f(beta)score
    report_1 = model_eval_report(model, X, t, threshold=result[f'best_threshold(f{beta}-score)'])
    report_2 = model_eval_report(model, X, t, threshold=0.5)

    # model parameters and eval scores
    metadata = {
        f'results' : result,
        f'classification_report(threshold={result[f'best_threshold(f{beta}-score)']})': report_1,
        f'classification_report(threshold=0.5)' : report_2
    }

    # save model metadata
    with open(eval_result_path, 'w') as f:
        json.dump(metadata, f, indent=4)

def save_best_model(model, metadata, model_path:str|None=None, metadata_path:str|None=None):
    '''
    save best trained model and it's parameters as metadat
    :param model: trained model
    :param metadata: parameters
    :param model_path: where model will be saved
    :param metadata_path: where metadata will be saved
    :return: None
    '''

    # save model
    joblib.dump(model, model_path)

    # save metadata
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)



def compare_evals_retrain_best_model(config=None):
    '''
    1- Compare all evaluations for all models and get model with best f1-score
    2- Retrain best model on full dataset (train + val)
    3- Save retrained model & it's metadata (best model dir)
    :param config: configurations
    :return: None
    '''

    # models and samples
    models = [
        'logistic_regression', 'random_forest', 'neural_network',
        'neural_network_fl', 'knn_classifier','voting_classifier_1',
        'voting_classifier_2', 'voting_classifier_3'
    ]
    samples = [
        'rus', 'enn', 'smote',
        'smoteenn', 'smotetomek'
    ]

    # initial values
    best_f1_score = 0
    best_sample = ''
    best_model = ''
    eval_data = ''

    for model in models:
        for sample in samples:
            eval_path = config.MODELS[model]['sample'][sample]['eval']

            # load eval json file
            with open(eval_path, 'r') as f:
                eval = json.load(f)

            # check for better f1-score
            if float(eval['classification_report(threshold=0.5)']['1']['f1-score']) > best_f1_score:
                best_f1_score = float(eval['classification_report(threshold=0.5)']['1']['f1-score'])
                best_model = model
                best_sample = sample
                eval_data = eval

    # load model and it's metadata
    model_path = config.MODELS[best_model]['sample'][best_sample]['model']
    meta_path = config.MODELS[best_model]['sample'][best_sample]['metadata']
    best_model = joblib.load(model_path)    # best model
    with open(meta_path, 'r') as f:
        metadata = json.load(f)

    # load sampled full dataset (train + val)
    X, t = get_processed_data(data_path=config.DATASET['sampled'][best_sample]['train_val']['data'],
                              dtype='df', meta_path=config.DATASET['sampled'][best_sample]['train_val']['metadata'])
    best_model.fit(X, t)    #refit on full dataset

    all_data = {
        **metadata,
        **eval_data
    }

    # save best model that trained on full dataset (train + val)
    joblib.dump(best_model, config.MODELS['best_model']['model'])
    with open(config.MODELS['best_model']['metadata'], 'w') as f:
        json.dump(all_data, f, indent=4)



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
                        choices=['lr', 'rf', 'nn','nn_fl','knn', 'vc1', 'vc2', 'vc3'], default='lr',
                        help='model algorithm {'
                             'lr: linear regression / '
                             'rf: random forest / '
                             'nn: neural network / '
                             'nn_fl: neural network with Focal Loss  /  '
                             'knn: knn classifier       /   '
                             'vc1: voting classifier 1'
                             'vc2: voting classifier 2'
                             'vc3: voting classifier 3'
                             '}')

    # add mode argument for processes ( train, eval, full )
    parser.add_argument('-m','--mode', type=str, choices=['train', 'eval', 'full'],
                        default='full', help='pipeline mode (train or eval or full)')

    # add plot argument for plotting precision recall curve show or not
    parser.add_argument('-p', '--plot', type=str, choices=['false','true'],
                        default='false', help='show plot of precision recall curve')

    return parser.parse_args()

