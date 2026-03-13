import numpy as np
import pandas as pd
import argparse
import json, joblib
from configs import config
from credit_fraud_utils_eval import avg_pr_fb_score, model_eval_report

def get_processed_train_data(sample_technique: (str | None)=None):
    '''
    get saved processed data for training
    :param sample_technique: get data according to applied technique
    :return x_train: input
    :return t_train: ground truth
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

    return x_train, t_train


def get_processed_val_data():
    '''
    get saved processed data for evaluation
    :return x_val: input data
    :return t_val: ground truth
    '''

    # get metadata to get columns names
    with open(config.DATASET['prepared']['val_metadata'], 'r') as f:
        metadata = json.load(f)
    cols_names = metadata['input_cols_names']

    # load validation data for evaluation
    val_data = np.load(config.DATASET['prepared']['val'])
    x_val, t_val = pd.DataFrame(val_data['x_prepared'], columns=cols_names), pd.Series(val_data['y_prepared'])

    return x_val, t_val


def model_eval(model, x, t, *,
                    model_eval_path:str|None=None,
                    show_plot:bool=False, beta:int=2):
    '''
    :param model: trained model
    :param x: validation input
    :param t: validation ground truth
    :param model_eval_path: saving path file for eval metrics scores (.json)
    :param show_plot: plot precision recall curve
    :param beta: fscore (1, 2, 0.5)
    :return: None
    '''

    # evaluate using avg precision score and f(beta)score and show plot with best threshold
    result = avg_pr_fb_score(model, x, t, beta=beta, show_plot=show_plot)

    # get classification report by using the best threshold with respect to the highest f(beta)score
    report_1 = model_eval_report(model, x, t, threshold=result[f'best_threshold(f{config.EVALUATION['beta']}-score)'])
    report_2 = model_eval_report(model, x, t, threshold=0.5)

    # model parameters and eval scores
    metadata = {
        f'results' : result,
        f'classification_report(threshold={result[f'best_threshold(f{config.EVALUATION['beta']}-score)']})': report_1,
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
    parser = argparse.ArgumentParser(description='Fraud-Detection-Pipeline')

    parser.add_argument('-c','--config', type=str, default='configurations.yaml'
                        ,help='configuration path (.yaml)')

    parser.add_argument('-s','--sampling', type=str, default=None,
                        choices =['enn', 'rus', 'smote', 'smoteenn', 'smotetomek', 'None'],
                        help='sampling technique: {'
                             'enn, rus, smote, smoteenn, smotetomek, None'
                             '}')

    parser.add_argument('-alg', '--algorithm', type=str,
                        choices=['lr', 'rf', 'nn', 'vc'], default='lr',
                        help='model algorithm {'
                             'lr: linear regression / '
                             'rf: random forest / '
                             'nn: neural network / '
                             'vc: voting classifier'
                             '}')

    parser.add_argument('-m','--mode', type=str, choices=['train', 'eval', 'full'],
                        default='full', help='pipeline mode (train or eval or full)')



    parser.add_argument('-o', '--output', type=str, default=None, help='output trained model dir')

    return parser.parse_args()

