from credit_fraud_utils_helper import parse_arg, model_eval
from credit_fraud_train import logistic_regression_model, random_forest_model, neural_network_classifier, \
    neural_network_fl, voting_classifier_1,voting_classifier_2,voting_classifier_3, knn_classifier
from credit_fraud_utils_helper import load_data, compare_evals_retrain_best_model
import torch
import numpy as np
from load_configs import Config


def main():
    # train dict models
    train = {
        'lr'    : logistic_regression_model,
        'rf'    : random_forest_model,
        'nn'    : neural_network_classifier,
        'nn_fl' : neural_network_fl,
        'knn'   : knn_classifier,
        'vc1'   : voting_classifier_1,
        'vc2'   : voting_classifier_2,
        'vc3'   : voting_classifier_3,
    }
    # evaluation dict models
    eval = {
        'lr'    : 'logistic_regression',
        'rf'    : 'random_forest',
        'nn'    : 'neural_network',
        'nn_fl' : 'neural_network_fl',
        'knn'   : 'knn_classifier',
        'vc1'   : 'voting_classifier_1',
        'vc2'   : 'voting_classifier_2',
        'vc3'   : 'voting_classifier_3'

    }

    # argument parser
    arg = parse_arg()
    print(arg)

    # load configurations
    config = Config(arg.config)

    # reproducibility
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)


    # check if plot is true
    show_plot = False
    if arg.plot.lower() == 'true':
        show_plot = True


    # check user choices
    if arg.mode == 'train':
        train[arg.algorithm](sample_technique=arg.sampling, config=config)

    elif arg.mode == 'eval':
        X_val, t_val = load_data(path=config.DATASET['unprocessed']['val'])
        model_eval(X=X_val, t=t_val,
                   model_path=config.MODELS[eval[arg.algorithm]]['sample'][arg.sampling]['model'],
                   eval_result_path=config.MODELS[eval[arg.algorithm]]['sample'][arg.sampling]['eval'],
                   show_plot=show_plot, beta=config.EVALUATION['beta'],
                   plot_path=f'{config.EVALUATION['dir']}/pr-curve_{eval[arg.algorithm]}_{arg.sampling}.png')

        compare_evals_retrain_best_model(config=config)


    elif arg.mode == 'full':
        train[arg.algorithm](sample_technique=arg.sampling, config=config)

        X_val, t_val = load_data(path=config.DATASET['unprocessed']['val'])
        model_eval(X=X_val, t=t_val,
                   model_path=config.MODELS[eval[arg.algorithm]]['sample'][arg.sampling]['model'],
                   eval_result_path=config.MODELS[eval[arg.algorithm]]['sample'][arg.sampling]['eval'],
                   show_plot=show_plot, beta=config.EVALUATION['beta'],
                   plot_path=f'{config.EVALUATION['dir']}/pr-curve_{eval[arg.algorithm]}_{arg.sampling}.png')

        compare_evals_retrain_best_model(config=config)

if __name__ == '__main__':
    main()