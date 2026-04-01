from credit_fraud_utils_helper import parse_arg, model_eval
from credit_fraud_train import logistic_regression_model, random_forest_model, neural_network_classifier, \
    neural_network_fl, voting_classifier, knn_classifier
import torch
import numpy as np
from load_configs import Config


def main():
    # train dict models
    train = {
        'lr' : logistic_regression_model,
        'rf' : random_forest_model,
        'nn' : neural_network_classifier,
        'nn_fl' : neural_network_fl,
        'knn': knn_classifier,
        'vc' : voting_classifier
    }
    # evaluation dict models
    eval = {
        'lr' : 'logistic_regression',
        'rf' : 'random_forest',
        'nn' : 'neural_network',
        'nn_fl' : 'neural_network_fl',
        'knn': 'knn_classifier',
        'vc' : 'voting_classifier'
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
        model_eval(val_data_path=config.DATASET['prepared']['val']['data'],val_meta_path=config.DATASET['prepared']['val']['metadata'],
                   model_path=config.MODELS[eval[arg.algorithm]]['sample'][arg.sampling]['model'],
                   model_eval_path=config.MODELS[eval[arg.algorithm]]['sample'][arg.sampling]['eval'],
                   show_plot=show_plot, beta=config.EVALUATION['beta'])

    elif arg.mode == 'full':
        train[arg.algorithm](sample_technique=arg.sampling, config=config)
        model_eval(val_data_path=config.DATASET['prepared']['val']['data'],val_meta_path=config.DATASET['prepared']['val']['metadata'],
                   model_path=config.MODELS[eval[arg.algorithm]]['sample'][arg.sampling]['model'],
                   model_eval_path=config.MODELS[eval[arg.algorithm]]['sample'][arg.sampling]['eval'],
                   show_plot=show_plot, beta=config.EVALUATION['beta'])


if __name__ == '__main__':
    main()