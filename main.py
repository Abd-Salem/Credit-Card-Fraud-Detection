from credit_fraud_utils_helper import parse_arg, model_eval, get_processed_train_data
from credit_fraud_train import logistic_regression_model, random_forest_model, neural_network_classifier, voting_classifier
from configs import Config


def main():
    # train dict models
    train = {
        'lr' : logistic_regression_model,
        'rf' : random_forest_model,
        'nn' : neural_network_classifier,
        'vc' : voting_classifier
    }
    # evaluation dict models
    eval = {
        'lr' : 'logistic_regression',
        'rf' : 'random_forest',
        'nn' : 'neural_network',
        'vc' : 'voting_classifier'
    }
    # argument parser
    arg = parse_arg()
    print(arg)

    # load configurations
    config = Config(arg.config)

    # check for different dirs(prepared or sampled)
    if arg.sampling == 'none':
        x_train, t_train = get_processed_train_data(sample_technique=arg.sampling,
                                                    train_path=config.DATASET['prepared']['train']['data'],
                                                    train_meta_path=config.DATASET['prepared']['train']['metadata'])
    else:
        x_train, t_train = get_processed_train_data(sample_technique=arg.sampling,
                                                    train_path=config.DATASET['sampled'][arg.sampling]['train'],
                                                    train_meta_path=config.DATASET['sampled'][arg.sampling]['train_metadata'])
    # check if plot is true
    show_plot = False
    if arg.plot.lower() == 'true':
        show_plot = True

    # check user choices
    if arg.mode == 'train':
        train[arg.algorithm](x_train=x_train,t_train=t_train,
                             sample_technique=arg.sampling, config=config)

    elif arg.mode == 'eval':
        model_eval(model_path=config.MODELS[eval[arg.algorithm]]['sample'][arg.sampling]['model'],
                   model_eval_path=config.MODELS[eval[arg.algorithm]]['sample'][arg.sampling]['eval'],
                   val_path=config.DATASET['prepared']['val']['data'],
                   meta_val_path=config.DATASET['prepared']['val']['metadata'],
                   show_plot=show_plot, beta=config.EVALUATION['beta'])

    elif arg.mode == 'full':
        train[arg.algorithm](x_train=x_train,t_train=t_train,
                             sample_technique=arg.sampling, config=config)
        model_eval(model_path=config.MODELS[eval[arg.algorithm]]['sample'][arg.sampling]['model'],
                   model_eval_path=config.MODELS[eval[arg.algorithm]]['sample'][arg.sampling]['eval'],
                   val_path=config.DATASET['prepared']['val']['data'],
                   meta_val_path=config.DATASET['prepared']['val']['metadata'],
                   show_plot=show_plot, beta=config.EVALUATION['beta'])



if __name__ == '__main__':
    main()