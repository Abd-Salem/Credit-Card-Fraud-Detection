from load_configs import Config
from credit_fraud_utils_helper import load_data, model_eval

def test_best_model(config=Config()):
    '''
    test script for best model
    :param config: configurations
    :return: None
    '''

    # get configs
    model_path = config.MODELS['best_model']['model']
    eval_result_path = config.MODELS['best_model']['eval']
    beta = config.EVALUATION['beta']

    # get test data
    X_test, t_test = load_data(config.DATASET['unprocessed']['test'])

    #TODO   F1-score of the best model


if __name__ == '__main__':
    test_best_model()