import joblib, json
from sklearn.metrics import f1_score, average_precision_score
from credit_fraud_utils_helper import load_data
from load_configs import Config

def test_best_model(config=None):
    '''
    test script for best model
    :param config: configurations
    :return: None
    '''

    # get configs
    model_path = config.MODELS['best_model']['model']
    model = joblib.load(model_path)

    # get test data
    X_test, t_test = load_data(config.DATASET['unprocessed']['test'])

    # get test scores
    y_pred = model.predict(X_test)
    f1_standard = f1_score(t_test, y_pred)

    # prepare for testing over different thresholds searching for best f1-score for production

    thresholds = {
        'th_1' : 0.55,
        'th_2' : 0.6,
        'th_3' : 0.65,
        'th_4' : 0.7
    }

    f1_scores = {
                 f'f1_test_1(th={thresholds['th_1']})': f1_score(t_test, (model.predict_proba(X_test)[:, 1] >= thresholds['th_1'])),
                 f'f1_test_2(th={thresholds['th_2']})': f1_score(t_test, (model.predict_proba(X_test)[:, 1] >= thresholds['th_2'])),
                 f'f1_test_3(th={thresholds['th_3']})': f1_score(t_test, (model.predict_proba(X_test)[:, 1] >= thresholds['th_3'])),
                 f'f1_test_4(th={thresholds['th_3']})': f1_score(t_test, (model.predict_proba(X_test)[:, 1] >= thresholds['th_4']))
    }

    y_probs = model.predict_proba(X_test)[:, 1]
    auprc = average_precision_score(t_test, y_probs)

    test_scores = {
        'AUPRC' : auprc,
        'f1-score(th=0.5)' : f1_standard,
        **f1_scores
    }

    with open(config.MODELS['best_model']['eval'], 'w') as f:
        json.dump(test_scores, f, indent=4)



if __name__ == '__main__':
    test_best_model(config=Config())