from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from imblearn.pipeline import  Pipeline
from config import config
from credit_fraud_utils_data import feature_transformation, get_processed_data
from credit_fraud_utils_eval import model_eval_report, avg_pr_fb_score
from collections import Counter
import json, joblib


def logistic_regression_model(sample_technique:(str | None) = None):
    '''
    Train Logistic Regression Model
    parameter:
        sample_technique(str|None): technique which data processed with
    '''

    x_train, x_val, t_train, t_val = get_processed_data(sample_technique=sample_technique)

    # stratified is a good way to keep class distribution as it is in each fold.
    skf = StratifiedKFold(n_splits=config.EVALUATION['cv_folds'], shuffle=True, random_state=config.RANDOM_STATE)

    # pipelining feature engineering, sampling and model training
    pipeline = Pipeline([
        ('col_trans', feature_transformation(preprocessing_type=config.PREPROCESSING['scaler'])),
        ('model', LogisticRegression(random_state=config.RANDOM_STATE))
    ])

    # classes ratio
    count = Counter(t_train)
    ratio = count[1]/count[0]

    # grid parameters
    param_grid = {
        'model__solver':            config.MODELS['logistic_regression']['params']['solver'],
        'model__class_weight':      [{1:wt} for wt in config.MODELS['logistic_regression']['params']['class_weight']+[ratio]],
        'model__max_iter':          config.MODELS['logistic_regression']['params']['max_iter']
    }

    # apply grid search with different parameters and get the best estimator
    grid = GridSearchCV(pipeline,param_grid=param_grid,
                        scoring=config.EVALUATION['scoring'], cv=skf, n_jobs=3)

    grid.fit(x_train, t_train)                  # fit model
    model = grid.best_estimator_                # best model

    # save model
    joblib.dump(model, config.MODELS['logistic_regression']['model'])

    # evaluate using avg precision score and f-beta metrics and show plot with best threshold
    result = avg_pr_fb_score(model, x_val, t_val, beta=config.EVALUATION['beta'], show_plot=False)

    # classification report using best threshold given from evaluation
    report_1 = model_eval_report(model, x_val, t_val, threshold=result[f'best_threshold(f{config.EVALUATION['beta']}-score)'])
    report_2 = model_eval_report(model, x_val, t_val, threshold=0.5)

    metadata = {
        'model_params': grid.best_params_,
        f'results' : result,
        f'classification_report(threshold={result[f'best_threshold(f{config.EVALUATION['beta']}-score)']})': report_1,
        f'classification_report(threshold=0.5)' : report_2
    }

    # save model metadata
    with open(config.MODELS['logistic_regression']['metadata'], 'w') as f:
        json.dump(metadata, f, indent=4)



def random_forest_model(sample_technique:(str | None) = None):
    '''
    - train random forest model
    parameter:
        sample_technique(list | None): technique which data processed with
    '''

    # get processed data
    x_train, x_val, t_train, t_val = get_processed_data(sample_technique=sample_technique)

    # stratified is a good way to keep class distribution as it is in each fold.
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)

    # pipelining feature engineering, sampling and model training
    pipeline = Pipeline([
        ('data_preprocessing', feature_transformation(preprocessing_type=config.PREPROCESSING['scaler'])),
        ('model', RandomForestClassifier(random_state=config.RANDOM_STATE ,n_jobs=3))
    ])

    params = {
        'model__max_depth':         config.MODELS['random_forest']['params']['max_depth'],
        'model__n_estimators':      config.MODELS['random_forest']['params']['n_estimators'],
        'model__min_samples_leaf':  config.MODELS['random_forest']['params']['min_samples_leaf'],
        'model__class_weight':      [{1 : wt} for wt in config.MODELS['random_forest']['params']['class_weight']]
    }

    rand_grid = RandomizedSearchCV(pipeline, param_distributions=params,
                        n_iter=config.MODELS['random_forest']['n_iter'],scoring=config.EVALUATION['scoring'],
                              cv=skf, n_jobs=3)

    rand_grid.fit(x_train, t_train)           # best parameters
    model = rand_grid.best_estimator_                # best model

    # save model
    joblib.dump(model, config.MODELS['random_forest']['model'])

    # evaluate using avg precision score and f-beta metrics and show plot with best threshold
    result = avg_pr_fb_score(model, x_val, t_val, beta=config.EVALUATION['beta'], show_plot=False)

    # classification report using best threshold given from evaluation
    report_1 = model_eval_report(model, x_val, t_val, threshold=result[f'best_threshold(f{config.EVALUATION['beta']}-score)'])
    report_2 = model_eval_report(model, x_val, t_val, threshold=0.5)


    # save model metadata
    metadata = {
        'model_params': rand_grid.best_params_,
        f'results' : result,
        f'classification_report(threshold={result[f'best_threshold(f{config.EVALUATION['beta']}-score)']})': report_1,
        'classification_report(threshold=0.5)' : report_2
    }

    with open(config.MODELS['random_forest']['metadata'], 'w') as f:
        json.dump(metadata, f, indent=4)



def neural_network_classifier(sample_technique:(str|None) = None):

    '''
    - Train neural network model (MLPClassifier)
    parameter:
        sample_technique(list | None): technique which data processed with
    '''

    # get processed data
    x_train, x_val, t_train, t_val = get_processed_data(sample_technique=sample_technique)

    # stratified is a good way to keep class distribution as it is in each fold.
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)

    # pipelining feature engineering, sampling and model training
    pipeline = Pipeline([
        ('data_preprocessing', feature_transformation(preprocessing_type=config.PREPROCESSING['scaler'])),
        ('model', MLPClassifier(random_state=config.RANDOM_STATE))
    ])

    params = {
        'model__hidden_layer_sizes':         [(32,),(64,), (64, 32),(128, 64),(128, 64, 32)],
        'model__activation':                  config.MODELS['neural_network']['params']['activation'],
        'model__alpha':                       config.MODELS['neural_network']['params']['alpha'],
        'model__learning_rate_init':          config.MODELS['neural_network']['params']['learning_rate'],
        'model__batch_size':                  config.MODELS['neural_network']['params']['batch_size'],
        'model__max_iter':                    config.MODELS['neural_network']['params']['max_iter']
    }

    rand_grid = RandomizedSearchCV(pipeline, param_distributions=params,
                        n_iter=config.MODELS['neural_network']['n_iter'],scoring=config.EVALUATION['scoring'],
                              cv=skf, n_jobs=3)

    rand_grid.fit(x_train, t_train)           # best parameters
    model = rand_grid.best_estimator_                # best model


    # save model
    joblib.dump(model, config.MODELS['neural_network']['model'])

    # evaluate using avg precision score and f-beta metrics and show plot with best threshold
    result = avg_pr_fb_score(model, x_val, t_val, beta=config.EVALUATION['beta'], show_plot=False)

    # classification report using best threshold given from evaluation
    report_1 = model_eval_report(model, x_val, t_val, threshold=result[f'best_threshold(f{config.EVALUATION['beta']}-score)'])
    report_2 = model_eval_report(model, x_val, t_val, threshold=0.5)


    # save model metadata
    metadata = {
        'model_params': rand_grid.best_params_,
        f'results' : result,
        f'classification_report(threshold={result[f'best_threshold(f{config.EVALUATION['beta']}-score)']})': report_1,
        'classification_report(threshold=0.5)' : report_2
    }

    with open(config.MODELS['neural_network']['metadata'], 'w') as f:
        json.dump(metadata, f, indent=4)


def voting_classifier(sample_technique:(str|None)=None):

    # load trained models
    lr_model = joblib.load(config.MODELS['logistic_regression']['model'])
    rf_model = joblib.load(config.MODELS['random_forest']['model'])
    nn_model = joblib.load(config.MODELS['neural_network']['model'])

    # prepare estimators
    names = config.MODELS['voting_classifier']['params']['estimators']
    models = [lr_model, rf_model, nn_model]
    estimators = list(zip(names, models))

    # get prepared data not sampled
    x_train,x_val, t_train, t_val = get_processed_data(sample_technique=sample_technique)

    # voting classifier
    voting = VotingClassifier(estimators=estimators, voting=config.MODELS['voting_classifier']['params']['voting'],
                              weights=config.MODELS['voting_classifier']['params']['weights'])

    # fit model
    voting.fit(x_train, t_train)

    # save model
    joblib.dump(voting, config.MODELS['voting_classifier']['model'])

    # evaluate using avg precision score and f-beta metrics and show plot with best threshold
    result = avg_pr_fb_score(voting, x_val, t_val, beta=config.EVALUATION['beta'], show_plot=False)

    # classification report using best threshold given from evaluation
    report_1 = model_eval_report(voting, x_val, t_val, threshold=result[f'best_threshold(f{config.EVALUATION['beta']}-score)'])
    report_2 = model_eval_report(voting, x_val, t_val, threshold=0.5)

    # save model metadata
    metadata = {
        f'results' : result,
        f'classification_report(threshold={result[f'best_threshold(f{config.EVALUATION['beta']}-score)']})': report_1,
        'classification_report(threshold=0.5)': report_2
    }
    with open(config.MODELS['voting_classifier']['metadata'], 'w') as f:
        json.dump(metadata, f, indent=4)


if __name__ == '__main__':
    # prepare models for training
    train_models = [
        logistic_regression_model, random_forest_model,
        neural_network_classifier, voting_classifier
    ]
    # run training for all models
    for model in train_models:
        model(sample_technique='smoteenn')
