from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV, cross_val_score
from imblearn.pipeline import  Pipeline
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from credit_fraud_utils_helper import get_processed_train_data, get_preprocessing_methods, save_best_model
import json, joblib
import pandas as pd


def logistic_regression_model(x_train:pd.DataFrame, t_train:pd.Series, *,
                              sample_technique:str='none', config=None):
    '''
    Train logistic regression model
    :param x_train: input features
    :param t_train: target feature
    :param sample_technique: which data processed with
    :param config: config loader
    :return: None
    '''

    # stratified is a good way to keep class distribution as it is in each fold.
    skf = StratifiedKFold(n_splits=config.EVALUATION['cv_folds'], shuffle=True, random_state=config.RANDOM_STATE)

    # get scalers
    scalers = get_preprocessing_methods()

    # pipelining feature engineering, sampling and model training
    pipeline = Pipeline([
        ('scaler', scalers['standard']),
        ('model', LogisticRegression(random_state=config.RANDOM_STATE))
    ])

    # get classes ratio and add it to class weights
    count = Counter(t_train)
    ratio = count[1]/count[0]
    class_weight = config.MODELS['logistic_regression']['params']['class_weight']
    class_weight.append({1:1, 0:ratio})

    # grid parameters
    param_grid = {
        'scaler':                   list(scalers.values()),
        'model__solver':            config.MODELS['logistic_regression']['params']['solver'],
        'model__class_weight':      class_weight,
        'model__max_iter':          config.MODELS['logistic_regression']['params']['max_iter']
    }

    # apply grid search with different parameters and get the best estimator
    grid = GridSearchCV(pipeline,param_grid=param_grid,
                        scoring=config.EVALUATION['scoring'], cv=skf, n_jobs=3)

    grid.fit(x_train, t_train)                  # fit model

    # clean strings no objects
    best_params = {k: str(v) for k, v in grid.best_params_.items()}
    rest = {
        'best_score' : grid.best_score_,
        'sample_technique' : sample_technique
    }

    # dict of model params & score
    lr_meta = {
        **best_params,
        **rest
    }

    save_best_model(grid.best_estimator_, lr_meta,
                    model_path=config.MODELS['logistic_regression']['sample'][sample_technique]['model'],
                    metadata_path=config.MODELS['logistic_regression']['sample'][sample_technique]['metadata'])

def random_forest_model(x_train:pd.DataFrame, t_train:pd.Series, *,
                        sample_technique:str='none', config=None):
    '''
    Train random forest model (tree-based algorithm)
    :param x_train: input features
    :param t_train: target feature
    :param sample_technique: which data processed with
    :param config: config loader
    :return: None
    '''


    # stratified is a good way to keep class distribution as it is in each fold.
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)

    # get scalers
    scalers = get_preprocessing_methods()

    # pipelining feature engineering, sampling and model training
    pipeline = Pipeline([
        ('scaler' , scalers['standard']),
        ('model', RandomForestClassifier(random_state=config.RANDOM_STATE ,n_jobs=3))
    ])

    params = {
        'scaler':                   list(scalers.values()),
        'model__max_depth':         config.MODELS['random_forest']['params']['max_depth'],
        'model__n_estimators':      config.MODELS['random_forest']['params']['n_estimators'],
        'model__min_samples_leaf':  config.MODELS['random_forest']['params']['min_samples_leaf'],
        'model__class_weight':      config.MODELS['random_forest']['params']['class_weight']
    }

    rand_grid = RandomizedSearchCV(pipeline, param_distributions=params,
                        n_iter=config.MODELS['random_forest']['n_iter'],scoring=config.EVALUATION['scoring'],
                              cv=skf, n_jobs=3)

    rand_grid.fit(x_train, t_train)           # fit model

    # clean strings no objects
    best_params = {k: str(v) for k, v in rand_grid.best_params_.items()}
    rest = {
        'best_score' : rand_grid.best_score_,
        'sample_technique' : sample_technique
    }

    # dict of model params & score
    rf_meta = {
        **best_params,
        **rest
    }

    # save best model with its parameters
    save_best_model(rand_grid.best_estimator_, rf_meta,
                    model_path=config.MODELS['random_forest']['sample'][sample_technique]['model']
                    , metadata_path=config.MODELS['random_forest']['sample'][sample_technique]['metadata'])


def neural_network_classifier(x_train:pd.DataFrame, t_train:pd.Series, *,
                              sample_technique:str='none', config=None):

    '''
    train neural network model (MLPClassifier)
    :param x_train: input features
    :param t_train: target feature
    :param sample_technique: which data processed with
    :param config: config loader
    :return: None
    '''

    # stratified is a good way to keep class distribution as it is in each fold.
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)

    # get scalers
    scalers = get_preprocessing_methods()

    # pipelining feature engineering, sampling and model training
    pipeline = Pipeline([
        ('scaler', scalers['standard']),
        ('model', MLPClassifier(random_state=config.RANDOM_STATE))
    ])

    params = {
        'scaler':                             list(scalers.values()),
        'model__hidden_layer_sizes':          [tuple(hid_sze) for hid_sze in config.MODELS['neural_network']['params']['hidden_layers']],
        'model__activation':                  config.MODELS['neural_network']['params']['activation'],
        'model__alpha':                       config.MODELS['neural_network']['params']['alpha'],
        'model__learning_rate_init':          config.MODELS['neural_network']['params']['learning_rate'],
        'model__batch_size':                  config.MODELS['neural_network']['params']['batch_size'],
        'model__max_iter':                    config.MODELS['neural_network']['params']['max_iter']
    }

    rand_grid = RandomizedSearchCV(pipeline, param_distributions=params,
                        n_iter=config.MODELS['neural_network']['n_iter'],scoring=config.EVALUATION['scoring'],
                              cv=skf, n_jobs=3)

    rand_grid.fit(x_train, t_train)           # fit model


    # clean strings no objects
    best_params = {k: str(v) for k, v in rand_grid.best_params_.items()}
    rest = {
        'best_score' : rand_grid.best_score_,
        'sample_technique' : sample_technique
    }

    # dict of model params & score
    nn_meta = {
        **best_params,
        **rest
    }

    # save best model with its parameters
    save_best_model(rand_grid.best_estimator_, nn_meta,
                    model_path=config.MODELS['neural_network']['sample'][sample_technique]['model']
                    , metadata_path=config.MODELS['neural_network']['sample'][sample_technique]['metadata'])


def voting_classifier(x_train:pd.DataFrame, t_train:pd.Series, *,
                      sample_technique:str='none', config=None):
    '''
    Train voting classifier model
    :param x_train: input features
    :param t_train: target feature
    :param sample_technique: which data processed with
    :param config: config loader
    :return: None
    '''

    # load trained models
    lr_model = joblib.load(config.MODELS['logistic_regression']['model'])
    rf_model = joblib.load(config.MODELS['random_forest']['model'])
    nn_model = joblib.load(config.MODELS['neural_network']['model'])

    # prepare estimators
    names = config.MODELS['voting_classifier']['params']['estimators']
    models = [lr_model, rf_model, nn_model]
    estimators = list(zip(names, models))

    # stratified is a good way to keep class distribution as it is in each fold.
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)

    # initial values
    best_score = -1
    best_params = {}

    # loop manually over different parameters to ensure freezing best trained models (no refit)
    for weights in config.MODELS['voting_classifier']['params']['weights']:
        voting = VotingClassifier(estimators=estimators, voting=config.MODELS['voting_classifier']['params']['voting']
                                  , weights=weights)

        # freeze models (no refit)
        voting.estimators_ = [lr_model, rf_model, nn_model]
        voting.le_ = LabelEncoder().fit(t_train)
        voting.classes_ = voting.le_.classes_

        # get scores
        scores = cross_val_score(voting, x_train, t_train, cv=skf,
                                 scoring=config.EVALUATION['scoring'], n_jobs=3)
        # mean score
        mean_score = scores.mean()

        # get the best one
        if mean_score > best_score:
            best_score = mean_score
            best_params = {'weights': weights, 'voting': config.MODELS['voting_classifier']['params']['voting']}

    # set voting model with best pretrained models
    voting = VotingClassifier(estimators=estimators, voting=best_params['voting'], weights=best_params['weights'])
    voting.estimators_ = [lr_model, rf_model, nn_model]
    voting.le_ = LabelEncoder().fit(t_train)
    voting.classes_ = voting.le_.classes_

    # get models metadata
    meta_paths = [
        config.MODELS['logistic_regression']['sample'][sample_technique]['metadata'],
        config.MODELS['random_forest']['sample'][sample_technique]['metadata'],
        config.MODELS['neural_network']['sample'][sample_technique]['metadata']
    ]
    # add best parameters
    voting_meta = [best_params]
    for path in meta_paths:
        with open(path,'r') as f:
            voting_meta.append(json.load(f))

    # save best model
    save_best_model(voting, voting_meta,
                    model_path=config.MODELS['voting_classifier']['sample'][sample_technique]['model'],
                    metadata_path=config.MODELS['voting_classifier']['sample'][sample_technique]['metadata'])

