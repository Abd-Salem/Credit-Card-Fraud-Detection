from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import  Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from collections import Counter
from credit_fraud_utils_helper import save_best_model, get_scaling_method, get_processed_data
from focal_loss_mlp import MLP_FL
import json, joblib


def logistic_regression_model(sample_technique:str='none', config=None):
    '''
    Train logistic regression model
    :param sample_technique: which data processed with
    :param config: config loader
    :return: None
    '''

    # get configs
    n_splits = config.EVALUATION['cv_folds']
    random_state = config.RANDOM_STATE
    preprocessing_types = config.PREPROCESSING['scaler']
    c_weights = config.MODELS['logistic_regression']['params']['class_weight']
    solvers = config.MODELS['logistic_regression']['params']['solver']
    iters = config.MODELS['logistic_regression']['params']['max_iter']
    penalty = config.MODELS['logistic_regression']['params']['penalty']
    lamda = config.MODELS['logistic_regression']['params']['lamda']
    scoring = config.EVALUATION['scoring']
    model_save_path = config.MODELS['logistic_regression']['sample'][sample_technique]['model']
    meta_save_path = config.MODELS['logistic_regression']['sample'][sample_technique]['metadata']

    # get train data according to chosen sample technique
    if sample_technique == 'none':
        x_train, t_train = get_processed_data(data_path=config.DATASET['prepared']['train']['data'],
                                              dtype='df', meta_path=config.DATASET['prepared']['train']['metadata'])
    else:
        x_train, t_train = get_processed_data(data_path=config.DATASET['sampled'][sample_technique]['train']['data'],
                                              dtype='df', meta_path=config.DATASET['sampled'][sample_technique]['train']['metadata'])


    # stratified is a good way to keep class distribution as it is in each fold.
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # get scalers
    scalers = get_scaling_method(scalers_names=preprocessing_types)

    # pipeline
    pipeline = Pipeline([
        ('scaler', scalers[0]),
        ('model', LogisticRegression(random_state=random_state))
    ])

    # get classes ratio and add it to class weights
    count = Counter(t_train)
    ratio = count[1]/count[0]
    class_weight = c_weights
    class_weight.append({1:1, 0:ratio})

    # grid parameters
    param_grid = {
        'scaler' :                  scalers,
        'model__solver':            solvers,
        'model__class_weight':      class_weight,
        'model__max_iter':          iters,
        'model__penalty':           penalty,
        'model__C':                 lamda
    }

    # apply grid search with different parameters and get the best estimator
    grid = GridSearchCV(pipeline,param_grid=param_grid,
                        scoring=scoring, cv=skf, n_jobs=3)

    grid.fit(x_train, t_train)                  # fit model

    # clean strings no objects
    best_params = {k: str(v) for k, v in grid.best_params_.items()}
    others = {
        'sample_technique': sample_technique,
        'best_score' : grid.best_score_
    }

    # dict of model params & score
    lr_meta = {
        **best_params,
        **others
    }

    save_best_model(grid.best_estimator_ , lr_meta,
                    model_path=model_save_path,
                    metadata_path=meta_save_path)


def random_forest_model(sample_technique:str='none', config=None):
    '''
    Train random forest model (tree-based algorithm)
    :param sample_technique: which data processed with
    :param config: config loader
    :return: None
    '''

    #get configs
    random_state = config.RANDOM_STATE
    preprocessing_methods = config.PREPROCESSING['scaler']
    max_depth =  config.MODELS['random_forest']['params']['max_depth']
    n_estimators = config.MODELS['random_forest']['params']['n_estimators']
    min_samples_leaf = config.MODELS['random_forest']['params']['min_samples_leaf']
    class_weight = config.MODELS['random_forest']['params']['class_weight']
    scoring = config.EVALUATION['scoring']
    model_save_path = config.MODELS['random_forest']['sample'][sample_technique]['model']
    model_meta_save_path = config.MODELS['random_forest']['sample'][sample_technique]['metadata']


    # get train data according to chosen sample technique
    if sample_technique == 'none':
        x_train, t_train = get_processed_data(data_path=config.DATASET['prepared']['train']['data'],
                                              dtype='df', meta_path=config.DATASET['prepared']['train']['metadata'])
    else:
        x_train, t_train = get_processed_data(data_path=config.DATASET['sampled'][sample_technique]['train']['data'],
                                              dtype='df', meta_path=config.DATASET['sampled'][sample_technique]['train']['metadata'])


    # stratified is a good way to keep class distribution as it is in each fold.
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    # get scalers
    scalers = get_scaling_method(scalers_names=preprocessing_methods)

    # pipelining feature engineering, sampling and model training
    pipeline = Pipeline([
        ('scaler', scalers[0]),
        ('model', RandomForestClassifier(random_state=random_state ,n_jobs=3))
    ])

    params = {
        'scaler':                   scalers,
        'model__max_depth':         max_depth,
        'model__n_estimators':      n_estimators,
        'model__min_samples_leaf':  min_samples_leaf,
        'model__class_weight':      class_weight
    }

    rand_grid = RandomizedSearchCV(pipeline, param_distributions=params,
                        n_iter=config.MODELS['random_forest']['n_iter'],scoring=scoring,
                              cv=skf, n_jobs=3)

    rand_grid.fit(x_train, t_train)           # fit model

    # clean strings no objects
    best_params = {k: str(v) for k, v in rand_grid.best_params_.items()}
    others = {
        'sample_technique': sample_technique,
        'best_score' : rand_grid.best_score_
    }

    # dict of model params & score
    rf_meta = {
        **best_params,
        **others
    }

    # save model, scaler and metadata
    save_best_model(rand_grid.best_estimator_, rf_meta,
                    model_path=model_save_path,
                    metadata_path=model_meta_save_path)


def neural_network_classifier(sample_technique:str='none', config=None):
    '''
    train neural network model (MLPClassifier)
    :param sample_technique: which data processed with
    :param config: config loader
    :return: None
    '''

    # get configs
    random_state = config.RANDOM_STATE
    preprocessing_methods = config.PREPROCESSING['scaler']
    hidden_layer_sizes =  [tuple(hid_sze) for hid_sze in config.MODELS['neural_network']['params']['hidden_layers']]
    activation = config.MODELS['neural_network']['params']['activation']
    alpha = config.MODELS['neural_network']['params']['alpha']
    learning_rate_init = config.MODELS['neural_network']['params']['learning_rate']
    batch_size = config.MODELS['neural_network']['params']['batch_size']
    max_iter = config.MODELS['neural_network']['params']['max_iter']
    scoring = config.EVALUATION['scoring']
    n_iter = config.MODELS['neural_network']['n_iter']
    model_save_path = config.MODELS['neural_network']['sample'][sample_technique]['model']
    model_meta_save_path = config.MODELS['neural_network']['sample'][sample_technique]['metadata']

    # get train data according to chosen sample technique
    if sample_technique == 'none':
        x_train, t_train = get_processed_data(data_path=config.DATASET['prepared']['train']['data'],
                                              dtype='df', meta_path=config.DATASET['prepared']['train']['metadata'])
    else:
        x_train, t_train = get_processed_data(data_path=config.DATASET['sampled'][sample_technique]['train']['data'],
                                              dtype='df', meta_path=config.DATASET['sampled'][sample_technique]['train']['metadata'])


    # stratified is a good way to keep class distribution as it is in each fold.
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    # get scaling methods
    scalers = get_scaling_method(scalers_names=preprocessing_methods)

    # pipelining feature engineering, sampling and model training
    pipeline = Pipeline([
        ('scaler', scalers[0]),
        ('model', MLPClassifier(random_state=random_state))
    ])

    params = {
        'scaler':                             scalers,
        'model__hidden_layer_sizes':          hidden_layer_sizes,
        'model__activation':                  activation,
        'model__alpha':                       alpha,
        'model__learning_rate_init':          learning_rate_init,
        'model__batch_size':                  batch_size,
        'model__max_iter':                    max_iter
    }

    rand_grid = RandomizedSearchCV(pipeline, param_distributions=params,
                        n_iter=n_iter,scoring=scoring,
                              cv=skf, n_jobs=3)

    rand_grid.fit(x_train, t_train)           # fit model


    # clean strings no objects
    best_params = {k: str(v) for k, v in rand_grid.best_params_.items()}
    others = {
        'sample_technique': sample_technique,
        'best_score' : rand_grid.best_score_
    }

    # dict of model params & score
    nn_meta = {
        **best_params,
        **others
    }

    # save model, scaler and metadata
    save_best_model(rand_grid.best_estimator_, nn_meta,
                    model_path=model_save_path,
                    metadata_path=model_meta_save_path)

def neural_network_fl(sample_technique:str='none', config=None):
    '''
    train neural network model with focal loss as cost function
    :param sample_technique: which data processed with
    :param config: config loader
    :return:
    '''

    # get train data according to chosen sample technique
    if sample_technique == 'none':
        x_train, t_train = get_processed_data(data_path=config.DATASET['prepared']['train']['data'],
                                              dtype='df', meta_path=config.DATASET['prepared']['train']['metadata'])
    else:
        x_train, t_train = get_processed_data(data_path=config.DATASET['sampled'][sample_technique]['train']['data'],
                                              dtype='df', meta_path=config.DATASET['sampled'][sample_technique]['train']['metadata'])


    preprocessing_methods = config.PREPROCESSING['scaler']
    random_state = config.RANDOM_STATE
    input_dim = x_train.shape[1]        # input nodes
    hidden_layers =  [tuple(hid_sze) for hid_sze in config.MODELS['neural_network_fl']['params']['hidden_layers']]
    activation = config.MODELS['neural_network_fl']['params']['activation']
    fl_alpha = config.MODELS['neural_network_fl']['params']['alpha']       # alpha
    fl_gamma = config.MODELS['neural_network_fl']['params']['gamma']       # gamma
    epochs = config.MODELS['neural_network_fl']['params']['epochs']         # epochs num
    patience = config.MODELS['neural_network_fl']['params']['patience']     # patience num
    opts = config.MODELS['neural_network_fl']['params']['optimizer']['type'] # optimizer type
    opt_lr = config.MODELS['neural_network_fl']['params']['optimizer']['lr']    # learning rate
    opt_weight_decay = config.MODELS['neural_network_fl']['params']['optimizer']['weight_decay']     # weight decay
    model_save_path = config.MODELS['neural_network']['sample'][sample_technique]['model']
    model_meta_save_path = config.MODELS['neural_network']['sample'][sample_technique]['metadata']
    cv_folds = config.EVALUATION['cv_folds']
    scoring = config.EVALUATION['scoring']


    # scale data using standard scaling
    scalers = get_scaling_method(preprocessing_methods)

    skf = StratifiedKFold(n_splits=cv_folds, random_state=random_state, shuffle=True)

    pipeline = Pipeline([
        ('scaler', scalers[0]),
        ('model', MLP_FL(input_dim=input_dim, random_state=random_state, stratify=t_train))
    ])

    params = {
        'scaler' :                              scalers,
        'model__hidden_layers':                 hidden_layers,
        'model__activation'   :                 activation,
        'model__epochs':                        epochs,
        'model__patience':                      patience,
        'model__optimizer':                     opts,
        'model__optimizer_lr':                  opt_lr,
        'model__optimizer_weight_decay':        opt_weight_decay,
        'model__focal_loss_alpha':              fl_alpha,
        'model__focal_loss_gamma':              fl_gamma
    }

    rand_grid = RandomizedSearchCV(pipeline,param_distributions=params, cv=skf ,scoring=scoring, n_jobs=1)

    from sklearn.utils.estimator_checks import check_is_fitted
    from sklearn.base import is_classifier

    print(is_classifier(rand_grid))  # should print True

    rand_grid.fit(x_train, t_train)

    best_params = {k: str(v) for k, v in rand_grid.best_params_.items()}

    others = {
        'sample_technieque' : sample_technique,
        'best_score'        : rand_grid.best_score_
    }
    nn_fl_meta = {
        **best_params,
        **others
    }


    # save best model with its parameters
    save_best_model(rand_grid.best_estimator_, nn_fl_meta,
                    model_path=model_save_path,
                    metadata_path=model_meta_save_path)


def knn_classifier(sample_technique:str='none', config=None):
    '''
    Train knn classifier model
    :param sample_technique: which data processed with
    :param config: config loader
    :return: None
    '''

    # get configs
    cv_folds = config.EVALUATION['cv_folds']
    random_state = config.RANDOM_STATE
    preprocessing_types = config.PREPROCESSING['scaler']
    scoring = config.EVALUATION['scoring']
    k = config.MODELS['knn_classifier']['params']['k']
    weights = config.MODELS['knn_classifier']['params']['weights']
    metric = config.MODELS['knn_classifier']['params']['metric']
    model_save_path = config.MODELS['knn_classifier']['sample'][sample_technique]['model']
    meta_save_path = config.MODELS['knn_classifier']['sample'][sample_technique]['metadata']

    # get train data according to chosen sample technique
    if sample_technique == 'none':
        x_train, t_train = get_processed_data(data_path=config.DATASET['prepared']['train']['data'],
                                              dtype='df', meta_path=config.DATASET['prepared']['train']['metadata'])
    else:
        x_train, t_train = get_processed_data(data_path=config.DATASET['sampled'][sample_technique]['train']['data'],
                                              dtype='df', meta_path=config.DATASET['sampled'][sample_technique]['train']['metadata'])

    # get scaler methods
    scalers = get_scaling_method(preprocessing_types)

    # stratified k-fold to preserve dist of data
    skf = StratifiedKFold(n_splits=cv_folds,random_state=random_state, shuffle=True)

    # create pipeline
    pipeline = Pipeline([
        ('scaler', scalers[0]),
        ('model', KNeighborsClassifier())
    ])

    # hyper parameters
    params = {
        'scaler' : scalers,
        'model__n_neighbors': k,
        'model__weights': weights,
        'model__metric': metric
    }


    # create grid
    grid = GridSearchCV(pipeline, param_grid=params, cv=skf, scoring=scoring, n_jobs=3)

    # try all fits and tune over parameters
    grid.fit(x_train, t_train)

    # clean strings no objects
    best_params = {k: str(v) for k, v in grid.best_params_.items()}
    others = {
        'sample_technique': sample_technique,
        'best_score' : grid.best_score_
    }

    # dict of model params & score
    knn_meta = {
        **best_params,
        **others
    }
    # save best model with it's parameters
    save_best_model(grid.best_estimator_ , knn_meta,
                    model_path=model_save_path,
                    metadata_path=meta_save_path)


def voting_classifier(sample_technique:str='none', config=None):
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
    save_best_model(voting,StandardScaler() , voting_meta,
                    model_path=config.MODELS['voting_classifier']['sample'][sample_technique]['model'],
                    scaler_path=config.MODELS['voting_classifier']['sample'][sample_technique]['scaler'],
                    metadata_path=config.MODELS['voting_classifier']['sample'][sample_technique]['metadata'])

