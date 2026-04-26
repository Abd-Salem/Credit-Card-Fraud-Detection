from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import  Pipeline
from sklearn.preprocessing import FunctionTransformer
from collections import Counter
import joblib, json
from credit_fraud_utils_data import feature_construction, feature_transformation, load_data
from credit_fraud_utils_helper import save_best_model, get_scaling_method, get_processed_data
from mlp_focal_loss import MLP_FL, MLP_Pipeline


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
    lamda = config.MODELS['logistic_regression']['params']['lamda']
    scoring = config.EVALUATION['scoring']
    model_save_path = config.MODELS['logistic_regression']['sample'][sample_technique]['model']
    meta_save_path = config.MODELS['logistic_regression']['sample'][sample_technique]['metadata']

    # get train data according to chosen sample technique
    if sample_technique == 'none':
        x_train, t_train = load_data(path=config.DATASET['unprocessed']['train'])
    else:
        x_train, t_train = get_processed_data(data_path=config.DATASET['sampled'][sample_technique]['train']['data'],
                                              dtype='df', meta_path=config.DATASET['sampled'][sample_technique]['train']['metadata'])


    # stratified is a good way to keep class distribution as it is in each fold.
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # get scalers
    scalers = get_scaling_method(scalers_names=preprocessing_types)


    # pipeline
    pipeline = Pipeline([
        ('feature_construction', FunctionTransformer(feature_construction)),
        ('feature_transformation', FunctionTransformer(feature_transformation)),
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
        'model__C':                 lamda
    }

    # apply grid search with different parameters and get the best estimator
    grid = GridSearchCV(pipeline,param_grid=param_grid,
                        scoring=scoring, cv=skf, n_jobs=-1)

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
    n_iter = config.MODELS['random_forest']['n_iter']
    min_samples_leaf = config.MODELS['random_forest']['params']['min_samples_leaf']
    class_weight = config.MODELS['random_forest']['params']['class_weight']
    scoring = config.EVALUATION['scoring']
    model_save_path = config.MODELS['random_forest']['sample'][sample_technique]['model']
    model_meta_save_path = config.MODELS['random_forest']['sample'][sample_technique]['metadata']


    # get train data according to chosen sample technique
    if sample_technique == 'none':
        x_train, t_train = load_data(path=config.DATASET['unprocessed']['train'])
    else:
        x_train, t_train = get_processed_data(data_path=config.DATASET['sampled'][sample_technique]['train']['data'],
                                              dtype='df', meta_path=config.DATASET['sampled'][sample_technique]['train']['metadata'])


    # stratified is a good way to keep class distribution as it is in each fold.
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    # get scalers
    scalers = get_scaling_method(scalers_names=preprocessing_methods)

    # pipelining feature engineering, sampling and model training
    pipeline = Pipeline([
        ('feature_construction', FunctionTransformer(feature_construction)),
        ('feature_transformation', FunctionTransformer(feature_transformation)),
        ('scaler', scalers[0]),
        ('model', RandomForestClassifier(random_state=random_state ,n_jobs=-1))
    ])

    params = {
        'scaler':                   scalers,
        'model__max_depth':         max_depth,
        'model__n_estimators':      n_estimators,
        'model__min_samples_leaf':  min_samples_leaf,
        'model__class_weight':      class_weight
    }

    rand_grid = RandomizedSearchCV(pipeline, param_distributions=params,
                        n_iter=n_iter,scoring=scoring, refit=True,
                              cv=skf, n_jobs=4)

    rand_grid.fit(x_train, t_train)           # fit model

    # clean strings no objects
    best_params = {k: str(v) for k, v in rand_grid.best_params_.items()}

    # dict of model params & score
    rf_meta = {
        'sample_technique': sample_technique,
        **best_params,
        'best_score': rand_grid.best_score_
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
    batch_size = config.MODELS['neural_network']['params']['batch_size']
    max_iter = config.MODELS['neural_network']['params']['max_iter']
    scoring = config.EVALUATION['scoring']
    cv_folds = config.EVALUATION['cv_folds']
    n_iter = config.MODELS['neural_network']['n_iter']
    model_save_path = config.MODELS['neural_network']['sample'][sample_technique]['model']
    model_meta_save_path = config.MODELS['neural_network']['sample'][sample_technique]['metadata']

    # get train data according to chosen sample technique
    if sample_technique == 'none':
        x_train, t_train = load_data(path=config.DATASET['unprocessed']['train'])
    else:
        x_train, t_train = get_processed_data(data_path=config.DATASET['sampled'][sample_technique]['train']['data'],
                                              dtype='df', meta_path=config.DATASET['sampled'][sample_technique]['train']['metadata'])


    # stratified is a good way to keep class distribution as it is in each fold.
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    # get scaling methods
    scalers = get_scaling_method(scalers_names=preprocessing_methods)

    # pipelining feature engineering, sampling and model training
    pipeline = Pipeline([
        ('feature_construction', FunctionTransformer(feature_construction)),
        ('feature_transformation', FunctionTransformer(feature_transformation)),
        ('scaler', scalers[0]),
        ('model', MLPClassifier(random_state=random_state, solver='adam'))
    ])

    params = {
        'scaler':                             scalers,
        'model__hidden_layer_sizes':          hidden_layer_sizes,
        'model__activation':                  activation,
        'model__alpha':                       alpha,
        'model__batch_size':                  batch_size,
        'model__max_iter':                    max_iter
    }

    rand_grid = RandomizedSearchCV(pipeline, param_distributions=params,
                        n_iter=n_iter,scoring=scoring,
                              cv=skf, n_jobs=-1)

    rand_grid.fit(x_train, t_train)           # fit model


    # clean strings no objects
    best_params = {k: str(v) for k, v in rand_grid.best_params_.items()}

    # metadata
    nn_meta = {
        'sample_technique': sample_technique,
        **best_params,
        'best_score': rand_grid.best_score_
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
        x_train, t_train = load_data(path=config.DATASET['unprocessed']['train'])
    else:
        x_train, t_train = get_processed_data(data_path=config.DATASET['sampled'][sample_technique]['train']['data'],
                                              dtype='df', meta_path=config.DATASET['sampled'][sample_technique]['train']['metadata'])


    preprocessing_methods = config.PREPROCESSING['scaler']
    random_state = config.RANDOM_STATE
    hidden_layers =  [tuple(hid_sze) for hid_sze in config.MODELS['neural_network_fl']['params']['hidden_layers']]
    fl_alphas = config.MODELS['neural_network_fl']['params']['alpha']       # alpha
    fl_gammas = config.MODELS['neural_network_fl']['params']['gamma']       # gamma
    epochs = config.MODELS['neural_network_fl']['params']['epochs']         # epochs num
    patience = config.MODELS['neural_network_fl']['params']['patience']     # patience num
    opt_lr = config.MODELS['neural_network_fl']['params']['optimizer']['lr']    # learning rate
    weight_decays = config.MODELS['neural_network_fl']['params']['optimizer']['weight_decay']
    model_save_path = config.MODELS['neural_network_fl']['sample'][sample_technique]['model']
    model_meta_save_path = config.MODELS['neural_network_fl']['sample'][sample_technique]['metadata']



    x_tr, x_val, t_tr, t_val = train_test_split(x_train, t_train, test_size=0.1,
                                                random_state=random_state, shuffle=True, stratify=t_train)
    # scale data using standard scaling
    scalers = get_scaling_method(preprocessing_methods)

    best_score = -1
    best_model = None
    best_scaler = None
    for scaler in scalers:
        for hl in hidden_layers:
            for lr in opt_lr:
                for wd in weight_decays:
                    for fl_alpha in fl_alphas:
                        for fl_gamma in fl_gammas:
                            for epoch in epochs:
                                preprocessing = Pipeline([
                                    ('feature_construction', FunctionTransformer(feature_construction)),
                                    ('feature_transformation', FunctionTransformer(feature_transformation)),
                                    ('scaler', scaler)
                                ])
                                model = model=MLP_FL(random_state=random_state,
                                                focal_loss_alpha=fl_alpha,
                                                focal_loss_gamma=fl_gamma,
                                                patience=patience,
                                                hidden_layers=hl,
                                                optimizer_lr=lr,
                                                optimizer_weight_decay=wd,
                                                batch_size=64,
                                                epochs=epoch)
                                pipeline = MLP_Pipeline(steps=[
                                    ('preprocessing', preprocessing),
                                    ('model', model)
                                ])
                                pipeline.fit(x_tr, t_tr)
                                score = pipeline.score(x_val, t_val)
                                if score > best_score:
                                    best_score = score
                                    best_scaler = scaler
                                    best_model = pipeline


    # get parameters and prepare metadata
    best_params = {k: str(v) for k, v in best_model.get_params().items()}

    nn_fl_meta = {
        'sample_technique' : sample_technique,
        'scaler': str(best_scaler),
        **best_params,
        'best_score': float(best_score)
    }


    # save best model with its parameters
    save_best_model(best_model, nn_fl_meta,
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
        x_train, t_train = load_data(path=config.DATASET['unprocessed']['train'])
    else:
        x_train, t_train = get_processed_data(data_path=config.DATASET['sampled'][sample_technique]['train']['data'],
                                              dtype='df', meta_path=config.DATASET['sampled'][sample_technique]['train']['metadata'])

    # get scaler methods
    scalers = get_scaling_method(preprocessing_types)

    # stratified k-fold to preserve dist of data
    skf = StratifiedKFold(n_splits=cv_folds,random_state=random_state, shuffle=True)

    # create pipeline
    pipeline = Pipeline([
        ('feature_construction', FunctionTransformer(feature_construction)),
        ('feature_transformation', FunctionTransformer(feature_transformation)),
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
    grid = GridSearchCV(pipeline, param_grid=params, cv=skf, scoring=scoring, n_jobs=-1)

    # try all fits and tune over parameters
    grid.fit(x_train, t_train)

    # clean strings no objects
    best_params = {k: str(v) for k, v in grid.best_params_.items()}

    # metadata
    knn_meta = {
        'sample_technique': sample_technique,
        **best_params,
        'best_score': grid.best_score_
    }

    # save best model with it's parameters
    save_best_model(grid.best_estimator_ , knn_meta,
                    model_path=model_save_path,
                    metadata_path=meta_save_path)


def voting_classifier_1(sample_technique:str='none', config=None):
    '''
    Train voting classifier model
    :param sample_technique: which data processed with
    :param config: config loader
    :return: None
    '''

    # save paths
    model_save_path = config.MODELS['voting_classifier_1']['sample'][sample_technique]['model']
    model_meta_save_path = config.MODELS['voting_classifier_1']['sample'][sample_technique]['metadata']

    # get configs
    n_splits = config.EVALUATION['cv_folds']
    random_state = config.RANDOM_STATE
    weights = config.MODELS['voting_classifier_1']['params']['weights']



    # get train data according to chosen sample technique
    if sample_technique == 'none':
        x_train, t_train = load_data(path=config.DATASET['unprocessed']['train'])
    else:
        x_train, t_train = get_processed_data(data_path=config.DATASET['sampled'][sample_technique]['train']['data'],
                                              dtype='df', meta_path=config.DATASET['sampled'][sample_technique]['train']['metadata'])



    # best pretrained models on the same sample technique
    rf_model = joblib.load(config.MODELS['random_forest']['sample'][sample_technique]['model'])
    nn_model = joblib.load(config.MODELS['neural_network']['sample'][sample_technique]['model'])
    lr_model = joblib.load(config.MODELS['logistic_regression']['sample'][sample_technique]['model'])


    # preserve the distribution
    skf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)

    # prepare estimators
    models = [
        ('rf' , rf_model),
        ('nn' , nn_model),
        ('lr', lr_model)
    ]

    pipeline = Pipeline([('model', VotingClassifier(estimators=models, voting='soft'))])
    params = {
        'model__weights' : weights
    }

    # create voting model
    grid = GridSearchCV(pipeline, param_grid=params,cv=skf, scoring='average_precision', n_jobs=-1)

    # fit data
    grid.fit(x_train, t_train)

    # models metadata
    with open(config.MODELS['random_forest']['sample'][sample_technique]['metadata'], 'r') as f:
        rf_meta = json.load(f)

    with open(config.MODELS['neural_network']['sample'][sample_technique]['metadata'], 'r') as f:
        nn_meta = json.load(f)

    with open(config.MODELS['logistic_regression']['sample'][sample_technique]['metadata'], 'r') as f:
        lr_meta = json.load(f)

    # collect all models metadata
    voting_meta = {
        'vc' : grid.best_params_,
        'rf' : rf_meta,
        'nn' : nn_meta,
        'lr' : lr_meta
    }

    # save best model
    save_best_model(grid.best_estimator_ , voting_meta,
                    model_path=model_save_path,
                    metadata_path=model_meta_save_path)


def voting_classifier_2(sample_technique:str='none', config=None):
    '''
    Train voting classifier model
    :param sample_technique: which data processed with
    :param config: config loader
    :return: None
    '''

    # save paths
    model_save_path = config.MODELS['voting_classifier_2']['sample'][sample_technique]['model']
    model_meta_save_path = config.MODELS['voting_classifier_2']['sample'][sample_technique]['metadata']

    # get configs
    n_splits = config.EVALUATION['cv_folds']
    random_state = config.RANDOM_STATE
    weights = config.MODELS['voting_classifier_2']['params']['weights']



    # get train data according to chosen sample technique
    if sample_technique == 'none':
        x_train, t_train = load_data(path=config.DATASET['unprocessed']['train'])
    else:
        x_train, t_train = get_processed_data(data_path=config.DATASET['sampled'][sample_technique]['train']['data'],
                                              dtype='df', meta_path=config.DATASET['sampled'][sample_technique]['train']['metadata'])



    # best pretrained models on the same sample technique
    rf_model = joblib.load(config.MODELS['random_forest']['sample'][sample_technique]['model'])
    nn_model = joblib.load(config.MODELS['neural_network']['sample'][sample_technique]['model'])
    knn_model = joblib.load(config.MODELS['knn_classifier']['sample'][sample_technique]['model'])


    # preserve the distribution
    skf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)

    # prepare estimators
    models = [
        ('rf' , rf_model),
        ('nn' , nn_model),
        ('knn', knn_model)
    ]

    pipeline = Pipeline([('model', VotingClassifier(estimators=models, voting='soft'))])
    params = {
        'model__weights' : weights
    }

    # create voting model
    grid = GridSearchCV(pipeline, param_grid=params,cv=skf, scoring='average_precision', n_jobs=-1)

    # fit data
    grid.fit(x_train, t_train)

    # models metadata
    with open(config.MODELS['random_forest']['sample'][sample_technique]['metadata'], 'r') as f:
        rf_meta = json.load(f)

    with open(config.MODELS['neural_network']['sample'][sample_technique]['metadata'], 'r') as f:
        nn_meta = json.load(f)

    with open(config.MODELS['knn_classifier']['sample'][sample_technique]['metadata'], 'r') as f:
        knn_meta = json.load(f)

    # collect all models metadata
    voting_meta = {
        'vc' : grid.best_params_,
        'rf' : rf_meta,
        'nn' : nn_meta,
        'knn': knn_meta
    }

    # save best model
    save_best_model(grid.best_estimator_ , voting_meta,
                    model_path=model_save_path,
                    metadata_path=model_meta_save_path)


def voting_classifier_3(sample_technique:str='none', config=None):
    '''
    Train voting classifier model
    :param sample_technique: which data processed with
    :param config: config loader
    :return: None
    '''

    # save paths
    model_save_path = config.MODELS['voting_classifier_3']['sample'][sample_technique]['model']
    model_meta_save_path = config.MODELS['voting_classifier_3']['sample'][sample_technique]['metadata']

    # get configs
    n_splits = config.EVALUATION['cv_folds']
    random_state = config.RANDOM_STATE
    weights = config.MODELS['voting_classifier_3']['params']['weights']



    # get train data according to chosen sample technique
    if sample_technique == 'none':
        x_train, t_train = load_data(path=config.DATASET['unprocessed']['train'])
    else:
        x_train, t_train = get_processed_data(data_path=config.DATASET['sampled'][sample_technique]['train']['data'],
                                              dtype='df', meta_path=config.DATASET['sampled'][sample_technique]['train']['metadata'])



    # best pretrained models on the same sample technique
    rf_model = joblib.load(config.MODELS['random_forest']['sample'][sample_technique]['model'])
    nn_model = joblib.load(config.MODELS['neural_network']['sample'][sample_technique]['model'])
    lr_model = joblib.load(config.MODELS['logistic_regression']['sample'][sample_technique]['model'])
    knn_model = joblib.load(config.MODELS['knn_classifier']['sample'][sample_technique]['model'])


    # preserve the distribution
    skf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)

    # prepare estimators
    models = [
        ('rf' , rf_model),
        ('nn' , nn_model),
        ('lr', lr_model),
        ('knn', knn_model)
    ]

    pipeline = Pipeline([('model', VotingClassifier(estimators=models, voting='soft'))])
    params = {
        'model__weights' : weights
    }

    # create voting model
    grid = GridSearchCV(pipeline, param_grid=params,cv=skf, scoring='average_precision', n_jobs=-1)

    # fit data
    grid.fit(x_train, t_train)

    # models metadata
    with open(config.MODELS['random_forest']['sample'][sample_technique]['metadata'], 'r') as f:
        rf_meta = json.load(f)

    with open(config.MODELS['neural_network']['sample'][sample_technique]['metadata'], 'r') as f:
        nn_meta = json.load(f)

    with open(config.MODELS['logistic_regression']['sample'][sample_technique]['metadata'], 'r') as f:
        lr_meta = json.load(f)

    with open(config.MODELS['knn_classifier']['sample'][sample_technique]['metadata'], 'r') as f:
        knn_meta = json.load(f)

    # collect all models metadata
    voting_meta = {
        'vc' : grid.best_params_,
        'rf' : rf_meta,
        'nn' : nn_meta,
        'lr' : lr_meta,
        'knn': knn_meta
    }

    # save best model
    save_best_model(grid.best_estimator_ , voting_meta,
                    model_path=model_save_path,
                    metadata_path=model_meta_save_path)