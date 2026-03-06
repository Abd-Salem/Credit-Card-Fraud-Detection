from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from imblearn.pipeline import  Pipeline
from config import config
from credit_fraud_utils_data import feature_transformation, get_processed_data
from credit_fraud_utils_eval import model_eval_report, avg_pr_fb_score
from collections import Counter


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
        'model__solver':            config.MODELS['logistic_regression']['solver'],
        'model__class_weight':      [{1:wt} for wt in config.MODELS['logistic_regression']['class_weight']+[ratio]],
        'model__max_iter':          config.MODELS['logistic_regression']['max_iter']
    }

    # apply grid search with different parameters and get the best estimator
    grid = GridSearchCV(pipeline,param_grid=param_grid,
                        scoring=config.EVALUATION['scoring'], cv=skf, n_jobs=-1)

    grid.fit(x_train, t_train)                  # fit model
    print(f'Best Hyperparameters: {grid.best_params_}')
    print(f'Best Score: {grid.best_score_}')
    model = grid.best_estimator_                # best model

    # evaluate using avg precision score and f-beta metrics and show plot with best threshold
    result = avg_pr_fb_score(model, x_val, t_val, beta_score=config.EVALUATION['beta'], show_plot=True)

    # classification report using best threshold given from evaluation
    report, harmonic_mean = model_eval_report(model, x_val, t_val, result['best_threshold'])

    # Showing results
    print(f'AUPRC: {result['auprc']:.3f}')
    print(f'Best Threshold: {result['best_threshold']:.3f}')
    print(f'f2-score(1): {result['fscore_1']:.3f}')
    print(f'f2-score(0): {result['fscore_0']:.3f}')
    print('==============================================')
    print('Classification report using best threshold:')
    print(report)
    print(f'Harmonic mean(f1-scores): {harmonic_mean:.3f}')


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
        ('model', RandomForestClassifier(random_state=config.RANDOM_STATE ,n_jobs=-1))
    ])

    params = {
        'model__max_depth':         config.MODELS['random_forest']['max_depth'],
        'model__n_estimators':      config.MODELS['random_forest']['n_estimators'],
        'model__min_samples_leaf':  config.MODELS['random_forest']['min_samples_leaf'],
        'model__class_weight':      [{1 : wt} for wt in config.MODELS['random_forest']['class_weight']]
    }

    grid = GridSearchCV(pipeline, param_grid=params,
                        scoring='average_precision', n_jobs=-1, cv=skf)

    grid.fit(x_train, t_train)           # best parameters
    print(f'Best Hyperparameters: {grid.best_params_}')
    print(f'Best Score: {grid.best_score_}')
    model = grid.best_estimator_                # best model

    # evaluate using avg precision score and f-beta metrics and show plot with best threshold
    result = avg_pr_fb_score(model, x_val, t_val, beta_score=config.EVALUATION['beta'], show_plot=True)

    # classification report using best threshold given from evaluation
    report, harmonic_mean = model_eval_report(model, x_val, t_val, result['best_threshold'])

    # Showing results
    print(f'AUPRC: {result['auprc']:.3f}')
    print(f'Best Threshold: {result['best_threshold']:.3f}')
    print(f'f2-score(1): {result['fscore_1']:.3f}')
    print(f'f2-score(0): {result['fscore_0']:.3f}')
    print('==============================================')
    print('Classification report using best threshold:')
    print(report)
    print(f'Harmonic mean(f1-scores): {harmonic_mean:.3f}')