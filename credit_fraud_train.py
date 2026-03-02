from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from imblearn.pipeline import  Pipeline
import config
from credit_fraud_utils_data import prepare_data, load_data, sample_data
from credit_fraud_utils_eval import model_eval_report, pr_curve_fbeta_score
from collections import Counter

def logistic_regression_model(sample_technique:str ='oversampling'):
    '''
    Train Logistic Regression Model
    Parameter:
        sample_technique (str): oversampling - undersampling - both
    '''

    # load train-val dataset
    x_train, t_train = load_data(config.DATASET['train_path'])
    x_val, t_val = load_data(config.DATASET['val_path'])

    # Applying preparing steps(feature extraction, feature transformation) on train & val dataset
    col_trans = prepare_data(x_train, inplace=True)
    prepare_data(x_val, inplace=True)

    # sampling data
    x_train_sampled, t_train_sampled = sample_data(X=x_train, y=t_train,
                                                   technique=sample_technique,
                                                   sample_strategy='auto')

    # stratified is a good way to keep distribution as it is in each fold.
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)

    # pipelining feature engineering, sampling and model training
    pipeline = Pipeline([
        ('data_preprocessing', col_trans),
        ('model', LogisticRegression(random_state=config.RANDOM_STATE))
    ])

    # Classes counts
    count = Counter(t_train)
    class_ratio = count[1] / count[0]

    # grid parameters
    param_grid = {
        'model__solver' : ['lbfgs', 'newton-cg'],
        'model__class_weight' : [{1 : wt} for wt in [class_ratio, 0.05, 0.1, 0.5, 1]],
        'model__max_iter': [100, 500, 1000]
    }

    # apply grid search with different parameters and get the best estimator
    grid = GridSearchCV(pipeline,param_grid=param_grid,
                        scoring='average_precision', cv=skf)
    grid.fit(x_train_sampled, t_train_sampled)
    best_params = grid.best_params_             # best parameters
    print("Best Hyperparameters:", best_params)

    model = grid.best_estimator_                # best model

    # avg precision score, best threshold and f-beta scores
    # get best threshold using f2-score as recall is important in our problem
    result = pr_curve_fbeta_score(model, x_val, t_val, beta_score=2)

    # classification report with specific threshold
    report, harmonic_mean = model_eval_report(model, x_val, t_val, result['best_threshold'])

    # Showing results
    print(f'AUPRC: {result['auprc']}')
    print(f'Best Threshold: {result['best_threshold']}')
    print(f'f2-score for class-1: {result['f-score_1']}')
    print(f'f2-score for class-0: {result['f-score_0']}')
    print('==============================================')
    print('Classification report using best threshold:')
    print(report)
    print(f'Harmonic mean of f1-scores: {harmonic_mean}')

    # Best Hyperparameters: {'model__class_weight': {1: 1}, 'model__max_iter': 100, 'model__solver': 'lbfgs'}
    # AUPRC: 0.74
    # Best Threshold: 0.999997
    # f2-score for class-1: 0.79
    # f2-score for class-0: 0.0
    # ==============================================
    # Classification report using best threshold:
    #               precision    recall  f1-score   support
    #
    #            0     0.9996    0.9998    0.9997     56870
    #            1     0.8537    0.7778    0.8140        90
    #
    #     accuracy                         0.9994     56960
    #    macro avg     0.9267    0.8888    0.9068     56960
    # weighted avg     0.9994    0.9994    0.9994     56960
    #
    # Harmonic mean of f1-scores: 0.897