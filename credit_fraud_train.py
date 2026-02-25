from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import Pipeline as imb_pipline
import config
from credit_fraud_utils_data import prepare_data, load_data, sample_data
from credit_fraud_utils_eval import model_eval_report, pr_curve_avg_precision_score
from collections import Counter
from tabulate import tabulate

def logistic_regression_model(sample_technique:str ='', weighted:bool =False):
    '''
    Train Logistic Regression Model
    Parameter:
        sample_technique (str): oversampling - undersampling - both - none
        weighted (bool): cost sensitive training
    '''

    # raise error if isn't bool
    if not isinstance(weighted, bool):
        raise TypeError('Weighted isn\'t boolean')

    # load train-val dataset
    x_train, t_train = load_data(config.DATASET['train_path'])
    x_val, t_val = load_data(config.DATASET['val_path'])

    # Applying preparing steps(feature extraction, feature transformation) on train-val dataset
    col_trans = prepare_data(x_train, inplace=True)
    x_val, _ = prepare_data(x_val, inplace=False)

    # get classes count
    count = Counter(t_train)

    # check for weighted classes for training
    wt = 1
    if weighted:
        wt = count[1] / count[0]

    # sampling technique
    technique = sample_data(y=t_train, technique=sample_technique, sample_strategy='auto')

    # pipelining feature engineering, sampling and model training
    pipeline = imb_pipline([
        ('data_preprocessing', col_trans),
        ('sampling', technique),
        ('model', LogisticRegression(solver="lbfgs",class_weight={1:1, 0:wt},
                                     max_iter=1000, random_state=config.RANDOM_STATE))
    ])

    # train model
    model = pipeline.fit(x_train, t_train)

    # precision recall curve for val dataset
    avg_pr_score = pr_curve_avg_precision_score(model, x_val, t_val)

    # Metrics' values
    report = model_eval_report(model, x_val, t_val)

    # Printing Vales and statistics in a fancy style using lib tabulate
    data = [
        ['1 (Fraud)', f'{report['1']['precision']:.3f}',f'{report['1']['recall']:.3f}',
         f'{report['1']['f1-score']:.4f}'],
        ['0 (Genuine)', f'{report['0']['precision']:.3f}', f'{report['0']['recall']:.4f}',
         f'{report['0']['f1-score']:.4f}']
    ]

    headers = ['Class', 'Precision', 'Recall', 'F1-score']      # table headers
    print(tabulate(data, headers=headers, tablefmt='github'))

    data = [
        ['Logistic Regression',
         f'{report['accuracy']:.3f}',f'{report['macro avg']['f1-score']:.3f}',
         f'{report['weighted avg']['f1-score']:.3f}', f'{report['harmonic avg']:.3f}', f'{avg_pr_score:.3f}']
    ]
    headers = ['Model','Accuracy','Macro Avg', 'Weighted Avg', 'Harmonic Avg', 'AUPRC']     # table headers
    print('#' * 85 + '\n')
    print('\t' * 8 +'** Summary Statistics **')
    print(tabulate(data,headers=headers, tablefmt='github'))

if __name__ == '__main__':
    logistic_regression_model(sample_technique='oversampling')
#     | Class         | Precision     | Recall     | F1 - score   |
#     | ------------- | ------------- | ---------- | ------------ |
#     | 1 (Fraud)     | 0.894         | 0.656      | 0.7564       |
#     | 0 (Genuine)   | 0.999         | 0.9999     | 0.9997       |
#     #############################################################################################
#                                ** Summary Statistics **
#     |       Model           |  Accuracy    |   Macro Avg   |  Weighted Avg    |   Harmonic Avg   |   AUPRC  |
#     | --------------------- | ------------ | ------------- | ---------------- | ---------------- |----------|
#     | Logistic Regression   |     0.999    |      0.878    |      0.999       |       0.861      |   0.76   |