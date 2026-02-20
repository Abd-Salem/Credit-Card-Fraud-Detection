from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import  Pipeline
import config
from credit_fraud_utils_data import prepare_data, load_data
from credit_fraud_utils_eval import model_eval_report, pr_curve_plot
from tabulate import tabulate

def train_model_1():
    '''
    - Train our dataset using logistic regression algorithm (baseline model)
    - No techniques are used for Imbalanced data
    '''

    # load train-val dataset
    train_df = load_data(config.DATASET['train_path'])
    val_df = load_data(config.DATASET['val_path'])

    # Applying preparing steps(feature extraction, feature transformation) on train-val dataset
    col_trans = prepare_data(train_df, inplace=True,
                                       numeric_features=config.DATASET['numeric_features'],
                                       log_trans_cols=config.DATASET['log_cols'])
    prepare_data(val_df, inplace=True,
                 numeric_features=config.DATASET['numeric_features'],
                 log_trans_cols=config.DATASET['log_cols'])

    # get our final results after preparing data (input columns names)
    input_cols_names = train_df.columns.tolist()
    input_cols_names.remove(config.DATASET['target_feature'])

    # pipeline data preparing with modeling
    pipeline = Pipeline([
        ('data_preprocessing', col_trans),
        ('model', LogisticRegression(solver="lbfgs", max_iter=1000, random_state=config.RANDOM_STATE))
    ])

    # train model
    model = pipeline.fit(train_df[input_cols_names], train_df[config.DATASET['target_feature']])

    # precision recall curve for val dataset as inference
    pr_curve_plot(model, val_df[input_cols_names], val_df[config.DATASET['target_feature']])

    # Metrics' values
    report = model_eval_report(model, val_df[input_cols_names], val_df[config.DATASET['target_feature']])

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
         f'{report['weighted avg']['f1-score']:.3f}', f'{report['harmonic avg']:.3f}']
    ]
    headers = ['Model','Accuracy','Macro Avg', 'Weighted Avg', 'Harmonic Avg']     # table headers
    print('####################################################################################\n')
    print('\t' * 8 +'** Summary Statistics **')
    print(tabulate(data,headers=headers, tablefmt='github'))


if __name__ == '__main__':
    
    train_model_1()
#     | Class         | Precision     | Recall     | F1 - score   |
#     | ------------- | ------------- | ---------- | ------------ |
#     | 1(Fraud)      | 0.875         | 0.544      | 0.6712       |
#     | 0(Genuine)    | 0.999         | 0.9999     | 0.9996       |
#     #############################################################################################
#                                ** Summary Statistics **
#     |       Model           |  Accuracy    |   Macro Avg   |  Weighted Avg    |   Harmonic Avg   |
#     | --------------------- | ------------ | ------------- | ---------------- | ---------------- |
#     | Logistic Regression   |     0.999    |      0.835    |      0.999       |       0.803      |