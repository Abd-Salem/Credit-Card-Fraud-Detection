import matplotlib.pyplot as plt
from scipy.stats import hmean
from sklearn.metrics import precision_recall_curve, classification_report


def model_eval_report(model, X, y_true):
    '''
    Calculate metrics:
        1- precision
        2- recall
        3- f1-score (each class - micro avg - macro avg - weighted avg)
     parameters:
        model: trained model
        X: inference data for prediction
        y_true: ground truth of the class
    return:
        report: dict contains metrics' values
    '''
    y_pred = model.predict(X)   # predict classes
    report = classification_report(y_true, y_pred, output_dict=True)    # get results in dict

    # calculate harmonic mean of f1-scores of classes (sense small values which show weak classification angles of the model)
    report['harmonic avg'] = hmean([report['1']['f1-score'], report['0']['f1-score']])
    return report


def pr_curve_plot(model, X, y_true):
    '''
    Plotting precision recall curve that show precision recall values with different thresholds
    parameter:
        model:trained model
        X: inference data for prediction
        y_true: ground truth of the class
    '''
    y_proba = model.predict_proba(X)[:, 1]      # get predicted probabilities
    precision, recall, threshold = precision_recall_curve(y_true, y_proba)      # precisions, recalls with different thresholds

    precision, recall = precision[:-1], recall[:-1]     # exclude last value
    plt.plot(threshold, precision, linestyle = '--', color='blue', label='Precision')       # threshold vs precision
    plt.plot(threshold, recall, linestyle='--', color='red', label='Recall')            # threshold vs recall
    plt.title('Precision Recall Curve', fontweight=12, fontstyle='italic', color='grey')
    plt.legend(loc='best')
    plt.show()