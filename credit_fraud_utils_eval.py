import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import hmean
from sklearn.metrics import precision_recall_curve, classification_report, average_precision_score, fbeta_score


def model_eval_report(model, X, y_true, threshold=0.5):
    '''
    Calculate metrics:
        1- precision
        2- recall
        3- f1-score (each class - micro avg - macro avg - weighted avg - harmonic mean)
     parameters:
        model: trained model
        X: input features
        y_true: ground truth
        threshold: control metrics calculations
    return:
        report: harmonic mean & report
    '''
    y_pred = (model.predict_proba(X)[:, 1] >= threshold).astype(int)   # get predictions with specific threshold

    # calculate harmonic mean of f1-scores of both classes (sense small values which show weak classification angles of the model)
    report_dict = classification_report(y_true, y_pred, output_dict=True, digits=3)    # get results in dict
    harmonic_mean = hmean([report_dict['1']['f1-score'], report_dict['0']['f1-score']])

    # get report
    report = classification_report(y_true, y_pred, output_dict=False)

    return report, harmonic_mean     # return harmonic mean & report


def pr_curve_fbeta_score(model, X, y_true, beta_score=1):
    '''
    Plotting precision recall curve that show precision recall values with different thresholds
    and calculate best threshold and average precision
    parameter:
        model:trained model
        X: input features for prediction
        y_true: ground truth
        beta_score: indicates f-score(f1, f0.5, f2)
    return:
        result as a dict type (auprc, best_threshold(respect-to-class-1), f-beta scores of both classes)
    '''
    y_proba_1 = model.predict_proba(X)[:, 1]      # get predicted probabilities
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba_1)      # precisions, recalls & thresholds

    # plot precision recall curve
    # precision, recall = precisions[:-1], recalls[:-1]     # exclude last value
    # plt.plot(thresholds, precision, linestyle = '--', color='blue', label='Precision')       # threshold vs precision
    # plt.plot(thresholds, recall, linestyle='--', color='red', label='Recall')            # threshold vs recall
    # plt.title('Threshold VS Precision Recall', fontweight=12, fontstyle='italic', color='grey')
    # plt.xlabel('Threshold', color='green')
    # plt.ylabel('Precision-Recall', color='green')
    # plt.legend(loc='best')
    # plt.show()

    # get average precision score
    auprc = average_precision_score(y_true, y_proba_1)

    # get our best_threshold(respect to class 1) and f-beta score for class 1
    scores = [fbeta_score(y_true, y_proba_1 >= th, beta=beta_score) for th in thresholds]     # list of f-scores with different thresholds
    best_idx = np.argmax(scores)                # return index of highest f-score
    score_1 = scores[best_idx]                  # get highest f-score for class 1
    best_threshold = thresholds[best_idx]       # get corresponding threshold of highest f-score

    # let's calculate f-score for class-0 using best_threshold(class-1)
    y_proba_0 = model.predict_proba(X)[:, 0]
    score_0 = fbeta_score(y_true, y_proba_0 >= best_threshold, beta=beta_score)

    # dict for auprc score, best_threshold and f-beta scores of both classes(according to best_threshold)
    result = {
        'auprc': auprc,
        'best_threshold': best_threshold,
        'f-score_0': score_0,
        'f-score_1': score_1,
    }

    return result