import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import hmean
from sklearn.metrics import precision_recall_curve, classification_report, average_precision_score, fbeta_score


def model_eval_report(model, X, y_true, threshold=0.5):
    '''
    get classification report with customized threshold
    :param model: trained model
    :param X: input
    :param y_true: target
    :param threshold: more control (default=0.5)
    :return:
    '''

    y_pred = (model.predict_proba(X)[:, 1] >= threshold).astype(int)   # get predictions with specific threshold

    # calculate harmonic mean of f1-scores of both classes (sense small values which show weak classification angles of the model)
    report_dict = classification_report(y_true, y_pred, output_dict=True)    # get results in dict
    report_dict['hmean'] = hmean([report_dict['1']['f1-score'], report_dict['0']['f1-score']])

    return report_dict     # return harmonic mean & report


def avg_pr_fb_score(model, X, y_true, beta=1, show_plot=False):
    '''
    calculate avg precision and plot precision recall curve with the best threshold
    :param model: trianed model
    :param X: input
    :param y_true: target
    :param beta: f(beat)score
    :param show_plot: if True it'll show the plot
    :return result: dic with metrics scores
    '''

    y_proba = model.predict_proba(X)[:, 1]      # get predicted probabilities
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)      # precisions, recalls & thresholds

    # get our best_threshold(respect to class 1) and f-beta score for class 1
    fscores = [fbeta_score(y_true, y_proba >= th, beta=beta, pos_label=1) for th in thresholds]     # list of f-scores with different thresholds
    best_idx = np.argmax(fscores)                # return index of highest f-score
    best_threshold = thresholds[best_idx]       # get corresponding threshold of highest f-score

    # let's calculate f-scores for both classes with respect to best threshold
    score_1 = fscores[best_idx]          # class 1
    score_0 = fbeta_score(y_true, y_proba >= best_threshold, beta=beta, pos_label=0)      # class 0
    auprc = average_precision_score(y_true, y_proba)        # avg precision score

    # results in dict
    result = {
        'AUPRC': auprc,
        f'best_threshold(f{beta}-score)': best_threshold,
        f'f{beta}-score class_0': score_0,
        f'f{beta}score class_1': score_1,
    }


    if show_plot:
        # plot precision recall curve
        precision, recall = precisions[:-1], recalls[:-1]     # exclude last value
        plt.plot(thresholds, precision, linestyle = '--', color='blue', label='Precision')       # threshold vs precision
        plt.plot(thresholds, recall, linestyle='--', color='green', label='Recall')            # threshold vs recall
        plt.axvline(best_threshold, color='red', linestyle='-', label=f'Best threshold: {best_threshold:.3f}')
        plt.title('Threshold VS Precision Recall', fontweight=12, fontstyle='italic', color='grey')
        plt.xlabel('Threshold', color='orange')
        plt.ylabel('Precision-Recall', color='orange')
        plt.legend(loc='best')
        plt.show()

    return result