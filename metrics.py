import os, json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    accuracy_score,
    precision_recall_fscore_support
)

def AUC_Confidence_Interval(y_true, y_pred, CI_index=0.95):
    '''
    This function can help calculate the AUC value and the confidence intervals. It is note the confidence interval is
    not calculated by the standard deviation. The auc is calculated by sklearn and the auc of the group are bootstraped
    1000 times. the confidence interval are extracted from the bootstrap result.

    Ref: https://onlinelibrary.wiley.com/doi/abs/10.1002/%28SICI%291097-0258%2820000515%2919%3A9%3C1141%3A%3AAID-SIM479%3E3.0.CO%3B2-F
    :param y_true: The label, dim should be 1.
    :param y_pred: The prediction, dim should be 1
    :param CI_index: The range of confidence interval. Default is 95%
    :return: The AUC value, a list of the confidence interval, the boot strap result.
    '''

    single_auc = roc_auc_score(y_true, y_pred)

    bootstrapped_scores = []

    np.random.seed(42) # control reproducibility
    seed_index = np.random.randint(0, 65535, 1000)
    for seed in seed_index.tolist():
        np.random.seed(seed)
        pred_one_sample = np.random.choice(y_pred, size=y_pred.size, replace=True)
        np.random.seed(seed)
        label_one_sample = np.random.choice(y_true, size=y_pred.size, replace=True)

        if len(np.unique(label_one_sample)) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue

        score = roc_auc_score(label_one_sample, pred_one_sample)
        bootstrapped_scores.append(score)

    sorted_scores = np.array(bootstrapped_scores)
    std_auc = np.std(sorted_scores)
    mean_auc = np.mean(sorted_scores)
    sorted_scores.sort()

    # Computing the lower and upper bound of the 90% confidence interval
    # You can change the bounds percentiles to 0.025 and 0.975 to get
    # a 95% confidence interval instead.
    confidence_lower = sorted_scores[int((1.0 - CI_index) / 2 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(1.0 - (1.0 - CI_index) / 2 * len(sorted_scores))]
    CI = [confidence_lower, confidence_upper]
    # final_auc = (confidence_lower+confidence_upper)/2
    # print('AUC is {:.3f}, Confidence interval : [{:0.3f} - {:0.3}]'.format(AUC, confidence_lower, confidence_upper))
    return single_auc, mean_auc, CI, sorted_scores, std_auc


def cutoff_youdens(fpr,tpr,thresholds):
    scores = tpr-fpr
    orders = sorted(zip(scores,thresholds, range(len(scores))))
    return orders[-1][1], orders[-1][-1]


def save_roc_curve_fn(y_preds, y_targets, save_dir, MOD, is_multilabel=False):
    fpr, tpr, thresholds = roc_curve(y_targets, y_preds)
    auc = roc_auc_score(y_targets, y_preds)

    np.savetxt(
        os.path.join(save_dir, 'roc_scores.csv'),
        np.stack([thresholds, fpr, tpr]).transpose(),
        delimiter=',',
        fmt='%f',
        header='Threshold,FPR,TPR',
        comments=''
    )

    average_type = 'binary' if not is_multilabel else None
    best_th, best_idx = cutoff_youdens(fpr, tpr, thresholds)
    print(best_th)
    precision, recall, f1, _ = precision_recall_fscore_support(y_targets, y_preds>=best_th, average=average_type)
    acc = accuracy_score(y_targets, y_preds>=best_th)
    print('Best precision, recall, f1:', precision, recall, f1)
    with open(os.path.join(save_dir, 'classification_results{}.json'.format(MOD)), 'w') as f:
        json.dump( {
            'Best threshold': float(best_th),
            'Precision': float(precision),
            'Recall': float(recall),
            'Accuracy': float(acc),
            'False positive rate': float(fpr[best_idx]),
            'True positive rate': float(tpr[best_idx]),
            'f1': float(f1),
            }, f, indent=2 )

    return 0


