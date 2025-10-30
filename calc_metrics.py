import pandas as pd
import numpy as np
import math
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score, precision_score
import os
import json

label_csv = pd.read_csv('/dir_path/dataset_multi_abnormality_labels_valid_predicted_labels.csv')

def find_threshold(probabilities, true_labels):
    """
    Finds the optimal threshold for binary classification based on ROC curve.

    Args:
        probabilities (numpy.ndarray): Predicted probabilities.
        true_labels (numpy.ndarray): True labels.

    Returns:
        float: Optimal threshold.
    """
    best_threshold = 0
    best_roc = 10000

    # Iterate over potential thresholds
    thresholds = np.linspace(0, 1, 100)
    for threshold in thresholds:
        predictions = (probabilities > threshold).astype(int)
        confusion = confusion_matrix(true_labels, predictions)
        TP = confusion[1, 1]
        TN = confusion[0, 0]
        FP = confusion[0, 1]
        FN = confusion[1, 0]
        TP_r = TP / (TP + FN)
        FP_r = FP / (FP + TN)
        current_roc = math.sqrt(((1 - TP_r) ** 2) + (FP_r ** 2))
        if current_roc <= best_roc:
            best_roc = current_roc
            best_threshold = threshold

    return best_threshold



csv_path = 'ZeroShotResult_rate.csv'
result = pd.read_csv(csv_path)

columns = list(result.columns[1:])

auc_scores = {}
spec_scores = {}
sens_scores = {}
acc_scores = {}
f1_scores = {}
prec_scores = {}

for column in columns:
    labels = []
    probs = []
    
    abnormality = column.split('_')[1]

    num = 0
    for file_name, prob in zip(result['file_name'], result[column]):
        if np.isnan(prob):
            num += 1
            continue
        probs.append(prob)
        labels.append(label_csv[label_csv['VolumeName'] == file_name][abnormality].values[0])

    print(column, 'AUC:', round(roc_auc_score(labels, probs), 3))

    auc_scores[abnormality] = max(auc_scores.get(abnormality, 0), roc_auc_score(labels, probs))

    threshold = find_threshold(np.array(probs), np.array(labels))
    
    pd_labels = (np.array(probs) > threshold).astype(int)
    gt_labels = np.array(labels)

    tp = ((pd_labels == 1) & (gt_labels == 1)).sum().item()
    fp = ((pd_labels == 1) & (gt_labels == 0)).sum().item()
    tn = ((pd_labels == 0) & (gt_labels == 0)).sum().item()
    fn = ((pd_labels == 0) & (gt_labels == 1)).sum().item()

    if tn + fp != 0:
        spec = tn / (tn + fp)
    else:
        spec = np.nan

    if tp + fn != 0:
        sens = tp / (tp + fn)
    else:
        sens = np.nan

    spec_scores[abnormality] = max(spec_scores.get(abnormality, 0), spec)
    sens_scores[abnormality] = max(sens_scores.get(abnormality, 0), sens)

    f1 = f1_score(gt_labels, pd_labels, average="weighted")
    prec = precision_score(gt_labels, pd_labels)

    f1_scores[abnormality] = max(f1_scores.get(abnormality, 0), f1)
    prec_scores[abnormality] = max(prec_scores.get(abnormality, 0), prec)

    acc = (tp + tn) / (tp + tn + fp + fn)
    acc_scores[abnormality] = max(acc_scores.get(abnormality, 0), acc)


print('\n')

print('Average AUC of Ours:', round(np.mean(list(auc_scores.values())), 3))
print('Average ACC of Ours:', round(np.mean(list(acc_scores.values())), 3))
print('Average Spec of Ours:', round(np.mean(list(spec_scores.values())), 3))
print('Average Sens of Ours:', round(np.mean(list(sens_scores.values())), 3))
print('Average F1 of Ours:', round(np.mean(list(f1_scores.values())), 3))
print('Average Prec of Ours:', round(np.mean(list(prec_scores.values())), 3))

