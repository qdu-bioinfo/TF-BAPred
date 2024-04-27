import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, auc
import matplotlib.pyplot as plt


def calculate_performace(test_num, pred_y, labels):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for index in range(test_num):
        if labels[index] == 1:
            if labels[index] == pred_y[index]:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if labels[index] == pred_y[index]:
                tn = tn + 1
            else:
                fp = fp + 1
    print("TP:", tp)
    print("FP:", fp)
    print("TN:", tn)
    print("FN:", fn)
    accuracy = float(tp + tn) / (tp + fp + tn + fn)  # 准确率
    precision = float(tp) / (tp + fp)  # 精准度
    sensitivity = float(tp) / (tp + fn)  # 召回率
    specificity = float(tn) / (tn + fp)
    f1_score = 2 * precision * sensitivity / (precision + sensitivity)
    MCC = float(tp * tn - fp * fn) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))

    return accuracy, precision, sensitivity, specificity, f1_score, MCC


def plot_roc_curve(labels, probality, legend_text, auc_tag=True):  # AUC
    # fpr2, tpr2, thresholds = roc_curve(labels, pred_y)
    fpr, tpr, thresholds = roc_curve(labels, probality)  # probas_[:, 1])
    roc_auc = auc(fpr, tpr)
    if auc_tag:
        rects1 = plt.plot(fpr, tpr, label=legend_text + ' (AUC=%6.3f) ' % roc_auc)
    else:
        rects1 = plt.plot(fpr, tpr, label=legend_text)
