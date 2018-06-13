# -*- coding: utf-8 -*-
# @Time    : 2018/5/21 19:30
# @Author  : LeonHardt
# @File    : score.py



import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import roc_curve

def score_roc(y, pred):
    fpr, tpr, thresholds = roc_curve(y, pred, pos_label=1)
    score_final = 0.4*tpr[np.where(fpr >= 0.001)[0][0]]\
                  +0.3*tpr[np.where(fpr >= 0.005)[0][0]]+0.3*tpr[np.where(fpr >= 0.01)[0][0]]
    return score_final

def score(y_true, y_score):
    """ Evaluation metric
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    score = 0.4 * tpr[np.where(fpr >= 0.001)[0][0]] + \
            0.3 * tpr[np.where(fpr >= 0.005)[0][0]] + \
            0.3 * tpr[np.where(fpr >= 0.01)[0][0]]
    return score

def evaluate(y_true, y_pred, y_prob):
    """ 估计结果: precision, recall, f1, auc, score
    """
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    mayi = score(y_true, y_prob)

    return [p, r, f1, auc, mayi]