# src/eval.py
from sklearn.metrics import fbeta_score, f1_score, precision_recall_fscore_support
import numpy as np

def f2_score(y_true, y_pred):
    # expects binary labels 0/1
    return fbeta_score(y_true, y_pred, beta=2, zero_division=0)

def summary(y_true, y_pred):
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    f2 = f2_score(y_true, y_pred)
    return {'precision': p, 'recall': r, 'f1': f1, 'f2': f2}
