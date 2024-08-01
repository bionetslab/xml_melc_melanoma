import tensorflow as tf 
import numpy as np
import cv2
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score, precision_score, confusion_matrix

def rgb_on_background(background, r=None, g=None, b=None):
    r = np.zeros(background.shape[:2]) if not isinstance(r, np.ndarray) else r    
    g = np.zeros(background.shape[:2]) if not isinstance(g, np.ndarray) else g
    b = np.zeros(background.shape[:2]) if not isinstance(b, np.ndarray) else b
    rgb = np.array([r, g, b], dtype=np.uint8).transpose((1, 2, 0))
    return cv2.addWeighted(background, 0.4, rgb, 0.6, 0)  
    

def print_metrics(y_true, y_pred):
    print("Accuracy", accuracy_score(y_true, y_pred))
    print("F1", f1_score(y_true, y_pred))
    print("AUC-ROC", roc_auc_score(y_true, y_pred))
    print("Recall", recall_score(y_true, y_pred))
    print("Precision", precision_score(y_true, y_pred))
    print("Accuracy in negative class", accuracy_score(y_true[np.where(y_true==0)], y_pred[np.where(y_true==0)]))
    print("Accuracy in positive class", accuracy_score(y_true[np.where(y_true==1)], y_pred[np.where(y_true==1)]))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    print(f'Sensitivity (Recall): {sensitivity}')
    specificity = tn / (tn + fp)
    print(f'Specificity: {specificity}')
    

def get_scalar_run_tensorboard(tag, filepath):
    values,steps = [],[]
    for e in tf.compat.v1.train.summary_iterator(filepath):
        if len(e.summary.value) > 0:
            if e.summary.value[0].tag == tag:
                value, step = (e.summary.value[0].simple_value, e.step)
                values.append(value)
                steps.append(step)
    return values, steps