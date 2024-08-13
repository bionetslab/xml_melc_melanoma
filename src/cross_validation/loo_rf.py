import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
import sys
sys.path.append("..")
from src import *


def get_histo_id_level_predictions(data, predictions):
    data = data.copy().set_index("Histo ID")
    predictions = predictions.copy().set_index("Histo ID")
    y_pred = list()
    y_true = list()
    for histo_id in data.index.unique():
        y_true.append(np.unique(data.loc[histo_id]["PFS < 5"])[0])
        #y_true.append(np.unique(data.loc[histo_id]["PFS label"])[0])
        y_pred.append(np.mean(predictions.loc[histo_id]).item())
    return y_true, y_pred
    

def fit_forest(X_train, y_train, X_val, y_val, clf=RandomForestClassifier):
    clf = clf(class_weight="balanced")
    clf.fit(X_train.values, y_train)
    if len(X_val.shape) == 1:
        # special treatment for single-sample patients
        X_val = X_val.values.reshape(1, -1)
        y_val = np.array(y_val).reshape(1, -1)
    else:
        X_val = X_val.values
    y_pred = clf.predict(X_val)
    return y_pred, clf.feature_importances_
    

def loo_forest(data, features, label):
    assert "Patient ID" in data.columns
    predictions = pd.DataFrame(index=data["Patient ID"])
    patients = data["Patient ID"].unique()
    data = data.copy()
    data.set_index("Patient ID", inplace=True)
    
    data[features] = data[features].astype(float)
    data[features] = (data[features] - data[features].mean(axis=0)) / data[features].std(axis=0)
    
    feature_importances = np.ndarray((len(patients), len(features)))
    
    for i, patient in enumerate(patients):
        X_val = data.loc[patient][features]
        y_val = data.loc[patient][label]       
        X_train = data.loc[data.index.difference([patient])][features]
        y_train = data.loc[data.index.difference([patient])][label]
        y_pred, feature_importance = fit_forest(X_train, y_train, X_val, y_val)
        try:
            predictions.loc[patient, "Prediction"] = y_pred 
            predictions.loc[patient, "Histo ID"] = data.loc[patient, "Histo ID"]
        except Exception as e:
            predictions.loc[patient, "Prediction"] = y_pred[0]
        feature_importances[i] = feature_importance * len(X_train)
    #data["Prediction"] = np.concatenate(y_preds)
    data.reset_index(inplace=True)
    feature_importances = np.mean(feature_importances, axis=0)
    feature_importances /= np.sum(feature_importances)
    feature_importance_df = pd.DataFrame(feature_importances, index=features).T
    return predictions, feature_importance_df