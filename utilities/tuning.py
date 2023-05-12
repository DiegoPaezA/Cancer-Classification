"""
This module contains functions for tuning the hyperparameters of a model.
"""
import numpy as np
import pandas as pd
import json

from sklearn.model_selection import GridSearchCV
from utilities.preprocessing import split_data, get_xy, oversample_data

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier


firts_time = True

def save_tuned_params(params:dict, param_file:str):
    """
    Save the tuned parameters to a json file.
    
    Args:
        params (dict): dictionary with the parameters
        param_file (str): path to the YAML file
    """
    with open(param_file, 'w') as f:
        json.dump(params, f, indent=4)
    f.close()
    print("Tuned parameters saved to ...")

def load_params(param_file:json):
    """
    Load the parameters from a json file and return a dict with the parameters.
    
    Args:
        param_file (str): path to the json file
    Returns: 
        params (dict): dictionary with the parameters
    """
    with open(param_file, 'r') as f:
        params = json.load(f)
    return params

def tune_hyperparameters(model,data:pd.DataFrame,target:str,Ename:str, parameters:dict):
    """
    Tune the hyperparameters of a model using GridSearchCV.
    
    Args:
        model
    """
    global firts_time
    if Ename == "E4":
        data = oversample_data(data, target)
        # if (firts_time):
        #     print("Oversampling done...")
        #     print("Class distribution in the oversampled data:")
        #     counter = Counter(data["Class"])
        #     print('%s : %d' % ('no Recurrence', counter[0]))
        #     print('%s : %d' % ('Recurrence', counter[1]))
        #     firts_time = False
            
    X, y = get_xy(data, target)
    X_train, X_test, y_train, y_test = split_data(X, y)
    grid_search = GridSearchCV(model, parameters, cv=10, scoring="accuracy", n_jobs=-1, verbose=1, refit=True)
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_params_

def tune_experiment(data:pd.DataFrame,target:str,Ename:str,tuneparams:dict, defaultparams:dict):
    """
    Tune the hyperparameters of a model using GridSearchCV and return the 
    accuracy score on the test set.
    
    Args:
        data (pandas.DataFrame): the data
        target (str): the name of the target column
        Ename (str): The name of the experiment (E1, E2, E3, E4)
        tuneparams (dict): the parameters to tune
    Returns:
        tuned_params (dict): the tuned parameters
    """
    # update nayive bayes parameters
    tuneparams["naive_bayes"]["var_smoothing"] = np.logspace(0,-9, num=1000)
    keys = list(tuneparams.keys())
    models = []
    models.append(('LR', LogisticRegression(**defaultparams["log_reg"])))
    models.append(('NB', GaussianNB(**defaultparams["naive_bayes"])))
    models.append(('SVM', SVC(**defaultparams["svm"])))
    models.append(('KNN', KNeighborsClassifier(**defaultparams["knn"])))
    models.append(('XGB', XGBClassifier(**defaultparams["xgb"])))
    results = []
    
    dict_best_params = {
        "log_reg":{},
        "naive_bayes":{},
        "svm":{},
        "knn":{},
        "xgb":{},
    }

    for idx, model in enumerate(models):
        name = model[0]
        model = model[1]
        print(f"Tuning hyperparameters for {name}...")
        best_param = (tune_hyperparameters(model,data,target,Ename,tuneparams[keys[idx]]))
        dict_best_params[keys[idx]] = best_param
        
    return dict_best_params