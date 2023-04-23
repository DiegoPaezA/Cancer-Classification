"""
This module contains functions for tuning the hyperparameters of a model.
"""
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import GridSearchCV
from utilities.preprocessing import split_data, get_xy, oversample_data

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from collections import Counter

firts_time = True


def load_params(param_file):
    """
    Load the parameters from a YAML file and return a dict with the parameters.
    
    Args:
        param_file (str): path to the YAML file
    Returns: 
        params (dict): dictionary with the parameters
    """
    with open(param_file, 'r') as f:
        params = yaml.safe_load(f)
    return params

def tune_hyperparameters(model,data:pd.Dataframe,target:str,Ename:str, parameters:dict):
    """
    Tune the hyperparameters of a model using GridSearchCV.
    
    Args:
        model
    """
    global firts_time
    if Ename == "E4":
        data = oversample_data(data, target)
        if (firts_time):
            print("Oversampling data...")
            print("Class distribution in the oversampled data:")
            counter = Counter(data["Class"])
            print('%s : %d' % ('no Recurrence', counter[0]))
            print('%s : %d' % ('Recurrence', counter[1]))
            firts_time = False
    
    X, y = get_xy(data, target)
    X_train, X_test, y_train, y_test = split_data(data, target, test_size=0.2)
    grid_search = GridSearchCV(model, parameters, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    return grid_search.best_params_

def tune_experiment(data:pd.Dataframe,target:str,Ename:str,parameters:dict):
    """
    Tune the hyperparameters of a model using GridSearchCV and return the 
    accuracy score on the test set.
    
    Args:
        data (pandas.DataFrame): the data
        target (str): the name of the target column
        Ename (str): The name of the experiment (E1, E2, E3, E4)
        parameters (dict): the parameters to tune
    Returns:
        tuned_params (dict): the tuned parameters
    """
    
    models = []
    models.append(('LR', LogisticRegression()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('XGB', XGBClassifier()))
    results = []
    
    for name, model in models:
        results.append(tune_hyperparameters(model,data, target, Ename, parameters))
    
    return 1