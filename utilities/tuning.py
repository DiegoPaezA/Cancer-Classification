"""
This module contains functions for tuning the hyperparameters of a model.
"""
import yaml
from sklearn.model_selection import GridSearchCV

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

def tune_hyperparameters(model, parameters, X_train, y_train):
    """
    Tune the hyperparameters of a model using GridSearchCV.
    """
    grid_search = GridSearchCV(model, parameters, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    return grid_search.best_params_