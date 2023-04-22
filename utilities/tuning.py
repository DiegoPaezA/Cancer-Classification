"""
This module contains functions for tuning the hyperparameters of a model.
"""
import yaml
from sklearn.model_selection import GridSearchCV
from utilities.preprocessing import split_data, get_xy, oversample_data

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

def tune_experiment(data,target, parameters):
    """
    Tune the hyperparameters of a model using GridSearchCV and return the 
    accuracy score on the test set.
    
    Args:
        data (pandas.DataFrame): the data
        target (str): the name of the target column
        parameters (dict): the parameters to tune
    Returns:
        tuned_params (dict): the tuned parameters
    """
    
    grid_search = GridSearchCV(model, parameters, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    return grid_search.score(X_test, y_test)