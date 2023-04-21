"""
This module contains functions for preprocessing of the data.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from mrmr import mrmr_classif


def drop_columns(data, columns):
    """
    Drop the columns from the dataset.
    Args:
        data (pandas.DataFrame): The dataset to drop the columns from.
        columns (list): The list of columns to drop.
    Returns:
        pandas.DataFrame: The dataset with the columns dropped.
    """
    return data.drop(columns, axis=1)

def standardize_data(data, columns):
    """
    Standardize the data by subtracting the mean and dividing by the standard deviation.
    Args:
        data (pandas.DataFrame): The dataset to standardize.
        columns (list): The list of columns to standardize.
    Returns:
        pandas.DataFrame: The standardized dataset.
    """
    scaler = StandardScaler()
    data[columns] = scaler.fit_transform(data[columns])
    return data

def oversample_data(data, target):
    """
    Oversample the data using SMOTE.
    Args:
        data (pandas.DataFrame): The dataset to oversample.
        target (str): The name of the target column.
    Returns:
        pandas.DataFrame: The oversampled dataset.
    """
    
    sm = SMOTE(random_state=42)
    X = data.drop(target, axis=1)
    y = data[target]
    X_res, y_res = sm.fit_resample(X, y)
    return pd.concat([X_res, y_res], axis=1)

def split_data(data, target, test_size=0.2):
    """
    Split the data into training and testing sets.
    Args:
        data (pandas.DataFrame): The dataset to split.
        target (str): The name of the target column.
        test_size (float): The size of the test set.
    Returns:
        tuple: The training and testing sets.
    """
    
    X = data.drop(target, axis=1)
    y = data[target]
    return train_test_split(X, y, test_size=test_size, random_state=42)

def get_xy(data, target):
    """
    Split the data into X and y.
    Args:
        data (pandas.DataFrame): The dataset to split.
        target (str): The name of the target column.
    Returns:
        tuple: The X and y.
    """
    
    X = data.drop(target, axis=1)
    y = data[target]
    return X, y

def mrmr_preprocess(data, target, n_selected_features=10):
    """
    Preprocess the data by selecting the best features using mRMR.
    
    Ref: https://towardsdatascience.com/mrmr-explained-exactly-how-you-wished-someone-explained-to-you-9cf4ed27458b
    Args:
        data (pandas.DataFrame): The dataset to preprocess.
        target (str): The name of the target column.
        n_selected_features (int): The number of features to select.
    Returns:
        pandas.DataFrame: The preprocessed dataset.
    """
    X = data.drop(target, axis=1)
    y = data[target]
    selected_features = mrmr_classif(X, y, n_selected_features)
    new_data = data[selected_features]
    new_data[target] = y
    return new_data
    
    

def gpp_preprocess(data, columns_to_drop, columns_to_standardize):
    """
    Preprocess the data by dropping columns, standardizing the data, and oversampling the data.
    Args:
        data (pandas.DataFrame): The dataset to preprocess.
        target (str): The name of the target column.
        columns_to_drop (list): The list of columns to drop.
        columns_to_standardize (list): The list of columns to standardize.
    Returns:
        pandas.DataFrame: The preprocessed dataset.
    """
    
    data = drop_columns(data, columns_to_drop)
    data = standardize_data(data, columns_to_standardize)
    return data