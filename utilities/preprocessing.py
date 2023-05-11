"""
This module contains functions for preprocessing of the data.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from mrmr import mrmr_classif


def drop_columns(data:pd.DataFrame, columns:list):
    """
    Drop the columns from the dataset.
    Args:
        data (pandas.DataFrame): The dataset to drop the columns from.
        columns (list): The list of columns to drop.
    Returns:
        pandas.DataFrame: The dataset with the columns dropped.
    """
    return data.drop(columns, axis=1)
def drop_rows(data:pd.DataFrame, rows:list):
    """
    Drop the rows from the dataset.
    Args:
        data (pandas.DataFrame): The dataset to drop the rows from.
        rows (list): The list of rows to drop.
    Returns:
        pandas.DataFrame: The dataset with the rows dropped.
    """
    return data.drop(rows, axis=0)

def standardize_data(data:pd.DataFrame, columns:list):
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

def oversample_data(data:pd.DataFrame, target:str):
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

def split_data(data:pd.DataFrame, target:str, test_size=0.2):
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
    return train_test_split(X, y, test_size=test_size)

def get_xy(data:pd.DataFrame, target:str):
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

def kmeans_preprocess(data:pd.DataFrame, target:str, n_clusters=2):
    """
    This function preprocesses the data by clustering the target column using KMeans.
    
    Args:
        data : pandas.DataFrame
            The dataset to preprocess
        target : str 
            The name of the target column
        n_clusters : int 
            The number of clusters to use for KMeans
    Returns:
        pandas.DataFrame: The preprocessed dataset.
    """
    target_data = data[target].values.reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init="auto").fit(target_data)
    target_kmeans = kmeans.predict(target_data)
    y = pd.DataFrame(target_kmeans, columns=[target])
    X = data.drop(target, axis=1)
    data = pd.concat([X, y], axis=1)
    return data

def mrmr_preprocess(data:pd.DataFrame, target:str, n_selected_features=10):
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
    
    
def gpp_preprocess(data:pd.DataFrame, columns_to_drop:str, columns_to_standardize:str):
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
    
    idx_Men = np.where(data.Sex == 1)
    data = drop_rows(data, idx_Men[0])
    data = drop_columns(data, columns_to_drop)
    data = standardize_data(data, columns_to_standardize)
    return data