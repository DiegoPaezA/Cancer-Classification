"""
This file contains the functions to evaluate the models
"""
import numpy as np
np.random.seed(42)

from numpy import mean
from numpy import std
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from preprocessing import split_data, get_xy


def train_model(model, data, target, test_size=0.2):
  """
  This functions trains a model with kfold cross validation and also `model.fit`.
  
  Args:
    model (sklearn model): The model to train.
    data (pandas.DataFrame): The dataset to train the model.
    target (str): The name of the target column.
    test_size (float): The size of the test set.
  """
  X, y = get_xy(data, target)
  X_train, X_test, y_train, y_test = split_data(data, target, test_size)
  
  cv = KFold(n_splits=10, random_state=42, shuffle=True)
  scores_origin = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
  
  accuracy_kfold = round(mean(scores_origin),3)
  std_kfold = round(std(scores_origin),3)
  model.fit(X_train, y_train)
  model_predic = model.predict(X_test)

  # ['accuracy', 'average_precision', 'f1','recall', 'roc_auc']

  return [accuracy_kfold, std_kfold, model_predic]
