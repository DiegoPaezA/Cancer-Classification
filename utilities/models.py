"""
This file contains the functions to evaluate the models
"""
import numpy as np
import pandas as pd
import random

from numpy import mean
from numpy import std
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from utilities.preprocessing import split_data, get_xy, oversample_data
from utilities.metrics import eval_performance

from collections import Counter


#np.random.seed(42)

firts_time=True

def convert_to_df(results:dict):
  """
  This function gets the results of the models and returns a dataframe.
  
  Args:
    results (list): The list of dictionaris with the results of the models that 
                    includes the kfold accuracy, std, and predictions of trained models.
  Returns:
    pandas.DataFrame: The dataframe of the results.
  """
  results_df = pd.DataFrame(results)
  models_names = ["LR", "NB", "SVM", "KNN", "XGB"]
  results_df.insert (0,"Models", models_names, True)
  return results_df

def evaluate_model(model, data:pd.DataFrame, target:str,Ename:str, test_size=0.2):
  """
  This functions trains a model with kfold cross validation and also `model.fit`.
  
  Args:
    model (sklearn model): The model to train.
    data (pandas.DataFrame): The dataset to train the model.
    target (str): The name of the target column.
    Ename (str): The name of the experiment (E1, E2, E3, E4).
    test_size (float): The size of the test set.
  """
  global firts_time
  if Ename == "E4":
    data = oversample_data(data, target)
    if (firts_time):
      print("Oversampling data...")
      print("Class distribution in the oversampled data:")
      firts_time = False
  
  X, y = get_xy(data, target)
  X_train, X_test, y_train, y_test = split_data(data, target, test_size)
  
  cv = KFold(n_splits=10, random_state=42, shuffle=True)
  scores_origin = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    
  accuracy_kfold = round(mean(scores_origin),5)
  std_kfold = round(std(scores_origin),5)
  
  model.fit(X_train, y_train)
  model_predic = model.predict(X_test)
  
  if len(np.unique(y_test))>2:
    model_prob = model.predict_proba(X_test)
  else:
    model_prob = []  

  metrics = eval_performance(y_test, model_predic,model_prob)
  
  cv_results = {
    "Accuracy (kfold)": accuracy_kfold,
    "Std (kfold)": std_kfold
  }

  result = {**cv_results, **metrics}
  
  return result

def run_experiment(data:pd.DataFrame,target:str,Ename:str,params:dict,num_exp:int=1,test_size=0.2)-> pd.DataFrame:
  """
  This function runs the experiment with the models.
  
  Args:
    data (pandas.DataFrame): The dataset to run the experiment on.
    target (str): The name of the target column.
    Ename (str): The name of the experiment (E1, E2, E3, E4)
    num_exp (int): The number of experiments to run.
    test_size (float): The size of the test set.
  Returns:
    
    results (pd.Dataframe): The list of the results of the models that 
                    includes the kfold accuracy, std, and predictions of trained models.
    
  """
  #Tuple[Dict[str, List[float]], torch.Tensor]
  models = []
  models.append(('LR', LogisticRegression(**params["log_reg"])))
  models.append(('NB', GaussianNB(**params["naive_bayes"])))
  models.append(('SVM', SVC(**params["svm"])))
  models.append(('KNN', KNeighborsClassifier(**params["knn"])))
  models.append(('XGB', XGBClassifier(**params["xgb"])))

  results = []
  all_exp_results = []
  for num in range(num_exp):
    print(f"Running experiment {num+1}...")
    for name, model in models:
      #print(f"Running {name} model...")
      results.append(evaluate_model(model,data,target,Ename, test_size))
    all_exp_results.append(results)
      
  #results_df = convert_to_df(results)
  return all_exp_results

