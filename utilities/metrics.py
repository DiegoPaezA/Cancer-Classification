"""
This module contains the metrics used to evaluate the performance of the model.
"""

from sklearn.metrics import confusion_matrix,classification_report,cohen_kappa_score, accuracy_score  
from sklearn.metrics import roc_auc_score, f1_score,precision_score,recall_score  
import numpy as np
def eval_performance(ytest: np.array, ypredict: np.array):
  """
    Calculation of performance metrics:
    - Confusion Matrix:
    - Accuracy Score: https://is.gd/YwoQJf
    - Precision Score: https://is.gd/Xz6Nu0
    - F1-Score: https://is.gd/YVGWCs
    - Recall Score: https://is.gd/9PUCgT 
    Arguments:
        ytest: ndarray - list
        ypredict: ndarray - list
    Returns:
        List with result of performance metrics: matrixconfu,accuracyscore,
        precisionscore,f1score,recallscore,cohenkappa,auroc_score
    """

  average_state = None if len(np.unique(ytest)) > 2 else "binary"
  
  #matrixconfu  = confusion_matrix(ytest, ypredict)
  #matrixreport = classification_report(ytest, ypredict)
  
  accuracyscore = accuracy_score(ytest, ypredict)
  f1score = f1_score(ytest, ypredict, average=average_state)
  precisionscore = precision_score(ytest, ypredict, average=average_state)
  recallscore = recall_score(ytest, ypredict, average=average_state)
  
  # if multi-class classification, return only the last class metrics
  if len(np.unique(ytest)) > 2:
    f1score = f1score[-1]
    precisionscore = precisionscore[-1]
    recallscore = recallscore[-1]
  

  result = {
    "Accuracy Score": accuracyscore,
    "Precision Score": precisionscore,
    "Recall Score": recallscore,
    "F1 Score": f1score
  }
  return result