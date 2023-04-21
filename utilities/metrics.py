"""
This module contains the metrics used to evaluate the performance of the model.
"""
from sklearn.metrics import confusion_matrix,classification_report,cohen_kappa_score, accuracy_score  
from sklearn.metrics import roc_auc_score, f1_score,precision_score,recall_score  

def eval_performance(ytest, ypredict):
  """
    Calculation of performance metrics:
    - Confusion Matrix:
    - Accuracy Score: https://is.gd/YwoQJf
    - Precision Score: https://is.gd/Xz6Nu0
    - F1-Score: https://is.gd/YVGWCs
    - Recall Score: https://is.gd/9PUCgT 
    - Cohen Kappa Score:
    - AUROC Score: https://is.gd/JIAktb
    Arguments:
        ytest: ndarray - list
        ypredict: ndarray - list
    Returns:
        List with result of performance metrics: matrixconfu,accuracyscore,
        precisionscore,f1score,recallscore,cohenkappa,auroc_score
    """

  matrixconfu  = confusion_matrix(ytest, ypredict)
  matrixreport = classification_report(ytest, ypredict)
  
  accuracyscore = round(accuracy_score(ytest, ypredict),3)
  f1score = round(f1_score(ytest, ypredict, average='weighted'),3)
  precisionscore = round(precision_score(ytest, ypredict, average='weighted'),3)
  recallscore = round(recall_score(ytest, ypredict, average='weighted'),3)
  

  cohenkappa= round(cohen_kappa_score(ytest, ypredict),3)
  auroc_score = round(roc_auc_score(ytest, ypredict),3)


  result = {
    "Accuracy Score": accuracyscore,
    "Precision Score": precisionscore,
    "F1 Score": f1score,
    "Recall Score": recallscore,
    "Cohen Kappa Score": cohenkappa,
    "AUROC Score": auroc_score,
  }
  return result