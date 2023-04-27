"""
This module contains functions for visualizing of the data.
"""
import seaborn as sns
import matplotlib.pyplot as plt

def plot_density_bin(data,target,fig_size=(20,20), num_cols=4, num_rows=5):
    """
    Plot the density of a binary classification problem. 
    The density is plotted for each feature in the dataset.
    Args:
        data (pandas.DataFrame): The dataset to plot.
        target (str): The name of the target column.
        fig_size (tuple): The size of the figure.
        num_cols (int): The number of columns in the figure.
        num_rows (int): The number of rows in the figure.
    """
    
    sns.set(style="darkgrid")
    plt.subplots(num_rows,num_cols,figsize=fig_size)

    for idx, col in enumerate(data.columns):
        ax = plt.subplot(num_rows,num_cols,idx+1)
        ax.yaxis.set_ticklabels([])
        fig = sns.kdeplot(data.loc[data[target] == 0][col], shade=True, color="r", linestyle='--')
        fig = sns.kdeplot(data.loc[data[target] == 1][col], shade=True, color="b")
        ax.legend(labels=['Rec','No Rec']) # No Recurrence; # Recurrence

    plt.subplot(num_rows,num_cols,19).set_visible(False)
    plt.subplot(num_rows,num_cols,20).set_visible(False)
    plt.show()
    
def plot_confusion_matrix(data):
    """
    Plot the heatmap of the confusion matrix.
    
    Args:
        data (ndarray of shape (n_classes, n_classes)): The confusion matrix.
    """
    ax = sns.heatmap(data, annot=True,xticklabels=['No Recurrence','Recurrence'],
                 yticklabels=['No Recurrence','Recurrence'],cbar=False, cmap='Blues')
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Actual")