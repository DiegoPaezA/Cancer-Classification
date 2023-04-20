"""
This module contains functions for visualizing of the data.
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

def plot_density_bin(data,fig_size=(20,20), num_cols=4, num_rows=5):
    """
    Plot the density of a binary classification problem. 
    The density is plotted for each feature in the dataset.
    """
    sns.set(style="darkgrid")
    plt.subplots(num_rows,num_cols,figsize=fig_size)

    for idx, col in enumerate(data.columns):
        ax = plt.subplot(num_rows,num_cols,idx+1)
        ax.yaxis.set_ticklabels([])
        fig = sns.kdeplot(data.loc[data.Class == 0][col], shade=True, color="r", linestyle='--')
        fig = sns.kdeplot(data.loc[data.Class == 1][col], shade=True, color="b")
        ax.legend(labels=['No Rec','Rec']) # No Recurrence; # Recurrence

    plt.subplot(5,4,19).set_visible(False)
    plt.subplot(5,4,20).set_visible(False)
    plt.show()