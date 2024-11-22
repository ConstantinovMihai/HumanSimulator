import pandas as pd
from torch.utils.data import Dataset
import torch
import seaborn as sns
from pyts.classification import TimeSeriesForest
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import numpy as np
from tsai.all import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_curve, auc, confusion_matrix, roc_auc_score, f1_score, accuracy_score, classification_report
import sklearn.metrics as skm
from sklearn.preprocessing import StandardScaler
from utils import *


def plot_confusion_matrix(conf_matrix):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Predicted: No', 'Predicted: Yes'], 
                yticklabels=['Actual: No', 'Actual: Yes'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()


def read_data(data_signal_string, idx=0):
    """Read data from file.
    """
    # read data from all files of CSV_data/PreviewDistractionExpData_S1/PRDCE/u.csv
    # afterwards, read changepoint detections from tc.csv, where chancging from 0 to 1 or 1 to 0 indicates a change point
    # return data, cps

    data = np.loadtxt(data_signal_string, delimiter=',')[:,idx]

    # Use regex to find the substring before the last '/'
    prefix = re.match(r'(.*)/', data_signal_string).group(1)

    # Append 'tc.csv' to the result
    tc_signal = f"{prefix}/tc.csv"

    tc = np.loadtxt(tc_signal, delimiter=',', dtype=int)[:,idx]

    cps = np.where(np.diff(tc, prepend=np.nan))[0]

    return data, cps, tc

error_signal = np.loadtxt(r'/home/mihai/Thesis/Data/Clean_CSV_data/updated_data/PRDPE/e.csv', delimiter=',')
u_signal =np.loadtxt(r'/home/mihai/Thesis/Data/Clean_CSV_data/updated_data/PRDPE/u.csv', delimiter=',')
tc = np.loadtxt(r'/home/mihai/Thesis/Data/Clean_CSV_data/updated_data/PRDPE/mdist.csv', delimiter=',')

error_signal = error_signal[:, 0]
u_signal = u_signal[:, 0]
tc = tc[:, 0]