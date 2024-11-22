import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.preprocessing import StandardScaler

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
    """Read data of A SINGLE RUN OF A SINGLE PARTCIPANT from a file.
    """
    # read data from all files of CSV_data/PreviewDistractionExpData_S1/PRDCE/u.csv format
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

def read_data_participant(data_string):
    """
    Read data of all runs of a single participant from a file.
    """

    data = np.loadtxt(data_string, delimiter=',')
    # Use regex to find the substring before the last '/'
    prefix = re.match(r'(.*)/', data_string).group(1)

    # Append 'tc.csv' to the result
    tc_signal = f"{prefix}/tc.csv"

    tc = np.loadtxt(tc_signal, delimiter=',', dtype=int)

    return data, tc


def process_signals(error_file, u_file, x_file):
    """
    Processes error_signal and u_signal from CSV files, groups their columns into lists of 4.

    Args:
        error_file (str): Path to the CSV file for error_signal.
        u_file (str): Path to the CSV file for u_signal.

    Returns:
        tuple: Two lists - one for error_signal and one for u_signal, each containing grouped arrays (4 per group).
    """
    # Load the CSV data
    error_signal = np.loadtxt(error_file, delimiter=',')
    u_signal = np.loadtxt(u_file, delimiter=',')
    x_signal = np.loadtxt(x_file, delimiter=',')

    # Initialize lists for storing grouped arrays (4 per group)
    error_not_distracted_list = []
    u_signal_not_distracted_list = []
    x_not_distracted_list = []

    # Grouping error_signal in steps of 4
    for i in range(0, error_signal.shape[1], 4):
        error_group = [error_signal[:, j] for j in range(i, min(i + 4, error_signal.shape[1]))]
        error_not_distracted_list.append(error_group)

    # Grouping u_signal in steps of 4
    for i in range(0, u_signal.shape[1], 4):
        u_group = [u_signal[:, j] for j in range(i, min(i + 4, u_signal.shape[1]))]
        u_signal_not_distracted_list.append(u_group)

    # Grouping x_signal in steps of 4
    for i in range(0, x_signal.shape[1], 4):
        x_group = [x_signal[:, j] for j in range(i, min(i + 4, x_signal.shape[1]))]
        x_not_distracted_list.append(x_group)

    return error_not_distracted_list, u_signal_not_distracted_list, x_not_distracted_list


def normalize_data_signal(data_signal):
    """
    Normalize the data signal.
    """
    
    # Normalize the data signal
    mean = np.mean(data_signal, axis=2, keepdims=True)
    std = np.std(data_signal, axis=2, keepdims=True)

    # Reshape the normalized data signal back to its original shape
    data_signal_normalized = (data_signal - mean) / std

    return data_signal_normalized

def process_signals(error_file, u_file, x_file):
    """
    Processes error_signal and u_signal from CSV files, groups their columns into lists of 4.

    Args:
        error_file (str): Path to the CSV file for error_signal.
        u_file (str): Path to the CSV file for u_signal.

    Returns:
        tuple: Two lists - one for error_signal and one for u_signal, each containing grouped arrays (4 per group).
    """
    # Load the CSV data
    error_signal = np.loadtxt(error_file, delimiter=',')
    u_signal = np.loadtxt(u_file, delimiter=',')
    x_signal = np.loadtxt(x_file, delimiter=',')

    # Initialize lists for storing grouped arrays (4 per group)
    error_not_distracted_list = []
    u_signal_not_distracted_list = []
    x_signal_not_distracted_list = []
    
    # Grouping error_signal in steps of 4
    for i in range(0, error_signal.shape[1], 4):
        scaler = StandardScaler()
        error_group = [scaler.fit_transform(error_signal[:, j].reshape(-1,1)).flatten() for j in range(i, min(i + 4, error_signal.shape[1]))]
        error_not_distracted_list.append(error_group)

    # Grouping u_signal in steps of 4
    for i in range(0, u_signal.shape[1], 4):
        u_group = [scaler.fit_transform(u_signal[:, j].reshape(-1,1)).flatten() for j in range(i, min(i + 4, u_signal.shape[1]))]
        u_signal_not_distracted_list.append(u_group)

    for i in range(0, x_signal.shape[1], 4):
        x_group = [scaler.fit_transform(x_signal[:, j].reshape(-1,1)).flatten() for j in range(i, min(i + 4, x_signal.shape[1]))]
        x_signal_not_distracted_list.append(x_group)

    return error_not_distracted_list, u_signal_not_distracted_list, x_signal_not_distracted_list


def plot_masked_signal(tc_test, distances):
    """
    Plots the error signal with different colors based on the real label.

    Parameters:
    - tc_test: The true labels array.
    - distances: The distances array.
    - run_idx: The index of the run to plot.
    - column_idx: The column index of tc_test to use for masking (default is 2).
    - run_length: The length of each run (default is 12000).
    """
   
    # Create a mask for tc_test == 1 and tc_test == 0
    mask_1 = tc_test == 1
    mask_0 = tc_test == 0

    # Plotting
    plt.figure(figsize=(20, 10))
    plt.scatter(np.arange(len(distances))[mask_1], distances[mask_1], color='red', label='Distracted')
    plt.scatter(np.arange(len(distances))[mask_0], distances[mask_0], color='blue', label='Not distracted')
    plt.grid()
    # Add labels and title
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    plt.title('Distances plot with Different Colors Based on mdist (real label)')
    plt.legend()
    plt.show()