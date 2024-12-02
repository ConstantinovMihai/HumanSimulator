import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, f1_score
import torch


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

def apply_windowing_to_tc(data, window_length=150, overlap=0.5):
    """
    Applies windowing to each column of the input array.

    Args:
        data (numpy.ndarray): 2D array where each column is to be windowed.
        window_length (int): The length of each window.
        overlap (float): The percentage of overlap (0 < overlap < 1).

    Returns:
        numpy.ndarray: 2D array where each column has been windowed.
    """
    # Initialize a list to hold windowed columns
    windowed_columns = []

    # Apply windowing to each column
    for col in range(data.shape[1]):
        # Extract the column
        column_data = data[:, col]

        # Apply create_windows to the column
        windowed_column = create_windows(column_data, window_length, overlap)

        # Add the windowed column to the list
        windowed_columns.append(windowed_column)

    return np.array(windowed_columns)


def create_windows(signal_column, window_length, overlap):
    """
    Creates overlapping windows for a given signal column.

    Args:
        signal_column (numpy.ndarray): A 1D array of the signal.
        window_length (int): The length of each window.
        overlap (float): The percentage of overlap (0 < overlap < 1).

    Returns:
        numpy.ndarray: A 2D array where each row is a windowed segment of the signal.
    """
    step = int(window_length * (1 - overlap))  # Compute the step size based on the overlap
    windows = [signal_column[i:i + window_length] for i in range(0, len(signal_column) - window_length + 1, step)]
    return np.array(windows)

def compute_derivatives(array):
    """
    Computes the derivative of each column in a 2D array.

    Args:
        array (numpy.ndarray): 2D array where each column represents a time series.

    Returns:
        numpy.ndarray: 2D array containing the derivative of each column.
    """
    return np.diff(array, axis=0)


def process_signals_from_file(file_path, window_length, overlap, runs_per_participant=4):
    """
    Loads a CSV file, applies StandardScaler to each column, computes the derivatives,
    groups columns into lists of 4, and applies windowing to both the scaled and derivative signals.

    Args:
        file_path (str): Path to the CSV file.
        window_length (int): The length of each window.
        overlap (float): The percentage of overlap (0 < overlap < 1).

    Returns:
        tuple: Two lists - one for the windowed scaled signals and one for the windowed derivative signals,
               each containing windowed arrays (4 per group).
    """
    # Load the CSV data
    signal = np.loadtxt(file_path, delimiter=',')

    # Initialize lists for storing windowed arrays (4 per group)
    signal_windowed_list = []
    derivative_signal_windowed_list = []

    # Standardize each column and compute derivatives
    scaled_signal = np.empty_like(signal)
    for j in range(signal.shape[1]):
        scaler = StandardScaler()
        scaled_signal[:, j] = scaler.fit_transform(signal[:, j].reshape(-1, 1)).flatten()

    # Compute the derivatives for each column
    derivative_signal = compute_derivatives(signal)
    for j in range(derivative_signal.shape[1]):
        scaler = StandardScaler()
        derivative_signal[:, j] = scaler.fit_transform(derivative_signal[:, j].reshape(-1, 1)).flatten()

    # Group the signal in steps of 4 and apply windowing
    for i in range(0, scaled_signal.shape[1], runs_per_participant):
        # For the scaled signal
        scaled_group = []
        for j in range(i, min(i + runs_per_participant, scaled_signal.shape[1])):
            # Apply windowing to the scaled signal
            windowed_column = create_windows(scaled_signal[:, j], window_length, overlap)
            scaled_group.append(windowed_column)

        signal_windowed_list.append(np.array(scaled_group))

        # For the derivative signal
        derivative_group = []
        for j in range(i, min(i + runs_per_participant, derivative_signal.shape[1])):
            # Apply windowing to the derivative signal
            windowed_derivative_column = create_windows(derivative_signal[:, j], window_length, overlap)
            derivative_group.append(windowed_derivative_column)

        derivative_signal_windowed_list.append(np.array(derivative_group))

    return np.array(signal_windowed_list)[:, :, 1:, :].reshape(-1, window_length), np.array(derivative_signal_windowed_list).reshape(-1, window_length)

def process_signals_from_file(file_path, window_length, overlap, runs_per_participant=4):
    """
    Loads a CSV file, applies StandardScaler to each column, computes the derivatives,
    groups columns into lists of 4, and applies windowing to both the scaled and derivative signals.

    Args:
        file_path (str): Path to the CSV file.
        window_length (int): The length of each window.
        overlap (float): The percentage of overlap (0 < overlap < 1).

    Returns:
        tuple: Two lists - one for the windowed scaled signals and one for the windowed derivative signals,
               each containing windowed arrays (4 per group).
    """
    # Load the CSV data
    signal = np.loadtxt(file_path, delimiter=',')

    # Initialize lists for storing windowed arrays (4 per group)
    signal_windowed_list = []
    derivative_signal_windowed_list = []

    # Standardize each column and compute derivatives
    scaled_signal = np.empty_like(signal)

    for j in range(signal.shape[1]):
        scaler = StandardScaler()
        scaled_signal[:, j] = scaler.fit_transform(signal[:, j].reshape(-1, 1)).flatten()

    # Compute the derivatives for each column
    derivative_signal = compute_derivatives(signal)
    scaled_signal = scaled_signal[1:,:]

    for j in range(derivative_signal.shape[1]):
        scaler = StandardScaler()
        derivative_signal[:, j] = scaler.fit_transform(derivative_signal[:, j].reshape(-1, 1)).flatten()

    # Group the signal in steps of 4 and apply windowing
    for i in range(0, scaled_signal.shape[1], runs_per_participant):
        # For the scaled signal
        scaled_group = []
        for j in range(i, min(i + runs_per_participant, scaled_signal.shape[1])):
            # Apply windowing to the scaled signal
            windowed_column = create_windows(scaled_signal[:, j], window_length, overlap)
            scaled_group.append(windowed_column)

        signal_windowed_list.append(np.array(scaled_group))

        # For the derivative signal
        derivative_group = []
        for j in range(i, min(i + runs_per_participant, derivative_signal.shape[1])):
            # Apply windowing to the derivative signal
            windowed_derivative_column = create_windows(derivative_signal[:, j], window_length, overlap)
            derivative_group.append(windowed_derivative_column)

        derivative_signal_windowed_list.append(np.array(derivative_group))

    return np.array(signal_windowed_list).reshape(-1, window_length), np.array(derivative_signal_windowed_list).reshape(-1, window_length)


def filter_distracted_columns(distracted_times_1d):
    """
    Filters out rows from the input array where the sum across each row is zero.

    Args:
        distracted_times_1d (numpy.ndarray): 2D array where each row represents a time series
                                             and each column represents a time step.

    Returns:
        numpy.ndarray: Filtered array with only rows where the sum is not zero.
    """

    # Find the indices of the rows where the sum across each row is not zero
    distracted_columns = np.array(np.where(np.sum(distracted_times_1d, axis=1) > 0)[0])

    return distracted_columns.astype(int)


def compute_derivatives(array):
    """
    Computes the derivative of each column in a 2D array.

    Args:
        array (numpy.ndarray): 2D array where each column represents a time series.

    Returns:
        numpy.ndarray: 2D array containing the derivative of each column.
    """
    # Compute the derivative for each column (along axis=0)
    derivatives = np.diff(array, axis=0)

    return derivatives


def compute_changepoints(tc_signal):
    cps_array = []
    for col_idx in range(tc_signal.shape[1]):
        # Compute the difference along the column with NaN prepended
        diffs = np.diff(tc_signal[:, col_idx], prepend=np.nan)
        # Find the change points (from 0 to 1)
        cps = np.where((diffs == 1))[0]
        cps_array.append(cps)

    return cps_array


def plot_heatmap(true_labels, predicted_labels):
  plt.figure(figsize=(10, 7))

  cm = confusion_matrix(true_labels, predicted_labels)
  sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
              xticklabels=np.arange(cm.shape[1]),
              yticklabels=np.arange(cm.shape[0]))
  plt.xlabel('Predicted Label')
  plt.ylabel('True Label')
  plt.title('Confusion Matrix')
  plt.show()

def f1_score_metric(preds, y):
    preds = preds.argmax(dim=1)  # Convert predictions to class labels
    preds = preds.cpu().numpy()  # Convert to numpy array
    y = y.cpu().numpy()  # Convert to numpy array
    f1 = f1_score(y, preds)  # Compute the weighted F1 score
    return torch.tensor(f1)  # Return as a Tensor