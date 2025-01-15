import numpy as np
from performance_utils import *
from scipy.stats import chi2

def generate_heatmap(precision_grid, recall_grid, f1_score_grid, tolerances, thresholds, title1='Precision Heatmap', title2='Recall Heatmap', title3='F1 Score Heatmap'):
    # Format thresholds and tolerances to two decimal places
    thresholds_formatted = [f"{x:.2f}" for x in thresholds]
    tolerances_formatted = [f"{x:.2f}" for x in tolerances]

    # Plot the heatmaps
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    sns.heatmap(
        precision_grid, 
        xticklabels=thresholds_formatted, 
        yticklabels=tolerances_formatted, 
        ax=axs[0], 
        cmap='Blues', 
        annot=True, 
        fmt=".2f"
    )
    axs[0].set_title(title1)
    axs[0].set_xlabel("Threshold (distance from decision boundary)")
    axs[0].set_ylabel("Tolerance (maximum time between real and detected changepoints)")

    sns.heatmap(
        recall_grid, 
        xticklabels=thresholds_formatted, 
        yticklabels=tolerances_formatted, 
        ax=axs[1], 
        cmap='Greens', 
        annot=True, 
        fmt=".2f"
    )
    axs[1].set_title(title2)
    axs[1].set_xlabel("Threshold (distance from decision boundary)")
    axs[1].set_ylabel("Tolerance (maximum time between real and detected changepoints)")

    sns.heatmap(
        f1_score_grid, 
        xticklabels=thresholds_formatted, 
        yticklabels=tolerances_formatted, 
        ax=axs[2], 
        cmap='Reds', 
        annot=True, 
        fmt=".2f"
    )
    axs[2].set_title(title3)
    axs[2].set_xlabel("Threshold (distance from decision boundary)")
    axs[2].set_ylabel("Tolerance (maximum time between real and detected changepoints)")

    plt.tight_layout()
    plt.show()

def trigger_ocsvm_alarm(distances, threshold=-500, count_limit=10):
    """
    Detect when 10 consecutive samples have distances below a specified threshold and trigger an alarm.
    
    Parameters:
        distances (numpy.ndarray): Array of distances from the decision boundary (signed distances).
    
    Returns:
        alarm_indices (list): List of starting indices for alarms where the condition is met.
    """

    # Initialize variables
    alarm_indices = []
    consecutive_count = 0
    
    # Iterate through the distances
    for i in range(len(distances)):
        if distances[i] < threshold:
            consecutive_count += 1
            # If 10 consecutive samples meet the condition, trigger the alarm
            if consecutive_count == count_limit:
                # Record the starting index of the alarm
                alarm_indices.append(i - 9)  # Subtract 9 to get the starting index of the 10-sample sequence
        else:
            # Reset the consecutive count if condition is broken
            consecutive_count = 0
    
    return alarm_indices

def trigger_ocsvm_alarms_dataset(ocsvm_scores, threshold=-300, count_limit=10):
    ocsvm_alarms = []

    for run_idx, ocsvm_score_run in enumerate(ocsvm_scores): 
        ocsvm_alarms_run = trigger_ocsvm_alarm(ocsvm_score_run, threshold, count_limit=count_limit)
        ocsvm_alarms.append(ocsvm_alarms_run)

    return ocsvm_alarms


def exponential_window_accumulation(
        distances,
        window_size=20,
        decay_factor=0.95):
    """
    Accumulate values over time using exponential decay, within a sliding window.

    Parameters:
        distances (array-like): Values to accumulate over time.
        window_size (int): Size of the sliding window.
        decay_factor (float): Exponential decay factor (0 < decay_factor < 1).

    Returns:
        accumulated_values (list): Accumulated values over time.
    """
    # Initialize the accumulated values list
    accumulated_values = []
    num_samples = len(distances)

    # Iterate through the data using a sliding window
    for i in range(num_samples):
        # Define the current window
        window_start = max(0, i - window_size + 1)  # Ensure the window does not go negative
        window_end = i + 1  # Inclusive of the current point
        window = distances[window_start:window_end]
        
        # Compute accumulated value for the current window with exponential decay
        accumulated_value = 0
        for j, value in enumerate(reversed(window)):
            accumulated_value += decay_factor**j * value
        
        # Append the accumulated value for the current point
        accumulated_values.append(accumulated_value)

    return np.array(accumulated_values)


def md_alarms_dataset_threshold(mahalanobis_scores, chi_squared_limit=0.8, mean_not_distracted_shape=2, evidence_threshold=15):
    md_alarms = []
    for idx in range(mahalanobis_scores.shape[0]):
        alarms, _ = detect_distractions_mahalanobis_threshold(mahalanobis_scores[idx], chi_squared_limit=chi_squared_limit, mean_not_distracted_shape=mean_not_distracted_shape, evidence_threshold=evidence_threshold)
        md_alarms.append(alarms)
    return md_alarms


def md_alarms_dataset_cumsum(mahalanobis_scores, chi_squared_limit=0.8, mean_not_distracted_shape=2, evidence_threshold=1.5):
    md_alarms = []
    for idx in range(mahalanobis_scores.shape[0]):
        alarms, _ = detect_distractions_mahalanobis_cusum(mahalanobis_scores[idx], chi_squared_limit=chi_squared_limit, mean_not_distracted_shape=mean_not_distracted_shape, evidence_threshold=evidence_threshold)
        md_alarms.append(alarms)
    return md_alarms

def detect_distractions_mahalanobis_threshold(distances_distracted, chi_squared_limit=0.8, mean_not_distracted_shape=2, evidence_threshold=15):
    # Calculate thresholds
    threshold_distracted = chi2.ppf(chi_squared_limit, mean_not_distracted_shape)
    threshold_reset = chi2.ppf(chi_squared_limit - 0.1, mean_not_distracted_shape)

    # Initialize variables
    counter = 0  # Counter for consecutive values above the threshold
    distractions_detected = []  # Indices where distractions are detected
    distraction_active = False  # Whether the alarm is currently active
    counter_list = []  # For visualization of counter over time

    # Loop through Mahalanobis distances
    for i, distance_value in enumerate(distances_distracted):
        if distraction_active:
            # Reset logic: If below the reset threshold, deactivate the alarm and reset the counter
            if distance_value < threshold_reset:
                counter = 0
                distraction_active = False
            counter_list.append(counter)
            continue

        # Update counter based on the distance value
        if distance_value > threshold_distracted:
            counter += 1
        else:
            counter = 0  # Reset counter if the value is below the distracted threshold

        # Trigger distraction detection if the counter exceeds the evidence threshold
        if counter >= evidence_threshold:
            distractions_detected.append(i)  # Log the index where distraction is detected
            distraction_active = True  # Activate distraction mode
            counter = 0  # Reset counter after detection

        counter_list.append(counter)

    return distractions_detected, counter_list


def detect_distractions_mahalanobis_cusum(distances_distracted, chi_squared_limit=0.8, mean_not_distracted_shape=2, evidence_threshold=1.5):
    threshold_distracted = chi2.ppf(chi_squared_limit, mean_not_distracted_shape)  # Threshold for distraction
    threshold_reset = chi2.ppf(chi_squared_limit-0.1, mean_not_distracted_shape)      # Threshold to resume search
    
    # Initialize variables for CUSUM
    S_pos = 0  # Positive cumulative sum (for upward shifts, i.e., distraction)
    distractions_detected = []  # List to store timestamps of detected distractions
    distraction_active = False  # Whether distraction detection is currently paused
    S_pos_list = [S_pos]  # List to store the cumulative sum values for visualization
    # Loop through Mahalanobis distances and apply CUSUM logic
    for i, distance_value in enumerate(distances_distracted):
        if distraction_active:
            # If distraction mode is active, wait until the distance falls below the reset threshold
            if distance_value < threshold_reset:
                S_pos = 0  # Reset the cumulative sum after normal behavior
                distraction_active = False  # Resume checking distractions
            continue  # Skip further checks while in distraction mode

        # If the Mahalanobis distance exceeds the distraction threshold, accumulate the difference in S_pos
        if distance_value > threshold_distracted:
            S_pos += distance_value - threshold_distracted
        else:
            S_pos = 0  # Reset S_pos if the distance does not exceed the threshold

        # If S_pos exceeds the threshold h, activate distraction mode
        if S_pos > evidence_threshold:
            distractions_detected.append(i)  # Log the index where distraction is detected
            distraction_active = True  # Activate distraction mode
            S_pos = 0  # Reset cumulative sum to prepare for next detection
        S_pos_list.append(S_pos)
    return distractions_detected, S_pos_list