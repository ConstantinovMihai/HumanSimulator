import numpy as np
import matplotlib.pyplot as plt
from tsai.all import *
from utils import *
from scipy.stats import chi2
from compute_mahalanobis_distances import *


def detect_distractions_mahalanobis_thresholding(distances_distracted, mean_shape=2, chi_threshold=0.80, n_consecutive_threshold=10):   
    threshold_distracted = chi2.ppf(chi_threshold, mean_shape) 
    threshold_reset = chi2.ppf(chi_threshold-0.05, mean_shape)       
    print(f"Chi-squared distraction threshold (0.85): {threshold_distracted}")
    print(f"Chi-squared reset threshold (0.75): {threshold_reset}")
    
    # Detect distractions by checking whether the Mahalanobis distance exceeds the 80% threshold
    detected_distracted = np.array(distances_distracted) > threshold_distracted

    # Initialize variables to keep track of consecutive exceedances and distraction reset mechanism
    consecutive_count = 0
    distractions_detected = []  # This will store the timestamps where distractions were detected
    distraction_active = False  # Whether distraction detection is currently paused

    # Loop through the detected_distracted array to find consecutive distractions
    for i, (is_distracted, distance_value) in enumerate(zip(detected_distracted, distances_distracted)):
        if distraction_active:
            # If we're in distraction mode, wait until the distance falls below the reset threshold
            if distance_value < threshold_reset:
                consecutive_count = 0  # Reset the counter once we detect a "non-distracted" sample below the reset threshold
                distraction_active = False  # Resume checking distractions after returning below reset threshold
            continue  # Skip further checks until distraction ends
        
        if is_distracted:
            consecutive_count += 1  # Increment the counter if a distraction is detected
        else:
            consecutive_count = 0  # Reset the counter if a distraction is not detected
        
        # If the number of consecutive distractions reaches the threshold, log the distraction
        if consecutive_count == n_consecutive_threshold:
            distractions_detected.append(i)  # Store the time (or index) of the threshold-reaching distraction
            consecutive_count = 0  # Reset the counter after detecting a distraction
            distraction_active = True  # Pause detection until behavior falls below reset threshold
    
    return distractions_detected, detected_distracted


def detect_distractions_mahalanobis_momentum(distances_distracted, mean_shape=2, chi_threshold=0.80, 
                                                          n_consecutive_threshold=10, velocity_weight=0.5):
    # Calculate the chi-squared thresholds
    threshold_distracted = chi2.ppf(0.85, mean_shape) 
    threshold_reset = chi2.ppf(0.75, mean_shape)
    threshold_velocity = 1.5  # Threshold for the velocity of the Mahalanobis distance       
    print(f"Chi-squared distraction threshold (0.85): {threshold_distracted}")
    print(f"Chi-squared reset threshold (0.75): {threshold_reset}")
    
    velocities = np.diff(distances_distracted, prepend=distances_distracted[0])

    # Detect distractions by checking whether the Mahalanobis distance exceeds the 80% threshold
    detected_distracted = np.array(distances_distracted) > threshold_distracted
    velocities_excesive = np.array(velocities) > threshold_velocity
    
    print(f'{detected_distracted.shape=}; {velocities_excesive.shape=}')

    # Initialize variables to keep track of consecutive exceedances and distraction reset mechanism
    consecutive_count = 0
    distractions_detected = []  # This will store the timestamps where distractions were detected
    distraction_active = False  # Whether distraction detection is currently paused

    # Loop through the momentum values to find distractions
    for i, (is_distracted, distance_value, exceed_velocity) in enumerate(zip(detected_distracted, distances_distracted, velocities_excesive)):
        if distraction_active:
            # If we're in distraction mode, wait until the distance falls below the reset threshold
            if distance_value < threshold_reset:
                consecutive_count = 0  # Reset the counter once we detect a "non-distracted" sample below the reset threshold
                distraction_active = False  # Resume checking distractions after returning below reset threshold
            continue  # Skip further checks until distraction ends
        
        # Check if the momentum exceeds the distraction threshold
        
        if is_distracted and exceed_velocity:
            consecutive_count += 1  # Increment the counter if a distraction is detected
        else:
            consecutive_count = 0  # Reset the counter if a distraction is not detected
        
        # If the number of consecutive distractions reaches the threshold, log the distraction
        if consecutive_count == n_consecutive_threshold:
            distractions_detected.append(i)  # Store the time (or index) of the threshold-reaching distraction
            consecutive_count = 0  # Reset the counter after detecting a distraction
            distraction_active = True  # Pause detection until behavior falls below reset threshold
    
    return distractions_detected, detected_distracted, velocities


def detect_distractions_mahalanobis_velocity(distances_distracted, mean_shape=2, chi_threshold=0.80, 
                                                          n_consecutive_threshold=10, velocity_weight=0.5):
    # Calculate the chi-squared thresholds
    threshold_distracted = chi2.ppf(0.95, mean_shape) 
    threshold_reset = chi2.ppf(0.65, mean_shape)       
    print(f"Chi-squared distraction threshold (0.85): {threshold_distracted}")
    print(f"Chi-squared reset threshold (0.75): {threshold_reset}")
    
    # Detect distractions by checking whether the Mahalanobis distance exceeds the 80% threshold
    detected_distracted = np.array(distances_distracted) > threshold_distracted
    
    # Compute the velocity (rate of change) of the Mahalanobis distance
    velocities = np.diff(distances_distracted, prepend=distances_distracted[0]) / 1  # Assuming uniform time steps (Δt = 1)
    threshold_velocity = 1.5  # Threshold for the velocity of the Mahalanobis distance

    # Initialize variables to keep track of consecutive exceedances and distraction reset mechanism
    consecutive_count = 0
    distractions_detected = []  # This will store the timestamps where distractions were detected
    distraction_active = False  # Whether distraction detection is currently paused

    # Loop through the detected_distracted array to find consecutive distractions
    for i, (is_distracted, distance_value) in enumerate(zip(detected_distracted, distances_distracted)):
        if distraction_active:
            # If we're in distraction mode, wait until the distance falls below the reset threshold
            if distance_value < threshold_reset:
                consecutive_count = 0  # Reset the counter once we detect a "non-distracted" sample below the reset threshold
                distraction_active = False  # Resume checking distractions after returning below reset threshold
            continue  # Skip further checks until distraction ends
        
        if is_distracted and velocities[i] > threshold_velocity:
            consecutive_count += 1  # Increment the counter if a distraction is detected
        else:
            consecutive_count = 0  # Reset the counter if a distraction is not detected
        
        # If the number of consecutive distractions reaches the threshold, log the distraction
        if consecutive_count == n_consecutive_threshold:
            distractions_detected.append(i)  # Store the time (or index) of the threshold-reaching distraction
            consecutive_count = 0  # Reset the counter after detecting a distraction
            distraction_active = True  # Pause detection until behavior falls below reset threshold
    
    return distractions_detected, detected_distracted


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


def detect_distractions_mahalanobis_aux(distances_from_prior, threshold=0.8):
    # Initialize lists to store outputs
    distraction_start_timesample_list = []
    detected_distracted_samples_list = []

    # Iterate over each column
    for column_index in range(distances_from_prior.shape[0]):
        print(f"Processing column {column_index}...")
        # Use the corresponding precomputed distances for this column
        distances_per_run = distances_from_prior[column_index]

        # Call the detect_distractions_mahalanobis function, assuming it needs to check the distraction using provided distances
        distraction_start_timesample, detected_distracted_samples = detect_distractions_mahalanobis_cusum(distances_per_run, threshold)

        # Append results to corresponding lists
        distraction_start_timesample_list.append(distraction_start_timesample)
        detected_distracted_samples_list.append(detected_distracted_samples)

    # Return the lists (no need to return distances since they're already available)
    return distraction_start_timesample_list, detected_distracted_samples_list



def plot_distractions(tc_test_column, distraction_start_timesample, detected_distracted_samples, run_idx=0):
    """
    Plots tc_test_column and detected_distracted side by side, marking detected distractions.

    Parameters:
    - tc_test_column: The time series data to plot on the left.
    - distractions_detected: A list of indices where distractions were detected.
    - detected_distracted: An array indicating whether a distraction was detected (0 or 1).
    """
    plt.figure(figsize=(10, 5))

    # Plot tc_test_column
    plt.subplot(1, 2, 1)
    for i in range(len(distraction_start_timesample)):
        plt.axvline(x=distraction_start_timesample[i], color='red', linestyle='--')
    plt.grid()
    plt.plot(tc_test_column, label="tc_test_column")
    plt.title(f"Run {run_idx}. Predicted distraction start times (red vertical lines) vs the real distracted times")
    plt.xlabel("Time index", fontsize=14)
    plt.ylabel("Distraction (0 or 1)", fontsize=14)

    # Plot detected_distracted
    plt.subplot(1, 2, 2)
    plt.grid()
    plt.plot(detected_distracted_samples, label="detected_distracted", color='orange')
    plt.title(f"Run {run_idx}. Evidence accumulated")
    plt.xlabel("Sample index")
    plt.ylabel("Evidence")

    plt.tight_layout()
    plt.show()


def analyze_performance(distraction_start_timesample_list, cps_list, tc_test, sampling_rate=100, detection_window=1):
    """
    Analyzes changepoint detection performance across multiple runs and calculates precision, recall, and F1-score for each column.

    Parameters:
    - distraction_start_timesample_list: List of predicted changepoints for each run
    - cps_list: List of actual changepoints for each run
    - tc_test: List of ground truth signals indicating distractions (1 if distraction, 0 otherwise)
    - sampling_rate: Sampling rate in samples per second
    - detection_window: Detection window in seconds
    
    Returns:
    - results: A list of dictionaries containing precision, recall, and F1-score for each column
    """
    detection_window_samples = detection_window * sampling_rate
    results = []  # Store results for each column
    
    num_columns = len(distraction_start_timesample_list[0])  # Assuming each list has the same length
    
    for col_idx in range(num_columns):
        true_positives = 0
        pred_positives = 0
        total_true = 0
        total_pred = 0

        # Iterate through each row (run) to compute the values for each column
        for run_idx in range(len(distraction_start_timesample_list)):
            predicted_changepoints = distraction_start_timesample_list[run_idx][col_idx]
            real_changepoints = cps_list[run_idx][col_idx]
            tc_run = tc_test[run_idx]
            
            # Count the true positives, predicted positives, total true, and total predicted
            for pred_cp in predicted_changepoints:
                if any(abs(pred_cp - real_cp) <= detection_window_samples for real_cp in real_changepoints):
                    true_positives += 1
                else:
                    pred_positives += 1

            # Count total true changepoints
            total_true += len(real_changepoints)

            # Count total predicted changepoints
            total_pred += len(predicted_changepoints)

        # Calculate precision, recall, and F1 score for the column
        precision = true_positives / total_pred if total_pred > 0 else 0
        recall = true_positives / total_true if total_true > 0 else 0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Store results for the column
        results.append({
            "column": col_idx + 1,  # Column index (1-based)
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score
        })

        # Print the results for the current column
        print(f"Column {col_idx + 1}: Precision = {precision:.2f}, Recall = {recall:.2f}, F1 Score = {f1_score:.2f}")

    return results


def match_changepoints(true_changepoints, predicted_changepoints, tolerance=5):
    """
    Matches each predicted changepoint to a true changepoint within a tolerance window
    and returns counts of matches for precision and recall calculation.
    """
    # Ensure true_changepoints and predicted_changepoints are lists of individual points
    true_changepoints = list(true_changepoints)
    predicted_changepoints = list(predicted_changepoints)

    true_matched = set()  # Keeps track of matched true changepoints
    pred_matched = set()  # Keeps track of matched predicted changepoints

    # Try to match each predicted changepoint to a true changepoint
    for pred_cp in predicted_changepoints:
        for true_cp in true_changepoints:
            # Check if the difference between pred_cp and true_cp is within tolerance
            if abs(pred_cp - true_cp) <= tolerance and true_cp not in true_matched:
                true_matched.add(true_cp)
                pred_matched.add(pred_cp)
                break  # Move to next predicted changepoint after finding a match
    
    # True positives, predicted positives, and total true changepoints
    true_positives = len(true_matched)
    pred_positives = len(pred_matched)
    total_true = len(true_changepoints)
    total_pred = len(predicted_changepoints)
    
    return true_positives, pred_positives, total_true, total_pred



def compute_precision_recall_f1(true_changepoints, predicted_changepoints, tolerance=5):
    """
    Computes precision, recall, and F1-score for changepoint detection based on
    matched changepoints within a given tolerance window.
    """
    true_positives, _, total_true, total_pred = match_changepoints(
        true_changepoints, predicted_changepoints, tolerance
    )
    
    # Precision: fraction of predicted changepoints that are correct
    precision = true_positives / total_pred if total_pred > 0 else 0
    
    # Recall: fraction of true changepoints that are correctly predicted
    recall = true_positives / total_true if total_true > 0 else 0
    
    # F1 Score: harmonic mean of precision and recall
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1_score

def compute_performance_cusum(distances, real_changepoints, threshold=0.8):
    
    # Call the detect_distractions_mahalanobis_aux function to detect distractions
    detected_distractions, _= detect_distractions_mahalanobis_aux(distances, threshold)

    total_true_positives, total_true, total_pred = 0, 0, 0

    # save the distraction start timesample list 
    for idx, detected_distractions_per_run in enumerate(detected_distractions):
        true_positives, _, local_true, local_pred = match_changepoints(real_changepoints[idx], detected_distractions_per_run, tolerance=200)       
        total_true_positives += true_positives
        total_true += local_true
        total_pred += local_pred

        precision, recall, f1_score = compute_precision_recall_f1(real_changepoints[idx], detected_distractions_per_run, tolerance=200)


        print(f"Run {idx+1}: Precision = {precision:.2f}, Recall = {recall:.2f}, F1 Score = {f1_score:.2f}")
    print(f"{total_true_positives=}, {total_true=}, {total_pred=}")
    
    total_precision = total_true_positives/total_pred
    total_recall = total_true_positives/total_true
    total_f1_score = 2 * total_true_positives / (total_true + total_pred)

    print(f"Total Precision = {total_precision:.2f}, Total Recall = {total_recall:.2f}, Total F1 Score = {total_f1_score:.2f}")
    
    return total_precision, total_recall, total_f1_score


if __name__ == '__main__':
    plot = True
    analyse_performance = False

    test_tc_file = r'/home/mihai/Thesis/Data/Clean_CSV_data/updated_data/PRDPE/mdist.csv'

    tc_test = load_data(test_tc_file)
    cps_test = compute_changepoints(tc_test)
    processed_tc_test = process_data_array(tc_test)

    distances = load_data("distances_per_run.csv")
    # distances = load_data("distances_per_run.csv")
    # for distancess in distances:
    #     # plot the distances
    #     plt.plot(distancess)
    #     plt.grid()
    #     plt.show()

    reshaped_tc_test = processed_tc_test.reshape(-1, processed_tc_test.shape[-1])

    # Call the detect_distractions_mahalanobis_aux function to detect distractions
    distraction_start_timesample_list, detected_distracted_samples_list= detect_distractions_mahalanobis_aux(distances, threshold=0.75)
    
    # save the distraction_start_timesample_list
    np.save('/home/mihai/Thesis/Data/Models/distraction_start_timesample_list.npy', np.array(distraction_start_timesample_list, dtype=object), allow_pickle=True)
    real_changepoints = compute_changepoints(reshaped_tc_test.transpose())
    
    thresholds = np.linspace(0.6, 0.95, 20)
    precisions, recalls, f1_scores = [], [], []
    for threshold in thresholds:
        print(f"Threshold: {threshold}")
        precision, recall, f1_score = compute_performance_cusum(distances, real_changepoints, threshold)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)

    # plot the results
    print(f"{len(thresholds)=}; {len(precisions)=}, {len(recalls)=}, {len(f1_scores)=}")

    plt.figure(figsize=(10, 5))
    plt.plot(thresholds, precisions, label='Precision')
    plt.plot(thresholds, recalls, label='Recall')
    plt.plot(thresholds, f1_scores, label='F1 Score')
    plt.xlabel('Threshold')
    plt.ylabel('Performance')
    plt.legend()
    plt.grid()
    plt.title('Performance of CUSUM with varying thresholds (with velocities) ')
    plt.show()
    
    
    if analyse_performance:
        # Analyze the performance of the detector
        good_detections, false_alarms, missed_detections = analyze_performance(distraction_start_timesample_list, cps_test, processed_tc_test.reshape(40,-1))
        print(f"Performance measures:")
        print(f"Good detections: {good_detections}")
        print(f"False alarms: {false_alarms}")
        print(f"Missed detections: {missed_detections}")


    if plot:
        for run_idx in range(reshaped_tc_test.shape[0]):
            # Plot distractions for the first run
            plot_distractions(reshaped_tc_test[run_idx], distraction_start_timesample_list[run_idx], detected_distracted_samples_list[run_idx], run_idx=run_idx)

    print("Done!")
