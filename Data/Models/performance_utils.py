import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

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

def plot_roc_curve(true_labels, predicted_scores):
    """
    Plots the ROC curve and computes AUC.

    Parameters:
    - true_labels: Array of true binary labels (0 or 1).
    - predicted_scores: Array of predicted scores (e.g., probabilities or raw outputs).
    """
    fpr, tpr, thresholds = roc_curve(true_labels, predicted_scores)
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})', color='blue')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')  # Diagonal line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate (Recall)')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    return roc_auc


def compute_precision_recall_f1_from_numbers(matched, missed, false_alarms):
    """
    Computes precision, recall, and F1-score based on matched intervals,
    missed intervals, and false alarms.

    Parameters:
    - matched: int, count of matched intervals
    - missed: int, count of intervals in tc_true without a match
    - false_alarms: int, count of intervals in tc_predicted without a match

    Returns:
    - precision: float, ratio of true positives to all predicted positives
    - recall: float, ratio of true positives to all actual positives
    - f1_score: float, harmonic mean of precision and recall
    """

    # True positives are the matched intervals
    true_positives = matched

    # Predicted positives include matched intervals and false alarms
    predicted_positives = matched + false_alarms

    # Actual positives include matched intervals and missed intervals
    actual_positives = matched + missed

    # Compute precision, recall, and F1-score
    precision = true_positives / predicted_positives if predicted_positives > 0 else 0.0
    recall = true_positives / actual_positives if actual_positives > 0 else 0.0
    f1_score = (
        2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    )

    return precision, recall, f1_score


def compute_total_precision_recall_f1(real_changepoints, predicted_changepoints, tolerance=5):
    total_precision = 0
    total_recall = 0
    total_f1_score = 0
    for real_cp, pred_cp in zip(real_changepoints, predicted_changepoints):
        precision, recall, f1_score = compute_precision_recall_f1(real_cp, pred_cp, tolerance)
        total_precision += precision
        total_recall += recall
        total_f1_score += f1_score

    return total_precision / len(real_changepoints), total_recall / len(real_changepoints), total_f1_score / len(real_changepoints)


def generate_heatmap(real_changepoints, distances_ocsvm, tolerances, thresholds):
    precision_grid = np.zeros((len(tolerances), len(thresholds)))
    recall_grid = np.zeros((len(tolerances), len(thresholds)))
    f1_score_grid = np.zeros((len(tolerances), len(thresholds)))

    # Iterate over tolerances and thresholds
    for i, tolerance in enumerate(tolerances):
        for j, threshold in enumerate(thresholds):
            alarm_ocsvm_indices = []

            # Generate alarms for the current threshold
            for distances_ocsvm_run in distances_ocsvm:
                alarm_indices = trigger_ocsvm_alarm(distances_ocsvm_run, threshold=threshold)
                alarm_ocsvm_indices.append(alarm_indices)

            # Compute precision, recall, F1 score
            total_precision, total_recall, total_f1_score = compute_total_precision_recall_f1(
                real_changepoints, alarm_ocsvm_indices, tolerance=tolerance
            )

            # Store the results in the grid
            precision_grid[i, j] = total_precision
            recall_grid[i, j] = total_recall
            f1_score_grid[i, j] = total_f1_score

    # Plot the heatmaps
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    sns.heatmap(precision_grid, xticklabels=thresholds, yticklabels=tolerances, ax=axs[0], cmap='Blues', annot=True, fmt=".2f")
    axs[0].set_title("Precision Heatmap")
    axs[0].set_xlabel("Threshold")
    axs[0].set_ylabel("Tolerance")

    sns.heatmap(recall_grid, xticklabels=thresholds, yticklabels=tolerances, ax=axs[1], cmap='Greens', annot=True, fmt=".2f")
    axs[1].set_title("Recall Heatmap")
    axs[1].set_xlabel("Threshold")
    axs[1].set_ylabel("Tolerance")

    sns.heatmap(f1_score_grid, xticklabels=thresholds, yticklabels=tolerances, ax=axs[2], cmap='Reds', annot=True, fmt=".2f")
    axs[2].set_title("F1 Score Heatmap")
    axs[2].set_xlabel("Threshold")
    axs[2].set_ylabel("Tolerance")

    plt.tight_layout()
    plt.show()

    return precision_grid, recall_grid, f1_score_grid


def compute_avg_delay_aux(true_changepoints, predicted_changepoints, tolerance=5):
    """
    Matches predicted changepoints to true changepoints within a tolerance window and computes metrics.
    """
    # Ensure inputs are lists
    true_changepoints = list(true_changepoints)
    predicted_changepoints = list(predicted_changepoints)

    true_matched = set()  # Tracks matched true changepoints
    pred_matched = set()  # Tracks matched predicted changepoints
    detection_delays = []  # Stores time differences between matched pairs

    for pred_cp in predicted_changepoints:
        for true_cp in true_changepoints:
            if abs(pred_cp - true_cp) <= tolerance and true_cp not in true_matched:
                true_matched.add(true_cp)
                pred_matched.add(pred_cp)
                detection_delays.append(abs(pred_cp - true_cp))  # Record delay
                break

    avg_detection_delay = sum(detection_delays) / len(detection_delays) if detection_delays else None

    return avg_detection_delay

def compute_avg_delay(true_changepoints, predicted_changepoints, tolerance=5):
    avg_detection_delays = []
    for idx, true_cp in enumerate(true_changepoints):
        avg_detection_delay = compute_avg_delay_aux(true_cp, predicted_changepoints[idx], tolerance=tolerance)
        avg_detection_delays.append(avg_detection_delay)

    return np.array(avg_detection_delays)

