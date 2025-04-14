# File: evaluate_tracker.py
# Phase 4: Evaluation of the Tracking Performance

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

try:
    from tracker import track_sequence, calculate_iou
    from motion_detector import (process_fg_mask,
                                 DEFAULT_MIN_CONTOUR_AREA,
                                 DEFAULT_MAX_CONTOUR_AREA,
                                 DEFAULT_MORPH_KERNEL_SIZE,
                                 DEFAULT_MORPH_OPEN_ITERATIONS,
                                 DEFAULT_MORPH_CLOSE_ITERATIONS)
    from data_loader import DATASET_BASE_PATH
except ImportError as e:
    print("="*60)
    print(f"ERROR: Could not import necessary functions: {e}")
    print("Ensure data_loader.py, motion_detector.py, and tracker.py are present")
    print("and contain the required functions (e.g., process_fg_mask, track_sequence).")
    print("="*60)
    exit()

# --- Evaluation Configuration ---
IOU_SUCCESS_THRESHOLD = 0.5 # Standard threshold for considering tracking successful in a frame

# --- Helper Function (Center Location Error) ---
def calculate_cle(boxA, boxB):
    """ Calculates Center Location Error (CLE) between two boxes [x, y, w, h]. """
    if boxA is None or boxB is None: return float('inf') # Handle missing boxes

    # Center of box A
    centerAx = boxA[0] + boxA[2] / 2.0
    centerAy = boxA[1] + boxA[3] / 2.0
    # Center of box B
    centerBx = boxB[0] + boxB[2] / 2.0
    centerBy = boxB[1] + boxB[3] / 2.0

    # Euclidean distance
    error = np.sqrt((centerAx - centerBx)**2 + (centerAy - centerBy)**2)
    return error

# --- Main Evaluation Function ---
def evaluate_sequence(sequence_name, tracked_bboxes, ground_truth_bboxes):
    """
    Evaluates tracking performance for a single sequence.

    Args:
        sequence_name (str): Name of the sequence.
        tracked_bboxes (list): List of tracker output boxes [(x,y,w,h), ...]
        ground_truth_bboxes (list): List of ground truth boxes [(x,y,w,h), ...]

    Returns:
        dict: Dictionary containing calculated metrics for the sequence.
              Returns None if input lists are invalid or mismatched.
    """
    print(f"Evaluating sequence: {sequence_name}")
    if not tracked_bboxes or not ground_truth_bboxes or len(tracked_bboxes) != len(ground_truth_bboxes):
        print(f"Error: Mismatched or empty lists for sequence {sequence_name}. "
              f"Tracked: {len(tracked_bboxes) if tracked_bboxes else 0}, GT: {len(ground_truth_bboxes) if ground_truth_bboxes else 0}")
        return None

    num_frames = len(tracked_bboxes)
    iou_scores = []
    cle_scores = []
    success_frames = 0

    for i in range(num_frames):
        track_box = tracked_bboxes[i]
        gt_box = ground_truth_bboxes[i]

        # Handle potential None values if tracker failed catastrophically (or GT missing)
        if track_box is None or gt_box is None:
            # Assign worst scores if either box is missing for a frame
            iou_scores.append(0.0)
            cle_scores.append(float('inf'))
            continue # Skip to next frame

        # Calculate Metrics
        iou = calculate_iou(track_box, gt_box)
        cle = calculate_cle(track_box, gt_box)

        iou_scores.append(iou)
        cle_scores.append(cle)

        if iou >= IOU_SUCCESS_THRESHOLD:
            success_frames += 1

    # Calculate aggregate metrics
    average_iou = np.mean(iou_scores) if iou_scores else 0.0
    average_cle = np.mean([s for s in cle_scores if s != float('inf')]) if any(s != float('inf') for s in cle_scores) else float('inf') # Avg finite CLE
    success_rate = (success_frames / num_frames) * 100 if num_frames > 0 else 0.0

    print(f"  Avg IoU: {average_iou:.3f}")
    print(f"  Avg CLE: {average_cle:.3f}" if average_cle != float('inf') else "  Avg CLE: Inf")
    print(f"  Success Rate (IoU >= {IOU_SUCCESS_THRESHOLD}): {success_rate:.2f}%")

    results = {
        "sequence": sequence_name,
        "num_frames": num_frames,
        "avg_iou": average_iou,
        "avg_cle": average_cle,
        "success_rate": success_rate,
        "iou_per_frame": iou_scores,
        "cle_per_frame": cle_scores
    }
    return results

# --- Plotting Function (Optional) ---
def plot_success_curve(all_sequence_results, title="Success Plot"):
    """ Generates a Success Plot (Precision Plot based on IoU threshold). """
    if not all_sequence_results: return

    # Combine all per-frame IoU scores from all sequences
    all_ious = []
    for result in all_sequence_results.values():
        if result and "iou_per_frame" in result:
            all_ious.extend(result["iou_per_frame"])

    if not all_ious:
        print("No IoU scores available for plotting.")
        return

    thresholds = np.linspace(0, 1, 101) # 101 points from 0 to 1
    success_rates_at_threshold = []

    total_frames = len(all_ious)
    if total_frames == 0: return

    for thresh in thresholds:
        successful_frames = sum(1 for iou in all_ious if iou >= thresh)
        success_rates_at_threshold.append(successful_frames / total_frames)

    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, success_rates_at_threshold, lw=2)
    plt.xlabel("IoU Threshold")
    plt.ylabel("Success Rate (Fraction of frames)")
    plt.title(title)
    plt.grid(True)
    plt.axis([0, 1, 0, 1]) # Set axis limits
    # Calculate Area Under Curve (AUC) - simple approximation
    auc = np.trapz(success_rates_at_threshold, thresholds)
    plt.text(0.1, 0.1, f"AUC: {auc:.3f}", fontsize=12)
    print(f"\nSuccess Plot AUC (Area Under Curve): {auc:.4f}")
    plt.show()


# --- Main Execution ---
if __name__ == "__main__":
    print("--- Phase 4: Tracker Evaluation ---")

    # 1. Define Tuning and Test Sets
    #    Use the list of 20 sequence names from Phase 0.
    all_sequences = [
        '8_birds', '8_car', '8_crossing', '8_crouching', '8_crowd',
        '8_depthwise_crossing', '8_garden', '8_hiding', '8_horse', '8_jacket',
        '8_mixed_distractors', '8_quadrocopter', '8_quadrocopter2',
        '8_rhino_behind_tree', '8_running_rhino', '8_saturated', '8_selma',
        '8_soccer', '8_street', '8_trees'
    ]
    # Shuffle or manually assign
    # random.shuffle(all_sequences) # If you want random split
    # tuning_sequences = all_sequences[:14] # Example 70%
    # test_sequences = all_sequences[14:]  # Example 30%

    tuning_sequences = [
    '8_rhino_behind_tree', '8_garden', '8_hiding', '8_saturated',
    '8_car', '8_crowd', '8_birds', '8_depthwise_crossing',
    '8_quadrocopter', '8_selma', '8_trees', '8_soccer'
    ]
    test_sequences = [
    '8_running_rhino', '8_horse', '8_mixed_distractors', '8_street',
    '8_crouching', '8_crossing', '8_jacket', '8_quadrocopter2'
    ]

    print(f"Tuning Set Sequences: {tuning_sequences}")
    print(f"Test Set Sequences: {test_sequences}")

    # 2. Choose Which Set to Evaluate (e.g., for tuning or final test)
    sequences_to_evaluate = test_sequences # Evaluate the TEST set for final results
    # sequences_to_evaluate = tuning_sequences # Evaluate the TUNING set while adjusting parameters

    # 3. Define Detector Configuration (Use the parameters you are testing/tuned)
    #    These should match the setup you intend to evaluate.
    #    Example using MOG2 detector (via process_fg_mask)
    #    If using Adaptive Thresh, set detector_func and params accordingly.
    detector_params_for_eval = { # Parameters for process_fg_mask
        "min_area": DEFAULT_MIN_CONTOUR_AREA,
        "max_area": DEFAULT_MAX_CONTOUR_AREA,
        "morph_kernel_size": DEFAULT_MORPH_KERNEL_SIZE,
        "open_iterations": DEFAULT_MORPH_OPEN_ITERATIONS,
        "close_iterations": DEFAULT_MORPH_CLOSE_ITERATIONS
    }
    # Note: MOG2 parameters (History, VarThreshold) are inside tracker.py's track_sequence
    # Note: Kalman parameters (Noise, IOU Threshold) are also inside tracker.py

    if 'PASTE_YOUR_FULL_LTIR_DATASET_PATH_HERE' in DATASET_BASE_PATH or not os.path.isdir(DATASET_BASE_PATH):
         print("="*60+"\n!!! ERROR: Set DATASET_BASE_PATH in data_loader.py !!!\n"+"="*60); exit()

    # 4. Run Tracking and Evaluation for Each Sequence
    all_results = {}
    print(f"\nRunning evaluation on {len(sequences_to_evaluate)} sequences...")
    for seq_name in sequences_to_evaluate:
        # Run the tracker (Phase 3 function) - Disable visualization for speed
        # NOTE: track_sequence needs modification if detector params are passed differently
        # Assuming track_sequence internally uses the imported defaults or has them hardcoded
        tracked_boxes, gt_boxes = track_sequence(seq_name, visualize=False)

        # Evaluate the results
        sequence_results = evaluate_sequence(seq_name, tracked_boxes, gt_boxes)
        if sequence_results:
            all_results[seq_name] = sequence_results

    # 5. Calculate and Print Overall Performance Summary
    print("\n--- Overall Evaluation Summary ---")
    if all_results:
        overall_avg_iou = np.mean([res['avg_iou'] for res in all_results.values() if res])
        valid_cles = [res['avg_cle'] for res in all_results.values() if res and res['avg_cle'] != float('inf')]
        overall_avg_cle = np.mean(valid_cles) if valid_cles else float('inf')
        overall_success_rate = np.mean([res['success_rate'] for res in all_results.values() if res])

        print(f"Sequences Evaluated: {len(all_results)}")
        print(f"Overall Average IoU: {overall_avg_iou:.4f}")
        print(f"Overall Average CLE (finite only): {overall_avg_cle:.4f}" if overall_avg_cle != float('inf') else "Overall Average CLE: Inf")
        print(f"Overall Average Success Rate (IoU >= {IOU_SUCCESS_THRESHOLD}): {overall_success_rate:.2f}%")

        # 6. Generate Success Plot (Optional)
        plot_success_curve(all_results, title=f"Success Plot ({len(sequences_to_evaluate)} Sequences)")

    else:
        print("No sequences were successfully evaluated.")