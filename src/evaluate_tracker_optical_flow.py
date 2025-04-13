# File: evaluate_tracker_optical_flow.py
# Phase 4: Evaluation of the Optical Flow Tracker's Performance

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt # For plotting results

# --- Attempt to Import ---
try:
    # Need the OF tracking function and its dependencies
    from optical_flow_tracker import track_sequence_optical_flow
    # Need evaluation helpers (IoU, CLE) - copy or import from tracker/evaluate_tracker
    from tracker import calculate_iou # Assuming it's in tracker.py
    # Need calculate_cle (copy from evaluate_tracker.py or define here)
    # Need base path config
    from data_loader import DATASET_BASE_PATH
except ImportError as e:
    print("="*60)
    print(f"ERROR: Could not import necessary functions: {e}")
    print("Ensure data_loader.py, optical_flow_tracker.py, and potentially tracker.py (for IoU) are present.")
    print("="*60)
    exit()

# --- Evaluation Configuration ---
IOU_SUCCESS_THRESHOLD = 0.5 # Standard threshold

# --- Helper Function (Center Location Error - Copied for self-containment) ---
def calculate_cle(boxA, boxB):
    """ Calculates Center Location Error (CLE) between two boxes [x, y, w, h]. """
    if boxA is None or boxB is None: return float('inf')
    centerAx = boxA[0] + boxA[2] / 2.0; centerAy = boxA[1] + boxA[3] / 2.0
    centerBx = boxB[0] + boxB[2] / 2.0; centerBy = boxB[1] + boxB[3] / 2.0
    error = np.sqrt((centerAx - centerBx)**2 + (centerAy - centerBy)**2)
    return error

# --- Main Evaluation Function (Identical to previous evaluate_tracker.py) ---
def evaluate_sequence(sequence_name, tracked_bboxes, ground_truth_bboxes):
    """ Evaluates tracking performance for a single sequence. """
    print(f"Evaluating sequence: {sequence_name}")
    if not tracked_bboxes or not ground_truth_bboxes or len(tracked_bboxes) != len(ground_truth_bboxes):
        print(f"Error: Mismatched/empty lists for {sequence_name}. Tracked: {len(tracked_bboxes)}, GT: {len(ground_truth_bboxes)}")
        return None

    num_frames = len(tracked_bboxes)
    iou_scores = []
    cle_scores = []
    success_frames = 0

    for i in range(num_frames):
        track_box = tracked_bboxes[i]
        gt_box = ground_truth_bboxes[i]
        if track_box is None or gt_box is None:
            iou_scores.append(0.0); cle_scores.append(float('inf')); continue

        iou = calculate_iou(track_box, gt_box)
        cle = calculate_cle(track_box, gt_box)
        iou_scores.append(iou); cle_scores.append(cle)
        if iou >= IOU_SUCCESS_THRESHOLD: success_frames += 1

    average_iou = np.mean(iou_scores) if iou_scores else 0.0
    valid_cle = [s for s in cle_scores if s != float('inf')]
    average_cle = np.mean(valid_cle) if valid_cle else float('inf')
    success_rate = (success_frames / num_frames) * 100 if num_frames > 0 else 0.0

    print(f"  Avg IoU: {average_iou:.3f}")
    print(f"  Avg CLE: {average_cle:.3f}" if average_cle != float('inf') else "  Avg CLE: Inf")
    print(f"  Success Rate (IoU >= {IOU_SUCCESS_THRESHOLD}): {success_rate:.2f}%")

    results = {
        "sequence": sequence_name, "num_frames": num_frames, "avg_iou": average_iou,
        "avg_cle": average_cle, "success_rate": success_rate,
        "iou_per_frame": iou_scores, "cle_per_frame": cle_scores
    }
    return results

# --- Plotting Function (Identical to previous evaluate_tracker.py) ---
def plot_success_curve(all_sequence_results, title="Success Plot - Optical Flow"): # Modified Title
    """ Generates a Success Plot (Precision Plot based on IoU threshold). """
    if not all_sequence_results: return
    all_ious = [iou for res in all_sequence_results.values() if res for iou in res.get("iou_per_frame", [])]
    if not all_ious: print("No IoU scores for plotting."); return

    thresholds = np.linspace(0, 1, 101)
    success_rates_at_threshold = []
    total_frames = len(all_ious)
    if total_frames == 0: return

    for thresh in thresholds:
        success_rates_at_threshold.append(sum(1 for iou in all_ious if iou >= thresh) / total_frames)

    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, success_rates_at_threshold, lw=2, label='Optical Flow Tracker') # Added label
    plt.xlabel("IoU Threshold")
    plt.ylabel("Success Rate (Fraction of frames)")
    plt.title(title)
    plt.grid(True)
    plt.axis([0, 1, 0, 1])
    auc = np.trapz(success_rates_at_threshold, thresholds)
    plt.text(0.1, 0.1, f"AUC: {auc:.3f}", fontsize=12)
    plt.legend() # Show label
    print(f"\nSuccess Plot AUC (Area Under Curve) for Optical Flow: {auc:.4f}")
    plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    print("--- Phase 4: Evaluation for OPTICAL FLOW Tracker ---") # Modified Title

    # 1. Define Tuning and Test Sets (Use the SAME split as for Kalman)
    #    CRITICAL: Ensure these lists EXACTLY match the ones used for Kalman evaluation for fair comparison.
    all_sequences = [ # Reference list
        '8_birds', '8_car', '8_crossing', '8_crouching', '8_crowd',
        '8_depthwise_crossing', '8_garden', '8_hiding', '8_horse', '8_jacket',
        '8_mixed_distractors', '8_quadrocopter', '8_quadrocopter2',
        '8_rhino_behind_tree', '8_running_rhino', '8_saturated', '8_selma',
        '8_soccer', '8_street', '8_trees'
    ]
    # >>>>> PASTE YOUR ACTUAL LISTS HERE <<<<<
    tuning_sequences = ['8_car', '8_crossing', '8_crouching', '8_crowd', '8_depthwise_crossing',
                        '8_garden', '8_hiding', '8_horse', '8_jacket', '8_mixed_distractors',
                        '8_quadrocopter', '8_quadrocopter2', '8_rhino_behind_tree', '8_running_rhino'] # Example
    test_sequences = ['8_birds', '8_saturated', '8_selma', '8_soccer', '8_street', '8_trees'] # Example
    # >>>>> END PASTE <<<<<

    print(f"Using Tuning Set: {tuning_sequences}")
    print(f"Using Test Set: {test_sequences}")

    # 2. Choose Which Set to Evaluate
    sequences_to_evaluate = test_sequences # Evaluate the TEST set for final results
    # sequences_to_evaluate = tuning_sequences # Evaluate the TUNING set while adjusting parameters

    # 3. Detector/Tracker Configuration (NOTE for Optical Flow)
    #    The 'track_sequence_optical_flow' function currently uses parameters
    #    defined INTERNALLY within its file (e.g., FLOW_MAGNITUDE_THRESHOLD).
    #    This evaluation script assumes those defaults are being used.
    #    For rigorous comparison, refactor 'track_sequence_optical_flow'
    #    to accept these parameters as arguments.
    print("\nNOTE: Evaluating Optical Flow tracker using its internally defined parameters.")
    print("(Refactor track_sequence_optical_flow to accept parameters for rigorous tuning/comparison)")

    # Verify data path is set
    if 'PASTE_YOUR_FULL_LTIR_DATASET_PATH_HERE' in DATASET_BASE_PATH or not os.path.isdir(DATASET_BASE_PATH):
         print("="*60+"\n!!! ERROR: Set DATASET_BASE_PATH in data_loader.py !!!\n"+"="*60); exit()

    # 4. Run Tracking and Evaluation for Each Sequence
    all_results_of = {} # Store optical flow results
    print(f"\nRunning OPTICAL FLOW evaluation on {len(sequences_to_evaluate)} sequences...")
    for seq_name in sequences_to_evaluate:
        # Run the OPTICAL FLOW tracker (Phase 3 comparison function)
        tracked_boxes, gt_boxes = track_sequence_optical_flow(seq_name, visualize=False)

        # Evaluate the results using the standard evaluation function
        sequence_results = evaluate_sequence(seq_name, tracked_boxes, gt_boxes)
        if sequence_results:
            all_results_of[seq_name] = sequence_results

    # 5. Calculate and Print Overall Performance Summary for OPTICAL FLOW
    print("\n--- Overall OPTICAL FLOW Evaluation Summary ---") # Modified Title
    if all_results_of:
        overall_avg_iou_of = np.mean([res['avg_iou'] for res in all_results_of.values() if res])
        valid_cles_of = [res['avg_cle'] for res in all_results_of.values() if res and res['avg_cle'] != float('inf')]
        overall_avg_cle_of = np.mean(valid_cles_of) if valid_cles_of else float('inf')
        overall_success_rate_of = np.mean([res['success_rate'] for res in all_results_of.values() if res])

        print(f"Sequences Evaluated: {len(all_results_of)}")
        print(f"Overall Average IoU (OF): {overall_avg_iou_of:.4f}")
        print(f"Overall Average CLE (OF, finite only): {overall_avg_cle_of:.4f}" if overall_avg_cle_of != float('inf') else "Overall Average CLE (OF): Inf")
        print(f"Overall Average Success Rate (OF, IoU >= {IOU_SUCCESS_THRESHOLD}): {overall_success_rate_of:.2f}%")

        # 6. Generate Success Plot for OPTICAL FLOW
        plot_success_curve(all_results_of, title=f"Success Plot - Optical Flow ({len(sequences_to_evaluate)} Sequences)")

    else:
        print("No sequences were successfully evaluated for the Optical Flow tracker.")