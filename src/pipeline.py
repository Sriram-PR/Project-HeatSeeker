# File: pipeline.py
# Purpose: Master script to run and evaluate tracking pipelines (MOG2+Kalman vs. Optical Flow).

import os
import time
import numpy as np
import matplotlib.pyplot as plt

try:
    from data_loader import DATASET_BASE_PATH
    from tracker import track_sequence as track_sequence_kalman
    from optical_flow_tracker import track_sequence_optical_flow
    from evaluate_tracker import evaluate_sequence, calculate_cle, calculate_iou, plot_success_curve

except ImportError as e:
    print("="*60)
    print(f"ERROR: Could not import necessary functions: {e}")
    print("Please ensure all required .py files (data_loader, tracker, optical_flow_tracker, evaluate_tracker) exist in the same directory.")
    print("="*60)
    exit()

# --- Configuration ---

# 1. Select which trackers to run
RUN_KALMAN_TRACKER = True
RUN_OPTICAL_FLOW_TRACKER = True

# 2. Select which dataset split to evaluate on
EVALUATE_TUNING_SET = False
EVALUATE_TEST_SET = True  # Typically evaluate on the test set for final results

# 3. Enable/Disable visualization during tracking
VISUALIZE_TRACKING = False # Set to True to see tracker output frame-by-frame (slow)

# 4. Define Sequence Splits
all_sequences_8bit = [
    '8_birds', '8_car', '8_crossing', '8_crouching', '8_crowd',
    '8_depthwise_crossing', '8_garden', '8_hiding', '8_horse', '8_jacket',
    '8_mixed_distractors', '8_quadrocopter', '8_quadrocopter2',
    '8_rhino_behind_tree', '8_running_rhino', '8_saturated', '8_selma',
    '8_soccer', '8_street', '8_trees'
]

TUNING_SEQUENCES = [
'8_rhino_behind_tree', '8_garden', '8_hiding', '8_saturated',
'8_car', '8_crowd', '8_birds', '8_depthwise_crossing',
'8_quadrocopter', '8_selma', '8_trees', '8_soccer'
]
TEST_SEQUENCES = [
'8_running_rhino', '8_horse', '8_mixed_distractors', '8_street',
'8_crouching', '8_crossing', '8_jacket', '8_quadrocopter2'
]

# 5. Evaluation Parameters
IOU_SUCCESS_THRESHOLD = 0.5

# --- Helper Functions ---

def run_and_evaluate_tracker(tracker_func, sequence_list, tracker_name):
    """Runs a specific tracker function on a list of sequences and evaluates."""
    print(f"\n--- Running Pipeline for: {tracker_name} ---")
    all_results = {}
    total_start_time = time.time()

    for i, seq_name in enumerate(sequence_list):
        print(f"Processing Sequence {i+1}/{len(sequence_list)}: {seq_name}...")
        start_time = time.time()

        # Run the tracking function
        try:
            tracked_boxes, gt_boxes = tracker_func(seq_name, visualize=VISUALIZE_TRACKING)
        except Exception as e:
            print(f"  ERROR during tracking for {seq_name}: {e}")
            continue # Skip to next sequence

        tracking_time = time.time() - start_time

        # Evaluate the results
        sequence_results = evaluate_sequence(seq_name, tracked_boxes, gt_boxes)
        if sequence_results:
            sequence_results["tracking_time_sec"] = tracking_time
            all_results[seq_name] = sequence_results
            print(f"  Sequence Time: {tracking_time:.2f} sec")
        else:
            print(f"  Evaluation failed for {seq_name}.")

    total_end_time = time.time()
    print(f"--- Finished Pipeline for: {tracker_name} ---")
    print(f"Total time for {len(sequence_list)} sequences: {total_end_time - total_start_time:.2f} sec")
    return all_results

def calculate_overall_metrics(results_dict):
    """Calculates average metrics across all evaluated sequences."""
    if not results_dict:
        return {"avg_iou": 0, "avg_cle": float('inf'), "avg_success_rate": 0, "avg_time": 0, "evaluated_count": 0}

    all_ious = [res['avg_iou'] for res in results_dict.values() if res]
    all_cles = [res['avg_cle'] for res in results_dict.values() if res and res['avg_cle'] != float('inf')]
    all_success_rates = [res['success_rate'] for res in results_dict.values() if res]
    all_times = [res.get('tracking_time_sec', 0) for res in results_dict.values() if res] # Use get for safety

    overall_avg_iou = np.mean(all_ious) if all_ious else 0.0
    overall_avg_cle = np.mean(all_cles) if all_cles else float('inf')
    overall_success_rate = np.mean(all_success_rates) if all_success_rates else 0.0
    overall_avg_time = np.mean(all_times) if all_times else 0.0

    return {
        "avg_iou": overall_avg_iou,
        "avg_cle": overall_avg_cle,
        "avg_success_rate": overall_success_rate,
        "avg_time": overall_avg_time,
        "evaluated_count": len(results_dict)
    }

# --- Main Execution ---
if __name__ == "__main__":
    print("="*60)
    print("Starting Tracking and Evaluation Pipeline")
    print("="*60)

    # 1. Verify Dataset Path
    if 'PASTE_YOUR_FULL_LTIR_DATASET_PATH_HERE' in DATASET_BASE_PATH or not os.path.isdir(DATASET_BASE_PATH):
         print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
         print("!!! ERROR: Please set the DATASET_BASE_PATH variable   !!!")
         print("!!! correctly inside the 'data_loader.py' file first. !!!")
         print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
         exit()
    else:
         print(f"Using Dataset Path: {DATASET_BASE_PATH}")

    # 2. Determine Sequences to Process
    sequences_to_process = []
    set_name = ""
    if EVALUATE_TUNING_SET:
        sequences_to_process.extend(TUNING_SEQUENCES)
        set_name += "Tuning Set"
    if EVALUATE_TEST_SET:
        sequences_to_process.extend(TEST_SEQUENCES)
        set_name = "Test Set" if not set_name else "Tuning & Test Sets"

    if not sequences_to_process:
        print("Warning: No dataset split selected for evaluation (EVALUATE_TUNING_SET or EVALUATE_TEST_SET is False).")
        exit()

    # Remove duplicates if both sets were selected
    sequences_to_process = sorted(list(set(sequences_to_process)))
    print(f"\nEvaluating on: {set_name} ({len(sequences_to_process)} sequences)")
    # print(f"Sequences: {sequences_to_process}") # Uncomment to list all sequences

    # 3. Run Selected Trackers and Store Results
    kalman_results = None
    optical_flow_results = None

    if RUN_KALMAN_TRACKER:
        kalman_results = run_and_evaluate_tracker(
            tracker_func=track_sequence_kalman,
            sequence_list=sequences_to_process,
            tracker_name="MOG2 + Kalman Filter"
        )

    if RUN_OPTICAL_FLOW_TRACKER:
        optical_flow_results = run_and_evaluate_tracker(
            tracker_func=track_sequence_optical_flow,
            sequence_list=sequences_to_process,
            tracker_name="Optical Flow + Centroid"
        )

    # 4. Summarize and Compare Results
    print("\n" + "="*60)
    print(f"Overall Performance Summary ({set_name})")
    print("="*60)

    kalman_summary = {}
    of_summary = {}

    if kalman_results:
        kalman_summary = calculate_overall_metrics(kalman_results)
        print("\n--- MOG2 + Kalman Filter ---")
        print(f" Sequences Evaluated: {kalman_summary['evaluated_count']}")
        print(f" Average IoU:         {kalman_summary['avg_iou']:.4f}")
        print(f" Average CLE (finite):{kalman_summary['avg_cle']:.4f}" if kalman_summary['avg_cle'] != float('inf') else " Average CLE (finite): Inf")
        print(f" Average Success Rate: {kalman_summary['avg_success_rate']:.2f}% (IoU >= {IOU_SUCCESS_THRESHOLD})")
        print(f" Average Time/Seq:    {kalman_summary['avg_time']:.2f} sec")

    if optical_flow_results:
        of_summary = calculate_overall_metrics(optical_flow_results)
        print("\n--- Optical Flow + Centroid ---")
        print(f" Sequences Evaluated: {of_summary['evaluated_count']}")
        print(f" Average IoU:         {of_summary['avg_iou']:.4f}")
        print(f" Average CLE (finite):{of_summary['avg_cle']:.4f}" if of_summary['avg_cle'] != float('inf') else " Average CLE (finite): Inf")
        print(f" Average Success Rate: {of_summary['avg_success_rate']:.2f}% (IoU >= {IOU_SUCCESS_THRESHOLD})")
        print(f" Average Time/Seq:    {of_summary['avg_time']:.2f} sec")

    # 5. Plot Success Curves (if results exist)
    if kalman_results or optical_flow_results:
        print("\nGenerating Success Plot(s)...")
        plt.figure(figsize=(10, 7)) # Create a single figure for combined plots

        if kalman_results:
            # Combine all per-frame IoU scores
            all_kalman_ious = [iou for res in kalman_results.values() if res for iou in res.get("iou_per_frame", [])]
            if all_kalman_ious:
                thresholds = np.linspace(0, 1, 101)
                kalman_success_rates = []
                total_frames = len(all_kalman_ious)
                for thresh in thresholds:
                    kalman_success_rates.append(sum(1 for iou in all_kalman_ious if iou >= thresh) / total_frames)
                auc_kalman = np.trapz(kalman_success_rates, thresholds)
                plt.plot(thresholds, kalman_success_rates, lw=2, label=f'Kalman (AUC={auc_kalman:.3f})')
                print(f"  Kalman Tracker AUC: {auc_kalman:.4f}")

        if optical_flow_results:
             # Combine all per-frame IoU scores
            all_of_ious = [iou for res in optical_flow_results.values() if res for iou in res.get("iou_per_frame", [])]
            if all_of_ious:
                thresholds = np.linspace(0, 1, 101)
                of_success_rates = []
                total_frames = len(all_of_ious)
                for thresh in thresholds:
                    of_success_rates.append(sum(1 for iou in all_of_ious if iou >= thresh) / total_frames)
                auc_of = np.trapz(of_success_rates, thresholds)
                plt.plot(thresholds, of_success_rates, lw=2, label=f'Optical Flow (AUC={auc_of:.3f})')
                print(f"  Optical Flow Tracker AUC: {auc_of:.4f}")

        # Finalize Plot
        plt.xlabel("IoU Threshold")
        plt.ylabel("Success Rate (Fraction of frames)")
        plt.title(f"Tracker Success Plot ({set_name} - {len(sequences_to_process)} Sequences)")
        plt.grid(True)
        plt.legend(loc='lower left')
        plt.axis([0, 1, 0, 1])
        plt.show()

    print("\n" + "="*60)
    print("Pipeline Execution Finished.")
    print("="*60)