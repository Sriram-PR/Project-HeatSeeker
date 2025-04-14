# File: tune_kalman_tracker.py
# Purpose: Automated tuning of MOG2+Kalman tracker parameters using Random Search
#          on the Tuning Set.

import os
import cv2
import numpy as np
import random
import time
import csv

try:
    from data_loader import DATASET_BASE_PATH
    from tracker import track_sequence
    from evaluate_tracker import evaluate_sequence, IOU_SUCCESS_THRESHOLD
except ImportError as e:
    print("="*60)
    print(f"ERROR: Could not import necessary functions: {e}")
    print("Please ensure refactored tracker.py, data_loader.py, evaluate_tracker.py exist.")
    print("="*60)
    exit()

# --- Tuning Configuration ---

# 1. Parameter Search Space (Define ranges or lists of values to sample from)
PARAM_SPACE = {
    'mog2_history': [50, 100, 150, 200, 250],              # List of specific values
    'mog2_var_threshold': [16, 25, 36, 49, 64],           # List of specific values
    'kf_process_noise': [1e-3, 5e-3, 1e-2, 5e-2, 1e-1],    # Logarithmic-like steps
    'kf_measurement_noise': [1e-2, 5e-2, 1e-1, 5e-1, 1.0],# Logarithmic-like steps
    'iou_threshold': [0.05, 0.1, 0.15, 0.2, 0.25],          # Linear steps
    'min_contour_area': [30, 50, 75, 100, 150],           # List of specific values
    'max_contour_area': [10000, 15000, 20000, 30000],      # List of specific values
    'morph_kernel_size': [(3,3), (5,5)],                  # Specific tuples
    'morph_open_iter': [1, 2],                            # Specific integers
    'morph_close_iter': [1, 2, 3]                         # Specific integers
}

# 2. Number of Random Trials to Run
NUM_RANDOM_TRIALS = 100

# 3. Dataset Split for Tuning (MUST BE THE TUNING SET)
#    Use the same list definition as in pipeline.py/evaluate_tracker.py
# >>>>> PASTE YOUR ACTUAL TUNING LIST HERE <<<<<
TUNING_SEQUENCES = [
'8_rhino_behind_tree', '8_garden', '8_hiding', '8_saturated',
'8_car', '8_crowd', '8_birds', '8_depthwise_crossing',
'8_quadrocopter', '8_selma', '8_trees', '8_soccer'
]
# >>>>> END PASTE <<<<<

# 4. Metric to Optimize (Primary goal for ranking results)
#    Options: 'avg_iou', 'success_rate', 'avg_cle' (lower is better for CLE)
OPTIMIZATION_METRIC = 'avg_iou'
HIGHER_IS_BETTER = True # Set to False if optimizing for CLE

# 5. Results Output File
OUTPUT_FOLDER = 'output'
RESULTS_CSV_FILENAME = 'kalman_tuning_results.csv'
RESULTS_CSV_FILE = os.path.join(OUTPUT_FOLDER, RESULTS_CSV_FILENAME)

# --- Helper Function ---
def sample_parameters(param_space):
    """ Randomly samples one value for each parameter from the defined space. """
    sampled_params = {}
    for key, values in param_space.items():
        sampled_params[key] = random.choice(values)
    return sampled_params

# --- Main Tuning Loop ---
if __name__ == "__main__":
    print("="*60)
    print("Starting MOG2+Kalman Tracker Parameter Tuning (Random Search)")
    print(f"Number of Trials: {NUM_RANDOM_TRIALS}")
    print(f"Tuning Sequences ({len(TUNING_SEQUENCES)}): {TUNING_SEQUENCES}")
    print(f"Optimizing for: {OPTIMIZATION_METRIC} ({'Higher' if HIGHER_IS_BETTER else 'Lower'} is better)")
    print("="*60)

    # Verify dataset path
    if 'PASTE_YOUR_FULL_LTIR_DATASET_PATH_HERE' in DATASET_BASE_PATH or not os.path.isdir(DATASET_BASE_PATH):
         print("="*60+"\n!!! ERROR: Set DATASET_BASE_PATH in data_loader.py !!!\n"+"="*60); exit()
         
    try:
        if not os.path.exists(OUTPUT_FOLDER):
            os.makedirs(OUTPUT_FOLDER)
            print(f"Created output directory: {OUTPUT_FOLDER}")
    except OSError as e:
        print(f"Error creating output directory {OUTPUT_FOLDER}: {e}")
        print("Results may not be saved correctly.")

    all_trial_results = []
    start_tuning_time = time.time()

    # Prepare CSV file
    try:
        with open(RESULTS_CSV_FILE, 'w', newline='') as csvfile:
            # Define header: Parameters first, then metrics
            fieldnames = list(PARAM_SPACE.keys()) + ['avg_iou', 'avg_cle', 'success_rate', 'avg_time_sec']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
    except IOError as e:
        print(f"Warning: Could not open CSV file {RESULTS_CSV_FILE} for writing: {e}")
        RESULTS_CSV_FILE = None # Disable CSV writing

    for trial in range(NUM_RANDOM_TRIALS):
        print(f"\n--- Trial {trial + 1}/{NUM_RANDOM_TRIALS} ---")
        # 1. Sample a parameter combination
        current_params = sample_parameters(PARAM_SPACE)
        print(f"Parameters: {current_params}")

        # 2. Evaluate this combination on the Tuning Set
        trial_sequence_results = {}
        trial_start_time = time.time()

        for seq_name in TUNING_SEQUENCES:
            # Run tracking with the sampled parameters (no visualization)
            try:
                tracked_boxes, gt_boxes = track_sequence(seq_name, visualize=False, **current_params)
                # Evaluate the sequence
                seq_result = evaluate_sequence(seq_name, tracked_boxes, gt_boxes)
                if seq_result:
                    trial_sequence_results[seq_name] = seq_result
                else:
                     print(f"  Warning: Evaluation failed for {seq_name} in trial {trial+1}")
            except Exception as e:
                print(f"  ERROR during tracking/evaluation for {seq_name} in trial {trial+1}: {e}")
                # Optionally decide whether to skip the trial or assign a bad score

        trial_end_time = time.time()
        trial_duration = trial_end_time - trial_start_time

        # 3. Calculate Average Performance for this trial
        if trial_sequence_results:
            num_eval_seq = len(trial_sequence_results)
            avg_iou = np.mean([res['avg_iou'] for res in trial_sequence_results.values()])
            valid_cles = [res['avg_cle'] for res in trial_sequence_results.values() if res['avg_cle'] != float('inf')]
            avg_cle = np.mean(valid_cles) if valid_cles else float('inf')
            avg_success_rate = np.mean([res['success_rate'] for res in trial_sequence_results.values()])
            avg_time = trial_duration / num_eval_seq if num_eval_seq > 0 else 0

            print(f"Trial Avg Performance: IoU={avg_iou:.4f}, CLE={avg_cle:.2f}, Success={avg_success_rate:.2f}%, Time/Seq={avg_time:.2f}s")

            # Store results
            result_entry = {
                **current_params, # Add parameter values
                'avg_iou': avg_iou,
                'avg_cle': avg_cle,
                'success_rate': avg_success_rate,
                'avg_time_sec': avg_time
            }
            all_trial_results.append(result_entry)

            # Write to CSV immediately
            if RESULTS_CSV_FILE:
                try:
                    with open(RESULTS_CSV_FILE, 'a', newline='') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        # Format results for CSV writer
                        row_data = {k: result_entry.get(k, '') for k in fieldnames}
                        writer.writerow(row_data)
                except IOError as e:
                    print(f"Warning: Could not write to CSV file {RESULTS_CSV_FILE}: {e}")
        else:
            print("Trial failed: No sequences evaluated successfully.")
            # Optionally append a failure entry
            # all_trial_results.append({'params': current_params, 'score': -float('inf') if HIGHER_IS_BETTER else float('inf')})


    end_tuning_time = time.time()
    print("\n" + "="*60)
    print(f"Tuning Finished. Total time: {end_tuning_time - start_tuning_time:.2f} seconds")
    print("="*60)

    # 4. Report Best Results
    if not all_trial_results:
        print("No successful trials completed.")
    else:
        # Sort results based on the chosen metric
        all_trial_results.sort(key=lambda x: x.get(OPTIMIZATION_METRIC, -float('inf') if HIGHER_IS_BETTER else float('inf')),
                               reverse=HIGHER_IS_BETTER)

        print(f"\n--- Top 5 Parameter Combinations (Optimized for {OPTIMIZATION_METRIC}) ---")
        for i, result in enumerate(all_trial_results[:5]):
            print(f"\nRank {i+1}: Score ({OPTIMIZATION_METRIC}) = {result.get(OPTIMIZATION_METRIC):.4f}")
            print(f"  IoU={result.get('avg_iou'):.4f}, CLE={result.get('avg_cle'):.2f}, Success={result.get('success_rate'):.2f}%")
            print(f"  Params:")
            param_keys = PARAM_SPACE.keys()
            for key in param_keys:
                print(f"    {key}: {result.get(key)}")

        print("\nFull results saved to:", RESULTS_CSV_FILE if RESULTS_CSV_FILE else "CSV saving disabled.")

    print("\nReminder: Use the best parameters found here to configure your tracker")
    print("for the final evaluation run on the TEST SET using pipeline.py or evaluate_tracker.py.")