# File: tune_kalman_grid.py
# Purpose: Automated tuning of MOG2+Kalman tracker parameters using Grid Search
#          on the Tuning Set, with checkpointing.

import os
import cv2
import numpy as np
import itertools
import time
import csv
import ast # For safely evaluating tuples read from CSV

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

# 1. Parameter Grid (Define specific values for each parameter)
PARAM_GRID = {
    # --- MOG2 ---
    'mog2_history': [100, 150, 200],
    'mog2_var_threshold': [16, 25, 36],
    # --- Kalman Filter ---
    'kf_process_noise': [5e-3, 1e-2, 5e-2],
    'kf_measurement_noise': [5e-2, 1e-1, 5e-1],
    # --- Association ---
    'iou_threshold': [0.1, 0.15, 0.2],
    # --- Detection Filtering ---
    'min_contour_area': [50, 75, 100],
    'max_contour_area': [15000, 20000],
    # --- Morphological Ops ---
    'morph_kernel_size': [(3,3), (5,5)],
    'morph_open_iter': [1], # Keep simple for grid size
    'morph_close_iter': [2] # Keep simple for grid size
}
# Note: Grid size grows exponentially! Adjust carefully.

# 2. Dataset Split for Tuning (MUST BE THE TUNING SET)
#    Use the same list definition as in pipeline.py/evaluate_tracker.py
TUNING_SEQUENCES = [
'8_rhino_behind_tree', '8_garden', '8_hiding', '8_saturated',
'8_car', '8_crowd', '8_birds', '8_depthwise_crossing',
'8_quadrocopter', '8_selma', '8_trees', '8_soccer'
]

# 3. Metric to Optimize (Primary goal for ranking results)
OPTIMIZATION_METRIC = 'avg_iou'
HIGHER_IS_BETTER = True # Set to False if optimizing for CLE

# 4. Results/Checkpoint File
OUTPUT_FOLDER = 'output'
RESULTS_CSV_FILENAME = 'kalman_grid_search_results.csv'
RESULTS_CSV_FILE = os.path.join(OUTPUT_FOLDER, RESULTS_CSV_FILENAME)

# --- Helper Functions ---

def generate_combinations(param_grid):
    """Generates all combinations from the parameter grid."""
    keys = param_grid.keys()
    values = param_grid.values()
    combinations = list(itertools.product(*values))
    param_combinations = [dict(zip(keys, combo)) for combo in combinations]
    return param_combinations, keys # Return keys for consistent ordering

def load_previous_results(filepath, param_names):
    """Loads previous results and returns processed signatures and full results."""
    processed_signatures = set()
    previous_results = []
    if not os.path.exists(filepath):
        return previous_results, processed_signatures

    try:
        with open(filepath, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                try:
                    # Recreate the parameter signature from the row
                    # Need to handle potential tuples stored as strings
                    param_values = []
                    for name in param_names:
                        val_str = row.get(name, '')
                        try:
                            # Safely evaluate if it looks like a tuple/list/etc.
                            if val_str.startswith(('(','[')):
                                val = ast.literal_eval(val_str)
                            # Otherwise, try converting to float/int if possible, else keep as string
                            elif '.' in val_str:
                                val = float(val_str)
                            else:
                                val = int(val_str)
                        except (ValueError, SyntaxError):
                            val = val_str # Keep as string if conversion fails
                        param_values.append(val)

                    signature = tuple(param_values) # Use tuple as signature
                    processed_signatures.add(signature)
                    # Convert metric values back to numeric types
                    numeric_row = {k: v for k, v in row.items() if k not in param_names}
                    for k in numeric_row:
                         try: numeric_row[k] = float(row[k])
                         except ValueError: pass # Keep as string if not floatable
                    full_result = {**dict(zip(param_names, param_values)), **numeric_row}
                    previous_results.append(full_result)
                except Exception as e:
                    print(f"Warning: Skipping malformed row in {filepath}: {row}. Error: {e}")
                    continue # Skip malformed rows
    except Exception as e:
        print(f"Error loading previous results from {filepath}: {e}")
        # Decide how to handle: continue without loading, or exit?
        # For safety, let's proceed without loading if error occurs.
        return [], set()

    print(f"Loaded {len(previous_results)} previous results. Found {len(processed_signatures)} unique processed signatures.")
    return previous_results, processed_signatures

def save_result(filepath, fieldnames, result_entry):
    """Appends a single result entry to the CSV file."""
    file_exists = os.path.exists(filepath)
    try:
        with open(filepath, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists or os.path.getsize(filepath) == 0:
                writer.writeheader() # Write header only if file is new/empty
            # Ensure all fieldnames are present in the row, fill with '' if missing
            row_data = {k: result_entry.get(k, '') for k in fieldnames}
            writer.writerow(row_data)
        return True
    except IOError as e:
        print(f"Error writing to CSV file {filepath}: {e}")
        return False

# --- Main Tuning Loop ---
if __name__ == "__main__":
    print("="*60)
    print("Starting MOG2+Kalman Tracker Parameter Tuning (Grid Search)")
    print(f"Parameter Grid:\n{PARAM_GRID}")
    print(f"Tuning Sequences ({len(TUNING_SEQUENCES)}): {TUNING_SEQUENCES}")
    print(f"Results/Checkpoint File: {RESULTS_CSV_FILE}")
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

    # Generate all combinations
    all_combinations, param_names_ordered = generate_combinations(PARAM_GRID)
    total_combinations = len(all_combinations)
    print(f"Total combinations to evaluate: {total_combinations}")
    if total_combinations > 1000: # Warning for large grids
        print("Warning: Grid size is very large, this may take a significant amount of time!")

    # Load previous results for checkpointing
    all_trial_results, processed_signatures = load_previous_results(RESULTS_CSV_FILE, param_names_ordered)
    start_index = len(all_trial_results) # Approximate start, relies on loading order

    start_tuning_time = time.time()
    combinations_processed_this_run = 0

    # Define CSV header order
    csv_fieldnames = list(param_names_ordered) + ['avg_iou', 'avg_cle', 'success_rate', 'avg_time_sec']

    for i, current_params in enumerate(all_combinations):
        # --- Checkpoint Check ---
        # Create signature based on the *ordered* parameter names
        current_signature = tuple(current_params[name] for name in param_names_ordered)

        if current_signature in processed_signatures:
            # print(f"Skipping Trial {i + 1}/{total_combinations} (Already processed)") # Can be verbose
            continue # Skip this combination

        # --- Run Evaluation ---
        combinations_processed_this_run += 1
        print(f"\n--- Processing Trial {i + 1}/{total_combinations} (Actual Run #{combinations_processed_this_run}) ---")
        print(f"Parameters: {current_params}")

        trial_sequence_results = {}
        trial_start_time = time.time()
        trial_successful = True

        for seq_name in TUNING_SEQUENCES:
            print(f"  Running on sequence: {seq_name}...")
            try:
                tracked_boxes, gt_boxes = track_sequence(seq_name, visualize=False, **current_params)
                seq_result = evaluate_sequence(seq_name, tracked_boxes, gt_boxes)
                if seq_result:
                    trial_sequence_results[seq_name] = seq_result
                else:
                     print(f"    Warning: Evaluation failed for {seq_name} in trial {i+1}")
                     # Decide if a single sequence failure invalidates the trial?
                     # For now, we continue and average over successful sequences.
            except Exception as e:
                print(f"    ERROR during tracking/evaluation for {seq_name} in trial {i+1}: {e}")
                # If one sequence crashes, maybe skip the rest of this trial?
                # trial_successful = False # Mark trial as potentially problematic
                # break # Stop processing sequences for this trial

        trial_end_time = time.time()
        trial_duration = trial_end_time - trial_start_time

        # --- Calculate Average Performance & Save ---
        if trial_sequence_results and trial_successful: # Only process if results exist and no major error occurred
            num_eval_seq = len(trial_sequence_results)
            avg_iou = np.mean([res['avg_iou'] for res in trial_sequence_results.values()])
            valid_cles = [res['avg_cle'] for res in trial_sequence_results.values() if res['avg_cle'] != float('inf')]
            avg_cle = np.mean(valid_cles) if valid_cles else float('inf')
            avg_success_rate = np.mean([res['success_rate'] for res in trial_sequence_results.values()])
            avg_time = trial_duration / num_eval_seq if num_eval_seq > 0 else 0

            print(f"Trial Avg Performance: IoU={avg_iou:.4f}, CLE={avg_cle:.2f}, Success={avg_success_rate:.2f}%, Time/Seq={avg_time:.2f}s")

            # Prepare entry for CSV and in-memory list
            result_entry = {
                **current_params,
                'avg_iou': avg_iou,
                'avg_cle': avg_cle,
                'success_rate': avg_success_rate,
                'avg_time_sec': avg_time
            }

            # Save immediately to CSV
            if save_result(RESULTS_CSV_FILE, csv_fieldnames, result_entry):
                # Add to in-memory list only after successful save
                all_trial_results.append(result_entry)
                processed_signatures.add(current_signature) # Mark as processed
            else:
                print("Error saving result, stopping execution to prevent data loss.")
                exit() # Stop if we can't save checkpoint

        else:
            print(f"Trial {i + 1} completed with errors or no successful evaluations. Skipping save.")
            # We don't save failed trials, they will be re-attempted on restart

        # Estimate remaining time (very rough)
        if combinations_processed_this_run > 0:
             elapsed_time = time.time() - start_tuning_time
             avg_time_per_run = elapsed_time / combinations_processed_this_run
             remaining_combinations = total_combinations - (i + 1) # How many are left in total loop
             # Refine remaining count based on how many we *expect* to run
             expected_to_run = total_combinations - (len(all_trial_results))
             estimated_remaining_time = avg_time_per_run * expected_to_run
             if estimated_remaining_time > 0:
                  print(f"Estimated time remaining: {time.strftime('%H:%M:%S', time.gmtime(estimated_remaining_time))}")


    end_tuning_time = time.time()
    print("\n" + "="*60)
    print(f"Grid Search Finished. Total time: {time.strftime('%H:%M:%S', time.gmtime(end_tuning_time - start_tuning_time))}")
    print(f"Total combinations processed across all runs: {len(all_trial_results)}")
    print("="*60)

    # --- Report Best Results ---
    if not all_trial_results:
        print("No successful trials completed.")
    else:
        # Sort results based on the chosen metric
        # Handle potential missing metrics during sorting
        def get_sort_key(x):
             val = x.get(OPTIMIZATION_METRIC)
             if val is None:
                  return -float('inf') if HIGHER_IS_BETTER else float('inf')
             # Ensure val is float for comparison if possible
             try: return float(val)
             except ValueError: return -float('inf') if HIGHER_IS_BETTER else float('inf')

        all_trial_results.sort(key=get_sort_key, reverse=HIGHER_IS_BETTER)

        print(f"\n--- Top 5 Parameter Combinations (Optimized for {OPTIMIZATION_METRIC}) ---")
        for i, result in enumerate(all_trial_results[:5]):
            # Safely get metric score
            score = result.get(OPTIMIZATION_METRIC, 'N/A')
            score_str = f"{score:.4f}" if isinstance(score, (float, int)) else str(score)
            print(f"\nRank {i+1}: Score ({OPTIMIZATION_METRIC}) = {score_str}")
            # Safely print other metrics
            iou_str = f"{result.get('avg_iou', 'N/A'):.4f}" if isinstance(result.get('avg_iou'), (float, int)) else 'N/A'
            cle_str = f"{result.get('avg_cle', 'N/A'):.2f}" if isinstance(result.get('avg_cle'), (float, int)) else 'N/A'
            succ_str = f"{result.get('success_rate', 'N/A'):.2f}%" if isinstance(result.get('success_rate'), (float, int)) else 'N/A'

            print(f"  IoU={iou_str}, CLE={cle_str}, Success={succ_str}")
            print(f"  Params:")
            for key in param_names_ordered: # Use ordered names
                print(f"    {key}: {result.get(key, 'N/A')}")

        print(f"\nFull results saved to: {RESULTS_CSV_FILE}")

    print("\nReminder: Use the best parameters found here to configure your tracker")
    print("for the final evaluation run on the TEST SET using pipeline.py or evaluate_tracker.py.")