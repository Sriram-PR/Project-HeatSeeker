"""
Bayesian Optimization script for tuning the Object Tracking Pipeline.
(GIF saving disabled during optimization)
"""
import os
import time
import traceback
import pandas as pd
import motmetrics as mm
import numpy as np
import cv2

try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
except ImportError:
    print("Error: scikit-optimize not installed. Use: pip install scikit-optimize"); exit()

from config import PipelineConfig
from data_loader_mot import DatasetLoader
from pipeline_runner import MotionDetectionPipeline
from evaluation import calculate_mot_metrics

print(f"OpenCV Threads: {cv2.getNumThreads()}")
print(f"motmetrics version: {mm.__version__}")
try: import skopt; print(f"scikit-optimize version: {skopt.__version__}")
except Exception: pass

# ============================================
# ======== Optimization Configuration ========
# ============================================
param_space = [
    Integer(50, 500, name='min_blob_area'),
    Integer(100, 800, name='mog2_history'),
    Real(10.0, 100.0, name='mog2_varThreshold'),
    Categorical([3, 5, 7], name='morph_kernel_size'),
    Categorical([3, 5, 7], name='gaussian_ksize'),
    Real(1.0, 5.0, name='clahe_clip_limit'),
    Categorical(['8x8', '12x12', '16x16'], name='clahe_tile_grid_size'),
    Integer(2, 10, name='max_misses'),
    Real(25.0, 150.0, name='association_threshold'),
    Real(1e-4, 0.8, prior='log-uniform', name='Q'),
    Real(15.0, 200.0, name='R')
]
param_names = [space.name for space in param_space]

OPTIMIZATION_TARGET_METRIC = 'idf1'
MAXIMIZE_METRIC = True

BASE_PATH = './data/ltir_v1_0_8bit_16bit/'

TRAINING_SEQUENCES = [
    '8_car', '8_garden', '8_hiding', '8_saturated', '8_crowd', '8_birds',
    '8_depthwise_crossing', '8_quadrocopter', '8_selma', '8_trees', '8_soccer'
]
VALIDATION_SEQUENCES = [
    '8_rhino_behind_tree', '8_running_rhino', '8_horse', '8_mixed_distractors', '8_street',
    '8_crouching', '8_crossing', '8_jacket', '8_quadrocopter2'
]

N_OPTIMIZATION_CALLS = 100
N_INITIAL_RANDOM_POINTS = 20
ACQUISITION_FUNCTION = 'EI'
RANDOM_STATE_SEED = 42

TIMESTAMP = time.strftime("%Y%m%d-%H%M%S")
OPTIMIZATION_NAME = f"{OPTIMIZATION_TARGET_METRIC}_optimization"
OUTPUT_DIR = "output"
TRIALS_CSV_PATH = os.path.join(OUTPUT_DIR, f'{OPTIMIZATION_NAME}_trials_{TIMESTAMP}.csv')
VALIDATION_CSV_PATH = os.path.join(OUTPUT_DIR, f'{OPTIMIZATION_NAME}_validation_{TIMESTAMP}.csv')

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Globals ---
optimization_trials_data = []
optimization_call_count = 0

# ========== Objective Function ==========
@use_named_args(param_space)
def objective_function(**params):
    global optimization_trials_data, optimization_call_count
    optimization_call_count += 1
    trial_start_time = time.time()
    trial_log = {'trial_num': optimization_call_count}
    current_params = params.copy()

    # Convert CLAHE grid size string back to tuple
    tuple_param_name = 'clahe_tile_grid_size'
    if tuple_param_name in current_params:
        grid_str = current_params[tuple_param_name]
        try:
            dims = list(map(int, grid_str.split('x')))
            if len(dims) == 2: current_params[tuple_param_name] = tuple(dims)
            else: raise ValueError("Invalid grid string format")
        except Exception as e:
             print(f"  ERROR: Invalid format for {tuple_param_name}: '{grid_str}'. Using (8, 8). Error: {e}")
             current_params[tuple_param_name] = (8, 8)
    trial_log.update(current_params)

    print(f"\n--- Trial {optimization_call_count}/{N_OPTIMIZATION_CALLS} ---")

    # --- Create Config ---
    try:
         config = PipelineConfig(**current_params, iou_threshold=0.3)
    except Exception as e:
         print(f"  ERROR: Failed to create PipelineConfig: {e}"); traceback.print_exc()
         trial_log['error_creating_config'] = str(e)[:200]
         return 2.0 if MAXIMIZE_METRIC else 1e9 # Bad score

    combined_accumulator = mm.MOTAccumulator(auto_id=True)
    processed_sequences = []; sequences_with_errors = []; total_frames_processed = 0

    # --- Run on Training Sequences ---
    for seq_name in TRAINING_SEQUENCES:
        seq_start_time = time.time()
        # print(f"  Processing training sequence: {seq_name}") # Can be verbose
        seq_success = False; error_msg = None; seq_frame_updates = []
        try:
            data = DatasetLoader(BASE_PATH, seq_name)
            if not data.frame_paths: error_msg = "No frames"
            elif data.gt_corners is None or not any(gt is not None for gt in data.gt_bboxes_with_id): error_msg = "No valid GT"
            if error_msg:
                print(f"    Warning: {error_msg} for {seq_name}. Skipping.")
                pass
            else:
                pipeline = MotionDetectionPipeline(data, config)
                seq_frame_updates, _ = pipeline.run(save_gif_path=None, save_csv_path=None) # NO GIF/CSV during opt
                if seq_frame_updates:
                    for frame_data in seq_frame_updates: combined_accumulator.update(*frame_data)
                    seq_success = True; total_frames_processed += len(seq_frame_updates)
                else: error_msg = "Pipeline yielded no frame data" # Can be verbose
        except Exception as e:
            error_msg = f"Pipeline Run ERROR: {e}"; traceback.print_exc()
        if error_msg: sequences_with_errors.append(f"{seq_name}:{error_msg[:50]}") # Log truncated error
        trial_log[f'seq_{seq_name}_success'] = seq_success
        trial_log[f'seq_{seq_name}_time'] = time.time() - seq_start_time

    # --- Calculate Metric ---
    worst_score = 2.0 if MAXIMIZE_METRIC else 1e9
    worst_metric_value = -1.0 if MAXIMIZE_METRIC else worst_score
    score = worst_score; metric_value = worst_metric_value; calculated_metric = None

    if combined_accumulator.events.shape[0] > 0:
        try:
            mh = mm.metrics.create()
            summary = mh.compute(combined_accumulator, metrics=[OPTIMIZATION_TARGET_METRIC], name='opt_eval')
            if not summary.empty and OPTIMIZATION_TARGET_METRIC in summary.columns:
                calculated_metric = summary[OPTIMIZATION_TARGET_METRIC].iloc[0]
                if pd.isna(calculated_metric): metric_value = worst_metric_value; score = worst_score
                else:
                     metric_value = float(calculated_metric)
                     score = (1.0 - metric_value) if MAXIMIZE_METRIC else metric_value
                     if MAXIMIZE_METRIC: score = max(0.0, score)
            else: trial_log['error_metric'] = f"Metric '{OPTIMIZATION_TARGET_METRIC}' not found"
        except Exception as e:
            print(f"  ERROR calculating combined metrics: {e}"); traceback.print_exc()
            trial_log['error_metric'] = str(e)[:200]; score = worst_score
    else: trial_log['error_metric'] = "No events accumulated"; score = worst_score

    # --- Finalize Log ---
    trial_duration = time.time() - trial_start_time
    print(f"  Score: {score:.5f} ({OPTIMIZATION_TARGET_METRIC.upper()}: {metric_value:.5f}) (Time: {trial_duration:.2f}s)")
    trial_log.update({'target_metric': OPTIMIZATION_TARGET_METRIC, 'metric_value': metric_value,
                      'score_to_minimize': score, 'processed_seq': ";".join(processed_sequences),
                      'errors': "; ".join(sequences_with_errors), 'frames': total_frames_processed,
                      'time_sec': trial_duration})
    optimization_trials_data.append(trial_log)

    # --- Save Log Periodically ---
    if optimization_call_count > 0 and optimization_call_count % 10 == 0:
         try:
             pd.DataFrame(optimization_trials_data).to_csv(TRIALS_CSV_PATH, index=False, float_format='%.5f')
             print(f"  (Saved intermediate log to {TRIALS_CSV_PATH})")
         except Exception as e: print(f"  (Warning: Failed to save intermediate log: {e})")
    return score

# ========== Main Execution ==========
if __name__ == "__main__":
    start_time_main = time.time()
    print("="*60 + "\n Bayesian Optimization Started\n" + "="*60)

    # --- Run Optimization ---
    result = gp_minimize(func=objective_function, dimensions=param_space,
                         n_calls=N_OPTIMIZATION_CALLS, n_initial_points=N_INITIAL_RANDOM_POINTS,
                         acq_func=ACQUISITION_FUNCTION, random_state=RANDOM_STATE_SEED, verbose=True)

    # --- Process Results ---
    print("\n" + "="*60 + "\n Optimization Finished\n" + "="*60)
    best_score_minimized = result.fun
    best_metric_value_actual = (1.0 - best_score_minimized) if MAXIMIZE_METRIC else best_score_minimized
    print(f"Best score (minimized): {best_score_minimized:.5f}")
    print(f"Best {OPTIMIZATION_TARGET_METRIC.upper()}: {best_metric_value_actual:.5f}")

    best_params_list = result.x
    best_params_dict = {space.name: value for space, value in zip(param_space, best_params_list)}
    # Convert best CLAHE grid back to tuple for printing/use
    tuple_param_name = 'clahe_tile_grid_size'
    if tuple_param_name in best_params_dict:
        grid_str = best_params_dict[tuple_param_name]
        try:
            dims = list(map(int, grid_str.split('x')))
            if len(dims) == 2: best_params_dict[tuple_param_name] = tuple(dims)
        except Exception: best_params_dict[tuple_param_name] = (8, 8) # Fallback

    print("\nBest parameters found:")
    for name, value in best_params_dict.items(): print(f"  {name}: {value}")

    # --- Save Final Trials Log ---
    if optimization_trials_data:
        try:
            trials_df = pd.DataFrame(optimization_trials_data)
            trials_df.to_csv(TRIALS_CSV_PATH, index=False, float_format='%.5f')
            print(f"\nSaved FINAL detailed optimization trials log to: {TRIALS_CSV_PATH}")
        except Exception as e: print(f"ERROR saving final trials log: {e}")

    # --- Evaluate on Validation Set ---
    print("\n" + "="*60 + "\n Evaluating on Validation Set\n" + "="*60 + "\n")
    validation_results = []
    try:
         best_config = PipelineConfig(**best_params_dict, iou_threshold=0.3)
    except Exception as e: print(f"FATAL ERROR creating best config: {e}"); exit()

    mh_val = mm.metrics.create() # Metric host for validation

    for seq_name in VALIDATION_SEQUENCES:
        print(f"--- Processing validation sequence: {seq_name} ---")
        seq_results = {'sequence': seq_name}
        # Add best params to seq_results (converting tuple to string)
        for k, v in best_params_dict.items(): seq_results[f'best_{k}'] = str(v) if isinstance(v, tuple) else v

        start_val_time = time.time(); val_status = 'Unknown'; val_error_msg = None
        try:
            data_val = DatasetLoader(BASE_PATH, seq_name)
            if not data_val.frame_paths: val_error_msg = 'No frames'; val_status = 'Skipped'
            elif data_val.gt_corners is None or not any(gt is not None for gt in data_val.gt_bboxes_with_id):
                 val_error_msg = 'No valid GT'; val_status = 'Skipped (No GT)'
            else:
                pipeline_val = MotionDetectionPipeline(data_val, best_config)
                csv_path_val = os.path.join(OUTPUT_DIR, f"{seq_name}_VALIDATION_{TIMESTAMP}.csv")
                print(f"    Running pipeline (saving CSV: {csv_path_val})...")
                val_frame_updates, _ = pipeline_val.run(save_gif_path=None, save_csv_path=csv_path_val)

                if val_frame_updates:
                    val_acc = mm.MOTAccumulator(auto_id=True)
                    for frame_data in val_frame_updates: val_acc.update(*frame_data)
                    if not val_acc.events.empty:
                         summary_val = calculate_mot_metrics(val_acc) # Prints metrics
                         if not summary_val.empty: seq_results.update(summary_val.iloc[0].to_dict()); val_status = 'Success'
                         else: val_error_msg = 'Metric calc empty summary'; val_status = 'Error'
                    else: val_error_msg = 'Accumulator empty'; val_status = 'Error'
                else: val_error_msg = 'No tracking events'; val_status = 'No Events'
        except Exception as e:
            print(f"    ERROR processing validation sequence {seq_name}: {e}"); traceback.print_exc()
            val_error_msg = str(e)[:200]; val_status = 'Error'
        seq_results['status'] = val_status
        if val_error_msg: seq_results['error'] = val_error_msg
        seq_results['validation_time_seconds'] = time.time() - start_val_time
        validation_results.append(seq_results)
        print("-" * 30)

    # --- Save Validation Results ---
    if validation_results:
        try:
            val_df = pd.DataFrame(validation_results)
            val_df.to_csv(VALIDATION_CSV_PATH, index=False, float_format='%.5f')
            print(f"\nSaved detailed validation results to: {VALIDATION_CSV_PATH}")
        except Exception as e: print(f"ERROR saving validation log: {e}")

    total_script_time = time.time() - start_time_main
    print(f"\nTotal script execution time: {total_script_time:.2f} seconds.")
    print("Script finished.")