"""
Example script to run the modular tracking pipeline on a single sequence.

This script demonstrates how to use the defined classes (Config, DataLoader,
PipelineRunner, etc.) to process one sequence, save outputs (CSV and optionally GIF),
and calculate MOT metrics if ground truth is available. It uses a fixed
set of parameters defined within this script.
"""

import os
import time
import pandas as pd
import motmetrics as mm
import numpy as np
import cv2

print(f"OpenCV Threads: {cv2.getNumThreads()}")

# Import necessary components from the modular files
from config import PipelineConfig
from data_loader_mot import DatasetLoader
from pipeline_runner import MotionDetectionPipeline
from evaluation import calculate_mot_metrics, per_frame_analysis

# --- Configuration for this Example Run ---

# 1. Dataset Information
BASE_PATH = './data/ltir_v1_0_8bit_16bit/'
SEQUENCE_NAME = '8_birds'

# 2. Fixed Pipeline Parameters (Example Values - NOT OPTIMIZED)
#    Use values known to give some results, or defaults.
# EXAMPLE_CONFIG = PipelineConfig(
#     min_blob_area=100,
#     mog2_history=300,
#     mog2_varThreshold=25.0,
#     morph_kernel_size=5,
#     max_misses=6,
#     association_threshold=50.0,
#     Q=0.05,                     # Process noise
#     R=50.0,                     # Measurement noise
#     gaussian_ksize=5,
#     clahe_clip_limit=2.0,
#     clahe_tile_grid_size=(8, 8),
#     iou_threshold=0.3           # IoU threshold for MOT evaluation matching
# )

EXAMPLE_CONFIG = PipelineConfig(
    min_blob_area=493,
    mog2_history=723,
    mog2_varThreshold=99.60038,
    morph_kernel_size=3,
    max_misses=10,
    association_threshold=142.36825,
    Q=0.77262,                     # Process noise
    R=64.25058,                     # Measurement noise
    gaussian_ksize=3,
    clahe_clip_limit=1.57559,
    clahe_tile_grid_size=(8, 8),
    iou_threshold=0.3           # IoU threshold for MOT evaluation matching
)

# 3. Output Settings
OUTPUT_DIR = "output_example"
SAVE_CSV = True
SAVE_GIF = True
SHOW_INLINE_EVERY = None

# --- Main Execution Block ---
if __name__ == "__main__":

    print("="*60)
    print(f" Running Tracking Pipeline Example ")
    print("="*60)
    print(f"Sequence: {SEQUENCE_NAME}")
    print(f"Base Path: {BASE_PATH}")
    print(f"Using Fixed Parameters:")
    for key, value in vars(EXAMPLE_CONFIG).items():
         print(f"  - {key}: {value}")
    print("-" * 60)

    # --- Ensure Output Directory Exists ---
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"Output directory: '{OUTPUT_DIR}'")
    except OSError as e:
        print(f"Error creating output directory {OUTPUT_DIR}: {e}. Outputs might fail.")

    # --- Load Data ---
    print("Loading data...")
    try:
        data_loader = DatasetLoader(BASE_PATH, SEQUENCE_NAME)
    except Exception as e:
        print(f"FATAL ERROR loading data: {e}")
        exit()

    # --- Check Data Validity ---
    if not data_loader.frame_paths:
        print(f"Error: No frames found for sequence {SEQUENCE_NAME}. Exiting.")
        exit()

    # Check if ground truth is available for evaluation
    has_valid_gt = data_loader.gt_corners is not None and any(gt is not None for gt in data_loader.gt_bboxes_with_id)
    if not has_valid_gt:
        print(f"Warning: No valid ground truth found for sequence {SEQUENCE_NAME}. Metrics cannot be calculated.")
    else:
        print(f"Loaded {len(data_loader)} frames. Ground truth found.")

    # --- Initialize and Run Pipeline ---
    print("Initializing pipeline...")
    pipeline = MotionDetectionPipeline(data_loader, EXAMPLE_CONFIG)

    # Define output paths for this specific run
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    gif_output_path = os.path.join(OUTPUT_DIR, f"{SEQUENCE_NAME}_example_{timestamp}.gif") if SAVE_GIF else None
    csv_output_path = os.path.join(OUTPUT_DIR, f"{SEQUENCE_NAME}_example_{timestamp}.csv") if SAVE_CSV else None

    print("Running pipeline...")
    start_run_time = time.time()
    try:
        # Execute the pipeline
        frame_update_data, results_df = pipeline.run(
            show_inline_every=SHOW_INLINE_EVERY,
            save_gif_path=gif_output_path,
            save_csv_path=csv_output_path
        )
    except Exception as e:
        print(f"FATAL ERROR during pipeline run: {e}")
        import traceback
        traceback.print_exc()
        exit()
    end_run_time = time.time()
    print(f"Pipeline finished in {end_run_time - start_run_time:.2f} seconds.")

    # --- Perform Evaluation (if GT was available) ---
    if has_valid_gt:
        if frame_update_data:
            print("\nAccumulating metrics...")
            # Create a new accumulator instance for this run
            mot_accumulator_run = mm.MOTAccumulator(auto_id=True)

            # Update the accumulator with data from the pipeline run
            for frame_data in frame_update_data:
                mot_accumulator_run.update(*frame_data) # Unpack (gt_ids, hyp_ids, dist_matrix)

            # --- Calculate and Display Metrics ---
            if not mot_accumulator_run.events.empty:
                print("Calculating MOT Metrics...")
                metrics_summary_df = calculate_mot_metrics(mot_accumulator_run) # Prints the summary table

                # Optional: Simple analysis based on the CSV data
                if SAVE_CSV and not results_df.empty:
                    print("\n--- Simple Analysis based on CSV output ---")
                    _ = per_frame_analysis(results_df) # Prints basic stats

                # Access and print specific metrics if needed
                if not metrics_summary_df.empty:
                    print(f"\nSpecific Metrics:")
                    mota = metrics_summary_df['mota'].iloc[0] if 'mota' in metrics_summary_df.columns else 'N/A'
                    motp = metrics_summary_df['motp'].iloc[0] if 'motp' in metrics_summary_df.columns else 'N/A'
                    idf1 = metrics_summary_df['idf1'].iloc[0] if 'idf1' in metrics_summary_df.columns else 'N/A'
                    print(f"  MOTA: {mota:.3f}" if isinstance(mota, (float, int)) else f"  MOTA: {mota}")
                    print(f"  MOTP: {motp:.3f}" if isinstance(motp, (float, int)) else f"  MOTP: {motp}")
                    print(f"  IDF1: {idf1:.3f}" if isinstance(idf1, (float, int)) else f"  IDF1: {idf1}")
                else:
                    print("Metrics summary DataFrame was empty.")
            else:
                 print("\nNo tracking events recorded by accumulator. Cannot compute metrics.")
        else:
            print("\nPipeline run did not produce frame data for metric accumulation (or GT was missing).")
    else:
        print("\nSkipped MOT metrics calculation as no valid ground truth was found.")
        if SAVE_CSV and not results_df.empty:
             print("\n--- Simple Analysis based on CSV output (Tracks Only) ---")
             _ = per_frame_analysis(results_df)


    print("\n" + "="*60)
    print(" Example Run Script Finished")
    print("="*60)