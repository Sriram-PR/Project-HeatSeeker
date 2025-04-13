import os
import json
import cv2
import numpy as np
import pandas as pd
import itertools # To generate parameter combinations
import time
import motmetrics as mm

# Import your existing modules
from data_loader import (load_coco_annotations, filter_night_images_and_annotations,
                         load_image, preprocess_image, DATASET_BASE_DIR)
from tracker import Tracker # Assuming tracker.py exists

# --- Configuration ---

# --- 1. Define Parameter Grid ---
# Choose parameters and ranges carefully! Grid size grows exponentially.
param_grid = {
    'MOG2_VAR_THRESHOLD': [8, 16, 32],
    'MIN_CONTOUR_AREA': [50, 100, 200],
    'TRACKER_IOU_THRESHOLD': [0.1, 0.2, 0.3],
    # Add more parameters if desired, e.g.:
    'TRACKER_MAX_AGE': [15, 30],
    'TRACKER_MIN_HITS': [2, 3],
    # 'MORPH_KERNEL_SIZE_VAL': [3, 5], # Need to handle tuple creation if tuning kernel size
}

# --- 2. Evaluation Setup ---
# Use the validation set
thermal_val_img_dir = os.path.join(DATASET_BASE_DIR, 'images_thermal_val', 'data')
val_annotation_file = os.path.join(DATASET_BASE_DIR, 'images_thermal_val', 'coco.json')

# Choose the sequence(s) for evaluation
# Using the single long sequence identified earlier
EVAL_SEQUENCE_ID = 'JhYLiFCieHQHaY8o7'

# Choose the optimization metric from motmetrics summary keys
OPTIMIZATION_METRIC = 'idf1' # e.g., 'idf1', 'mota', 'motp'
HIGHER_IS_BETTER = True # Set to False if minimizing (e.g., for FP, FN)

# Constants from previous steps (or make them tunable too)
MOG2_HISTORY = 100
MOG2_DETECT_SHADOWS = False
MORPH_KERNEL_SIZE = (3, 3) # Assuming fixed for now, or handle tuning
MORPH_OPEN_ITERATIONS = 1
MORPH_CLOSE_ITERATIONS = 2
MAX_CONTOUR_AREA = 15000 # Assuming fixed

# --- Helper Functions ---

# Keep group_images_by_video (from run_tracker_and_save.py)
def group_images_by_video(coco_data, night_image_ids, image_id_to_filename):
    # (Same implementation as before)
    video_sequences = {}
    image_id_to_video_id = {}
    print("Grouping night images by video sequence...")
    if 'images' in coco_data:
        for img_entry in coco_data['images']:
            img_id = img_entry.get('id')
            if img_id in night_image_ids:
                extra_info = img_entry.get('extra_info', {})
                video_id = extra_info.get('video_id', 'unknown_video')
                image_id_to_video_id[img_id] = video_id
        for img_id in night_image_ids:
            video_id = image_id_to_video_id.get(img_id)
            filename = image_id_to_filename.get(img_id)
            if video_id and filename:
                if video_id not in video_sequences:
                    video_sequences[video_id] = []
                video_sequences[video_id].append((img_id, filename))
    for video_id in video_sequences:
        try: video_sequences[video_id].sort(key=lambda item: item[1])
        except Exception as e: print(f"Warning: Could not sort frames for video {video_id}. Error: {e}")
    print(f"Grouped into {len(video_sequences)} night video sequences.")
    # Let's use min_seq_length=1 here to ensure our target sequence isn't filtered out
    min_seq_length = 1
    filtered_sequences = {vid: frames for vid, frames in video_sequences.items() if len(frames) >= min_seq_length}
    # print(f"Filtered to {len(filtered_sequences)} sequences with at least {min_seq_length} frames.")
    return filtered_sequences


# Keep prepare_gt_dataframe (from run_tracker_and_save.py)
def prepare_gt_dataframe(image_ids, image_id_to_annotations):
    # (Same implementation as before)
    gt_data = []
    frame_num_map = {img_id: i + 1 for i, img_id in enumerate(image_ids)}
    for img_id in image_ids:
        frame_num = frame_num_map.get(img_id)
        if frame_num is None: continue
        annotations = image_id_to_annotations.get(img_id, [])
        for ann in annotations:
            bbox = ann.get('bbox')
            if bbox and len(bbox) == 4:
                x, y, w, h = [float(c) for c in bbox]
                gt_data.append([frame_num, ann.get('id', -1), x, y, w, h, 1.0, -1, -1, -1])
    gt_df = pd.DataFrame(gt_data, columns=['FrameId', 'Id', 'X', 'Y', 'Width', 'Height', 'Confidence', 'WorldX', 'WorldY', 'WorldZ'])
    # Ensure types are correct for motmetrics processing later
    if not gt_df.empty:
        gt_df[['FrameId', 'Id']] = gt_df[['FrameId', 'Id']].astype(int)
        gt_df[['X', 'Y', 'Width', 'Height', 'Confidence']] = gt_df[['X', 'Y', 'Width', 'Height', 'Confidence']].astype(float)
    return gt_df

# --- NEW: Function to run pipeline and evaluate ONE combination ---
def evaluate_parameter_combination(params, sequence_data, gt_df):
    """
    Runs the detection and tracking pipeline for one set of parameters
    and returns the value of the OPTIMIZATION_METRIC.
    """
    print(f"  Testing params: {params}")
    start_time = time.time()

    # Extract parameters for this run
    var_thresh = params['MOG2_VAR_THRESHOLD']
    min_area = params['MIN_CONTOUR_AREA']
    iou_thresh = params['TRACKER_IOU_THRESHOLD']
    # Add others if tuning: max_age, min_hits, etc.
    max_age = params.get('TRACKER_MAX_AGE', 20) # Use default if not in grid
    min_hits = params.get('TRACKER_MIN_HITS', 3) # Use default if not in grid

    # Initialize Detector and Tracker with current params
    backSub = cv2.createBackgroundSubtractorMOG2(history=MOG2_HISTORY, varThreshold=var_thresh, detectShadows=MOG2_DETECT_SHADOWS)
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_KERNEL_SIZE)
    tracker = Tracker(iou_threshold=iou_thresh, max_age=max_age, min_hits=min_hits)

    tracker_output_data = []

    # Process the sequence
    for frame_index, (image_id, filename) in enumerate(sequence_data):
        frame_num = frame_index + 1
        frame = load_image(thermal_val_img_dir, filename)
        if frame is None: continue
        preprocessed_frame = preprocess_image(frame)
        if preprocessed_frame is None: continue

        # --- Detection ---
        fgMask = backSub.apply(preprocessed_frame)
        fgMask_opened = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, morph_kernel, iterations=MORPH_OPEN_ITERATIONS)
        fgMask_closed = cv2.morphologyEx(fgMask_opened, cv2.MORPH_CLOSE, morph_kernel, iterations=MORPH_CLOSE_ITERATIONS)
        contours, _ = cv2.findContours(fgMask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections_bboxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < MAX_CONTOUR_AREA:
                x, y, w, h = cv2.boundingRect(contour)
                detections_bboxes.append([x, y, w, h])

        # --- Tracking ---
        tracked_objects = tracker.update(detections_bboxes)

        # --- Store Tracker Output ---
        for x_tr, y_tr, w_tr, h_tr, track_id in tracked_objects:
             tracker_output_data.append([frame_num, track_id, float(x_tr), float(y_tr), float(w_tr), float(h_tr), 1.0, -1, -1, -1])

    # --- Evaluation for this run ---
    if not tracker_output_data:
        print("  Warning: No tracker output generated for these parameters.")
        # Return a very bad score if no output
        return -1.0 if HIGHER_IS_BETTER else float('inf')

    tracker_df = pd.DataFrame(tracker_output_data, columns=['FrameId', 'Id', 'X', 'Y', 'Width', 'Height', 'Confidence', 'WorldX', 'WorldY', 'WorldZ'])
    tracker_df[['FrameId', 'Id']] = tracker_df[['FrameId', 'Id']].astype(int)
    tracker_df[['X', 'Y', 'Width', 'Height', 'Confidence']] = tracker_df[['X', 'Y', 'Width', 'Height', 'Confidence']].astype(float)

    # Run motmetrics
    acc = mm.MOTAccumulator(auto_id=True)
    # Need to handle potentially empty tracker_df for certain frames if GT exists but tracker output doesn't
    for frame_id in gt_df['FrameId'].unique():
        gt_frame = gt_df[gt_df['FrameId'] == frame_id]
        tracker_frame = tracker_df[tracker_df['FrameId'] == frame_id]
        gt_ids = gt_frame['Id'].values
        tracker_ids = tracker_frame['Id'].values
        gt_boxes = gt_frame[['X', 'Y', 'Width', 'Height']].values
        tracker_boxes = tracker_frame[['X', 'Y', 'Width', 'Height']].values
        similarity = mm.distances.iou_matrix(gt_boxes, tracker_boxes, max_iou=0.5)
        distance_matrix = 1.0 - similarity
        acc.update(gt_ids, tracker_ids, distance_matrix)

    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=[OPTIMIZATION_METRIC], name='GridSearchRun') # Only compute needed metric

    metric_value = summary[OPTIMIZATION_METRIC].iloc[0] # Get the value
    end_time = time.time()
    print(f"  Finished in {end_time - start_time:.2f}s. {OPTIMIZATION_METRIC}: {metric_value:.4f}")
    return metric_value


# --- Main Grid Search Execution ---
if __name__ == "__main__":
    print("--- Phase 5: Grid Search for Parameter Tuning ---")

    # 1. Load Validation Annotations
    val_coco_data = load_coco_annotations(val_annotation_file)
    if not val_coco_data: exit()
    night_ids_val, id_to_fname_val, id_to_anns_val, _ = filter_night_images_and_annotations(val_coco_data)
    if not night_ids_val: print("\nNo night images found in validation set. Exiting."); exit()

    # 2. Group sequences and select the target sequence data
    night_sequences_val = group_images_by_video(val_coco_data, night_ids_val, id_to_fname_val)
    if EVAL_SEQUENCE_ID not in night_sequences_val:
        print(f"ERROR: Target evaluation sequence ID '{EVAL_SEQUENCE_ID}' not found.")
        exit()
    eval_sequence_data = night_sequences_val[EVAL_SEQUENCE_ID]
    print(f"\nUsing sequence '{EVAL_SEQUENCE_ID}' ({len(eval_sequence_data)} frames) for tuning.")

    # 3. Prepare Ground Truth DataFrame for the target sequence
    sequence_image_ids = [item[0] for item in eval_sequence_data]
    gt_df_eval = prepare_gt_dataframe(sequence_image_ids, id_to_anns_val)
    if gt_df_eval.empty:
        print("ERROR: Could not generate ground truth data for the sequence.")
        exit()
    print(f"Ground truth data prepared for {len(gt_df_eval['FrameId'].unique())} frames.")

    # 4. Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_value_lists = list(param_grid.values())
    all_combinations = list(itertools.product(*param_value_lists))
    total_combinations = len(all_combinations)
    print(f"\nStarting Grid Search with {total_combinations} parameter combinations...")

    # 5. Iterate through combinations and evaluate
    best_score = -float('inf') if HIGHER_IS_BETTER else float('inf')
    best_params = None
    results = []

    for i, combo_values in enumerate(all_combinations):
        current_params = dict(zip(param_names, combo_values))
        print(f"\n[Combination {i+1}/{total_combinations}]")

        # Run evaluation for this combination
        score = evaluate_parameter_combination(current_params, eval_sequence_data, gt_df_eval)
        results.append({'params': current_params, 'score': score})

        # Update best score
        if HIGHER_IS_BETTER:
            if score > best_score:
                best_score = score
                best_params = current_params
        else:
            if score < best_score:
                best_score = score
                best_params = current_params

    # 6. Report Best Results
    print("\n--- Grid Search Complete ---")
    print(f"Optimization Metric: {OPTIMIZATION_METRIC}")
    print(f"Best Score achieved: {best_score:.4f}")
    print("Best Parameter Combination:")
    if best_params:
        for name, value in best_params.items():
            print(f"  {name}: {value}")
    else:
        print("  No successful runs completed.")

    # Optional: Save all results to a file for more analysis
    results_df = pd.DataFrame(results)
    results_df.to_csv('./output/grid_search_results.csv', index=False)
    print("\nFull grid search results saved to ./output/grid_search_results.csv")