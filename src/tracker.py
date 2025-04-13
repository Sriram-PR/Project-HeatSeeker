# File: tracker.py
# Phase 3: Tracking using Simple Kalman Filter and Association (using MOG2 detector)

import os
import cv2
import numpy as np

# --- Attempt to Import from Previous Phases ---
try:
    from data_loader import (DATASET_BASE_PATH,
                             get_sequence_frames,
                             parse_ground_truth,
                             convert_corners_to_bbox,
                             load_frame,
                             preprocess_frame_pipeline,
                             normalize_frame)
except ImportError:
    print("="*60+"\nERROR: Could not import from data_loader.py.\n"+"="*60); exit()

try:
    # Import the MASK PROCESSING function and its default parameters
    from motion_detector import (process_fg_mask,
                                 DEFAULT_MIN_CONTOUR_AREA,
                                 DEFAULT_MAX_CONTOUR_AREA,
                                 DEFAULT_MORPH_KERNEL_SIZE,
                                 DEFAULT_MORPH_OPEN_ITERATIONS,
                                 DEFAULT_MORPH_CLOSE_ITERATIONS)
except ImportError:
    print("="*60+"\nERROR: Could not import from motion_detector.py.\n"+"="*60); exit()

# --- Configuration for Tracking (Tunable Parameters) ---

# Kalman Filter Setup
KF_STATE_DIM = 4
KF_MEASURE_DIM = 4
KF_CONTROL_DIM = 0
KF_PROCESS_NOISE = 1e-2
KF_MEASUREMENT_NOISE = 1e-1
KF_ERROR_COV_POST = 1.0

# Association Parameters
IOU_THRESHOLD = 0.1 # Minimum IoU for association

# MOG2 Parameters (Used within tracker.py now)
MOG2_HISTORY = 150
MOG2_VAR_THRESHOLD = 25
MOG2_DETECT_SHADOWS = False

# --- Utility Functions (setup_kalman_filter, bbox_to_state, state_to_bbox, calculate_iou - Keep these as before) ---
def setup_kalman_filter(state_dim=KF_STATE_DIM, measure_dim=KF_MEASURE_DIM, control_dim=KF_CONTROL_DIM):
    kf = cv2.KalmanFilter(state_dim, measure_dim, control_dim)
    kf.transitionMatrix = np.eye(state_dim, dtype=np.float32)
    kf.measurementMatrix = np.eye(measure_dim, state_dim, dtype=np.float32)
    kf.processNoiseCov = np.eye(state_dim, dtype=np.float32) * KF_PROCESS_NOISE
    kf.measurementNoiseCov = np.eye(measure_dim, dtype=np.float32) * KF_MEASUREMENT_NOISE
    kf.errorCovPost = np.eye(state_dim, dtype=np.float32) * KF_ERROR_COV_POST
    kf.statePost = np.zeros((state_dim, 1), dtype=np.float32)
    return kf

def bbox_to_state(bbox):
    x, y, w, h = bbox
    cx = x + w / 2.0; cy = y + h / 2.0
    return np.array([[cx], [cy], [w], [h]], dtype=np.float32)

def state_to_bbox(state):
    cx, cy, w, h = state.flatten().astype(float)
    x = max(0.0, cx - w / 2.0); y = max(0.0, cy - h / 2.0)
    return int(round(x)), int(round(y)), int(round(w)), int(round(h))

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]; boxBArea = boxB[2] * boxB[3]
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou
# --- End Utility Functions ---


# --- Main Tracking Function ---
def track_sequence(sequence_name, visualize=True):
    """ Performs tracking on a single LTIR sequence using MOG2 + Simple Kalman. """
    print(f"\n--- Starting Tracking for Sequence: {sequence_name} ---")

    # 1. Load Sequence Data
    frame_paths = get_sequence_frames(DATASET_BASE_PATH, sequence_name)
    gt_corners_list = parse_ground_truth(DATASET_BASE_PATH, sequence_name)
    if not frame_paths or gt_corners_list is None or len(frame_paths) != len(gt_corners_list):
        print(f"Error: Data mismatch for {sequence_name}."); return [], []
    ground_truth_bboxes = [convert_corners_to_bbox(c) for c in gt_corners_list]

    # 2. Initialize Kalman Filter & Background Subtractor
    kf = setup_kalman_filter()
    backSub = cv2.createBackgroundSubtractorMOG2(history=MOG2_HISTORY,
                                                 varThreshold=MOG2_VAR_THRESHOLD,
                                                 detectShadows=MOG2_DETECT_SHADOWS)
    print(f"Using MOG2 (Hist={MOG2_HISTORY}, VarThresh={MOG2_VAR_THRESHOLD})")
    print(f"Using KF (ProcNoise={KF_PROCESS_NOISE}, MeasNoise={KF_MEASUREMENT_NOISE}) and IoU Threshold={IOU_THRESHOLD}")


    first_bbox = ground_truth_bboxes[0]
    if first_bbox is None: print("Invalid first GT box."); return [], ground_truth_bboxes
    kf.statePost = bbox_to_state(first_bbox)

    tracked_bboxes = [first_bbox] # Store results

    # 3. Tracking Loop (from frame 2 onwards)
    for i in range(1, len(frame_paths)):
        original_frame, _ = load_frame(frame_paths[i])
        if original_frame is None: tracked_bboxes.append(state_to_bbox(kf.predict())); continue # Predict if load fails

        # --- Predict ---
        predicted_state_raw = kf.predict()
        predicted_bbox = state_to_bbox(predicted_state_raw)

        # --- Detect ---
        preprocessed_frame = preprocess_frame_pipeline(original_frame)
        if preprocessed_frame is None: tracked_bboxes.append(predicted_bbox); continue # Predict if preprocess fails

        # Get foreground mask using MOG2
        fg_mask = backSub.apply(preprocessed_frame)

        # Process the mask to get detections using the imported function
        detected_boxes, _ = process_fg_mask(
            fg_mask,
            min_area=DEFAULT_MIN_CONTOUR_AREA, # Using defaults from motion_detector
            max_area=DEFAULT_MAX_CONTOUR_AREA,
            morph_kernel_size=DEFAULT_MORPH_KERNEL_SIZE,
            open_iterations=DEFAULT_MORPH_OPEN_ITERATIONS,
            close_iterations=DEFAULT_MORPH_CLOSE_ITERATIONS
        )

        # --- Associate ---
        best_match_box = None
        best_iou = IOU_THRESHOLD
        if detected_boxes:
            for det_box in detected_boxes:
                iou = calculate_iou(predicted_bbox, det_box)
                if iou > best_iou:
                    best_iou = iou
                    best_match_box = det_box

        # --- Update ---
        final_bbox_for_frame = None
        if best_match_box is not None:
            measurement = bbox_to_state(best_match_box)
            kf.correct(measurement)
            final_bbox_for_frame = state_to_bbox(kf.statePost)
        else:
            final_bbox_for_frame = predicted_bbox # Use prediction

        tracked_bboxes.append(final_bbox_for_frame)

        # --- Visualization ---
        if visualize:
            display_frame = normalize_frame(original_frame)
            if display_frame is None: continue
            display_frame_color = cv2.cvtColor(display_frame, cv2.COLOR_GRAY2BGR)
            # Draw GT (Green)
            gt_box = ground_truth_bboxes[i];
            if gt_box: cv2.rectangle(display_frame_color, gt_box[:2], (gt_box[0]+gt_box[2], gt_box[1]+gt_box[3]), (0, 255, 0), 1)
            # Draw Detections (Red)
            for dbox in detected_boxes: cv2.rectangle(display_frame_color, dbox[:2], (dbox[0]+dbox[2], dbox[1]+dbox[3]), (0, 0, 255), 1)
             # Draw Prediction (Yellow Circle at center) - Optional change
            # cv2.circle(display_frame_color, (int(predicted_bbox[0]+predicted_bbox[2]/2), int(predicted_bbox[1]+predicted_bbox[3]/2)), 5, (0, 255, 255), -1)
            # Draw Final Tracked Box (Blue)
            if final_bbox_for_frame: cv2.rectangle(display_frame_color, final_bbox_for_frame[:2], (final_bbox_for_frame[0]+final_bbox_for_frame[2], final_bbox_for_frame[1]+final_bbox_for_frame[3]), (255, 0, 0), 2)

            # Add Status Text
            status = f"Update (IoU:{best_iou:.2f})" if best_match_box else "Predict"
            cv2.putText(display_frame_color, f"Frame {i} - {status}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

            cv2.imshow(f"Tracking - {sequence_name}", display_frame_color)
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'): break
            elif key == ord('p'): cv2.waitKey(-1)

    if visualize: cv2.destroyAllWindows()
    print(f"--- Finished Tracking: {sequence_name} ---")
    return tracked_bboxes, ground_truth_bboxes


# --- Example Usage ---
if __name__ == "__main__":
    # Check data_loader path
    if 'PASTE_YOUR_FULL_LTIR_DATASET_PATH_HERE' in DATASET_BASE_PATH or not os.path.isdir(DATASET_BASE_PATH):
         print("="*60+"\n!!! ERROR: Set DATASET_BASE_PATH in data_loader.py !!!\n"+"="*60); exit()

    sequence_to_track = '8_car' # <-- CHANGE THIS TO TEST

    tracked_results, gt_results = track_sequence(sequence_to_track, visualize=True)

    if tracked_results:
        print(f"\nTracking complete. Generated {len(tracked_results)} tracked boxes.")
        # Basic check
        print(f"GT boxes available: {len(gt_results)}")
        # Next phase: Evaluate tracked_results against gt_results
    else:
        print("Tracking failed.")