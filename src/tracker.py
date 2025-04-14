# File: tracker.py (Refactored Version)
# Phase 3: Tracking using Simple Kalman Filter and Association (using MOG2 detector)

import os
import cv2
import numpy as np

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
    from motion_detector import process_fg_mask
except ImportError:
    print("="*60+"\nERROR: Could not import from motion_detector.py.\n"+"="*60); exit()

# --- Default Configuration Parameters (used if not overridden by function arguments) ---
# These are now defaults for the track_sequence function
DEFAULT_MOG2_HISTORY = 150
DEFAULT_MOG2_VAR_THRESHOLD = 25
DEFAULT_MOG2_DETECT_SHADOWS = False

DEFAULT_MIN_CONTOUR_AREA = 75
DEFAULT_MAX_CONTOUR_AREA = 20000
DEFAULT_MORPH_KERNEL_SIZE = (3, 3)
DEFAULT_MORPH_OPEN_ITERATIONS = 1
DEFAULT_MORPH_CLOSE_ITERATIONS = 2

DEFAULT_KF_STATE_DIM = 4
DEFAULT_KF_MEASURE_DIM = 4
DEFAULT_KF_CONTROL_DIM = 0
DEFAULT_KF_PROCESS_NOISE = 1e-2
DEFAULT_KF_MEASUREMENT_NOISE = 1e-1
DEFAULT_KF_ERROR_COV_POST = 1.0

DEFAULT_IOU_THRESHOLD = 0.1 # Minimum IoU for association

# --- Utility Functions (setup_kalman_filter, bbox_to_state, state_to_bbox, calculate_iou) ---
# Modified setup_kalman_filter to accept noise parameters
def setup_kalman_filter(state_dim=DEFAULT_KF_STATE_DIM,
                        measure_dim=DEFAULT_KF_MEASURE_DIM,
                        control_dim=DEFAULT_KF_CONTROL_DIM,
                        process_noise=DEFAULT_KF_PROCESS_NOISE,
                        measurement_noise=DEFAULT_KF_MEASUREMENT_NOISE,
                        error_cov_post=DEFAULT_KF_ERROR_COV_POST):
    """Initializes Kalman Filter with specified noise parameters."""
    kf = cv2.KalmanFilter(state_dim, measure_dim, control_dim)
    kf.transitionMatrix = np.eye(state_dim, dtype=np.float32)
    kf.measurementMatrix = np.eye(measure_dim, state_dim, dtype=np.float32)
    kf.processNoiseCov = np.eye(state_dim, dtype=np.float32) * process_noise
    kf.measurementNoiseCov = np.eye(measure_dim, dtype=np.float32) * measurement_noise
    kf.errorCovPost = np.eye(state_dim, dtype=np.float32) * error_cov_post
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
    # Handle None inputs
    if boxA is None or boxB is None: return 0.0
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]; boxBArea = boxB[2] * boxB[3]
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6) # Add epsilon for stability
    return iou
# --- End Utility Functions ---


# --- Main Tracking Function (Refactored to accept parameters) ---
def track_sequence(sequence_name,
                   visualize=False,
                   # --- Parameters to Tune ---
                   mog2_history=DEFAULT_MOG2_HISTORY,
                   mog2_var_threshold=DEFAULT_MOG2_VAR_THRESHOLD,
                   kf_process_noise=DEFAULT_KF_PROCESS_NOISE,
                   kf_measurement_noise=DEFAULT_KF_MEASUREMENT_NOISE,
                   iou_threshold=DEFAULT_IOU_THRESHOLD,
                   min_contour_area=DEFAULT_MIN_CONTOUR_AREA,
                   max_contour_area=DEFAULT_MAX_CONTOUR_AREA,
                   morph_kernel_size=DEFAULT_MORPH_KERNEL_SIZE,
                   morph_open_iter=DEFAULT_MORPH_OPEN_ITERATIONS,
                   morph_close_iter=DEFAULT_MORPH_CLOSE_ITERATIONS,
                   # --- Other parameters (could be tuned too) ---
                   kf_error_cov_post=DEFAULT_KF_ERROR_COV_POST,
                   mog2_detect_shadows=DEFAULT_MOG2_DETECT_SHADOWS
                  ):
    """
    Performs tracking on a single LTIR sequence using MOG2 + Simple Kalman.
    Accepts key parameters for tuning.
    """
    # Use passed-in parameters or defaults
    print(f"\n--- Tracking {sequence_name} with Params ---")
    print(f"  MOG2: Hist={mog2_history}, VarThresh={mog2_var_threshold}")
    print(f"  KF: ProcNoise={kf_process_noise:.1e}, MeasNoise={kf_measurement_noise:.1e}")
    print(f"  Detect: Area=[{min_contour_area}-{max_contour_area}], IoU Thresh={iou_threshold}")
    print(f"  Morph: Kernel={morph_kernel_size}, Open={morph_open_iter}, Close={morph_close_iter}")
    # --------------------------------------------------

    # 1. Load Sequence Data
    frame_paths = get_sequence_frames(DATASET_BASE_PATH, sequence_name)
    gt_corners_list = parse_ground_truth(DATASET_BASE_PATH, sequence_name)
    if not frame_paths or gt_corners_list is None or len(frame_paths) != len(gt_corners_list):
        print(f"Error: Data mismatch for {sequence_name}. Skipping."); return [], []
    ground_truth_bboxes = [convert_corners_to_bbox(c) for c in gt_corners_list]

    # 2. Initialize Kalman Filter & Background Subtractor with specified parameters
    kf = setup_kalman_filter(
        process_noise=kf_process_noise,
        measurement_noise=kf_measurement_noise,
        error_cov_post=kf_error_cov_post
    )
    backSub = cv2.createBackgroundSubtractorMOG2(
        history=mog2_history,
        varThreshold=mog2_var_threshold,
        detectShadows=mog2_detect_shadows
    )

    first_bbox = ground_truth_bboxes[0]
    if first_bbox is None or first_bbox[2] <= 0 or first_bbox[3] <= 0: # Check for valid first box
        print(f"Error: Invalid first GT box for {sequence_name}: {first_bbox}. Skipping."); return [], ground_truth_bboxes
    kf.statePost = bbox_to_state(first_bbox)

    tracked_bboxes = [first_bbox] # Store results

    # 3. Tracking Loop (from frame 2 onwards)
    for i in range(1, len(frame_paths)):
        original_frame, _ = load_frame(frame_paths[i])
        if original_frame is None:
            # Predict if load fails, append predicted box
            predicted_state_raw = kf.predict()
            predicted_bbox = state_to_bbox(predicted_state_raw)
            tracked_bboxes.append(predicted_bbox)
            continue

        # --- Predict ---
        predicted_state_raw = kf.predict()
        predicted_bbox = state_to_bbox(predicted_state_raw)

        # --- Detect ---
        # Preprocessing (using defaults for now, could be params too)
        preprocessed_frame = preprocess_frame_pipeline(original_frame)
        if preprocessed_frame is None:
            tracked_bboxes.append(predicted_bbox) # Predict if preprocess fails
            continue

        # Get foreground mask using MOG2 (initialized with specific params)
        fg_mask = backSub.apply(preprocessed_frame)

        # Process the mask using the imported function and specified parameters
        detected_boxes, _ = process_fg_mask(
            fg_mask,
            min_area=min_contour_area,
            max_area=max_contour_area,
            morph_kernel_size=morph_kernel_size,
            open_iterations=morph_open_iter,
            close_iterations=morph_close_iter
        )

        # --- Associate ---
        best_match_box = None
        best_iou = iou_threshold # Use the passed-in threshold
        if detected_boxes:
            # Ensure predicted_bbox has positive width/height before IoU calculation
            if predicted_bbox[2] > 0 and predicted_bbox[3] > 0:
                for det_box in detected_boxes:
                     # Ensure detection has positive w/h
                     if det_box[2] > 0 and det_box[3] > 0:
                         iou = calculate_iou(predicted_bbox, det_box)
                         if iou > best_iou:
                             best_iou = iou
                             best_match_box = det_box
            # else:
                 # print(f"Warning frame {i}: Predicted bbox has zero area: {predicted_bbox}")

        # --- Update ---
        final_bbox_for_frame = None
        if best_match_box is not None:
            measurement = bbox_to_state(best_match_box)
            kf.correct(measurement)
            final_bbox_for_frame = state_to_bbox(kf.statePost)
        else:
            # No good match found, use prediction
            kf.statePost = predicted_state_raw # Reset state to prediction if no update
            final_bbox_for_frame = predicted_bbox

        # Ensure final box has positive dimensions
        if final_bbox_for_frame[2] <= 0 or final_bbox_for_frame[3] <= 0:
             # print(f"Warning frame {i}: Final bbox has zero area: {final_bbox_for_frame}. Using prediction.")
             # Fallback to predicted box if correction resulted in invalid box
             kf.statePost = predicted_state_raw
             final_bbox_for_frame = predicted_bbox
             # If predicted is also invalid, this might indicate a deeper issue.
             # For now, we store it, but evaluation (IoU/CLE) will handle it appropriately.


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
            # Draw Final Tracked Box (Blue)
            if final_bbox_for_frame: cv2.rectangle(display_frame_color, final_bbox_for_frame[:2], (final_bbox_for_frame[0]+final_bbox_for_frame[2], final_bbox_for_frame[1]+final_bbox_for_frame[3]), (255, 0, 0), 2)

            status = f"Update (IoU:{best_iou:.2f})" if best_match_box else "Predict"
            cv2.putText(display_frame_color, f"Frame {i} - {status}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

            cv2.imshow(f"Tracking - {sequence_name}", display_frame_color)
            key = cv2.waitKey(1) & 0xFF # Reduce delay for tuning runs
            if key == ord('q'):
                 visualize = False # Stop visualizing if q is pressed, but continue sequence
                 cv2.destroyAllWindows()
            # elif key == ord('p'): cv2.waitKey(-1) # Pause removed for tuning speed

    if visualize: cv2.destroyAllWindows()
    # print(f"--- Finished Tracking: {sequence_name} ---") # Make less verbose for tuning
    return tracked_bboxes, ground_truth_bboxes


# --- Example Usage ---
if __name__ == "__main__":
    if 'PASTE_YOUR_FULL_LTIR_DATASET_PATH_HERE' in DATASET_BASE_PATH or not os.path.isdir(DATASET_BASE_PATH):
         print("="*60+"\n!!! ERROR: Set DATASET_BASE_PATH in data_loader.py !!!\n"+"="*60); exit()

    sequence_to_track = '8_car'
    print(f"--- Running tracker.py standalone demo on {sequence_to_track} ---")
    tracked_results, gt_results = track_sequence(sequence_to_track, visualize=True)

    if tracked_results:
        print(f"\nTracking complete. Generated {len(tracked_results)} tracked boxes.")
    else:
        print("Tracking failed.")