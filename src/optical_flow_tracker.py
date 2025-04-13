# File: optical_flow_tracker.py
# Phase 3/Comparison: Tracking using Dense Optical Flow Detection + Centroid Association

import os
import cv2
import numpy as np

# --- Attempt to Import from Phase 1 ---
try:
    from data_loader import (DATASET_BASE_PATH,
                             get_sequence_frames,
                             parse_ground_truth,
                             convert_corners_to_bbox,
                             load_frame,
                             preprocess_frame_pipeline, # Reuse preprocessing
                             normalize_frame)
except ImportError:
    print("="*60+"\nERROR: Could not import from data_loader.py.\n"+"="*60); exit()

# --- Configuration (Tunable Parameters) ---

# Farneback Optical Flow Parameters
# See OpenCV docs for details - defaults are often okay to start
# pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags
FARNEBACK_PARAMS = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3,
                        poly_n=5, poly_sigma=1.2, flags=0)

# Motion Mask Thresholding
FLOW_MAGNITUDE_THRESHOLD = 1.0 # Minimum flow magnitude to consider as motion. CRUCIAL TO TUNE.

# Morphological Operations (for flow mask)
FLOW_MORPH_KERNEL_SIZE = (5, 5) # Might need larger kernel for flow
FLOW_MORPH_OPEN_ITERATIONS = 1
FLOW_MORPH_CLOSE_ITERATIONS = 3 # May need more closing

# Contour Filtering (for flow mask)
FLOW_MIN_CONTOUR_AREA = 100 # Adjust based on flow blob sizes
FLOW_MAX_CONTOUR_AREA = 20000 # Adjust

# Centroid Tracking Parameters
MAX_CENTROID_DISTANCE = 50 # Max distance (pixels) to associate centroid. TUNE THIS.
TRACK_MEMORY = 10 # Number of frames to remember a lost track's position.

# --- Helper Functions ---

def calculate_flow_magnitude_angle(flow):
    """ Calculates magnitude and angle of dense optical flow. """
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return magnitude, angle

def calculate_centroid(bbox):
    """ Calculates the center (cx, cy) of a bbox [x, y, w, h]. """
    x, y, w, h = bbox
    return (x + w // 2, y + h // 2)

# --- Core Tracking Function ---

# Store active tracks between frames for centroid tracking
# Simple dict: {track_id: {'bbox': [x,y,w,h], 'centroid': (cx,cy), 'age': 0, 'last_frame': 0}}
active_centroid_tracks = {}
next_centroid_track_id = 1

def track_sequence_optical_flow(sequence_name, visualize=True):
    """
    Performs tracking using Dense Optical Flow (Farneback) based detection
    and simple centroid association.
    """
    global active_centroid_tracks, next_centroid_track_id # Use global for simplicity in example
    active_centroid_tracks = {} # Reset for each sequence
    next_centroid_track_id = 1

    print(f"\n--- Starting Optical Flow Tracking: {sequence_name} ---")

    # 1. Load Sequence Data
    frame_paths = get_sequence_frames(DATASET_BASE_PATH, sequence_name)
    gt_corners_list = parse_ground_truth(DATASET_BASE_PATH, sequence_name)
    if not frame_paths or gt_corners_list is None or len(frame_paths) != len(gt_corners_list):
        print(f"Error: Data mismatch for {sequence_name}."); return [], [] # Return empty lists on error
    ground_truth_bboxes = [convert_corners_to_bbox(c) for c in gt_corners_list]

    tracked_bboxes_output = [] # Store the final bbox for the *main* target
    prev_gray_frame = None
    frame_count = 0

    # 2. Tracking Loop
    for i in range(len(frame_paths)):
        frame_count = i
        original_frame, _ = load_frame(frame_paths[i])
        if original_frame is None: continue

        # Preprocess frame (important for flow stability)
        # Using slightly different params might be beneficial for flow, e.g., less aggressive CLAHE
        preprocessed_gray = preprocess_frame_pipeline(original_frame,
                                                      median_ksize=5, # Maybe more blur?
                                                      clahe_clip_limit=1.5) # Less aggressive contrast?
        if preprocessed_gray is None: continue

        current_tracked_centroids = {} # Centroids tracked in this frame {track_id: (cx, cy)}

        # --- Calculate Optical Flow (if we have a previous frame) ---
        if prev_gray_frame is not None:
            # Calculate Dense Optical Flow
            flow = cv2.calcOpticalFlowFarneback(prev_gray_frame, preprocessed_gray,
                                                None, **FARNEBACK_PARAMS)
            magnitude, _ = calculate_flow_magnitude_angle(flow)

            # --- Detect Motion Blobs from Flow ---
            # Threshold the magnitude
            motion_mask = cv2.threshold(magnitude, FLOW_MAGNITUDE_THRESHOLD, 255, cv2.THRESH_BINARY)[1]
            motion_mask = motion_mask.astype(np.uint8) # Convert to uint8

            # Morphological Operations
            flow_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, FLOW_MORPH_KERNEL_SIZE)
            mask_opened = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, flow_kernel, iterations=FLOW_MORPH_OPEN_ITERATIONS)
            mask_closed = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, flow_kernel, iterations=FLOW_MORPH_CLOSE_ITERATIONS)

            # Find Contours
            contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Get Centroids of detected motion blobs
            detected_centroids = []
            detected_flow_bboxes = [] # Store bboxes for association reference
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if FLOW_MIN_CONTOUR_AREA < area < FLOW_MAX_CONTOUR_AREA:
                    x, y, w, h = cv2.boundingRect(cnt)
                    cx = x + w // 2
                    cy = y + h // 2
                    detected_centroids.append((cx, cy))
                    detected_flow_bboxes.append([x, y, w, h])

            # --- Simple Centroid Association ---
            matched_detection_indices = set()
            current_track_ids = list(active_centroid_tracks.keys())

            # Match existing tracks to detections
            if detected_centroids and current_track_ids:
                 dist_matrix = np.zeros((len(current_track_ids), len(detected_centroids)))
                 for t_idx, track_id in enumerate(current_track_ids):
                     track_centroid = active_centroid_tracks[track_id]['centroid']
                     for d_idx, det_centroid in enumerate(detected_centroids):
                          dist = np.linalg.norm(np.array(track_centroid) - np.array(det_centroid))
                          dist_matrix[t_idx, d_idx] = dist

                 # Simple greedy matching (nearest neighbor below threshold)
                 # Could use Hungarian here too if desired
                 assigned_dets = set()
                 sorted_track_indices = np.argsort([active_centroid_tracks[tid]['age'] for tid in current_track_ids]) # Prioritize older tracks

                 for t_idx in sorted_track_indices:
                     track_id = current_track_ids[t_idx]
                     # Find nearest valid detection for this track
                     best_dist = MAX_CENTROID_DISTANCE
                     best_d_idx = -1
                     for d_idx in range(len(detected_centroids)):
                         if d_idx not in assigned_dets and dist_matrix[t_idx, d_idx] < best_dist:
                             best_dist = dist_matrix[t_idx, d_idx]
                             best_d_idx = d_idx

                     if best_d_idx != -1:
                         # Match found: Update track
                         track = active_centroid_tracks[track_id]
                         track['centroid'] = detected_centroids[best_d_idx]
                         track['bbox'] = detected_flow_bboxes[best_d_idx] # Update bbox too
                         track['age'] = 0 # Reset age
                         track['last_frame'] = frame_count
                         current_tracked_centroids[track_id] = track['centroid']
                         assigned_dets.add(best_d_idx)
                         matched_detection_indices.add(best_d_idx)

            # Increment age for unmatched tracks
            for track_id in current_track_ids:
                 if track_id not in current_tracked_centroids: # If not matched this frame
                      active_centroid_tracks[track_id]['age'] += 1

            # Remove old tracks
            track_ids_to_remove = [tid for tid, track in active_centroid_tracks.items() if track['age'] > TRACK_MEMORY]
            for tid in track_ids_to_remove:
                 # print(f"Removing old track {tid}")
                 del active_centroid_tracks[tid]

            # Add new tracks for unmatched detections
            for d_idx, centroid in enumerate(detected_centroids):
                 if d_idx not in matched_detection_indices:
                     new_id = next_centroid_track_id
                     active_centroid_tracks[new_id] = {
                         'centroid': centroid,
                         'bbox': detected_flow_bboxes[d_idx],
                         'age': 0,
                         'last_frame': frame_count
                     }
                     current_tracked_centroids[new_id] = centroid
                     # print(f"Creating new track {new_id}")
                     next_centroid_track_id += 1

        # --- Prepare Output for SOT Evaluation ---
        # Find the track closest to the Ground Truth for this frame
        # (This simulates identifying the primary target in an MOT scenario for SOT evaluation)
        current_gt_box = ground_truth_bboxes[i]
        final_bbox_for_frame = None
        if current_gt_box and active_centroid_tracks:
            gt_centroid = calculate_centroid(current_gt_box)
            min_dist_to_gt = float('inf')
            best_track_id = -1

            for track_id, track_data in active_centroid_tracks.items():
                 # Only consider tracks updated recently? Optional.
                 # if track_data['last_frame'] == frame_count:
                 dist = np.linalg.norm(np.array(gt_centroid) - np.array(track_data['centroid']))
                 if dist < min_dist_to_gt:
                     min_dist_to_gt = dist
                     best_track_id = track_id

            if best_track_id != -1 : #and min_dist_to_gt < MAX_CENTROID_DISTANCE * 2 : # Optional distance check
                final_bbox_for_frame = active_centroid_tracks[best_track_id]['bbox']

        tracked_bboxes_output.append(final_bbox_for_frame) # Append best guess or None


        # --- Visualization ---
        if visualize:
            display_frame_norm = normalize_frame(original_frame)
            if display_frame_norm is None: continue
            display_frame_color = cv2.cvtColor(display_frame_norm, cv2.COLOR_GRAY2BGR)

            # Draw GT
            if current_gt_box: cv2.rectangle(display_frame_color, current_gt_box[:2], (current_gt_box[0]+current_gt_box[2], current_gt_box[1]+current_gt_box[3]), (0, 255, 0), 1)

            # Draw all active tracks
            for track_id, track_data in active_centroid_tracks.items():
                 x, y, w, h = [int(c) for c in track_data['bbox']]
                 cx, cy = [int(c) for c in track_data['centroid']]
                 color = (255, 0, 255) # Magenta for tracks
                 cv2.rectangle(display_frame_color, (x, y), (x+w, y+h), color, 1)
                 cv2.putText(display_frame_color, str(track_id), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Highlight the chosen output box (if any)
            if final_bbox_for_frame:
                 x,y,w,h = final_bbox_for_frame
                 cv2.rectangle(display_frame_color, (x, y), (x+w, y+h), (255, 0, 0), 2) # Blue thick


            cv2.putText(display_frame_color, f"Frame {i} (#Trks: {len(active_centroid_tracks)})", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            cv2.imshow(f"Optical Flow Tracking - {sequence_name}", display_frame_color)

            # Also show flow mask (optional)
            # if 'mask_closed' in locals() and mask_closed is not None:
            #      cv2.imshow("Flow Motion Mask", mask_closed)

            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'): break
            elif key == ord('p'): cv2.waitKey(-1)


        # Update previous frame
        prev_gray_frame = preprocessed_gray.copy()


    if visualize: cv2.destroyAllWindows()
    print(f"--- Finished Optical Flow Tracking: {sequence_name} ---")
    return tracked_bboxes_output, ground_truth_bboxes


# --- Main Execution ---
if __name__ == "__main__":
    print("--- Optical Flow Tracker Demonstration ---")
    # Verify data path is set
    if 'PASTE_YOUR_FULL_LTIR_DATASET_PATH_HERE' in DATASET_BASE_PATH or not os.path.isdir(DATASET_BASE_PATH):
         print("="*60+"\n!!! ERROR: Set DATASET_BASE_PATH in data_loader.py !!!\n"+"="*60); exit()

    sequence_to_track = '8_car' # Try '8_crossing', '8_selma'

    # Run the Optical Flow tracker
    # Note: This currently doesn't pass detector params, uses defaults inside function
    # Modify if needed to pass flow_magnitude_threshold, etc.
    tracked_results_of, gt_results_of = track_sequence_optical_flow(sequence_to_track,
                                                                    visualize=True)

    # Basic Output (Evaluation comes next by comparing this to Kalman tracker results)
    if tracked_results_of:
        print(f"\nOptical Flow tracking complete for {sequence_to_track}.")
        print(f"Generated {len(tracked_results_of)} tracked bounding boxes.")
        # You would save these results and compare them to the Kalman tracker's results in the evaluation phase.
    else:
        print("Optical Flow tracking failed or produced no results.")