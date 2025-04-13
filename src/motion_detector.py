# # File: motion_detector.py
# # Phase 2: Motion Detection and Object Segmentation using MOG2 Background Subtraction

# import os
# import cv2
# import numpy as np
# import time # To add delay for visualization

# # --- Attempt to Import from Phase 1 ---
# try:
#     from data_loader import (DATASET_BASE_PATH,
#                              get_sequence_frames,
#                              load_frame,
#                              preprocess_frame_pipeline,
#                              normalize_frame) # Added normalize_frame for display
# except ImportError:
#     print("="*60)
#     print("ERROR: Could not import functions from data_loader.py.")
#     print("Please ensure 'data_loader.py' exists in the same directory as this script.")
#     print("="*60)
#     exit()

# # --- Configuration for Motion Detection (Tunable Parameters) ---
# # Background Subtractor Params (MOG2)
# MOG2_HISTORY = 150        # Number of frames used to build the background model. Longer history adapts slower.
# MOG2_VAR_THRESHOLD = 25   # Threshold on the squared Mahalanobis distance. Lower = more sensitive. CRUCIAL TO TUNE.
# MOG2_DETECT_SHADOWS = False # Shadows are not distinct in thermal, set to False.

# # Morphological Operations Params
# MORPH_KERNEL_SIZE = (3, 3) # Kernel size for Opening/Closing. (3,3) or (5,5) are common.
# MORPH_OPEN_ITERATIONS = 1  # Iterations for Morphological Opening (removes noise). 1 is usually enough.
# MORPH_CLOSE_ITERATIONS = 2 # Iterations for Morphological Closing (fills gaps). 1 or 2 are common.

# # Contour Filtering Params
# MIN_CONTOUR_AREA = 75     # Minimum pixel area to consider a contour. CRUCIAL TO TUNE based on target size and distance.
# MAX_CONTOUR_AREA = 20000  # Maximum pixel area. Helps filter out huge changes/scene flashes. CRUCIAL TO TUNE.


# # --- Main Execution ---
# if __name__ == "__main__":
#     print("--- Phase 2: Motion Detection and Segmentation (MOG2 for LTIR) ---")

#     # 1. Verify DATASET_BASE_PATH is set in data_loader.py
#     #    (Basic check, user must ensure the path is correct within data_loader.py)
#     if 'PASTE_YOUR_FULL_LTIR_DATASET_PATH_HERE' in DATASET_BASE_PATH or not os.path.isdir(DATASET_BASE_PATH):
#          print("="*60)
#          print("!!! ERROR: Please set the DATASET_BASE_PATH variable   !!!")
#          print("!!! correctly inside the 'data_loader.py' file first. !!!")
#          print("="*60)
#          exit()

#     # 2. Select an LTIR Sequence to Process
#     #    You will later loop through your Tuning/Test sets here.
#     #    Available 8-bit examples: '8_car', '8_crossing', '8_crouching', '8_trees', '8_selma'
#     selected_sequence_name = '8_car' # <-- CHANGE THIS TO TEST DIFFERENT SEQUENCES
#     print(f"\nSelected LTIR sequence for demonstration: {selected_sequence_name}")

#     frame_paths = get_sequence_frames(DATASET_BASE_PATH, selected_sequence_name)
#     if not frame_paths:
#         print(f"Could not find frames for sequence '{selected_sequence_name}'. Exiting.")
#         exit()
#     print(f"Processing {len(frame_paths)} frames...")

#     # 3. Initialize Background Subtractor
#     print(f"Initializing MOG2 (History={MOG2_HISTORY}, VarThreshold={MOG2_VAR_THRESHOLD})")
#     backSub = cv2.createBackgroundSubtractorMOG2(history=MOG2_HISTORY,
#                                                  varThreshold=MOG2_VAR_THRESHOLD,
#                                                  detectShadows=MOG2_DETECT_SHADOWS)

#     # 4. Initialize Morphological Kernel
#     morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_KERNEL_SIZE)
#     print(f"Using Morph Kernel Size: {MORPH_KERNEL_SIZE}, Open Iter: {MORPH_OPEN_ITERATIONS}, Close Iter: {MORPH_CLOSE_ITERATIONS}")

#     # 5. Process the Selected Sequence Frame by Frame
#     print(f"Contour Area Filter: Min={MIN_CONTOUR_AREA}, Max={MAX_CONTOUR_AREA}")
#     print("\nStarting frame processing...")
#     print("Press 'q' to quit, 'p' to pause/resume during visualization.")

#     for i, frame_path in enumerate(frame_paths):
#         # Load the original frame using LTIR loader
#         original_frame, bit_depth = load_frame(frame_path)
#         if original_frame is None:
#             print(f"Warning: Skipping frame {os.path.basename(frame_path)} (index {i}) - could not load.")
#             continue

#         # Preprocess the frame (using LTIR compatible function)
#         # These preprocessing params can also be tuned. Using defaults from data_loader example:
#         preprocessed_frame = preprocess_frame_pipeline(original_frame,
#                                                        median_ksize=3,
#                                                        clahe_clip_limit=2.0,
#                                                        clahe_tile_grid_size=(8,8))
#         if preprocessed_frame is None:
#             print(f"Warning: Preprocessing failed for frame {i}.")
#             continue # Skip frame if preprocessing fails

#         # Apply Background Subtraction
#         # The apply method updates the background model internally
#         fgMask = backSub.apply(preprocessed_frame)

#         # Apply Morphological Operations to clean the mask
#         fgMask_opened = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, morph_kernel, iterations=MORPH_OPEN_ITERATIONS)
#         fgMask_closed = cv2.morphologyEx(fgMask_opened, cv2.MORPH_CLOSE, morph_kernel, iterations=MORPH_CLOSE_ITERATIONS)
#         # Now fgMask_closed is the cleaned mask we use for contour detection

#         # Find Contours on the cleaned mask
#         contours, _ = cv2.findContours(fgMask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#         # Filter Contours and get Bounding Boxes
#         detected_boxes = []
#         # Ensure original frame is 8-bit normalized for display before converting to color
#         display_original_norm = normalize_frame(original_frame)
#         if display_original_norm is None: continue # Skip if normalization failed

#         frame_display = cv2.cvtColor(display_original_norm, cv2.COLOR_GRAY2BGR) # For drawing detections

#         for contour in contours:
#             area = cv2.contourArea(contour)
#             # Filter by area based on defined parameters
#             if MIN_CONTOUR_AREA < area < MAX_CONTOUR_AREA:
#                 x, y, w, h = cv2.boundingRect(contour)
#                 detected_boxes.append((x, y, w, h))
#                 # Draw the bounding box (e.g., in Red)
#                 cv2.rectangle(frame_display, (x, y), (x + w, y + h), (0, 0, 255), 2) # Red box, thickness 2

#         # --- Visualization ---
#         # Prepare other frames for the 2x2 display
#         fgMask_display = cv2.cvtColor(fgMask, cv2.COLOR_GRAY2BGR) # Raw MOG2 output
#         fgMask_closed_display = cv2.cvtColor(fgMask_closed, cv2.COLOR_GRAY2BGR) # Cleaned mask
#         preprocessed_display = cv2.cvtColor(preprocessed_frame, cv2.COLOR_GRAY2BGR) # Input to MOG2

#         # Add text labels to each quadrant
#         label_font = cv2.FONT_HERSHEY_SIMPLEX
#         label_scale = 0.5
#         label_color = (0, 255, 0) # Green
#         label_thickness = 1
#         cv2.putText(frame_display, f"Orig+Detect ({len(detected_boxes)}) Fr {i}", (10, 20), label_font, label_scale, label_color, label_thickness)
#         cv2.putText(fgMask_display, "Raw FG Mask", (10, 20), label_font, label_scale, label_color, label_thickness)
#         cv2.putText(preprocessed_display, "Preprocessed In", (10, 20), label_font, label_scale, label_color, label_thickness)
#         cv2.putText(fgMask_closed_display, "Cleaned FG Mask", (10, 20), label_font, label_scale, label_color, label_thickness)

#         # Check if dimensions match before stacking (should usually be the same)
#         if frame_display.shape == fgMask_display.shape == preprocessed_display.shape == fgMask_closed_display.shape:
#             top_row = np.hstack((frame_display, fgMask_display))
#             bottom_row = np.hstack((preprocessed_display, fgMask_closed_display))
#             combined_display = np.vstack((top_row, bottom_row))

#             # Resize for better screen fit if images are large (optional)
#             max_disp_w = 1280
#             max_disp_h = 720
#             h_orig, w_orig = combined_display.shape[:2]
#             scale = min(max_disp_w / w_orig, max_disp_h / h_orig)
#             if scale < 1.0:
#                 new_w = int(w_orig * scale)
#                 new_h = int(h_orig * scale)
#                 combined_display = cv2.resize(combined_display, (new_w, new_h), interpolation=cv2.INTER_AREA)

#             cv2.imshow('Motion Detection Results (MOG2)', combined_display)
#         else:
#              print(f"Warning: Frame dimension mismatch at frame {i}, cannot display combined view.")
#              # Fallback: display just the detections
#              cv2.imshow('Detections', frame_display)


#         # --- Frame Delay and User Input ---
#         key = cv2.waitKey(10) & 0xFF # Display frame for ~10ms, check key
#         if key == ord('q'): # Quit if 'q' is pressed
#             print("\n'q' pressed. Exiting visualization loop.")
#             break
#         elif key == ord('p'): # Pause if 'p' is pressed
#             print("\nPaused. Press any key in the OpenCV window to continue...")
#             cv2.waitKey(-1) # Wait indefinitely

#     # --- Cleanup ---
#     cv2.destroyAllWindows()
#     print("\nFinished processing sequence.")

# File: motion_detector.py
# Phase 2: Provides function to process MOG2 foreground mask and detect objects.

import os
import cv2
import numpy as np
import time

# --- Attempt to Import from Phase 1 ---
try:
    from data_loader import (DATASET_BASE_PATH,
                             get_sequence_frames,
                             load_frame,
                             preprocess_frame_pipeline,
                             normalize_frame)
except ImportError:
    print("="*60+"\nERROR: Could not import from data_loader.py.\n"+"="*60)
    exit()

# --- Default Configuration Parameters (can be overridden) ---
# These are defaults used if not specified when calling the function,
# AND used by the __main__ demo section.
DEFAULT_MORPH_KERNEL_SIZE = (3, 3)
DEFAULT_MORPH_OPEN_ITERATIONS = 1
DEFAULT_MORPH_CLOSE_ITERATIONS = 2
DEFAULT_MIN_CONTOUR_AREA = 75
DEFAULT_MAX_CONTOUR_AREA = 20000

# --- Core Mask Processing Function (Importable) ---

def process_fg_mask(fg_mask,
                    min_area=DEFAULT_MIN_CONTOUR_AREA,
                    max_area=DEFAULT_MAX_CONTOUR_AREA,
                    morph_kernel_size=DEFAULT_MORPH_KERNEL_SIZE,
                    open_iterations=DEFAULT_MORPH_OPEN_ITERATIONS,
                    close_iterations=DEFAULT_MORPH_CLOSE_ITERATIONS):
    """
    Processes a foreground mask (e.g., from MOG2) to find object contours and bounding boxes.

    Args:
        fg_mask (np.ndarray): The input foreground mask (8-bit single channel).
        min_area (int): Minimum contour area.
        max_area (int): Maximum contour area.
        morph_kernel_size (tuple): Kernel size (width, height) for morphology.
        open_iterations (int): Iterations for MORPH_OPEN.
        close_iterations (int): Iterations for MORPH_CLOSE.

    Returns:
        list: A list of detected bounding boxes in (x, y, w, h) format.
        np.ndarray: The morphologically cleaned mask used for contour finding.
    """
    detected_boxes = []
    if fg_mask is None:
        return detected_boxes, None

    # 1. Morphological Operations
    if morph_kernel_size is not None and morph_kernel_size[0] > 0 and morph_kernel_size[1] > 0:
        morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, morph_kernel_size)
        # Opening: Remove noise
        mask_opened = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, morph_kernel, iterations=open_iterations)
        # Closing: Fill gaps
        mask_closed = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, morph_kernel, iterations=close_iterations)
    else:
        mask_closed = fg_mask # Skip morphology if kernel size is invalid

    # 2. Find Contours
    contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 3. Filter Contours and Get Bounding Boxes
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(contour)
            detected_boxes.append((x, y, w, h))

    return detected_boxes, mask_closed


# --- Example Usage / Demonstration ---
if __name__ == "__main__":
    print("--- Phase 2 Demonstration: Motion Detection (MOG2) ---")

    # --- MOG2 Parameters for Demo ---
    MOG2_HISTORY = 150
    MOG2_VAR_THRESHOLD = 25
    MOG2_DETECT_SHADOWS = False

    # 1. Verify DATASET_BASE_PATH is set in data_loader.py
    if 'PASTE_YOUR_FULL_LTIR_DATASET_PATH_HERE' in DATASET_BASE_PATH or not os.path.isdir(DATASET_BASE_PATH):
         print("="*60+"\n!!! ERROR: Set DATASET_BASE_PATH in data_loader.py !!!\n"+"="*60); exit()

    # 2. Select Sequence for Demo
    selected_sequence_name = '8_car'
    print(f"\nSelected sequence: {selected_sequence_name}")
    frame_paths = get_sequence_frames(DATASET_BASE_PATH, selected_sequence_name)
    if not frame_paths: print("Frames not found."); exit()
    print(f"Processing {len(frame_paths)} frames...")

    # 3. Initialize Background Subtractor for this demo run
    backSub = cv2.createBackgroundSubtractorMOG2(history=MOG2_HISTORY,
                                                 varThreshold=MOG2_VAR_THRESHOLD,
                                                 detectShadows=MOG2_DETECT_SHADOWS)
    print(f"Initializing MOG2 (Hist={MOG2_HISTORY}, VarThresh={MOG2_VAR_THRESHOLD})")

    # 4. Process Frames for Demo
    print(f"Using Detection Params: Area=[{DEFAULT_MIN_CONTOUR_AREA}-{DEFAULT_MAX_CONTOUR_AREA}], MorphKernel={DEFAULT_MORPH_KERNEL_SIZE}")
    print("Press 'q' to quit, 'p' to pause/resume.")

    for i, frame_path in enumerate(frame_paths):
        original_frame, _ = load_frame(frame_path)
        if original_frame is None: continue
        preprocessed_frame = preprocess_frame_pipeline(original_frame)
        if preprocessed_frame is None: continue

        # Apply Background Subtraction
        fgMask_raw = backSub.apply(preprocessed_frame)

        # Process the Mask using the importable function
        # Pass parameters explicitly or use defaults defined above
        detected_boxes, fgMask_processed = process_fg_mask(
            fgMask_raw,
            min_area=DEFAULT_MIN_CONTOUR_AREA,
            max_area=DEFAULT_MAX_CONTOUR_AREA,
            morph_kernel_size=DEFAULT_MORPH_KERNEL_SIZE,
            open_iterations=DEFAULT_MORPH_OPEN_ITERATIONS,
            close_iterations=DEFAULT_MORPH_CLOSE_ITERATIONS
        )

        # --- Visualization ---
        display_original_norm = normalize_frame(original_frame)
        if display_original_norm is None: continue
        frame_display = cv2.cvtColor(display_original_norm, cv2.COLOR_GRAY2BGR)
        for (x, y, w, h) in detected_boxes:
            cv2.rectangle(frame_display, (x, y), (x + w, y + h), (0, 0, 255), 2)

        fgMask_raw_display = cv2.cvtColor(fgMask_raw, cv2.COLOR_GRAY2BGR)
        fgMask_processed_display = cv2.cvtColor(fgMask_processed, cv2.COLOR_GRAY2BGR)
        preprocessed_display = cv2.cvtColor(preprocessed_frame, cv2.COLOR_GRAY2BGR)

        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX; scale = 0.5; color = (0, 255, 0); thick = 1
        cv2.putText(frame_display, f"Orig+Detect ({len(detected_boxes)}) Fr {i}", (10, 20), font, scale, color, thick)
        cv2.putText(fgMask_raw_display, "Raw FG Mask", (10, 20), font, scale, color, thick)
        cv2.putText(preprocessed_display, "Preprocessed In", (10, 20), font, scale, color, thick)
        cv2.putText(fgMask_processed_display, "Cleaned FG Mask", (10, 20), font, scale, color, thick)

        # Combine and display
        if frame_display.shape == fgMask_raw_display.shape == preprocessed_display.shape == fgMask_processed_display.shape:
            top_row = np.hstack((frame_display, fgMask_raw_display))
            bottom_row = np.hstack((preprocessed_display, fgMask_processed_display))
            combined = np.vstack((top_row, bottom_row))
            # Optional resize
            max_w=1280; max_h=720; h, w = combined.shape[:2]
            s = min(max_w/w, max_h/h);
            if s < 1.0: combined = cv2.resize(combined, (int(w*s), int(h*s)))
            cv2.imshow('Motion Detector Demo (MOG2)', combined)
        else:
            cv2.imshow('Detections', frame_display) # Fallback

        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'): break
        elif key == ord('p'): cv2.waitKey(-1)

    cv2.destroyAllWindows()
    print("\nDemo finished.")