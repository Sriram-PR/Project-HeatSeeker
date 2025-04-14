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