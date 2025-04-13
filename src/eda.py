# File: eda.py
# Purpose: Exploratory Data Analysis - Load LTIR sequence data, parse ground truth,
#          and visualize the first frame with its annotation.

import cv2
import numpy as np
import os
import glob # For finding files easily
import matplotlib.pyplot as plt # Added for potential future plotting

# --- Configuration ---
# >>> IMPORTANT: Set this to the actual path on your system <<<
# Example for Windows: DATASET_BASE_PATH = r'C:\datasets\ltir_v1_0_8bit_16bit'
# Example for Linux/macOS: DATASET_BASE_PATH = '/home/user/datasets/ltir_v1_0_8bit_16bit'
DATASET_BASE_PATH = r'./data/ltir_v1_0_8bit_16bit' # Use 'r' for raw string on Windows

# --- Utility Functions ---

def get_sequence_frames(base_path, sequence_name):
    """Gets a sorted list of frame image paths for a given sequence."""
    sequence_path = os.path.join(base_path, sequence_name)
    # Look specifically for .png files to avoid other files if any
    frames_pattern = os.path.join(sequence_path, '*.png')
    frame_paths = sorted(glob.glob(frames_pattern)) # glob finds files matching pattern
                                                    # sorted ensures correct order
    if not frame_paths:
        print(f"Warning: No PNG frames found for sequence '{sequence_name}' at {sequence_path}")
    return frame_paths

def parse_ground_truth(base_path, sequence_name):
    """Parses the groundtruth.txt file for a sequence."""
    gt_path = os.path.join(base_path, sequence_name, 'groundtruth.txt')
    ground_truth_boxes_corners = [] # Store the raw 8 corner coords
    try:
        with open(gt_path, 'r') as f:
            for line in f:
                # Split the line by comma and convert to float
                coords = list(map(float, line.strip().split(',')))
                if len(coords) == 8:
                    ground_truth_boxes_corners.append(coords)
                else:
                    # Handle potential special cases from VOT format if needed (e.g., skipped frames)
                    # For now, just warn about unexpected lines
                    print(f"Warning: Unexpected line format in {gt_path}: {line.strip()}")
    except FileNotFoundError:
        print(f"Error: groundtruth.txt not found for sequence '{sequence_name}' at {gt_path}")
        return None
    return ground_truth_boxes_corners

def convert_corners_to_bbox(corners):
    """Converts 8 corner coordinates [x1, y1, ..., y4] to [x_min, y_min, width, height]."""
    if not isinstance(corners, (list, tuple)) or len(corners) != 8:
        print(f"Warning: Invalid input for corner conversion: {corners}")
        return None # Invalid input

    try:
        # Extract x and y coordinates
        x_coords = [corners[i] for i in range(0, 8, 2)]
        y_coords = [corners[i] for i in range(1, 8, 2)]

        x_min = min(x_coords)
        y_min = min(y_coords)
        x_max = max(x_coords)
        y_max = max(y_coords)

        # Ensure width and height are non-negative
        width = max(0.0, x_max - x_min)
        height = max(0.0, y_max - y_min)

        # Return as integers, as typically used for bounding boxes in OpenCV drawing
        return int(round(x_min)), int(round(y_min)), int(round(width)), int(round(height))
    except Exception as e:
        print(f"Error converting corners {corners}: {e}")
        return None


def load_frame(frame_path):
    """Loads a single frame image, attempting to determine bit depth."""
    # Try loading as anydepth first
    frame = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED) # More general loader

    if frame is None:
        print(f"Error: Failed to load frame '{frame_path}'")
        return None, None # Return frame and bit depth

    # Check the depth
    if frame.dtype == np.uint8:
        bit_depth = 8
        # Ensure it's treated as grayscale if it has only 2 dimensions
        if len(frame.shape) == 2:
             pass # Already grayscale
        elif len(frame.shape) == 3 and frame.shape[2] == 3:
             # Convert to grayscale if loaded as color (unlikely for LTIR but safe)
             print(f"Warning: Frame {frame_path} loaded as color, converting to grayscale.")
             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
             print(f"Warning: Unexpected shape for 8-bit image {frame_path}: {frame.shape}")

    elif frame.dtype == np.uint16:
        bit_depth = 16
        # Ensure it's grayscale (should be for 16-bit thermal)
        if len(frame.shape) != 2:
             print(f"Warning: Expected 16-bit image {frame_path} to be grayscale, but shape is {frame.shape}")
             # Attempt to handle common cases if necessary, otherwise raise error or return None
    else:
        print(f"Warning: Unsupported dtype {frame.dtype} for frame '{frame_path}'. Attempting grayscale load.")
        # Fallback to grayscale load
        frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
        if frame is None:
             print(f"Error: Fallback grayscale load failed for '{frame_path}'")
             return None, None
        bit_depth = 8 # Assume 8-bit after fallback

    return frame, bit_depth

# --- Verification Script / Main Execution ---
if __name__ == "__main__":
    # 1. Choose a sequence to test
    test_sequence_name = '8_car' # Change this to test other sequences
    print(f"--- Testing Sequence: {test_sequence_name} ---")

    # Basic check if base path is set
    if 'PASTE_YOUR_FULL_LTIR_DATASET_PATH_HERE' in DATASET_BASE_PATH or not os.path.isdir(DATASET_BASE_PATH):
         print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
         print("!!! ERROR: Please set the DATASET_BASE_PATH variable   !!!")
         print("!!! in the eda.py script to your actual dataset location !!!")
         print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
         exit()


    # 2. Get frame paths
    frame_paths = get_sequence_frames(DATASET_BASE_PATH, test_sequence_name)
    if not frame_paths:
        print("Exiting due to missing frames.")
        exit()
    print(f"Found {len(frame_paths)} frames.")

    # 3. Parse ground truth corners
    gt_corners_list = parse_ground_truth(DATASET_BASE_PATH, test_sequence_name)
    if gt_corners_list is None:
        print("Exiting due to missing ground truth.")
        exit()

    # Check if number of ground truth entries matches frames
    # Note: Some VOT sequences might have GT starting/ending differently,
    # but for basic exploration, we assume 1-to-1 for now.
    if len(gt_corners_list) != len(frame_paths):
        print(f"Warning: Number of ground truth entries ({len(gt_corners_list)}) "
              f"does not match number of frames ({len(frame_paths)}).")
        # Decide how to handle this - stop, or just proceed with available data?
        # For now, let's only use frames that have corresponding GT
        min_len = min(len(frame_paths), len(gt_corners_list))
        frame_paths = frame_paths[:min_len]
        gt_corners_list = gt_corners_list[:min_len]
        print(f"Processing the first {min_len} frames/GT entries.")

    if not frame_paths: # Check if we have anything left to process
         print("No frames left to process after aligning with ground truth.")
         exit()

    print(f"Found {len(gt_corners_list)} ground truth entries aligned with frames.")

    # 4. Load the FIRST frame
    first_frame_path = frame_paths[0]
    first_frame, detected_bit_depth = load_frame(first_frame_path)

    if first_frame is None:
        print("Exiting due to frame loading error.")
        exit()
    print(f"Loaded first frame. Detected bit depth: {detected_bit_depth}-bit.")

    # 5. Get the FIRST ground truth box
    first_gt_corners = gt_corners_list[0]
    first_gt_bbox = convert_corners_to_bbox(first_gt_corners) # (x, y, w, h) format

    if first_gt_bbox is None:
        print("Error converting first ground truth corners to bbox.")
        exit()
    print(f"First GT corners: {first_gt_corners}")
    print(f"First GT bbox (x, y, w, h): {first_gt_bbox}")


    # 6. Prepare Frame for Display (Handle Grayscale / 16-bit)
    display_frame = None
    if len(first_frame.shape) == 2: # Grayscale (Could be 8-bit or 16-bit)
        if detected_bit_depth == 16:
            # Normalize 16-bit to 0-255 for display
            print("Normalizing 16-bit frame for display.")
            normalized_frame = cv2.normalize(first_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            display_frame = cv2.cvtColor(normalized_frame, cv2.COLOR_GRAY2BGR) # Convert to color for drawing
        else: # 8-bit grayscale
            display_frame = cv2.cvtColor(first_frame, cv2.COLOR_GRAY2BGR) # Convert to color for drawing
    elif len(first_frame.shape) == 3: # Should technically not happen often with LTIR
         print("Warning: Frame appears to be color, displaying as is.")
         display_frame = first_frame.copy()
    else:
         print("Error: Cannot determine frame format for display.")
         exit()


    # 7. Draw the bounding box on the display frame
    if display_frame is not None and first_gt_bbox is not None:
        x, y, w, h = first_gt_bbox
        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2) # Green box, thickness 2
        cv2.putText(display_frame, 'Ground Truth', (x, y - 10 if y>10 else y+10 ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 8. Display the frame
        cv2.imshow(f"First Frame - {test_sequence_name}", display_frame)
        print("\nShowing the first frame with the ground truth bounding box.")
        print("Press any key to exit.")
        cv2.waitKey(0) # Wait indefinitely until a key is pressed
        cv2.destroyAllWindows()
    else:
        print("Could not display frame.")