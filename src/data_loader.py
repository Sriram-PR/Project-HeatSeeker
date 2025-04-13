# File: data_loader.py
# Purpose: Functions for loading LTIR sequence data and applying common preprocessing steps.

import cv2
import numpy as np
import os
import glob

# --- Configuration ---
# >>> IMPORTANT: Set this to the actual path on your system <<<
# Example for Windows: DATASET_BASE_PATH = r'C:\datasets\ltir_v1_0_8bit_16bit'
# Example for Linux/macOS: DATASET_BASE_PATH = '/home/user/datasets/ltir_v1_0_8bit_16bit'
# You might want to remove this later and pass the path as an argument for better modularity
DATASET_BASE_PATH = r'./data/ltir_v1_0_8bit_16bit'

# --- Loading Functions (Adapted from eda.py) ---

def get_sequence_frames(base_path, sequence_name):
    """Gets a sorted list of frame image paths for a given sequence."""
    sequence_path = os.path.join(base_path, sequence_name)
    frames_pattern = os.path.join(sequence_path, '*.png')
    frame_paths = sorted(glob.glob(frames_pattern))
    if not frame_paths:
        print(f"Warning: No PNG frames found for sequence '{sequence_name}' at {sequence_path}")
    return frame_paths

def parse_ground_truth(base_path, sequence_name):
    """Parses the groundtruth.txt file for a sequence into a list of corner lists."""
    gt_path = os.path.join(base_path, sequence_name, 'groundtruth.txt')
    ground_truth_boxes_corners = []
    try:
        with open(gt_path, 'r') as f:
            for line in f:
                coords = list(map(float, line.strip().split(',')))
                if len(coords) == 8:
                    ground_truth_boxes_corners.append(coords)
                else:
                    print(f"Warning: Unexpected line format in {gt_path}: {line.strip()}")
    except FileNotFoundError:
        print(f"Error: groundtruth.txt not found for sequence '{sequence_name}' at {gt_path}")
        return None
    return ground_truth_boxes_corners

def convert_corners_to_bbox(corners):
    """Converts 8 corner coordinates [x1, y1, ..., y4] to [x_min, y_min, width, height]."""
    if not isinstance(corners, (list, tuple)) or len(corners) != 8:
        # print(f"Warning: Invalid input for corner conversion: {corners}") # Can be verbose
        return None
    try:
        x_coords = [corners[i] for i in range(0, 8, 2)]
        y_coords = [corners[i] for i in range(1, 8, 2)]
        x_min = min(x_coords)
        y_min = min(y_coords)
        x_max = max(x_coords)
        y_max = max(y_coords)
        width = max(0.0, x_max - x_min)
        height = max(0.0, y_max - y_min)
        return int(round(x_min)), int(round(y_min)), int(round(width)), int(round(height))
    except Exception as e:
        print(f"Error converting corners {corners}: {e}")
        return None

def load_frame(frame_path):
    """Loads a single frame image, returning the frame and its detected bit depth (8 or 16)."""
    frame = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
    if frame is None:
        print(f"Error: Failed to load frame '{frame_path}'")
        return None, None
    if frame.dtype == np.uint8:
        bit_depth = 8
        if len(frame.shape) == 3: # If loaded as color, convert to gray
             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif frame.dtype == np.uint16:
        bit_depth = 16
        if len(frame.shape) != 2: # Should be grayscale
             print(f"Warning: Expected 16-bit image {frame_path} to be grayscale, shape is {frame.shape}")
             # Handle potential errors or return None if critical
    else:
        print(f"Warning: Unsupported dtype {frame.dtype} for frame '{frame_path}'. Attempting grayscale.")
        frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE) # Fallback
        bit_depth = 8 if frame is not None else None
    return frame, bit_depth

# --- Preprocessing Functions ---

def normalize_frame(frame):
    """Normalizes a frame (typically 16-bit) to 0-255 8-bit range for consistent processing."""
    if frame is None: return None
    if frame.dtype == np.uint16:
        normalized = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return normalized
    elif frame.dtype == np.uint8:
        return frame # Already 8-bit
    else:
        print(f"Warning: Unsupported dtype {frame.dtype} for normalization.")
        return None

def apply_median_blur(frame_8bit, ksize=3):
    """Applies Median Blur for noise reduction. Expects 8-bit input."""
    if frame_8bit is None or frame_8bit.dtype != np.uint8:
        # print("Median blur requires 8-bit input.") # Can be verbose
        return frame_8bit # Return input if not suitable
    ksize = max(1, ksize)
    if ksize % 2 == 0: ksize += 1 # Ensure ksize is odd
    blurred = cv2.medianBlur(frame_8bit, ksize)
    return blurred

def apply_clahe(frame_8bit_gray, clip_limit=2.0, tile_grid_size=(8, 8)):
    """Applies CLAHE for contrast enhancement. Expects 8-bit grayscale input."""
    if frame_8bit_gray is None or frame_8bit_gray.dtype != np.uint8 or len(frame_8bit_gray.shape) != 2:
        # print("CLAHE requires 8-bit grayscale input.") # Can be verbose
        return frame_8bit_gray # Return input if not suitable
    try:
        clahe_processor = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        enhanced = clahe_processor.apply(frame_8bit_gray)
        return enhanced
    except cv2.error as e:
        print(f"OpenCV Error during CLAHE: {e}")
        print(f"Input frame dtype: {frame_8bit_gray.dtype}, shape: {frame_8bit_gray.shape}")
        return frame_8bit_gray # Return original on error

def preprocess_frame_pipeline(frame, median_ksize=3, clahe_clip_limit=2.0, clahe_tile_grid_size=(8, 8)):
    """Applies the full preprocessing pipeline: Normalize -> Median Blur -> CLAHE."""
    if frame is None: return None

    # 1. Normalize (handles 16-bit to 8-bit conversion)
    frame_8bit = normalize_frame(frame)
    if frame_8bit is None: return None

    # 2. Apply Median Blur
    blurred_8bit = apply_median_blur(frame_8bit, ksize=median_ksize)
    # blurred_8bit should still be 8-bit if input was 8-bit

    # 3. Apply CLAHE (expects 8-bit grayscale)
    enhanced_8bit = apply_clahe(blurred_8bit, clip_limit=clahe_clip_limit, tile_grid_size=clahe_tile_grid_size)

    return enhanced_8bit


# --- Example Usage (Demonstrates loading and preprocessing) ---
if __name__ == "__main__":

    # Basic check if base path is set
    if 'PASTE_YOUR_FULL_LTIR_DATASET_PATH_HERE' in DATASET_BASE_PATH or not os.path.isdir(DATASET_BASE_PATH):
         print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
         print("!!! ERROR: Please set the DATASET_BASE_PATH variable   !!!")
         print("!!! in the data_loader.py script before running!       !!!")
         print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
         exit()

    # --- Configuration for Example ---
    test_sequence_name = '8_car' # Try '8_trees' or a 16-bit sequence if available
    frame_index_to_show = 50 # Pick a frame number

    # --- Load Data ---
    print(f"--- Loading data for sequence: {test_sequence_name} ---")
    frame_paths = get_sequence_frames(DATASET_BASE_PATH, test_sequence_name)
    if not frame_paths or frame_index_to_show >= len(frame_paths):
        print(f"Error: Frame paths not found or index {frame_index_to_show} out of bounds ({len(frame_paths)} frames total).")
        exit()

    # --- Load Specific Frame ---
    frame_path = frame_paths[frame_index_to_show]
    print(f"Loading frame: {frame_path}")
    original_frame, bit_depth = load_frame(frame_path)
    if original_frame is None: exit()
    print(f"Original frame loaded ({bit_depth}-bit).")

    # --- Apply Preprocessing ---
    print("Applying preprocessing pipeline...")
    # Tune parameters here if desired for the example:
    preprocessed_frame = preprocess_frame_pipeline(original_frame,
                                                   median_ksize=3,
                                                   clahe_clip_limit=3.0,
                                                   clahe_tile_grid_size=(8, 8))
    if preprocessed_frame is None:
        print("Preprocessing failed.")
        exit()

    # --- Display Results ---
    display_original_8bit = normalize_frame(original_frame) # Ensure 8-bit for display

    if display_original_8bit is not None and preprocessed_frame is not None:
        # Add text labels to the images
        cv2.putText(display_original_8bit, 'Original (Normalized)', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(preprocessed_frame, 'Preprocessed (Median+CLAHE)', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Combine images side-by-side (requires them to have same height)
        comparison_image = cv2.hconcat([display_original_8bit, preprocessed_frame])

        cv2.imshow(f"Preprocessing Comparison - Frame {frame_index_to_show} of {test_sequence_name}", comparison_image)
        print("\nShowing comparison. Left: Original (Normalized). Right: Preprocessed.")
        print("Observe the difference in contrast and noise.")
        print("Press any key to exit.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Could not prepare images for display.")