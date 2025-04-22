"""
Image preprocessing functions suitable for thermal imagery.
"""
import cv2
import numpy as np

def normalize_image(img: np.ndarray, bit_depth: int) -> np.ndarray | None:
    """
    Normalizes an image to 8-bit grayscale [0, 255].

    Handles 16-bit to 8-bit conversion via min-max normalization and ensures
    the output is a single-channel 8-bit unsigned integer array.

    Args:
        img: The input image (NumPy array). Can be 8-bit or 16-bit, grayscale or color.
        bit_depth: The original bit depth of the image (e.g., 8 or 16).

    Returns:
        The normalized 8-bit grayscale image as a NumPy array, or None if input is invalid.
    """
    if img is None:
        return None
    if bit_depth == 16:
        # Ensure input is valid before normalize (e.g., handle potential all-zero images if necessary)
        min_val, max_val = img.min(), img.max()
        if max_val > min_val:
             img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        elif max_val == min_val: # Handle flat image
             img = np.full_like(img, int(255.0 * min_val / 65535.0) if min_val > 0 else 0, dtype=np.uint8)
        else: # Should not happen if input dtype is uint16
             img = np.zeros_like(img, dtype=np.uint8)

    if len(img.shape) > 2 and img.shape[2] > 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        # Clip and cast if somehow normalization resulted in different type (unlikely with CV_8U)
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def preprocess(img: np.ndarray, bit_depth: int,
               gaussian_ksize: int, clahe_clip_limit: float,
               clahe_tile_grid_size: tuple[int, int]) -> np.ndarray | None:
    """
    Applies a preprocessing pipeline: Normalize -> Gaussian Blur -> CLAHE.

    Args:
        img: The input image (NumPy array, possibly 16-bit).
        bit_depth: The original bit depth of the image (8 or 16).
        gaussian_ksize: Kernel size for Gaussian blur (must be odd).
        clahe_clip_limit: Contrast limit for CLAHE.
        clahe_tile_grid_size: Tile grid size for CLAHE.

    Returns:
        The preprocessed 8-bit grayscale image, or None if normalization fails.
    """
    img_norm = normalize_image(img, bit_depth)
    if img_norm is None:
        # print("Error: Normalization failed in preprocess.") # Can be verbose
        return None

    if len(img_norm.shape) > 2 or img_norm.dtype != np.uint8:
       # print("Warning: Image not 8-bit grayscale after normalization, attempting conversion.")
       if len(img_norm.shape) > 2: img_norm = cv2.cvtColor(img_norm, cv2.COLOR_BGR2GRAY)
       if img_norm.dtype != np.uint8: img_norm = cv2.normalize(img_norm, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    ksize = gaussian_ksize // 2 * 2 + 1 # Ensure odd
    img_blur = cv2.GaussianBlur(img_norm, (ksize, ksize), 0)

    try:
        clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_grid_size)
        img_clahe = clahe.apply(img_blur)
    except cv2.error as e:
         print(f"Error applying CLAHE: {e}. Returning blurred image.")
         return img_blur

    return img_clahe