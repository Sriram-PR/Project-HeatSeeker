import cv2
import numpy as np
from config import PipelineConfig

class ForegroundDetector:
    """
    Detects foreground objects using MOG2 background subtraction and morphological filtering.
    """
    def __init__(self, config: PipelineConfig):
        """
        Initializes the detector with MOG2 and a morphological kernel.

        Args:
            config: The pipeline configuration object containing MOG2/morph parameters.
        """
        self.bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=config.mog2_history,
            varThreshold=config.mog2_varThreshold,
            detectShadows=False
        )
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (config.morph_kernel_size, config.morph_kernel_size)
        )
        self.min_blob_area = config.min_blob_area # Store min_blob_area from config

    def detect(self, img: np.ndarray) -> np.ndarray:
        """
        Applies background subtraction and morphological opening to detect foreground.

        Args:
            img: The preprocessed input frame (8-bit grayscale).

        Returns:
            A binary foreground mask (0 for background, 255 for foreground).
        """
        fgmask = self.bg_sub.apply(img)
        _, fgmask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, self.kernel)
        return fgmask

    def blobs(self, fgmask: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int, int, int]]]:
        """
        Finds contours (blobs) in the foreground mask and filters them by area.

        Args:
            fgmask: The binary foreground mask obtained from detect().

        Returns:
            A tuple containing:
            - detections: A list of centroid coordinates (cx, cy) for valid blobs.
            - det_boxes: A list of bounding boxes (x, y, w, h) for valid blobs.
        """
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        det_boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= self.min_blob_area: # Use stored min_blob_area
                x, y, w, h = cv2.boundingRect(cnt)
                w = max(1, w); h = max(1, h) # Ensure min size 1
                cx = x + w // 2
                cy = y + h // 2
                detections.append((cx, cy))
                det_boxes.append((x, y, w, h))
        return detections, det_boxes