import cv2
import numpy as np
import matplotlib.pyplot as plt
from ukf_tracker import UKFTrack

def draw_tracks_on_frame(frame: np.ndarray, tracks: list[UKFTrack]) -> np.ndarray:
    """Draws bounding boxes and IDs for active tracks on a frame."""
    for trk in tracks:
        if trk.misses == 0 and trk.last_box is not None:
            x, y, w, h = [int(c) for c in trk.last_box] # Ensure int
            clr = trk.color
            cv2.rectangle(frame, (x, y), (x + w, y + h), clr, 2)
            cv2.putText(frame, f'ID{trk.id}', (x, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, clr, 1, cv2.LINE_AA)
    return frame

def draw_gt_on_frame(frame: np.ndarray, gt_info: dict | None) -> np.ndarray:
    """Draws the ground truth bounding box and ID on a frame."""
    if gt_info and 'bbox' in gt_info and gt_info['bbox'] is not None:
        xg, yg, wg, hg = [int(c) for c in gt_info['bbox']] # Ensure int
        gt_color = (0, 0, 255) # Red in BGR
        cv2.rectangle(frame, (xg, yg), (xg + wg, yg + hg), gt_color, 2)
        cv2.putText(frame, f'GT{gt_info["id"]}', (xg, yg - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, gt_color, 1, cv2.LINE_AA)
    return frame

def show_frame(frame_bgr: np.ndarray, title: str = "Frame", size: tuple[int, int] = (8, 6)):
    """Displays a single frame inline using Matplotlib."""
    if frame_bgr is None:
         print("Warning: Cannot show None frame.")
         return
    plt.figure(figsize=size)
    plt.imshow(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)) # Convert BGR to RGB
    plt.title(title)
    plt.axis('off')
    plt.show()