import glob
import os

import cv2
import numpy as np


class DatasetLoader:
    """
    Handles loading image sequences and corresponding ground truth data.

    Specifically designed for datasets like LTIR where frames are PNG images
    and ground truth is a text file with corner coordinates for (typically)
    a single object per frame.
    """

    def __init__(self, base_path: str, sequence_name: str):
        """
        Initializes the loader for a specific sequence.

        Args:
            base_path: The root directory containing all sequence folders.
            sequence_name: The name of the specific sequence folder to load.
        """
        self.base_path: str = base_path
        self.sequence_name: str = sequence_name
        self.sequence_path: str = os.path.join(base_path, sequence_name)
        self.frame_paths: list[str] = self.get_sequence_frames()
        self.gt_corners: list[list[float]] | None = self.parse_ground_truth()
        self.gt_bboxes_with_id: list[dict | None] = []
        if self.gt_corners:
            for corners in self.gt_corners:
                bbox = self.convert_corners_to_bbox(corners)
                if bbox:
                    self.gt_bboxes_with_id.append({"id": 1, "bbox": bbox})
                else:
                    self.gt_bboxes_with_id.append(None)
        else:
            self.gt_bboxes_with_id = [None] * len(self.frame_paths)

    def get_sequence_frames(self) -> list[str]:
        """Finds and sorts all PNG image files in the sequence directory."""
        pattern = os.path.join(self.sequence_path, "*.png")
        frame_paths = sorted(glob.glob(pattern))
        if not frame_paths:
            print(f"Warning: No PNG frames found in {self.sequence_path}")
        return frame_paths

    def parse_ground_truth(self) -> list[list[float]] | None:
        """Parses the 'groundtruth.txt' file if it exists."""
        gt_path = os.path.join(self.sequence_path, "groundtruth.txt")
        gt_corners_list = []
        try:
            with open(gt_path, "r") as f:
                for line in f:
                    coords = list(map(float, line.strip().split(",")))
                    if len(coords) == 8:
                        gt_corners_list.append(coords)
                    else:
                        print(
                            f"Warning: Skipping line in {gt_path} with unexpected number of coordinates ({len(coords)}): {line.strip()}"
                        )
        except FileNotFoundError:
            print(f"Warning: Ground truth file not found: {gt_path}")
            return None
        except ValueError as e:
            print(f"Error parsing ground truth file {gt_path}: {e}. Check file format.")
            return None
        return gt_corners_list

    @staticmethod
    def convert_corners_to_bbox(
        corners: list[float],
    ) -> tuple[int, int, int, int] | None:
        """Converts 8 corner coordinates [x1,y1,...,x4,y4] to a bounding box [x, y, w, h]."""
        if not isinstance(corners, (list, tuple)) or len(corners) != 8:
            # print(f"Warning: Invalid corner format for bbox conversion: {corners}") # Can be verbose
            return None
        try:
            x_coords = [corners[i] for i in range(0, 8, 2)]
            y_coords = [corners[i] for i in range(1, 8, 2)]
            x_min, y_min = min(x_coords), min(y_coords)
            x_max, y_max = max(x_coords), max(y_coords)
            width = max(0.0, x_max - x_min)
            height = max(0.0, y_max - y_min)
            if width == 0 and x_max != x_min:
                width = 1.0
            if height == 0 and y_max != y_min:
                height = 1.0
            return (
                int(round(x_min)),
                int(round(y_min)),
                int(round(width)),
                int(round(height)),
            )
        except Exception as e:
            print(f"Error converting corners to bbox: {e}, corners={corners}")
            return None

    def load_frame(self, idx: int) -> tuple[np.ndarray | None, int | None]:
        """Loads a single frame from the sequence by its index."""
        if not 0 <= idx < len(self.frame_paths):
            # print(f"Error: Frame index {idx} out of bounds (0-{len(self.frame_paths)-1}).") # Can be verbose
            return None, None
        frame_path = self.frame_paths[idx]
        try:
            frame = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
            if frame is None:
                print(f"Warning: Failed to load frame {frame_path}")
                return None, None
            if frame.dtype == np.uint8:
                bit_depth = 8
            elif frame.dtype == np.uint16:
                bit_depth = 16
            else:
                print(
                    f"Warning: Unexpected frame dtype {frame.dtype} for {frame_path}. Reloading as grayscale."
                )
                frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
                if frame is None:
                    print(f"Warning: Failed to reload frame {frame_path} as grayscale.")
                    return None, None
                bit_depth = 8
            return frame, bit_depth
        except Exception as e:
            print(f"Error loading frame {frame_path}: {e}")
            return None, None

    def __len__(self) -> int:
        """Returns the number of frames available in the sequence."""
        if self.gt_bboxes_with_id:
            return min(len(self.frame_paths), len(self.gt_bboxes_with_id))
        else:
            return len(self.frame_paths)
