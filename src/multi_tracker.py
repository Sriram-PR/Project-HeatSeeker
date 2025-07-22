import numpy as np
from scipy.optimize import linear_sum_assignment

from config import PipelineConfig  # Import configuration
from ukf_tracker import UKFTrack  # Import the single track class


class MultiObjectTracker:
    """Manages multiple UKFTrack objects and performs data association."""

    def __init__(self, config: PipelineConfig):
        """
        Initializes the multi-object tracker.

        Args:
            config: The pipeline configuration object.
        """
        self.tracks: list[UKFTrack] = []
        self.config: PipelineConfig = config
        self.next_id: int = 1  # Initialize track ID counter

    def associate_and_track(
        self,
        detections: list[tuple[int, int]],
        det_boxes: list[tuple[int, int, int, int]],
    ) -> list[UKFTrack]:
        """
        Performs prediction, association, update, and track management.

        Args:
            detections: List of detected centroids (cx, cy).
            det_boxes: List of corresponding bounding boxes (x, y, w, h).

        Returns:
            A list of currently active tracks (misses == 0).
        """
        # 1. Predict
        trk_preds = (
            np.array([trk.predict() for trk in self.tracks])
            if self.tracks
            else np.zeros((0, 2))
        )
        dets_np = np.array(detections) if detections else np.zeros((0, 2))

        matches = []
        unmatched_tracks = list(range(len(self.tracks)))
        unmatched_detections = list(range(len(detections)))

        # 2. Associate (Hungarian)
        if len(self.tracks) > 0 and len(detections) > 0:
            cost_matrix = np.linalg.norm(
                trk_preds[:, None, :] - dets_np[None, :, :], axis=2
            )
            cost_matrix[cost_matrix > self.config.association_threshold] = 1e8
            try:
                trk_idx, det_idx = linear_sum_assignment(cost_matrix)
                valid_matches_mask = (
                    cost_matrix[trk_idx, det_idx] < self.config.association_threshold
                )
                matched_trk_indices = trk_idx[valid_matches_mask]
                matched_det_indices = det_idx[valid_matches_mask]
                matches = list(zip(matched_trk_indices, matched_det_indices))
                unmatched_tracks = [
                    t for t in range(len(self.tracks)) if t not in matched_trk_indices
                ]
                unmatched_detections = [
                    d for d in range(len(detections)) if d not in matched_det_indices
                ]
            except ValueError as e:
                print(
                    f"Warning: linear_sum_assignment failed: {e}. Skipping association."
                )
                # Proceed with all tracks/detections as unmatched

        # 3. Update Matched
        for t_idx, d_idx in matches:
            self.tracks[t_idx].update(np.array(detections[d_idx]), det_boxes[d_idx])

        # 4. Handle Unmatched Tracks
        for t_idx in unmatched_tracks:
            self.tracks[t_idx].no_update()

        # 5. Create New Tracks
        for d_idx in unmatched_detections:
            trk = UKFTrack(*detections[d_idx], self.next_id, config=self.config)
            trk.last_box = det_boxes[d_idx]
            self.tracks.append(trk)
            self.next_id += 1

        # 6. Prune Dead Tracks
        self.tracks = [
            trk for trk in self.tracks if trk.misses <= self.config.max_misses
        ]

        # 7. Return Active Tracks
        active_tracks = [trk for trk in self.tracks if trk.misses == 0]
        return active_tracks
