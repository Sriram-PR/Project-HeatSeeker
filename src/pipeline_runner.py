"""
Orchestrates the tracking pipeline for a single sequence.
"""
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import imageio
import motmetrics as mm

# Import necessary components from other modules
from config import PipelineConfig
from data_loader_mot import DatasetLoader
from preprocessing import preprocess, normalize_image
from detector import ForegroundDetector
from multi_tracker import MultiObjectTracker
from visualization import draw_tracks_on_frame, draw_gt_on_frame, show_frame
# Note: compute_iou might not be directly needed here if using mm.distances

class MotionDetectionPipeline:
    """Orchestrates the entire motion detection and tracking pipeline."""
    def __init__(self, dataset_loader: DatasetLoader, config: PipelineConfig):
        self.data: DatasetLoader = dataset_loader
        self.config: PipelineConfig = config
        self.detector = ForegroundDetector(config)
        self.tracker = MultiObjectTracker(config)

    def run(self, show_inline_every: int | None = None,
            save_gif_path: str | None = None,
            save_csv_path: str | None = None) -> tuple[list[tuple], pd.DataFrame]:
        """
        Executes the tracking pipeline for the loaded sequence.

        Returns:
            A tuple containing:
            - frame_update_data: List of (gt_ids, hyp_ids, dist_matrix) tuples for motmetrics.
            - results_df: Pandas DataFrame of per-frame track outputs.
        """
        vis_frames = []
        results_for_csv = []
        frame_update_data = []
        num_frames = len(self.data)
        # print(f"Starting pipeline run for {self.data.sequence_name} ({num_frames} frames)...") # Moved to optimize.py

        # for idx in range(num_frames): # Without tqdm
        for idx in tqdm(range(num_frames), desc=f"Pipeline:{self.data.sequence_name}", unit="frame", leave=True):
            frame, bit_depth = self.data.load_frame(idx)
            if frame is None: continue

            proc_frame = preprocess(
                frame, bit_depth,
                gaussian_ksize=self.config.gaussian_ksize,
                clahe_clip_limit=self.config.clahe_clip_limit,
                clahe_tile_grid_size=self.config.clahe_tile_grid_size
            )
            if proc_frame is None: continue

            fgmask = self.detector.detect(proc_frame)
            detections, det_boxes = self.detector.blobs(fgmask) # Uses config.min_blob_area internally
            active_tracks = self.tracker.associate_and_track(detections, det_boxes)

            # --- Prepare for MOT Metrics ---
            gt_info = self.data.gt_bboxes_with_id[idx]
            gt_ids = [gt_info['id']] if gt_info and 'bbox' in gt_info and gt_info['bbox'] else []
            gt_boxes = [gt_info['bbox']] if gt_info and 'bbox' in gt_info and gt_info['bbox'] else []
            hyp_ids = [trk.id for trk in active_tracks if trk.last_box]
            hyp_boxes = [trk.last_box for trk in active_tracks if trk.last_box]
            # Use motmetrics distance calculation
            dist_matrix = mm.distances.iou_matrix(gt_boxes, hyp_boxes, max_iou=1.0) # Range [0, 1] is correct
            frame_update_data.append((gt_ids, hyp_ids, dist_matrix))

            # --- Prepare for CSV ---
            gt_bbox_simple = gt_info['bbox'] if gt_info and 'bbox' in gt_info else None
            for trk in active_tracks:
                 if trk.last_box:
                    bx, by, bw, bh = trk.last_box
                    row = dict(
                        frame=idx, track_id=trk.id, x=bx, y=by, w=bw, h=bh,
                        gt_x=gt_bbox_simple[0] if gt_bbox_simple else np.nan,
                        gt_y=gt_bbox_simple[1] if gt_bbox_simple else np.nan,
                        gt_w=gt_bbox_simple[2] if gt_bbox_simple else np.nan,
                        gt_h=gt_bbox_simple[3] if gt_bbox_simple else np.nan
                    )
                    results_for_csv.append(row)

            # --- Visualization ---
            if save_gif_path: # Only prepare vis if saving GIF (show_inline disabled for optimization)
                vis_frame_8bit = normalize_image(frame, bit_depth)
                vis = cv2.cvtColor(vis_frame_8bit, cv2.COLOR_GRAY2BGR)
                vis = draw_tracks_on_frame(vis, active_tracks)
                vis = draw_gt_on_frame(vis, gt_info)
                vis_frames.append(vis.copy())

        # --- Save Outputs ---
        if save_gif_path and vis_frames:
            print(f"    (Saving GIF: {save_gif_path})") # Less verbose saving message
            try:
                vis_frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in vis_frames]
                imageio.mimsave(save_gif_path, vis_frames_rgb, fps=10)
            except Exception as e:
                print(f"    Error saving GIF {save_gif_path}: {e}")

        results_df = pd.DataFrame(results_for_csv)
        if save_csv_path:
            # print(f"    (Saving CSV: {save_csv_path})") # Less verbose saving message
            try:
                results_df.to_csv(save_csv_path, index=False, float_format='%.2f')
            except Exception as e:
                print(f"    Error saving CSV {save_csv_path}: {e}")

        # print(f"Pipeline run finished for {self.data.sequence_name}.") # Moved to optimize.py
        return frame_update_data, results_df