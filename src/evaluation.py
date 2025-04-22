"""
Helper functions for calculating IoU and standard MOT metrics.
"""
import numpy as np
import pandas as pd
import motmetrics as mm

def compute_iou(boxA: tuple[int, int, int, int], boxB: tuple[int, int, int, int]) -> float:
    """Computes the Intersection over Union (IoU) between two bounding boxes."""
    if boxA is None or boxB is None: return 0.0
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interW = max(0, xB - xA); interH = max(0, yB - yA)
    interArea = float(interW * interH)
    boxAArea = float(max(0, boxA[2]) * max(0, boxA[3]))
    boxBArea = float(max(0, boxB[2]) * max(0, boxB[3]))
    unionArea = boxAArea + boxBArea - interArea
    iou = interArea / (unionArea + 1e-6) # Add epsilon
    return max(0.0, min(1.0, iou)) # Clamp

def per_frame_analysis(results_df: pd.DataFrame) -> dict:
    """Performs a simple analysis based on the saved tracker output CSV."""
    if results_df is None or results_df.empty: return {}
    analysis_results = {}
    if 'frame' in results_df.columns and 'track_id' in results_df.columns:
        avg_tracks = results_df.groupby('frame')['track_id'].nunique().mean()
        analysis_results['avg_tracks_csv'] = avg_tracks
    else: analysis_results['avg_tracks_csv'] = 0

    required_cols = ['x', 'y', 'w', 'h', 'gt_x', 'gt_y', 'gt_w', 'gt_h']
    if all(col in results_df.columns for col in required_cols):
        valid_iou_df = results_df.dropna(subset=required_cols)
        if not valid_iou_df.empty:
            ious = [compute_iou((r.x, r.y, r.w, r.h), (r.gt_x, r.gt_y, r.gt_w, r.gt_h))
                    for _, r in valid_iou_df.iterrows()]
            mean_iou_csv = np.mean(ious) if ious else 0
            analysis_results['mean_iou_csv'] = mean_iou_csv
        else: analysis_results['mean_iou_csv'] = 0
    else: analysis_results['mean_iou_csv'] = 0
    # Print results within this function if desired, or just return the dict
    print(f"CSV Analysis - Avg Tracks: {analysis_results['avg_tracks_csv']:.2f}, Mean Pair IoU: {analysis_results['mean_iou_csv']:.4f}")
    return analysis_results

def calculate_mot_metrics(mot_accumulator: mm.MOTAccumulator) -> pd.DataFrame:
    """Calculates and displays standard MOT metrics using a populated MOTAccumulator."""
    if mot_accumulator is None or mot_accumulator.events.empty:
         print("Warning: MOTAccumulator is empty. Cannot calculate metrics.")
         return pd.DataFrame()
    mh = mm.metrics.create()
    summary = mh.compute(
        mot_accumulator,
        metrics=mm.metrics.motchallenge_metrics,
        name='acc_summary'
    )
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print("\n--- MOTChallenge Metrics ---")
    print(strsummary)
    return summary