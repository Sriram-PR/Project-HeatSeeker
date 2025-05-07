class PipelineConfig:
    """
    Holds configuration parameters for the entire tracking pipeline.

    Centralizes tunable values for easy modification and experimentation,
    including parameters for preprocessing, detection, tracking, and association.
    """
    def __init__(self,
        # --- Detection & Background Subtraction ---
        min_blob_area: int = 80,
        mog2_history: int = 200,
        mog2_varThreshold: float = 16.0,
        morph_kernel_size: int = 5,

        # --- Tracking (UKF & Association) ---
        max_misses: int = 8,
        association_threshold: float = 60.0,
        Q: float = 0.1,
        R: float = 70.0,

        # --- Evaluation ---
        iou_threshold: float = 0.5,

        # --- Preprocessing (Bilateral + CLAHE) ---
        bilateral_d: int = 9,
        bilateral_sigmaColor: float = 75.0,
        bilateral_sigmaSpace: float = 75.0,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid_size: tuple[int, int] = (8, 8)
        ):
        # --- Assign detection/background params ---
        self.min_blob_area: int = min_blob_area
        self.mog2_history: int = mog2_history
        self.mog2_varThreshold: float = mog2_varThreshold
        self.morph_kernel_size: int = morph_kernel_size // 2 * 2 + 1 # Ensure odd

        # --- Assign tracking params ---
        self.max_misses: int = max_misses
        self.association_threshold: float = association_threshold
        self.Q: float = Q
        self.R: float = R

        # --- Assign evaluation params ---
        self.iou_threshold: float = iou_threshold # Used by motmetrics setup

        # --- Assign preprocessing params ---
        self.bilateral_d: int = bilateral_d
        self.bilateral_sigmaColor: float = bilateral_sigmaColor
        self.bilateral_sigmaSpace: float = bilateral_sigmaSpace
        self.clahe_clip_limit: float = clahe_clip_limit
        self.clahe_tile_grid_size: tuple[int, int] = clahe_tile_grid_size