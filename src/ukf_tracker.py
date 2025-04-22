"""
Defines the UKF state transition/measurement functions and the UKFTrack class.
"""
import numpy as np
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
from config import PipelineConfig # Import configuration class

# ========== UKF State Transition and Measurement Functions ==========
def fx(x: np.ndarray, dt: float) -> np.ndarray:
    """State transition function (fx) for a constant velocity model."""
    return np.array([x[0] + x[2]*dt, x[1] + x[3]*dt, x[2], x[3]])

def hx(x: np.ndarray) -> np.ndarray:
    """Measurement function (hx) - measures position only."""
    return x[:2]

# ========== UKF Track Class ==========
class UKFTrack:
    """Represents a single tracked object using an Unscented Kalman Filter (UKF)."""
    count = 0 # Static variable for unique IDs (if needed, currently managed by MultiObjectTracker)

    def __init__(self, x: int, y: int, track_id: int, config: PipelineConfig, dt: float = 1.0):
        """
        Initializes a new track with a UKF filter.

        Args:
            x: Initial x-coordinate of the detection centroid.
            y: Initial y-coordinate of the detection centroid.
            track_id: The unique ID assigned by the MultiObjectTracker.
            config: The pipeline configuration object (for noise parameters R, Q).
            dt: Time step (typically 1.0 frame).
        """
        sp = MerweScaledSigmaPoints(n=4, alpha=.1, beta=2., kappa=-1)
        self.ukf = UKF(dim_x=4, dim_z=2, dt=dt, hx=hx, fx=fx, points=sp)
        self.ukf.x = np.array([x, y, 0., 0.])
        self.ukf.P = np.diag([100., 100., 50., 50.]) # Initial state uncertainty
        self.ukf.R = np.diag([config.R, config.R])   # Measurement noise
        self.ukf.Q = np.diag([config.Q, config.Q, config.Q/2, config.Q/2]) # Process noise

        self.id: int = track_id
        self.misses: int = 0
        self.age: int = 1
        self.color: tuple[int, int, int] = tuple(np.random.randint(0, 255, 3).tolist())
        self.last_box: tuple[int, int, int, int] | None = None

    def predict(self) -> np.ndarray:
        """Performs the UKF predict step and returns the predicted position."""
        self.ukf.predict()
        return self.ukf.x[:2]

    def update(self, z: np.ndarray, box: tuple[int, int, int, int]):
        """Performs the UKF update step with a new measurement."""
        self.ukf.update(z)
        self.misses = 0
        self.age += 1
        self.last_box = box

    def no_update(self):
        """Handles the case where the track was not associated with any detection."""
        self.misses += 1
        self.age += 1
        # Optional: self.ukf.P *= 1.05