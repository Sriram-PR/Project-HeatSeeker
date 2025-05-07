# Classical Object Tracking Pipeline for Thermal Infrared Video

This project implements and optimizes a classical computer vision pipeline for single-object tracking in thermal infrared (TIR) video, primarily using the Linköping Thermal Infrared (LTIR) dataset. The pipeline combines MOG2 background subtraction for detection, an Unscented Kalman Filter (UKF) for state estimation, and the Hungarian algorithm for data association, with tailored preprocessing for thermal imagery. Bayesian Optimization is used to tune pipeline parameters.

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Setup](#setup)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Dataset](#dataset)
- [Usage](#usage)
  - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
  - [Running a Single Example](#running-a-single-example)
  - [Running Parameter Optimization](#running-parameter-optimization)
- [Pipeline Components](#pipeline-components)
- [Parameter Tuning](#parameter-tuning)
- [Performance Overview](#performance-overview)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

## Features

-   **Modular Design:** Code is organized into logical modules for configuration, data loading, preprocessing, detection, tracking, evaluation, and visualization.
-   **Thermal-Specific Preprocessing:** Includes normalization, Bilateral Filtering (edge-preserving), and CLAHE (local contrast enhancement).
-   **Classical Detection:** Utilizes MOG2 background subtraction with morphological filtering.
-   **Robust State Estimation:** Employs an Unscented Kalman Filter (UKF) with a constant velocity motion model for each track.
-   **Optimal Data Association:** Uses the Hungarian algorithm for matching detections to tracks.
-   **MOT Evaluation:** Integrates with the `motmetrics` library for standard Multiple Object Tracking (MOT) metric calculation (MOTA, MOTP, IDF1, etc.).
-   **Parameter Optimization:** Includes a script using Bayesian Optimization (`scikit-optimize`) to tune critical pipeline parameters.
-   **Example Runner:** Provides a script to run the pipeline on a single sequence with fixed parameters.
-   **EDA Script:** Includes an initial script for basic data exploration and verification.

## Project Structure

```
Project-HeatSeeker/
├── config.py                   # PipelineConfig class
├── data_loader_mot.py          # DatasetLoader class
├── preprocessing.py            # Preprocessing functions (normalize, bilateral, clahe)
├── detector.py                 # ForegroundDetector class (MOG2, blobs)
├── ukf_tracker.py              # UKF state models (fx, hx) and UKFTrack class
├── multi_tracker.py            # MultiObjectTracker class (manages tracks, Hungarian assoc.)
├── visualization.py            # Drawing functions for frames
├── evaluation.py               # IoU calculation, MOT metrics calculation
├── pipeline_runner.py          # MotionDetectionPipeline class (orchestrates a single run)
├── optimize.py                 # Bayesian Optimization script (main runnable for tuning)
├── run_example.py              # Script to run pipeline on a single example sequence
├── eda.py                      # Initial script for Exploratory Data Analysis
├── output/                     # Default directory for optimization trial logs and validation results
│   └── (generated CSV files)
└── output_example/             # Default directory for example run outputs (GIFs, CSVs)
    └── (generated GIFs and CSVs)
└── README.md                   # This file
```

## Setup

### Prerequisites

-   Python 3.9+
-   `pip` (Python package installer)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Sriram-PR/Project-HeatSeeker.git
    cd Project-HeatSeeker
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install required packages:**
    ```bash
    pip install -r requirements.txt
    ```

### Dataset

1.  **Download the LTIR Dataset:** Obtain the "[LTIR Dataset v1.0 (8-bit and 16-bit)](https://www.cvl.isy.liu.se/en/research/datasets/ltir/version1.0/)" from its official source.
2.  **Extract the dataset.**
3.  **Update `DATASET_BASE_PATH` / `BASE_PATH`:**
    *   In `eda.py`, update `DATASET_BASE_PATH`.
    *   In `optimize.py` and `run_example.py`, update the `BASE_PATH` variable to point to the root directory of your extracted LTIR dataset.
    ```python
    # Example:
    BASE_PATH = '/path/to/your/ltir_v1_0_8bit_16bit/ltir_v1_0_8bit_16bit'
    ```

## Usage

### Exploratory Data Analysis (EDA)

The `eda.py` script can be used for initial dataset exploration, such as loading and visualizing the first frame of a sequence with its ground truth.

1.  **Configure `eda.py`:**
    *   Ensure `DATASET_BASE_PATH` is correctly set.
    *   Change `test_sequence_name` to the sequence you want to inspect.
2.  **Run:**
    ```bash
    python eda.py
    ```

### Running a Single Example

The `run_example.py` script allows you to run the tracking pipeline on a single sequence with a predefined set of parameters.

1.  **Configure `run_example.py`:** Set `BASE_PATH`, `SEQUENCE_NAME`, and optionally `EXAMPLE_CONFIG`, `SAVE_CSV`, `SAVE_GIF`.
2.  **Run:**
    ```bash
    python run_example.py
    ```
    Outputs will be saved to `output_example/`.

### Running Parameter Optimization

The `optimize.py` script uses Bayesian Optimization to find optimal pipeline parameters.

1.  **Configure `optimize.py`:** Set `BASE_PATH`, `TRAINING_SEQUENCES`, `VALIDATION_SEQUENCES`, `param_space`, `OPTIMIZATION_TARGET_METRIC`, and optimization settings.
2.  **Run:**
    ```bash
    python optimize.py
    ```
    This can take a significant amount of time. Logs and results are saved to the `output/` directory.

## Pipeline Components

-   **`config.py` (Configuration)**: Centralized `PipelineConfig` class for all tunable parameters.
-   **`data_loader_mot.py` (Data Loading)**: `DatasetLoader` for LTIR sequences and ground truth, used by the main pipeline.
-   **`preprocessing.py` (Preprocessing)**: Functions for normalization, Bilateral Filtering, and CLAHE.
-   **`detector.py` (Detection)**: `ForegroundDetector` using MOG2 and morphological operations for blob detection.
-   **`ukf_tracker.py` (Tracking - UKF)**: `UKFTrack` class for single object state estimation using UKF with constant velocity model.
-   **`multi_tracker.py` (Multi-Object Tracker & Association)**: `MultiObjectTracker` managing multiple tracks and performing Hungarian algorithm-based data association.
-   **`visualization.py` (Visualization)**: Drawing utilities for tracks and ground truth on frames.
-   **`evaluation.py` (Evaluation)**: Functions for IoU and standard MOT metrics calculation via `motmetrics`.
-   **`pipeline_runner.py` (Pipeline Orchestration)**: `MotionDetectionPipeline` class orchestrating the frame-by-frame processing for a sequence.
-   **`eda.py` (Exploratory Data Analysis Script)**: Contains initial functions for basic dataset loading, ground truth parsing, and visualization of a single frame with its annotation. Used for early-stage dataset familiarization.

## Parameter Tuning

Parameter tuning is performed by the `optimize.py` script using Bayesian Optimization (`scikit-optimize`).
-   The script is configured to target a specific preprocessing method (e.g., Bilateral Filter or Gaussian Blur) by adjusting the `PipelineConfig` defaults and the `param_space` within `optimize.py`.
-   It searches the defined `param_space` to find parameters that optimize a target MOT metric (e.g., IDF1) on the `TRAINING_SEQUENCES`.
-   The best found parameters for a given preprocessing configuration are then evaluated on the separate `VALIDATION_SEQUENCES`.
-   Detailed trial logs for each optimization run (e.g., one for Bilateral, one for Gaussian) are saved to `output/`.

*(To reproduce the results for both Gaussian and Bilateral pipelines, `optimize.py` would need to be configured and run separately for each preprocessing method, adjusting the `param_space` and the default filter choice in `config.py` and `preprocessing.py` accordingly.)*

## Performance Overview

*(This section reflects results from optimized parameters on the validation set for both Bilateral and Gaussian pipeline variants)*

The classical tracking pipeline was implemented with two primary preprocessing variants: one using **Bilateral Filtering** and another using **Gaussian Blurring**. Both variants underwent **separate parameter optimization processes** using Bayesian Optimization, primarily targeting the **IDF1 (ID F1 Score)** metric. The optimized parameters for each variant were then evaluated on the LTIR validation set (9 sequences). Key MOT (Multiple Object Tracking) metrics were calculated, and the average results across the validation set are summarized below:

| Metric             | Bilateral Pipeline (Optimized) | Gaussian Pipeline (Optimized) |
| :----------------- | :----------------------------- | :---------------------------- |
| MOTA               | -0.0907                        | 0.117                         |
| MOTP               | 0.8282                         | 0.826                         |
| IDF1               | 0.2115                         | 0.179                         |
| FP (Average)       | 183.78                         | 119.1                         |
| FN (Average)       | 318.0                          | 391.89                        |
| IDs (Average)      | 5.89                           | 4.56                          |
| Recall (Average)   | 0.4940                         | 0.4181                        |
| Precision (Average)| 0.7238                         | 0.7618                        |

The results clearly show performance trade-offs between the two preprocessing approaches after their respective optimizations:

-   The **Gaussian pipeline** achieved a better average **MOTA (0.117 vs -0.091)**, primarily due to significantly fewer **False Positives (FP)** (119.1 vs 183.8). This suggests that, with its optimized parameters, the Gaussian blur variant produced fewer spurious detections.
-   The **Bilateral pipeline**, on the other hand, achieved a better average **IDF1 score (0.2115 vs 0.179)** and higher average **Recall (0.4940 vs 0.4181)**. This indicates better track continuity and fewer missed targets (**False Negatives (FN)**: 318.0 vs 391.89), likely due to the edge-preserving nature of the bilateral filter helping the detector maintain a lock on targets.
-   Both methods showed comparable localization precision (**MOTP** around 0.83) and relatively low average ID switches (around 5), indicating that when a target was correctly tracked, its bounding box alignment was similar, and identity swaps were not the primary issue.
-   The overall modest MOTA and IDF1 scores underscore the inherent difficulty of the LTIR dataset for classical tracking methods, even with extensive parameter optimization.

Further details, including per-sequence breakdowns and the exact optimized parameters for each pipeline variant, can be found in the CSV files generated in the `output/` directory (e.g., results from the Bilateral optimization run and a separate Gaussian optimization run).

## Future Work

-   Implement and evaluate advanced motion models (e.g., Interacting Multiple Models - IMM) for the UKF.
-   Explore alternative background subtraction methods (e.g., KNN).
-   Enhance data association with IoU or Mahalanobis distance.
-   Further refine preprocessing steps or explore adaptive techniques.
-   Compare performance against modern deep learning-based trackers.
-   Add camera motion compensation for broader applicability.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the [MIT License](https://github.com/Sriram-PR/Project-HeatSeeker/blob/main/LICENSE.txt).