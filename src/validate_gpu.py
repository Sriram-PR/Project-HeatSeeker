import os
import time

import numpy as np

DATASET_BASE_PATH = r"./data/ltir_v1_0_8bit_16bit"

try:
    import cv2 as cv2_cpu

    print(f"Standard OpenCV version: {cv2_cpu.__version__}")
except ImportError:
    print("ERROR: Standard 'opencv-python' not found.")
    exit()

try:
    import opencv_cuda as cv2_gpu

    print(f"CUDA OpenCV version: {cv2_gpu.__version__}")
    if cv2_gpu.cuda.getCudaEnabledDeviceCount() == 0:
        print("ERROR: No CUDA devices found.")
        exit()
    print(f"CUDA Device Count: {cv2_gpu.cuda.getCudaEnabledDeviceCount()}")
except ImportError:
    print("ERROR: 'opencv_cuda' module not found.")
    exit()

try:
    from data_loader_mot import DatasetLoader
except ImportError:
    print("ERROR: Could not import from data_loader_mot.py.")
    exit()


def normalize_image(frame):
    if frame is None:
        return None

    if frame.dtype == np.uint16:
        return cv2_cpu.normalize(
            frame, None, 0, 255, cv2_cpu.NORM_MINMAX, dtype=cv2_cpu.CV_8U
        )
    elif frame.dtype == np.uint8:
        if len(frame.shape) == 3:
            return cv2_cpu.cvtColor(frame, cv2_cpu.COLOR_BGR2GRAY)
        return frame

    return None


def check_gpu_mat(
    gpu_mat, expected_shape=None, expected_dtype=None, operation_name="Operation"
):
    if gpu_mat is None or gpu_mat.empty():
        print(f"  [FAIL] {operation_name}: Output GpuMat empty.")
        return None, False

    try:
        cpu_mat = gpu_mat.download()
        if not isinstance(cpu_mat, np.ndarray):
            print(f"  [FAIL] {operation_name}: Downloaded result not NumPy.")
            return None, False

        if expected_shape is not None and cpu_mat.shape != expected_shape:
            print(
                f"  [FAIL] {operation_name}: Shape mismatch. Exp {expected_shape}, Got {cpu_mat.shape}"
            )
            return cpu_mat, False

        if expected_dtype is not None and cpu_mat.dtype != expected_dtype:
            print(
                f"  [FAIL] {operation_name}: Dtype mismatch. Exp {expected_dtype}, Got {cpu_mat.dtype}"
            )
            return cpu_mat, False

        print(
            f"  [ OK ] {operation_name}: Output verified (Shape: {cpu_mat.shape}, Dtype: {cpu_mat.dtype})."
        )
        return cpu_mat, True

    except cv2_gpu.error as e:
        print(f"  [FAIL] {operation_name}: Error during download: {e}")
        return None, False


def time_operation(func, args_list, num_iterations, warmup_iterations):
    for _ in range(warmup_iterations):
        _ = func(*args_list)

    start_t = time.perf_counter()

    for _ in range(num_iterations):
        _ = func(*args_list)

    end_t = time.perf_counter()

    return ((end_t - start_t) / num_iterations) * 1000


# --- Configuration ---
TEST_SEQUENCE = "8_car"
NUM_TEST_FRAMES = 30
TIMING_ITERATIONS = 50
WARMUP_ITERATIONS = 10


if __name__ == "__main__":
    print(
        "--- OpenCV CUDA Functionality & Timing Validation (Gaussian, CLAHE, MOG2, Farneback) ---"
    )

    if not os.path.isdir(DATASET_BASE_PATH):
        print(f"ERROR: DATASET_BASE_PATH '{DATASET_BASE_PATH}' not found.")
        exit()

    print(f"\nLoading {NUM_TEST_FRAMES} frames from '{TEST_SEQUENCE}'...")

    try:
        loader = DatasetLoader(DATASET_BASE_PATH, TEST_SEQUENCE)
    except Exception as e:
        print(f"Error initializing Loader: {e}")
        exit()

    if len(loader) < NUM_TEST_FRAMES:
        print(f"Error: Not enough frames ({len(loader)}).")
        exit()

    cpu_frames_gray_8u = []
    original_shape = None

    for i in range(NUM_TEST_FRAMES):
        frame, _ = loader.load_frame(i)

        if frame is None:
            print(f"Error: Load failed frame {i}")
            exit()

        norm_frame = normalize_image(frame)

        if norm_frame is None:
            print(f"Error: Norm failed frame {i}")
            exit()

        if original_shape is None:
            original_shape = norm_frame.shape

        cpu_frames_gray_8u.append(norm_frame)

    print(f"Frames loaded. Shape: {original_shape}")

    gpu_frame_in = cv2_gpu.cuda_GpuMat()
    gpu_frame_out = cv2_gpu.cuda_GpuMat()
    stream = cv2_gpu.cuda.Stream()
    validation_passed = True
    test_results = {}

    print(
        f"\n--- Running GPU Function & Timing Tests (Iterations: {TIMING_ITERATIONS}, Warmup: {WARMUP_ITERATIONS}) ---"
    )

    # Test 1: Gaussian Filter
    print("\n[Test 1: Gaussian Filter (ksize=5x5, sigmaX=1.5)]")
    gauss_ksize = (5, 5)
    gauss_sigmaX = 1.5
    test_results["Gaussian"] = False  # Default to False

    cpu_time = time_operation(
        cv2_cpu.GaussianBlur,
        [cpu_frames_gray_8u[0], gauss_ksize, gauss_sigmaX],
        TIMING_ITERATIONS,
        WARMUP_ITERATIONS,
    )
    print(f"  Avg CPU Time: {cpu_time:.4f} ms")

    try:
        gpu_filter = cv2_gpu.cuda.createGaussianFilter(
            cv2_gpu.CV_8UC1, cv2_gpu.CV_8UC1, ksize=gauss_ksize, sigma1=gauss_sigmaX
        )
        gpu_frame_out.upload(cpu_frames_gray_8u[0])

        # Warmup
        for _ in range(WARMUP_ITERATIONS):
            gpu_frame_in.upload(cpu_frames_gray_8u[0], stream=stream)
            gpu_filter.apply(gpu_frame_in, gpu_frame_out, stream=stream)

        stream.waitForCompletion()
        start_t = time.perf_counter()

        # Timing
        for i in range(TIMING_ITERATIONS):
            gpu_frame_in.upload(
                cpu_frames_gray_8u[i % len(cpu_frames_gray_8u)], stream=stream
            )
            gpu_filter.apply(gpu_frame_in, gpu_frame_out, stream=stream)
            stream.waitForCompletion()

        end_t = time.perf_counter()
        gpu_time = ((end_t - start_t) / TIMING_ITERATIONS) * 1000
        gpu_result = gpu_frame_out
        print(f"  Avg GPU Time: {gpu_time:.4f} ms")

        if gpu_time > 0:
            print(f"  Speedup: {cpu_time / gpu_time:.2f}x")

        _, success = check_gpu_mat(
            gpu_result, original_shape, np.uint8, "GPU Gaussian Result"
        )
        test_results["Gaussian"] = success  # Store result

        if not success:
            validation_passed = False

    except (cv2_gpu.error, AttributeError, TypeError) as e:
        print(f"  [FAIL] GPU Gaussian Filter: {e}")
        validation_passed = False

    # Test 2: CLAHE
    print("\n[Test 2: CLAHE (clip=2.0, grid=(8,8))]")
    test_results["CLAHE"] = False  # Default to False

    cpu_clahe = cv2_cpu.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cpu_time = time_operation(
        cpu_clahe.apply, [cpu_frames_gray_8u[0]], TIMING_ITERATIONS, WARMUP_ITERATIONS
    )
    print(f"  Avg CPU Time: {cpu_time:.4f} ms")

    try:
        results_clahe = []

        # Warmup
        for _ in range(WARMUP_ITERATIONS):
            gpu_frame_in.upload(cpu_frames_gray_8u[0], stream=stream)
            clahe_gpu = cv2_gpu.cuda.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            _ = clahe_gpu.apply(gpu_frame_in, stream)

        stream.waitForCompletion()
        start_t = time.perf_counter()

        # Timing
        for i in range(TIMING_ITERATIONS):
            gpu_frame_in.upload(
                cpu_frames_gray_8u[i % len(cpu_frames_gray_8u)], stream=stream
            )
            clahe_gpu = cv2_gpu.cuda.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gpu_result_clahe = clahe_gpu.apply(gpu_frame_in, stream)
            results_clahe.append(gpu_result_clahe)
            stream.waitForCompletion()

        end_t = time.perf_counter()
        gpu_time = ((end_t - start_t) / TIMING_ITERATIONS) * 1000
        gpu_result = results_clahe[-1]
        print(f"  Avg GPU Time: {gpu_time:.4f} ms")

        if gpu_time > 0:
            print(f"  Speedup: {cpu_time / gpu_time:.2f}x")

        _, success = check_gpu_mat(
            gpu_result, original_shape, np.uint8, "GPU CLAHE Result"
        )
        test_results["CLAHE"] = success  # Store result

        if not success:
            validation_passed = False

    except (cv2_gpu.error, AttributeError) as e:
        print(f"  [FAIL] GPU CLAHE: {e}")
        validation_passed = False

    # Test 3: MOG2 Background Subtraction
    print("\n[Test 3: MOG2 Apply (History=50, VarThresh=16)]")
    test_results["MOG2"] = False  # Default to False

    try:
        cpu_mog2 = cv2_cpu.createBackgroundSubtractorMOG2(
            history=50, varThreshold=16, detectShadows=False
        )
        gpu_mog2 = cv2_gpu.cuda.createBackgroundSubtractorMOG2(
            history=50, varThreshold=16, detectShadows=False
        )
        gpu_fgMask = cv2_gpu.cuda_GpuMat(
            original_shape[0], original_shape[1], cv2_gpu.CV_8UC1
        )

        print("  Building background models...")
        build_frames = max(WARMUP_ITERATIONS, NUM_TEST_FRAMES - 1)

        for i in range(build_frames):
            _ = cpu_mog2.apply(cpu_frames_gray_8u[i])
            gpu_frame_in.upload(cpu_frames_gray_8u[i], stream=stream)
            gpu_mog2.apply(
                gpu_frame_in, fgmask=gpu_fgMask, learningRate=-1, stream=stream
            )

        stream.waitForCompletion()
        print("  Models built. Timing apply...")

        cpu_time = time_operation(
            cpu_mog2.apply, [cpu_frames_gray_8u[-1]], TIMING_ITERATIONS, 0
        )
        print(f"  Avg CPU Time: {cpu_time:.4f} ms")

        start_t = time.perf_counter()

        for i in range(TIMING_ITERATIONS):
            gpu_frame_in.upload(cpu_frames_gray_8u[-1], stream=stream)
            gpu_mog2.apply(
                gpu_frame_in, fgmask=gpu_fgMask, learningRate=-1, stream=stream
            )
            stream.waitForCompletion()

        end_t = time.perf_counter()
        gpu_time = ((end_t - start_t) / TIMING_ITERATIONS) * 1000
        gpu_result = gpu_fgMask
        print(f"  Avg GPU Time: {gpu_time:.4f} ms")

        if gpu_time > 0:
            print(f"  Speedup: {cpu_time / gpu_time:.2f}x")

        _, success = check_gpu_mat(
            gpu_result, original_shape, np.uint8, "GPU MOG2 Result"
        )
        test_results["MOG2"] = success  # Store result

        if not success:
            validation_passed = False

    except (cv2_gpu.error, AttributeError) as e:
        print(f"  [FAIL] GPU MOG2: {e}")
        validation_passed = False

    # Test 4: Farneback Optical Flow
    print("\n[Test 4: Farneback Optical Flow]")
    test_results["Farneback"] = False  # Default to False

    if NUM_TEST_FRAMES >= 2:
        frame0_cpu = cpu_frames_gray_8u[0]
        frame1_cpu = cpu_frames_gray_8u[1]
        reduced_iterations = max(1, TIMING_ITERATIONS // 5)
        reduced_warmup = max(1, WARMUP_ITERATIONS // 5)

        cpu_time = time_operation(
            cv2_cpu.calcOpticalFlowFarneback,
            [frame0_cpu, frame1_cpu, None, 0.5, 3, 15, 3, 5, 1.2, 0],
            reduced_iterations,
            reduced_warmup,
        )
        print(f"  Avg CPU Time: {cpu_time:.4f} ms")

        try:
            gpu_frame0 = cv2_gpu.cuda_GpuMat()
            gpu_frame1 = cv2_gpu.cuda_GpuMat()
            gpu_frame0.upload(frame0_cpu)
            gpu_frame1.upload(frame1_cpu)
            gpu_flow_out = cv2_gpu.cuda_GpuMat(
                original_shape[0], original_shape[1], cv2_gpu.CV_32FC2
            )

            try:
                flow_calculator = cv2_gpu.cuda.FarnebackOpticalFlow_create(
                    numLevels=3,
                    pyrScale=0.5,
                    winSize=15,
                    numIters=3,
                    polyN=5,
                    polySigma=1.2,
                    flags=0,
                )
            except AttributeError:
                flow_calculator = cv2_gpu.cuda_FarnebackOpticalFlow.create(
                    numLevels=3,
                    pyrScale=0.5,
                    winSize=15,
                    numIters=3,
                    polyN=5,
                    polySigma=1.2,
                    flags=0,
                )

            results = []

            # Warmup
            for _ in range(reduced_warmup):
                flow_calculator.calc(
                    gpu_frame0, gpu_frame1, gpu_flow_out, stream=stream
                )

            stream.waitForCompletion()
            start_t = time.perf_counter()

            # Timing
            for _ in range(reduced_iterations):
                flow_calculator.calc(
                    gpu_frame0, gpu_frame1, gpu_flow_out, stream=stream
                )
                results.append(gpu_flow_out)
                stream.waitForCompletion()

            end_t = time.perf_counter()
            gpu_time = ((end_t - start_t) / reduced_iterations) * 1000
            gpu_result = results[-1]
            print(f"  Avg GPU Time: {gpu_time:.4f} ms")

            if gpu_time > 0:
                print(f"  Speedup: {cpu_time / gpu_time:.2f}x")

            expected_flow_shape = (original_shape[0], original_shape[1], 2)
            _, success = check_gpu_mat(
                gpu_result, expected_flow_shape, np.float32, "GPU Farneback Result"
            )
            test_results["Farneback"] = success  # Store result

            if not success:
                validation_passed = False

        except (cv2_gpu.error, AttributeError) as e:
            print(f"  [FAIL] GPU Farneback Flow: {e}")
            validation_passed = False

    else:
        print("  [SKIP] Farneback Flow: Need at least 2 frames.")

    print("\n--- Validation Summary ---")

    if validation_passed:
        print(
            "✓ All tested OpenCV CUDA functions executed successfully and passed verification."
        )
    else:
        print(
            "✗ One or more OpenCV CUDA function tests failed execution or verification."
        )
        print("  Failed Tests:", [k for k, v in test_results.items() if not v])
        print("  Check the error messages above for details.")
