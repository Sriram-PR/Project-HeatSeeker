import os
import time
import numpy as np

# --- Import OpenCV versions with clear aliases ---
try:
    import cv2 as cv2_cpu
    print(f"Standard OpenCV version: {cv2_cpu.__version__}")
except ImportError:
    print("="*60)
    print("ERROR: Standard 'opencv-python' not found.")
    print("Please install it using: pip install opencv-python")
    print("="*60)
    exit()

try:
    import opencv_cuda as cv2_gpu
    print(f"CUDA OpenCV version: {cv2_gpu.__version__}")
    
    if cv2_gpu.cuda.getCudaEnabledDeviceCount() == 0:
        print("="*60)
        print("ERROR: No CUDA devices found by opencv_cuda.")
        print("Check your GPU drivers and CUDA installation.")
        print("="*60)
        exit()
    
    print(f"CUDA Device Count: {cv2_gpu.cuda.getCudaEnabledDeviceCount()}")
    
except ImportError:
    print("="*60)
    print("ERROR: 'opencv_cuda' not found.")
    print("Ensure a CUDA-enabled OpenCV version is installed and properly named.")
    print("="*60)
    exit()

# --- Attempt to Import from data_loader ---
try:
    from data_loader import (
        DATASET_BASE_PATH,
        get_sequence_frames,
        load_frame,
        normalize_frame
    )
except ImportError:
    print("="*60)
    print("ERROR: Could not import from data_loader.py.")
    print("Make sure the file exists in the current directory.")
    print("="*60)
    exit()

# --- Configuration ---
TEST_SEQUENCE = '8_car'
NUM_TEST_FRAMES = 30     # Load more frames for more stable timing/MOG2 buildup
TIMING_ITERATIONS = 50   # Number of times to run the operation for timing
WARMUP_ITERATIONS = 10   # Number of initial runs before timing

# --- Validation/Timing Helper Functions ---
def check_gpu_mat(gpu_mat, expected_shape=None, expected_dtype=None, operation_name="Operation"):
    """
    Checks GPU mat validity and downloads result for verification.
    
    Args:
        gpu_mat: The GpuMat to check
        expected_shape: Expected shape of downloaded NumPy array
        expected_dtype: Expected dtype of downloaded NumPy array
        operation_name: Name of operation for error messages
        
    Returns:
        (cpu_mat, is_valid): Tuple of downloaded array and boolean validity
    """
    if gpu_mat is None or gpu_mat.empty():
        print(f"  [FAIL] {operation_name}: Output GpuMat is empty.")
        return None, False
        
    try:
        cpu_mat = gpu_mat.download()
        
        if not isinstance(cpu_mat, np.ndarray):
            print(f"  [FAIL] {operation_name}: Downloaded result is not a NumPy array.")
            return None, False
            
        if expected_shape is not None and cpu_mat.shape != expected_shape:
            print(f"  [FAIL] {operation_name}: Shape mismatch. Expected {expected_shape}, Got {cpu_mat.shape}")
            return cpu_mat, False
            
        if expected_dtype is not None and cpu_mat.dtype != expected_dtype:
            print(f"  [FAIL] {operation_name}: Dtype mismatch. Expected {expected_dtype}, Got {cpu_mat.dtype}")
            return cpu_mat, False
            
        print(f"  [ OK ] {operation_name}: Output verified (Shape: {cpu_mat.shape}, Dtype: {cpu_mat.dtype}).")
        return cpu_mat, True
        
    except cv2_gpu.error as e:
        print(f"  [FAIL] {operation_name}: Error during download: {e}")
        return None, False

def time_operation(func, args_list, num_iterations, warmup_iterations):
    """
    Times a function call over multiple iterations.
    
    Args:
        func: Function to time
        args_list: List of arguments to pass to func
        num_iterations: Number of timed iterations
        warmup_iterations: Number of warmup iterations
        
    Returns:
        Average execution time in milliseconds
    """
    # Warm-up
    for _ in range(warmup_iterations):
        _ = func(*args_list)
        
    # Timing
    start_t = time.perf_counter()
    for _ in range(num_iterations):
        _ = func(*args_list)
    end_t = time.perf_counter()
    
    avg_time_ms = ((end_t - start_t) / num_iterations) * 1000
    return avg_time_ms

def time_gpu_operation(gpu_func, input_data, input_gpu_mat, output_gpu_mat,
                       num_iterations, warmup_iterations, stream, *extra_args):
    """
    Times a GPU operation with proper stream synchronization.
    
    Args:
        gpu_func: GPU function to call (should accept input, output, and stream)
        input_data: List of CPU input data (for upload)
        input_gpu_mat: Input GpuMat to upload to
        output_gpu_mat: Output GpuMat or None if function returns result
        num_iterations: Number of timed iterations
        warmup_iterations: Number of warmup iterations
        stream: CUDA stream
        *extra_args: Additional arguments to pass to gpu_func
        
    Returns:
        Average execution time in milliseconds
    """
    # Wrap the GPU function call to handle different API patterns
    def wrapped_gpu_call(input_mat, data_idx):
        # Only upload if we have valid input data and mat
        if input_data and input_mat is not None and data_idx < len(input_data) and input_data[data_idx % len(input_data)] is not None:
            input_mat.upload(input_data[data_idx % len(input_data)], stream=stream)
        
        if output_gpu_mat is not None:
            # Function that modifies output argument
            gpu_func(input_mat, output_gpu_mat, *extra_args, stream=stream)
            return output_gpu_mat
        else:
            # Function that returns new GpuMat
            return gpu_func(input_mat, output_gpu_mat, stream=stream)
    
    # Warm-up phase
    results = []  # Keep references to prevent premature deallocation
    for i in range(warmup_iterations):
        result = wrapped_gpu_call(input_gpu_mat, i)
        results.append(result)
    stream.waitForCompletion()
    results.clear()  # Clear references after warmup
    
    # Timing phase
    start_t = time.perf_counter()
    for i in range(num_iterations):
        result = wrapped_gpu_call(input_gpu_mat, i)
        results.append(result)
        stream.waitForCompletion()  # Ensure each operation is complete before timing next
    end_t = time.perf_counter()
    
    avg_time_ms = ((end_t - start_t) / num_iterations) * 1000
    
    # Return the last result for verification and time
    return avg_time_ms, results[-1] if results else None

# --- Main Validation ---
if __name__ == "__main__":
    print("--- OpenCV CUDA Functionality & Timing Validation ---")

    # Verify dataset path
    if 'PASTE_YOUR_FULL_LTIR_DATASET_PATH_HERE' in DATASET_BASE_PATH or not os.path.isdir(DATASET_BASE_PATH):
        print("="*60)
        print("ERROR: Set DATASET_BASE_PATH in data_loader.py")
        print("="*60)
        exit()

    # Load test frames
    print(f"\nLoading {NUM_TEST_FRAMES} frames from '{TEST_SEQUENCE}'...")
    frame_paths = get_sequence_frames(DATASET_BASE_PATH, TEST_SEQUENCE)
    if not frame_paths or len(frame_paths) < NUM_TEST_FRAMES:
        print("Error: Not enough frames found.")
        exit()
        
    cpu_frames_gray_8u = []
    original_shape = None
    
    for i in range(NUM_TEST_FRAMES):
        frame, _ = load_frame(frame_paths[i])
        if frame is None:
            print(f"Error: Failed to load frame {i}")
            exit()
            
        norm_frame = normalize_frame(frame)
        if norm_frame is None or norm_frame.dtype != np.uint8 or len(norm_frame.shape) != 2:
            print(f"Error: Frame {i} normalization failed.")
            exit()
            
        if original_shape is None:
            original_shape = norm_frame.shape
            
        cpu_frames_gray_8u.append(norm_frame)
        
    print(f"Frames loaded. Shape: {original_shape}")

    # Prepare GPU resources
    gpu_frame_in = cv2_gpu.cuda_GpuMat()
    gpu_frame_out = cv2_gpu.cuda_GpuMat()
    stream = cv2_gpu.cuda.Stream()

    # --- Test Suite ---
    validation_passed = True
    print(f"\n--- Running GPU Function & Timing Tests (Iterations: {TIMING_ITERATIONS}, Warmup: {WARMUP_ITERATIONS}) ---")

    # Test 1: Median Filter
    print("\n[Test 1: Median Filter (windowSize=5)]")
    cpu_time = time_operation(cv2_cpu.medianBlur, [cpu_frames_gray_8u[0], 5], 
                              TIMING_ITERATIONS, WARMUP_ITERATIONS)
    print(f"  Avg CPU Time: {cpu_time:.4f} ms")
    
    try:
        gpu_filter = cv2_gpu.cuda.createMedianFilter(cv2_gpu.CV_8UC1, windowSize=5)
        
        # Pre-allocate output with correct dimensions
        gpu_frame_out = cv2_gpu.cuda_GpuMat(
            (original_shape[1], original_shape[0]),  # OpenCV uses (width, height)
            cv2_gpu.CV_8UC1
        )
        
        # Time GPU operation
        gpu_time, gpu_result = time_gpu_operation(
            gpu_filter.apply, 
            cpu_frames_gray_8u, 
            gpu_frame_in, 
            gpu_frame_out,
            TIMING_ITERATIONS, 
            WARMUP_ITERATIONS, 
            stream
        )
        
        print(f"  Avg GPU Time: {gpu_time:.4f} ms")
        if gpu_time > 0:
            print(f"  Speedup: {cpu_time / gpu_time:.2f}x")
            
        # Verify result
        _, success = check_gpu_mat(gpu_result, original_shape, np.uint8, "GPU Median Result")
        if not success:
            validation_passed = False
            
    except (cv2_gpu.error, AttributeError, TypeError) as e:
        print(f"  [FAIL] GPU Median Filter: {e}")
        validation_passed = False

    # Test 2: CLAHE
    print("\n[Test 2: CLAHE (clip=2.0, grid=(8,8))]")
    cpu_clahe = cv2_cpu.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cpu_time = time_operation(cpu_clahe.apply, [cpu_frames_gray_8u[0]], 
                              TIMING_ITERATIONS, WARMUP_ITERATIONS)
    print(f"  Avg CPU Time: {cpu_time:.4f} ms")
    
    try:
        # Define CLAHE wrapper function that creates new CLAHE object each time
        def gpu_clahe_wrapper(in_mat, out_mat, stream):
            # out_mat is not used but included to match the expected interface
            clahe_gpu = cv2_gpu.cuda.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe_gpu.apply(in_mat, stream)

        # Time GPU operation with wrapper
        gpu_time, gpu_result = time_gpu_operation(
            gpu_clahe_wrapper,
            cpu_frames_gray_8u,
            gpu_frame_in,
            None,  # Output is returned by the function
            TIMING_ITERATIONS,
            WARMUP_ITERATIONS,
            stream
        )
        
        print(f"  Avg GPU Time: {gpu_time:.4f} ms")
        if gpu_time > 0:
            print(f"  Speedup: {cpu_time / gpu_time:.2f}x")
            
        # Verify result
        _, success = check_gpu_mat(gpu_result, original_shape, np.uint8, "GPU CLAHE Result")
        if not success:
            validation_passed = False
            
    except (cv2_gpu.error, AttributeError) as e:
        print(f"  [FAIL] GPU CLAHE: {e}")
        validation_passed = False

    # Test 3: MOG2 Background Subtraction
    print("\n[Test 3: MOG2 Apply (History=50, VarThresh=16)]")
    try:
        # Setup background subtractors
        cpu_mog2 = cv2_cpu.createBackgroundSubtractorMOG2(
            history=50, varThreshold=16, detectShadows=False)
        gpu_mog2 = cv2_gpu.cuda.createBackgroundSubtractorMOG2(
            history=50, varThreshold=16, detectShadows=False)
        
        # Pre-allocate output
        gpu_fgMask = cv2_gpu.cuda_GpuMat(
            (original_shape[1], original_shape[0]),
            cv2_gpu.CV_8UC1
        )
        
        # Build background models
        print("  Building background models...")
        build_frames = max(WARMUP_ITERATIONS, NUM_TEST_FRAMES - 1)
        
        for i in range(build_frames):
            # Train CPU model
            _ = cpu_mog2.apply(cpu_frames_gray_8u[i])
            
            # Train GPU model
            gpu_frame_in.upload(cpu_frames_gray_8u[i], stream=stream)
            gpu_mog2.apply(gpu_frame_in, fgmask=gpu_fgMask, learningRate=-1, stream=stream)
            
        stream.waitForCompletion()
        print("  Models built. Timing apply...")

        # Time CPU apply
        cpu_time = time_operation(
            cpu_mog2.apply, 
            [cpu_frames_gray_8u[-1]], 
            TIMING_ITERATIONS, 
            0  # No more warmup needed
        )
        print(f"  Avg CPU Time: {cpu_time:.4f} ms")

        # Define GPU MOG2 wrapper
        def gpu_mog2_wrapper(in_mat, out_mat, stream):
            gpu_mog2.apply(in_mat, fgmask=out_mat, learningRate=-1, stream=stream)
            return out_mat

        # Time GPU operation
        gpu_time, gpu_result = time_gpu_operation(
            gpu_mog2_wrapper,
            [cpu_frames_gray_8u[-1]],  # Use last frame repeatedly for timing
            gpu_frame_in,
            gpu_fgMask,
            TIMING_ITERATIONS,
            0,  # No more warmup needed
            stream
        )
        
        print(f"  Avg GPU Time: {gpu_time:.4f} ms")
        if gpu_time > 0:
            print(f"  Speedup: {cpu_time / gpu_time:.2f}x")
            
        # Verify result
        _, success = check_gpu_mat(gpu_result, original_shape, np.uint8, "GPU MOG2 Result")
        if not success:
            validation_passed = False
            
    except (cv2_gpu.error, AttributeError) as e:
        print(f"  [FAIL] GPU MOG2: {e}")
        validation_passed = False

    # Test 4: Farneback Optical Flow
    print("\n[Test 4: Farneback Optical Flow]")
    if NUM_TEST_FRAMES >= 2:
        frame0_cpu = cpu_frames_gray_8u[0]
        frame1_cpu = cpu_frames_gray_8u[1]
        
        # Reduce iterations for this slower operation
        reduced_iterations = max(1, TIMING_ITERATIONS // 5)
        reduced_warmup = max(1, WARMUP_ITERATIONS // 5)
        
        # Time CPU operation
        cpu_time = time_operation(
            cv2_cpu.calcOpticalFlowFarneback,
            [frame0_cpu, frame1_cpu, None, 0.5, 3, 15, 3, 5, 1.2, 0],
            reduced_iterations,
            reduced_warmup
        )
        print(f"  Avg CPU Time: {cpu_time:.4f} ms")
        
        try:
            # Upload frames to GPU
            gpu_frame0 = cv2_gpu.cuda_GpuMat()
            gpu_frame1 = cv2_gpu.cuda_GpuMat()
            gpu_frame0.upload(frame0_cpu)
            gpu_frame1.upload(frame1_cpu)
            
            # Pre-allocate output
            gpu_flow_out = cv2_gpu.cuda_GpuMat(
                (original_shape[1], original_shape[0]),  # Width, height
                cv2_gpu.CV_32FC2  # 2-channel float for optical flow
            )

            # Create flow calculator with API adaptability
            try:
                flow_calculator = cv2_gpu.cuda_FarnebackOpticalFlow.create(
                    numLevels=5, pyrScale=0.5, winSize=13, numIters=10, 
                    polyN=5, polySigma=1.1, flags=0
                )
            except AttributeError:
                flow_calculator = cv2_gpu.cuda.FarnebackOpticalFlow_create(
                    numLevels=5, pyrScale=0.5, winSize=13, numIters=10, 
                    polyN=5, polySigma=1.1, flags=0
                )

            # Define flow wrapper
            def gpu_flow_wrapper(input_mat, output_mat, stream):
                # This function doesn't use the standard input arguments from time_gpu_operation
                # because we already have our frames uploaded separately
                flow_calculator.calc(gpu_frame0, gpu_frame1, gpu_flow_out, stream=stream)
                return gpu_flow_out

            # Time GPU operation with special wrapper
            gpu_time, gpu_result = time_gpu_operation(
                gpu_flow_wrapper,
                [None],  # Dummy input list - not used by wrapper
                None,    # Dummy input mat - not used by wrapper
                None,    # Output returned by function
                reduced_iterations,
                reduced_warmup,
                stream
            )
            
            print(f"  Avg GPU Time: {gpu_time:.4f} ms")
            if gpu_time > 0:
                print(f"  Speedup: {cpu_time / gpu_time:.2f}x")
                
            # Verify result - expected shape is (height, width, 2)
            expected_flow_shape_np = (original_shape[0], original_shape[1], 2)
            _, success = check_gpu_mat(
                gpu_result, 
                expected_flow_shape_np, 
                np.float32, 
                "GPU Farneback Result"
            )
            if not success:
                validation_passed = False
                
        except (cv2_gpu.error, AttributeError) as e:
            print(f"  [FAIL] GPU Farneback Flow: {e}")
            validation_passed = False
    else:
        print("  [SKIP] Farneback Flow: Need at least 2 frames.")

    # --- Final Summary ---
    print("\n--- Validation Summary ---")
    if validation_passed:
        print("✓ All tested OpenCV CUDA functions were available and executed without errors.")
        print("✓ Timing comparisons show potential speedups from GPU acceleration.")
    else:
        print("✗ One or more OpenCV CUDA function tests failed or verification checks failed.")
        print("  Check the error messages above for details.")