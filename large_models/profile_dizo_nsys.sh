#!/bin/bash
# NSYS Profiling Script for DiZO
#
# This script uses NVIDIA Nsight Systems (nsys) to profile DiZO training.
# nsys provides detailed GPU kernel traces, CUDA API calls, and CPU-GPU synchronization info.
#
# Prerequisites:
#   - NVIDIA Nsight Systems installed (comes with CUDA toolkit)
#   - nsys command available in PATH
#
# Usage:
#   ./profile_dizo_nsys.sh
#
# Output:
#   - profiler_logs/dizo_nsys_profile.nsys-rep - Main profile file
#   - Open with NVIDIA Nsight Systems GUI (nsight-sys) or export to other formats

# Configuration
export CUDA_VISIBLE_DEVICES=6
MODEL="facebook/opt-350m"
TASK="SST2"
NUM_STEPS=5
BATCH_SIZE=4
OUTPUT_DIR="./profiler_logs"

# Create output directory
mkdir -p $OUTPUT_DIR

echo "============================================"
echo "DiZO NSYS Profiling"
echo "============================================"
echo "Model: $MODEL"
echo "Task: $TASK"
echo "Steps: $NUM_STEPS"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Output: $OUTPUT_DIR"
echo "============================================"

# Check if nsys is available
if ! command -v nsys &> /dev/null; then
    echo "Warning: nsys not found in PATH"
    echo "Install NVIDIA Nsight Systems or add it to PATH"
    echo "Typical location: /usr/local/cuda/bin/nsys"
    echo ""
    echo "Falling back to PyTorch profiler only..."
    python profile_dizo.py \
        --model_name $MODEL \
        --task_name $TASK \
        --num_steps $NUM_STEPS \
        --batch_size $BATCH_SIZE \
        --output_dir $OUTPUT_DIR
    exit 0
fi

echo "Running nsys profiling..."

# nsys profile options:
# -t cuda,nvtx,osrt,cudnn,cublas: trace CUDA, NVTX markers, OS runtime, cuDNN, cuBLAS
# --cuda-memory-usage=true: track CUDA memory allocations
# --cudabacktrace=true: capture CUDA API backtraces
# --python-sampling=true: Python stack sampling
# --python-backtrace=cuda: Python backtraces for CUDA calls
# -o: output file name (without extension)
# --force-overwrite=true: overwrite existing profile

nsys profile \
    -t cuda,nvtx,osrt,cudnn,cublas \
    --cuda-memory-usage=true \
    --stats=true \
    --force-overwrite=true \
    -o $OUTPUT_DIR/dizo_nsys_profile \
    python profile_dizo.py \
        --model_name $MODEL \
        --task_name $TASK \
        --num_steps $NUM_STEPS \
        --batch_size $BATCH_SIZE \
        --output_dir $OUTPUT_DIR

echo ""
echo "============================================"
echo "Profiling Complete!"
echo "============================================"
echo ""
echo "Output files:"
echo "  - PyTorch Profiler: $OUTPUT_DIR/dizo_trace.json"
echo "  - TensorBoard logs: $OUTPUT_DIR/"
echo "  - NSYS Profile: $OUTPUT_DIR/dizo_nsys_profile.nsys-rep"
echo ""
echo "Visualization options:"
echo "  1. Chrome Trace Viewer:"
echo "     - Open chrome://tracing"
echo "     - Load $OUTPUT_DIR/dizo_trace.json"
echo ""
echo "  2. TensorBoard:"
echo "     tensorboard --logdir=$OUTPUT_DIR"
echo ""
echo "  3. NVIDIA Nsight Systems GUI:"
echo "     nsight-sys $OUTPUT_DIR/dizo_nsys_profile.nsys-rep"
echo ""
echo "  4. Export nsys to text report:"
echo "     nsys stats $OUTPUT_DIR/dizo_nsys_profile.nsys-rep"
echo "============================================"
