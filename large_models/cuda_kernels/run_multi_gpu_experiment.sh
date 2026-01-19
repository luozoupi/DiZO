#!/bin/bash
# Multi-GPU Parallel MeZO Experiment Runner
# For debugging with GPUs 1,2 and conda venv py310_2

set -e

# Change to script directory first
cd "$(dirname "$0")"

# Try to activate conda environment if not already activated
if [[ -z "$CONDA_DEFAULT_ENV" ]] || [[ "$CONDA_DEFAULT_ENV" != "py310_2" ]]; then
    echo "Attempting to activate conda environment py310_2..."
    
    # Try different conda initialization paths
    CONDASCRIPT=""
    if [[ -f ~/anaconda3/etc/profile.d/conda.sh ]]; then
        CONDASCRIPT=~/anaconda3/etc/profile.d/conda.sh
    elif [[ -f ~/miniconda3/etc/profile.d/conda.sh ]]; then
        CONDASCRIPT=~/miniconda3/etc/profile.d/conda.sh
    elif [[ -f ~/.conda/etc/profile.d/conda.sh ]]; then
        CONDASCRIPT=~/.conda/etc/profile.d/conda.sh
    fi
    
    if [[ -n "$CONDASCRIPT" ]]; then
        source "$CONDASCRIPT"
        conda activate py310_2 || {
            echo "Error: Failed to activate conda environment py310_2"
            echo "Please activate it manually: conda activate py310_2"
            exit 1
        }
    elif command -v conda &> /dev/null; then
        # Conda might already be in PATH
        echo "Conda found in PATH, using directly..."
        conda activate py310_2 || {
            echo "Warning: Could not activate py310_2 via conda command"
            echo "Continuing with current environment..."
        }
    else
        echo "Warning: Could not find conda initialization script"
        echo "If you're already in py310_2, continuing..."
        if [[ "$CONDA_DEFAULT_ENV" != "py310_2" ]]; then
            echo "Error: Not in py310_2 environment and cannot activate it"
            echo "Please activate manually: conda activate py310_2"
            exit 1
        fi
    fi
else
    echo "Already in conda environment: $CONDA_DEFAULT_ENV"
fi

# Verify Python and check for torch
echo "Checking Python environment..."
if ! command -v python &> /dev/null && ! command -v python3 &> /dev/null; then
    echo "Error: Python not found in PATH"
    exit 1
fi

# Find the correct Python - prioritize conda environment's Python
PYTHON_CMD=""

# First check if we're in a conda environment and use its Python directly
if [[ -n "$CONDA_PREFIX" ]]; then
    CONDA_PYTHON="$CONDA_PREFIX/bin/python"
    if [[ -f "$CONDA_PYTHON" ]]; then
        PYTHON_CMD="$CONDA_PYTHON"
        echo "Using conda environment Python: $PYTHON_CMD"
    fi
fi

# If not found, try from PATH (should be conda's if env is active)
if [[ -z "$PYTHON_CMD" ]]; then
    if command -v python &> /dev/null; then
        PYTHON_CMD=$(which python)
        echo "Using Python from PATH: $PYTHON_CMD"
    elif command -v python3 &> /dev/null; then
        PYTHON_CMD=$(which python3)
        echo "Using Python3 from PATH: $PYTHON_CMD"
    else
        echo "Error: Python not found in PATH"
        exit 1
    fi
fi

# Verify Python path
echo "Python executable: $PYTHON_CMD"

# Verify Python works
$PYTHON_CMD -c "import sys; print(f'Python: {sys.version}'); print(f'Python executable: {sys.executable}')" || {
    echo "Error: Python not working: $PYTHON_CMD"
    exit 1
}

# Check for torch
echo "Checking for PyTorch in: $PYTHON_CMD"
if ! $PYTHON_CMD -c "import torch" 2>/dev/null; then
    echo "Error: PyTorch not found in current Python environment"
    echo "Python executable: $PYTHON_CMD"
    echo "Current environment: ${CONDA_DEFAULT_ENV:-unknown}"
    echo "CONDA_PREFIX: ${CONDA_PREFIX:-not set}"
    echo ""
    echo "Trying to locate PyTorch installation..."
    $PYTHON_CMD -c "import sys; print('Python path:', sys.executable); print('Python paths:'); import sys; [print('  ', p) for p in sys.path]" 2>&1 || true
    echo ""
    echo "Please install PyTorch:"
    echo "  conda install pytorch torchvision torchaudio pytorch-cuda -c pytorch -c nvidia"
    echo "  or"
    echo "  pip install torch torchvision torchaudio"
    exit 1
fi

# Print environment info
$PYTHON_CMD -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'CUDA devices: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('Warning: CUDA not available')
" || {
    echo "Warning: Could not query PyTorch/CUDA information"
}

# Set CUDA devices (use GPUs 1 and 2 for debugging)
export CUDA_VISIBLE_DEVICES=1,2

echo ""
echo "=========================================="
echo "Multi-GPU Parallel MeZO Experiment"
echo "=========================================="
echo "Conda env: ${CONDA_DEFAULT_ENV:-unknown}"
echo "CUDA devices: 1,2 (visible as 0,1 in script)"
echo ""

# Run with small model first for testing (can override with command line args)
MODEL=${1:-opt-350m}
MAIN_GPU=${2:-0}  # GPU 1 becomes visible as 0
SIDE_GPU=${3:-1}  # GPU 2 becomes visible as 1
N_ITER=${4:-20}

# Environment variables for configuration:
# USE_REAL_MODEL=1 - Use real OPT model for forward passes
# USE_PYTORCH_PERTURB=1 - Use safe PyTorch perturb/update
# BATCH_SIZE=N - Override batch size (auto-selected if not set)
# SEQ_LEN=N - Sequence length (default: 512)
# DTYPE=auto|fp32|fp16|bf16 - Data type (auto for automatic selection)
# INCLUDE_OPTIMIZED=1 - Include optimized multi-GPU benchmark

echo "Configuration:"
echo "  Model: $MODEL"
echo "  Main GPU: $MAIN_GPU (physical GPU 1)"
echo "  Side GPU: $SIDE_GPU (physical GPU 2)"
echo "  Iterations: $N_ITER"
echo ""

# Export DEBUG_PYTHON to help debug if needed
export DEBUG_PYTHON=1

# Build command options
EXTRA_FLAGS=""

# Use safe PyTorch perturb/update if requested
if [[ "${USE_PYTORCH_PERTURB:-0}" == "1" ]]; then
    EXTRA_FLAGS="$EXTRA_FLAGS --use_pytorch_perturb"
    echo "Using safe PyTorch perturb/update (no CUDA extensions)"
fi

# Use real OPT model for actual forward passes
if [[ "${USE_REAL_MODEL:-0}" == "1" ]]; then
    EXTRA_FLAGS="$EXTRA_FLAGS --use_real_model"
    
    # Batch size (auto-selected if not specified)
    if [[ -n "${BATCH_SIZE:-}" ]]; then
        EXTRA_FLAGS="$EXTRA_FLAGS --batch_size $BATCH_SIZE"
        echo "Using batch_size=$BATCH_SIZE"
    else
        echo "Batch size: auto (based on model size and GPU memory)"
    fi
    
    # Sequence length (default: 512)
    SEQ_LEN=${SEQ_LEN:-512}
    EXTRA_FLAGS="$EXTRA_FLAGS --seq_len $SEQ_LEN"
    echo "Using seq_len=$SEQ_LEN"
    
    # Data type
    DTYPE=${DTYPE:-auto}
    EXTRA_FLAGS="$EXTRA_FLAGS --dtype $DTYPE"
    echo "Using dtype=$DTYPE"
    
    echo "Using REAL OPT model for forward passes"
fi

# Include optimized multi-GPU benchmark
if [[ "${INCLUDE_OPTIMIZED:-0}" == "1" ]]; then
    EXTRA_FLAGS="$EXTRA_FLAGS --compare_optimized"
    echo "Including OPTIMIZED multi-GPU benchmark"
fi

echo ""
echo "Running benchmark..."
$PYTHON_CMD benchmark_multi_gpu_parallel.py \
    --model "$MODEL" \
    --main_gpu "$MAIN_GPU" \
    --side_gpu "$SIDE_GPU" \
    --n_iter "$N_ITER" \
    --compare_all \
    --debug \
    $EXTRA_FLAGS

echo ""
echo "Experiment completed!"

# Print helpful commands for larger models
echo ""
echo "=========================================="
echo "Commands for larger models:"
echo "=========================================="
echo ""
echo "# OPT-1.3B (auto batch size and dtype):"
echo "USE_REAL_MODEL=1 INCLUDE_OPTIMIZED=1 bash run_multi_gpu_experiment.sh opt-1.3b"
echo ""
echo "# OPT-2.7B (auto batch size and dtype):"
echo "USE_REAL_MODEL=1 INCLUDE_OPTIMIZED=1 bash run_multi_gpu_experiment.sh opt-2.7b"
echo ""
echo "# Custom batch size and dtype:"
echo "USE_REAL_MODEL=1 BATCH_SIZE=4 SEQ_LEN=1024 DTYPE=fp16 bash run_multi_gpu_experiment.sh opt-1.3b"
