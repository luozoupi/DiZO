#!/bin/bash
# Run integrated MeZO/DiZO training step benchmarks

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default values
MODEL="opt-350m"
N_ITER=20
WARMUP=5
OUTPUT_DIR="benchmark_results/full_step"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --all)
            RUN_ALL=1
            shift
            ;;
        --n_iter)
            N_ITER="$2"
            shift 2
            ;;
        --breakdown)
            BREAKDOWN="--breakdown"
            shift
            ;;
        --output)
            OUTPUT="--output"
            shift
            ;;
        --skip_dizo)
            SKIP_DIZO="--skip_dizo"
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model MODEL     Model size (opt-350m, opt-1.3b, opt-2.7b, opt-6.7b, opt-13b)"
            echo "  --all             Run all model sizes"
            echo "  --n_iter N        Number of iterations (default: 20)"
            echo "  --breakdown       Show operation breakdown"
            echo "  --output          Save results to file"
            echo "  --skip_dizo       Skip DiZO benchmarks (MeZO only)"
            echo "  --help            Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Print banner
echo "=============================================================="
echo "  MeZO/DiZO Integrated Training Step Benchmark"
echo "=============================================================="
echo ""

# Check CUDA availability
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" || {
    echo "ERROR: CUDA not available"
    exit 1
}

# Print GPU info
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Check kernel availability
echo "Checking kernel availability..."
python -c "
import sys
sys.path.insert(0, 'Perturb_wise')
sys.path.insert(0, 'zo_foward_wise')

try:
    import triton
    print('  ✓ Triton available')
except ImportError:
    print('  ✗ Triton NOT available')

try:
    from triton_fused_perturb import fused_perturb_kernel_autotuned
    print('  ✓ Perturb kernels available')
except ImportError as e:
    print(f'  ✗ Perturb kernels NOT available: {e}')

try:
    from dizo_fused_kernels import fused_compute_norms
    print('  ✓ ZO-forward kernels available')
except ImportError as e:
    print(f'  ✗ ZO-forward kernels NOT available: {e}')
"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run benchmarks
if [[ -n "$RUN_ALL" ]]; then
    echo "Running benchmarks for ALL model sizes..."
    python benchmark_full_training_step.py \
        --all \
        --n_iter "$N_ITER" \
        --warmup "$WARMUP" \
        $BREAKDOWN $OUTPUT $SKIP_DIZO
else
    echo "Running benchmark for $MODEL..."
    python benchmark_full_training_step.py \
        --model "$MODEL" \
        --n_iter "$N_ITER" \
        --warmup "$WARMUP" \
        $BREAKDOWN $OUTPUT $SKIP_DIZO
fi

echo ""
echo "=============================================================="
echo "  Benchmark complete!"
echo "=============================================================="
