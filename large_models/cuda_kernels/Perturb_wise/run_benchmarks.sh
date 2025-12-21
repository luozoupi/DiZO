#!/bin/bash
# =============================================================================
# Sequential Benchmark Script for DiZO Kernel Performance Testing
# =============================================================================
#
# This script runs benchmarks sequentially to ensure memory isolation between
# different model sizes and methods. Each method runs in a separate Python
# process, which fully releases GPU memory before the next method.
#
# Usage:
#   ./run_benchmarks.sh                              # Run all methods for all models
#   ./run_benchmarks.sh opt-13b                      # Run all methods for OPT-13B
#   ./run_benchmarks.sh opt-13b cuda                 # Run only CUDA for OPT-13B
#   ./run_benchmarks.sh opt-13b all complete         # Run all + complete step for OPT-13B
#   ./run_benchmarks.sh opt-13b cuda complete        # Run CUDA + complete step for OPT-13B
#
# Methods: cuda, triton_auto, triton_v1, triton_v2, pytorch_baseline, pytorch_fused, chunked, all
# For complete step: pytorch, triton, cuda, all
#
# Output:
#   Results are saved to benchmark_results/ directory with timestamps
#
# =============================================================================

set -e  # Exit on error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default parameters
N_ITER=${N_ITER:-50}
WARMUP=${WARMUP:-10}
CUDA_DEVICE=${CUDA_VISIBLE_DEVICES:-0}

# Output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="benchmark_results/${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

# Log file
LOG_FILE="${OUTPUT_DIR}/benchmark.log"

# Available methods for individual kernel benchmarks
INDIVIDUAL_METHODS=("pytorch_baseline" "pytorch_fused" "chunked" "triton_v1" "triton_v2" "triton_auto" "cuda")
# Available methods for complete step benchmarks  
COMPLETE_METHODS=("pytorch" "triton" "cuda")

echo "=============================================="
echo "DiZO Kernel Benchmark Suite"
echo "=============================================="
echo "Script directory: $SCRIPT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "CUDA device: $CUDA_DEVICE"
echo "Iterations: $N_ITER"
echo "Warmup: $WARMUP"
echo "=============================================="

# Function to run a single method benchmark
run_single_method() {
    local model=$1
    local method=$2
    local output_file="${OUTPUT_DIR}/${model}_${method}.txt"
    
    echo ""
    echo "======================================"
    echo "Running: $model --method $method"
    echo "Output: $output_file"
    echo "======================================"
    
    # Run benchmark and capture output
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python benchmark_kernels.py \
        --model "$model" \
        --method "$method" \
        --n_iter "$N_ITER" \
        --warmup "$WARMUP" \
        --skip_correctness 2>&1 | tee "$output_file"
    
    local exit_code=${PIPESTATUS[0]}
    
    if [ $exit_code -eq 0 ]; then
        echo "[SUCCESS] $model $method completed"
    else
        echo "[WARNING] $model $method exited with code $exit_code"
    fi
    
    # Give GPU time to release memory
    sleep 2
    
    return $exit_code
}

# Function to run complete step benchmark for one method
run_complete_step_method() {
    local model=$1
    local method=$2
    local output_file="${OUTPUT_DIR}/${model}_complete_${method}.txt"
    
    echo ""
    echo "======================================"
    echo "Running COMPLETE STEP: $model --method $method"
    echo "Output: $output_file"
    echo "======================================"
    
    # Map method name for complete step
    local complete_method="$method"
    case "$method" in
        pytorch_baseline|pytorch_fused|chunked)
            complete_method="pytorch_baseline"  # Will be mapped to pytorch in script
            ;;
        triton_v1|triton_v2|triton_auto)
            complete_method="triton_auto"  # Will be mapped to triton in script
            ;;
    esac
    
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python benchmark_kernels.py \
        --model "$model" \
        --method "$complete_method" \
        --n_iter "$N_ITER" \
        --warmup "$WARMUP" \
        --skip_correctness \
        --complete_step_only 2>&1 | tee "$output_file"
    
    local exit_code=${PIPESTATUS[0]}
    
    if [ $exit_code -eq 0 ]; then
        echo "[SUCCESS] $model complete step ($method) completed"
    else
        echo "[WARNING] $model complete step ($method) exited with code $exit_code"
    fi
    
    sleep 2
    return $exit_code
}

# Parse arguments
MODEL_ARG=${1:-all}
METHOD_ARG=${2:-all}
COMPLETE_STEP=${3:-}

# Define model sizes to test
if [ "$MODEL_ARG" == "all" ]; then
    MODELS=("opt-350m" "opt-2.7b" "opt-6.7b" "opt-13b")
else
    MODELS=("$MODEL_ARG")
fi

# Define methods to test
if [ "$METHOD_ARG" == "all" ]; then
    METHODS=("${INDIVIDUAL_METHODS[@]}")
else
    METHODS=("$METHOD_ARG")
fi

# Run benchmarks
echo ""
echo "=============================================="
echo "Starting benchmark suite..."
echo "Models: ${MODELS[*]}"
echo "Methods: ${METHODS[*]}"
echo "Complete step: ${COMPLETE_STEP:-no}"
echo "=============================================="

# Run individual kernel benchmarks for each model and method
for model in "${MODELS[@]}"; do
    echo ""
    echo "=============================================="
    echo ">>> Processing model: $model <<<"
    echo "=============================================="
    
    for method in "${METHODS[@]}"; do
        run_single_method "$model" "$method" || true
    done
done

# If complete step is requested, run those separately (with fresh memory)
if [ "$COMPLETE_STEP" == "complete" ]; then
    echo ""
    echo "=============================================="
    echo "Running COMPLETE MeZO STEP benchmarks..."
    echo "=============================================="
    
    # Determine which complete step methods to run based on METHOD_ARG
    if [ "$METHOD_ARG" == "all" ]; then
        COMPLETE_TO_RUN=("${COMPLETE_METHODS[@]}")
    else
        # Map individual method to complete step method
        case "$METHOD_ARG" in
            pytorch_baseline|pytorch_fused|chunked)
                COMPLETE_TO_RUN=("pytorch")
                ;;
            triton_v1|triton_v2|triton_auto)
                COMPLETE_TO_RUN=("triton")
                ;;
            cuda)
                COMPLETE_TO_RUN=("cuda")
                ;;
            *)
                COMPLETE_TO_RUN=("$METHOD_ARG")
                ;;
        esac
    fi
    
    for model in "${MODELS[@]}"; do
        echo ""
        echo ">>> Complete step for: $model <<<"
        
        for method in "${COMPLETE_TO_RUN[@]}"; do
            run_complete_step_method "$model" "$method" || true
        done
    done
fi

# Generate summary
echo ""
echo "=============================================="
echo "BENCHMARK SUITE COMPLETE"
echo "=============================================="
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Files generated:"
ls -la "$OUTPUT_DIR"
echo ""
echo "To view results:"
echo "  cat ${OUTPUT_DIR}/*.txt"
echo ""

# Create a combined summary file
SUMMARY_FILE="${OUTPUT_DIR}/summary.txt"
echo "DiZO Kernel Benchmark Summary" > "$SUMMARY_FILE"
echo "Generated: $(date)" >> "$SUMMARY_FILE"
echo "Models: ${MODELS[*]}" >> "$SUMMARY_FILE"
echo "Methods: ${METHODS[*]}" >> "$SUMMARY_FILE"
echo "======================================" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

for f in "${OUTPUT_DIR}"/*.txt; do
    if [ "$f" != "$SUMMARY_FILE" ]; then
        echo "--- $(basename "$f") ---" >> "$SUMMARY_FILE"
        # Extract just the summary tables and timing info
        grep -E "(Time:|Memory:|Speedup|SUMMARY)" "$f" >> "$SUMMARY_FILE" 2>/dev/null || true
        echo "" >> "$SUMMARY_FILE"
    fi
done

echo "Summary saved to: $SUMMARY_FILE"
echo ""
echo "Quick commands:"
echo "  # View all CUDA results:"
echo "  grep -h 'Time:' ${OUTPUT_DIR}/*cuda*.txt"
echo "  # Compare all methods for a model:"
echo "  cat ${OUTPUT_DIR}/opt-13b_*.txt | grep Time:"
