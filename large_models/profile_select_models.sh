#!/bin/bash
# ============================================================================
# Selective Model Profiling Script for DiZO
# ============================================================================
# 
# Profile specific OPT models based on command-line arguments.
#
# Usage:
#   ./profile_select_models.sh [models...] [-g GPU_ID] [-s NUM_STEPS]
#
# Examples:
#   ./profile_select_models.sh 350m 2.7b          # Profile OPT-350M and OPT-2.7B
#   ./profile_select_models.sh all                # Profile all models
#   ./profile_select_models.sh 2.7b 6.7b -g 4     # Use GPU 4
#   ./profile_select_models.sh 13b -s 3           # Only 3 steps for OPT-13B
#
# Available models: 350m, 1.3b, 2.7b, 6.7b, 13b, 30b, 66b
# ============================================================================

set -e

# Default configuration
GPU_ID=6
NUM_STEPS=5
TASK="SST2"
OUTPUT_DIR="./profiler_logs"

# Model name mapping and batch sizes
declare -A MODEL_NAMES=(
    ["350m"]="facebook/opt-350m"
    ["1.3b"]="facebook/opt-1.3b"
    ["2.7b"]="facebook/opt-2.7b"
    ["6.7b"]="facebook/opt-6.7b"
    ["13b"]="facebook/opt-13b"
    ["30b"]="facebook/opt-30b"
    ["66b"]="facebook/opt-66b"
)

declare -A BATCH_SIZES=(
    ["350m"]="4"
    ["1.3b"]="4"
    ["2.7b"]="2"
    ["6.7b"]="1"
    ["13b"]="1"
    ["30b"]="1"
    ["66b"]="1"
)

# Parse arguments
SELECTED_MODELS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        -g|--gpu)
            GPU_ID="$2"
            shift 2
            ;;
        -s|--steps)
            NUM_STEPS="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [models...] [-g GPU_ID] [-s NUM_STEPS] [-o OUTPUT_DIR]"
            echo ""
            echo "Models: 350m, 1.3b, 2.7b, 6.7b, 13b, 30b, 66b, all"
            echo ""
            echo "Options:"
            echo "  -g, --gpu      GPU ID (default: 6)"
            echo "  -s, --steps    Number of profiling steps (default: 5)"
            echo "  -o, --output   Output directory (default: ./profiler_logs)"
            exit 0
            ;;
        all)
            SELECTED_MODELS=("350m" "2.7b" "6.7b" "13b")
            shift
            ;;
        *)
            if [[ -n "${MODEL_NAMES[$1]}" ]]; then
                SELECTED_MODELS+=("$1")
            else
                echo "Unknown model: $1"
                echo "Available: 350m, 1.3b, 2.7b, 6.7b, 13b, 30b, 66b"
                exit 1
            fi
            shift
            ;;
    esac
done

# Default to 350m if no models specified
if [ ${#SELECTED_MODELS[@]} -eq 0 ]; then
    SELECTED_MODELS=("350m")
fi

export CUDA_VISIBLE_DEVICES=$GPU_ID

echo "============================================================================"
echo "DiZO Selective Model Profiling"
echo "============================================================================"
echo "GPU: $GPU_ID"
echo "Steps: $NUM_STEPS"
echo "Models: ${SELECTED_MODELS[*]}"
echo "Output: $OUTPUT_DIR"
echo "============================================================================"

mkdir -p "$OUTPUT_DIR"

# Profile each selected model
for MODEL_KEY in "${SELECTED_MODELS[@]}"; do
    MODEL_NAME="${MODEL_NAMES[$MODEL_KEY]}"
    BATCH_SIZE="${BATCH_SIZES[$MODEL_KEY]}"
    
    echo ""
    echo ">>> Profiling $MODEL_NAME (batch_size=$BATCH_SIZE)"
    echo ""
    
    python profile_dizo.py \
        --model_name "$MODEL_NAME" \
        --task_name "$TASK" \
        --num_steps "$NUM_STEPS" \
        --batch_size "$BATCH_SIZE" \
        --output_dir "$OUTPUT_DIR" \
        --profile_memory
    
    echo ""
    echo "✓ $MODEL_NAME complete"
    sleep 2
done

echo ""
echo "============================================================================"
echo "All profiles complete!"
echo "View with: tensorboard --logdir=$OUTPUT_DIR"
echo "============================================================================"
