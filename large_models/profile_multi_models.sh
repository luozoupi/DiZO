#!/bin/bash
# ============================================================================
# Multi-Model Profiling Script for DiZO
# ============================================================================
# 
# This script profiles DiZO training across multiple OPT model sizes:
#   - OPT-350M (baseline, quick)
#   - OPT-2.7B
#   - OPT-6.7B
#   - OPT-13B
#
# Each model's profile is saved to a separate subdirectory.
#
# Usage:
#   ./profile_multi_models.sh [GPU_ID]
#
# Example:
#   ./profile_multi_models.sh 4      # Use GPU 4
#   ./profile_multi_models.sh        # Default: GPU 6
#
# Output Structure:
#   profiler_logs/
#   ├── profile_opt-350m/
#   │   ├── opt-350m_profiler_summary.txt
#   │   └── *.pt.trace.json
#   ├── profile_opt-2_7b/
#   │   ├── opt-2_7b_profiler_summary.txt
#   │   └── *.pt.trace.json
#   ├── profile_opt-6_7b/
#   │   └── ...
#   ├── profile_opt-13b/
#   │   └── ...
#   └── comparison_summary.txt
# ============================================================================

set -e  # Exit on error

# Configuration
GPU_ID=${1:-6}
export CUDA_VISIBLE_DEVICES=$GPU_ID

OUTPUT_DIR="./profiler_logs"
TASK="SST2"
NUM_STEPS=5

# Model configurations: name, batch_size
# Larger models need smaller batch sizes to fit in memory
declare -A MODEL_CONFIGS=(
    ["facebook/opt-350m"]="4"
    ["facebook/opt-2.7b"]="2"
    ["facebook/opt-6.7b"]="1"
    ["facebook/opt-13b"]="1"
)

# Models to profile (in order)
MODELS=(
    "facebook/opt-350m"
    "facebook/opt-2.7b"
    "facebook/opt-6.7b"
    "facebook/opt-13b"
)

# Create output directory
mkdir -p $OUTPUT_DIR

echo "============================================================================"
echo "DiZO Multi-Model Profiling"
echo "============================================================================"
echo "GPU: $GPU_ID"
echo "Task: $TASK"
echo "Steps per model: $NUM_STEPS"
echo "Output: $OUTPUT_DIR"
echo "Models to profile: ${MODELS[*]}"
echo "============================================================================"
echo ""

# Track timing and results
declare -A PROFILE_TIMES
declare -A PROFILE_STATUS
START_TIME=$(date +%s)

# Function to get model short name
get_short_name() {
    echo "$1" | sed 's/facebook\///' | tr '.' '_'
}

# Profile each model
for MODEL in "${MODELS[@]}"; do
    BATCH_SIZE=${MODEL_CONFIGS[$MODEL]}
    SHORT_NAME=$(get_short_name "$MODEL")
    
    echo ""
    echo "============================================================================"
    echo "Profiling: $MODEL (batch_size=$BATCH_SIZE)"
    echo "============================================================================"
    
    MODEL_START=$(date +%s)
    
    # Run profiling
    if python profile_dizo.py \
        --model_name "$MODEL" \
        --task_name "$TASK" \
        --num_steps "$NUM_STEPS" \
        --batch_size "$BATCH_SIZE" \
        --output_dir "$OUTPUT_DIR" \
        --profile_memory; then
        
        MODEL_END=$(date +%s)
        PROFILE_TIMES[$MODEL]=$((MODEL_END - MODEL_START))
        PROFILE_STATUS[$MODEL]="SUCCESS"
        echo ""
        echo "✓ $MODEL profiling complete (${PROFILE_TIMES[$MODEL]}s)"
    else
        MODEL_END=$(date +%s)
        PROFILE_TIMES[$MODEL]=$((MODEL_END - MODEL_START))
        PROFILE_STATUS[$MODEL]="FAILED"
        echo ""
        echo "✗ $MODEL profiling failed"
    fi
    
    # Brief pause between models to let GPU cool down
    sleep 5
done

END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))

# Generate comparison summary
SUMMARY_FILE="$OUTPUT_DIR/comparison_summary.txt"
echo ""
echo "============================================================================"
echo "Generating Comparison Summary"
echo "============================================================================"

cat > "$SUMMARY_FILE" << EOF
============================================================================
DiZO MULTI-MODEL PROFILING SUMMARY
============================================================================
Date: $(date)
GPU: $GPU_ID
Task: $TASK
Steps: $NUM_STEPS
Total Time: ${TOTAL_TIME}s

============================================================================
PROFILING RESULTS
============================================================================
EOF

for MODEL in "${MODELS[@]}"; do
    SHORT_NAME=$(get_short_name "$MODEL")
    STATUS=${PROFILE_STATUS[$MODEL]}
    TIME=${PROFILE_TIMES[$MODEL]}
    
    echo "Model: $MODEL" >> "$SUMMARY_FILE"
    echo "  Status: $STATUS" >> "$SUMMARY_FILE"
    echo "  Time: ${TIME}s" >> "$SUMMARY_FILE"
    echo "  Output: $OUTPUT_DIR/profile_$SHORT_NAME/" >> "$SUMMARY_FILE"
    echo "" >> "$SUMMARY_FILE"
done

# Add visualization commands
cat >> "$SUMMARY_FILE" << EOF
============================================================================
VISUALIZATION COMMANDS
============================================================================

# TensorBoard (view all models):
tensorboard --logdir=$OUTPUT_DIR

# Individual model traces (Chrome/Perfetto):
EOF

for MODEL in "${MODELS[@]}"; do
    SHORT_NAME=$(get_short_name "$MODEL")
    echo "# $MODEL:" >> "$SUMMARY_FILE"
    echo "#   Open chrome://tracing and load: $OUTPUT_DIR/profile_$SHORT_NAME/*.pt.trace.json" >> "$SUMMARY_FILE"
done

cat >> "$SUMMARY_FILE" << EOF

============================================================================
OUTPUT FILES
============================================================================
EOF

# List output files
for MODEL in "${MODELS[@]}"; do
    SHORT_NAME=$(get_short_name "$MODEL")
    SUBDIR="$OUTPUT_DIR/profile_$SHORT_NAME"
    if [ -d "$SUBDIR" ]; then
        echo "" >> "$SUMMARY_FILE"
        echo "[$MODEL]" >> "$SUMMARY_FILE"
        ls -lh "$SUBDIR" 2>/dev/null | head -20 >> "$SUMMARY_FILE"
    fi
done

# Print summary
echo ""
echo "============================================================================"
echo "PROFILING COMPLETE"
echo "============================================================================"
echo ""
echo "Results:"
for MODEL in "${MODELS[@]}"; do
    STATUS=${PROFILE_STATUS[$MODEL]}
    TIME=${PROFILE_TIMES[$MODEL]}
    if [ "$STATUS" = "SUCCESS" ]; then
        echo "  ✓ $MODEL (${TIME}s)"
    else
        echo "  ✗ $MODEL (${TIME}s) - FAILED"
    fi
done
echo ""
echo "Total time: ${TOTAL_TIME}s"
echo ""
echo "Output files:"
echo "  Summary: $SUMMARY_FILE"
echo "  Profiles: $OUTPUT_DIR/profile_*/"
echo ""
echo "To view with TensorBoard:"
echo "  tensorboard --logdir=$OUTPUT_DIR"
echo ""
echo "============================================================================"
