#!/bin/bash
# Run all baseline experiments for comparison
# Usage: bash scripts/run_baselines.sh [model_scale] [max_chunks]

set -e  # Exit on error

MODEL_SCALE=${1:-125m}
MAX_CHUNKS=${2:-100}
OUTPUT_DIR="outputs/baselines_${MODEL_SCALE}"
SEED=42

echo "============================================================"
echo "Running All Baseline Experiments"
echo "============================================================"
echo "Model scale: ${MODEL_SCALE}"
echo "Max chunks: ${MAX_CHUNKS}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Seed: ${SEED}"
echo "============================================================"
echo ""

mkdir -p ${OUTPUT_DIR}

# Array of actions
ACTIONS=("SKIP" "UPDATE_1" "UPDATE_2" "UPDATE_4")

for ACTION in "${ACTIONS[@]}"; do
    echo ""
    echo "------------------------------------------------------------"
    echo "Running: ${ACTION}"
    echo "------------------------------------------------------------"

    uv run python -m ponderttt.experiments.train_baseline \
        --model_scale ${MODEL_SCALE} \
        --action ${ACTION} \
        --max_chunks ${MAX_CHUNKS} \
        --output_dir ${OUTPUT_DIR} \
        --seed ${SEED}

    echo "✓ ${ACTION} completed"
done

echo ""
echo "============================================================"
echo "All baselines completed!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  1. Compare results in ${OUTPUT_DIR}/"
echo "  2. Run policy training: python -m ponderttt.experiments.train_policy"
echo "  3. Visualize results: python scripts/visualize_results.py"
