#!/bin/bash
# Run hyperparameter sweep for PonderTTT
# Usage: ./scripts/run_sweep.sh [gpu|tpu]

set -e  # Exit on error

DEVICE=${1:-gpu}
MODEL_SCALE="125m"
STEPS=1000
OUTPUT_ROOT="outputs/sweep"

echo "Starting sweep on $DEVICE with model $MODEL_SCALE..."

# 1. Budget 1.5
echo "=== Running Budget 1.5 ==="
python -m ponderttt.experiments.train_differentiable \
    --model_scale $MODEL_SCALE \
    --num_iterations $STEPS \
    --budget_limit 1.5 \
    --output_dir $OUTPUT_ROOT/budget_1.5 \
    --max_steps 4.0

# 2. Budget 2.0
echo "=== Running Budget 2.0 ==="
python -m ponderttt.experiments.train_differentiable \
    --model_scale $MODEL_SCALE \
    --num_iterations $STEPS \
    --budget_limit 2.0 \
    --output_dir $OUTPUT_ROOT/budget_2.0 \
    --max_steps 4.0

# 3. Budget 3.0
echo "=== Running Budget 3.0 ==="
python -m ponderttt.experiments.train_differentiable \
    --model_scale $MODEL_SCALE \
    --num_iterations $STEPS \
    --budget_limit 3.0 \
    --output_dir $OUTPUT_ROOT/budget_3.0 \
    --max_steps 4.0

echo "Sweep training complete."

# 4. Evaluate All
echo "=== Evaluating All ==="
for BUDGET in 1.5 2.0 3.0; do
    echo "Evaluating Budget $BUDGET..."
    python -m ponderttt.experiments.compare_methods \
        --model_scale $MODEL_SCALE \
        --budget $BUDGET \
        --num_eval_batches 50 \
        --diff_checkpoint $OUTPUT_ROOT/budget_$BUDGET/checkpoint_$STEPS \
        --output_dir $OUTPUT_ROOT/eval_budget_$BUDGET
done

echo "All done! Results in $OUTPUT_ROOT"
