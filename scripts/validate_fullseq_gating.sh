#!/bin/bash
# ============================================================
# PonderTTT: Full-Sequence Gating Signal Validation Experiments
# ============================================================
#
# This script validates the new Full-Sequence Reconstruction Loss
# gating signal (ttt_loss_init) which showed r=0.77 correlation
# with Oracle advantage on XL models.
#
# Key Questions to Answer:
# 1. Does Full-Seq Gating outperform Random Skip baseline?
# 2. How does performance scale with model size?
# 3. Does the improvement hold on OOD languages?
#
# Expected Output:
# - outputs/fullseq_gating/125m_python.json
# - outputs/fullseq_gating/xl_python.json
# - outputs/fullseq_gating/xl_javascript_ood.json
# ============================================================

set -e

# Configuration
OUTPUT_DIR="outputs/fullseq_gating"
NUM_BATCHES=200
BUDGET=1.4  # 20% update rate (sparse gating)

mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo "PonderTTT Full-Sequence Gating Validation"
echo "============================================================"
echo "Output Directory: $OUTPUT_DIR"
echo "Batches per experiment: $NUM_BATCHES"
echo "Budget: $BUDGET (20% update rate)"
echo "============================================================"
echo ""

# Check for required checkpoints
check_checkpoint() {
    local scale=$1
    local ckpt_path="outputs/baselines/${scale}_update1/checkpoints/checkpoint_*"
    if ls $ckpt_path 1> /dev/null 2>&1; then
        echo "✓ Found checkpoint for $scale"
        return 0
    else
        echo "✗ Missing checkpoint for $scale at $ckpt_path"
        return 1
    fi
}

echo "Checking required checkpoints..."
check_checkpoint "125m" || exit 1
check_checkpoint "xl" || exit 1
echo ""

# Helper function to find latest checkpoint (highest step number)
find_checkpoint() {
    local scale=$1
    # Extract step number from path like checkpoint_160000 and sort numerically
    ls -d outputs/baselines/${scale}_update1/checkpoints/checkpoint_* 2>/dev/null | \
        sed 's/.*checkpoint_//' | sort -n | tail -1 | \
        xargs -I{} echo "outputs/baselines/${scale}_update1/checkpoints/checkpoint_{}"
}

# ============================================================
# Experiment 1: 125M Python (In-Distribution)
# ============================================================
echo "============================================================"
echo "[1/4] 125M Python (In-Distribution)"
echo "============================================================"

CKPT_125M=$(find_checkpoint "125m")
echo "Checkpoint: $CKPT_125M"

python -m ponderttt.experiments.compare_methods \
    --model_scale 125m \
    --update1_checkpoint "$CKPT_125M" \
    --eval_ttt_loss \
    --budget $BUDGET \
    --num_eval_batches $NUM_BATCHES \
    --language Python \
    --output_dir "$OUTPUT_DIR/125m_python"

echo "✓ 125M Python complete"
echo ""

# ============================================================
# Experiment 2: XL Python (In-Distribution) - Main Test
# ============================================================
echo "============================================================"
echo "[2/4] XL Python (In-Distribution) - MAIN TEST"
echo "          Expected: Full-Seq Gating should beat Random Skip"
echo "          Signal correlation: r=0.77"
echo "============================================================"

CKPT_XL=$(find_checkpoint "xl")
echo "Checkpoint: $CKPT_XL"

python -m ponderttt.experiments.compare_methods \
    --model_scale xl \
    --update1_checkpoint "$CKPT_XL" \
    --eval_ttt_loss \
    --budget $BUDGET \
    --num_eval_batches $NUM_BATCHES \
    --language Python \
    --output_dir "$OUTPUT_DIR/xl_python"

echo "✓ XL Python complete"
echo ""

# ============================================================
# Experiment 3: XL JavaScript (OOD)
# ============================================================
echo "============================================================"
echo "[3/4] XL JavaScript (Out-of-Distribution)"
echo "          Testing OOD generalization of Full-Seq signal"
echo "============================================================"

python -m ponderttt.experiments.compare_methods \
    --model_scale xl \
    --update1_checkpoint "$CKPT_XL" \
    --eval_ttt_loss \
    --budget $BUDGET \
    --num_eval_batches $NUM_BATCHES \
    --language JavaScript \
    --output_dir "$OUTPUT_DIR/xl_javascript_ood"

echo "✓ XL JavaScript OOD complete"
echo ""

# ============================================================
# Experiment 4: XL Go (OOD - Hardest)
# ============================================================
echo "============================================================"
echo "[4/4] XL Go (Out-of-Distribution - Hardest)"
echo "          Go has lowest baseline accuracy"
echo "============================================================"

python -m ponderttt.experiments.compare_methods \
    --model_scale xl \
    --update1_checkpoint "$CKPT_XL" \
    --eval_ttt_loss \
    --budget $BUDGET \
    --num_eval_batches $NUM_BATCHES \
    --language Go \
    --output_dir "$OUTPUT_DIR/xl_go_ood"

echo "✓ XL Go OOD complete"
echo ""

# ============================================================
# Summary
# ============================================================
echo "============================================================"
echo "ALL EXPERIMENTS COMPLETE"
echo "============================================================"
echo ""
echo "Results saved to: $OUTPUT_DIR/"
echo ""
echo "Key files to check:"
echo "  - $OUTPUT_DIR/xl_python/results.csv"
echo "  - $OUTPUT_DIR/xl_javascript_ood/results.csv"
echo "  - $OUTPUT_DIR/xl_go_ood/results.csv"
echo ""
echo "What to look for:"
echo "  1. Compare 'TTT Loss-Gating' vs 'Random Skip' loss"
echo "  2. Positive delta = Full-Seq Gating is BETTER"
echo "  3. Check 'Oracle Capture' percentage"
echo ""
echo "Expected results (based on r=0.77 correlation):"
echo "  - XL Python: Full-Seq should recover 70%+ of Oracle gap"
echo "  - 125M Python: Marginal improvement (r=0.42)"
echo ""
