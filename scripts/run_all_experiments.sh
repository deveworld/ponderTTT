#!/bin/bash
# PonderTTT Full Experiment Pipeline
# Runs ALL experiments from paper/main.tex
#
# Usage:
#   ./scripts/run_all_experiments.sh           # Run all phases
#   ./scripts/run_all_experiments.sh phase1    # Run specific phase
#   ./scripts/run_all_experiments.sh phase1 phase2  # Run multiple phases
#
# Paper Tables Covered:
#   - Table 1 (Main Results): Phase 3
#   - Table 2 (OOD): Phase 4
#   - Table 3 (Latency): Phase 5
#   - Appendix Training Dynamics: Phase 2 outputs
#   - Appendix Baseline Results: Phase 1 outputs
#   - Appendix OOD Full: Phase 4
#   - Table 8 (Shuffled Input): Phase 6
#   - Table 9 (Causal Mask Ablation): Phase 7

# NOTE: No 'set -e' - we want to continue even if individual experiments fail

# Configuration
NUM_WORKERS=128
BATCH_SIZE=16
NUM_ITERATIONS=10000
MAX_CHUNKS=100000
NUM_EVAL_BATCHES=1000
NUM_EVAL_BATCHES_OOD=500

# Track failures
declare -a FAILED_EXPERIMENTS=()
TOTAL_EXPERIMENTS=0
PASSED_EXPERIMENTS=0

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_phase() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

# Find the latest checkpoint in a directory (highest number)
get_latest_checkpoint() {
    local dir="$1"
    if [ ! -d "$dir" ]; then
        echo ""
        return 1
    fi

    # Find checkpoint_* directories, extract numbers, sort, get highest
    local latest=$(ls -d "${dir}"/checkpoint_* 2>/dev/null | \
        sed 's/.*checkpoint_//' | \
        sort -n | \
        tail -1)

    if [ -z "$latest" ]; then
        echo ""
        return 1
    fi

    echo "${dir}/checkpoint_${latest}"
}

# Run a single experiment with error handling
run_experiment() {
    local name="$1"
    shift

    TOTAL_EXPERIMENTS=$((TOTAL_EXPERIMENTS + 1))
    log_info "Starting: $name"

    if "$@"; then
        log_info "SUCCESS: $name"
        PASSED_EXPERIMENTS=$((PASSED_EXPERIMENTS + 1))
        return 0
    else
        log_error "FAILED: $name"
        FAILED_EXPERIMENTS+=("$name")
        return 1
    fi
}

# ============================================================
# Phase 1: Baseline Training (UPDATE_1, UPDATE_2, UPDATE_4)
# Paper: Appendix Table (Baseline Results on Training Data)
# SKIP doesn't need training - no learnable parameters
# ============================================================
phase1_baselines() {
    log_phase "Phase 1: Training Baselines (Appendix Baseline Table)"

    # 125M Baselines
    run_experiment "125M UPDATE_1" \
        python -m ponderttt.experiments.train_baseline \
            --model_scale 125m --action UPDATE_1 --max_chunks $MAX_CHUNKS \
            --output_dir outputs/baselines/125m_update1 \
            --num_workers $NUM_WORKERS --batch_size $BATCH_SIZE \
            --wandb_project ponderttt-125m

    run_experiment "125M UPDATE_2" \
        python -m ponderttt.experiments.train_baseline \
            --model_scale 125m --action UPDATE_2 --max_chunks $MAX_CHUNKS \
            --output_dir outputs/baselines/125m_update2 \
            --num_workers $NUM_WORKERS --batch_size $BATCH_SIZE \
            --wandb_project ponderttt-125m

    run_experiment "125M UPDATE_4" \
        python -m ponderttt.experiments.train_baseline \
            --model_scale 125m --action UPDATE_4 --max_chunks $MAX_CHUNKS \
            --output_dir outputs/baselines/125m_update4 \
            --num_workers $NUM_WORKERS --batch_size $BATCH_SIZE \
            --wandb_project ponderttt-125m

    # 350M Baselines
    run_experiment "350M UPDATE_1" \
        python -m ponderttt.experiments.train_baseline \
            --model_scale 350m --action UPDATE_1 --max_chunks $MAX_CHUNKS \
            --output_dir outputs/baselines/350m_update1 \
            --num_workers $NUM_WORKERS --batch_size $BATCH_SIZE \
            --wandb_project ponderttt-350m

    run_experiment "350M UPDATE_2" \
        python -m ponderttt.experiments.train_baseline \
            --model_scale 350m --action UPDATE_2 --max_chunks $MAX_CHUNKS \
            --output_dir outputs/baselines/350m_update2 \
            --num_workers $NUM_WORKERS --batch_size $BATCH_SIZE \
            --wandb_project ponderttt-350m

    run_experiment "350M UPDATE_4" \
        python -m ponderttt.experiments.train_baseline \
            --model_scale 350m --action UPDATE_4 --max_chunks $MAX_CHUNKS \
            --output_dir outputs/baselines/350m_update4 \
            --num_workers $NUM_WORKERS --batch_size $BATCH_SIZE \
            --wandb_project ponderttt-350m

    log_info "Phase 1 Complete!"
}

# ============================================================
# Phase 2: Hard Skip (Binary Gating) Training
# Paper: Table 1 (Main Results), Appendix Training Dynamics
# Note: target_update_rate=0.5 corresponds to "Target Skip 0.5" in paper
#       target_update_rate=0.2 corresponds to "Target Skip 0.8" in paper
# ============================================================
phase2_hard_skip() {
    log_phase "Phase 2: Training Hard Skip (Table 1, Appendix Training Dynamics)"

    # 125M Scale
    # Paper "Target Skip 0.5" = 50% skip target = 50% update target
    run_experiment "125M Hard Skip (target_update=0.5)" \
        python -m ponderttt.experiments.train_hard_skip \
            --model_scale 125m --target_update_rate 0.5 \
            --num_iterations $NUM_ITERATIONS --batch_size $BATCH_SIZE \
            --output_dir outputs/hard_skip/125m_update0.5 \
            --num_workers $NUM_WORKERS --wandb_project ponderttt-125m

    # Paper "Target Skip 0.8" = 80% skip target = 20% update target
    run_experiment "125M Hard Skip (target_update=0.2)" \
        python -m ponderttt.experiments.train_hard_skip \
            --model_scale 125m --target_update_rate 0.2 \
            --num_iterations $NUM_ITERATIONS --batch_size $BATCH_SIZE \
            --output_dir outputs/hard_skip/125m_update0.2 \
            --num_workers $NUM_WORKERS --wandb_project ponderttt-125m

    # 350M Scale
    run_experiment "350M Hard Skip (target_update=0.5)" \
        python -m ponderttt.experiments.train_hard_skip \
            --model_scale 350m --target_update_rate 0.5 \
            --num_iterations $NUM_ITERATIONS --batch_size $BATCH_SIZE \
            --output_dir outputs/hard_skip/350m_update0.5 \
            --num_workers $NUM_WORKERS --wandb_project ponderttt-350m

    run_experiment "350M Hard Skip (target_update=0.2)" \
        python -m ponderttt.experiments.train_hard_skip \
            --model_scale 350m --target_update_rate 0.2 \
            --num_iterations $NUM_ITERATIONS --batch_size $BATCH_SIZE \
            --output_dir outputs/hard_skip/350m_update0.2 \
            --num_workers $NUM_WORKERS --wandb_project ponderttt-350m

    log_info "Phase 2 Complete!"
}

# ============================================================
# Phase 3: Evaluation - In-Distribution (Python)
# Paper: Table 1 (Main Results)
# ============================================================
phase3_eval_id() {
    log_phase "Phase 3: Evaluating In-Distribution Python (Table 1)"

    # Number of examples to skip for held-out evaluation
    # Training uses ~160K examples, so skip those for fair evaluation
    local SKIP_EXAMPLES=160000

    # 125M - Use target_update=0.5 checkpoint (primary result in paper)
    local ckpt_125m=$(get_latest_checkpoint "outputs/hard_skip/125m_update0.5")
    local ckpt_125m_update1=$(get_latest_checkpoint "outputs/baselines/125m_update1/checkpoints")
    if [ -z "$ckpt_125m" ]; then
        log_error "No checkpoint found for 125M Hard Skip"
    else
        log_info "Using Hard Skip checkpoint: $ckpt_125m"
        log_info "Using UPDATE_1 checkpoint: $ckpt_125m_update1"
        run_experiment "Eval 125M Python (Table 1)" \
            python -m ponderttt.experiments.compare_methods \
                --model_scale 125m \
                --binary_gating_checkpoint "$ckpt_125m" \
                --update1_checkpoint "$ckpt_125m_update1" \
                --num_eval_batches $NUM_EVAL_BATCHES \
                --language Python \
                --skip_examples $SKIP_EXAMPLES \
                --output_dir outputs/eval/125m_python
    fi

    # 350M
    local ckpt_350m=$(get_latest_checkpoint "outputs/hard_skip/350m_update0.5")
    local ckpt_350m_update1=$(get_latest_checkpoint "outputs/baselines/350m_update1/checkpoints")
    if [ -z "$ckpt_350m" ]; then
        log_error "No checkpoint found for 350M Hard Skip"
    else
        log_info "Using Hard Skip checkpoint: $ckpt_350m"
        log_info "Using UPDATE_1 checkpoint: $ckpt_350m_update1"
        run_experiment "Eval 350M Python (Table 1)" \
            python -m ponderttt.experiments.compare_methods \
                --model_scale 350m \
                --binary_gating_checkpoint "$ckpt_350m" \
                --update1_checkpoint "$ckpt_350m_update1" \
                --num_eval_batches $NUM_EVAL_BATCHES \
                --language Python \
                --skip_examples $SKIP_EXAMPLES \
                --output_dir outputs/eval/350m_python
    fi

    log_info "Phase 3 Complete!"
}

# ============================================================
# Phase 4: Evaluation - Out-of-Distribution (JS, Java, Go)
# Paper: Table 2 (OOD), Appendix OOD Full
# ============================================================
phase4_eval_ood() {
    log_phase "Phase 4: Evaluating Out-of-Distribution (Table 2, Appendix OOD)"

    local languages=("JavaScript" "Java" "Go")

    # Get checkpoints
    local ckpt_125m=$(get_latest_checkpoint "outputs/hard_skip/125m_update0.5")
    local ckpt_350m=$(get_latest_checkpoint "outputs/hard_skip/350m_update0.5")

    # Get UPDATE_1 checkpoints for baselines
    local ckpt_125m_update1=$(get_latest_checkpoint "outputs/baselines/125m_update1/checkpoints")
    local ckpt_350m_update1=$(get_latest_checkpoint "outputs/baselines/350m_update1/checkpoints")

    # 125M OOD (Table 2)
    if [ -z "$ckpt_125m" ]; then
        log_error "No checkpoint found for 125M Hard Skip"
    else
        log_info "Using 125M Hard Skip checkpoint: $ckpt_125m"
        for lang in "${languages[@]}"; do
            local lang_lower=$(echo "$lang" | tr '[:upper:]' '[:lower:]')
            run_experiment "Eval 125M $lang (Table 2)" \
                python -m ponderttt.experiments.compare_methods \
                    --model_scale 125m \
                    --binary_gating_checkpoint "$ckpt_125m" \
                    --update1_checkpoint "$ckpt_125m_update1" \
                    --num_eval_batches $NUM_EVAL_BATCHES_OOD \
                    --language "$lang" \
                    --output_dir "outputs/eval/125m_${lang_lower}"
        done
    fi

    # 350M OOD (Appendix OOD Full)
    if [ -z "$ckpt_350m" ]; then
        log_error "No checkpoint found for 350M Hard Skip"
    else
        log_info "Using 350M Hard Skip checkpoint: $ckpt_350m"
        for lang in "${languages[@]}"; do
            local lang_lower=$(echo "$lang" | tr '[:upper:]' '[:lower:]')
            run_experiment "Eval 350M $lang (Appendix OOD)" \
                python -m ponderttt.experiments.compare_methods \
                    --model_scale 350m \
                    --binary_gating_checkpoint "$ckpt_350m" \
                    --update1_checkpoint "$ckpt_350m_update1" \
                    --num_eval_batches $NUM_EVAL_BATCHES_OOD \
                    --language "$lang" \
                    --output_dir "outputs/eval/350m_${lang_lower}"
        done
    fi

    log_info "Phase 4 Complete!"
}

# ============================================================
# Phase 5: Latency Measurement
# Paper: Table 3 (Latency Analysis)
# ============================================================
phase5_latency() {
    log_phase "Phase 5: Measuring Latency (Table 3)"

    local ckpt_125m=$(get_latest_checkpoint "outputs/hard_skip/125m_update0.5")

    if [ -z "$ckpt_125m" ]; then
        log_error "No checkpoint found for 125M Hard Skip"
    else
        run_experiment "Latency Measurement (Table 3)" \
            python scripts/measure_latency.py \
                --checkpoint "$ckpt_125m" \
                --model_scale 125m \
                --num_iterations 100 \
                --warmup_iterations 10
    fi

    log_info "Phase 5 Complete!"
}

# ============================================================
# Phase 6: Shuffled Input Sanity Check
# Paper: Table 8 (Shuffled Input Test)
# ============================================================
phase6_shuffled() {
    log_phase "Phase 6: Shuffled Input Test (Table 8)"

    local ckpt_125m=$(get_latest_checkpoint "outputs/hard_skip/125m_update0.5")

    if [ -z "$ckpt_125m" ]; then
        log_error "No checkpoint found for 125M Hard Skip"
    else
        run_experiment "Shuffled Input Test (Table 8)" \
            python scripts/test_shuffled_input.py \
                --checkpoint "$ckpt_125m" \
                --model_scale 125m \
                --num_batches 100 \
                --batch_size 4
    fi

    log_info "Phase 6 Complete!"
}

# ============================================================
# Phase 7: Causal Mask Diagonal Ablation
# Paper: Table 9 (Causal Mask Diagonal Ablation)
# ============================================================
phase7_causal_ablation() {
    log_phase "Phase 7: Causal Mask Ablation (Table 9)"

    local ckpt_125m=$(get_latest_checkpoint "outputs/hard_skip/125m_update0.5")

    if [ -z "$ckpt_125m" ]; then
        log_error "No checkpoint found for 125M Hard Skip"
    else
        run_experiment "Causal Mask Ablation k=0 vs k=-1 (Table 9)" \
            python scripts/ablation_strict_causal.py \
                --checkpoint "$ckpt_125m" \
                --model_scale 125m \
                --num_batches 100 \
                --batch_size 4
    fi

    log_info "Phase 7 Complete!"
}

# ============================================================
# Print Summary
# ============================================================
print_summary() {
    log_phase "EXPERIMENT SUMMARY"

    echo -e "Total Experiments: $TOTAL_EXPERIMENTS"
    echo -e "${GREEN}Passed: $PASSED_EXPERIMENTS${NC}"
    echo -e "${RED}Failed: ${#FAILED_EXPERIMENTS[@]}${NC}"

    if [ ${#FAILED_EXPERIMENTS[@]} -gt 0 ]; then
        echo ""
        echo -e "${RED}Failed Experiments:${NC}"
        for exp in "${FAILED_EXPERIMENTS[@]}"; do
            echo -e "  - $exp"
        done
        echo ""
        log_warn "Some experiments failed. Check logs above for details."
        return 1
    else
        echo ""
        log_info "All experiments completed successfully!"
        return 0
    fi
}

# ============================================================
# Main
# ============================================================
run_all() {
    log_info "Running ALL experiment phases..."
    log_info "This will reproduce all tables from paper/main.tex"
    echo ""
    echo "Phase 1: Baseline Training      -> Appendix Baseline Table"
    echo "Phase 2: Hard Skip Training     -> Table 1, Appendix Training Dynamics"
    echo "Phase 3: Eval Python (ID)       -> Table 1"
    echo "Phase 4: Eval OOD               -> Table 2, Appendix OOD Full"
    echo "Phase 5: Latency                -> Table 3"
    echo "Phase 6: Shuffled Input         -> Table 8"
    echo "Phase 7: Causal Mask Ablation   -> Table 9"
    echo ""

    phase1_baselines
    phase2_hard_skip
    phase3_eval_id
    phase4_eval_ood
    phase5_latency
    phase6_shuffled
    phase7_causal_ablation
    print_summary
}

# Parse arguments
if [ $# -eq 0 ]; then
    run_all
else
    for phase in "$@"; do
        case $phase in
            phase1|baselines)
                phase1_baselines
                ;;
            phase2|hard_skip)
                phase2_hard_skip
                ;;
            phase3|eval_id)
                phase3_eval_id
                ;;
            phase4|eval_ood)
                phase4_eval_ood
                ;;
            phase5|latency)
                phase5_latency
                ;;
            phase6|shuffled)
                phase6_shuffled
                ;;
            phase7|causal_ablation)
                phase7_causal_ablation
                ;;
            all)
                run_all
                exit $?
                ;;
            *)
                log_error "Unknown phase: $phase"
                echo "Available phases:"
                echo "  phase1, baselines       - Train UPDATE_1/2/4 baselines"
                echo "  phase2, hard_skip       - Train Hard Skip (PonderTTT)"
                echo "  phase3, eval_id         - Evaluate on Python (Table 1)"
                echo "  phase4, eval_ood        - Evaluate OOD (Table 2)"
                echo "  phase5, latency         - Measure latency (Table 3)"
                echo "  phase6, shuffled        - Shuffled input test (Table 8)"
                echo "  phase7, causal_ablation - Causal mask ablation (Table 9)"
                echo "  all                     - Run all phases"
                exit 1
                ;;
        esac
    done
    print_summary
fi
