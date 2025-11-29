#!/bin/bash
# PonderTTT Full Experiment Pipeline
# Run all experiments from scratch with Hard Skip implementation
#
# Usage:
#   ./scripts/run_all_experiments.sh           # Run all phases
#   ./scripts/run_all_experiments.sh phase1    # Run specific phase
#   ./scripts/run_all_experiments.sh phase1 phase2  # Run multiple phases

# NOTE: No 'set -e' - we want to continue even if individual experiments fail

# Configuration
NUM_WORKERS=128
BATCH_SIZE=16
NUM_ITERATIONS=10000
MAX_CHUNKS=10000
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
# SKIP doesn't need training - no learnable parameters
# ============================================================
phase1_baselines() {
    log_phase "Phase 1: Training Baselines"

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
# ============================================================
phase2_hard_skip() {
    log_phase "Phase 2: Training Hard Skip (Binary Gating)"

    # 125M Scale
    run_experiment "125M Hard Skip (skip=0.8)" \
        python -m ponderttt.experiments.train_hard_skip \
            --model_scale 125m --target_skip_rate 0.8 --cost_weight 0.1 \
            --num_iterations $NUM_ITERATIONS --batch_size $BATCH_SIZE \
            --output_dir outputs/hard_skip/125m_skip0.8 \
            --num_workers $NUM_WORKERS --wandb_project ponderttt-125m

    run_experiment "125M Hard Skip (skip=0.5)" \
        python -m ponderttt.experiments.train_hard_skip \
            --model_scale 125m --target_skip_rate 0.5 --cost_weight 0.1 \
            --num_iterations $NUM_ITERATIONS --batch_size $BATCH_SIZE \
            --output_dir outputs/hard_skip/125m_skip0.5 \
            --num_workers $NUM_WORKERS --wandb_project ponderttt-125m

    # 350M Scale
    run_experiment "350M Hard Skip (skip=0.8)" \
        python -m ponderttt.experiments.train_hard_skip \
            --model_scale 350m --target_skip_rate 0.8 --cost_weight 0.1 \
            --num_iterations $NUM_ITERATIONS --batch_size $BATCH_SIZE \
            --output_dir outputs/hard_skip/350m_skip0.8 \
            --num_workers $NUM_WORKERS --wandb_project ponderttt-350m

    run_experiment "350M Hard Skip (skip=0.5)" \
        python -m ponderttt.experiments.train_hard_skip \
            --model_scale 350m --target_skip_rate 0.5 --cost_weight 0.1 \
            --num_iterations $NUM_ITERATIONS --batch_size $BATCH_SIZE \
            --output_dir outputs/hard_skip/350m_skip0.5 \
            --num_workers $NUM_WORKERS --wandb_project ponderttt-350m

    log_info "Phase 2 Complete!"
}

# ============================================================
# Phase 3: Evaluation - In-Distribution (Python)
# ============================================================
phase3_eval_id() {
    log_phase "Phase 3: Evaluating In-Distribution (Python)"

    # 125M
    local ckpt_125m=$(get_latest_checkpoint "outputs/hard_skip/125m_skip0.8")
    if [ -z "$ckpt_125m" ]; then
        log_error "No checkpoint found for 125M Hard Skip"
    else
        log_info "Using checkpoint: $ckpt_125m"
        run_experiment "Eval 125M Python" \
            python -m ponderttt.experiments.compare_methods \
                --model_scale 125m \
                --binary_gating_checkpoint "$ckpt_125m" \
                --num_eval_batches $NUM_EVAL_BATCHES \
                --language Python \
                --output_dir outputs/eval/125m_hard_skip_python
    fi

    # 350M
    local ckpt_350m=$(get_latest_checkpoint "outputs/hard_skip/350m_skip0.8")
    if [ -z "$ckpt_350m" ]; then
        log_error "No checkpoint found for 350M Hard Skip"
    else
        log_info "Using checkpoint: $ckpt_350m"
        run_experiment "Eval 350M Python" \
            python -m ponderttt.experiments.compare_methods \
                --model_scale 350m \
                --binary_gating_checkpoint "$ckpt_350m" \
                --num_eval_batches $NUM_EVAL_BATCHES \
                --language Python \
                --output_dir outputs/eval/350m_hard_skip_python
    fi

    log_info "Phase 3 Complete!"
}

# ============================================================
# Phase 4: Evaluation - Out-of-Distribution (JS, Java, Go)
# ============================================================
phase4_eval_ood() {
    log_phase "Phase 4: Evaluating Out-of-Distribution"

    local languages=("JavaScript" "Java" "Go")

    # Get checkpoints
    local ckpt_125m=$(get_latest_checkpoint "outputs/hard_skip/125m_skip0.8")
    local ckpt_350m=$(get_latest_checkpoint "outputs/hard_skip/350m_skip0.8")

    # 125M OOD
    if [ -z "$ckpt_125m" ]; then
        log_error "No checkpoint found for 125M Hard Skip"
    else
        log_info "Using 125M checkpoint: $ckpt_125m"
        for lang in "${languages[@]}"; do
            local lang_lower=$(echo "$lang" | tr '[:upper:]' '[:lower:]')
            run_experiment "Eval 125M $lang" \
                python -m ponderttt.experiments.compare_methods \
                    --model_scale 125m \
                    --binary_gating_checkpoint "$ckpt_125m" \
                    --num_eval_batches $NUM_EVAL_BATCHES_OOD \
                    --language "$lang" \
                    --output_dir "outputs/eval/125m_hard_skip_${lang_lower}"
        done
    fi

    # 350M OOD
    if [ -z "$ckpt_350m" ]; then
        log_error "No checkpoint found for 350M Hard Skip"
    else
        log_info "Using 350M checkpoint: $ckpt_350m"
        for lang in "${languages[@]}"; do
            local lang_lower=$(echo "$lang" | tr '[:upper:]' '[:lower:]')
            run_experiment "Eval 350M $lang" \
                python -m ponderttt.experiments.compare_methods \
                    --model_scale 350m \
                    --binary_gating_checkpoint "$ckpt_350m" \
                    --num_eval_batches $NUM_EVAL_BATCHES_OOD \
                    --language "$lang" \
                    --output_dir "outputs/eval/350m_hard_skip_${lang_lower}"
        done
    fi

    log_info "Phase 4 Complete!"
}

# ============================================================
# Phase 5: Ablation Studies
# ============================================================
phase5_ablation() {
    log_phase "Phase 5: Running Ablation Studies"

    # Temperature annealing ablation
    run_experiment "Ablation: Temp 2.0->0.1" \
        python -m ponderttt.experiments.train_hard_skip \
            --model_scale 125m \
            --initial_temperature 2.0 --min_temperature 0.1 \
            --num_iterations $NUM_ITERATIONS \
            --output_dir outputs/ablation/temp_2.0_to_0.1 \
            --wandb_project ponderttt-ablation

    run_experiment "Ablation: Temp 1.0->0.5" \
        python -m ponderttt.experiments.train_hard_skip \
            --model_scale 125m \
            --initial_temperature 1.0 --min_temperature 0.5 \
            --num_iterations $NUM_ITERATIONS \
            --output_dir outputs/ablation/temp_1.0_to_0.5 \
            --wandb_project ponderttt-ablation

    # Cost weight ablation
    run_experiment "Ablation: Cost 0.01" \
        python -m ponderttt.experiments.train_hard_skip \
            --model_scale 125m \
            --cost_weight 0.01 \
            --num_iterations $NUM_ITERATIONS \
            --output_dir outputs/ablation/cost_0.01 \
            --wandb_project ponderttt-ablation

    run_experiment "Ablation: Cost 0.5" \
        python -m ponderttt.experiments.train_hard_skip \
            --model_scale 125m \
            --cost_weight 0.5 \
            --num_iterations $NUM_ITERATIONS \
            --output_dir outputs/ablation/cost_0.5 \
            --wandb_project ponderttt-ablation

    log_info "Phase 5 Complete!"
}

# ============================================================
# Phase 6: RL Comparison (Optional)
# ============================================================
phase6_rl() {
    log_phase "Phase 6: Training and Evaluating RL (PPO)"

    run_experiment "RL Training" \
        python -m ponderttt.experiments.train_policy \
            --model_scale 125m \
            --num_iterations 1000 \
            --budget_limit 2.0 \
            --output_dir outputs/rl/125m_budget2.0 \
            --wandb_project ponderttt-rl

    local rl_ckpt=$(get_latest_checkpoint "outputs/rl/125m_budget2.0")
    if [ -z "$rl_ckpt" ]; then
        log_error "No checkpoint found for RL policy"
    else
        log_info "Using RL checkpoint: $rl_ckpt"
        run_experiment "RL Evaluation" \
            python -m ponderttt.experiments.compare_methods \
                --model_scale 125m \
                --rl_checkpoint "$rl_ckpt" \
                --num_eval_batches $NUM_EVAL_BATCHES_OOD \
                --output_dir outputs/eval/125m_rl
    fi

    log_info "Phase 6 Complete!"
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
    phase1_baselines
    phase2_hard_skip
    phase3_eval_id
    phase4_eval_ood
    phase5_ablation
    phase6_rl
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
            phase5|ablation)
                phase5_ablation
                ;;
            phase6|rl)
                phase6_rl
                ;;
            all)
                run_all
                exit $?
                ;;
            *)
                log_error "Unknown phase: $phase"
                echo "Available phases: phase1, phase2, phase3, phase4, phase5, phase6, all"
                exit 1
                ;;
        esac
    done
    print_summary
fi
