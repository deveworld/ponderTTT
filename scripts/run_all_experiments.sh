#!/bin/bash
# PonderTTT Full Experiment Pipeline
#
# Usage:
#   ./scripts/run_all_experiments.sh                    # Run all phases, all models
#   ./scripts/run_all_experiments.sh --125m             # Run only 125M model
#   ./scripts/run_all_experiments.sh --350m             # Run only 350M model
#   ./scripts/run_all_experiments.sh --125m phase1      # Run specific phase for 125M
#   ./scripts/run_all_experiments.sh phase1 phase2      # Run multiple phases
#
# Model Selection:
#   --125m    Run only 125M experiments
#   --350m    Run only 350M experiments
#   (default) Run both 125M and 350M
#
# Paper Tables Covered:
#   - Phase 1: Baseline Training (UPDATE_1, UPDATE_2, UPDATE_4)
#   - Phase 2: Evaluation - In-Distribution (Python) with TTT Improvement Gating
#   - Phase 3: Evaluation - Out-of-Distribution (JS, Java, Go)
#   - Phase 4: Latency Benchmark
#   - Phase 5: Shuffled Input Ablation
#   - Phase 6: Diagonal Mask Ablation

# NOTE: No 'set -e' - we want to continue even if individual experiments fail

# Model selection flags (default: run both)
RUN_125M=false
RUN_350M=false
RUN_1B=false
RUN_XL=false

# Configuration - Common
NUM_WORKERS=128

# Configuration - 125M Model
BATCH_SIZE_125M=16
MAX_CHUNKS_125M=100000
NUM_EVAL_BATCHES_125M=1000
NUM_EVAL_BATCHES_OOD_125M=500

# Configuration - 350M Model
BATCH_SIZE_350M=16
MAX_CHUNKS_350M=100000
NUM_EVAL_BATCHES_350M=1000
NUM_EVAL_BATCHES_OOD_350M=500

# Configuration - Larger Models
BATCH_SIZE_LARGE=2
MAX_CHUNKS_LARGE=10000
NUM_EVAL_BATCHES_LARGE=200

# Save frequency (auto-calculated: save once at midpoint)
SAVE_EVERY_BASELINE_125M=$((MAX_CHUNKS_125M / 2))
SAVE_EVERY_BASELINE_350M=$((MAX_CHUNKS_350M / 2))

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
    if [ "$RUN_125M" = true ]; then
        run_experiment "125M UPDATE_1" \
            python -m ponderttt.experiments.train_baseline \
                --model_scale 125m --action UPDATE_1 --max_chunks $MAX_CHUNKS_125M \
                --output_dir outputs/baselines/125m_update1 \
                --num_workers $NUM_WORKERS --batch_size $BATCH_SIZE_125M \
                --wandb_project ponderttt-125m --save_every $SAVE_EVERY_BASELINE_125M

        run_experiment "125M UPDATE_2" \
            python -m ponderttt.experiments.train_baseline \
                --model_scale 125m --action UPDATE_2 --max_chunks $MAX_CHUNKS_125M \
                --output_dir outputs/baselines/125m_update2 \
                --num_workers $NUM_WORKERS --batch_size $BATCH_SIZE_125M \
                --wandb_project ponderttt-125m --save_every $SAVE_EVERY_BASELINE_125M

        run_experiment "125M UPDATE_4" \
            python -m ponderttt.experiments.train_baseline \
                --model_scale 125m --action UPDATE_4 --max_chunks $MAX_CHUNKS_125M \
                --output_dir outputs/baselines/125m_update4 \
                --num_workers $NUM_WORKERS --batch_size $BATCH_SIZE_125M \
                --wandb_project ponderttt-125m --save_every $SAVE_EVERY_BASELINE_125M
    fi

    # 350M Baselines
    if [ "$RUN_350M" = true ]; then
        run_experiment "350M UPDATE_1" \
            python -m ponderttt.experiments.train_baseline \
                --model_scale 350m --action UPDATE_1 --max_chunks $MAX_CHUNKS_350M \
                --output_dir outputs/baselines/350m_update1 \
                --num_workers $NUM_WORKERS --batch_size $BATCH_SIZE_350M \
                --wandb_project ponderttt-350m --save_every $SAVE_EVERY_BASELINE_350M

        run_experiment "350M UPDATE_2" \
            python -m ponderttt.experiments.train_baseline \
                --model_scale 350m --action UPDATE_2 --max_chunks $MAX_CHUNKS_350M \
                --output_dir outputs/baselines/350m_update2 \
                --num_workers $NUM_WORKERS --batch_size $BATCH_SIZE_350M \
                --wandb_project ponderttt-350m --save_every $SAVE_EVERY_BASELINE_350M

        run_experiment "350M UPDATE_4" \
            python -m ponderttt.experiments.train_baseline \
                --model_scale 350m --action UPDATE_4 --max_chunks $MAX_CHUNKS_350M \
                --output_dir outputs/baselines/350m_update4 \
                --num_workers $NUM_WORKERS --batch_size $BATCH_SIZE_350M \
                --wandb_project ponderttt-350m --save_every $SAVE_EVERY_BASELINE_350M
    fi

    # 1B Baselines
    if [ "$RUN_1B" = true ]; then
        run_experiment "1B UPDATE_1" \
            python -m ponderttt.experiments.train_baseline \
                --model_scale 1b --action UPDATE_1 --max_chunks $MAX_CHUNKS_LARGE \
                --output_dir outputs/baselines/1b_update1 \
                --num_workers $NUM_WORKERS --batch_size $BATCH_SIZE_LARGE \
                --wandb_project ponderttt-1b --save_every $((MAX_CHUNKS_LARGE / 2))
    fi

    # XL Baselines
    if [ "$RUN_XL" = true ]; then
        run_experiment "XL UPDATE_1" \
            python -m ponderttt.experiments.train_baseline \
                --model_scale xl --action UPDATE_1 --max_chunks $MAX_CHUNKS_LARGE \
                --output_dir outputs/baselines/xl_update1 \
                --num_workers $NUM_WORKERS --batch_size $BATCH_SIZE_LARGE \
                --wandb_project ponderttt-xl --save_every $((MAX_CHUNKS_LARGE / 2))
    fi

    log_info "Phase 1 Complete!"
}

# ============================================================
# Phase 2: Evaluation - In-Distribution (Python)
# Compares: SKIP, UPDATE_1, Random Skip, Oracle, TTT Improvement, Loss Skip Gating
# ============================================================
phase2_eval_id() {
    log_phase "Phase 2: Evaluating In-Distribution Python"

    # Number of examples to skip for held-out evaluation
    # Training uses ~160K examples, so skip those for fair evaluation
    local SKIP_EXAMPLES=160000

    # 125M Evaluation
    if [ "$RUN_125M" = true ]; then
        local ckpt_125m_update1=$(get_latest_checkpoint "outputs/baselines/125m_update1/checkpoints")
        if [ -z "$ckpt_125m_update1" ]; then
            log_error "No UPDATE_1 checkpoint found for 125M. Run Phase 1 first!"
        else
            log_info "Using UPDATE_1 checkpoint: $ckpt_125m_update1"
            run_experiment "Eval 125M Python" \
                python -m ponderttt.experiments.compare_methods \
                    --model_scale 125m \
                    --update1_checkpoint "$ckpt_125m_update1" \
                    --num_eval_batches $NUM_EVAL_BATCHES_125M \
                    --language Python \
                    --skip_examples $SKIP_EXAMPLES \
                    --output_dir outputs/eval/125m_python \
                    --eval_ttt_loss \
                    --eval_ttt_improvement \
                    $INVERT_SIGNAL \
                    $TTT_BASE_LR_ARG
        fi
    fi

    # 350M Evaluation
    if [ "$RUN_350M" = true ]; then
        local ckpt_350m_update1=$(get_latest_checkpoint "outputs/baselines/350m_update1/checkpoints")
        if [ -z "$ckpt_350m_update1" ]; then
            log_error "No UPDATE_1 checkpoint found for 350M. Run Phase 1 first!"
        else
            log_info "Using UPDATE_1 checkpoint: $ckpt_350m_update1"
            run_experiment "Eval 350M Python" \
                python -m ponderttt.experiments.compare_methods \
                    --model_scale 350m \
                    --update1_checkpoint "$ckpt_350m_update1" \
                    --num_eval_batches $NUM_EVAL_BATCHES_350M \
                    --language Python \
                    --skip_examples $SKIP_EXAMPLES \
                    --output_dir outputs/eval/350m_python \
                    --eval_ttt_loss \
                    --eval_ttt_improvement \
                    $INVERT_SIGNAL \
                    $TTT_BASE_LR_ARG
        fi
    fi

    # 1B Evaluation (Fresh)
    if [ "$RUN_1B" = true ]; then
        log_info "Evaluating 1B (Fresh Weights)..."
        run_experiment "Eval 1B Python" \
            python -m ponderttt.experiments.compare_methods \
                --model_scale 1b \
                --num_eval_batches $NUM_EVAL_BATCHES_LARGE \
                --batch_size $BATCH_SIZE_LARGE \
                --language Python \
                --skip_examples $SKIP_EXAMPLES \
                --output_dir outputs/eval/1b_python \
                --eval_ttt_loss \
                --eval_ttt_improvement \
                $INVERT_SIGNAL \
                $TTT_BASE_LR_ARG
    fi

    # XL Evaluation (Fresh)
    if [ "$RUN_XL" = true ]; then
        log_info "Evaluating XL (Fresh Weights)..."
        run_experiment "Eval XL Python" \
            python -m ponderttt.experiments.compare_methods \
                --model_scale xl \
                --num_eval_batches $NUM_EVAL_BATCHES_LARGE \
                --batch_size $BATCH_SIZE_LARGE \
                --language Python \
                --skip_examples $SKIP_EXAMPLES \
                --output_dir outputs/eval/xl_python \
                --eval_ttt_loss \
                --eval_ttt_improvement \
                $INVERT_SIGNAL \
                $TTT_BASE_LR_ARG
    fi

    log_info "Phase 2 Complete!"
}

# ============================================================
# Phase 3: Evaluation - Out-of-Distribution (JS, Java, Go)
# ============================================================
phase3_eval_ood() {
    log_phase "Phase 3: Evaluating Out-of-Distribution"

    local languages=("JavaScript" "Java" "Go")

    # 125M OOD
    if [ "$RUN_125M" = true ]; then
        local ckpt_125m_update1=$(get_latest_checkpoint "outputs/baselines/125m_update1/checkpoints")
        if [ -z "$ckpt_125m_update1" ]; then
            log_error "No UPDATE_1 checkpoint found for 125M. Run Phase 1 first!"
        else
            log_info "Using 125M UPDATE_1 checkpoint: $ckpt_125m_update1"
            for lang in "${languages[@]}"; do
                local lang_lower=$(echo "$lang" | tr '[:upper:]' '[:lower:]')
                run_experiment "Eval 125M $lang" \
                    python -m ponderttt.experiments.compare_methods \
                        --model_scale 125m \
                        --update1_checkpoint "$ckpt_125m_update1" \
                        --num_eval_batches $NUM_EVAL_BATCHES_OOD_125M \
                        --language "$lang" \
                        --output_dir "outputs/eval/125m_${lang_lower}" \
                        --eval_ttt_loss \
                        --eval_ttt_improvement \
                        $INVERT_SIGNAL
            done
        fi
    fi

    # 350M OOD
    if [ "$RUN_350M" = true ]; then
        local ckpt_350m_update1=$(get_latest_checkpoint "outputs/baselines/350m_update1/checkpoints")
        if [ -z "$ckpt_350m_update1" ]; then
            log_error "No UPDATE_1 checkpoint found for 350M. Run Phase 1 first!"
        else
            log_info "Using 350M UPDATE_1 checkpoint: $ckpt_350m_update1"
            for lang in "${languages[@]}"; do
                local lang_lower=$(echo "$lang" | tr '[:upper:]' '[:lower:]')
                run_experiment "Eval 350M $lang" \
                    python -m ponderttt.experiments.compare_methods \
                        --model_scale 350m \
                        --update1_checkpoint "$ckpt_350m_update1" \
                        --num_eval_batches $NUM_EVAL_BATCHES_OOD_350M \
                        --language "$lang" \
                        --output_dir "outputs/eval/350m_${lang_lower}" \
                        --eval_ttt_loss \
                        --eval_ttt_improvement \
                        $INVERT_SIGNAL
            done
        fi
    fi

    log_info "Phase 3 Complete!"
}

# ============================================================
# Phase 4: Latency Benchmark
# ============================================================
phase4_latency() {
    log_phase "Phase 4: Latency Benchmark"

    # Latency script handles both 125M and 350M if we pass arguments, 
    # but the script provided seems to hardcode setups or expect modifications.
    # Let's check measure_latency.py usage. It seems to just run.
    # We will assume it's standalone or update it if needed. 
    # For now, running it directly as it appears to measure TTT overhead generally.
    
    run_experiment "Latency Benchmark" \
        python scripts/measure_latency.py

    log_info "Phase 4 Complete!"
}

# ============================================================
# Phase 5: Shuffled Input Ablation
# ============================================================
phase5_shuffle() {
    log_phase "Phase 5: Shuffled Input Ablation"

    local SKIP_EXAMPLES=160000

    if [ "$RUN_125M" = true ]; then
        local ckpt_125m_update1=$(get_latest_checkpoint "outputs/baselines/125m_update1/checkpoints")
         if [ -z "$ckpt_125m_update1" ]; then
            log_error "No UPDATE_1 checkpoint found for 125M. Cannot run Shuffle Ablation."
        else
            log_info "Using UPDATE_1 checkpoint: $ckpt_125m_update1"
            run_experiment "Shuffle Ablation 125M" \
                python -m ponderttt.experiments.compare_methods \
                    --model_scale 125m \
                    --update1_checkpoint "$ckpt_125m_update1" \
                    --num_eval_batches $NUM_EVAL_BATCHES_125M \
                    --language Python \
                    --skip_examples $SKIP_EXAMPLES \
                    --output_dir outputs/eval/125m_shuffle \
                    --eval_ttt_loss \
                    --eval_ttt_improvement \
                    $INVERT_SIGNAL \
                    --shuffle
        fi
    fi

    if [ "$RUN_350M" = true ]; then
        local ckpt_350m_update1=$(get_latest_checkpoint "outputs/baselines/350m_update1/checkpoints")
        if [ -z "$ckpt_350m_update1" ]; then
            log_error "No UPDATE_1 checkpoint found for 350M. Cannot run Shuffle Ablation."
        else
            log_info "Using UPDATE_1 checkpoint: $ckpt_350m_update1"
            run_experiment "Shuffle Ablation 350M" \
                python -m ponderttt.experiments.compare_methods \
                    --model_scale 350m \
                    --update1_checkpoint "$ckpt_350m_update1" \
                    --num_eval_batches $NUM_EVAL_BATCHES_350M \
                    --language Python \
                    --skip_examples $SKIP_EXAMPLES \
                    --output_dir outputs/eval/350m_shuffle \
                    --eval_ttt_loss \
                    --eval_ttt_improvement \
                    $INVERT_SIGNAL \
                    --shuffle
        fi
    fi

    log_info "Phase 5 Complete!"
}

# ============================================================
# Phase 6: Diagonal Mask Ablation
# ============================================================
phase6_diagonal() {
    log_phase "Phase 6: Diagonal Mask Ablation"
    
    # We compare diagonal_offset=0 (standard, baseline) vs diagonal_offset=-1 (no diagonal info)
    # The standard run covers 0. We run -1 here.

    local SKIP_EXAMPLES=160000

    if [ "$RUN_125M" = true ]; then
        local ckpt_125m_update1=$(get_latest_checkpoint "outputs/baselines/125m_update1/checkpoints")
         if [ -z "$ckpt_125m_update1" ]; then
            log_error "No UPDATE_1 checkpoint found for 125M. Cannot run Diagonal Ablation."
        else
            log_info "Using UPDATE_1 checkpoint: $ckpt_125m_update1"
            run_experiment "Diagonal Ablation 125M (k=-1)" \
                python -m ponderttt.experiments.compare_methods \
                    --model_scale 125m \
                    --update1_checkpoint "$ckpt_125m_update1" \
                    --num_eval_batches $NUM_EVAL_BATCHES_125M \
                    --language Python \
                    --skip_examples $SKIP_EXAMPLES \
                    --output_dir outputs/eval/125m_diagonal_k_minus_1 \
                    --eval_ttt_loss \
                    --eval_ttt_improvement \
                    $INVERT_SIGNAL \
                    --diagonal_offset -1
        fi
    fi

    if [ "$RUN_350M" = true ]; then
        local ckpt_350m_update1=$(get_latest_checkpoint "outputs/baselines/350m_update1/checkpoints")
        if [ -z "$ckpt_350m_update1" ]; then
            log_error "No UPDATE_1 checkpoint found for 350M. Cannot run Diagonal Ablation."
        else
            log_info "Using UPDATE_1 checkpoint: $ckpt_350m_update1"
            run_experiment "Diagonal Ablation 350M (k=-1)" \
                python -m ponderttt.experiments.compare_methods \
                    --model_scale 350m \
                    --update1_checkpoint "$ckpt_350m_update1" \
                    --num_eval_batches $NUM_EVAL_BATCHES_350M \
                    --language Python \
                    --skip_examples $SKIP_EXAMPLES \
                    --output_dir outputs/eval/350m_diagonal_k_minus_1 \
                    --eval_ttt_loss \
                    --eval_ttt_improvement \
                    $INVERT_SIGNAL \
                    --diagonal_offset -1
        fi
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
    echo ""
    echo "Phase 1: Baseline Training      -> UPDATE_1, UPDATE_2, UPDATE_4"
    echo "Phase 2: Eval Python (ID)       -> SKIP, UPDATE_1, Random, Oracle, TTT Improvement"
    echo "Phase 3: Eval OOD               -> JavaScript, Java, Go"
    echo ""

    phase1_baselines
    phase2_eval_id
    phase3_eval_ood
    phase4_latency
    phase5_shuffle
    phase6_diagonal
    print_summary
}


# Parse arguments - first pass: extract model flags
PHASES=()
INVERT_SIGNAL=""
TTT_BASE_LR_ARG=""

for arg in "$@"; do
    case $arg in
        --125m)
            RUN_125M=true
            ;;
        --350m)
            RUN_350M=true
            ;;
        --1b)
            RUN_1B=true
            ;;
        --xl)
            RUN_XL=true
            ;;
        --invert_signal)
            INVERT_SIGNAL="--invert_signal"
            ;;
        --ttt_base_lr=*)
            TTT_BASE_LR_ARG="${arg}"
            ;;
        *)
            PHASES+=("$arg")
            ;;
    esac
done

# If neither flag specified, run both
# If neither flag specified, run both small models (default)
if [ "$RUN_125M" = false ] && [ "$RUN_350M" = false ] && [ "$RUN_1B" = false ] && [ "$RUN_XL" = false ]; then
    RUN_125M=true
    RUN_350M=true
fi

# Log which models will be run
if [ "$RUN_125M" = true ] && [ "$RUN_350M" = true ]; then
    log_info "Running experiments for: 125M and 350M"
elif [ "$RUN_125M" = true ]; then
    log_info "Running experiments for: 125M only"
else
    log_info "Running experiments for selected models:"
    [ "$RUN_125M" = true ] && echo "  - 125M"
    [ "$RUN_350M" = true ] && echo "  - 350M"
    [ "$RUN_1B" = true ] && echo "  - 1B"
    [ "$RUN_XL" = true ] && echo "  - XL"
fi

# Parse phase arguments
if [ ${#PHASES[@]} -eq 0 ]; then
    run_all
else
    for phase in "${PHASES[@]}"; do
        case $phase in
            phase1|baselines)
                phase1_baselines
                ;;
            phase2|eval_id)
                phase2_eval_id
                ;;
            phase3|eval_ood)
                phase3_eval_ood
                ;;
            phase4|latency)
                phase4_latency
                ;;
            phase5|shuffle)
                phase5_shuffle
                ;;
            phase6|diagonal)
                phase6_diagonal
                ;;
            all)
                run_all
                exit $?
                ;;
            *)
                log_error "Unknown phase: $phase"
                echo "Available phases:"
                echo "  phase1, baselines       - Train UPDATE_1/2/4 baselines"
                echo "  phase2, eval_id         - Evaluate on Python (In-Distribution)"
                echo "  phase3, eval_ood        - Evaluate OOD (JavaScript, Java, Go)"
                echo "  phase4, latency         - Benchmark TTT latency"
                echo "  phase5, shuffle         - Shuffle ablation"
                echo "  phase6, diagonal        - Diagonal mask ablation"
                echo "  all                     - Run all phases"
                echo ""
                echo "Model selection:"
                echo "  --125m                  - Run only 125M experiments"
                echo "  --350m                  - Run only 350M experiments"
                echo "  --1b                    - Run only 1B (GPT2-Large) experiments"
                echo "  --xl                    - Run only XL (GPT2-XL) experiments"
                exit 1
                ;;
        esac
    done
    print_summary
fi
