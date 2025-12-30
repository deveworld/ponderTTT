#!/bin/bash
# PonderTTT Full Experiment Pipeline
#
# Usage:
#   ./scripts/run_all_experiments.sh                    # Run all phases, all models
#   ./scripts/run_all_experiments.sh --small            # Run only GPT-2 Small (124M)
#   ./scripts/run_all_experiments.sh --medium           # Run only GPT-2 Medium (355M)
#   ./scripts/run_all_experiments.sh --small phase1     # Run specific phase for small
#   ./scripts/run_all_experiments.sh phase1 phase2      # Run multiple phases
#
# Model Selection:
#   --small   Run only GPT-2 Small experiments
#   --medium  Run only GPT-2 Medium experiments
#   (default) Run both small and medium
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
RUN_SMALL=false
RUN_MEDIUM=false
RUN_LARGE=false
RUN_XL=false

# Configuration - Common
NUM_WORKERS=128

# Configuration - GPT-2 Small (124M)
BATCH_SIZE_SMALL=16
MAX_CHUNKS_SMALL=160000
NUM_EVAL_BATCHES_SMALL=1000
NUM_EVAL_BATCHES_OOD_SMALL=500

# Configuration - GPT-2 Medium (355M)
BATCH_SIZE_MEDIUM=16
MAX_CHUNKS_MEDIUM=160000
NUM_EVAL_BATCHES_MEDIUM=1000
NUM_EVAL_BATCHES_OOD_MEDIUM=500

# Configuration - Larger Models
BATCH_SIZE_LARGE=16
MAX_CHUNKS_LARGE=160000
NUM_EVAL_BATCHES_LARGE=1000

# Save frequency (auto-calculated: save once at midpoint)
# Save frequency (auto-calculated: save once at midpoint)
SAVE_EVERY_BASELINE_SMALL=$((MAX_CHUNKS_SMALL / 2))
SAVE_EVERY_BASELINE_MEDIUM=$((MAX_CHUNKS_MEDIUM / 2))

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

    # GPT-2 Small (124M) Baselines
    if [ "$RUN_SMALL" = true ]; then
        run_experiment "GPT-2 Small UPDATE_1" \
            python -m ponderttt.experiments.train_baseline \
                --model_scale small --action UPDATE_1 --max_chunks $MAX_CHUNKS_SMALL \
                --output_dir outputs/baselines/small_update1 \
                --num_workers $NUM_WORKERS --batch_size $BATCH_SIZE_SMALL \
                --wandb_project ponderttt-small --save_every $SAVE_EVERY_BASELINE_SMALL

        run_experiment "GPT-2 Small UPDATE_2" \
            python -m ponderttt.experiments.train_baseline \
                --model_scale small --action UPDATE_2 --max_chunks $MAX_CHUNKS_SMALL \
                --output_dir outputs/baselines/small_update2 \
                --num_workers $NUM_WORKERS --batch_size $BATCH_SIZE_SMALL \
                --wandb_project ponderttt-small --save_every $SAVE_EVERY_BASELINE_SMALL

        run_experiment "GPT-2 Small UPDATE_4" \
            python -m ponderttt.experiments.train_baseline \
                --model_scale small --action UPDATE_4 --max_chunks $MAX_CHUNKS_SMALL \
                --output_dir outputs/baselines/small_update4 \
                --num_workers $NUM_WORKERS --batch_size $BATCH_SIZE_SMALL \
                --wandb_project ponderttt-small --save_every $SAVE_EVERY_BASELINE_SMALL
    fi

    # GPT-2 Medium (355M) Baselines
    if [ "$RUN_MEDIUM" = true ]; then
        run_experiment "GPT-2 Medium UPDATE_1" \
            python -m ponderttt.experiments.train_baseline \
                --model_scale medium --action UPDATE_1 --max_chunks $MAX_CHUNKS_MEDIUM \
                --output_dir outputs/baselines/medium_update1 \
                --num_workers $NUM_WORKERS --batch_size $BATCH_SIZE_MEDIUM \
                --wandb_project ponderttt-medium --save_every $SAVE_EVERY_BASELINE_MEDIUM

        run_experiment "GPT-2 Medium UPDATE_2" \
            python -m ponderttt.experiments.train_baseline \
                --model_scale medium --action UPDATE_2 --max_chunks $MAX_CHUNKS_MEDIUM \
                --output_dir outputs/baselines/medium_update2 \
                --num_workers $NUM_WORKERS --batch_size $BATCH_SIZE_MEDIUM \
                --wandb_project ponderttt-medium --save_every $SAVE_EVERY_BASELINE_MEDIUM

        run_experiment "GPT-2 Medium UPDATE_4" \
            python -m ponderttt.experiments.train_baseline \
                --model_scale medium --action UPDATE_4 --max_chunks $MAX_CHUNKS_MEDIUM \
                --output_dir outputs/baselines/medium_update4 \
                --num_workers $NUM_WORKERS --batch_size $BATCH_SIZE_MEDIUM \
                --wandb_project ponderttt-medium --save_every $SAVE_EVERY_BASELINE_MEDIUM
    fi

    # GPT-2 Large Baselines
    if [ "$RUN_LARGE" = true ]; then
        run_experiment "GPT-2 Large UPDATE_1" \
            python -m ponderttt.experiments.train_baseline \
                --model_scale large --action UPDATE_1 --max_chunks $MAX_CHUNKS_LARGE \
                --output_dir outputs/baselines/large_update1 \
                --num_workers $NUM_WORKERS --batch_size $BATCH_SIZE_LARGE \
                --wandb_project ponderttt-large --save_every $((MAX_CHUNKS_LARGE / 2))
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

    # GPT-2 Small (124M) Evaluation (Standard Gating)
    if [ "$RUN_SMALL" = true ]; then
        local ckpt_small_update1=$(get_latest_checkpoint "outputs/baselines/small_update1/checkpoints")
        if [ -z "$ckpt_small_update1" ]; then
            log_error "No UPDATE_1 checkpoint found for GPT-2 Small. Run Phase 1 first!"
        else
            log_info "Using UPDATE_1 checkpoint: $ckpt_small_update1"
            run_experiment "Eval GPT-2 Small Python" \
                python -m ponderttt.experiments.compare_methods \
                    --model_scale small \
                    --checkpoint_path "$ckpt_small_update1" \
                    --num_eval_batches $NUM_EVAL_BATCHES_SMALL \
                    --batch_size $BATCH_SIZE_SMALL \
                    --language Python \
                    --skip_examples $SKIP_EXAMPLES \
                    --output_dir outputs/eval/small_python \
                    
        fi
    fi

    # GPT-2 Medium (355M) Evaluation
    if [ "$RUN_MEDIUM" = true ]; then
        local ckpt_medium_update1=$(get_latest_checkpoint "outputs/baselines/medium_update1/checkpoints")
        if [ -z "$ckpt_medium_update1" ]; then
            log_error "No UPDATE_1 checkpoint found for GPT-2 Medium. Run Phase 1 first!"
        else
            log_info "Using UPDATE_1 checkpoint: $ckpt_medium_update1"
            
            # 1. Standard Gating (Comparison/Control - expected to fail/underperform)
            log_info "Running GPT-2 Medium Standard Gating (Control)"
            run_experiment "Eval GPT-2 Medium Python (Standard)" \
                python -m ponderttt.experiments.compare_methods \
                    --model_scale medium \
                    --checkpoint_path "$ckpt_medium_update1" \
                    --num_eval_batches $NUM_EVAL_BATCHES_MEDIUM \
                    --batch_size $BATCH_SIZE_MEDIUM \
                    --language Python \
                    --skip_examples $SKIP_EXAMPLES \
                    --output_dir outputs/eval/medium_python \
                    
        fi
    fi

    # GPT-2 Large Evaluation
    if [ "$RUN_LARGE" = true ]; then
        local ckpt_large_update1=$(get_latest_checkpoint "outputs/baselines/large_update1/checkpoints")
        if [ -z "$ckpt_large_update1" ]; then
            log_error "No UPDATE_1 checkpoint found for GPT-2 Large. Run Phase 1 first!"
        else
            log_info "Using UPDATE_1 checkpoint: $ckpt_large_update1"
            
            # 1. Standard Gating
            log_info "Running GPT-2 Large Standard Gating (Control)"
            run_experiment "Eval GPT-2 Large Python (Standard)" \
                python -m ponderttt.experiments.compare_methods \
                    --model_scale large \
                    --checkpoint_path "$ckpt_large_update1" \
                    --num_eval_batches $NUM_EVAL_BATCHES_LARGE \
                    --batch_size $BATCH_SIZE_LARGE \
                    --language Python \
                    --skip_examples $SKIP_EXAMPLES \
                    --output_dir outputs/eval/large_python \
                    
            
        fi
    fi

    # XL Evaluation
    if [ "$RUN_XL" = true ]; then
        local ckpt_xl_update1=$(get_latest_checkpoint "outputs/baselines/xl_update1/checkpoints")
        if [ -z "$ckpt_xl_update1" ]; then
            log_error "No UPDATE_1 checkpoint found for XL. Run Phase 1 first!"
        else
            log_info "Using UPDATE_1 checkpoint: $ckpt_xl_update1"
            
            # 1. Standard Gating
            log_info "Running XL Standard Gating (Control)"
            run_experiment "Eval XL Python (Standard)" \
                python -m ponderttt.experiments.compare_methods \
                    --model_scale xl \
                    --checkpoint_path "$ckpt_xl_update1" \
                    --num_eval_batches $NUM_EVAL_BATCHES_LARGE \
                    --batch_size $BATCH_SIZE_LARGE \
                    --language Python \
                    --skip_examples $SKIP_EXAMPLES \
                    --output_dir outputs/eval/xl_python \
                    
            
        fi
    fi

    log_info "Phase 2 Complete!"
}

# ============================================================
# Phase 3: Evaluation - Out-of-Distribution (JS, Java, Go)
# ============================================================
phase3_eval_ood() {
    log_phase "Phase 3: Evaluating Out-of-Distribution"

    local languages=("JavaScript" "Java" "Go")

    # GPT-2 Small (124M) OOD
    if [ "$RUN_SMALL" = true ]; then
        local ckpt_small_update1=$(get_latest_checkpoint "outputs/baselines/small_update1/checkpoints")
        if [ -z "$ckpt_small_update1" ]; then
            log_error "No UPDATE_1 checkpoint found for GPT-2 Small. Run Phase 1 first!"
        else
            log_info "Using GPT-2 Small UPDATE_1 checkpoint: $ckpt_small_update1"
            for lang in "${languages[@]}"; do
                local lang_lower=$(echo "$lang" | tr '[:upper:]' '[:lower:]')
                run_experiment "Eval GPT-2 Small $lang" \
                    python -m ponderttt.experiments.compare_methods \
                        --model_scale small \
                        --checkpoint_path "$ckpt_small_update1" \
                        --num_eval_batches $NUM_EVAL_BATCHES_OOD_SMALL \
                        --batch_size $BATCH_SIZE_SMALL \
                        --language "$lang" \
                        --output_dir "outputs/eval/small_${lang_lower}" \
            done
        fi
    fi

    # GPT-2 Medium (355M) OOD
    if [ "$RUN_MEDIUM" = true ]; then
        local ckpt_medium_update1=$(get_latest_checkpoint "outputs/baselines/medium_update1/checkpoints")
        if [ -z "$ckpt_medium_update1" ]; then
            log_error "No UPDATE_1 checkpoint found for GPT-2 Medium. Run Phase 1 first!"
        else
            log_info "Using GPT-2 Medium UPDATE_1 checkpoint: $ckpt_medium_update1"
            
            # 1. Standard Gating (Control - Verify Failure)
            log_info "Running GPT-2 Medium OOD Standard Gating (Control)"
            for lang in "${languages[@]}"; do
                local lang_lower=$(echo "$lang" | tr '[:upper:]' '[:lower:]')
                run_experiment "Eval GPT-2 Medium $lang (Standard)" \
                    python -m ponderttt.experiments.compare_methods \
                        --model_scale medium \
                        --checkpoint_path "$ckpt_medium_update1" \
                        --num_eval_batches $NUM_EVAL_BATCHES_OOD_MEDIUM \
                        --batch_size $BATCH_SIZE_MEDIUM \
                        --language "$lang" \
                        --output_dir "outputs/eval/medium_${lang_lower}" \
            done

        fi
    fi

    # GPT-2 Large OOD
    if [ "$RUN_LARGE" = true ]; then
        local ckpt_large_update1=$(get_latest_checkpoint "outputs/baselines/large_update1/checkpoints")
        if [ -z "$ckpt_large_update1" ]; then
            log_error "No UPDATE_1 checkpoint found for GPT-2 Large. Run Phase 1 first!"
        else
            log_info "Using GPT-2 Large UPDATE_1 checkpoint: $ckpt_large_update1"
            for lang in "${languages[@]}"; do
                local lang_lower=$(echo "$lang" | tr '[:upper:]' '[:lower:]')
                run_experiment "Eval GPT-2 Large $lang" \
                    python -m ponderttt.experiments.compare_methods \
                        --model_scale large \
                        --checkpoint_path "$ckpt_large_update1" \
                        --num_eval_batches $NUM_EVAL_BATCHES_LARGE \
                        --batch_size $BATCH_SIZE_LARGE \
                        --language "$lang" \
                        --output_dir "outputs/eval/large_${lang_lower}" \
            done
        fi
    fi

    # XL OOD (Important: Full-Seq Gating shows r=0.53-0.79 on XL OOD)
    if [ "$RUN_XL" = true ]; then
        local ckpt_xl_update1=$(get_latest_checkpoint "outputs/baselines/xl_update1/checkpoints")
        if [ -z "$ckpt_xl_update1" ]; then
            log_error "No UPDATE_1 checkpoint found for XL. Run Phase 1 first!"
        else
            log_info "Using XL UPDATE_1 checkpoint: $ckpt_xl_update1"
            for lang in "${languages[@]}"; do
                local lang_lower=$(echo "$lang" | tr '[:upper:]' '[:lower:]')
                run_experiment "Eval XL $lang" \
                    python -m ponderttt.experiments.compare_methods \
                        --model_scale xl \
                        --checkpoint_path "$ckpt_xl_update1" \
                        --num_eval_batches $NUM_EVAL_BATCHES_LARGE \
                        --batch_size $BATCH_SIZE_LARGE \
                        --language "$lang" \
                        --output_dir "outputs/eval/xl_${lang_lower}" \
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

    # GPU Utilization Benchmark (more meaningful metric than wall-clock latency)
    if [ "$RUN_SMALL" = true ]; then
        run_experiment "GPU Util GPT-2 Small" \
            python scripts/measure_gpu_util.py --model_scale small --batch_size 1
    fi

    if [ "$RUN_MEDIUM" = true ]; then
        run_experiment "GPU Util GPT-2 Medium" \
            python scripts/measure_gpu_util.py --model_scale medium --batch_size 1
    fi

    if [ "$RUN_LARGE" = true ]; then
        run_experiment "GPU Util GPT-2 Large" \
            python scripts/measure_gpu_util.py --model_scale large --batch_size 1
    fi

    if [ "$RUN_XL" = true ]; then
        run_experiment "GPU Util XL" \
            python scripts/measure_gpu_util.py --model_scale xl --batch_size 1
    fi

    log_info "Phase 4 Complete!"
}

# ============================================================
# Phase 5: Shuffled Input Ablation
# ============================================================
phase5_shuffle() {
    log_phase "Phase 5: Shuffled Input Ablation"

    local SKIP_EXAMPLES=160000

    if [ "$RUN_SMALL" = true ]; then
        local ckpt_small_update1=$(get_latest_checkpoint "outputs/baselines/small_update1/checkpoints")
         if [ -z "$ckpt_small_update1" ]; then
            log_error "No UPDATE_1 checkpoint found for GPT-2 Small. Cannot run Shuffle Ablation."
        else
            log_info "Using UPDATE_1 checkpoint: $ckpt_small_update1"
            run_experiment "Shuffle Ablation GPT-2 Small" \
                python -m ponderttt.experiments.compare_methods \
                    --model_scale small \
                    --checkpoint_path "$ckpt_small_update1" \
                    --num_eval_batches $NUM_EVAL_BATCHES_SMALL \
                    --batch_size $BATCH_SIZE_SMALL \
                    --language Python \
                    --skip_examples $SKIP_EXAMPLES \
                    --output_dir outputs/eval/small_shuffle \
                    --shuffle
        fi
    fi

    if [ "$RUN_MEDIUM" = true ]; then
        local ckpt_medium_update1=$(get_latest_checkpoint "outputs/baselines/medium_update1/checkpoints")
        if [ -z "$ckpt_medium_update1" ]; then
            log_error "No UPDATE_1 checkpoint found for GPT-2 Medium. No Shuffle Ablation."
        else
            log_info "Using UPDATE_1 checkpoint: $ckpt_medium_update1"
            # Medium Shuffle Ablation
            run_experiment "Shuffle Ablation GPT-2 Medium" \
                python -m ponderttt.experiments.compare_methods \
                    --model_scale medium \
                    --checkpoint_path "$ckpt_medium_update1" \
                    --num_eval_batches $NUM_EVAL_BATCHES_MEDIUM \
                    --batch_size $BATCH_SIZE_MEDIUM \
                    --language Python \
                    --skip_examples $SKIP_EXAMPLES \
                    --output_dir outputs/eval/medium_shuffle \
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

    if [ "$RUN_SMALL" = true ]; then
        local ckpt_small_update1=$(get_latest_checkpoint "outputs/baselines/small_update1/checkpoints")
         if [ -z "$ckpt_small_update1" ]; then
            log_error "No UPDATE_1 checkpoint found for GPT-2 Small. Cannot run Diagonal Ablation."
        else
            log_info "Using UPDATE_1 checkpoint: $ckpt_small_update1"
            run_experiment "Diagonal Ablation GPT-2 Small (k=-1)" \
                python -m ponderttt.experiments.compare_methods \
                    --model_scale small \
                    --checkpoint_path "$ckpt_small_update1" \
                    --num_eval_batches $NUM_EVAL_BATCHES_SMALL \
                    --batch_size $BATCH_SIZE_SMALL \
                    --language Python \
                    --skip_examples $SKIP_EXAMPLES \
                    --output_dir outputs/eval/small_diagonal_k_minus_1 \
                    --diagonal_offset -1
        fi
    fi

    if [ "$RUN_MEDIUM" = true ]; then
        local ckpt_medium_update1=$(get_latest_checkpoint "outputs/baselines/medium_update1/checkpoints")
        if [ -z "$ckpt_medium_update1" ]; then
            log_error "No UPDATE_1 checkpoint found for GPT-2 Medium. No Diagonal Ablation."
        else
            log_info "Using UPDATE_1 checkpoint: $ckpt_medium_update1"

            # 1. Medium Standard Gating k=-1 (for Paper Table 12 verification)
            run_experiment "Diagonal Ablation GPT-2 Medium (k=-1, Standard)" \
                python -m ponderttt.experiments.compare_methods \
                    --model_scale medium \
                    --checkpoint_path "$ckpt_medium_update1" \
                    --num_eval_batches $NUM_EVAL_BATCHES_MEDIUM \
                    --batch_size $BATCH_SIZE_MEDIUM \
                    --language Python \
                    --skip_examples $SKIP_EXAMPLES \
                    --output_dir outputs/eval/medium_diagonal_k_minus_1 \
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
TTT_BASE_LR_ARG=""

for arg in "$@"; do
    case $arg in
        --small)
            RUN_SMALL=true
            ;;
        --medium)
            RUN_MEDIUM=true
            ;;
        --large)
            RUN_LARGE=true
            ;;
        --xl)
            RUN_XL=true
            ;;
        --ttt_base_lr=*)
            TTT_BASE_LR_ARG="${arg}"
            ;;
        --all-models)
            RUN_SMALL=true
            RUN_MEDIUM=true
            RUN_LARGE=true
            RUN_XL=true
            ;;
        *)
            PHASES+=("$arg")
            ;;
    esac
done

# If neither flag specified, run both
# If neither flag specified, run both small models (default)
if [ "$RUN_SMALL" = false ] && [ "$RUN_MEDIUM" = false ] && [ "$RUN_LARGE" = false ] && [ "$RUN_XL" = false ]; then
    RUN_SMALL=true
    RUN_MEDIUM=true
fi


# Log which models will be run
if [ "$RUN_SMALL" = true ] && [ "$RUN_MEDIUM" = true ]; then
    log_info "Running experiments for: GPT-2 Small and Medium"
elif [ "$RUN_SMALL" = true ]; then
    log_info "Running experiments for: GPT-2 Small only"
else
    log_info "Running experiments for selected models:"
    [ "$RUN_SMALL" = true ] && echo "  - GPT-2 Small"
    [ "$RUN_MEDIUM" = true ] && echo "  - GPT-2 Medium"
    [ "$RUN_LARGE" = true ] && echo "  - GPT-2 Large"
    [ "$RUN_XL" = true ] && echo "  - GPT-2 XL"
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
                echo "  --small                 - Run only GPT-2 Small (124M) experiments"
                echo "  --medium                - Run only GPT-2 Medium (355M) experiments"
                echo "  --large                 - Run only GPT-2 Large (774M) experiments"
                echo "  --xl                    - Run only GPT-2 XL (1.5B) experiments"
                exit 1
                ;;
        esac
    done
    print_summary
fi
