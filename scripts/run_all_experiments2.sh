#!/bin/bash
# PonderTTT Gemma 3 Experiment Pipeline
#
# Usage:
#   ./scripts/run_all_experiments2.sh                    # Run all phases, all Gemma 3 models
#   ./scripts/run_all_experiments2.sh --1b               # Run only Gemma 3 1B
#   ./scripts/run_all_experiments2.sh --4b               # Run only Gemma 3 4B
#   ./scripts/run_all_experiments2.sh --12b              # Run only Gemma 3 12B
#   ./scripts/run_all_experiments2.sh --4b phase1        # Run specific phase for 4B
#   ./scripts/run_all_experiments2.sh phase1 phase2      # Run multiple phases
#
# Model Selection:
#   --1b    Run only Gemma 3 1B experiments (for testing)
#   --4b    Run only Gemma 3 4B experiments
#   --12b   Run only Gemma 3 12B experiments
#   --27b   Run only Gemma 3 27B experiments
#   (default) Run 4B and 12B
#
# Paper Tables Covered:
#   - Phase 1: Baseline Training (UPDATE_1, UPDATE_2, UPDATE_4)
#   - Phase 2: Evaluation - In-Distribution (Python) with TTT Improvement Gating
#   - Phase 3: Evaluation - Out-of-Distribution (JS, Java, Go)
#   - Phase 4: Latency Benchmark
#   - Phase 5: Shuffled Input Ablation
#   - Phase 6: Diagonal Mask Ablation
#
# Requirements:
#   - NVIDIA RTX PRO 6000 Blackwell (96GB VRAM) or similar
#   - Gemma 3 weights (HuggingFace or Orbax checkpoint)

# NOTE: No 'set -e' - we want to continue even if individual experiments fail

# Model selection flags (default: run 4B and 12B)
RUN_1B=false
RUN_4B=false
RUN_12B=false
RUN_27B=false

# Configuration - Common (single GPU, reduced workers)
NUM_WORKERS=16

# Configuration - Gemma 3 1B (for testing)
# ~2GB model, fits easily with large batch
BATCH_SIZE_1B=32
MAX_CHUNKS_1B=80000
NUM_EVAL_BATCHES_1B=500
NUM_EVAL_BATCHES_OOD_1B=250
CHECKPOINT_1B="hf:google/gemma-3-1b-it"

# Configuration - Gemma 3 4B
# ~8GB model, 96GB VRAM allows larger batch
BATCH_SIZE_4B=16
MAX_CHUNKS_4B=160000
NUM_EVAL_BATCHES_4B=1000
NUM_EVAL_BATCHES_OOD_4B=500
CHECKPOINT_4B="hf:google/gemma-3-4b-it"

# Configuration - Gemma 3 12B
# ~24GB model, moderate batch size
BATCH_SIZE_12B=2
MAX_CHUNKS_12B=160000
NUM_EVAL_BATCHES_12B=1000
NUM_EVAL_BATCHES_OOD_12B=500
CHECKPOINT_12B="hf:google/gemma-3-12b-it"

# Configuration - Gemma 3 27B
# ~54GB model, limited batch size on 96GB
BATCH_SIZE_27B=2
MAX_CHUNKS_27B=80000
NUM_EVAL_BATCHES_27B=500
NUM_EVAL_BATCHES_OOD_27B=250
CHECKPOINT_27B="hf:google/gemma-3-27b-it"

# Save frequency
SAVE_EVERY_1B=$((MAX_CHUNKS_1B / 2))
SAVE_EVERY_4B=$((MAX_CHUNKS_4B / 2))
SAVE_EVERY_12B=$((MAX_CHUNKS_12B / 2))
SAVE_EVERY_27B=$((MAX_CHUNKS_27B / 2))

# Track failures
declare -a FAILED_EXPERIMENTS=()
TOTAL_EXPERIMENTS=0
PASSED_EXPERIMENTS=0

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
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
    echo -e "\n${MAGENTA}========================================${NC}"
    echo -e "${MAGENTA}$1${NC}"
    echo -e "${MAGENTA}========================================${NC}\n"
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
# Uses train_gemma3_ttt.py for frozen backbone training
# ============================================================
phase1_baselines() {
    log_phase "Phase 1: Training Gemma 3 Baselines"

    # Gemma 3 1B Baselines (for testing)
    if [ "$RUN_1B" = true ]; then
        log_info "Training Gemma 3 1B baseline..."
        
        run_experiment "Gemma 3 1B UPDATE_1" \
            python -m ponderttt.experiments.train_baseline \
                --model_scale 1b --action UPDATE_1 --max_chunks $MAX_CHUNKS_1B \
                --output_dir outputs/gemma3/baselines/1b_update1 \
                --num_workers $NUM_WORKERS --batch_size $BATCH_SIZE_1B \
                --wandb_project ponderttt-gemma3-1b --save_every $SAVE_EVERY_1B \
                --checkpoint_path "$CHECKPOINT_1B"
    fi

    # Gemma 3 4B Baselines
    if [ "$RUN_4B" = true ]; then
        log_info "Training Gemma 3 4B baseline..."
        
        run_experiment "Gemma 3 4B UPDATE_1" \
            python -m ponderttt.experiments.train_baseline \
                --model_scale 4b --action UPDATE_1 --max_chunks $MAX_CHUNKS_4B \
                --output_dir outputs/gemma3/baselines/4b_update1 \
                --num_workers $NUM_WORKERS --batch_size $BATCH_SIZE_4B \
                --wandb_project ponderttt-gemma3-4b --save_every $SAVE_EVERY_4B \
                --checkpoint_path "$CHECKPOINT_4B"
    fi

    # Gemma 3 12B Baselines
    if [ "$RUN_12B" = true ]; then
        log_info "Training Gemma 3 12B baseline..."
        
        run_experiment "Gemma 3 12B UPDATE_1" \
            python -m ponderttt.experiments.train_baseline \
                --model_scale 12b --action UPDATE_1 --max_chunks $MAX_CHUNKS_12B \
                --output_dir outputs/gemma3/baselines/12b_update1 \
                --num_workers $NUM_WORKERS --batch_size $BATCH_SIZE_12B \
                --wandb_project ponderttt-gemma3-12b --save_every $SAVE_EVERY_12B \
                --checkpoint_path "$CHECKPOINT_12B"
    fi

    # Gemma 3 27B Baselines
    if [ "$RUN_27B" = true ]; then
        log_info "Training Gemma 3 27B baselines..."
        
        run_experiment "Gemma 3 27B UPDATE_1" \
            python -m ponderttt.experiments.train_baseline \
                --model_scale 27b --action UPDATE_1 --max_chunks $MAX_CHUNKS_27B \
                --output_dir outputs/gemma3/baselines/27b_update1 \
                --num_workers $NUM_WORKERS --batch_size $BATCH_SIZE_27B \
                --wandb_project ponderttt-gemma3-27b --save_every $SAVE_EVERY_27B \
                --checkpoint_path "$CHECKPOINT_27B"
    fi

    log_info "Phase 1 Complete!"
}

# ============================================================
# Phase 2: Evaluation - In-Distribution (Python)
# Compares: SKIP, UPDATE_1, Random Skip, Oracle, TTT Improvement, Loss Skip Gating
# ============================================================
phase2_eval_id() {
    log_phase "Phase 2: Evaluating In-Distribution Python"

    local SKIP_EXAMPLES=160000

    # Gemma 3 1B Evaluation
    if [ "$RUN_1B" = true ]; then
        local ckpt_1b_update1=$(get_latest_checkpoint "outputs/gemma3/baselines/1b_update1/checkpoints")
        if [ -z "$ckpt_1b_update1" ]; then
            log_error "No UPDATE_1 checkpoint found for Gemma 3 1B. Run Phase 1 first!"
        else
            log_info "Using UPDATE_1 checkpoint: $ckpt_1b_update1"
            run_experiment "Eval Gemma 3 1B Python" \
                python -m ponderttt.experiments.compare_methods \
                    --model_scale 1b \
                    --checkpoint_path "$ckpt_1b_update1" \
                    --num_eval_batches $NUM_EVAL_BATCHES_1B \
                    --batch_size $BATCH_SIZE_1B \
                    --language Python \
                    --skip_examples $SKIP_EXAMPLES \
                    --output_dir outputs/gemma3/eval/1b_python
        fi
    fi

    # Gemma 3 4B Evaluation
    if [ "$RUN_4B" = true ]; then
        local ckpt_4b_update1=$(get_latest_checkpoint "outputs/gemma3/baselines/4b_update1/checkpoints")
        if [ -z "$ckpt_4b_update1" ]; then
            log_error "No UPDATE_1 checkpoint found for Gemma 3 4B. Run Phase 1 first!"
        else
            log_info "Using UPDATE_1 checkpoint: $ckpt_4b_update1"
            run_experiment "Eval Gemma 3 4B Python" \
                python -m ponderttt.experiments.compare_methods \
                    --model_scale 4b \
                    --checkpoint_path "$ckpt_4b_update1" \
                    --num_eval_batches $NUM_EVAL_BATCHES_4B \
                    --batch_size $BATCH_SIZE_4B \
                    --language Python \
                    --skip_examples $SKIP_EXAMPLES \
                    --output_dir outputs/gemma3/eval/4b_python
        fi
    fi

    # Gemma 3 12B Evaluation
    if [ "$RUN_12B" = true ]; then
        local ckpt_12b_update1=$(get_latest_checkpoint "outputs/gemma3/baselines/12b_update1/checkpoints")
        if [ -z "$ckpt_12b_update1" ]; then
            log_error "No UPDATE_1 checkpoint found for Gemma 3 12B. Run Phase 1 first!"
        else
            log_info "Using UPDATE_1 checkpoint: $ckpt_12b_update1"
            run_experiment "Eval Gemma 3 12B Python" \
                python -m ponderttt.experiments.compare_methods \
                    --model_scale 12b \
                    --checkpoint_path "$ckpt_12b_update1" \
                    --num_eval_batches $NUM_EVAL_BATCHES_12B \
                    --batch_size $BATCH_SIZE_12B \
                    --language Python \
                    --skip_examples $SKIP_EXAMPLES \
                    --output_dir outputs/gemma3/eval/12b_python
        fi
    fi

    # Gemma 3 27B Evaluation
    if [ "$RUN_27B" = true ]; then
        local ckpt_27b_update1=$(get_latest_checkpoint "outputs/gemma3/baselines/27b_update1/checkpoints")
        if [ -z "$ckpt_27b_update1" ]; then
            log_error "No UPDATE_1 checkpoint found for Gemma 3 27B. Run Phase 1 first!"
        else
            log_info "Using UPDATE_1 checkpoint: $ckpt_27b_update1"
            run_experiment "Eval Gemma 3 27B Python" \
                python -m ponderttt.experiments.compare_methods \
                    --model_scale 27b \
                    --checkpoint_path "$ckpt_27b_update1" \
                    --num_eval_batches $NUM_EVAL_BATCHES_27B \
                    --batch_size $BATCH_SIZE_27B \
                    --language Python \
                    --skip_examples $SKIP_EXAMPLES \
                    --output_dir outputs/gemma3/eval/27b_python
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

    # Gemma 3 4B OOD
    if [ "$RUN_4B" = true ]; then
        local ckpt_4b_update1=$(get_latest_checkpoint "outputs/gemma3/baselines/4b_update1/checkpoints")
        if [ -z "$ckpt_4b_update1" ]; then
            log_error "No UPDATE_1 checkpoint found for Gemma 3 4B. Run Phase 1 first!"
        else
            log_info "Using Gemma 3 4B UPDATE_1 checkpoint: $ckpt_4b_update1"
            for lang in "${languages[@]}"; do
                local lang_lower=$(echo "$lang" | tr '[:upper:]' '[:lower:]')
                run_experiment "Eval Gemma 3 4B $lang" \
                    python -m ponderttt.experiments.compare_methods \
                        --model_scale 4b \
                        --checkpoint_path "$ckpt_4b_update1" \
                        --num_eval_batches $NUM_EVAL_BATCHES_OOD_4B \
                        --batch_size $BATCH_SIZE_4B \
                        --language "$lang" \
                        --output_dir "outputs/gemma3/eval/4b_${lang_lower}"
            done
        fi
    fi

    # Gemma 3 12B OOD
    if [ "$RUN_12B" = true ]; then
        local ckpt_12b_update1=$(get_latest_checkpoint "outputs/gemma3/baselines/12b_update1/checkpoints")
        if [ -z "$ckpt_12b_update1" ]; then
            log_error "No UPDATE_1 checkpoint found for Gemma 3 12B. Run Phase 1 first!"
        else
            log_info "Using Gemma 3 12B UPDATE_1 checkpoint: $ckpt_12b_update1"
            for lang in "${languages[@]}"; do
                local lang_lower=$(echo "$lang" | tr '[:upper:]' '[:lower:]')
                run_experiment "Eval Gemma 3 12B $lang" \
                    python -m ponderttt.experiments.compare_methods \
                        --model_scale 12b \
                        --checkpoint_path "$ckpt_12b_update1" \
                        --num_eval_batches $NUM_EVAL_BATCHES_OOD_12B \
                        --batch_size $BATCH_SIZE_12B \
                        --language "$lang" \
                        --output_dir "outputs/gemma3/eval/12b_${lang_lower}"
            done
        fi
    fi

    log_info "Phase 3 Complete!"
}

# ============================================================
# Phase 4: Verification (Quick Sanity Check)
# ============================================================
phase4_verify() {
    log_phase "Phase 4: Model Verification"

    if [ "$RUN_1B" = true ]; then
        run_experiment "Verify Gemma 3 1B" \
            python scripts/verify_gemma3_nnx.py --model_scale 1b
    fi

    if [ "$RUN_4B" = true ]; then
        run_experiment "Verify Gemma 3 4B" \
            python scripts/verify_gemma3_nnx.py --model_scale 4b
    fi

    if [ "$RUN_12B" = true ]; then
        run_experiment "Verify Gemma 3 12B" \
            python scripts/verify_gemma3_nnx.py --model_scale 12b
    fi

    log_info "Phase 4 Complete!"
}

# ============================================================
# Phase 5: Shuffled Input Ablation
# ============================================================
phase5_shuffle() {
    log_phase "Phase 5: Shuffled Input Ablation"

    local SKIP_EXAMPLES=160000

    if [ "$RUN_4B" = true ]; then
        local ckpt_4b_update1=$(get_latest_checkpoint "outputs/gemma3/baselines/4b_update1/checkpoints")
        if [ -z "$ckpt_4b_update1" ]; then
            log_error "No UPDATE_1 checkpoint found for Gemma 3 4B."
        else
            run_experiment "Shuffle Ablation Gemma 3 4B" \
                python -m ponderttt.experiments.compare_methods \
                    --model_scale 4b \
                    --checkpoint_path "$ckpt_4b_update1" \
                    --num_eval_batches $NUM_EVAL_BATCHES_4B \
                    --batch_size $BATCH_SIZE_4B \
                    --language Python \
                    --skip_examples $SKIP_EXAMPLES \
                    --output_dir outputs/gemma3/eval/4b_shuffle \
                    --shuffle
        fi
    fi

    if [ "$RUN_12B" = true ]; then
        local ckpt_12b_update1=$(get_latest_checkpoint "outputs/gemma3/baselines/12b_update1/checkpoints")
        if [ -z "$ckpt_12b_update1" ]; then
            log_error "No UPDATE_1 checkpoint found for Gemma 3 12B."
        else
            run_experiment "Shuffle Ablation Gemma 3 12B" \
                python -m ponderttt.experiments.compare_methods \
                    --model_scale 12b \
                    --checkpoint_path "$ckpt_12b_update1" \
                    --num_eval_batches $NUM_EVAL_BATCHES_12B \
                    --batch_size $BATCH_SIZE_12B \
                    --language Python \
                    --skip_examples $SKIP_EXAMPLES \
                    --output_dir outputs/gemma3/eval/12b_shuffle \
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

    local SKIP_EXAMPLES=160000

    if [ "$RUN_4B" = true ]; then
        local ckpt_4b_update1=$(get_latest_checkpoint "outputs/gemma3/baselines/4b_update1/checkpoints")
        if [ -z "$ckpt_4b_update1" ]; then
            log_error "No UPDATE_1 checkpoint found for Gemma 3 4B."
        else
            run_experiment "Diagonal Ablation Gemma 3 4B (k=-1)" \
                python -m ponderttt.experiments.compare_methods \
                    --model_scale 4b \
                    --checkpoint_path "$ckpt_4b_update1" \
                    --num_eval_batches $NUM_EVAL_BATCHES_4B \
                    --batch_size $BATCH_SIZE_4B \
                    --language Python \
                    --skip_examples $SKIP_EXAMPLES \
                    --output_dir outputs/gemma3/eval/4b_diagonal_k_minus_1 \
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
    log_info "Running ALL Gemma 3 experiment phases..."
    echo ""
    echo "Phase 1: Baseline Training      -> UPDATE_1, UPDATE_2, UPDATE_4"
    echo "Phase 2: Eval Python (ID)       -> SKIP, UPDATE_1, Oracle, TTT Improvement"
    echo "Phase 3: Eval OOD               -> JavaScript, Java, Go"
    echo "Phase 4: Model Verification     -> Sanity checks"
    echo "Phase 5: Shuffle Ablation       -> Shuffled input ablation"
    echo "Phase 6: Diagonal Ablation      -> Diagonal mask ablation"
    echo ""

    phase4_verify   # Run verification first
    phase1_baselines
    phase2_eval_id
    phase3_eval_ood
    phase5_shuffle
    phase6_diagonal
    print_summary
}


# Parse arguments - first pass: extract model flags
PHASES=()

for arg in "$@"; do
    case $arg in
        --1b)
            RUN_1B=true
            ;;
        --4b)
            RUN_4B=true
            ;;
        --12b)
            RUN_12B=true
            ;;
        --27b)
            RUN_27B=true
            ;;
        --all-models)
            RUN_1B=true
            RUN_4B=true
            RUN_12B=true
            RUN_27B=true
            ;;
        *)
            PHASES+=("$arg")
            ;;
    esac
done

# If no model flag specified, default to 4B and 12B
if [ "$RUN_1B" = false ] && [ "$RUN_4B" = false ] && [ "$RUN_12B" = false ] && [ "$RUN_27B" = false ]; then
    RUN_4B=true
    RUN_12B=true
fi


# Log which models will be run
log_info "Running experiments for Gemma 3 models:"
[ "$RUN_1B" = true ] && echo "  - Gemma 3 1B (testing)"
[ "$RUN_4B" = true ] && echo "  - Gemma 3 4B"
[ "$RUN_12B" = true ] && echo "  - Gemma 3 12B"
[ "$RUN_27B" = true ] && echo "  - Gemma 3 27B"

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
            phase4|verify)
                phase4_verify
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
                echo "  phase4, verify          - Model verification"
                echo "  phase5, shuffle         - Shuffle ablation"
                echo "  phase6, diagonal        - Diagonal mask ablation"
                echo "  all                     - Run all phases"
                echo ""
                echo "Model selection:"
                echo "  --1b                    - Run only Gemma 3 1B (for testing)"
                echo "  --4b                    - Run only Gemma 3 4B"
                echo "  --12b                   - Run only Gemma 3 12B"
                echo "  --27b                   - Run only Gemma 3 27B"
                echo "  --all-models            - Run all models"
                exit 1
                ;;
        esac
    done
    print_summary
fi
