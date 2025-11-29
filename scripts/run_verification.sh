#!/bin/bash
# PonderTTT Verification Experiments
# Runs shuffled input test and latency measurement for Hard Skip models
#
# Usage:
#   ./scripts/run_verification.sh                    # Run all
#   ./scripts/run_verification.sh shuffled           # Shuffled input only
#   ./scripts/run_verification.sh latency            # Latency only

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_phase() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

# Find latest checkpoint
get_latest_checkpoint() {
    local dir="$1"
    if [ ! -d "$dir" ]; then
        echo ""
        return 1
    fi
    local latest=$(ls -d "${dir}"/checkpoint_* 2>/dev/null | \
        sed 's/.*checkpoint_//' | sort -n | tail -1)
    if [ -z "$latest" ]; then
        echo ""
        return 1
    fi
    echo "${dir}/checkpoint_${latest}"
}

# Shuffled Input Test
run_shuffled_test() {
    log_phase "Shuffled Input Sanity Check"

    local ckpt_125m=$(get_latest_checkpoint "outputs/hard_skip/125m_skip0.8")

    if [ -z "$ckpt_125m" ]; then
        log_error "No checkpoint found for 125M Hard Skip"
        return 1
    fi

    log_info "Using checkpoint: $ckpt_125m"

    python scripts/test_shuffled_input.py \
        --checkpoint "$ckpt_125m" \
        --model_scale 125m \
        --num_batches 30 \
        2>&1 | tee outputs/log/shuffled_input_hard_skip.log
}

# Latency Measurement
run_latency_test() {
    log_phase "Latency Measurement"

    local ckpt_125m=$(get_latest_checkpoint "outputs/hard_skip/125m_skip0.8")

    if [ -z "$ckpt_125m" ]; then
        log_error "No checkpoint found for 125M Hard Skip"
        return 1
    fi

    log_info "Using checkpoint: $ckpt_125m"

    python scripts/measure_latency.py \
        --checkpoint "$ckpt_125m" \
        --model_scale 125m \
        --chunk_size 512 \
        --num_trials 100 \
        2>&1 | tee outputs/log/latency_hard_skip.log
}

# Main
mkdir -p outputs/log

if [ $# -eq 0 ]; then
    run_shuffled_test
    run_latency_test
else
    for test in "$@"; do
        case $test in
            shuffled)
                run_shuffled_test
                ;;
            latency)
                run_latency_test
                ;;
            *)
                log_error "Unknown test: $test"
                echo "Available: shuffled, latency"
                exit 1
                ;;
        esac
    done
fi

log_info "Verification complete! Check outputs/log/ for results."
