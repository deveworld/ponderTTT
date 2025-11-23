#!/bin/bash
# Setup script for PonderTTT development environment

set -e

echo "Setting up PonderTTT environment..."

# Check Python version
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Install dependencies
echo "Installing development dependencies..."
uv pip install -Ue . --group dev

# Install GPU dependencies
echo "Installing development dependencies..."
uv pip install -Ue . --group gpu

# Create necessary directories
echo "Creating output directories..."
mkdir -p outputs/baselines
mkdir -p outputs/policy
mkdir -p logs

# Configure XLA for stability on modern GPUs (Hopper/Blackwell)
# These flags prevent crashes on RTX 4090/5090/H100
export XLA_FLAGS="--xla_gpu_enable_cublaslt=true --xla_gpu_cublas_fallback=true --xla_gpu_enable_command_buffer=''"
echo "Configured XLA_FLAGS for high-end GPU stability."

echo "Environment setup complete!"
echo ""
echo "IMPORTANT: Run 'source scripts/setup_environment.sh' to apply environment variables."
echo "To test the installation, run:"
echo "  python scripts/quick_test.py"
echo "  python scripts/test_weight_tying.py"
echo "  python scripts/test_pipeline.py"
