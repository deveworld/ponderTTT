#!/bin/bash
# Setup script for PonderTTT development environment

set -e

echo "Setting up PonderTTT environment..."

# Check Python version
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Install dependencies
echo "Installing dependencies..."
uv pip install -e .

# Install development dependencies
echo "Installing development dependencies..."
uv pip install -e ".[dev]"

# Create necessary directories
echo "Creating output directories..."
mkdir -p outputs/baselines
mkdir -p outputs/policy
mkdir -p logs

echo "Environment setup complete!"
echo ""
echo "To test the installation, run:"
echo "  python scripts/test_pipeline.py"
