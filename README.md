<h1 align="center">When to Ponder: Adaptive Compute Allocation for Code Generation via Test-Time Training</h1>

<p align="center">
    <a href="https://github.com/deveworld">Gihyeon Sim</a>
</p>

<p align="center">
    <a href="https://arxiv.org/abs/2601.00894"><img src="https://img.shields.io/badge/arXiv-2601.00894-b31b1b.svg"></a>
    <a href="https://doi.org/10.5281/zenodo.18160850"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.18160850.svg"></a>
    <a href="https://ponderttt.worldsw.dev"><img src="https://img.shields.io/badge/Project-Page-blue"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg"></a>
</p>

<p align="center">
    <a href="https://arxiv.org/abs/2601.00894"><b>Paper</b></a> |
    <a href="https://ponderttt.worldsw.dev"><b>Project Page</b></a> |
    <a href="#setup"><b>Setup</b></a> |
    <a href="#usage"><b>Usage</b></a> |
    <a href="#citation"><b>Citation</b></a>
</p>

## Abstract

**PonderTTT** applies selective TTT updates based on input difficulty using the reconstruction loss as a training-free gating signal. A single scalar threshold, calibrated on unlabeled data and adapted during inference, governs update frequency. Testing on GPT-2 models (124M to 1.5B parameters) shows **82–89% Oracle Recovery** while being fully training-free.

## Results

| Model | SKIP | Oracle | Ours | Recovery |
| :---- | :--: | :----: | :--: | :--------: |
| Small (124M) | 2.324 | 1.935 | **1.977** | 89.2% |
| Medium (355M) | 1.909 | 1.653 | **1.697** | 82.8% |
| Large (774M) | 2.005 | 1.580 | **1.656** | 82.1% |
| XL (1.5B) | 1.875 | 1.518 | **1.576** | 83.8% |

## Setup

This codebase is implemented in [JAX](https://github.com/google/jax) and has been tested on both GPUs and Cloud TPU VMs.

### Installation

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install the project
uv pip install -e .              # CPU
uv pip install -e . --group gpu  # CUDA 13
uv pip install -e . --group tpu  # TPU
```

## Usage

### Reconstruction Gating

```python
output = model(input_ids, use_ttt=True)
recon_loss = output["ttt_stats"]["ttt_loss_step_0"]

if recon_loss > threshold:
    # UPDATE: re-forward with updated weights
    pass
else:
    # SKIP: use current weights
    pass
```

### Reproduce Paper Results

```bash
./scripts/run_all_experiments.sh          # All models
./scripts/run_all_experiments.sh --small  # Small (124M)
./scripts/run_all_experiments.sh --xl     # XL (1.5B)
```

## Citation

```bibtex
@article{sim2025ponderttt,
  title={When to Ponder: Adaptive Compute Allocation for Code Generation via Test-Time Training},
  author={Sim, Gihyeon},
  journal={arXiv preprint arXiv:2601.00894},
  year={2025}
}
```

## License

This project is licensed under the MIT License.
