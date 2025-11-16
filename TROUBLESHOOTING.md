# PonderTTT Troubleshooting Guide

## Common Issues and Solutions

### Installation Issues

#### Issue: `uv: command not found`

**Solution**: Install uv first:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### Issue: CUDA not found when JAX is installed

**Symptom**:
```
WARNING: An NVIDIA GPU may be present on this machine, but a CUDA-enabled jaxlib is not installed.
```

**Solution**: Install JAX with CUDA support:
```bash
uv pip install jax[cuda12_pip] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

---

### Memory Issues

#### Issue: Out of Memory (OOM) Error

**Symptom**:
```
jaxlib.xla_extension.XlaRuntimeError: RESOURCE_EXHAUSTED: Out of memory
```

**Solutions** (try in order):

1. **Reduce Batch Size** (v0.2.0 fix):
   ```python
   # In src/ponderttt/experiments/config.py
   @dataclass
   class TrainingConfig:
       batch_size: int = 2  # Reduce from 4 to 2
   ```

2. **Reduce Sequence Length**:
   ```python
   # In experiments/config.py
   @dataclass
   class ExperimentModelConfig:
       max_seq_length: int = 512  # Reduce from 1024
       chunk_size: int = 256       # Reduce from 512
   ```

3. **Use Gradient Accumulation**:
   ```python
   # Effective batch size = batch_size * accumulation_steps
   batch_size: int = 1
   gradient_accumulation_steps: int = 4
   ```

4. **Enable Mixed Precision** (if supported):
   ```python
   # Use float16 instead of float32
   jax.config.update('jax_default_matmul_precision', 'float16')
   ```

#### Issue: CPU RAM Exhausted

**Symptom**: System freezes or killed by OOM killer

**Solutions**:
1. Reduce `num_workers` in data loading
2. Enable streaming mode (already default for The Stack v2)
3. Reduce `max_examples` in data iterator

---

### Data Loading Issues

#### Issue: The Stack v2 Access Denied

**Symptom**:
```
DatasetNotFoundError: Dataset bigcode/the-stack-v2-dedup is gated.
```

**Solution**:
1. Go to https://huggingface.co/datasets/bigcode/the-stack-v2-dedup
2. Accept the terms and conditions
3. Login with your token:
   ```bash
   uv run huggingface-cli login
   ```

#### Issue: S3 Download Fails

**Symptom**:
```
botocore.exceptions.NoCredentialsError: Unable to locate credentials
```

**Solution**: This is expected. We use unsigned requests:
```python
# Already configured in dataset.py
self.s3_client = boto3.client(
    's3',
    config=Config(signature_version=UNSIGNED)
)
```

If still failing, check internet connection and S3 bucket availability.

---

### Testing Issues

#### Issue: Tests Fail with Import Errors

**Symptom**:
```
ImportError: cannot import name 'X' from 'ponderttt.Y'
```

**Solutions**:
1. Install in development mode:
   ```bash
   uv pip install -e .
   ```

2. Check Python path:
   ```bash
   export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
   ```

#### Issue: JAX Tests Fail on CPU

**Symptom**: Tests expect GPU but running on CPU

**Solution**: This is expected. CPU validation works but is slower:
```bash
# Run with CPU (default)
uv run pytest tests/ -v

# Force CPU even if GPU available
JAX_PLATFORMS=cpu uv run pytest tests/ -v
```

---

### Training Issues

#### Issue: Loss is NaN

**Possible Causes**:
1. Learning rate too high
2. Gradient explosion
3. Numerical instability

**Solutions**:
1. Reduce learning rate:
   ```python
   learning_rate: float = 1e-4  # Reduce from 3e-4
   ```

2. Enable gradient clipping:
   ```python
   max_grad_norm: float = 1.0
   ```

3. Check for inf/nan in data:
   ```python
   assert jnp.all(jnp.isfinite(batch['input_ids']))
   ```

#### Issue: Training is Very Slow on CPU

**Symptom**: Takes minutes per step

**Solution**: This is expected. CPU is ~10-100× slower than GPU.
- For development: Use smaller models and fewer steps
- For production: Use GPU or TPU

---

### TPU-Specific Issues

#### Issue: TPU Not Detected

**Symptom**:
```
RuntimeError: No TPU devices found
```

**Solutions**:
1. Check TPU is created:
   ```bash
   gcloud compute tpus list --zone=us-central2-b
   ```

2. SSH into correct TPU VM:
   ```bash
   gcloud compute tpus tpu-vm ssh ponderttt-v4-64 --zone=us-central2-b
   ```

3. Verify JAX sees TPU:
   ```python
   import jax
   print(jax.devices())  # Should show TPU devices
   ```

#### Issue: Multi-Host Training Hangs

**Symptom**: Training starts but hangs on first step

**Solutions**:
1. Ensure script runs on ALL hosts:
   ```bash
   gcloud compute tpus tpu-vm ssh ponderttt-v4-64 \
     --zone=us-central2-b \
     --worker=all \
     --command="cd ~/ponderttt && python scripts/train_tpu.py"
   ```

2. Check JAX distributed initialization:
   ```python
   jax.distributed.initialize()  # Must be called on all hosts
   ```

---

### Common Error Messages

#### Error: `chunk_size must be <= max_position_embeddings`

**Cause**: GPT-2 has max position 1024, but chunk_size is larger

**Solution**: Already fixed in v0.2.0:
```python
chunk_size: int = 512  # Was 4096, now 512
```

#### Error: `Expected RNG to be a PRNGKey, got None`

**Cause**: Missing or incorrect RNG handling

**Solution**: Ensure RNG is properly initialized:
```python
from ponderttt.utils import init_rng, next_rng

rng = init_rng(42)
rng, model_rng = next_rng(rng)
params = model.init(model_rng, inputs)
```

#### Error: `InvalidRngError: RNGs should be of shape (2,)`

**Cause**: Flax module expects RNG but none provided

**Solution**: Pass RNG during initialization:
```python
# Correct
params = model.init({'params': rng}, inputs)

# Also correct
params = model.init(rng, inputs, deterministic=True)
```

---

### Performance Issues

#### Issue: Training is Slower Than Expected

**Diagnosis**:
```python
import time
start = time.time()
# ... training step ...
print(f"Step time: {time.time() - start:.2f}s")
```

**Common Causes**:
1. **Data loading bottleneck**: Use profiling to check
2. **Not using JIT**: Ensure functions are jitted
3. **CPU fallback**: Verify GPU is actually being used
4. **Small batch size**: Increase if memory allows

**Solutions**:
1. Check device utilization:
   ```bash
   # For GPU
   nvidia-smi -l 1

   # For TPU
   watch -n 1 'gcloud compute tpus describe ponderttt-v4-64 --zone=us-central2-b'
   ```

2. Profile with JAX:
   ```python
   with jax.profiler.trace("/tmp/jax-trace"):
       # ... training code ...
   ```

---

### Code Quality Issues

#### Issue: Ruff Linting Errors

**Solution**: Auto-fix most errors:
```bash
uv run ruff check src/ --fix
```

#### Issue: Type Check Errors with mypy

**Solution**: Add type ignores for known issues:
```python
# type: ignore[attr-defined]
```

Or update to correct types.

---

## Getting Help

If you encounter an issue not covered here:

1. **Check Documentation**:
   - README.md
   - QUICKSTART.md
   - PLAN.md
   - PROJECT_STATUS.md

2. **Run Quick Test**:
   ```bash
   uv run python scripts/quick_test.py
   ```

3. **Check Recent Fixes**:
   - Review recent commits: `git log --oneline -10`
   - v0.2.0 fixed many critical issues

4. **Enable Debug Logging**:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

5. **Open GitHub Issue**:
   - Include error message
   - Include system info (GPU, OS, JAX version)
   - Include minimal reproducible example

---

## Known Limitations (v0.2.0)

1. **Synthetic Data Only**: Real data experiments require GPU
2. **No Unit Tests for Models**: Only metrics tests available
3. **No Safe Code Execution**: HumanEval/MBPP execution is placeholder
4. **TPU Untested**: Code ready but not validated on actual TPU hardware

See PROJECT_STATUS.md for full status and roadmap.
