import sys
sys.path.insert(0, 'src')

import torch
from src.ponderttt.models import (
    IterativeTransformerConfig,
    IterativeTransformerTTT,
    UniformPolicy,
)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Baseline config (uniform_k4)
baseline_config = IterativeTransformerConfig(
    hidden_dim=512,
    num_layers=6,
    num_heads=8,
    ffn_dim=2048,
    use_iterative_ttt=True,
    ttt_layer_indices=[2, 3, 4],
    fast_weight_hidden_dim=64,
    step_options=[1, 2, 4, 8],
    max_steps=4,
    use_learned_policy=False,  # No learned policy
    lambda_compute=0.0,
)

# Learned config
learned_config = IterativeTransformerConfig(
    hidden_dim=512,
    num_layers=6,
    num_heads=8,
    ffn_dim=2048,
    use_iterative_ttt=True,
    ttt_layer_indices=[2, 3, 4],
    fast_weight_hidden_dim=64,
    step_options=[1, 2, 4, 8],
    max_steps=8,
    use_learned_policy=True,  # Learned policy
    lambda_compute=0.01,
    target_avg_steps=4.0,
)

# Create models
print('Creating baseline model...')
baseline_model = IterativeTransformerTTT(baseline_config)

print('Creating learned model...')
learned_model = IterativeTransformerTTT(learned_config)

# Count before replacement
baseline_params_before = count_parameters(baseline_model)
learned_params = count_parameters(learned_model)

print(f'\nBefore policy replacement:')
print(f'Baseline (use_learned_policy=False): {baseline_params_before:,} ({baseline_params_before/1e6:.2f}M)')
print(f'Learned (use_learned_policy=True): {learned_params:,} ({learned_params/1e6:.2f}M)')
print(f'Difference: {learned_params - baseline_params_before:,} ({(learned_params - baseline_params_before)/1e6:.2f}M)')

# Now replace baseline policy with UniformPolicy (as done in full_comparison_suite.py)
print('\nReplacing baseline policies with UniformPolicy...')
for block in baseline_model.blocks:
    if hasattr(block, 'halting_policy') and block.halting_policy is not None:
        block.halting_policy = UniformPolicy(
            fixed_steps=4,
            step_options=[1, 2, 4, 8],
        )

baseline_params_after = count_parameters(baseline_model)

print(f'\nAfter policy replacement:')
print(f'Baseline (with UniformPolicy): {baseline_params_after:,} ({baseline_params_after/1e6:.2f}M)')
print(f'Learned (with HaltingPolicyNetwork): {learned_params:,} ({learned_params/1e6:.2f}M)')
print(f'Difference: {learned_params - baseline_params_after:,} ({(learned_params - baseline_params_after)/1e6:.2f}M)')
print(f'Difference percentage: {(learned_params - baseline_params_after) / baseline_params_after * 100:.1f}%')
