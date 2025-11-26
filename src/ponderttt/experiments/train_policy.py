"""
Train adaptive policy using PID-Lagrangian PPO (NNX version).

Usage:
    python -m ponderttt.experiments.train_policy --model_scale 125m --num_iterations 100
"""

import argparse
import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from tqdm import tqdm

from ..data import create_data_iterator, get_tokenizer
from ..models import PolicyConfig, PolicyNetwork, load_ttt_model
from ..models.policy_nnx import compute_gae
from ..training import PIDController
from ..utils import FeatureExtractor, cross_entropy_loss
from ..utils.checkpointing import save_checkpoint, load_checkpoint, finalize_checkpointing
from .training_utils import run_chunk_step
import wandb


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train adaptive policy (NNX)")

    parser.add_argument(
        "--model_scale",
        type=str,
        choices=["125m", "350m", "1b"],
        default="125m",
        help="Model scale (for tokenizer)",
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=10,
        help="Number of training iterations",
    )
    parser.add_argument(
        "--rollout_length",
        type=int,
        default=32,
        help="Number of chunks per rollout",
    )
    parser.add_argument(
        "--budget_limit",
        type=float,
        default=3.0,
        help="Target average computational cost",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--ppo_epochs",
        type=int,
        default=4,
        help="Number of PPO epochs per rollout",
    )
    parser.add_argument(
        "--ppo_minibatch_size",
        type=int,
        default=32,
        help="Minibatch size for PPO updates",
    )
    parser.add_argument(
        "--value_clip",
        type=float,
        default=0.2,
        help="Value function clipping epsilon",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Global gradient norm clip",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/policy_nnx",
        help="Output directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--fast_weight_type",
        type=str,
        choices=["ttt", "lora"],
        default="ttt",
        help="Type of fast weights: 'ttt' (TTT Layer) or 'lora' (LoRA)"
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=64,
        help="LoRA rank (only used if fast_weight_type='lora')"
    )
    parser.add_argument(
        "--ssl_weight",
        type=float,
        default=0.1,
        help="Weight for SSL auxiliary loss when using TTT/LoRA fast weights",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Optional comma-separated list of seeds for multi-seed runs (e.g., '0,1,2')",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="WandB project name (if None, WandB is disabled)",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=100,
        help="Save checkpoint every N steps",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to checkpoint directory to resume from",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=32,
        help="Number of parallel workers for data downloading",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Optional override for data cache size (defaults to required minimum)",
    )
    parser.add_argument(
        "--no_cache",
        action="store_true",
        help="Disable data caching (streaming mode)",
    )

    return parser.parse_args()


def get_model_name(model_scale: str) -> str:
    """Convert model scale to HuggingFace model name."""
    mapping = {
        "125m": "gpt2",
        "350m": "gpt2-medium",
        "1b": "gpt2-large",
    }
    return mapping[model_scale]


# GAE computation is now imported from policy_nnx.compute_gae
# which uses jax.lax.scan for JAX-compatible implementation


def main():
    args = parse_args()
    seeds = [args.seed] if args.seeds is None else [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    print("=" * 60)
    print("PonderTTT Policy Training (NNX)")
    print("=" * 60)
    print(f"Model scale: {args.model_scale}")
    print(f"Iterations: {args.num_iterations}")
    print(f"Budget limit: {args.budget_limit}x")
    print(f"Rollout length: {args.rollout_length}")
    print(f"Output dir: {args.output_dir}")
    print()
    print(f"Seeds: {seeds}")

    if args.budget_limit <= 0:
        raise ValueError("budget_limit must be positive")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get model configuration
    model_name = get_model_name(args.model_scale)

    if args.wandb_project:
        wandb.init(
            project=args.wandb_project,
            config=vars(args),
            name=f"rl_{args.model_scale}_{args.budget_limit}",
        )

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = get_tokenizer(model_name)

    # Create data iterator
    print("Creating data iterator...")
    batch_size = args.batch_size
    seq_length = 1024
    chunk_size = 512
    chunks_per_sequence = max(1, seq_length // chunk_size)

    # Estimate examples needed; each batch provides chunks_per_sequence chunk steps
    total_chunk_steps = args.num_iterations * args.rollout_length
    batches_needed = math.ceil(total_chunk_steps / chunks_per_sequence)
    min_examples = batches_needed * batch_size
    max_examples = min_examples if args.max_examples is None else args.max_examples
    if max_examples < min_examples:
        print(
            f"Warning: max_examples ({max_examples}) < required minimum ({min_examples}); "
            "data iterator may exhaust before training completes."
        )

    def build_data_iter():
        return create_data_iterator(
            tokenizer=tokenizer,
            split="train",
            batch_size=batch_size,
            seq_length=seq_length,
            chunk_size=chunk_size,
            max_examples=max_examples,
            num_workers=args.num_workers,
            cache_data=not args.no_cache,
        )

    # Initialize policy network
    print("\nInitializing policy network...")
    policy_config = PolicyConfig(
        feature_dim=32,
        hidden_dim=128,
        num_actions=4,
        dropout_rate=0.1,
    )

    rngs = nnx.Rngs(args.seed)
    policy = PolicyNetwork(config=policy_config, rngs=rngs)
    policy.train()  # Enable dropout for training
    policy_rng = jax.random.PRNGKey(args.seed + 2)

    print(f"OK Policy: {policy_config.feature_dim}D -> {policy_config.hidden_dim}D -> {policy_config.num_actions} actions")

    training_history = []
    seed_histories = []

    for seed in seeds:
        print(f"\n=== Running seed {seed} ===")
        
        # Initialize loop variables to prevent unbound errors
        rollout_features = []
        rollout_actions = []
        rollout_log_probs = []
        rollout_values = []
        rollout_rewards = []
        rollout_costs = []
        chunks_collected = 0
        iteration = 0
        avg_cost = 0.0
        avg_reward = 0.0
        loss = 0.0
        last_metrics = {}

        # reinitialize policy RNGs for each seed
        rngs = nnx.Rngs(seed)
        policy = PolicyNetwork(config=policy_config, rngs=rngs)
        policy.train()
        policy_rng = jax.random.PRNGKey(seed + 2)
        
        # Create optimizer for this seed's policy
        optimizer = nnx.Optimizer(
            policy,
            optax.chain(
                optax.clip_by_global_norm(args.max_grad_norm),
                optax.adam(args.learning_rate),
            ),
            wrt=nnx.All(nnx.Param),
        )
        
        start_iteration = 0
        
        # Resume logic (Per Seed Resume is tricky if seeds are in one command)
        # Assuming resume_from points to a specific seed checkpoint directory or general structure.
        # If resume_from is provided, we assume it matches the current seed if we run 1 seed.
        # If multiple seeds, standard practice is to not resume or handle paths carefully.
        # For simplicity, if len(seeds) == 1 and resume_from is set:
        if args.resume_from and len(seeds) == 1:
            print(f"Resuming from checkpoint: {args.resume_from}")
            target = {
                "state": {"policy": nnx.state(policy), "optimizer": nnx.state(optimizer)},
                "step": 0,
                "metadata": {
                    "seed": 0,
                    "model_scale": "",
                    "budget_limit": 0.0,
                }
            }
            ckpt = load_checkpoint(args.resume_from, target=target)
            nnx.update(policy, ckpt["state"]["policy"])
            nnx.update(optimizer, ckpt["state"]["optimizer"])
            start_iteration = ckpt.get("step", 0)
            print(f"Resumed from iteration {start_iteration}")

        # Prepare TTT model template per seed
        if args.fast_weight_type == "lora":
            from ponderttt.models import LoRAConfig

            lora_config = LoRAConfig(
                hidden_dim=768 if args.model_scale == "125m" else 1024 if args.model_scale == "350m" else 1280,
                rank=args.lora_rank,
                alpha=float(args.lora_rank),
                dropout_rate=0.1,
            )
            print(f"  LoRA rank: {args.lora_rank}")
            ttt_model_template, ttt_config = load_ttt_model(
                model_name=model_name,
                fast_weight_type="lora",
                lora_config=lora_config,
                seed=seed + 1,
                load_pretrained=True,
                vocab_size=tokenizer.get_vocab_size(),
            )
        else:
            ttt_model_template, ttt_config = load_ttt_model(
                model_name=model_name,
                fast_weight_type="ttt",
                seed=seed + 1,
                load_pretrained=True,
                vocab_size=tokenizer.get_vocab_size(),
            )

        ttt_model_template.train()
        model_graphdef, model_state = nnx.split(ttt_model_template)
        base_state = jax.tree_util.tree_map(lambda x: x.copy(), model_state)

        # Create initial instances once
        ttt_model = nnx.merge(model_graphdef, base_state)
        ttt_model.train()
        ttt_optimizer = nnx.Optimizer(
            ttt_model,
            optax.chain(
                optax.clip_by_global_norm(args.max_grad_norm),
                optax.adam(args.learning_rate),
            ),
            wrt=nnx.All(nnx.Param),
        )
        # Capture initial optimizer state
        _, ttt_opt_state_init = nnx.split(ttt_optimizer)

        def reset_ttt_model():
            """Reset TTT model and optimizer state."""
            # Reset model parameters
            nnx.update(ttt_model, base_state)
            # Reset optimizer state
            nnx.update(ttt_optimizer, ttt_opt_state_init)
            return ttt_model, ttt_optimizer

        print(f"OK Model template ready: {ttt_config.n_layer} layers")
        print(f"  Fast weight type: {args.fast_weight_type}")

        # Initialize feature extractor
        feature_extractor = FeatureExtractor(
            vocab_size=tokenizer.get_vocab_size(),
            pad_token_id=tokenizer.token_to_id("<|pad|>"),
            seq_length_norm=chunk_size,
        )
        print("OK Feature extractor initialized (32D features)")

        # PID controller for budget constraint
        pid = PIDController(
            kp=0.1,
            ki=0.01,
            kd=0.05,
        )
        print(f"OK PID controller: kp={pid.kp}, ki={pid.ki}, kd={pid.kd}")

        chunk_iterator = None

        data_iter = build_data_iter()

        def chunk_stream():
            while True:
                batch = next(data_iter)
                num_chunks = batch["chunks"].shape[1]
                for idx in range(num_chunks):
                    yield (
                        {
                            "input_ids": batch["chunks"][:, idx, :],
                            "attention_mask": batch["chunk_attention_mask"][:, idx, :],
                        },
                        idx == 0,
                        idx == num_chunks - 1, # is_last_chunk
                    )

        def get_chunk_batch():
            nonlocal chunk_iterator
            if chunk_iterator is None:
                chunk_iterator = chunk_stream()
            return next(chunk_iterator)

        chunks_per_sequence = seq_length // chunk_size
        training_history = []

        for iteration in range(start_iteration, args.num_iterations):
            print(f"\n{'=' * 60}")
            print(f"Iteration {iteration + 1}/{args.num_iterations}")
            print(f"{'=' * 60}")

            feature_extractor.reset_history()
            ttt_model, ttt_optimizer = reset_ttt_model()

            rollout_features = []
            rollout_actions = []
            rollout_log_probs = []
            rollout_values = []
            rollout_rewards = []
            rollout_costs = []
            rollout_dones = []

            # Cost model: 1 (base forward) + 2 * num_steps
            # SKIP=1, UPDATE_1=3, UPDATE_2=5, UPDATE_4=9
            costs_map = jnp.array([1.0, 3.0, 5.0, 9.0])
            step_map = [0, 1, 2, 4]
            total_cost_sum = 0.0  # Sum of all costs for averaging
            total_cost_count = 0  # Number of decisions made
            chunks_collected = 0
            exhausted = False

            # Initialize batch_states cache - only store fast layer state, not full model
            batch_states = [None] * args.batch_size

            with tqdm(total=args.rollout_length, desc="Collecting rollout") as pbar:
                while chunks_collected < args.rollout_length:
                    try:
                        chunk_batch, is_new_sequence, is_last_chunk = get_chunk_batch()
                    except StopIteration:
                        exhausted = True
                        break

                    # Handle batch size logic
                    current_batch_size = chunk_batch["input_ids"].shape[0]

                    # Check for valid tokens (batch-wide)
                    num_valid_tokens = jnp.sum(chunk_batch["attention_mask"][:, 1:])
                    if num_valid_tokens < 16:
                        continue

                    # Buffers for this batch
                    b_features = []
                    b_actions = []
                    b_log_probs = []
                    b_values = []
                    b_rewards = []
                    b_costs = []
                    
                    total_loss_baseline = 0.0
                    total_cost = 0.0
                    
                    # Reset model once if new sequence (more efficient than per-item)
                    if is_new_sequence:
                        ttt_model, ttt_optimizer = reset_ttt_model()
                        batch_states = [None] * args.batch_size  # Clear all states
                    
                    # Process each item in the batch serially to support diverse actions/states
                    for i in range(current_batch_size):
                        # 1. Restore State (if not new sequence)
                        if not is_new_sequence:
                            if i < len(batch_states) and batch_states[i] is not None:
                                fast_layer_state, fast_norm_state = batch_states[i]
                                nnx.update(ttt_model.fast_layer, fast_layer_state)
                                if fast_norm_state is not None and hasattr(ttt_model, 'fast_norm'):
                                    nnx.update(ttt_model.fast_norm, fast_norm_state)
                            else:
                                # Fallback if state missing
                                ttt_model, ttt_optimizer = reset_ttt_model()

                        # 2. Extract single item inputs (keep dims for model compatibility: 1, L)
                        input_ids_i = chunk_batch["input_ids"][i:i+1]
                        mask_i = chunk_batch["attention_mask"][i:i+1]
                        batch_i = {
                            "input_ids": input_ids_i,
                            "attention_mask": mask_i,
                        }
                        
                        # 3. Baseline Forward
                        outputs = ttt_model(input_ids_i, use_ttt=False)
                        logits = outputs["logits"]
                        
                        loss_baseline = float(cross_entropy_loss(
                            logits[:, :-1],
                            input_ids_i[:, 1:],
                            mask_i[:, 1:],
                        ))
                        
                        # 4. Features & Policy
                        # Calculate budget_remaining based on average cost so far vs target
                        if total_cost_count > 0:
                            avg_cost_so_far = total_cost_sum / total_cost_count
                            # How much of budget have we used? (1.0 = at limit, >1.0 = over)
                            budget_usage = avg_cost_so_far / args.budget_limit
                            # Remaining budget: 1.0 - usage, but allow negative to signal overspend
                            budget_remaining = 1.0 - budget_usage
                        else:
                            budget_remaining = 1.0  # Full budget at start
                        
                        features = feature_extractor.extract(
                            input_ids=input_ids_i,
                            logits=logits,
                            attention_mask=mask_i,
                            budget_remaining=budget_remaining,
                        )
                        
                        policy_rng, action_key = jax.random.split(policy_rng)
                        policy_output = policy(
                            features,
                            deterministic=False,
                            rng=action_key,
                        )
                        
                        action = policy_output["action"][0] # Scalar
                        action_idx = int(action)
                        action_steps = step_map[action_idx]
                        cost = float(costs_map[action_idx])
                        
                        # 5. TTT Updates (if any)
                        if action_steps == 0:
                            loss_after_ce = loss_baseline
                        else:
                            metrics = {}
                            for _ in range(action_steps):
                                metrics = run_chunk_step(
                                    ttt_model,
                                    ttt_optimizer,
                                    batch_i,
                                    use_ttt=True,
                                    apply_update=True,
                                    ssl_weight=args.ssl_weight,
                                )
                            loss_after_ce = float(metrics.get("loss_ce", loss_baseline))
                            
                        # 6. Compute Reward
                        reward = loss_baseline - loss_after_ce
                        
                        # 7. Store State - only store fast layer state for memory efficiency
                        batch_states[i] = (
                            nnx.state(ttt_model.fast_layer),
                            nnx.state(ttt_model.fast_norm) if hasattr(ttt_model, 'fast_norm') else None,
                        )
                        
                        # 8. Collect results
                        b_features.append(features)
                        b_actions.append(policy_output["action"])
                        b_log_probs.append(policy_output["log_prob"])
                        b_values.append(policy_output["value"])
                        b_rewards.append(reward)
                        b_costs.append(cost)
                        
                        total_loss_baseline += loss_baseline
                        total_cost += cost

                    # Aggregate results - track total cost for averaging
                    total_cost_sum += total_cost
                    total_cost_count += current_batch_size
                    feature_extractor.update_history(total_loss_baseline / current_batch_size, total_cost / current_batch_size)
                    
                    rollout_features.append(jnp.concatenate(b_features, axis=0))
                    rollout_actions.append(jnp.concatenate(b_actions, axis=0))
                    rollout_log_probs.append(jnp.concatenate(b_log_probs, axis=0))
                    rollout_values.append(jnp.concatenate(b_values, axis=0))
                    rollout_rewards.append(jnp.array(b_rewards))
                    rollout_costs.append(jnp.array(b_costs))
                    rollout_dones.append(jnp.full((current_batch_size,), is_last_chunk))

                    chunks_collected += 1
                    pbar.update(1)
                    pbar.set_postfix(
                        {
                            "avg_cost": f"{total_cost / current_batch_size:.1f}x",
                            "avg_rew": f"{sum(b_rewards) / current_batch_size:.3f}",
                        }
                    )

            if exhausted:
                print("\nData iterator exhausted during rollout collection")
                break

            if len(rollout_rewards) == 0:
                print("No rollout data collected, stopping")
                break

            # Convert to arrays [num_steps]
            feature_dim = rollout_features[0].shape[-1]
            rollout_features_array = jnp.reshape(
                jnp.stack(rollout_features), (-1, feature_dim)
            )
            rollout_actions_array = jnp.reshape(jnp.stack(rollout_actions), (-1,))
            rollout_log_probs_array = jnp.reshape(jnp.stack(rollout_log_probs), (-1,))
            rollout_values_array = jnp.reshape(jnp.stack(rollout_values), (-1,))
            rollout_rewards_array = jnp.reshape(jnp.stack(rollout_rewards), (-1,))
            rollout_costs_array = jnp.reshape(jnp.stack(rollout_costs), (-1,))
            rollout_dones_array = jnp.reshape(jnp.stack(rollout_dones), (-1,)).astype(jnp.float32)

            # Create dones array (now using collected dones)
            dones_array = rollout_dones_array

            # Cost-aware rewards
            adjusted_rewards = rollout_rewards_array - pid.lambda_value * rollout_costs_array

            # Compute advantages and returns using GAE (with jax.lax.scan)
            advantages, returns = compute_gae(
                rewards=adjusted_rewards,
                values=rollout_values_array,
                dones=dones_array,
                gamma=0.99,
                gae_lambda=0.95,
                last_value=0.0,
            )

            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Policy update using minibatches
            old_log_probs = rollout_log_probs_array
            old_values = rollout_values_array
            last_metrics = {}
            loss = math.inf
            data_size = rollout_features_array.shape[0]
            update_rng = jax.random.PRNGKey(args.seed + iteration + 100)

            @nnx.jit
            def train_step_policy(
                policy_model: PolicyNetwork,
                optimizer: nnx.Optimizer,
                mb_features: jax.Array,
                mb_actions: jax.Array,
                mb_old_log_probs: jax.Array,
                mb_old_values: jax.Array,
                mb_advantages: jax.Array,
                mb_returns: jax.Array,
            ):
                def policy_loss_fn(mdl):
                    """Compute PPO loss for minibatch."""
                    outputs = mdl.evaluate_actions(mb_features, mb_actions)
                    new_log_probs = outputs["log_prob"]
                    new_values = outputs["value"]
                    entropies = outputs["entropy"]

                    ratios = jnp.exp(new_log_probs - mb_old_log_probs)
                    clip_ratios = jnp.clip(ratios, 1.0 - args.value_clip, 1.0 + args.value_clip)
                    policy_loss = -jnp.mean(
                        jnp.minimum(ratios * mb_advantages, clip_ratios * mb_advantages)
                    )

                    value_pred_clipped = mb_old_values + jnp.clip(
                        new_values - mb_old_values, -args.value_clip, args.value_clip
                    )
                    value_losses = jnp.square(new_values - mb_returns)
                    value_losses_clipped = jnp.square(value_pred_clipped - mb_returns)
                    value_loss = 0.5 * jnp.mean(jnp.minimum(value_losses, value_losses_clipped))

                    entropy_loss = -0.01 * jnp.mean(entropies)

                    approx_kl = jnp.mean(mb_old_log_probs - new_log_probs)
                    total_loss = policy_loss + value_loss + entropy_loss
                    return total_loss, {
                        "policy_loss": policy_loss,
                        "value_loss": value_loss,
                        "entropy": jnp.mean(entropies),
                        "approx_kl": approx_kl,
                    }

                (loss, metrics), grads = nnx.value_and_grad(policy_loss_fn, has_aux=True)(policy_model)
                optimizer.update(policy_model, grads)
                return loss, metrics

            for _ in range(args.ppo_epochs):
                update_rng, perm_key = jax.random.split(update_rng)
                perm = jax.random.permutation(perm_key, data_size)
                for start in range(0, data_size, args.ppo_minibatch_size):
                    mb_idx = perm[start : start + args.ppo_minibatch_size]
                    mb_features = rollout_features_array[mb_idx]
                    mb_actions = rollout_actions_array[mb_idx]
                    mb_old_log_probs = old_log_probs[mb_idx]
                    mb_old_values = old_values[mb_idx]
                    mb_advantages = advantages[mb_idx]
                    mb_returns = returns[mb_idx]

                    loss, last_metrics = train_step_policy(
                        policy,
                        optimizer,
                        mb_features,
                        mb_actions,
                        mb_old_log_probs,
                        mb_old_values,
                        mb_advantages,
                        mb_returns,
                    )

            # Update PID controller
            avg_cost = float(jnp.mean(rollout_costs_array))
            cost_violation = avg_cost - args.budget_limit
            pid = pid.update(cost_violation)

            # Compute statistics
            avg_reward = float(jnp.mean(rollout_rewards_array))

            print("\nRollout summary:")
            print(f"  Average cost: {avg_cost:.2f}x (target: {args.budget_limit:.1f}x)")
            print(f"  Average reward: {avg_reward:.4f}")
            print(f"  Lambda (penalty): {pid.lambda_value:.4f}")
            print(f"  Policy loss: {loss:.4f}")
            print(f"  Approx KL: {float(last_metrics.get('approx_kl', 0.0)):.6f}")
            print(f"  Chunks collected: {chunks_collected}")

            # WandB Log
            if args.wandb_project:
                wandb.log({
                    f"seed_{seed}/avg_cost": avg_cost,
                    f"seed_{seed}/avg_reward": avg_reward,
                    f"seed_{seed}/lambda": float(pid.lambda_value),
                    f"seed_{seed}/policy_loss": float(loss),
                    f"seed_{seed}/approx_kl": float(last_metrics.get('approx_kl', 0.0)),
                    "iteration": iteration + 1,
                })

            # Save iteration results
            training_history.append({
                "iteration": iteration + 1,
                "avg_cost": avg_cost,
                "avg_reward": avg_reward,
                "lambda": float(pid.lambda_value),
                "policy_loss": float(loss),
                "chunks": chunks_collected,
                "approx_kl": float(last_metrics.get("approx_kl", 0.0)),
            })

            # Periodic Checkpoint
            print(f"DEBUG: Checking checkpoint for iter {iteration+1}")
            if (iteration + 1) % args.save_every == 0 and (iteration + 1) < args.num_iterations:
                checkpoint_dir = output_dir / f"seed_{seed}"
                print(f"Saving checkpoint to {checkpoint_dir} at iter {iteration + 1}...")
                save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    step=iteration + 1,
                    state={"policy": nnx.state(policy), "optimizer": nnx.state(optimizer)},
                    metadata={
                        "seed": seed,
                        "model_scale": args.model_scale,
                        "budget_limit": args.budget_limit,
                    }
                )

        # Save per-seed results
        if training_history:
            results = {
                "config": {
                    "model_scale": args.model_scale,
                    "num_iterations": args.num_iterations,
                    "budget_limit": args.budget_limit,
                    "rollout_length": args.rollout_length,
                    "learning_rate": args.learning_rate,
                    "seed": seed,
                    "ssl_weight": args.ssl_weight,
                    "ppo_minibatch_size": args.ppo_minibatch_size,
                },
                "training_history": training_history,
            }
            results_file = output_dir / f"policy_results_{args.model_scale}_seed{seed}.json"
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nOK Results saved to: {results_file}")
            seed_histories.append(results)
            
            # Save Checkpoint for this seed (Final)
            if iteration + 1 == args.num_iterations:
                checkpoint_dir = output_dir / f"seed_{seed}"
                print(f"Saving final checkpoint to {checkpoint_dir}...")
                save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    step=args.num_iterations,
                    state={"policy": nnx.state(policy), "optimizer": nnx.state(optimizer)},
                    metadata={
                        "seed": seed,
                        "model_scale": args.model_scale,
                        "budget_limit": args.budget_limit,
                    }
                )
                finalize_checkpointing()
        else:
            print("\nNo training completed for this seed!")

    # Final summary across seeds
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)

    if seed_histories:
        last_hist = seed_histories[-1]["training_history"]
        print(f"Total iterations (last seed): {len(last_hist)}")
        print(f"Final average cost (last seed): {last_hist[-1]['avg_cost']:.2f}x")
        print(f"Final average reward (last seed): {last_hist[-1]['avg_reward']:.4f}")

        if len(seed_histories) > 1:
            from ponderttt.utils.statistics import bootstrap_ci, compute_iqm
            final_costs = jnp.array([h["training_history"][-1]["avg_cost"] for h in seed_histories])
            final_rewards = jnp.array([h["training_history"][-1]["avg_reward"] for h in seed_histories])
            summary = {
                "seeds": [h["config"]["seed"] for h in seed_histories],
                "avg_cost_mean": float(final_costs.mean()),
                "avg_cost_iqm": compute_iqm(final_costs),
                "avg_cost_ci": bootstrap_ci(final_costs, n_bootstrap=1000),
                "avg_reward_mean": float(final_rewards.mean()),
                "avg_reward_iqm": compute_iqm(final_rewards),
                "avg_reward_ci": bootstrap_ci(final_rewards, n_bootstrap=1000),
            }
            summary_file = output_dir / f"policy_summary_{args.model_scale}.json"
            with open(summary_file, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"\nSeed summary saved to: {summary_file}")
    else:
        print("\nNo training completed!")


if __name__ == "__main__":
    main()