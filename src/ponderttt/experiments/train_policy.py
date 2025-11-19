"""
Train adaptive policy using PID-Lagrangian PPO (NNX version).

Usage:
    python -m ponderttt.experiments.train_policy --model_scale 125m --num_iterations 100
"""

import argparse
import json
import math
from pathlib import Path
from typing import cast

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from tokenizers import Tokenizer
from tqdm import tqdm

from ..data import create_data_iterator, get_tokenizer
from ..models import PolicyConfig, PolicyNetwork, load_ttt_model
from ..models.policy_nnx import compute_gae
from ..training import PIDController
from ..utils import FeatureExtractor, cross_entropy_loss
from .training_utils import run_chunk_step


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

    print("=" * 60)
    print("PonderTTT Policy Training (NNX)")
    print("=" * 60)
    print(f"Model scale: {args.model_scale}")
    print(f"Iterations: {args.num_iterations}")
    print(f"Budget limit: {args.budget_limit}x")
    print(f"Rollout length: {args.rollout_length}")
    print(f"Output dir: {args.output_dir}")
    print()

    if args.budget_limit <= 0:
        raise ValueError("budget_limit must be positive")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get model configuration
    model_name = get_model_name(args.model_scale)

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = cast(Tokenizer, get_tokenizer(model_name))

    # Create data iterator
    print("Creating data iterator...")
    batch_size = args.batch_size
    seq_length = 2048
    chunk_size = 512
    chunks_per_sequence = max(1, seq_length // chunk_size)

    # Estimate examples needed (2x safety margin)
    examples_per_iteration = math.ceil(args.rollout_length / chunks_per_sequence) * batch_size
    num_examples = examples_per_iteration * args.num_iterations * 2

    data_iter = create_data_iterator(
        tokenizer=tokenizer,
        split="train",
        batch_size=batch_size,
        seq_length=seq_length,
        chunk_size=chunk_size,
        max_examples=num_examples,
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

    print(f"OK Policy: {policy_config.feature_dim}D -> {policy_config.hidden_dim}D -> {policy_config.num_actions} actions")

    # Create optimizer
    print("Creating optimizer...")
    optimizer = nnx.Optimizer(policy, optax.adam(args.learning_rate), wrt=nnx.All(nnx.Param))
    print(f"OK Optimizer: Adam (lr={args.learning_rate})")

    # Initialize TTT model template for environment rollouts
    print(f"\nPreparing TTT model ({args.fast_weight_type.upper()}) for rollouts...")
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
            seed=args.seed + 1,
            load_pretrained=True,
        )
    else:
        ttt_model_template, ttt_config = load_ttt_model(
            model_name=model_name,
            fast_weight_type="ttt",
            seed=args.seed + 1,
            load_pretrained=True,
        )

    ttt_model_template.train()
    model_graphdef, model_state = nnx.split(ttt_model_template)
    base_state = jax.tree_util.tree_map(lambda x: x.copy(), model_state)

    def reset_ttt_model():
        """Create a fresh copy of the TTT model + optimizer."""
        model_state_copy = jax.tree_util.tree_map(lambda x: x.copy(), base_state)
        model = nnx.merge(model_graphdef, model_state_copy)
        model.train()
        optimizer = nnx.Optimizer(model, optax.adam(args.learning_rate), wrt=nnx.All(nnx.Param))
        return model, optimizer

    print(f"OK Model template ready: {ttt_config.n_layer} layers")
    print(f"  Fast weight type: {args.fast_weight_type}")

    # Initialize feature extractor
    feature_extractor = FeatureExtractor(vocab_size=tokenizer.get_vocab_size())
    print(f"OK Feature extractor initialized (32D features)")

    # PID controller for budget constraint
    pid = PIDController(
        kp=0.1,
        ki=0.01,
        kd=0.05,
    )
    print(f"OK PID controller: kp={pid.kp}, ki={pid.ki}, kd={pid.kd}")

    # Training loop
    print("\nStarting policy training...")
    print(f"Target budget: {args.budget_limit}x")
    print()

    training_history = []

    def chunk_stream():
        while True:
            batch = next(data_iter)
            num_chunks = batch["chunks"].shape[1]
            for idx in range(num_chunks):
                yield {
                    "input_ids": batch["chunks"][:, idx, :],
                    "attention_mask": batch["chunk_attention_mask"][:, idx, :],
                }

    chunk_iterator = None

    def get_chunk_batch():
        nonlocal chunk_iterator
        if chunk_iterator is None:
            chunk_iterator = chunk_stream()
        return next(chunk_iterator)

    chunks_per_sequence = seq_length // chunk_size

    for iteration in range(args.num_iterations):
        print(f"\n{'=' * 60}")
        print(f"Iteration {iteration + 1}/{args.num_iterations}")
        print(f"{'=' * 60}")

        feature_extractor.reset_history()
        ttt_model, ttt_optimizer = reset_ttt_model()

        # Collect rollout data
        rollout_features = []
        rollout_actions = []
        rollout_log_probs = []
        rollout_values = []
        rollout_rewards = []
        rollout_costs = []

        costs_map = jnp.array([1.0, 3.0, 6.0, 12.0])
        step_map = [0, 1, 2, 4]
        cost_accumulator = 0.0
        chunks_collected = 0
        exhausted = False

        with tqdm(total=args.rollout_length, desc="Collecting rollout") as pbar:
            while chunks_collected < args.rollout_length:
                try:
                    chunk_batch = get_chunk_batch()
                except StopIteration:
                    exhausted = True
                    break

                outputs = ttt_model(chunk_batch["input_ids"], use_ttt=False)
                logits = outputs["logits"]
                labels = chunk_batch["input_ids"]
                mask = chunk_batch["attention_mask"]
                loss_baseline = cross_entropy_loss(
                    logits[:, :-1],
                    labels[:, 1:],
                    mask[:, 1:],
                )

                budget_remaining = max(
                    0.0,
                    (args.budget_limit - cost_accumulator) / max(args.budget_limit, 1e-8),
                )

                features = feature_extractor.extract(
                    input_ids=chunk_batch["input_ids"],
                    logits=logits,
                    budget_remaining=budget_remaining,
                )
                features_mean = jnp.mean(features, axis=0, keepdims=True)

                policy_output = policy(
                    features_mean,
                    deterministic=False,
                    rng=rngs.action(),
                )

                action_idx = int(policy_output["action"][0])
                action_steps = step_map[action_idx]
                cost = float(costs_map[action_idx])

                if action_steps == 0:
                    loss_after = float(loss_baseline)
                else:
                    metrics = None
                    for _ in range(action_steps):
                        metrics = run_chunk_step(
                            ttt_model,
                            ttt_optimizer,
                            chunk_batch,
                            use_ttt=True,
                            apply_update=True,
                        )
                    assert metrics is not None
                    loss_after = metrics["loss"]

                quality_improvement = float(loss_baseline) - loss_after
                lambda_penalty = pid.lambda_value
                reward = quality_improvement - lambda_penalty * (cost / args.budget_limit)
                cost_accumulator += cost

                feature_extractor.update_history(float(loss_baseline), cost)

                rollout_features.append(features_mean)
                rollout_actions.append(jnp.array([action_idx], dtype=jnp.int32))
                rollout_log_probs.append(policy_output["log_prob"])
                rollout_values.append(policy_output["value"])
                rollout_rewards.append(reward)
                rollout_costs.append(cost)

                chunks_collected += 1
                pbar.update(1)
                pbar.set_postfix(
                    {
                        "action": action_idx,
                        "cost": f"{cost:.1f}x",
                        "reward": f"{reward:.3f}",
                    }
                )

        if exhausted:
            print("\nData iterator exhausted during rollout collection")
            break

        if len(rollout_rewards) == 0:
            print("No rollout data collected, stopping")
            break

        # Convert to arrays [num_steps]
        rollout_features_array = jnp.concatenate(rollout_features, axis=0)
        rollout_actions_array = jnp.concatenate(rollout_actions, axis=0).flatten()
        rollout_log_probs_array = jnp.concatenate(rollout_log_probs, axis=0).flatten()
        rollout_values_array = jnp.concatenate(rollout_values, axis=0).flatten()
        rollout_rewards_array = jnp.array(rollout_rewards)
        rollout_costs_array = jnp.array(rollout_costs)

        # Create dones array (all False for continuous rollout)
        dones_array = jnp.zeros_like(rollout_rewards_array)

        # Compute advantages and returns using GAE (with jax.lax.scan)
        advantages, returns = compute_gae(
            rewards=rollout_rewards_array,
            values=rollout_values_array,
            dones=dones_array,
            gamma=0.99,
            gae_lambda=0.95,
            last_value=float(rollout_values_array[-1]),
        )

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Policy update using vectorized operations (batch processing)
        def policy_loss_fn(mdl):
            """Compute PPO loss for entire rollout in one batch."""
            # Re-evaluate entire batch with current policy
            outputs = mdl.evaluate_actions(rollout_features_array, rollout_actions_array)
            new_log_probs = outputs["log_prob"]  # [rollout_length]
            new_values = outputs["value"]        # [rollout_length]
            entropies = outputs["entropy"]       # [rollout_length]

            # PPO clipped surrogate objective (vectorized)
            ratios = jnp.exp(new_log_probs - rollout_log_probs_array)
            clip_ratios = jnp.clip(ratios, 0.8, 1.2)
            policy_loss = -jnp.mean(jnp.minimum(
                ratios * advantages,
                clip_ratios * advantages,
            ))

            # Value loss (vectorized)
            value_loss = 0.5 * jnp.mean(jnp.square(new_values - returns))

            # Entropy bonus (vectorized)
            entropy_loss = -0.01 * jnp.mean(entropies)

            total_loss = policy_loss + value_loss + entropy_loss
            return total_loss

        # Compute gradients and update
        loss_fn = nnx.value_and_grad(policy_loss_fn)
        loss, grads = loss_fn(policy)
        optimizer.update(policy, grads)

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
        print(f"  Chunks collected: {len(rollout_rewards)}")

        # Save iteration results
        training_history.append({
            "iteration": iteration + 1,
            "avg_cost": avg_cost,
            "avg_reward": avg_reward,
            "lambda": float(pid.lambda_value),
            "policy_loss": float(loss),
            "chunks": len(rollout_rewards),
        })

    # Final summary
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)

    if training_history:
        print(f"Total iterations: {len(training_history)}")
        print(f"Final average cost: {training_history[-1]['avg_cost']:.2f}x")
        print(f"Final average reward: {training_history[-1]['avg_reward']:.4f}")

        # Save results
        results = {
            "config": {
                "model_scale": args.model_scale,
                "num_iterations": args.num_iterations,
                "budget_limit": args.budget_limit,
                "rollout_length": args.rollout_length,
                "learning_rate": args.learning_rate,
            },
            "training_history": training_history,
        }

        results_file = output_dir / f"policy_results_{args.model_scale}.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nOK Results saved to: {results_file}")
    else:
        print("\nNo training completed!")


if __name__ == "__main__":
    main()
