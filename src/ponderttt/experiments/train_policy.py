"""
Train adaptive policy using PID-Lagrangian PPO.

Usage:
    python -m ponderttt.experiments.train_policy --model_scale 125m --num_iterations 100
"""

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
from tqdm import tqdm

from ..data import create_data_iterator, get_tokenizer
from ..models import (
    PolicyConfig,
    PolicyNetwork,
)
from ..training import PIDController
from ..utils import FeatureExtractor, init_rng, next_rng
from .config import get_1b_config, get_125m_config, get_350m_config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train adaptive policy")

    parser.add_argument(
        "--model_scale",
        type=str,
        choices=["125m", "350m", "1b"],
        default="125m",
        help="Model scale",
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=10,
        help="Number of training iterations",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/policy",
        help="Output directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    print("=" * 60)
    print("PonderTTT Policy Training")
    print("=" * 60)
    print(f"Model scale: {args.model_scale}")
    print(f"Iterations: {args.num_iterations}")
    print(f"Output dir: {args.output_dir}")
    print()

    # Get configuration
    if args.model_scale == "125m":
        config = get_125m_config()
    elif args.model_scale == "350m":
        config = get_350m_config()
    else:
        config = get_1b_config()

    config.seed = args.seed
    config.output_dir = args.output_dir
    config.training.num_iterations = args.num_iterations

    # Initialize RNG
    init_rng(config.seed)

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = get_tokenizer(config.model.model_name)

    # Create data iterator
    print("Creating data iterator...")
    data_iter = create_data_iterator(
        tokenizer=tokenizer,
        split="train",
        batch_size=config.training.batch_size,
        seq_length=config.model.max_seq_length,
        chunk_size=config.model.chunk_size,
        max_examples=config.training.num_train_examples,
    )

    # Initialize models
    print("Initializing models...")

    # Policy network
    policy_config = PolicyConfig(
        feature_dim=32,
        hidden_dim=128,
        num_actions=4,
    )
    policy = PolicyNetwork(config=policy_config)

    # Initialize policy parameters
    rng = next_rng()
    dummy_features = jnp.ones((1, 32))
    policy_variables = policy.init(rng, dummy_features, deterministic=True)
    policy_params = policy_variables['params']

    # Feature extractor
    FeatureExtractor(vocab_size=tokenizer.vocab_size)

    # PID controller
    pid = PIDController(
        kp=config.training.pid_kp,
        ki=config.training.pid_ki,
        kd=config.training.pid_kd,
    )

    # Training loop
    print("\nStarting policy training...")
    print(f"Budget limit: {config.training.budget_limit:.1f}×")
    print(f"Rollout length: {config.training.rollout_length}")
    print()

    training_history = []

    for iteration in range(config.training.num_iterations):
        print(f"\n{'='*60}")
        print(f"Iteration {iteration + 1}/{config.training.num_iterations}")
        print(f"{'='*60}")

        # Collect rollout
        rollout_data = []
        rollout_cost = 0.0
        rollout_rewards = []

        with tqdm(total=config.training.rollout_length, desc="Collecting rollout") as pbar:
            batch = next(data_iter)
            chunks = batch["chunks"]
            batch_size, num_chunks, chunk_size = chunks.shape

            for i in range(min(num_chunks, config.training.rollout_length)):
                chunks[:, i, :]

                # Extract features (dummy for now)
                features = jnp.ones((batch_size, 32)) * (i / num_chunks)

                # Get action from policy
                policy_output = policy.apply(
                    {'params': policy_params},
                    features,
                    deterministic=False,
                    rngs={'action': next_rng()},
                )

                action = policy_output['action'][0]
                log_prob = policy_output['log_prob'][0]
                value = policy_output['value'][0]

                # Compute cost
                costs = jnp.array([1.0, 3.0, 5.0, 12.0])
                cost = costs[action]

                # Compute reward (quality improvement - cost penalty)
                quality_improvement = jnp.log(chunk_size + 1.0) / 10.0
                lambda_penalty = pid.lambda_value
                reward = quality_improvement - lambda_penalty * (cost / config.training.budget_limit)

                rollout_cost += float(cost)
                rollout_rewards.append(float(reward))

                rollout_data.append({
                    'features': features,
                    'action': action,
                    'log_prob': log_prob,
                    'value': value,
                    'reward': reward,
                    'cost': cost,
                })

                pbar.update(1)
                pbar.set_postfix({
                    'action': int(action),
                    'cost': f"{cost:.1f}×",
                    'reward': f"{reward:.3f}",
                })

        # Update PID controller
        avg_cost = rollout_cost / len(rollout_data)
        cost_violation = avg_cost - config.training.budget_limit
        pid = pid.update(cost_violation)

        # Compute statistics
        avg_reward = sum(rollout_rewards) / len(rollout_rewards)

        print("\nRollout summary:")
        print(f"  Average cost: {avg_cost:.2f}× (target: {config.training.budget_limit:.1f}×)")
        print(f"  Average reward: {avg_reward:.4f}")
        print(f"  Lambda (penalty): {pid.lambda_value:.4f}")

        # Save iteration results
        training_history.append({
            'iteration': iteration + 1,
            'avg_cost': avg_cost,
            'avg_reward': avg_reward,
            'lambda': float(pid.lambda_value),
        })

    # Final summary
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Total iterations: {config.training.num_iterations}")
    print(f"Final average cost: {training_history[-1]['avg_cost']:.2f}×")
    print(f"Final average reward: {training_history[-1]['avg_reward']:.4f}")

    # Save results
    output_dir = Path(args.output_dir) / args.model_scale
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'config': {
            'model_scale': args.model_scale,
            'num_iterations': args.num_iterations,
            'budget_limit': config.training.budget_limit,
            'rollout_length': config.training.rollout_length,
        },
        'training_history': training_history,
    }

    output_file = output_dir / "training_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
