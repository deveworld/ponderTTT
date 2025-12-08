"""
Train binary gating network for adaptive TTT with Hard Skip (Top-k Discriminative).

Uses Gumbel-Softmax for differentiable binary (SKIP/UPDATE) decisions.
- SKIP: No TTT computation (cost = 1.0x)
- UPDATE: Full TTT computation (cost = 3.0x)

Key insight: Instead of "UPDATE if advantage > 0", we ask
"Which k% of chunks benefit MOST from TTT?"

This ensures:
1. Fair comparison with random baseline (same budget)
2. Discriminative learning (relative ranking, not absolute threshold)
3. Automatic sparsity control via target_update_rate parameter

Usage:
    python -m ponderttt.experiments.train_hard_skip --model_scale 125m --target_update_rate 0.3 \\
        --ttt_checkpoint outputs/baselines_nnx/checkpoints/checkpoint_5000/
"""

import argparse
import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from typing import cast

from ..data import create_data_iterator, get_tokenizer
from ..models import GPT2Model, load_ttt_model
from ..models.gating_nnx import BinaryGatingConfig, BinaryGatingNetwork
from ..utils import FeatureExtractor, cross_entropy_loss, per_sample_cross_entropy_loss
from ..utils.checkpointing import save_checkpoint, load_checkpoint, finalize_checkpointing

import wandb


def unwrap_state(state):
    """Recursively unwrap Orbax-serialized NNX state dicts (remove 'value' wrappers).

    Also converts integer keys to strings to ensure consistent key types,
    which is required for NNX state sorting during optimizer creation.
    """
    if isinstance(state, dict):
        if "value" in state and len(state) == 1:
            return state["value"]
        # Convert integer keys to strings for consistency (nnx.List indices)
        return {str(k) if isinstance(k, int) else k: unwrap_state(v) for k, v in state.items()}
    return state


def normalize_state_keys(state):
    """Recursively convert all dict keys to strings for NNX state compatibility.

    NNX requires consistent key types for sorting. This converts int keys
    (from nnx.List indices) to strings.
    """
    from flax.nnx.statelib import State

    if isinstance(state, State):
        # Convert State's internal mapping
        normalized = {}
        for k, v in state._mapping.items():
            new_key = str(k) if isinstance(k, int) else k
            normalized[new_key] = normalize_state_keys(v)
        return State(normalized)
    elif isinstance(state, dict):
        return {str(k) if isinstance(k, int) else k: normalize_state_keys(v) for k, v in state.items()}
    else:
        return state


def compute_topk_targets(advantages: jax.Array, k: float) -> tuple[jax.Array, jax.Array]:
    """
    Compute top-k targets based on advantage values.

    Args:
        advantages: Per-sample advantage values [B]
        k: Target update rate (0.0 to 1.0), e.g., 0.3 means top 30%

    Returns:
        targets: Binary targets [B] where 1 = UPDATE (top k%), 0 = SKIP
        threshold: The threshold value used for this batch
    """
    # Compute threshold: (1-k) percentile
    # e.g., k=0.3 means top 30%, so threshold = 70th percentile
    percentile = (1.0 - k) * 100.0
    threshold = jnp.percentile(advantages, percentile)

    # Samples with advantage >= threshold are in top-k
    targets = (advantages >= threshold).astype(jnp.float32)

    return targets, threshold


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Hard Skip gating (Binary Gating with Gumbel-Softmax)")

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
        default=10000,
        help="Number of training iterations (batches)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate for gating network",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size (sequences)",
    )
    parser.add_argument(
        "--target_update_rate",
        type=float,
        default=0.3,
        help="Target update rate (0.3 = top 30%% get UPDATE, 70%% SKIP)",
    )
    parser.add_argument(
        "--ttt_checkpoint",
        type=str,
        default=None,
        help="Path to pre-trained TTT checkpoint (required for meaningful training)",
    )
    parser.add_argument(
        "--threshold_ema_decay",
        type=float,
        default=0.99,
        help="EMA decay for threshold tracking (for inference)",
    )
    parser.add_argument(
        "--initial_temperature",
        type=float,
        default=1.0,
        help="Initial Gumbel-Softmax temperature",
    )
    parser.add_argument(
        "--min_temperature",
        type=float,
        default=0.1,
        help="Minimum temperature after annealing",
    )
    parser.add_argument(
        "--temperature_anneal_steps",
        type=int,
        default=5000,
        help="Number of steps to anneal temperature",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/hard_skip",
        help="Output directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Gradient clipping norm",
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
        default=1000,
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
        "--chunk_size",
        type=int,
        default=512,
        help="Chunk size for TTT processing",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=500,
        help="Number of steps to disable cost penalty for warmup",
    )
    parser.add_argument(
        "--beta_ttt",
        type=float,
        default=0.1,
        help="Weight for TTT auxiliary loss",
    )
    parser.add_argument(
        "--listnet_temp",
        type=float,
        default=3.0,
        help="Temperature for ListNet softmax (higher = more uniform weight across samples)",
    )

    return parser.parse_args()


def get_model_name(model_scale: str) -> str:
    mapping = {
        "125m": "gpt2",
        "350m": "gpt2-medium",
        "1b": "gpt2-large",
    }
    return mapping[model_scale]


def get_temperature(step: int, initial: float, minimum: float, anneal_steps: int) -> float:
    """Calculate annealed temperature."""
    if step >= anneal_steps:
        return minimum
    # Exponential annealing
    decay = (minimum / initial) ** (step / anneal_steps)
    return initial * decay


def main():
    args = parse_args()

    print("=" * 60)
    print("PonderTTT Hard Skip Training (Top-k Discriminative Gating)")
    print("=" * 60)
    print(f"Model scale: {args.model_scale}")
    print(f"Target update rate: {args.target_update_rate} (top {args.target_update_rate*100:.0f}%)")
    print(f"ListNet temperature: {args.listnet_temp} (higher = more uniform weight)")
    print(f"Gumbel-Softmax temperature: {args.initial_temperature} -> {args.min_temperature} over {args.temperature_anneal_steps} steps")
    print(f"Warmup steps: {args.warmup_steps}")
    if args.ttt_checkpoint:
        print(f"TTT checkpoint: {args.ttt_checkpoint}")
    else:
        print("WARNING: No TTT checkpoint provided. Training with random TTT weights.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = get_model_name(args.model_scale)
    tokenizer = get_tokenizer(model_name)

    # Load TTT Model (Frozen Backbone)
    print("Loading TTT model...")
    ttt_model, ttt_config = load_ttt_model(
        model_name=model_name,
        fast_weight_type="ttt",
        seed=args.seed,
        load_pretrained=True,
        vocab_size=tokenizer.get_vocab_size(),
    )

    # NOTE: We no longer need to save/restore base_model state because we now do
    # SELECTIVE LOADING - only TTT parts (fast_layer, fast_norm) are loaded from checkpoint.
    # The base_model remains untouched as original HuggingFace weights.

    # Initialize Binary Gating Network
    print("Initializing Binary Gating Network...")
    gating_config = BinaryGatingConfig(
        feature_dim=32,
        hidden_dim=64,
        initial_temperature=args.initial_temperature,
        min_temperature=args.min_temperature,
        scale_when_update=1.0,  # Scale of 1.0 when UPDATE is chosen
    )
    rngs = nnx.Rngs(args.seed + 1)
    gating_net = BinaryGatingNetwork(config=gating_config, rngs=rngs)

    # System container for optimization (Only Trainable Parts)
    class TrainableSystem(nnx.Module):
        def __init__(self, ttt_model, gating_net):
            # Share references to trainable parts
            self.fast_layer = ttt_model.fast_layer
            self.fast_norm = ttt_model.fast_norm
            self.gating_net = gating_net
            # Handle LM Head
            if hasattr(ttt_model, 'lm_head'):
                self.lm_head = ttt_model.lm_head
            else:
                self.lm_head = None

    trainable_system = TrainableSystem(ttt_model, gating_net)

    # Initialize WandB
    if args.wandb_project:
        wandb.init(
            project=args.wandb_project,
            config=vars(args),
            name=f"hard_skip_{args.model_scale}_update{args.target_update_rate}",
        )

    # Optimizer - Create BEFORE loading checkpoint to avoid NNX state key issues
    optimizer = nnx.Optimizer(
        trainable_system,
        optax.chain(
            optax.clip_by_global_norm(args.max_grad_norm),
            optax.adam(args.learning_rate),
        ),
        wrt=nnx.All(nnx.Param),
    )

    # Load pre-trained TTT checkpoint if provided (AFTER optimizer creation)
    # IMPORTANT: We only load TTT-specific parts (fast_layer, fast_norm), NOT base_model.
    # The base_model should remain as original HuggingFace weights to ensure:
    # - Hidden states are computed correctly (proper LayerNorm, etc.)
    # - SKIP path uses embeddings aligned with raw hidden states
    if args.ttt_checkpoint:
        print(f"Loading TTT checkpoint from {args.ttt_checkpoint}...")
        try:
            ckpt = load_checkpoint(args.ttt_checkpoint, target=None)
            if "state" in ckpt and "model" in ckpt["state"]:
                model_state = unwrap_state(ckpt["state"]["model"])
                print(f"Checkpoint from step {ckpt.get('step', 'unknown')}")

                # SELECTIVE LOADING: Only load TTT-specific parts, NOT base_model
                # This preserves original HuggingFace weights for correct SKIP path
                ttt_parts_loaded = []

                # Load fast_layer if present
                if "fast_layer" in model_state:
                    nnx.update(ttt_model.fast_layer, model_state["fast_layer"])
                    ttt_parts_loaded.append("fast_layer")

                # Load fast_norm if present
                if "fast_norm" in model_state:
                    nnx.update(ttt_model.fast_norm, model_state["fast_norm"])
                    ttt_parts_loaded.append("fast_norm")

                # Skip base_model intentionally - keep original HuggingFace weights
                if "base_model" in model_state:
                    print("  ⚠ Skipping base_model from checkpoint (keeping original HuggingFace weights)")

                print(f"✓ Loaded TTT parts: {ttt_parts_loaded}")
                print("✓ base_model preserved as original HuggingFace (for correct SKIP path)")
            else:
                print("Warning: Could not find 'state.model' in TTT checkpoint.")
        except Exception as e:
            print(f"Warning: Could not load TTT checkpoint: {e}")
            print("Proceeding with random initialized TTT layer")

    # Resume from checkpoint if requested
    start_iteration = 0
    if args.resume_from:
        print(f"Resuming from checkpoint: {args.resume_from}")
        target = {
            "state": {"model": nnx.state(trainable_system), "optimizer": nnx.state(optimizer)},
            "step": 0,
            "metadata": {}
        }
        ckpt = load_checkpoint(args.resume_from, target=target)
        nnx.update(trainable_system, ckpt["state"]["model"])
        nnx.update(optimizer, ckpt["state"]["optimizer"])
        start_iteration = ckpt.get("step", 0)
        print(f"Resumed from iteration {start_iteration}")

    # Feature Extractor
    feature_extractor = FeatureExtractor(
        vocab_size=tokenizer.get_vocab_size(),
        pad_token_id=tokenizer.token_to_id("<|pad|>"),
        seq_length_norm=512,
    )

    # Data Iterator
    chunk_size = args.chunk_size
    seq_length = 1024
    # Each iteration processes one batch of batch_size sequences
    # So we need num_iterations * batch_size sequences total
    examples_needed = args.num_iterations * args.batch_size
    data_iter = create_data_iterator(
        tokenizer=tokenizer,
        split="train",
        batch_size=args.batch_size,
        seq_length=seq_length,
        chunk_size=chunk_size,
        max_examples=examples_needed,
        num_workers=args.num_workers,
    )

    # Random key for Gumbel sampling
    rng_key = jax.random.PRNGKey(args.seed + 100)

    # Threshold EMAs for inference
    # threshold_ema: advantage-space threshold (legacy)
    # prob_threshold_ema: probability-space threshold for update_prob soft outputs
    threshold_ema = 0.0
    threshold_ema_initialized = False
    prob_threshold_ema = 0.5
    prob_threshold_initialized = False

    # Training Step
    def train_step(
        trainable_sys: TrainableSystem,
        optimizer: nnx.Optimizer,
        base_model: GPT2Model,
        batch: dict,
        temperature: float,
        target_threshold: float,
        rng_key: jax.Array,
        is_warmup: bool,
        beta_ttt: float = 0.1,
        target_update_rate: float = 0.3,
        tie_word_embeddings: bool = True,
        padding_threshold: float = 0.1,
        listnet_temperature: float = 3.0,
    ):
        """
        Training step with Hard Skip (Top-k discriminative gating).

        Args:
            target_update_rate: Fraction of chunks to UPDATE (top k%).
            padding_threshold: Minimum fraction of valid tokens to consider a chunk as real code.
        """
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        position_ids = batch.get("position_ids")
        labels = input_ids

        # Compute valid token ratio to identify padding-only chunks
        # valid_ratio: [B] - fraction of non-padding tokens per sample
        valid_tokens_per_sample = jnp.sum(attention_mask, axis=-1)  # [B]
        seq_len = attention_mask.shape[-1]
        valid_ratio = valid_tokens_per_sample / seq_len  # [B]

        # Mask for real code chunks (not padding-dominated)
        is_real_code = valid_ratio > padding_threshold  # [B]

        # 1. Base Model Forward (Frozen) - Always needed for features
        hidden_states = jax.lax.stop_gradient(base_model(
            input_ids,
            position_ids=position_ids,
            train=False
        ))

        # Compute logits for features
        if tie_word_embeddings:
            embedding_kernel = jnp.asarray(base_model.wte.embedding)
            logits_base = hidden_states @ embedding_kernel.T
        elif trainable_sys.lm_head:
            logits_base = trainable_sys.lm_head(hidden_states)
        else:
            logits_base = hidden_states

        features = feature_extractor.extract(
            input_ids=input_ids,
            logits=logits_base,
            hidden_states=[hidden_states],
            attention_mask=attention_mask,
            budget_remaining=1.0,
        )
        features = jax.lax.stop_gradient(features)

        # DEBUG: Print features info (will be printed during JIT trace)
        # jax.debug.print("features shape: {}, first sample first 10: {}", features.shape, features[0, :10])

        def loss_fn(sys, rng):
            # 2. Get binary gating decision (Gumbel-Softmax)
            # Returns:
            # - gating_scale: Scaled hard decision (for TTT execution)
            # - decision_probs_hard: One-hot hard decision (for stats)
            # - decision_probs_soft: Soft probabilities (for BCE loss)
            # - decision_hard: Integer hard decision (for stats)
            # - gating_logits: Raw logits [batch, 2] for ranking loss
            gating_scale, decision_probs_hard, decision_probs_soft, decision_hard, gating_logits = sys.gating_net(
                features,
                train=True,
                temperature=temperature,
                rng_key=rng,
                return_logits=True,
            )

            # 3. Compute both SKIP and UPDATE outputs
            # SKIP output: use base hidden states directly
            embedding_kernel = None
            pad_token_id = None

            if tie_word_embeddings:
                embedding_kernel = jnp.asarray(base_model.wte.embedding)
                # Pad token is the last token (added when extending vocab from 50257 to 50258)
                # Its embedding is all zeros, causing logit=0 to dominate softmax
                pad_token_id = embedding_kernel.shape[0] - 1

                logits_skip = hidden_states @ embedding_kernel.T
                # Mask padding token to -inf (it has zero embedding, causing logit=0 to dominate)
                logits_skip = logits_skip.at[..., pad_token_id].set(-1e10)
            elif sys.lm_head:
                logits_skip = sys.lm_head(hidden_states)
            else:
                logits_skip = logits_base

            # UPDATE output: apply TTT
            hidden_states_normed = sys.fast_norm(hidden_states)
            fast_output, fast_stats = sys.fast_layer(
                hidden_states_normed,
                mask=attention_mask,
                position_ids=position_ids,
                train=True,
                gating_scale=jnp.ones_like(gating_scale),  # Full TTT when updating
            )
            adapted_hidden = hidden_states + fast_output

            if tie_word_embeddings and embedding_kernel is not None:
                logits_update = adapted_hidden @ embedding_kernel.T
                # Mask padding token
                logits_update = logits_update.at[..., pad_token_id].set(-1e10)
            elif sys.lm_head:
                logits_update = sys.lm_head(adapted_hidden)
            else:
                raise ValueError("LM Head not found")

            # 5. Compute CE loss for TTT training (always use update path)
            ce_loss_update = cross_entropy_loss(logits_update[:, :-1], labels[:, 1:], attention_mask[:, 1:])

            # === TOP-K DISCRIMINATIVE GATING ===
            #
            # Key insight: Instead of "UPDATE if advantage > 0", we ask
            # "Which k% of chunks benefit MOST from TTT?"
            #
            # This ensures:
            # 1. Fair comparison with random baseline (same budget)
            # 2. Discriminative learning (relative ranking, not absolute threshold)
            # 3. Automatic sparsity control via target_update_rate

            # 1. Compute per-sample advantage (oracle signal)
            ce_per_sample_skip = per_sample_cross_entropy_loss(
                logits_skip[:, :-1], labels[:, 1:], attention_mask[:, 1:]
            )
            ce_per_sample_update = per_sample_cross_entropy_loss(
                logits_update[:, :-1], labels[:, 1:], attention_mask[:, 1:]
            )

            # Debug: track raw loss values
            loss_skip_mean = jnp.mean(ce_per_sample_skip)
            loss_update_mean = jnp.mean(ce_per_sample_update)

            advantage_per_sample = ce_per_sample_skip - ce_per_sample_update  # [B]

            # Clip advantage to prevent outliers from destabilizing training
            # Normal range should be approximately -2 to +2 for mean-normalized CE loss
            advantage_min_raw = jnp.min(advantage_per_sample)
            advantage_max_raw = jnp.max(advantage_per_sample)
            advantage_per_sample = jnp.clip(advantage_per_sample, -10.0, 10.0)

            # 2. Compute Top-k targets (batch-relative) for threshold tracking
            # This is used to update the EMA threshold outside the loop
            _, batch_threshold = compute_topk_targets(
                advantage_per_sample, k=target_update_rate
            )
            batch_threshold = jax.lax.stop_gradient(batch_threshold)

            # 3. Compute Training Targets using STABLE threshold (for logging only)
            training_targets = (advantage_per_sample >= target_threshold).astype(jnp.float32)
            training_targets = jax.lax.stop_gradient(training_targets)

            # 4. Per-sample update probability from gating network
            update_prob_soft = decision_probs_soft[:, 1]  # [B]

            # === LISTNET LOSS (Listwise Ranking) ===
            # Reference: Cao et al., "Learning to Rank: From Pairwise Approach to Listwise Approach" (2007)
            #
            # Key insight: Learn the DISTRIBUTION of scores to match the distribution of advantages.
            # This handles both ranking AND calibration simultaneously:
            # - Ranking: Higher advantage → higher probability in target distribution
            # - Calibration: Softmax normalizes scores, preventing all-UPDATE collapse
            #
            # Loss: Cross-entropy between softmax(advantages) and softmax(scores)
            #       L = -sum(P_adv * log(P_score))
            #       where P_adv = softmax(advantage / temp), P_score = softmax(score / temp)

            # Gating score: logit difference (higher = more likely to UPDATE)
            gating_score = gating_logits[:, 1] - gating_logits[:, 0]  # [B]

            # Temperature for ListNet softmax (controls sharpness)
            # Lower temp = sharper distribution (more focus on top samples)
            # Higher temp = smoother distribution (more uniform learning)
            # NOTE: temp=1.0 caused over-focus on extreme samples (top 2 correct, rest random)
            # Higher temp spreads weight more evenly across all samples
            listnet_temp = listnet_temperature  # Passed as argument

            # Target distribution: softmax over advantages
            # High advantage samples get high probability mass
            adv_target_dist = jax.nn.softmax(advantage_per_sample / listnet_temp)  # [B]

            # Predicted distribution: softmax over gating scores
            score_pred_dist = jax.nn.softmax(gating_score / listnet_temp)  # [B]

            # Cross-entropy loss: -sum(target * log(pred))
            eps = 1e-7
            listnet_loss = -jnp.sum(adv_target_dist * jnp.log(score_pred_dist + eps))

            # === CALIBRATION REGULARIZATION ===
            # ListNet is shift-invariant (adding constant to all scores doesn't change softmax).
            # We need to anchor the scores to ensure ~50% update rate.
            #
            # Method: Penalize deviation of mean UPDATE probability from target rate.
            # This pushes scores such that top-k have prob > 0.5, bottom-k have prob < 0.5.
            actual_update_rate = jnp.mean(update_prob_soft)
            calibration_loss = jnp.square(actual_update_rate - target_update_rate)

            # Weight for calibration loss (balance between ranking and calibration)
            lambda_calibration = 1.0

            # For pairwise ranking accuracy (logging only)
            diff_adv = advantage_per_sample[:, None] - advantage_per_sample[None, :]
            diff_score = gating_score[:, None] - gating_score[None, :]
            ranking_margin = 0.1
            valid_pairs = diff_adv > ranking_margin
            num_valid_pairs = jnp.maximum(jnp.sum(valid_pairs.astype(jnp.float32)), 1.0)

            # Also compute BCE for comparison/logging (but don't use for training)
            update_prob_clamped = jnp.clip(update_prob_soft, eps, 1.0 - eps)
            bce_loss = -(
                training_targets * jnp.log(update_prob_clamped) +
                (1.0 - training_targets) * jnp.log(1.0 - update_prob_clamped)
            )
            gating_bce_loss = jnp.mean(bce_loss)

            # Combined loss: ListNet (ranking) + Calibration (absolute scale)
            ranking_loss = listnet_loss  # For logging compatibility
            gating_quality_loss = listnet_loss + lambda_calibration * calibration_loss

            # For logging
            advantage_per_sample = jax.lax.stop_gradient(advantage_per_sample)
            advantage = jnp.mean(advantage_per_sample)

            # CE loss for TTT parameters: ALWAYS use ce_loss_update
            # The gating decision affects inference cost, not training.
            # TTT should always learn to improve predictions.
            ce_loss = ce_loss_update

            # TTT auxiliary loss
            if fast_stats is not None and "ttt_loss_step_1" in fast_stats:
                l_ttt = jnp.mean(fast_stats["ttt_loss_step_1"])
                if l_ttt is None:
                    l_ttt = jnp.array(0.0)
            else:
                l_ttt = jnp.array(0.0)

            # For logging: compute stats on real code chunks
            update_prob_flat = update_prob_soft  # Use soft for logging
            num_real_code = jnp.maximum(jnp.sum(is_real_code.astype(jnp.float32)), 1.0)
            avg_update_prob = jnp.mean(update_prob_flat)

            # Apply warmup (disable gating loss during warmup to let TTT stabilize first)
            gating_quality_loss = jnp.where(is_warmup, 0.0, gating_quality_loss)

            # Total loss: CE (for TTT) + TTT aux + Gating loss
            total_loss = ce_loss + beta_ttt * l_ttt + gating_quality_loss

            # Compute actual cost for logging
            # Hard decision cost: SKIP=1.0, UPDATE=3.0
            hard_cost = jnp.mean(jnp.where(decision_hard == 1, 3.0, 1.0))

            # Skip rate on real code only
            skip_on_real = jnp.sum((decision_hard == 0).astype(jnp.float32) * is_real_code.astype(jnp.float32)) / num_real_code

            # Compute stats for logging
            advantage_std = jnp.std(advantage_per_sample)

            # Top-k match rate: how well does gating match the STABLE target?
            topk_match_rate = jnp.mean((decision_hard == training_targets.astype(jnp.int32)).astype(jnp.float32))

            # Probability-space threshold (UPDATE prob percentile) matching target_update_rate
            prob_threshold = jnp.percentile(update_prob_soft, 100.0 * (1.0 - target_update_rate))

            # Pairwise ranking accuracy: what fraction of valid pairs are correctly ordered?
            # correct_pairs[i,j] = 1 if (adv_i > adv_j) implies (score_i > score_j)
            correct_pairs = (diff_score > 0).astype(jnp.float32) * valid_pairs.astype(jnp.float32)
            ranking_accuracy = jnp.sum(correct_pairs) / num_valid_pairs

            # Track valid pairs ratio (for debugging)
            batch_size = advantage_per_sample.shape[0]
            total_pairs = batch_size * (batch_size - 1)  # Excluding diagonal
            valid_pairs_ratio = num_valid_pairs / jnp.maximum(total_pairs, 1.0)

            return total_loss, (
                ce_loss,
                l_ttt,
                gating_quality_loss,
                avg_update_prob,
                hard_cost,
                decision_hard,
                skip_on_real,
                num_real_code,
                advantage,
                gating_bce_loss,
                batch_threshold,
                topk_match_rate,
                advantage_std,
                prob_threshold,
                advantage_min_raw,
                advantage_max_raw,
                loss_skip_mean,
                loss_update_mean,
                ranking_loss,
                ranking_accuracy,
            )

        grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
        (loss, aux), grads = grad_fn(trainable_sys, rng_key)
        (
            ce_loss,
            l_ttt,
            gating_loss,
            avg_update_prob,
            hard_cost,
            decision_hard,
            skip_on_real,
            num_real_code,
            advantage,
            bce_loss,
            batch_threshold,
            topk_match,
            adv_std,
            prob_threshold,
            adv_min_raw,
            adv_max_raw,
            loss_skip_mean,
            loss_update_mean,
            ranking_loss,
            ranking_accuracy,
        ) = aux

        optimizer.update(trainable_sys, grads)

        # Compute skip rate from hard decisions (overall)
        skip_rate = jnp.mean(decision_hard == 0)

        return (
            loss,
            ce_loss,
            l_ttt,
            gating_loss,
            avg_update_prob,
            hard_cost,
            skip_rate,
            skip_on_real,
            num_real_code,
            advantage,
            bce_loss,
            batch_threshold,
            topk_match,
            adv_std,
            prob_threshold,
            adv_min_raw,
            adv_max_raw,
            loss_skip_mean,
            loss_update_mean,
            ranking_loss,
            ranking_accuracy,
        )

    # JIT compile training step
    train_step_jit = nnx.jit(
        train_step,
        static_argnames=("beta_ttt", "target_update_rate", "tie_word_embeddings", "padding_threshold", "listnet_temperature"),
    )

    # History tracking
    history = []

    print("Starting training...")
    if start_iteration > 0:
        print(f"Skipping first {start_iteration} batches to resume training...")

    iter_count = start_iteration
    temperature = args.initial_temperature
    for i, sequence_batch in enumerate(data_iter):
        if i < start_iteration:
            continue

        if iter_count >= args.num_iterations:
            break

        # Process chunks in sequence
        chunks = sequence_batch["chunks"]
        masks = sequence_batch["chunk_attention_mask"]
        num_chunks = chunks.shape[1]

        # Temperature annealing
        temperature = get_temperature(
            iter_count,
            args.initial_temperature,
            args.min_temperature,
            args.temperature_anneal_steps
        )

        # Check if in warmup period
        is_warmup = iter_count < args.warmup_steps

        # Aggregate stats
        total_loss = 0.0
        total_ce_loss = 0.0
        total_l_ttt = 0.0
        total_gating_loss = 0.0
        total_update_prob = 0.0
        total_hard_cost = 0.0
        total_skip_rate = 0.0
        total_skip_on_real = 0.0
        total_real_code_chunks = 0.0
        total_advantage = 0.0
        total_bce_loss = 0.0
        total_threshold = 0.0
        total_topk_match = 0.0
        total_adv_std = 0.0
        total_prob_threshold = 0.0
        min_adv_raw = float('inf')
        max_adv_raw = float('-inf')
        total_loss_skip = 0.0
        total_loss_update = 0.0
        total_ranking_loss = 0.0
        total_ranking_acc = 0.0
        valid_chunks = 0

        feature_extractor.reset_history()

        for c_idx in range(num_chunks):
            chunk_batch = {
                "input_ids": chunks[:, c_idx],
                "attention_mask": masks[:, c_idx],
                "position_ids": jnp.arange(
                    c_idx * chunk_size,
                    (c_idx + 1) * chunk_size,
                    dtype=jnp.int32
                )[None, :].repeat(chunks.shape[0], axis=0)
            }

            # Check for valid tokens
            num_valid_tokens = jnp.sum(chunk_batch["attention_mask"][:, 1:])
            if num_valid_tokens < 16:
                continue

            # Get new random key
            rng_key, subkey = jax.random.split(rng_key)

            (
                loss,
                ce,
                l_ttt,
                gate_loss,
                update_prob,
                hard_cost,
                skip_rate,
                skip_on_real,
                num_real,
                adv,
                bce_loss,
                batch_thresh,
                topk_match,
                adv_std,
                prob_thresh,
                adv_min,
                adv_max,
                loss_skip,
                loss_update,
                rank_loss,
                rank_acc,
            ) = train_step_jit(
                trainable_system,
                optimizer,
                cast(GPT2Model, ttt_model.base_model),  # train_hard_skip is GPT2-specific
                chunk_batch,
                temperature,
                threshold_ema,
                subkey,
                is_warmup,
                beta_ttt=args.beta_ttt,
                target_update_rate=args.target_update_rate,
                tie_word_embeddings=ttt_model.tie_word_embeddings,
                padding_threshold=0.1,
                listnet_temperature=args.listnet_temp,
            )

            # Stability check
            if not jnp.isfinite(loss) or float(ce) > 20.0:
                print(f"Warning: Skipping unstable batch (loss={loss:.4f}, ce={ce:.4f})")
                continue

            # Update threshold EMA
            if not threshold_ema_initialized:
                threshold_ema = float(batch_thresh)
                threshold_ema_initialized = True
            else:
                threshold_ema = args.threshold_ema_decay * threshold_ema + (1 - args.threshold_ema_decay) * float(batch_thresh)
            if not prob_threshold_initialized:
                prob_threshold_ema = float(prob_thresh)
                prob_threshold_initialized = True
            else:
                prob_threshold_ema = args.threshold_ema_decay * prob_threshold_ema + (1 - args.threshold_ema_decay) * float(prob_thresh)

            total_loss += float(loss)
            total_ce_loss += float(ce)
            total_l_ttt += float(l_ttt)
            total_gating_loss += float(gate_loss)
            total_update_prob += float(update_prob)
            total_hard_cost += float(hard_cost)
            total_skip_rate += float(skip_rate)
            total_skip_on_real += float(skip_on_real) * float(num_real)
            total_real_code_chunks += float(num_real)
            total_advantage += float(adv)
            total_bce_loss += float(bce_loss)
            total_threshold += float(batch_thresh)
            total_topk_match += float(topk_match)
            total_adv_std += float(adv_std)
            total_prob_threshold += float(prob_thresh)
            min_adv_raw = min(min_adv_raw, float(adv_min))
            max_adv_raw = max(max_adv_raw, float(adv_max))
            total_loss_skip += float(loss_skip)
            total_loss_update += float(loss_update)
            total_ranking_loss += float(rank_loss)
            total_ranking_acc += float(rank_acc)
            valid_chunks += 1

            feature_extractor.update_history(float(ce), float(hard_cost))

        if valid_chunks == 0:
            iter_count += 1
            continue

        # Average stats
        avg_loss = total_loss / valid_chunks
        avg_ce_loss = total_ce_loss / valid_chunks
        avg_l_ttt = total_l_ttt / valid_chunks
        avg_gating_loss = total_gating_loss / valid_chunks
        avg_update_prob = total_update_prob / valid_chunks
        avg_hard_cost = total_hard_cost / valid_chunks
        avg_skip_rate = total_skip_rate / valid_chunks
        avg_skip_on_real = total_skip_on_real / max(total_real_code_chunks, 1.0)
        avg_advantage = total_advantage / valid_chunks
        avg_bce_loss = total_bce_loss / valid_chunks
        avg_threshold = total_threshold / valid_chunks
        avg_topk_match = total_topk_match / valid_chunks
        avg_adv_std = total_adv_std / valid_chunks
        avg_prob_threshold = total_prob_threshold / valid_chunks
        avg_loss_skip = total_loss_skip / valid_chunks
        avg_loss_update = total_loss_update / valid_chunks
        avg_ranking_loss = total_ranking_loss / valid_chunks
        avg_ranking_acc = total_ranking_acc / valid_chunks
        perplexity = math.exp(min(avg_ce_loss, 10.0))  # Cap to avoid overflow

        warmup_status = " [WARMUP]" if is_warmup else ""
        print(
            f"Iter {iter_count+1}{warmup_status}: Loss={avg_loss:.4f}, "
            f"CE={avg_ce_loss:.4f}, PPL={perplexity:.2f}, "
            f"SkipRate={avg_skip_rate:.2%}, TopkMatch={avg_topk_match:.2%}, "
            f"Cost={avg_hard_cost:.2f}x, "
            f"LossSkip={avg_loss_skip:.2f}, LossUpdate={avg_loss_update:.2f}, "
            f"Adv=[{min_adv_raw:.2f},{max_adv_raw:.2f}], "
            f"RankLoss={avg_ranking_loss:.4f}, RankAcc={avg_ranking_acc:.2%}, "
            f"Temp={temperature:.3f}"
        )

        # WandB Logging
        if args.wandb_project:
            wandb.log({
                "loss_total": avg_loss,
                "loss_ce": avg_ce_loss,
                "perplexity": perplexity,
                "l_ttt": avg_l_ttt,
                "gating_loss": avg_gating_loss,
                "bce_loss": avg_bce_loss,
                "update_prob": avg_update_prob,
                "skip_rate": avg_skip_rate,
                "topk_match_rate": avg_topk_match,
                "skip_rate_real_code": avg_skip_on_real,
                "hard_cost": avg_hard_cost,
                "temperature": temperature,
                "is_warmup": float(is_warmup),
                "iteration": iter_count + 1,
                "real_code_chunks": total_real_code_chunks,
                "advantage": avg_advantage,
                "advantage_std": avg_adv_std,
                "advantage_min_raw": min_adv_raw,
                "advantage_max_raw": max_adv_raw,
                "loss_skip": avg_loss_skip,
                "loss_update": avg_loss_update,
                "ranking_loss": avg_ranking_loss,
                "ranking_accuracy": avg_ranking_acc,
                "batch_threshold": avg_threshold,
                "threshold_ema": threshold_ema,
                "prob_threshold": avg_prob_threshold,
                "prob_threshold_ema": prob_threshold_ema,
            })

        history.append({
            "iteration": iter_count,
            "loss_total": avg_loss,
            "loss_ce": avg_ce_loss,
            "perplexity": perplexity,
            "l_ttt": avg_l_ttt,
            "update_prob": avg_update_prob,
            "skip_rate": avg_skip_rate,
            "topk_match_rate": avg_topk_match,
            "skip_rate_real_code": avg_skip_on_real,
            "hard_cost": avg_hard_cost,
            "temperature": temperature,
            "is_warmup": is_warmup,
            "threshold_ema": threshold_ema,
            "prob_threshold": avg_prob_threshold,
            "prob_threshold_ema": prob_threshold_ema,
            "advantage_min_raw": min_adv_raw,
            "advantage_max_raw": max_adv_raw,
            "loss_skip": avg_loss_skip,
            "loss_update": avg_loss_update,
            "ranking_loss": avg_ranking_loss,
            "ranking_accuracy": avg_ranking_acc,
        })

        # Periodic Checkpoint
        if (iter_count + 1) % args.save_every == 0 and (iter_count + 1) < args.num_iterations:
            print(f"Saving checkpoint at iter {iter_count + 1}...")
            save_checkpoint(
                checkpoint_dir=output_dir,
                step=iter_count + 1,
                state={"model": nnx.state(trainable_system), "optimizer": nnx.state(optimizer)},
                metadata={
                    "model_scale": args.model_scale,
                    "target_update_rate": args.target_update_rate,
                    "temperature": temperature,
                    "threshold_ema": threshold_ema,
                    "prob_threshold_ema": prob_threshold_ema,
                }
            )

        iter_count += 1

    # Save results
    with open(output_dir / "history_hard_skip.json", "w") as f:
        json.dump(history, f, indent=2)

    # Save Final Checkpoint
    print("Saving final checkpoint...")
    save_checkpoint(
        checkpoint_dir=output_dir,
        step=iter_count,
        state={"model": nnx.state(trainable_system), "optimizer": nnx.state(optimizer)},
        metadata={
            "model_scale": args.model_scale,
            "target_update_rate": args.target_update_rate,
            "final_temperature": temperature,
            "threshold_ema": threshold_ema,
            "prob_threshold_ema": prob_threshold_ema,
        }
    )
    finalize_checkpointing()
    print(f"Checkpoint saved to {output_dir}")

    # Final summary
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    if history:
        final_stats = history[-1]
        print(f"Target Update Rate: {args.target_update_rate:.0%}")
        print(f"Final Skip Rate: {final_stats['skip_rate']:.2%}")
        print(f"Final Top-k Match Rate: {final_stats['topk_match_rate']:.2%}")
        print(f"Final Cost: {final_stats['hard_cost']:.2f}x")
        print(f"Final Perplexity: {final_stats['perplexity']:.2f}")
        print(f"Threshold EMA (for inference): {threshold_ema:.4f}")
        print(f"Prob Threshold EMA (for inference): {prob_threshold_ema:.4f}")

    if args.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    main()
