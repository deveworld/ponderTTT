"""
Train binary gating network with Top-k Gating for adaptive TTT.

Key insight: Instead of "UPDATE if advantage > 0", we ask
"Which k% of chunks benefit MOST from TTT?"

This ensures:
1. Fair comparison with random baseline (same budget)
2. Discriminative learning (relative ranking, not absolute threshold)
3. Automatic sparsity control via k parameter

Usage:
    python -m ponderttt.experiments.train_topk_gating --model_scale 125m --target_update_rate 0.3
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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Top-k Gating (Discriminative Selection)")

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
        help="Target update rate (0.3 = top 30%% get UPDATE)",
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
        default="outputs/topk_gating",
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
        help="Number of steps to disable gating loss for warmup",
    )
    parser.add_argument(
        "--beta_ttt",
        type=float,
        default=0.1,
        help="Weight for TTT auxiliary loss",
    )
    parser.add_argument(
        "--threshold_ema_decay",
        type=float,
        default=0.99,
        help="EMA decay for threshold tracking (for inference)",
    )
    parser.add_argument(
        "--use_soft_topk",
        action="store_true",
        help="Use soft top-k (differentiable) instead of hard top-k",
    )
    parser.add_argument(
        "--ranking_margin",
        type=float,
        default=0.0,
        help="Margin for ranking loss (0 = pure BCE on top-k targets)",
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


def soft_topk_targets(advantages: jax.Array, k: float, temperature: float = 1.0) -> jax.Array:
    """
    Compute soft top-k targets using a differentiable approximation.

    Instead of hard 0/1 targets, produces soft targets based on
    relative ranking within the batch.

    Args:
        advantages: Per-sample advantage values [B]
        k: Target update rate
        temperature: Softness of the ranking (lower = harder)

    Returns:
        soft_targets: Soft targets [B] in range [0, 1]
    """
    # Normalize advantages to [0, 1] range using sigmoid
    # This preserves relative ordering
    adv_mean = jnp.mean(advantages)
    adv_std = jnp.maximum(jnp.std(advantages), 1e-6)
    adv_normalized = (advantages - adv_mean) / adv_std

    # Shift to target k% having high probability
    # If k=0.3, we want top 30% to have target > 0.5
    # This means we need to shift by inverse_sigmoid(1-k)
    # inverse_sigmoid(0.7) ≈ 0.85
    shift = jnp.log(k / (1.0 - k + 1e-6))  # logit of k

    soft_targets = jax.nn.sigmoid((adv_normalized - shift) / temperature)

    return soft_targets


def main():
    args = parse_args()

    print("=" * 60)
    print("PonderTTT Top-k Gating Training (Discriminative Selection)")
    print("=" * 60)
    print(f"Model scale: {args.model_scale}")
    print(f"Target update rate: {args.target_update_rate} (top {args.target_update_rate*100:.0f}%)")
    print(f"Temperature: {args.initial_temperature} -> {args.min_temperature} over {args.temperature_anneal_steps} steps")
    print(f"Warmup steps: {args.warmup_steps}")
    print(f"Soft top-k: {args.use_soft_topk}")

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

    # Initialize Binary Gating Network
    print("Initializing Binary Gating Network...")
    gating_config = BinaryGatingConfig(
        feature_dim=32,
        hidden_dim=64,
        initial_temperature=args.initial_temperature,
        min_temperature=args.min_temperature,
        scale_when_update=1.0,
    )
    rngs = nnx.Rngs(args.seed + 1)
    gating_net = BinaryGatingNetwork(config=gating_config, rngs=rngs)

    # System container for optimization
    class TrainableSystem(nnx.Module):
        def __init__(self, ttt_model, gating_net):
            self.fast_layer = ttt_model.fast_layer
            self.fast_norm = ttt_model.fast_norm
            self.gating_net = gating_net
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
            name=f"topk_{args.model_scale}_k{args.target_update_rate}",
        )

    # Optimizer
    optimizer = nnx.Optimizer(
        trainable_system,
        optax.chain(
            optax.clip_by_global_norm(args.max_grad_norm),
            optax.adam(args.learning_rate),
        ),
        wrt=nnx.All(nnx.Param),
    )

    # Resume from checkpoint if requested
    start_iteration = 0
    threshold_ema = 0.0  # Will be initialized from first batch
    threshold_ema_initialized = False

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
        threshold_ema = ckpt.get("metadata", {}).get("threshold_ema", 0.0)
        if threshold_ema != 0.0:
            threshold_ema_initialized = True
        print(f"Resumed from iteration {start_iteration}, threshold_ema={threshold_ema:.4f}")

    # Feature Extractor
    feature_extractor = FeatureExtractor(
        vocab_size=tokenizer.get_vocab_size(),
        pad_token_id=tokenizer.token_to_id("<|pad|>"),
        seq_length_norm=512,
    )

    # Data Iterator
    chunk_size = args.chunk_size
    seq_length = 1024
    chunks_per_sequence = max(1, seq_length // chunk_size)
    total_chunks = args.num_iterations * chunks_per_sequence
    examples_needed = math.ceil(total_chunks / chunks_per_sequence)
    max_examples = examples_needed * args.batch_size
    data_iter = create_data_iterator(
        tokenizer=tokenizer,
        split="train",
        batch_size=args.batch_size,
        seq_length=seq_length,
        chunk_size=chunk_size,
        max_examples=max_examples,
        num_workers=args.num_workers,
    )

    # Random key for Gumbel sampling
    rng_key = jax.random.PRNGKey(args.seed + 100)

    # Training Step
    def train_step(
        trainable_sys: TrainableSystem,
        optimizer: nnx.Optimizer,
        base_model: GPT2Model,
        batch: dict,
        temperature: float,
        rng_key: jax.Array,
        is_warmup: bool,
        target_update_rate: float = 0.3,
        beta_ttt: float = 0.1,
        use_soft_topk: bool = False,
        ranking_margin: float = 0.0,
        tie_word_embeddings: bool = True,
    ):
        """
        Training step with Top-k Gating.

        Key difference from advantage-supervised:
        - Target is based on relative ranking (top k%) not absolute advantage
        - Forces same budget as random baseline for fair comparison
        - Learns discriminative selection rather than binary classification
        """
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        position_ids = batch.get("position_ids")
        labels = input_ids

        # 1. Base Model Forward (Frozen)
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

        def loss_fn(sys, rng):
            # 2. Get gating logits and decisions
            gating_scale, decision_probs_soft, decision_hard = sys.gating_net(
                features,
                train=True,
                temperature=temperature,
                rng_key=rng,
            )

            # Get raw logits for ranking loss (before softmax)
            # We need to recompute this from the network
            B = features.shape[0]
            x = features.astype(jnp.float32).reshape(B, -1)
            x = sys.gating_net.input_norm(x)
            x = sys.gating_net.fc1(x)
            x = jax.nn.relu(x)
            x = sys.gating_net.fc2(x)
            x = jax.nn.relu(x)
            gating_logits = sys.gating_net.head(x)  # [B, 2]
            update_logit = gating_logits[:, 1] - gating_logits[:, 0]  # [B] logit for UPDATE

            # 3. Compute SKIP and UPDATE outputs
            embedding_kernel = None
            if tie_word_embeddings:
                embedding_kernel = jnp.asarray(base_model.wte.embedding)
                logits_skip = hidden_states @ embedding_kernel.T
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
                gating_scale=jnp.ones_like(gating_scale),
            )
            adapted_hidden = hidden_states + fast_output

            if tie_word_embeddings and embedding_kernel is not None:
                logits_update = adapted_hidden @ embedding_kernel.T
            elif sys.lm_head:
                logits_update = sys.lm_head(adapted_hidden)
            else:
                raise ValueError("LM Head not found")

            # 4. Compute per-sample losses and advantage
            ce_per_sample_skip = per_sample_cross_entropy_loss(
                logits_skip[:, :-1], labels[:, 1:], attention_mask[:, 1:]
            )
            ce_per_sample_update = per_sample_cross_entropy_loss(
                logits_update[:, :-1], labels[:, 1:], attention_mask[:, 1:]
            )
            advantage_per_sample = ce_per_sample_skip - ce_per_sample_update  # [B]

            # === TOP-K GATING ===
            # Key insight: Instead of "advantage > 0 → UPDATE",
            # we ask "Is this sample in the top k% of advantages?"

            # 5. Compute top-k targets
            if use_soft_topk:
                # Soft targets based on relative ranking
                topk_targets = soft_topk_targets(
                    advantage_per_sample,
                    k=target_update_rate,
                    temperature=1.0
                )
                batch_threshold = jnp.array(0.0)  # Not used for soft
            else:
                # Hard targets: top k% get 1, rest get 0
                topk_targets, batch_threshold = compute_topk_targets(
                    advantage_per_sample,
                    k=target_update_rate
                )

            topk_targets = jax.lax.stop_gradient(topk_targets)
            batch_threshold = jax.lax.stop_gradient(batch_threshold)

            # 6. Gating loss: BCE on top-k targets
            update_prob_soft = decision_probs_soft[:, 1]  # [B]
            eps = 1e-6
            update_prob_clamped = jnp.clip(update_prob_soft, eps, 1.0 - eps)
            target_clamped = jnp.clip(topk_targets, eps, 1.0 - eps)

            bce_loss = -(
                target_clamped * jnp.log(update_prob_clamped) +
                (1 - target_clamped) * jnp.log(1 - update_prob_clamped)
            )
            gating_bce_loss = jnp.mean(bce_loss)

            # 7. Optional: Ranking margin loss
            # Encourages logit_i > logit_j + margin when adv_i > adv_j
            ranking_loss = jnp.array(0.0)
            if ranking_margin > 0:
                # Pairwise ranking loss (simplified: compare to batch mean)
                mean_logit = jnp.mean(update_logit)
                should_be_above = topk_targets  # Samples that should have high logits
                # Loss: max(0, margin - (logit - mean_logit)) for top-k samples
                # Loss: max(0, margin - (mean_logit - logit)) for non-top-k samples
                margin_violation_topk = jnp.maximum(0, ranking_margin - (update_logit - mean_logit))
                margin_violation_other = jnp.maximum(0, ranking_margin - (mean_logit - update_logit))
                ranking_loss = jnp.mean(
                    should_be_above * margin_violation_topk +
                    (1 - should_be_above) * margin_violation_other
                )

            # Combined gating loss (no skip rate regularizer needed - k controls rate)
            gating_quality_loss = gating_bce_loss + ranking_loss

            # 8. CE loss for TTT (always use update path for training TTT)
            ce_loss_update = cross_entropy_loss(logits_update[:, :-1], labels[:, 1:], attention_mask[:, 1:])
            ce_loss = ce_loss_update

            # TTT auxiliary loss
            if fast_stats is not None and "ttt_loss_step_1" in fast_stats:
                l_ttt = jnp.mean(fast_stats["ttt_loss_step_1"])
                if l_ttt is None:
                    l_ttt = jnp.array(0.0)
            else:
                l_ttt = jnp.array(0.0)

            # Apply warmup
            gating_quality_loss = jnp.where(is_warmup, 0.0, gating_quality_loss)

            # Total loss
            total_loss = ce_loss + beta_ttt * l_ttt + gating_quality_loss

            # Logging stats
            advantage = jnp.mean(advantage_per_sample)
            advantage_std = jnp.std(advantage_per_sample)
            hard_cost = jnp.mean(jnp.where(decision_hard == 1, 3.0, 1.0))
            actual_update_rate = jnp.mean(decision_hard.astype(jnp.float32))

            # How well does the learned gating match top-k oracle?
            topk_match_rate = jnp.mean((decision_hard == topk_targets.astype(jnp.int32)).astype(jnp.float32))

            return total_loss, (
                ce_loss, l_ttt, gating_quality_loss,
                jnp.mean(update_prob_soft), hard_cost, decision_hard,
                advantage, advantage_std,
                batch_threshold, topk_match_rate, actual_update_rate,
                gating_bce_loss, ranking_loss
            )

        grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
        (loss, aux), grads = grad_fn(trainable_sys, rng_key)
        (ce_loss, l_ttt, gating_loss, avg_update_prob, hard_cost, decision_hard,
         advantage, advantage_std, batch_threshold, topk_match_rate, actual_update_rate,
         bce_loss, ranking_loss) = aux

        optimizer.update(trainable_sys, grads)

        skip_rate = 1.0 - actual_update_rate

        return (loss, ce_loss, l_ttt, gating_loss, avg_update_prob, hard_cost,
                skip_rate, advantage, advantage_std, batch_threshold,
                topk_match_rate, bce_loss, ranking_loss)

    # JIT compile training step
    train_step_jit = nnx.jit(
        train_step,
        static_argnames=("target_update_rate", "beta_ttt", "use_soft_topk", "ranking_margin", "tie_word_embeddings"),
    )

    # History tracking
    history = []

    # Advantage distribution tracking (for analysis)
    advantage_history = []

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

        is_warmup = iter_count < args.warmup_steps

        # Aggregate stats
        total_loss = 0.0
        total_ce_loss = 0.0
        total_l_ttt = 0.0
        total_gating_loss = 0.0
        total_update_prob = 0.0
        total_hard_cost = 0.0
        total_skip_rate = 0.0
        total_advantage = 0.0
        total_advantage_std = 0.0
        total_threshold = 0.0
        total_topk_match = 0.0
        total_bce_loss = 0.0
        total_ranking_loss = 0.0
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

            num_valid_tokens = jnp.sum(chunk_batch["attention_mask"][:, 1:])
            if num_valid_tokens < 16:
                continue

            rng_key, subkey = jax.random.split(rng_key)

            results = train_step_jit(
                trainable_system,
                optimizer,
                cast(GPT2Model, ttt_model.base_model),
                chunk_batch,
                temperature,
                subkey,
                is_warmup,
                target_update_rate=args.target_update_rate,
                beta_ttt=args.beta_ttt,
                use_soft_topk=args.use_soft_topk,
                ranking_margin=args.ranking_margin,
                tie_word_embeddings=ttt_model.tie_word_embeddings,
            )

            (loss, ce, l_ttt, gating_loss, update_prob, hard_cost,
             skip_rate, adv, adv_std, batch_threshold, topk_match,
             bce_loss, ranking_loss) = results

            # Stability check
            if not jnp.isfinite(loss) or float(ce) > 20.0:
                print(f"Warning: Skipping unstable batch (loss={loss:.4f}, ce={ce:.4f})")
                continue

            # Update threshold EMA
            if not threshold_ema_initialized:
                threshold_ema = float(batch_threshold)
                threshold_ema_initialized = True
            else:
                threshold_ema = args.threshold_ema_decay * threshold_ema + (1 - args.threshold_ema_decay) * float(batch_threshold)

            total_loss += float(loss)
            total_ce_loss += float(ce)
            total_l_ttt += float(l_ttt)
            total_gating_loss += float(gating_loss)
            total_update_prob += float(update_prob)
            total_hard_cost += float(hard_cost)
            total_skip_rate += float(skip_rate)
            total_advantage += float(adv)
            total_advantage_std += float(adv_std)
            total_threshold += float(batch_threshold)
            total_topk_match += float(topk_match)
            total_bce_loss += float(bce_loss)
            total_ranking_loss += float(ranking_loss)
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
        avg_advantage = total_advantage / valid_chunks
        avg_advantage_std = total_advantage_std / valid_chunks
        avg_threshold = total_threshold / valid_chunks
        avg_topk_match = total_topk_match / valid_chunks
        avg_bce_loss = total_bce_loss / valid_chunks
        avg_ranking_loss = total_ranking_loss / valid_chunks
        perplexity = math.exp(min(avg_ce_loss, 10.0))

        warmup_status = " [WARMUP]" if is_warmup else ""
        print(
            f"Iter {iter_count+1}{warmup_status}: Loss={avg_loss:.4f}, "
            f"CE={avg_ce_loss:.4f}, PPL={perplexity:.2f}, "
            f"SkipRate={avg_skip_rate:.2%}, TopkMatch={avg_topk_match:.2%}, "
            f"Cost={avg_hard_cost:.2f}x, Adv={avg_advantage:.4f}±{avg_advantage_std:.4f}, "
            f"Thresh={avg_threshold:.4f} (EMA:{threshold_ema:.4f}), Temp={temperature:.3f}"
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
                "ranking_loss": avg_ranking_loss,
                "update_prob": avg_update_prob,
                "skip_rate": avg_skip_rate,
                "topk_match_rate": avg_topk_match,
                "hard_cost": avg_hard_cost,
                "temperature": temperature,
                "is_warmup": float(is_warmup),
                "iteration": iter_count + 1,
                "advantage_mean": avg_advantage,
                "advantage_std": avg_advantage_std,
                "batch_threshold": avg_threshold,
                "threshold_ema": threshold_ema,
            })

        history.append({
            "iteration": iter_count,
            "loss_total": avg_loss,
            "loss_ce": avg_ce_loss,
            "perplexity": perplexity,
            "skip_rate": avg_skip_rate,
            "topk_match_rate": avg_topk_match,
            "hard_cost": avg_hard_cost,
            "temperature": temperature,
            "is_warmup": is_warmup,
            "advantage_mean": avg_advantage,
            "advantage_std": avg_advantage_std,
            "threshold_ema": threshold_ema,
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
                }
            )

        iter_count += 1

    # Save results
    with open(output_dir / "history_topk_gating.json", "w") as f:
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

    if args.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    main()
