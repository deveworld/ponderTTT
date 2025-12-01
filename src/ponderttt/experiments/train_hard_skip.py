"""
Train binary gating network for adaptive TTT with Hard Skip.

Uses Gumbel-Softmax for differentiable binary (SKIP/UPDATE) decisions.
- SKIP: No TTT computation (cost = 1.0x)
- UPDATE: Full TTT computation (cost = 3.0x)

Usage:
    python -m ponderttt.experiments.train_hard_skip --model_scale 125m --target_skip_rate 0.5
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
from ..utils import FeatureExtractor, cross_entropy_loss
from ..utils.checkpointing import save_checkpoint, load_checkpoint, finalize_checkpointing

import wandb


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
        "--cost_weight",
        type=float,
        default=0.1,
        help="Weight for computational cost penalty",
    )
    parser.add_argument(
        "--target_skip_rate",
        type=float,
        default=0.5,
        help="Target skip rate (0.5 = skip 50%% of chunks, cost ~2.0x)",
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
    print("PonderTTT Hard Skip Training (Binary Gating with Gumbel-Softmax)")
    print("=" * 60)
    print(f"Model scale: {args.model_scale}")
    print(f"Target skip rate: {args.target_skip_rate}")
    print(f"Cost weight (gamma): {args.cost_weight}")
    print(f"Temperature: {args.initial_temperature} -> {args.min_temperature} over {args.temperature_anneal_steps} steps")
    print(f"Warmup steps: {args.warmup_steps}")

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
            name=f"hard_skip_{args.model_scale}_skip{args.target_skip_rate}",
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
        beta_ttt: float = 0.1,
        cost_weight: float = 0.1,
        tie_word_embeddings: bool = True,
        padding_threshold: float = 0.1,
    ):
        """
        Training step with Hard Skip (binary gating).

        Args:
            padding_threshold: Minimum fraction of valid tokens to consider a chunk as real code.
                              Chunks with fewer valid tokens are excluded from cost penalty.
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

        def loss_fn(sys, rng):
            # 2. Get binary gating decision (Gumbel-Softmax)
            gating_scale, decision_probs, decision_hard = sys.gating_net(
                features,
                train=True,
                temperature=temperature,
                rng_key=rng,
            )

            # 3. Compute both SKIP and UPDATE outputs
            # SKIP output: use base hidden states directly
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
                gating_scale=jnp.ones_like(gating_scale),  # Full TTT when updating
            )
            adapted_hidden = hidden_states + fast_output

            if tie_word_embeddings and embedding_kernel is not None:
                logits_update = adapted_hidden @ embedding_kernel.T
            elif sys.lm_head:
                logits_update = sys.lm_head(adapted_hidden)
            else:
                raise ValueError("LM Head not found")

            # 4. Blend outputs based on soft decision (for differentiability)
            # update_prob is the probability of choosing UPDATE
            update_prob = decision_probs[:, 1:2]  # [B, 1]

            # Expand for broadcasting: [B, 1, 1] for [B, T, V] logits
            update_prob_expanded = update_prob[:, :, None]

            # 5. Compute losses separately for skip and update
            ce_loss_skip = cross_entropy_loss(logits_skip[:, :-1], labels[:, 1:], attention_mask[:, 1:])
            ce_loss_update = cross_entropy_loss(logits_update[:, :-1], labels[:, 1:], attention_mask[:, 1:])

            # === ADVANTAGE-BASED GATING (Section 3.2) ===
            # The gating network needs TWO signals:
            # 1. Quality signal: How much does UPDATE help? (advantage)
            # 2. Cost signal: Stay close to target skip rate
            #
            # Without quality signal, gating learns to always SKIP (minimizes cost).
            # Without cost signal, gating learns to always UPDATE (maximizes quality).
            #
            # Advantage = ce_loss_skip - ce_loss_update (positive means UPDATE is better)
            # We use stop_gradient on advantage to prevent CE loss from directly training gating.
            # Instead, advantage acts as a "reward" signal for the gating decision.

            advantage = jax.lax.stop_gradient(ce_loss_skip - ce_loss_update)
            update_prob_scalar = jnp.mean(update_prob)

            # Gating loss: encourage UPDATE when advantage > 0, SKIP when advantage < 0
            # L_gating = -update_prob * advantage (minimize this = maximize update_prob when advantage > 0)
            gating_quality_loss = -update_prob_scalar * advantage

            # CE loss for TTT parameters (stop gradient to gating)
            update_prob_no_grad = jax.lax.stop_gradient(update_prob_scalar)
            ce_loss_blended = (1 - update_prob_no_grad) * ce_loss_skip + update_prob_no_grad * ce_loss_update

            # For logging
            ce_loss = ce_loss_blended

            # TTT auxiliary loss (only when updating)
            if fast_stats is not None and "ttt_loss_step_1" in fast_stats:
                l_ttt = jnp.mean(fast_stats["ttt_loss_step_1"])
                if l_ttt is None:
                    l_ttt = jnp.array(0.0)
            else:
                l_ttt = jnp.array(0.0)

            # Weighted TTT loss - stop gradient to gating network
            l_ttt_weighted = l_ttt * update_prob_no_grad

            # 6. Cost penalty for target skip rate
            update_prob_flat = update_prob[:, 0]  # [B]

            # Compute average update probability only on real code chunks
            num_real_code = jnp.maximum(jnp.sum(is_real_code.astype(jnp.float32)), 1.0)
            avg_update_prob_real = jnp.sum(update_prob_flat * is_real_code.astype(jnp.float32)) / num_real_code

            # Also compute overall for logging
            avg_update_prob = jnp.mean(update_prob_flat)

            # Cost penalty: L_cost = |d̄ - r_target| * γ (Eq. 7)
            target_update_rate = 1.0 - args.target_skip_rate
            cost_penalty = jnp.abs(avg_update_prob_real - target_update_rate) * cost_weight

            # Apply warmup (disable cost penalty during warmup to let quality signal dominate initially)
            cost_penalty = jnp.where(is_warmup, 0.0, cost_penalty)

            # Total loss: CE (for TTT) + TTT aux + Gating quality + Cost penalty
            total_loss = ce_loss + beta_ttt * l_ttt_weighted + gating_quality_loss + cost_penalty

            # Compute actual cost for logging
            # Hard decision cost: SKIP=1.0, UPDATE=3.0
            hard_cost = jnp.mean(jnp.where(decision_hard == 1, 3.0, 1.0))

            # Skip rate on real code only
            skip_on_real = jnp.sum((decision_hard == 0).astype(jnp.float32) * is_real_code.astype(jnp.float32)) / num_real_code

            return total_loss, (ce_loss, l_ttt, cost_penalty, avg_update_prob, hard_cost, decision_hard, skip_on_real, num_real_code)

        grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
        (loss, aux), grads = grad_fn(trainable_sys, rng_key)
        ce_loss, l_ttt, cost_loss, avg_update_prob, hard_cost, decision_hard, skip_on_real, num_real_code = aux

        optimizer.update(trainable_sys, grads)

        # Compute skip rate from hard decisions (overall)
        skip_rate = jnp.mean(decision_hard == 0)

        return loss, ce_loss, l_ttt, cost_loss, avg_update_prob, hard_cost, skip_rate, skip_on_real, num_real_code

    # JIT compile training step
    train_step_jit = nnx.jit(
        train_step,
        static_argnames=("beta_ttt", "cost_weight", "tie_word_embeddings", "padding_threshold"),
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
        total_cost = 0.0
        total_update_prob = 0.0
        total_hard_cost = 0.0
        total_skip_rate = 0.0
        total_skip_on_real = 0.0
        total_real_code_chunks = 0.0
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

            loss, ce, l_ttt, cost, update_prob, hard_cost, skip_rate, skip_on_real, num_real = train_step_jit(
                trainable_system,
                optimizer,
                cast(GPT2Model, ttt_model.base_model),  # train_hard_skip is GPT2-specific
                chunk_batch,
                temperature,
                subkey,
                is_warmup,
                beta_ttt=args.beta_ttt,
                cost_weight=args.cost_weight,
                tie_word_embeddings=ttt_model.tie_word_embeddings,
                padding_threshold=0.1,
            )

            # Stability check
            if not jnp.isfinite(loss) or float(ce) > 20.0:
                print(f"Warning: Skipping unstable batch (loss={loss:.4f}, ce={ce:.4f})")
                continue

            total_loss += float(loss)
            total_ce_loss += float(ce)
            total_l_ttt += float(l_ttt)
            total_cost += float(cost)
            total_update_prob += float(update_prob)
            total_hard_cost += float(hard_cost)
            total_skip_rate += float(skip_rate)
            total_skip_on_real += float(skip_on_real) * float(num_real)
            total_real_code_chunks += float(num_real)
            valid_chunks += 1

            feature_extractor.update_history(float(ce), float(hard_cost))

        if valid_chunks == 0:
            iter_count += 1
            continue

        # Average stats
        avg_loss = total_loss / valid_chunks
        avg_ce_loss = total_ce_loss / valid_chunks
        avg_l_ttt = total_l_ttt / valid_chunks
        avg_update_prob = total_update_prob / valid_chunks
        avg_hard_cost = total_hard_cost / valid_chunks
        avg_skip_rate = total_skip_rate / valid_chunks
        avg_skip_on_real = total_skip_on_real / max(total_real_code_chunks, 1.0)
        perplexity = math.exp(min(avg_ce_loss, 10.0))  # Cap to avoid overflow

        warmup_status = " [WARMUP]" if is_warmup else ""
        print(
            f"Iter {iter_count+1}{warmup_status}: Loss={avg_loss:.4f}, "
            f"CE={avg_ce_loss:.4f}, PPL={perplexity:.2f}, "
            f"SkipRate={avg_skip_rate:.2%} (real:{avg_skip_on_real:.2%}), Cost={avg_hard_cost:.2f}x, "
            f"Temp={temperature:.3f}"
        )

        # WandB Logging
        if args.wandb_project:
            wandb.log({
                "loss_total": avg_loss,
                "loss_ce": avg_ce_loss,
                "perplexity": perplexity,
                "l_ttt": avg_l_ttt,
                "update_prob": avg_update_prob,
                "skip_rate": avg_skip_rate,
                "skip_rate_real_code": avg_skip_on_real,
                "hard_cost": avg_hard_cost,
                "temperature": temperature,
                "is_warmup": float(is_warmup),
                "iteration": iter_count + 1,
                "real_code_chunks": total_real_code_chunks,
            })

        history.append({
            "iteration": iter_count,
            "loss_total": avg_loss,
            "loss_ce": avg_ce_loss,
            "perplexity": perplexity,
            "l_ttt": avg_l_ttt,
            "update_prob": avg_update_prob,
            "skip_rate": avg_skip_rate,
            "skip_rate_real_code": avg_skip_on_real,
            "hard_cost": avg_hard_cost,
            "temperature": temperature,
            "is_warmup": is_warmup,
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
                    "target_skip_rate": args.target_skip_rate,
                    "temperature": temperature,
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
            "target_skip_rate": args.target_skip_rate,
            "final_temperature": temperature,
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
        print(f"Final Skip Rate: {final_stats['skip_rate']:.2%}")
        print(f"Final Cost: {final_stats['hard_cost']:.2f}x")
        print(f"Final Perplexity: {final_stats['perplexity']:.2f}")

    if args.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    main()
