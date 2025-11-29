#!/usr/bin/env python3
"""
Shuffled Input Sanity Check for Hard Skip (Binary Gating).

Verifies that TTT only helps when there are learnable patterns,
confirming no data leakage.

Usage:
    python scripts/test_shuffled_input.py --checkpoint outputs/hard_skip/125m_skip0.8/checkpoint_XXXX
"""

import argparse
import math
import jax
import jax.numpy as jnp
from flax import nnx
from tqdm import tqdm

from ponderttt.data import create_data_iterator, get_tokenizer
from ponderttt.models import load_ttt_model
from ponderttt.models.gating_nnx import BinaryGatingConfig, BinaryGatingNetwork
from ponderttt.utils import FeatureExtractor, cross_entropy_loss
from ponderttt.utils.checkpointing import load_checkpoint


def unwrap_state(state):
    """Recursively unwrap Orbax-serialized NNX state dicts (remove 'value' wrappers)."""
    if isinstance(state, dict):
        if "value" in state and len(state) == 1:
            return state["value"]
        return {k: unwrap_state(v) for k, v in state.items()}
    return state


MODEL_SCALES = {"125m": "gpt2", "350m": "gpt2-medium", "1b": "gpt2-large"}


class TrainableSystem(nnx.Module):
    """Mirror of TrainableSystem from train_hard_skip.py for checkpoint loading."""
    def __init__(self, ttt_model, gating_net):
        self.fast_layer = ttt_model.fast_layer
        self.fast_norm = ttt_model.fast_norm
        self.gating_net = gating_net
        if hasattr(ttt_model, 'lm_head'):
            self.lm_head = ttt_model.lm_head
        else:
            self.lm_head = None


def parse_args():
    parser = argparse.ArgumentParser(description="Shuffled Input Sanity Check")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to Binary Gating checkpoint",
    )
    parser.add_argument(
        "--model_scale",
        type=str,
        default="125m",
        choices=["125m", "350m", "1b"],
    )
    parser.add_argument(
        "--num_batches",
        type=int,
        default=100,
        help="Number of batches to evaluate",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--min_valid_tokens",
        type=int,
        default=64,
        help="Minimum valid tokens in a chunk to include it",
    )
    return parser.parse_args()


def evaluate_chunks(ttt_model, gating_net, feature_extractor, batches, min_valid_tokens=64, shuffle=False):
    """Evaluate on chunks, optionally with shuffled tokens."""
    total_loss_skip = 0.0
    total_loss_ours = 0.0
    total_chunks = 0
    total_updates = 0
    skipped_empty = 0

    for batch in tqdm(batches, desc="Shuffled" if shuffle else "Normal"):
        # Use pre-chunked data from dataset
        chunks = batch["chunks"]  # [batch, num_chunks, chunk_size]
        masks = batch["chunk_attention_mask"]  # [batch, num_chunks, chunk_size]

        batch_size = chunks.shape[0]
        num_chunks = chunks.shape[1]

        for b_idx in range(batch_size):
            for c_idx in range(num_chunks):
                chunk_ids = chunks[b_idx, c_idx]  # [chunk_size]
                chunk_mask = masks[b_idx, c_idx]  # [chunk_size]

                # Check if chunk has enough valid tokens
                valid_tokens = int(chunk_mask.sum())
                if valid_tokens < min_valid_tokens:
                    skipped_empty += 1
                    continue

                # Shuffle tokens within chunk if requested
                if shuffle:
                    key = jax.random.PRNGKey(total_chunks + c_idx + b_idx * 1000)
                    # Shuffle only valid tokens
                    perm = jax.random.permutation(key, valid_tokens)
                    chunk_ids = chunk_ids.at[:valid_tokens].set(chunk_ids[:valid_tokens][perm])

                # Add batch dimension
                input_ids = chunk_ids[None, :]  # [1, chunk_size]
                attention_mask = chunk_mask[None, :]  # [1, chunk_size]

                # SKIP baseline
                out_skip = ttt_model(
                    input_ids,
                    use_ttt=False,
                    gating_scale=None,
                )
                loss_skip = cross_entropy_loss(
                    out_skip["logits"][:, :-1],
                    input_ids[:, 1:],
                    attention_mask[:, 1:],
                )

                # Binary Gating (Hard Skip)
                features = feature_extractor.extract(
                    input_ids=input_ids,
                    logits=out_skip["logits"],
                    attention_mask=attention_mask,
                    budget_remaining=1.0,
                )
                hard_scale, decision = gating_net.get_decision(features)
                decision_val = int(decision[0])

                if decision_val == 0:  # SKIP
                    loss_ours = loss_skip
                else:  # UPDATE
                    total_updates += 1
                    out_ttt = ttt_model(
                        input_ids,
                        use_ttt=True,
                        gating_scale=jnp.array([[1.0]]),
                    )
                    loss_ours = cross_entropy_loss(
                        out_ttt["logits"][:, :-1],
                        input_ids[:, 1:],
                        attention_mask[:, 1:],
                    )

                loss_skip_val = float(loss_skip)
                loss_ours_val = float(loss_ours)

                # Skip NaN or invalid values
                if math.isnan(loss_skip_val) or math.isnan(loss_ours_val):
                    continue
                if loss_skip_val < 0 or loss_ours_val < 0:
                    continue

                total_loss_skip += loss_skip_val
                total_loss_ours += loss_ours_val
                total_chunks += 1

    if total_chunks == 0:
        return float('nan'), float('nan'), 0, 0, 0

    avg_loss_skip = total_loss_skip / total_chunks
    avg_loss_ours = total_loss_ours / total_chunks
    update_rate = total_updates / total_chunks if total_chunks > 0 else 0

    return avg_loss_skip, avg_loss_ours, total_chunks, update_rate, skipped_empty


def main():
    args = parse_args()

    print("=" * 70)
    print("PonderTTT Shuffled Input Sanity Check (Hard Skip / Binary Gating)")
    print("=" * 70)

    # Load model
    print("\nLoading model...")
    model_name = MODEL_SCALES[args.model_scale]
    ttt_model, _ = load_ttt_model(model_name, load_pretrained=True)

    # Initialize Binary Gating network
    print("Initializing Binary Gating network...")
    config = BinaryGatingConfig(feature_dim=32, hidden_dim=64)
    rngs = nnx.Rngs(0)
    gating_net = BinaryGatingNetwork(config, rngs)

    # Create TrainableSystem for checkpoint loading
    trainable_system = TrainableSystem(ttt_model, gating_net)

    # Load checkpoint (model state only, no optimizer needed for inference)
    print(f"Loading checkpoint from {args.checkpoint}...")
    ckpt = load_checkpoint(args.checkpoint, target=None)

    # Update model with loaded state
    if "state" in ckpt and "model" in ckpt["state"]:
        model_state = unwrap_state(ckpt["state"]["model"])
        nnx.update(trainable_system, model_state)
    print(f"Checkpoint loaded (step {ckpt.get('step', 'unknown')})")

    # Load tokenizer and feature extractor
    print("Loading data...")
    tokenizer = get_tokenizer(model_name)
    feature_extractor = FeatureExtractor(
        vocab_size=tokenizer.get_vocab_size(),
        pad_token_id=tokenizer.token_to_id("<|pad|>"),
        seq_length_norm=args.chunk_size,
    )

    # Use shorter sequences to get more valid chunks
    seq_length = args.chunk_size * 4  # 4 chunks per sequence
    data_iter = create_data_iterator(
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        seq_length=seq_length,
        chunk_size=args.chunk_size,
        split="train",
        language="Python",
    )

    # Collect batches
    batches = []
    for i, batch in enumerate(data_iter):
        if i >= args.num_batches:
            break
        batches.append(batch)

    print(f"\nCollected {len(batches)} batches")
    print(f"Config: batch_size={args.batch_size}, chunk_size={args.chunk_size}, min_valid={args.min_valid_tokens}")

    # Evaluate normal text
    print("\n1. Evaluating on NORMAL text...")
    loss_skip_normal, loss_ours_normal, n_normal, update_rate_normal, skipped_normal = evaluate_chunks(
        ttt_model, gating_net, feature_extractor, batches,
        min_valid_tokens=args.min_valid_tokens, shuffle=False
    )
    ppl_skip_normal = math.exp(min(loss_skip_normal, 20)) if not math.isnan(loss_skip_normal) else float('inf')
    ppl_ours_normal = math.exp(min(loss_ours_normal, 20)) if not math.isnan(loss_ours_normal) else float('inf')
    improv_normal = (loss_skip_normal - loss_ours_normal) / loss_skip_normal * 100 if loss_skip_normal > 0 else 0

    # Reset feature extractor history
    feature_extractor.difficulty_ema = 0.0
    feature_extractor.difficulty_sq_ema = 0.0
    feature_extractor.cost_ema = 0.0

    # Evaluate shuffled text
    print("\n2. Evaluating on SHUFFLED text...")
    loss_skip_shuffled, loss_ours_shuffled, n_shuffled, update_rate_shuffled, skipped_shuffled = evaluate_chunks(
        ttt_model, gating_net, feature_extractor, batches,
        min_valid_tokens=args.min_valid_tokens, shuffle=True
    )
    ppl_skip_shuffled = math.exp(min(loss_skip_shuffled, 20)) if not math.isnan(loss_skip_shuffled) else float('inf')
    ppl_ours_shuffled = math.exp(min(loss_ours_shuffled, 20)) if not math.isnan(loss_ours_shuffled) else float('inf')
    improv_shuffled = (loss_skip_shuffled - loss_ours_shuffled) / loss_skip_shuffled * 100 if loss_skip_shuffled > 0 else 0

    # Results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\n1. Normal Text (N={n_normal} chunks, skipped {skipped_normal} empty):")
    print(f"   SKIP:      Loss = {loss_skip_normal:.4f}, PPL = {ppl_skip_normal:.2f}")
    print(f"   PonderTTT: Loss = {loss_ours_normal:.4f}, PPL = {ppl_ours_normal:.2f}")
    print(f"   Improvement: {improv_normal:.1f}%")
    print(f"   Update Rate: {update_rate_normal:.1%}")

    print(f"\n2. Shuffled Text (N={n_shuffled} chunks, skipped {skipped_shuffled} empty):")
    print(f"   SKIP:      Loss = {loss_skip_shuffled:.4f}, PPL = {ppl_skip_shuffled:.2f}")
    print(f"   PonderTTT: Loss = {loss_ours_shuffled:.4f}, PPL = {ppl_ours_shuffled:.2f}")
    print(f"   Improvement: {improv_shuffled:.1f}%")
    print(f"   Update Rate: {update_rate_shuffled:.1%}")

    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    if improv_normal > 10 and abs(improv_shuffled) < 10:
        print("PASS: TTT provides significant improvement on normal text")
        print("      but minimal/no improvement on shuffled text.")
        print("      This confirms the model learns patterns, NOT data leakage.")
    elif ppl_skip_shuffled > 50 and abs(improv_shuffled) < 10:
        print("PASS: Shuffled text has much higher PPL (as expected)")
        print("      and TTT provides no benefit, confirming no data leakage.")
    else:
        print("NOTE: Results may need investigation.")
        print(f"      Normal improvement: {improv_normal:.1f}%")
        print(f"      Shuffled improvement: {improv_shuffled:.1f}%")

    # LaTeX table
    print("\n" + "=" * 70)
    print("FOR PAPER (LaTeX)")
    print("=" * 70)
    print(f"""
\\begin{{table}}[h]
\\centering
\\caption{{Shuffled Input Sanity Check (Hard Skip). PonderTTT only improves on normal text.}}
\\begin{{tabular}}{{lccc}}
\\toprule
\\textbf{{Input Type}} & \\textbf{{SKIP PPL}} & \\textbf{{Ours PPL}} & \\textbf{{Improv.}} \\\\
\\midrule
Normal Text & {ppl_skip_normal:.2f} & {ppl_ours_normal:.2f} & {improv_normal:.1f}\\% \\\\
Shuffled Text & {ppl_skip_shuffled:.2f} & {ppl_ours_shuffled:.2f} & {improv_shuffled:.1f}\\% \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""")


if __name__ == "__main__":
    main()
