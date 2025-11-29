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
        default=30,
        help="Number of batches to evaluate",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=512,
    )
    return parser.parse_args()


def evaluate_chunks(ttt_model, gating_net, feature_extractor, batches, shuffle=False):
    """Evaluate on chunks, optionally with shuffled tokens."""
    total_loss_skip = 0.0
    total_loss_ours = 0.0
    total_chunks = 0

    for batch in tqdm(batches, desc="Shuffled" if shuffle else "Normal"):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        # Shuffle tokens within each sequence if requested
        if shuffle:
            key = jax.random.PRNGKey(total_chunks)
            for i in range(input_ids.shape[0]):
                seq_len = int(attention_mask[i].sum())
                perm = jax.random.permutation(key, seq_len)
                input_ids = input_ids.at[i, :seq_len].set(input_ids[i, perm])
                key = jax.random.split(key)[0]

        # Process chunks
        seq_len = input_ids.shape[1]
        chunk_size = 512

        for start in range(0, seq_len - chunk_size, chunk_size):
            chunk = {
                "input_ids": input_ids[:, start : start + chunk_size],
                "attention_mask": attention_mask[:, start : start + chunk_size],
            }

            # SKIP baseline
            out_skip = ttt_model(
                chunk["input_ids"],
                use_ttt=False,
                gating_scale=None,
            )
            loss_skip = cross_entropy_loss(
                out_skip["logits"][:, :-1],
                chunk["input_ids"][:, 1:],
                chunk["attention_mask"][:, 1:],
            )

            # Binary Gating (Hard Skip)
            features = feature_extractor.extract(
                input_ids=chunk["input_ids"],
                logits=out_skip["logits"],
                attention_mask=chunk["attention_mask"],
                budget_remaining=1.0,
            )
            hard_scale, decision = gating_net.get_decision(features)
            decision_val = int(decision[0])

            if decision_val == 0:  # SKIP
                loss_ours = loss_skip
            else:  # UPDATE
                out_ttt = ttt_model(
                    chunk["input_ids"],
                    use_ttt=True,
                    gating_scale=[[1.0]],
                )
                loss_ours = cross_entropy_loss(
                    out_ttt["logits"][:, :-1],
                    chunk["input_ids"][:, 1:],
                    chunk["attention_mask"][:, 1:],
                )

            total_loss_skip += float(loss_skip)
            total_loss_ours += float(loss_ours)
            total_chunks += 1

    avg_loss_skip = total_loss_skip / total_chunks
    avg_loss_ours = total_loss_ours / total_chunks

    return avg_loss_skip, avg_loss_ours, total_chunks


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
        nnx.update(trainable_system, ckpt["state"]["model"])
    print(f"Checkpoint loaded (step {ckpt.get('step', 'unknown')})")

    # Feature extractor
    feature_extractor = FeatureExtractor(
        vocab_size=tokenizer.get_vocab_size(),
        pad_token_id=tokenizer.token_to_id("<|pad|>"),
        seq_length_norm=512,
    )

    # Load data
    print("Loading data...")
    tokenizer = get_tokenizer(model_name)
    data_iter = create_data_iterator(
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        seq_length=1024,
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

    # Evaluate normal text
    print("\n1. Evaluating on NORMAL text...")
    loss_skip_normal, loss_ours_normal, n_normal = evaluate_chunks(
        ttt_model, gating_net, feature_extractor, batches, shuffle=False
    )
    ppl_skip_normal = math.exp(min(loss_skip_normal, 20))
    ppl_ours_normal = math.exp(min(loss_ours_normal, 20))
    improv_normal = (loss_skip_normal - loss_ours_normal) / loss_skip_normal * 100

    # Evaluate shuffled text
    print("\n2. Evaluating on SHUFFLED text...")
    loss_skip_shuffled, loss_ours_shuffled, n_shuffled = evaluate_chunks(
        ttt_model, gating_net, feature_extractor, batches, shuffle=True
    )
    ppl_skip_shuffled = math.exp(min(loss_skip_shuffled, 20))
    ppl_ours_shuffled = math.exp(min(loss_ours_shuffled, 20))
    improv_shuffled = (loss_skip_shuffled - loss_ours_shuffled) / loss_skip_shuffled * 100

    # Results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\n1. Normal Text (N={n_normal} chunks):")
    print(f"   SKIP:      Loss = {loss_skip_normal:.4f}, PPL = {ppl_skip_normal:.2f}")
    print(f"   PonderTTT: Loss = {loss_ours_normal:.4f}, PPL = {ppl_ours_normal:.2f}")
    print(f"   Improvement: {improv_normal:.1f}%")

    print(f"\n2. Shuffled Text (N={n_shuffled} chunks):")
    print(f"   SKIP:      Loss = {loss_skip_shuffled:.4f}, PPL = {ppl_skip_shuffled:.2f}")
    print(f"   PonderTTT: Loss = {loss_ours_shuffled:.4f}, PPL = {ppl_ours_shuffled:.2f}")
    print(f"   Improvement: {improv_shuffled:.1f}%")

    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    if improv_normal > 10 and abs(improv_shuffled) < 5:
        print("PASS: TTT only helps on normal text with learnable patterns.")
        print("      This confirms NO DATA LEAKAGE.")
    else:
        print("WARNING: Results may need investigation.")

    # LaTeX table
    print("\n" + "=" * 70)
    print("FOR PAPER (LaTeX)")
    print("=" * 70)
    print("""
\\begin{table}[h]
\\centering
\\caption{Shuffled Input Sanity Check (Hard Skip). PonderTTT only improves on normal text.}
\\begin{tabular}{lccc}
\\toprule
\\textbf{Input Type} & \\textbf{SKIP PPL} & \\textbf{Ours PPL} & \\textbf{Improv.} \\\\
\\midrule""")
    print(f"Normal Text & {ppl_skip_normal:.2f} & {ppl_ours_normal:.2f} & {improv_normal:.1f}\\% \\\\")
    if ppl_skip_shuffled > 1e6:
        print(f"Shuffled Text & $\\infty$ & $\\infty$ & {improv_shuffled:.1f}\\% \\\\")
    else:
        print(f"Shuffled Text & {ppl_skip_shuffled:.2f} & {ppl_ours_shuffled:.2f} & {improv_shuffled:.1f}\\% \\\\")
    print("""\\bottomrule
\\end{tabular}
\\end{table}
""")


if __name__ == "__main__":
    main()
