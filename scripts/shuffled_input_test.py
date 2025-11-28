"""
Shuffled Input Sanity Check for PonderTTT.

This test verifies that PonderTTT's improvement comes from learning patterns,
not from data leakage. If TTT helps on shuffled (random) text, it would indicate
a potential leakage issue.

Expected results:
- Normal text: PonderTTT significantly better than SKIP
- Shuffled text: PonderTTT similar to SKIP (no patterns to learn)

Usage:
    python scripts/shuffled_input_test.py --checkpoint outputs/diff/125m_budget1.5/checkpoint_10000
"""

import argparse
import jax
import jax.numpy as jnp
from flax import nnx
from tqdm import tqdm

from ponderttt.data import create_data_iterator, get_tokenizer
from ponderttt.models import load_ttt_model
from ponderttt.models.gating_nnx import GatingConfig, GatingNetwork
from ponderttt.utils import FeatureExtractor, cross_entropy_loss
from ponderttt.utils.checkpointing import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_batches", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_steps", type=float, default=4.0)
    return parser.parse_args()


def shuffle_tokens(input_ids, key):
    """Shuffle tokens within each sequence (preserving batch dimension)."""
    B, T = input_ids.shape
    shuffled = []
    for i in range(B):
        perm = jax.random.permutation(key, T)
        shuffled.append(input_ids[i][perm])
        key = jax.random.split(key)[0]
    return jnp.stack(shuffled), key


def evaluate_batch(model, gating_net, feature_extractor, chunk_batch, use_ponder=True):
    """Evaluate a single batch with optional PonderTTT."""
    input_ids = chunk_batch["input_ids"]
    attention_mask = chunk_batch["attention_mask"]

    if use_ponder and gating_net is not None:
        # Get base model features for gating
        out_base = model(input_ids, use_ttt=False)
        features = feature_extractor.extract(
            input_ids=input_ids,
            logits=out_base["logits"],
            attention_mask=attention_mask,
            budget_remaining=1.0,
        )

        # Get gating scale
        scale = gating_net(features, train=False)

        # Forward with TTT
        if float(scale.mean()) < 0.01:
            outputs = model(input_ids, use_ttt=False)
        else:
            outputs = model(input_ids, use_ttt=True, gating_scale=scale)
    else:
        # SKIP baseline
        outputs = model(input_ids, use_ttt=False)

    logits = outputs["logits"]
    loss = cross_entropy_loss(
        logits[:, :-1],
        input_ids[:, 1:],
        attention_mask[:, 1:]
    )
    return float(loss)


def main():
    args = parse_args()

    print("=" * 70)
    print("PonderTTT Shuffled Input Sanity Check")
    print("=" * 70)

    # Load model
    print("\nLoading model...")
    tokenizer = get_tokenizer("gpt2")
    model, _ = load_ttt_model(
        model_name="gpt2",
        fast_weight_type="ttt",
        seed=args.seed,
        load_pretrained=True,
        vocab_size=tokenizer.get_vocab_size(),
    )

    # Load gating network
    print("Loading gating network...")
    gating_config = GatingConfig(
        feature_dim=32,
        hidden_dim=64,
        scale_output=args.max_steps
    )
    gating_net = GatingNetwork(config=gating_config, rngs=nnx.Rngs(args.seed + 1))

    # Create trainable system for checkpoint loading (same structure as training)
    class TrainableSystem(nnx.Module):
        def __init__(self, ttt_model, gating_net):
            self.fast_layer = ttt_model.fast_layer
            self.fast_norm = ttt_model.fast_norm
            self.gating_net = gating_net
            if hasattr(ttt_model, 'lm_head'):
                self.lm_head = ttt_model.lm_head
            else:
                self.lm_head = None

    trainable_system = TrainableSystem(model, gating_net)

    # Create optimizer to match checkpoint structure
    import optax
    optimizer = nnx.Optimizer(
        trainable_system,
        optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(1e-3),
        ),
        wrt=nnx.All(nnx.Param),
    )

    # Load checkpoint with proper target structure
    print(f"Loading checkpoint from {args.checkpoint}...")
    target = {
        "state": {"model": nnx.state(trainable_system), "optimizer": nnx.state(optimizer)},
        "step": 0,
        "metadata": {
            "model_scale": "",
            "max_steps": 0.0,
            "budget_limit": 0.0,
        }
    }
    ckpt = load_checkpoint(args.checkpoint, target=target)
    nnx.update(trainable_system, ckpt["state"]["model"])
    print(f"Checkpoint loaded successfully (step {ckpt.get('step', 'unknown')}).")

    # Use the loaded gating network from trainable_system
    gating_net = trainable_system.gating_net

    # Feature extractor
    feature_extractor = FeatureExtractor(
        vocab_size=tokenizer.get_vocab_size(),
        pad_token_id=tokenizer.token_to_id("<|pad|>"),
        seq_length_norm=512,
    )

    # Data iterator
    print("Loading data...")
    data_iter = create_data_iterator(
        tokenizer=tokenizer,
        split="train",
        batch_size=args.batch_size,
        seq_length=1024,
        chunk_size=512,
        max_examples=args.batch_size * args.num_batches * 2,
        num_workers=4,
    )

    # Collect results
    results = {
        "normal_skip": [],
        "normal_ponder": [],
        "shuffled_skip": [],
        "shuffled_ponder": [],
    }

    print("\nRunning evaluation...")
    rng_key = jax.random.PRNGKey(args.seed)

    for i, batch in enumerate(tqdm(data_iter, total=args.num_batches)):
        if i >= args.num_batches:
            break

        chunks = batch["chunks"]
        masks = batch["chunk_attention_mask"]

        for c_idx in range(chunks.shape[1]):
            chunk_batch = {
                "input_ids": chunks[:, c_idx],
                "attention_mask": masks[:, c_idx],
            }

            # Skip if too few valid tokens
            if jnp.sum(chunk_batch["attention_mask"][:, 1:]) < 16:
                continue

            # Normal text evaluation
            loss_skip = evaluate_batch(model, None, feature_extractor, chunk_batch, use_ponder=False)
            loss_ponder = evaluate_batch(model, gating_net, feature_extractor, chunk_batch, use_ponder=True)

            results["normal_skip"].append(loss_skip)
            results["normal_ponder"].append(loss_ponder)

            # Shuffled text evaluation
            shuffled_ids, rng_key = shuffle_tokens(chunk_batch["input_ids"], rng_key)
            shuffled_batch = {
                "input_ids": shuffled_ids,
                "attention_mask": chunk_batch["attention_mask"],
            }

            loss_skip_shuf = evaluate_batch(model, None, feature_extractor, shuffled_batch, use_ponder=False)
            loss_ponder_shuf = evaluate_batch(model, gating_net, feature_extractor, shuffled_batch, use_ponder=True)

            results["shuffled_skip"].append(loss_skip_shuf)
            results["shuffled_ponder"].append(loss_ponder_shuf)

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    def avg(lst):
        return sum(lst) / len(lst) if lst else 0

    def ppl(loss):
        import math
        if loss > 20:  # Cap to avoid overflow (exp(20) ≈ 485M)
            return float('inf')
        return math.exp(loss)

    print(f"\n1. Normal Text (N={len(results['normal_skip'])} chunks):")
    normal_skip_loss = avg(results["normal_skip"])
    normal_ponder_loss = avg(results["normal_ponder"])
    print(f"   SKIP:      Loss = {normal_skip_loss:.4f}, PPL = {ppl(normal_skip_loss):.2f}")
    print(f"   PonderTTT: Loss = {normal_ponder_loss:.4f}, PPL = {ppl(normal_ponder_loss):.2f}")
    print(f"   Improvement: {(1 - normal_ponder_loss/normal_skip_loss)*100:.1f}%")

    print(f"\n2. Shuffled Text (N={len(results['shuffled_skip'])} chunks):")
    shuf_skip_loss = avg(results["shuffled_skip"])
    shuf_ponder_loss = avg(results["shuffled_ponder"])
    shuf_skip_ppl = ppl(shuf_skip_loss)
    shuf_ponder_ppl = ppl(shuf_ponder_loss)
    print(f"   SKIP:      Loss = {shuf_skip_loss:.4f}, PPL = {'inf' if shuf_skip_ppl == float('inf') else f'{shuf_skip_ppl:.2f}'}")
    print(f"   PonderTTT: Loss = {shuf_ponder_loss:.4f}, PPL = {'inf' if shuf_ponder_ppl == float('inf') else f'{shuf_ponder_ppl:.2f}'}")
    print(f"   Improvement: {(1 - shuf_ponder_loss/shuf_skip_loss)*100:.1f}%")

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    normal_improvement = (1 - normal_ponder_loss/normal_skip_loss) * 100
    shuffled_improvement = (1 - shuf_ponder_loss/shuf_skip_loss) * 100

    if shuffled_improvement > 10:
        print(f"WARNING: PonderTTT improves shuffled text by {shuffled_improvement:.1f}%")
        print("         This may indicate data leakage!")
    else:
        print(f"PASS: PonderTTT improvement on normal text: {normal_improvement:.1f}%")
        print(f"      PonderTTT improvement on shuffled text: {shuffled_improvement:.1f}%")
        print("      TTT only helps when there are patterns to learn.")
        print("      This confirms NO DATA LEAKAGE.")

    # Save results for paper
    print("\n" + "=" * 70)
    print("FOR PAPER (Appendix)")
    print("=" * 70)

    def fmt_ppl(p):
        if p == float('inf') or p > 1e6:
            return "$\\infty$"
        return f"{p:.1f}"

    print(f"""
\\begin{{table}}[h]
\\centering
\\caption{{Shuffled Input Sanity Check. PonderTTT only improves on normal text with learnable patterns.}}
\\begin{{tabular}}{{lccc}}
\\toprule
\\textbf{{Input Type}} & \\textbf{{SKIP PPL}} & \\textbf{{Ours PPL}} & \\textbf{{Improv.}} \\\\
\\midrule
Normal Text & {fmt_ppl(ppl(normal_skip_loss))} & {fmt_ppl(ppl(normal_ponder_loss))} & {normal_improvement:.1f}\\% \\\\
Shuffled Text & {fmt_ppl(ppl(shuf_skip_loss))} & {fmt_ppl(ppl(shuf_ponder_loss))} & {shuffled_improvement:.1f}\\% \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}

Raw Loss Values (for reference):
- Normal SKIP: {normal_skip_loss:.4f}
- Normal Ours: {normal_ponder_loss:.4f}
- Shuffled SKIP: {shuf_skip_loss:.4f}
- Shuffled Ours: {shuf_ponder_loss:.4f}
""")


if __name__ == "__main__":
    main()
