"""
Compare Learned Top-k Gating vs Random vs Oracle.

This is the key evaluation script that answers:
"Does learned gating select better chunks than random?"

All methods are evaluated with the SAME update budget for fair comparison.

Usage:
    python -m ponderttt.experiments.compare_topk_methods \
        --checkpoint outputs/topk_gating \
        --target_update_rate 0.3 \
        --num_samples 500
"""

import argparse
import json
import math
from pathlib import Path
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from tqdm import tqdm

from ..data import create_data_iterator, get_tokenizer
from ..models import GPT2Model, load_ttt_model
from ..models.gating_nnx import BinaryGatingConfig, BinaryGatingNetwork
from ..utils import FeatureExtractor, per_sample_cross_entropy_loss
from ..utils.checkpointing import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Compare Top-k Gating Methods")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained gating checkpoint")
    parser.add_argument("--model_scale", type=str, default="125m", choices=["125m", "350m"])
    parser.add_argument("--target_update_rate", type=float, default=0.3, help="Update budget (same for all methods)")
    parser.add_argument("--num_samples", type=int, default=500, help="Number of chunks to evaluate")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs/comparison")
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--eval_split", type=str, default="train", help="Data split to evaluate on")
    parser.add_argument("--skip_examples", type=int, default=160000, help="Skip N examples (for held-out eval)")
    return parser.parse_args()


def get_model_name(model_scale: str) -> str:
    return {"125m": "gpt2", "350m": "gpt2-medium", "1b": "gpt2-large"}[model_scale]


def main():
    args = parse_args()

    print("=" * 60)
    print("Comparing: Learned Gating vs Random vs Oracle")
    print("=" * 60)
    print(f"Update budget: {args.target_update_rate*100:.0f}% (same for all methods)")
    print(f"Checkpoint: {args.checkpoint}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = get_model_name(args.model_scale)
    tokenizer = get_tokenizer(model_name)

    # Load TTT Model
    print("\nLoading TTT model...")
    ttt_model, ttt_config = load_ttt_model(
        model_name=model_name,
        fast_weight_type="ttt",
        seed=args.seed,
        load_pretrained=True,
        vocab_size=tokenizer.get_vocab_size(),
    )

    # Initialize Gating Network
    gating_config = BinaryGatingConfig(
        feature_dim=32,
        hidden_dim=64,
        initial_temperature=1.0,
        min_temperature=0.1,
        scale_when_update=1.0,
    )
    rngs = nnx.Rngs(args.seed + 1)
    gating_net = BinaryGatingNetwork(config=gating_config, rngs=rngs)

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")

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

    # Load checkpoint with dummy optimizer
    optimizer = nnx.Optimizer(
        trainable_system,
        optax.adam(1e-3),
        wrt=nnx.All(nnx.Param),
    )

    try:
        import optax
        target = {
            "state": {"model": nnx.state(trainable_system), "optimizer": nnx.state(optimizer)},
            "step": 0,
            "metadata": {}
        }
        ckpt = load_checkpoint(args.checkpoint, target=target)
        nnx.update(trainable_system, ckpt["state"]["model"])
        threshold_ema = ckpt.get("metadata", {}).get("threshold_ema", 0.0)
        print(f"Loaded checkpoint, threshold_ema={threshold_ema:.4f}")
    except Exception as e:
        print(f"Warning: Could not load checkpoint: {e}")
        print("Proceeding with random initialized gating network")
        threshold_ema = 0.0

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
    max_examples = math.ceil(args.num_samples / chunks_per_sequence) * args.batch_size + args.skip_examples

    data_iter = create_data_iterator(
        tokenizer=tokenizer,
        split=args.eval_split,
        batch_size=args.batch_size,
        seq_length=seq_length,
        chunk_size=chunk_size,
        max_examples=max_examples,
        num_workers=args.num_workers,
    )

    # Evaluation function
    @jax.jit
    def evaluate_chunk(
        base_model: GPT2Model,
        trainable_sys: TrainableSystem,
        batch: dict,
        tie_word_embeddings: bool = True,
    ):
        """Evaluate a chunk and get all necessary metrics."""
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        position_ids = batch["position_ids"]
        labels = input_ids

        # Base model forward
        hidden_states = base_model(input_ids, position_ids=position_ids, train=False)

        # SKIP logits
        if tie_word_embeddings:
            embedding_kernel = jnp.asarray(base_model.wte.embedding)
            logits_skip = hidden_states @ embedding_kernel.T
        else:
            logits_skip = hidden_states

        # Features for gating
        features = feature_extractor.extract(
            input_ids=input_ids,
            logits=logits_skip,
            hidden_states=[hidden_states],
            attention_mask=attention_mask,
            budget_remaining=1.0,
        )

        # Gating prediction
        _, decision_probs, _ = trainable_sys.gating_net(features, train=False)
        gating_score = decision_probs[:, 1]  # P(UPDATE)

        # UPDATE logits (TTT)
        hidden_states_normed = trainable_sys.fast_norm(hidden_states)
        fast_output, _ = trainable_sys.fast_layer(
            hidden_states_normed,
            mask=attention_mask,
            position_ids=position_ids,
            train=False,
            gating_scale=jnp.ones((input_ids.shape[0], 1)),
        )
        adapted_hidden = hidden_states + fast_output

        if tie_word_embeddings:
            logits_update = adapted_hidden @ embedding_kernel.T
        else:
            logits_update = adapted_hidden

        # Per-sample losses
        ce_skip = per_sample_cross_entropy_loss(
            logits_skip[:, :-1], labels[:, 1:], attention_mask[:, 1:]
        )
        ce_update = per_sample_cross_entropy_loss(
            logits_update[:, :-1], labels[:, 1:], attention_mask[:, 1:]
        )

        advantage = ce_skip - ce_update

        return ce_skip, ce_update, advantage, gating_score

    # Collect data
    print(f"\nEvaluating {args.num_samples} chunks...")
    all_ce_skip = []
    all_ce_update = []
    all_advantages = []
    all_gating_scores = []

    chunks_collected = 0
    examples_seen = 0
    pbar = tqdm(total=args.num_samples)

    for sequence_batch in data_iter:
        examples_seen += args.batch_size

        # Skip examples for held-out evaluation
        if examples_seen <= args.skip_examples:
            continue

        if chunks_collected >= args.num_samples:
            break

        chunks = sequence_batch["chunks"]
        masks = sequence_batch["chunk_attention_mask"]
        num_chunks = chunks.shape[1]

        feature_extractor.reset_history()

        for c_idx in range(num_chunks):
            if chunks_collected >= args.num_samples:
                break

            chunk_batch = {
                "input_ids": chunks[:, c_idx],
                "attention_mask": masks[:, c_idx],
                "position_ids": jnp.arange(
                    c_idx * chunk_size,
                    (c_idx + 1) * chunk_size,
                    dtype=jnp.int32
                )[None, :].repeat(chunks.shape[0], axis=0)
            }

            valid_tokens = jnp.sum(chunk_batch["attention_mask"][:, 1:])
            if valid_tokens < 16:
                continue

            ce_skip, ce_update, advantage, gating_score = evaluate_chunk(
                cast(GPT2Model, ttt_model.base_model),
                trainable_system,
                chunk_batch,
                ttt_model.tie_word_embeddings,
            )

            all_ce_skip.extend(np.array(ce_skip).tolist())
            all_ce_update.extend(np.array(ce_update).tolist())
            all_advantages.extend(np.array(advantage).tolist())
            all_gating_scores.extend(np.array(gating_score).tolist())

            chunks_collected += len(advantage)
            pbar.update(len(advantage))

            feature_extractor.update_history(float(jnp.mean(ce_skip)), 1.0)

    pbar.close()

    # Convert to numpy
    ce_skip = np.array(all_ce_skip)
    ce_update = np.array(all_ce_update)
    advantages = np.array(all_advantages)
    gating_scores = np.array(all_gating_scores)

    n_samples = len(advantages)
    n_update = max(1, int(n_samples * args.target_update_rate))

    print(f"\nCollected {n_samples} samples")
    print(f"Will select {n_update} ({args.target_update_rate*100:.0f}%) for UPDATE\n")

    # === COMPARISON ===
    print("=" * 60)
    print("RESULTS: Same Budget Comparison")
    print("=" * 60)

    results = {}

    # 1. Baseline (100% SKIP)
    baseline_ce = ce_skip.mean()
    baseline_ppl = math.exp(min(baseline_ce, 10))
    print(f"\n1. Baseline (100% SKIP):")
    print(f"   CE: {baseline_ce:.4f}, PPL: {baseline_ppl:.2f}")
    results["baseline"] = {"ce": baseline_ce, "ppl": baseline_ppl}

    # 2. Full TTT (100% UPDATE)
    full_ce = ce_update.mean()
    full_ppl = math.exp(min(full_ce, 10))
    print(f"\n2. Full TTT (100% UPDATE):")
    print(f"   CE: {full_ce:.4f}, PPL: {full_ppl:.2f}")
    results["full_ttt"] = {"ce": full_ce, "ppl": full_ppl}

    # 3. Oracle Top-k (Upper Bound)
    oracle_indices = np.argsort(advantages)[-n_update:]
    oracle_mask = np.zeros(n_samples, dtype=bool)
    oracle_mask[oracle_indices] = True
    oracle_ce_values = np.where(oracle_mask, ce_update, ce_skip)
    oracle_ce = oracle_ce_values.mean()
    oracle_ppl = math.exp(min(oracle_ce, 10))
    print(f"\n3. Oracle Top-{args.target_update_rate*100:.0f}% (Upper Bound):")
    print(f"   CE: {oracle_ce:.4f}, PPL: {oracle_ppl:.2f}")
    print(f"   Improvement over baseline: {(baseline_ppl - oracle_ppl) / baseline_ppl * 100:.1f}%")
    results["oracle"] = {"ce": oracle_ce, "ppl": oracle_ppl}

    # 4. Random Selection (same budget)
    np.random.seed(args.seed)
    random_indices = np.random.choice(n_samples, n_update, replace=False)
    random_mask = np.zeros(n_samples, dtype=bool)
    random_mask[random_indices] = True
    random_ce_values = np.where(random_mask, ce_update, ce_skip)
    random_ce = random_ce_values.mean()
    random_ppl = math.exp(min(random_ce, 10))
    print(f"\n4. Random {args.target_update_rate*100:.0f}%:")
    print(f"   CE: {random_ce:.4f}, PPL: {random_ppl:.2f}")
    print(f"   Improvement over baseline: {(baseline_ppl - random_ppl) / baseline_ppl * 100:.1f}%")
    results["random"] = {"ce": random_ce, "ppl": random_ppl}

    # 5. Learned Gating (Top-k by gating score)
    learned_indices = np.argsort(gating_scores)[-n_update:]
    learned_mask = np.zeros(n_samples, dtype=bool)
    learned_mask[learned_indices] = True
    learned_ce_values = np.where(learned_mask, ce_update, ce_skip)
    learned_ce = learned_ce_values.mean()
    learned_ppl = math.exp(min(learned_ce, 10))
    print(f"\n5. Learned Gating (Top-{args.target_update_rate*100:.0f}% by score):")
    print(f"   CE: {learned_ce:.4f}, PPL: {learned_ppl:.2f}")
    print(f"   Improvement over baseline: {(baseline_ppl - learned_ppl) / baseline_ppl * 100:.1f}%")
    results["learned"] = {"ce": learned_ce, "ppl": learned_ppl}

    # 6. Learned Gating with threshold (as would be used in inference)
    if threshold_ema > 0:
        # Convert gating scores to decisions using threshold
        # Note: gating_score is P(UPDATE), we need to compare logit to threshold
        # Approximate: if score > 0.5 * (1 + threshold_scale), then UPDATE
        threshold_mask = gating_scores > 0.5
        threshold_update_rate = threshold_mask.mean()
        threshold_ce_values = np.where(threshold_mask, ce_update, ce_skip)
        threshold_ce = threshold_ce_values.mean()
        threshold_ppl = math.exp(min(threshold_ce, 10))
        print(f"\n6. Learned Gating (threshold-based, actual rate={threshold_update_rate*100:.1f}%):")
        print(f"   CE: {threshold_ce:.4f}, PPL: {threshold_ppl:.2f}")
        results["learned_threshold"] = {
            "ce": threshold_ce,
            "ppl": threshold_ppl,
            "actual_update_rate": threshold_update_rate
        }

    # === KEY METRICS ===
    print("\n" + "=" * 60)
    print("KEY METRICS")
    print("=" * 60)

    # Oracle vs Random gap
    oracle_random_gap = (random_ppl - oracle_ppl) / random_ppl * 100
    print(f"\nOracle is {oracle_random_gap:.1f}% better than Random")
    print("(This is the maximum possible improvement from learned selection)")

    # Learned vs Random gap
    learned_random_gap = (random_ppl - learned_ppl) / random_ppl * 100
    print(f"\nLearned is {learned_random_gap:.1f}% better than Random")

    # How much of oracle gap does learned capture?
    if oracle_random_gap > 0:
        gap_captured = learned_random_gap / oracle_random_gap * 100
        print(f"Learned captures {gap_captured:.1f}% of Oracle's advantage")
    else:
        gap_captured = 0
        print("Warning: Oracle-Random gap is 0 or negative")

    results["metrics"] = {
        "oracle_vs_random_pct": oracle_random_gap,
        "learned_vs_random_pct": learned_random_gap,
        "gap_captured_pct": gap_captured,
        "update_rate": args.target_update_rate,
        "n_samples": n_samples,
        "n_update": n_update,
    }

    # === SELECTION QUALITY ANALYSIS ===
    print("\n" + "=" * 60)
    print("SELECTION QUALITY ANALYSIS")
    print("=" * 60)

    # Overlap between learned and oracle selections
    overlap = np.sum(learned_mask & oracle_mask)
    overlap_pct = overlap / n_update * 100
    print(f"\nOverlap between Learned and Oracle selections: {overlap}/{n_update} ({overlap_pct:.1f}%)")

    # Correlation between gating scores and advantages
    correlation = np.corrcoef(gating_scores, advantages)[0, 1]
    print(f"Correlation(gating_score, advantage): {correlation:.3f}")

    # Spearman rank correlation
    from scipy import stats
    spearman_corr, _ = stats.spearmanr(gating_scores, advantages)
    print(f"Spearman rank correlation: {spearman_corr:.3f}")

    results["selection_quality"] = {
        "overlap_with_oracle": overlap,
        "overlap_pct": overlap_pct,
        "pearson_correlation": correlation,
        "spearman_correlation": spearman_corr,
    }

    # === SAVE RESULTS ===
    output_file = output_dir / f"comparison_{args.model_scale}_k{args.target_update_rate}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")

    # === CONCLUSION ===
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)

    if learned_random_gap > 1.0:
        print(f"\n SUCCESS: Learned gating is {learned_random_gap:.1f}% better than random!")
        print(f"   It captures {gap_captured:.1f}% of the oracle's advantage.")
    elif learned_random_gap > 0:
        print(f"\n PARTIAL SUCCESS: Learned gating is {learned_random_gap:.1f}% better than random.")
        print(f"   But only captures {gap_captured:.1f}% of oracle's advantage.")
        print("   Consider: more training, better features, or different k value.")
    else:
        print(f"\n NEEDS IMPROVEMENT: Learned gating is not better than random.")
        print("   Possible causes:")
        print("   - Insufficient training")
        print("   - Features don't capture difficulty well")
        print("   - k value might be suboptimal")


if __name__ == "__main__":
    import optax  # Import here to avoid issues in jit
    main()
