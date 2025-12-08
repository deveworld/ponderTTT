"""
Compare Differentiable Gating vs Fixed Baselines.

Usage:
    python -m ponderttt.experiments.compare_methods --model_scale 125m --budget 2.0
"""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from flax import nnx
from tqdm import tqdm
from typing import Optional, cast

from ..data import create_data_iterator, get_tokenizer
from ..models import load_ttt_model, TTTTransformerLM, TTTModel
from ..models.gating_nnx import GatingConfig, GatingNetwork, BinaryGatingConfig, BinaryGatingNetwork
from ..utils import FeatureExtractor, cross_entropy_loss
from ..utils.checkpointing import load_checkpoint


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


# --- JIT-compiled Helpers ---

@nnx.jit(static_argnames=("vocab_size", "pad_token_id", "seq_norm"))
def jit_base_forward_and_features(
    model: TTTTransformerLM,
    input_ids: jax.Array,
    attention_mask: jax.Array,
    position_ids: jax.Array,
    budget_remaining: float,
    diff_ema: float,
    diff_sq_ema: float,
    cost_ema: float,
    vocab_size: int,
    pad_token_id: int,
    seq_norm: float,
):
    """Run base model forward and extract features (JIT compiled)."""
    # 1. Base Forward (No TTT)
    out_base = model(
        input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_ttt=False,
    )
    logits = out_base["logits"]
    hidden_states = out_base["hidden_states"]

    # 2. Feature Extraction
    extractor = FeatureExtractor(
        vocab_size=vocab_size,
        pad_token_id=pad_token_id,
        seq_length_norm=seq_norm,
    )
    extractor.difficulty_ema = diff_ema
    extractor.difficulty_sq_ema = diff_sq_ema
    extractor.cost_ema = cost_ema

    # IMPORTANT: Pass hidden_states to match training feature extraction
    features = extractor.extract(
        input_ids=input_ids,
        logits=logits,
        hidden_states=[hidden_states],
        attention_mask=attention_mask,
        budget_remaining=budget_remaining,
    )

    return logits, features


@nnx.jit(static_argnames=("threshold",))
def jit_binary_decision(
    gating_net: BinaryGatingNetwork,
    features: jax.Array,
    threshold: float = 0.5,
    rng_key: jax.Array | None = None,
):
    """Get binary gating decision (JIT compiled)."""
    return gating_net.get_decision(features, threshold=threshold, rng_key=rng_key)


@nnx.jit
def jit_continuous_scale(gating_net: GatingNetwork, features: jax.Array):
    """Get continuous gating scale (JIT compiled)."""
    return gating_net(features, train=False)


@nnx.jit
def jit_eval_with_scale(
    model: TTTTransformerLM,
    input_ids: jax.Array,
    attention_mask: jax.Array,
    position_ids: jax.Array,
    gating_scale: jax.Array,
):
    """Run model with specific gating scale and compute loss (JIT compiled)."""
    out = model(
        input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_ttt=True,
        gating_scale=gating_scale,
    )
    loss = cross_entropy_loss(
        out["logits"][:, :-1],
        input_ids[:, 1:],
        attention_mask[:, 1:],
    )
    return loss


@nnx.jit
def jit_loss_from_logits(
    logits: jax.Array,
    input_ids: jax.Array,
    attention_mask: jax.Array,
):
    """Compute loss from existing logits (JIT compiled)."""
    return cross_entropy_loss(
        logits[:, :-1],
        input_ids[:, 1:],
        attention_mask[:, 1:],
    )


def evaluate_oracle(
    method_name: str,
    model_scale: str,
    update_rate: float,  # Same update rate as Binary Gating for fair comparison
    num_batches: int,
    batch_size: int,
    seed: int,
    model: Optional[TTTModel] = None,
    language: str = "Python",
    split: str = "test",
    skip_examples: int = 0,
    num_workers: int = 32,
):
    """
    Oracle baseline: compute advantage for each chunk and select top-k% to update.
    This provides the upper bound for learned gating.
    """
    print(f"\nEvaluating {method_name} (update_rate={update_rate:.1%}) on {language}...")

    model_name = {"125m": "gpt2", "350m": "gpt2-medium", "1b": "gpt2-large"}[model_scale]
    tokenizer = get_tokenizer(model_name)

    # Load TTT Model if not provided
    if model is None:
        ttt_model, _ = load_ttt_model(
            model_name=model_name,
            fast_weight_type="ttt",
            seed=seed,
            load_pretrained=True,
            vocab_size=tokenizer.get_vocab_size(),
        )
    else:
        ttt_model = model

    data_iter = create_data_iterator(
        tokenizer=tokenizer,
        split=split,
        language=language,
        batch_size=batch_size,
        seq_length=1024,
        chunk_size=512,
        max_examples=batch_size * num_batches * 2,
        skip_examples=skip_examples,
        num_workers=num_workers,
    )

    results = {
        "loss": [],
        "cost": [],
        "method": [],
        "decision": [],
        "text": [],
        "is_real_code": [],
    }

    assert isinstance(ttt_model, TTTTransformerLM)

    # JIT-compiled functions for Oracle
    @nnx.jit
    def jit_skip_loss(model, input_ids, attention_mask, position_ids):
        out = model(input_ids, attention_mask=attention_mask, position_ids=position_ids, use_ttt=False)
        return cross_entropy_loss(out["logits"][:, :-1], input_ids[:, 1:], attention_mask[:, 1:])

    @nnx.jit
    def jit_update_loss(model, input_ids, attention_mask, position_ids, gating_scale):
        out = model(input_ids, attention_mask=attention_mask, position_ids=position_ids, use_ttt=True, gating_scale=gating_scale)
        return cross_entropy_loss(out["logits"][:, :-1], input_ids[:, 1:], attention_mask[:, 1:])

    for i, batch in enumerate(tqdm(data_iter, total=num_batches, desc=method_name)):
        if i >= num_batches:
            break

        chunks = batch["chunks"]
        masks = batch["chunk_attention_mask"]
        num_chunks = chunks.shape[1]

        # Step 1: Compute advantage for all chunks in this batch
        chunk_data = []

        for c_idx in range(num_chunks):
            chunk_batch = {
                "input_ids": chunks[:, c_idx],
                "attention_mask": masks[:, c_idx]
            }

            chunk_len = chunk_batch["input_ids"].shape[-1]
            position_ids = jnp.arange(chunk_len, dtype=jnp.int32)
            position_ids = position_ids + c_idx * chunk_len
            position_ids = jnp.broadcast_to(position_ids, chunk_batch["input_ids"].shape)

            # Compute SKIP loss (JIT)
            loss_skip = float(jit_skip_loss(
                ttt_model,
                chunk_batch["input_ids"],
                chunk_batch["attention_mask"],
                position_ids,
            ))

            # Compute UPDATE loss (JIT)
            gating_scale = jnp.array([[1.0]], dtype=jnp.float32)
            loss_update = float(jit_update_loss(
                ttt_model,
                chunk_batch["input_ids"],
                chunk_batch["attention_mask"],
                position_ids,
                gating_scale,
            ))

            # Advantage: how much better is UPDATE vs SKIP
            advantage = loss_skip - loss_update

            # Decode text
            try:
                text = tokenizer.decode(chunk_batch["input_ids"][0], skip_special_tokens=False)
            except Exception:
                text = "[Decode Error]"

            # Valid ratio for is_real_code
            valid_ratio = float(jnp.sum(chunk_batch["attention_mask"])) / chunk_len
            is_real_code = valid_ratio > 0.1

            chunk_data.append({
                "c_idx": c_idx,
                "loss_skip": loss_skip,
                "loss_update": loss_update,
                "advantage": advantage,
                "text": text,
                "is_real_code": is_real_code,
            })

        # Step 2: Select top-k% chunks by advantage (Oracle selection)
        num_to_update = max(1, int(len(chunk_data) * update_rate))
        sorted_chunks = sorted(chunk_data, key=lambda x: x["advantage"], reverse=True)

        update_indices = set(c["c_idx"] for c in sorted_chunks[:num_to_update])

        # Step 3: Record results based on Oracle decision
        for c in chunk_data:
            if c["c_idx"] in update_indices:
                # Oracle says UPDATE
                results["loss"].append(c["loss_update"])
                results["cost"].append(3.0)
                results["decision"].append("UPDATE")
            else:
                # Oracle says SKIP
                results["loss"].append(c["loss_skip"])
                results["cost"].append(1.0)
                results["decision"].append("SKIP")

            results["method"].append(method_name)
            results["text"].append(c["text"])
            results["is_real_code"].append(c["is_real_code"])

    return pd.DataFrame(results)


def parse_args():
    parser = argparse.ArgumentParser(description="Compare optimization methods")
    parser.add_argument("--model_scale", type=str, default="125m", choices=["125m", "350m", "1b"])
    parser.add_argument("--budget", type=float, default=2.0, help="Target budget (avg steps)")
    parser.add_argument("--num_eval_batches", type=int, default=20, help="Number of batches for evaluation")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (must be 1 for dynamic gating evaluation)")
    parser.add_argument("--diff_checkpoint", type=str, help="Path to differentiable gating checkpoint (optional)")
    parser.add_argument("--binary_gating_checkpoint", type=str, help="Path to Binary Gating (Hard Skip) checkpoint (optional)")
    parser.add_argument("--update1_checkpoint", type=str, help="Path to UPDATE_1 baseline checkpoint (optional)")
    parser.add_argument("--output_dir", type=str, default="outputs/comparison")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--language", type=str, default="Python", help="Programming language for OOD testing")
    parser.add_argument("--split", type=str, default="train", help="Dataset split (train/validation/test). Note: The Stack v2 only has 'train'.")
    parser.add_argument("--skip_examples", type=int, default=0, help="Number of examples to skip (for held-out evaluation).")
    parser.add_argument("--num_workers", type=int, default=32, help="Number of parallel workers for data downloading")
    parser.add_argument("--hard_skip_threshold", type=float, default=0.1, help="Hard Skip threshold: skip TTT if gating scale < threshold (default: 0.1)")
    parser.add_argument("--use_checkpoint_threshold", action="store_true", help="Use probability threshold from checkpoint; otherwise auto-calibrate from budget")
    return parser.parse_args()


def evaluate_model(
    method_name: str,
    model_scale: str,
    budget_target: float,
    num_batches: int,
    batch_size: int,
    seed: int,
    gating_net: Optional[GatingNetwork | BinaryGatingNetwork] = None,
    model: Optional[TTTModel] = None,
    fixed_action: str = "SKIP",
    language: str = "Python",
    split: str = "test",
    skip_examples: int = 0,
    num_workers: int = 32,
    hard_skip_threshold: float = 0.1,
    binary_threshold: Optional[float] = None,  # Optional probability threshold; None -> auto from budget
    random_update_rate: Optional[float] = None,  # For Random Skip baseline (0.0-1.0); set explicitly
):
    # Convert budget (cost multiplier) to target update rate using cost model:
    # cost = 1 + 2 * update_rate => update_rate = (budget - 1) / 2
    target_update_rate: Optional[float] = None
    if budget_target is not None:
        target_update_rate = (budget_target - 1.0) / 2.0
        target_update_rate = float(min(max(target_update_rate, 0.0), 1.0))

    # NOTE: random_update_rate should only be set explicitly by callers.
    # Leaving it untouched avoids accidentally turning fixed-action baselines
    # (e.g., UPDATE_1) into random-skip evaluations when a budget is provided.

    if gating_net is not None:
        assert batch_size == 1, "Batch size must be 1 for dynamic gating evaluation (mixed SKIP/TTT strategies)."

    skip_info = f", skip={skip_examples}" if skip_examples > 0 else ""
    print(f"\nEvaluating {method_name} on {language} ({split}{skip_info})...")

    model_name = {"125m": "gpt2", "350m": "gpt2-medium", "1b": "gpt2-large"}[model_scale]
    tokenizer = get_tokenizer(model_name)

    # Load TTT Model if not provided
    if model is None:
        ttt_model, _ = load_ttt_model(
            model_name=model_name,
            fast_weight_type="ttt",
            seed=seed,
            load_pretrained=True,
            vocab_size=tokenizer.get_vocab_size(),
        )
    else:
        ttt_model = model
    # Ensure eval mode for inference (dropout off, deterministic)
    if hasattr(ttt_model, "eval"):
        ttt_model.eval()

    # Initialize feature extractor
    pad_id = tokenizer.token_to_id("<|pad|>")
    if pad_id is None:
        pad_id = -1

    feature_extractor = FeatureExtractor(
        vocab_size=tokenizer.get_vocab_size(),
        pad_token_id=pad_id,
        seq_length_norm=512,
    )

    data_iter = create_data_iterator(
        tokenizer=tokenizer,
        split=split,
        language=language,
        batch_size=batch_size,
        seq_length=1024,
        chunk_size=512,
        max_examples=batch_size * num_batches * 2,
        skip_examples=skip_examples,
        num_workers=num_workers,
    )

    results = {
        "loss": [],
        "cost": [],
        "method": [],
        "decision": [],
        "text": [],
        "is_real_code": [],  # True if chunk has >10% valid tokens
    }

    # For aggregate correlation analysis (Binary Gating only)
    all_update_probs: list[float] = []
    all_advantages: list[float] = []

    # RNG key for stochastic evaluation
    rng_key = jax.random.PRNGKey(seed)

    # Evaluation Loop
    for i, batch in enumerate(tqdm(data_iter, total=num_batches, desc=method_name)):
        if i >= num_batches:
            break

        chunks = batch["chunks"]
        masks = batch["chunk_attention_mask"]
        num_chunks = chunks.shape[1]
        current_spend = 0.0

        feature_extractor.reset_history()
        # Buffer to defer binary gating decisions until we choose a threshold for this sequence
        binary_chunk_buffer: list[dict] = []

        for c_idx in range(num_chunks):
            chunk_batch = {
                "input_ids": chunks[:, c_idx],
                "attention_mask": masks[:, c_idx]
            }
            
            # Decode text for qualitative analysis
            try:
                text = tokenizer.decode(chunk_batch["input_ids"][0], skip_special_tokens=False)
            except Exception:
                text = "[Decode Error]"

            chunk_len = chunk_batch["input_ids"].shape[-1]
            position_ids = jnp.arange(chunk_len, dtype=jnp.int32)
            position_ids = position_ids + c_idx * chunk_len
            position_ids = jnp.broadcast_to(position_ids, chunk_batch["input_ids"].shape)
            # Determine if this is a real code chunk (>10% valid tokens)
            valid_ratio = float(jnp.sum(chunk_batch["attention_mask"])) / chunk_len
            is_real_code = valid_ratio > 0.1

            # Budget Feature: Use 1.0 to match training (training hardcoded budget_remaining=1.0)
            # Note: The gating network was trained with budget_remaining=1.0 constant,
            # so evaluation must use the same value to avoid distribution shift.
            budget_rem = 1.0  # Match training

            assert isinstance(ttt_model, TTTTransformerLM)
            # 1. Extract Features (using JIT)
            logits_base, features = jit_base_forward_and_features(
                ttt_model,
                chunk_batch["input_ids"],
                chunk_batch["attention_mask"],
                position_ids,
                budget_remaining=budget_rem,
                diff_ema=feature_extractor.difficulty_ema,
                diff_sq_ema=feature_extractor.difficulty_sq_ema,
                cost_ema=feature_extractor.cost_ema,
                vocab_size=feature_extractor.vocab_size,
                pad_token_id=feature_extractor.pad_token_id if feature_extractor.pad_token_id is not None else -1,
                seq_norm=feature_extractor.seq_length_norm,
            )

            # Decision
            cost = 1.0
            loss = 0.0
            decision_str = ""

            if gating_net is None:
                if random_update_rate is not None:
                    if fixed_action != "SKIP":
                        raise ValueError("random_update_rate is only supported for SKIP/random baselines")

                    # Random Skip Baseline: randomly decide SKIP/UPDATE based on update_rate
                    rng_key, subkey = jax.random.split(rng_key)
                    do_update = jax.random.uniform(subkey) < random_update_rate

                    if do_update:
                        gating_scale = jnp.array([[1.0]], dtype=jnp.float32)
                        loss = float(jit_eval_with_scale(
                            ttt_model,
                            chunk_batch["input_ids"],
                            chunk_batch["attention_mask"],
                            position_ids,
                            gating_scale
                        ))
                        cost = 3.0
                        decision_str = "UPDATE"
                    else:
                        cost = 1.0
                        loss = float(jit_loss_from_logits(
                            logits_base,
                            chunk_batch["input_ids"],
                            chunk_batch["attention_mask"]
                        ))
                        decision_str = "SKIP"
                elif fixed_action == "SKIP":
                    # Fixed Baseline: SKIP
                    cost = 1.0
                    loss = float(jit_loss_from_logits(
                        logits_base,
                        chunk_batch["input_ids"],
                        chunk_batch["attention_mask"]
                    ))
                    decision_str = "SKIP"
                elif fixed_action == "UPDATE_1":
                    # Fixed Baseline: UPDATE_1
                    gating_scale = jnp.array([[1.0]], dtype=jnp.float32)
                    loss = float(jit_eval_with_scale(
                        ttt_model,
                        chunk_batch["input_ids"],
                        chunk_batch["attention_mask"],
                        position_ids,
                        gating_scale
                    ))
                    cost = 3.0
                    decision_str = "UPDATE"
                else:
                    raise ValueError(f"Unknown fixed_action: {fixed_action}")

            elif isinstance(gating_net, BinaryGatingNetwork):
                # Defer decision until we know the threshold for this sequence
                _, _, soft_probs, _, gating_logits = gating_net(features, train=False, return_logits=True)
                update_prob_val = float(soft_probs[0, 1])

                # Debug: print first 3 samples' features and outputs
                if i == 0 and c_idx < 3:
                    print(f"\n  [DEBUG] Batch {i}, Chunk {c_idx}:")
                    print(f"    features (first 10): {features[0, :10]}")
                    print(f"    features (mean, std): ({float(jnp.mean(features)):.4f}, {float(jnp.std(features)):.4f})")
                    print(f"    gating_logits: [{float(gating_logits[0, 0]):.4f}, {float(gating_logits[0, 1]):.4f}]")
                    print(f"    update_prob: {update_prob_val:.4f}")

                # Precompute both paths
                skip_loss = float(jit_loss_from_logits(
                    logits_base,
                    chunk_batch["input_ids"],
                    chunk_batch["attention_mask"]
                ))
                update_loss = float(jit_eval_with_scale(
                    ttt_model,
                    chunk_batch["input_ids"],
                    chunk_batch["attention_mask"],
                    position_ids,
                    jnp.array([[1.0]], dtype=jnp.float32),
                ))

                binary_chunk_buffer.append({
                    "update_prob": update_prob_val,
                    "skip_loss": skip_loss,
                    "update_loss": update_loss,
                    "text": text,
                    "is_real_code": is_real_code,
                })
                continue  # Defer appending results until after threshold selection

            elif isinstance(gating_net, GatingNetwork):
                # Continuous scale
                scale = float(jit_continuous_scale(gating_net, features)[0, 0])
                scale = max(0.0, scale)

                # Hard Skip check
                if scale < hard_skip_threshold:
                    cost = 1.0
                    loss = float(jit_loss_from_logits(
                        logits_base,
                        chunk_batch["input_ids"],
                        chunk_batch["attention_mask"]
                    ))
                    decision_str = "SKIP"
                else:
                    gating_scale = jnp.array([[scale]], dtype=jnp.float32)
                    loss = float(jit_eval_with_scale(
                        ttt_model,
                        chunk_batch["input_ids"],
                        chunk_batch["attention_mask"],
                        position_ids,
                        gating_scale
                    ))
                    cost = 3.0
                    decision_str = "UPDATE"

            results["loss"].append(loss)
            results["cost"].append(cost)
            results["method"].append(method_name)
            results["decision"].append(decision_str)
            results["text"].append(text)
            results["is_real_code"].append(is_real_code)

            current_spend += cost
            feature_extractor.update_history(loss, cost)

        # If we deferred binary gating decisions, finalize them now
        if isinstance(gating_net, BinaryGatingNetwork) and binary_chunk_buffer:
            update_probs = [c["update_prob"] for c in binary_chunk_buffer]
            advantages = [c["skip_loss"] - c["update_loss"] for c in binary_chunk_buffer]

            # Collect for aggregate analysis
            all_update_probs.extend(update_probs)
            all_advantages.extend(advantages)

            # Diagnostic: Check correlation between update_prob and advantage
            if len(update_probs) > 1:
                corr = np.corrcoef(update_probs, advantages)[0, 1]
                if i == 0:  # Print only for first batch
                    print(f"\n  [Diagnostic] Batch {i}: update_prob vs advantage correlation = {corr:.4f}")
                    print(f"    update_prob: min={min(update_probs):.4f}, max={max(update_probs):.4f}, mean={np.mean(update_probs):.4f}")
                    print(f"    advantage:   min={min(advantages):.4f}, max={max(advantages):.4f}, mean={np.mean(advantages):.4f}")

            # Choose threshold
            threshold_to_use: float
            if binary_threshold is not None and 0.0 <= binary_threshold <= 1.0:
                threshold_to_use = binary_threshold
            elif target_update_rate is not None:
                threshold_to_use = float(np.percentile(np.asarray(update_probs), 100 * (1 - target_update_rate)))
            else:
                threshold_to_use = 0.5

            for c in binary_chunk_buffer:
                decision = c["update_prob"] >= threshold_to_use
                loss = c["update_loss"] if decision else c["skip_loss"]
                cost = 3.0 if decision else 1.0
                decision_str = "UPDATE" if decision else "SKIP"

                results["loss"].append(loss)
                results["cost"].append(cost)
                results["method"].append(method_name)
                results["decision"].append(decision_str)
                results["text"].append(c["text"])
                results["is_real_code"].append(c["is_real_code"])

                current_spend += cost
                feature_extractor.update_history(loss, cost)

    # Print aggregate correlation for Binary Gating
    if all_update_probs and all_advantages and len(all_update_probs) > 1:
        probs_arr = np.array(all_update_probs)
        advs_arr = np.array(all_advantages)

        agg_corr = np.corrcoef(probs_arr, advs_arr)[0, 1]
        print(f"\n  [AGGREGATE] update_prob vs advantage correlation = {agg_corr:.4f} (n={len(all_update_probs)})")
        print(f"    update_prob: min={probs_arr.min():.4f}, max={probs_arr.max():.4f}, mean={probs_arr.mean():.4f}, std={probs_arr.std():.4f}")
        print(f"    advantage:   min={advs_arr.min():.4f}, max={advs_arr.max():.4f}, mean={advs_arr.mean():.4f}, std={advs_arr.std():.4f}")

        # Check ranking quality: what fraction of top-k by update_prob are also top-k by advantage?
        k = len(all_update_probs) // 2  # Top 50%
        prob_sorted_indices = np.argsort(probs_arr)
        adv_sorted_indices = np.argsort(advs_arr)

        prob_topk_indices = set(prob_sorted_indices[-k:])  # Top k by prob (highest)
        adv_topk_indices = set(adv_sorted_indices[-k:])    # Top k by advantage (highest)
        overlap = len(prob_topk_indices & adv_topk_indices) / k
        print(f"    Top-50% ranking overlap (gating vs oracle): {overlap:.2%}")

        # DETAILED DEBUG: Verify the logic is correct
        print(f"\n    [DEBUG] Sanity check:")
        print(f"      Total samples: {len(probs_arr)}, k (50%): {k}")
        print(f"      prob_topk_indices: first 5 = {list(prob_topk_indices)[:5]}, count = {len(prob_topk_indices)}")
        print(f"      adv_topk_indices:  first 5 = {list(adv_topk_indices)[:5]}, count = {len(adv_topk_indices)}")
        print(f"      intersection count: {len(prob_topk_indices & adv_topk_indices)}")

        # Verify: for samples in prob top-k, what's their average advantage rank?
        adv_ranks = np.argsort(np.argsort(advs_arr))  # Rank of each sample by advantage
        prob_topk_avg_adv_rank = np.mean([adv_ranks[i] for i in prob_topk_indices])
        print(f"      Avg advantage rank of prob top-k samples: {prob_topk_avg_adv_rank:.1f} (expected > {len(probs_arr)/2:.1f} if working)")

        # Print first 10 samples sorted by prob
        print(f"\n    [DEBUG] First 10 samples by descending update_prob:")
        for rank, idx in enumerate(prob_sorted_indices[::-1][:10]):
            print(f"      rank={rank}, idx={idx}, prob={probs_arr[idx]:.4f}, adv={advs_arr[idx]:.4f}, adv_rank={adv_ranks[idx]}")

        # Print first 10 samples sorted by advantage
        print(f"\n    [DEBUG] First 10 samples by descending advantage:")
        for rank, idx in enumerate(adv_sorted_indices[::-1][:10]):
            print(f"      rank={rank}, idx={idx}, prob={probs_arr[idx]:.4f}, adv={advs_arr[idx]:.4f}")

        # Compute RankAcc with margin (same as training) to verify consistency
        print(f"\n    [DEBUG] Pairwise ranking accuracy (same metric as training):")
        ranking_margin = 0.1
        diff_adv = advs_arr[:, None] - advs_arr[None, :]
        diff_score = probs_arr[:, None] - probs_arr[None, :]  # Use prob as score
        valid_pairs = diff_adv > ranking_margin
        num_valid_pairs = np.maximum(np.sum(valid_pairs), 1)
        correct_pairs = (diff_score > 0) * valid_pairs
        ranking_accuracy = np.sum(correct_pairs) / num_valid_pairs
        total_pairs = len(probs_arr) * (len(probs_arr) - 1)
        valid_pairs_ratio = num_valid_pairs / total_pairs

        print(f"      RankAcc (margin=0.1): {ranking_accuracy:.2%}")
        print(f"      Valid pairs ratio: {valid_pairs_ratio:.2%} ({num_valid_pairs}/{total_pairs})")

        # Also compute RankAcc without margin for comparison
        all_pairs = diff_adv > 0  # All pairs where adv_i > adv_j
        num_all_pairs = np.maximum(np.sum(all_pairs), 1)
        correct_all_pairs = (diff_score > 0) * all_pairs
        ranking_accuracy_no_margin = np.sum(correct_all_pairs) / num_all_pairs
        print(f"      RankAcc (no margin): {ranking_accuracy_no_margin:.2%}")

    return pd.DataFrame(results)


def main():
    args = parse_args()

    print("="*60)
    print("Comparison: Gating Methods vs Baseline")
    print("="*60)

    # 1. Initialize/Load Networks
    rngs = nnx.Rngs(args.seed)

    model_name = {"125m": "gpt2", "350m": "gpt2-medium", "1b": "gpt2-large"}[args.model_scale]
    tokenizer = get_tokenizer(model_name)
    vocab_size = tokenizer.get_vocab_size()

    # Differentiable Network (optional)
    diff_net: Optional[GatingNetwork] = None
    diff_ttt_model = None

    if args.diff_checkpoint:
        diff_net = GatingNetwork(
            config=GatingConfig(feature_dim=32, hidden_dim=64, scale_output=4.0),
            rngs=rngs
        )
        print(f"Loading Differentiable Gating checkpoint from {args.diff_checkpoint}...")

        diff_ttt_model, _ = load_ttt_model(
            model_name=model_name,
            fast_weight_type="ttt",
            seed=args.seed,
            load_pretrained=True,
            vocab_size=vocab_size,
        )

        ckpt = load_checkpoint(args.diff_checkpoint, target=None)

        if "state" in ckpt and "model" in ckpt["state"]:
            model_state = unwrap_state(ckpt["state"]["model"])

            # Direct component updates (avoid wrapper indirection issues)
            if "gating_net" in model_state:
                nnx.update(diff_net, model_state["gating_net"])
                print("  - gating_net updated")

            if "fast_layer" in model_state:
                nnx.update(diff_ttt_model.fast_layer, model_state["fast_layer"])
                print("  - fast_layer updated")

            if "fast_norm" in model_state:
                nnx.update(diff_ttt_model.fast_norm, model_state["fast_norm"])
                print("  - fast_norm updated")

            if "lm_head" in model_state and hasattr(diff_ttt_model, 'lm_head'):
                nnx.update(diff_ttt_model.lm_head, model_state["lm_head"])
                print("  - lm_head updated")

            print("Differentiable Gating and TTT weights loaded.")
        else:
            print("Warning: Could not find 'state.model' in checkpoint.")
    else:
        print("No Differentiable checkpoint supplied; skipping differentiable evaluation.")

    # Binary Gating Network (Hard Skip, trained with Gumbel-Softmax)
    binary_net: Optional[BinaryGatingNetwork] = None
    binary_ttt_model = None
    binary_threshold_ema: Optional[float] = None  # Advantage/prob threshold saved during training
    binary_prob_threshold_ema: Optional[float] = None  # Preferred probability threshold if available

    if args.binary_gating_checkpoint:
        binary_net = BinaryGatingNetwork(
            config=BinaryGatingConfig(feature_dim=32, hidden_dim=64, scale_when_update=1.0),
            rngs=rngs
        )
        print(f"Loading Binary Gating (Hard Skip) checkpoint from {args.binary_gating_checkpoint}...")

        binary_ttt_model, _ = load_ttt_model(
            model_name=model_name,
            fast_weight_type="ttt",
            seed=args.seed,
            load_pretrained=True,
            vocab_size=vocab_size,
        )

        ckpt = load_checkpoint(args.binary_gating_checkpoint, target=None)

        # Load thresholds from metadata for evaluation
        if "metadata" in ckpt:
            meta = ckpt["metadata"]
            if "prob_threshold_ema" in meta:
                binary_prob_threshold_ema = float(meta["prob_threshold_ema"])
                print(f"  Loaded prob_threshold_ema from metadata: {binary_prob_threshold_ema:.4f}")
            if "threshold_ema" in meta:
                binary_threshold_ema = float(meta["threshold_ema"])
                print(f"  Loaded threshold_ema from metadata (advantage space): {binary_threshold_ema:.4f}")

        if "state" in ckpt and "model" in ckpt["state"]:
            model_state = unwrap_state(ckpt["state"]["model"])
            print(f"  Checkpoint keys: {list(model_state.keys())}")

            # Verify weights change by capturing before/after
            fast_layer_before = float(jnp.mean(jnp.abs(binary_ttt_model.fast_layer.W1[...])))

            # Direct component updates (avoid wrapper indirection issues)
            # Update each trainable component directly from checkpoint state
            if "gating_net" in model_state:
                nnx.update(binary_net, model_state["gating_net"])
                bias_val = binary_net.head.bias[...] if binary_net.head.bias is not None else "None"
                print(f"  - gating_net updated (bias: {bias_val})")

            if "fast_layer" in model_state:
                nnx.update(binary_ttt_model.fast_layer, model_state["fast_layer"])
                fast_layer_after = float(jnp.mean(jnp.abs(binary_ttt_model.fast_layer.W1[...])))
                print(f"  - fast_layer updated (W1 mean abs: {fast_layer_before:.6f} -> {fast_layer_after:.6f})")

            if "fast_norm" in model_state:
                nnx.update(binary_ttt_model.fast_norm, model_state["fast_norm"])
                print("  - fast_norm updated")

            if "lm_head" in model_state and hasattr(binary_ttt_model, 'lm_head'):
                nnx.update(binary_ttt_model.lm_head, model_state["lm_head"])
                print("  - lm_head updated")

            print("Binary Gating (Hard Skip) and TTT weights loaded.")

            # Quick sanity check: run inference on dummy data
            dummy_features = jnp.zeros((1, 32))
            _, _, soft_probs_zero, _, dummy_logits = binary_net(dummy_features, train=False, return_logits=True)
            print(f"  Sanity check (zero features): UPDATE prob = {float(soft_probs_zero[0, 1]):.4f}")
            print(f"  Sanity check (zero features): logits = [{float(dummy_logits[0, 0]):.4f}, {float(dummy_logits[0, 1]):.4f}]")

            # Print network weights for debugging
            print(f"  Gating net head.bias: {binary_net.head.bias[...]}")
            print(f"  Gating net head.kernel shape: {binary_net.head.kernel[...].shape}")
            print(f"  Gating net head.kernel mean abs: {float(jnp.mean(jnp.abs(binary_net.head.kernel[...]))):.6f}")
        else:
            print("Warning: Could not find 'state.model' in checkpoint.")
    else:
        print("No Binary Gating checkpoint supplied; skipping binary gating evaluation.")

    # UPDATE_1 Model (Fixed)
    update1_ttt_model = None
    if args.update1_checkpoint:
        print(f"Loading UPDATE_1 checkpoint from {args.update1_checkpoint}...")
        update1_ttt_model, _ = load_ttt_model(
            model_name=model_name,
            fast_weight_type="ttt",
            seed=args.seed,
            load_pretrained=True,
            vocab_size=vocab_size,
        )
        ckpt = load_checkpoint(args.update1_checkpoint, target=None)
        if "state" in ckpt and "model" in ckpt["state"]:
            nnx.update(update1_ttt_model, unwrap_state(ckpt["state"]["model"]))
            print("UPDATE_1 weights loaded.")
        else:
            print("Warning: Could not find 'state.model' in UPDATE_1 checkpoint.")

    all_results = []

    # Convert budget to target update rate (cost = 1 + 2 * update_rate)
    target_update_rate = max(0.0, min(1.0, (args.budget - 1.0) / 2.0))
    print(f"\nTarget update rate from budget={args.budget:.2f}: {target_update_rate*100:.1f}%")

    # 2. Evaluate Baselines (SKIP)
    df_skip = evaluate_model(
        "SKIP (Baseline)",
        args.model_scale,
        0.0,
        args.num_eval_batches,
        args.batch_size,
        args.seed,
        None,
        fixed_action="SKIP",
        language=args.language,
        split=args.split,
        skip_examples=args.skip_examples,
        num_workers=args.num_workers,
    )
    all_results.append(df_skip)

    # 3. Evaluate UPDATE_1 (Fixed)
    if update1_ttt_model is not None:
        df_update1 = evaluate_model(
            "UPDATE_1 (Fixed)",
            args.model_scale,
            0.0,
            args.num_eval_batches,
            args.batch_size,
            args.seed,
            None,
            model=update1_ttt_model,
            fixed_action="UPDATE_1",
            language=args.language,
            split=args.split,
            skip_examples=args.skip_examples,
            num_workers=args.num_workers,
        )
        all_results.append(df_update1)

    # 4. Evaluate Differentiable (with Hard Skip)
    if diff_net is not None:
        df_diff = evaluate_model(
            "Differentiable (Hard Skip)",
            args.model_scale,
            args.budget,
            args.num_eval_batches,
            args.batch_size,
            args.seed,
            diff_net,
            model=diff_ttt_model,
            language=args.language,
            split=args.split,
            skip_examples=args.skip_examples,
            num_workers=args.num_workers,
            hard_skip_threshold=args.hard_skip_threshold,
        )
        all_results.append(df_diff)

    # 5. Evaluate Binary Gating (Hard Skip with Gumbel-Softmax)
    binary_update_rate = None
    if binary_net is not None:
        # Prefer probability-space threshold from checkpoint only if explicitly requested.
        eval_threshold = None
        if args.use_checkpoint_threshold:
            if binary_prob_threshold_ema is not None and 0.0 <= binary_prob_threshold_ema <= 1.0:
                eval_threshold = binary_prob_threshold_ema
                print(f"\n=== Using probability threshold from checkpoint: {eval_threshold:.4f} ===")
            elif binary_threshold_ema is not None and 0.0 <= binary_threshold_ema <= 1.0:
                eval_threshold = binary_threshold_ema
                print(f"\n=== Using probability threshold from checkpoint (legacy): {eval_threshold:.4f} ===")
            else:
                print("\n=== No valid probability threshold in checkpoint; falling back to auto-calibrated threshold from budget ===")
        else:
            print("\n=== Using auto-calibrated threshold from budget (per-sequence percentile) ===")

        df_binary = evaluate_model(
            "Binary Gating (Hard Skip)",
            args.model_scale,
            args.budget,
            args.num_eval_batches,
            args.batch_size,
            args.seed,
            binary_net,
            model=binary_ttt_model,
            language=args.language,
            split=args.split,
            skip_examples=args.skip_examples,
            num_workers=args.num_workers,
            binary_threshold=eval_threshold,
        )
        all_results.append(df_binary)

        # Calculate actual update rate from Binary Gating
        binary_update_rate = (df_binary["decision"] == "UPDATE").mean()
        print(f"\n=== Binary Gating Update Rate: {binary_update_rate:.2%} ===")

        # Print diagnostic info about gating predictions
        if "update_probs" in df_binary.columns:
            probs = df_binary["update_probs"].values
            print(f"    UPDATE prob distribution: min={np.min(probs):.4f}, max={np.max(probs):.4f}, "
                  f"mean={np.mean(probs):.4f}, std={np.std(probs):.4f}")
            print(f"    Prob percentiles: 10th={np.percentile(probs, 10):.4f}, "
                  f"50th={np.percentile(probs, 50):.4f}, 90th={np.percentile(probs, 90):.4f}")

    # 6. Evaluate Random Skip Baseline (same target update rate as budget)
    # IMPORTANT: Use binary_ttt_model (same TTT weights as Binary Gating) for fair comparison
    if binary_update_rate is not None and binary_ttt_model is not None:
        print(f"\n=== Running Random Skip Baseline (target update_rate={target_update_rate:.2%}) ===")
        print("    (Using same TTT weights as Binary Gating for fair comparison)")
        df_random = evaluate_model(
            f"Random Skip ({target_update_rate:.0%} update)",
            args.model_scale,
            args.budget,
            args.num_eval_batches,
            args.batch_size,
            args.seed + 1,  # Different seed for random decisions
            gating_net=None,
            model=binary_ttt_model,  # Use same TTT weights as Binary Gating
            language=args.language,
            split=args.split,
            skip_examples=args.skip_examples,
            num_workers=args.num_workers,
            random_update_rate=target_update_rate,
        )
        all_results.append(df_random)

        # 6b. Evaluate Oracle Baseline (upper bound)
        print(f"\n=== Running Oracle Baseline (update_rate={target_update_rate:.2%}) ===")
        print("    (Using same TTT weights as Binary Gating for fair comparison)")
        df_oracle = evaluate_oracle(
            f"Oracle ({target_update_rate:.0%} update)",
            args.model_scale,
            update_rate=target_update_rate,
            num_batches=args.num_eval_batches,
            batch_size=args.batch_size,
            seed=args.seed,
            model=binary_ttt_model,  # Use same TTT weights as Binary Gating
            language=args.language,
            split=args.split,
            skip_examples=args.skip_examples,
            num_workers=args.num_workers,
        )
        all_results.append(df_oracle)

    # 7. Visualize & Report
    full_df = pd.concat(all_results)

    print("\n=== Final Results (All Chunks) ===")
    summary = cast(pd.DataFrame, full_df.groupby("method").agg({"loss": "mean", "cost": "mean"})).sort_values(by="loss")
    print(summary)

    # Real code only analysis
    real_code_df = full_df[full_df["is_real_code"]]
    print("\n=== Final Results (Real Code Only, excluding padding) ===")
    summary_real = cast(pd.DataFrame, real_code_df.groupby("method").agg({"loss": "mean", "cost": "mean"})).sort_values(by="loss")
    print(summary_real)

    # Skip rate on real code for Binary Gating
    if "Binary Gating (Hard Skip)" in full_df["method"].values:
        binary_real = real_code_df[real_code_df["method"] == "Binary Gating (Hard Skip)"]
        if len(binary_real) > 0:
            skip_rate_real = (binary_real["decision"] == "SKIP").mean()
            print(f"\n=== Binary Gating Skip Rate on Real Code: {skip_rate_real:.2%} ===")

    # Random Skip vs Binary Gating vs Oracle comparison (Ablation)
    random_method = [m for m in summary.index if m.startswith("Random Skip")]
    oracle_method = [m for m in summary.index if m.startswith("Oracle")]

    if random_method and "Binary Gating (Hard Skip)" in summary.index:
        random_name = random_method[0]
        random_loss = float(summary.loc[random_name, "loss"])
        binary_loss = float(summary.loc["Binary Gating (Hard Skip)", "loss"])
        random_cost = float(summary.loc[random_name, "cost"])
        binary_cost = float(summary.loc["Binary Gating (Hard Skip)", "cost"])

        improvement_vs_random = (random_loss - binary_loss) / random_loss * 100

        print("\n" + "=" * 70)
        print("ABLATION: Random Skip vs Learned Gating vs Oracle")
        print("=" * 70)
        print(f"{'Method':<25} {'Loss':>10} {'Cost':>10} {'vs Random':>12}")
        print("-" * 70)
        print(f"{'Random Skip':<25} {random_loss:>10.4f} {random_cost:>10.2f}x {'baseline':>12}")
        print(f"{'Binary Gating (Learned)':<25} {binary_loss:>10.4f} {binary_cost:>10.2f}x {improvement_vs_random:>+11.2f}%")

        if oracle_method:
            oracle_name = oracle_method[0]
            oracle_loss = float(summary.loc[oracle_name, "loss"])
            oracle_cost = float(summary.loc[oracle_name, "cost"])
            improvement_oracle = (random_loss - oracle_loss) / random_loss * 100

            print(f"{'Oracle (Upper Bound)':<25} {oracle_loss:>10.4f} {oracle_cost:>10.2f}x {improvement_oracle:>+11.2f}%")
            print("-" * 70)

            # How much of the Oracle improvement does Binary Gating capture?
            gap_random_to_oracle = random_loss - oracle_loss
            gap_random_to_binary = random_loss - binary_loss
            capture_rate = (gap_random_to_binary / gap_random_to_oracle * 100) if gap_random_to_oracle > 0 else 0

            print(f"\nOracle Gap:    Random → Oracle = {gap_random_to_oracle:.4f}")
            print(f"Learned Gap:   Random → Binary = {gap_random_to_binary:.4f}")
            print(f"Capture Rate:  {capture_rate:.1f}% of Oracle improvement")

            print("\n" + "-" * 70)
            if gap_random_to_oracle < 0.01:
                print("CONCLUSION: Oracle provides NO improvement over Random.")
                print("            'Where to update' does NOT matter for this task.")
            elif capture_rate > 80:
                print("CONCLUSION: Learned gating captures most of Oracle improvement!")
                print("            Gating network is working well.")
            elif capture_rate > 50:
                print("CONCLUSION: Learned gating captures partial Oracle improvement.")
                print("            Room for improvement in gating network.")
            elif capture_rate > 20:
                print("CONCLUSION: Learned gating captures little Oracle improvement.")
                print("            Gating network needs significant improvement.")
            else:
                print("CONCLUSION: Learned gating fails to capture Oracle improvement.")
                print("            Gating training may be broken.")
        else:
            print("-" * 70)
            if improvement_vs_random > 5:
                print("CONCLUSION: Learned gating provides meaningful improvement over random.")
            elif improvement_vs_random > 0:
                print("CONCLUSION: Learned gating provides marginal improvement over random.")
            else:
                print("CONCLUSION: Learned gating provides NO improvement over random skip!")

        print("=" * 70)

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    summary.to_csv(output_path / "summary.csv")
    summary_real.to_csv(output_path / "summary_real_code.csv")
    full_df.to_csv(output_path / "detailed_results.csv")

    # Plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=full_df, x="cost", y="loss", hue="method", alpha=0.6)
    if "SKIP (Baseline)" in summary.index:
        baseline_loss = cast(float, summary.loc["SKIP (Baseline)", "loss"])
        plt.axhline(y=baseline_loss, color='gray', linestyle='--', label="Baseline Loss")
    plt.title(f"Cost-Quality Tradeoff (Budget ~{args.budget}x)")
    plt.savefig(output_path / "tradeoff_plot.png")
    print(f"\nPlots saved to {output_path}")

if __name__ == "__main__":
    main()
