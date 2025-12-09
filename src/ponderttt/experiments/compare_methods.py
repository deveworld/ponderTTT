"""
Compare gating methods: SKIP, UPDATE_1, Random Skip, Oracle, TTT Improvement Gating.

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
from ..utils import cross_entropy_loss
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

@nnx.jit
def jit_ttt_forward_with_stats(
    model: TTTTransformerLM,
    input_ids: jax.Array,
    attention_mask: jax.Array,
    position_ids: jax.Array,
):
    """Run TTT forward and return loss + ttt_stats (for TTT Improvement gating).

    Also computes SKIP loss in the same JIT call for consistent comparison.
    """
    # SKIP path (no TTT)
    out_skip = model(
        input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_ttt=False,
    )
    loss_skip = cross_entropy_loss(
        out_skip["logits"][:, :-1],
        input_ids[:, 1:],
        attention_mask[:, 1:],
    )

    # UPDATE path (with TTT)
    out_update = model(
        input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_ttt=True,
    )
    loss_update = cross_entropy_loss(
        out_update["logits"][:, :-1],
        input_ids[:, 1:],
        attention_mask[:, 1:],
    )

    # TTT internal stats (take mean across heads/batch)
    ttt_stats = out_update.get("ttt_stats", {})
    ttt_loss_step_0 = ttt_stats.get("ttt_loss_step_0", jnp.array(0.0))
    ttt_loss_step_1 = ttt_stats.get("ttt_loss_step_1", jnp.array(0.0))

    # Ensure scalars by taking mean (TTT stats are per-head arrays)
    ttt_loss_step_0 = jnp.mean(ttt_loss_step_0)
    ttt_loss_step_1 = jnp.mean(ttt_loss_step_1)

    return loss_skip, loss_update, ttt_loss_step_0, ttt_loss_step_1


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


def evaluate_ttt_improvement_gating(
    method_name: str,
    model_scale: str,
    update_rate: float,
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
    TTT Improvement-based Gating: Use TTT internal self-supervision loss improvement
    as the gating signal.

    Key insight: ttt_improvement = ttt_loss_step_0 - ttt_loss_step_1 strongly correlates
    with oracle advantage (ρ=0.689), so we can use it for threshold-based gating.

    For each chunk:
    1. Run TTT forward to get ttt_stats
    2. Compute ttt_improvement = ttt_loss_step_0 - ttt_loss_step_1
    3. Select top-k% chunks by ttt_improvement for UPDATE, rest SKIP

    Note: This requires running TTT for all chunks to measure improvement,
    so the decision cost is 3.0x per chunk (same as always-update).
    The benefit is in the final loss: we only "count" UPDATE loss for selected chunks.
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

    # For correlation analysis
    all_ttt_improvements: list[float] = []
    all_advantages: list[float] = []

    assert isinstance(ttt_model, TTTTransformerLM)

    # State leakage detection
    def get_fast_weight_checksum(model) -> float:
        w1 = model.fast_layer.W1[...]
        b1 = model.fast_layer.b1[...]
        return float(jnp.sum(w1) + jnp.sum(b1))

    initial_checksum = get_fast_weight_checksum(ttt_model)

    for i, batch in enumerate(tqdm(data_iter, total=num_batches, desc=method_name)):
        if i >= num_batches:
            break

        chunks = batch["chunks"]
        masks = batch["chunk_attention_mask"]
        num_chunks = chunks.shape[1]

        # Step 1: Compute TTT improvement for all chunks
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

            # Compute both SKIP and UPDATE losses + TTT stats in one JIT call
            # This ensures consistent results (same as analyze_gradient_norm.py)
            loss_skip, loss_update, ttt_step_0, ttt_step_1 = jit_ttt_forward_with_stats(
                ttt_model,
                chunk_batch["input_ids"],
                chunk_batch["attention_mask"],
                position_ids,
            )
            loss_skip = float(loss_skip)
            loss_update = float(loss_update)

            # TTT improvement: how much did TTT self-supervision loss decrease?
            ttt_step_0_val = float(ttt_step_0)
            ttt_step_1_val = float(ttt_step_1)
            ttt_improvement = ttt_step_0_val - ttt_step_1_val

            # Oracle advantage (for analysis)
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
                "ttt_improvement": ttt_improvement,
                "advantage": advantage,
                "text": text,
                "is_real_code": is_real_code,
            })

            all_ttt_improvements.append(ttt_improvement)
            all_advantages.append(advantage)

        # Step 2: Select top-k% chunks by TTT improvement
        num_to_update = max(1, int(len(chunk_data) * update_rate))
        sorted_chunks = sorted(chunk_data, key=lambda x: x["ttt_improvement"], reverse=True)

        update_indices = set(c["c_idx"] for c in sorted_chunks[:num_to_update])

        # Step 3: Record results based on TTT improvement decision
        for c in chunk_data:
            if c["c_idx"] in update_indices:
                # TTT Improvement says UPDATE
                results["loss"].append(c["loss_update"])
                results["cost"].append(3.0)
                results["decision"].append("UPDATE")
            else:
                # TTT Improvement says SKIP
                results["loss"].append(c["loss_skip"])
                results["cost"].append(1.0)
                results["decision"].append("SKIP")

            results["method"].append(method_name)
            results["text"].append(c["text"])
            results["is_real_code"].append(c["is_real_code"])

    # State leakage check
    final_checksum = get_fast_weight_checksum(ttt_model)
    if abs(final_checksum - initial_checksum) > 1e-6:
        print(f"\n  ⚠️  [STATE LEAKAGE] Fast weights changed during evaluation!")
        print(f"    Initial: {initial_checksum:.6f}, Final: {final_checksum:.6f}")
    else:
        print(f"\n  ✓ [State Check] No state leakage (fast weights unchanged)")

    # Print correlation analysis
    if all_ttt_improvements and all_advantages:
        from scipy import stats as scipy_stats
        ttt_arr = np.array(all_ttt_improvements)
        adv_arr = np.array(all_advantages)

        pearson_r, _ = scipy_stats.pearsonr(ttt_arr, adv_arr)
        spearman_r, _ = scipy_stats.spearmanr(ttt_arr, adv_arr)

        print(f"\n  [TTT Improvement Gating] Correlation with Oracle:")
        print(f"    Pearson r:  {pearson_r:.4f}")
        print(f"    Spearman ρ: {spearman_r:.4f}")

        # Top-k overlap
        k = len(all_ttt_improvements) // 2
        ttt_sorted = np.argsort(ttt_arr)
        adv_sorted = np.argsort(adv_arr)
        ttt_topk = set(ttt_sorted[-k:])
        adv_topk = set(adv_sorted[-k:])
        overlap = len(ttt_topk & adv_topk) / k
        print(f"    Top-50% overlap with Oracle: {overlap:.2%}")

    return pd.DataFrame(results)


def evaluate_threshold_gating(
    method_name: str,
    model_scale: str,
    update_rate: float,
    num_batches: int,
    batch_size: int,
    language: str,
    split: str,
    skip_examples: int,
    num_workers: int,
    model: TTTTransformerLM | None = None,
    threshold_mode: str = "fixed",  # "fixed", "ema", "prob"
    initial_threshold: float | None = None,  # If None, calibrate from first batch
    ema_alpha: float = 0.1,
    prob_temperature: float = 1.0,
    seed: int = 42,
):
    """
    Walk Phase: Threshold-based online gating.

    Unlike top-k selection (Crawl phase), this makes per-chunk decisions
    using an adaptive threshold - suitable for streaming inference.

    Modes:
    - "fixed": Use a fixed threshold (calibrated from first batch or specified)
    - "ema": Exponential moving average threshold that adapts to target update rate
    - "prob": Probability-space threshold using sigmoid mapping

    Args:
        threshold_mode: Gating strategy ("fixed", "ema", "prob")
        initial_threshold: Starting threshold (if None, calibrate from first batch)
        ema_alpha: EMA decay rate for threshold adaptation
        prob_temperature: Temperature for sigmoid in probability mode
    """
    print(f"\nEvaluating {method_name} (mode={threshold_mode}) on {language}...")

    model_name = {"125m": "gpt2", "350m": "gpt2-medium", "1b": "gpt2-large"}[model_scale]
    tokenizer = get_tokenizer(model_name)

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

    assert isinstance(ttt_model, TTTTransformerLM)

    # Create data iterator
    data_iter = create_data_iterator(
        tokenizer=tokenizer,
        split=split,
        language=language,
        batch_size=batch_size,
        seq_length=1024,
        chunk_size=512,
        skip_examples=skip_examples,
        max_examples=batch_size * num_batches * 2,
        num_workers=num_workers,
    )

    initial_checksum = get_fast_weight_checksum(ttt_model)
    rng_key = jax.random.PRNGKey(seed)

    results: dict[str, list] = {
        "loss": [],
        "cost": [],
        "decision": [],
        "is_real_code": [],
        "method": [],
        "ttt_improvement": [],
        "advantage": [],
        "threshold_used": [],
    }

    # Threshold state
    threshold = initial_threshold
    update_count = 0
    total_count = 0

    # Calibration buffer (for fixed mode when threshold not specified)
    calibration_buffer: list[float] = []
    calibration_done = False

    for i, batch in enumerate(tqdm(data_iter, total=num_batches, desc=method_name)):
        if i >= num_batches:
            break

        chunks = batch["input_ids"]
        masks = batch["attention_mask"]
        num_chunks = chunks.shape[1]

        for c_idx in range(num_chunks):
            chunk_batch = {
                "input_ids": chunks[:, c_idx],
                "attention_mask": masks[:, c_idx],
            }
            chunk_len = chunk_batch["input_ids"].shape[-1]
            position_ids = jnp.arange(chunk_len, dtype=jnp.int32)
            position_ids = position_ids + c_idx * chunk_len
            position_ids = jnp.broadcast_to(position_ids, chunk_batch["input_ids"].shape)

            # Valid ratio for is_real_code
            valid_ratio = float(jnp.sum(chunk_batch["attention_mask"])) / chunk_len
            is_real_code = valid_ratio > 0.1

            # Compute TTT stats for gating decision
            loss_skip, loss_update, ttt_step_0, ttt_step_1 = jit_ttt_forward_with_stats(
                ttt_model,
                chunk_batch["input_ids"],
                chunk_batch["attention_mask"],
                position_ids,
            )

            loss_skip = float(loss_skip)
            loss_update = float(loss_update)
            ttt_step_0_val = float(ttt_step_0)
            ttt_step_1_val = float(ttt_step_1)
            ttt_improvement = ttt_step_0_val - ttt_step_1_val
            advantage = loss_skip - loss_update

            # Calibration phase (first batch only)
            if threshold is None and not calibration_done:
                calibration_buffer.append(ttt_improvement)
                if len(calibration_buffer) >= num_chunks * 2:  # Use first 2 batches
                    # Set threshold to achieve target update rate
                    sorted_improvements = sorted(calibration_buffer, reverse=True)
                    target_idx = int(len(sorted_improvements) * update_rate)
                    threshold = sorted_improvements[min(target_idx, len(sorted_improvements) - 1)]
                    calibration_done = True
                    print(f"  [Calibrated threshold: {threshold:.6f}]")

            # Make gating decision based on mode
            if threshold is None:
                # Still calibrating - use UPDATE for now
                do_update = True
            elif threshold_mode == "fixed":
                do_update = ttt_improvement > threshold
            elif threshold_mode == "ema":
                do_update = ttt_improvement > threshold
                # Adapt threshold to maintain target update rate
                total_count += 1
                if do_update:
                    update_count += 1
                actual_rate = update_count / total_count if total_count > 0 else update_rate
                # Adjust threshold: increase if updating too much, decrease if too little
                if actual_rate > update_rate:
                    threshold = threshold * (1 + ema_alpha * 0.1)
                else:
                    threshold = threshold * (1 - ema_alpha * 0.1)
            elif threshold_mode == "prob":
                # Probability-space gating
                rng_key, subkey = jax.random.split(rng_key)
                # Sigmoid mapping: higher improvement -> higher probability
                logit = (ttt_improvement - threshold) / prob_temperature
                p_update = 1.0 / (1.0 + np.exp(-logit * 100))  # Scale for sensitivity
                do_update = float(jax.random.uniform(subkey)) < p_update
                total_count += 1
                if do_update:
                    update_count += 1
            else:
                raise ValueError(f"Unknown threshold_mode: {threshold_mode}")

            # Record result
            if do_update:
                results["loss"].append(loss_update)
                results["cost"].append(3.0)
                results["decision"].append("UPDATE")
            else:
                results["loss"].append(loss_skip)
                results["cost"].append(1.0)
                results["decision"].append("SKIP")

            results["is_real_code"].append(is_real_code)
            results["method"].append(method_name)
            results["ttt_improvement"].append(ttt_improvement)
            results["advantage"].append(advantage)
            results["threshold_used"].append(threshold if threshold is not None else 0.0)

    # Verify no state leakage
    final_checksum = get_fast_weight_checksum(ttt_model)
    if abs(final_checksum - initial_checksum) < 1e-6:
        print(f"\n  ✓ [State Check] No state leakage (fast weights unchanged)")
    else:
        print(f"\n  ✗ [State Check] WARNING: Fast weights changed! ({initial_checksum:.6f} → {final_checksum:.6f})")

    # Report statistics
    actual_update_rate = sum(1 for d in results["decision"] if d == "UPDATE") / len(results["decision"])
    print(f"  Actual update rate: {actual_update_rate:.2%} (target: {update_rate:.2%})")
    if threshold is not None:
        print(f"  Final threshold: {threshold:.6f}")

    # Correlation analysis
    ttt_arr = np.array(results["ttt_improvement"])
    adv_arr = np.array(results["advantage"])
    decisions = np.array([1 if d == "UPDATE" else 0 for d in results["decision"]])

    # Compare decisions with oracle
    oracle_decisions = (adv_arr > np.median(adv_arr)).astype(int)
    decision_accuracy = np.mean(decisions == oracle_decisions)
    print(f"  Decision accuracy vs Oracle: {decision_accuracy:.2%}")

    return pd.DataFrame(results)


def parse_args():
    parser = argparse.ArgumentParser(description="Compare optimization methods")
    parser.add_argument("--model_scale", type=str, default="125m", choices=["125m", "350m", "1b"])
    parser.add_argument("--budget", type=float, default=2.0, help="Target budget (avg steps)")
    parser.add_argument("--num_eval_batches", type=int, default=20, help="Number of batches for evaluation")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--update1_checkpoint", type=str, help="Path to UPDATE_1 baseline checkpoint (optional)")
    parser.add_argument("--output_dir", type=str, default="outputs/comparison")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--language", type=str, default="Python", help="Programming language for OOD testing")
    parser.add_argument("--split", type=str, default="train", help="Dataset split (train/validation/test). Note: The Stack v2 only has 'train'.")
    parser.add_argument("--skip_examples", type=int, default=0, help="Number of examples to skip (for held-out evaluation).")
    parser.add_argument("--num_workers", type=int, default=32, help="Number of parallel workers for data downloading")
    parser.add_argument("--eval_ttt_improvement", action="store_true", help="Evaluate TTT Improvement-based gating (training-free, uses TTT internal loss as signal)")
    parser.add_argument("--eval_threshold", action="store_true", help="Evaluate threshold-based gating (Walk phase)")
    parser.add_argument("--threshold_mode", type=str, default="ema", choices=["fixed", "ema", "prob"], help="Threshold gating mode")
    parser.add_argument("--initial_threshold", type=float, default=None, help="Initial threshold (if None, calibrate from data)")
    return parser.parse_args()


def evaluate_model(
    method_name: str,
    model_scale: str,
    budget_target: float,
    num_batches: int,
    batch_size: int,
    seed: int,
    model: Optional[TTTModel] = None,
    fixed_action: str = "SKIP",
    language: str = "Python",
    split: str = "test",
    skip_examples: int = 0,
    num_workers: int = 32,
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

    # RNG key for stochastic evaluation
    rng_key = jax.random.PRNGKey(seed)

    # JIT-compiled functions
    @nnx.jit
    def jit_skip_loss(model, input_ids, attention_mask, position_ids):
        out = model(input_ids, attention_mask=attention_mask, position_ids=position_ids, use_ttt=False)
        return cross_entropy_loss(out["logits"][:, :-1], input_ids[:, 1:], attention_mask[:, 1:])

    @nnx.jit
    def jit_update_loss(model, input_ids, attention_mask, position_ids):
        out = model(input_ids, attention_mask=attention_mask, position_ids=position_ids, use_ttt=True)
        return cross_entropy_loss(out["logits"][:, :-1], input_ids[:, 1:], attention_mask[:, 1:])

    assert isinstance(ttt_model, TTTTransformerLM)

    # Evaluation Loop
    for i, batch in enumerate(tqdm(data_iter, total=num_batches, desc=method_name)):
        if i >= num_batches:
            break

        chunks = batch["chunks"]
        masks = batch["chunk_attention_mask"]
        num_chunks = chunks.shape[1]

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

            # Decision
            cost = 1.0
            loss = 0.0
            decision_str = ""

            if random_update_rate is not None:
                # Random Skip Baseline: randomly decide SKIP/UPDATE based on update_rate
                rng_key, subkey = jax.random.split(rng_key)
                do_update = jax.random.uniform(subkey) < random_update_rate

                if do_update:
                    loss = float(jit_update_loss(
                        ttt_model,
                        chunk_batch["input_ids"],
                        chunk_batch["attention_mask"],
                        position_ids,
                    ))
                    cost = 3.0
                    decision_str = "UPDATE"
                else:
                    loss = float(jit_skip_loss(
                        ttt_model,
                        chunk_batch["input_ids"],
                        chunk_batch["attention_mask"],
                        position_ids,
                    ))
                    cost = 1.0
                    decision_str = "SKIP"
            elif fixed_action == "SKIP":
                # Fixed Baseline: SKIP
                loss = float(jit_skip_loss(
                    ttt_model,
                    chunk_batch["input_ids"],
                    chunk_batch["attention_mask"],
                    position_ids,
                ))
                cost = 1.0
                decision_str = "SKIP"
            elif fixed_action == "UPDATE_1":
                # Fixed Baseline: UPDATE_1
                loss = float(jit_update_loss(
                    ttt_model,
                    chunk_batch["input_ids"],
                    chunk_batch["attention_mask"],
                    position_ids,
                ))
                cost = 3.0
                decision_str = "UPDATE"
            else:
                raise ValueError(f"Unknown fixed_action: {fixed_action}")

            results["loss"].append(loss)
            results["cost"].append(cost)
            results["method"].append(method_name)
            results["decision"].append(decision_str)
            results["text"].append(text)
            results["is_real_code"].append(is_real_code)

    return pd.DataFrame(results)


def main():
    args = parse_args()

    print("="*60)
    print("Comparison: Gating Methods vs Baseline")
    print("="*60)

    model_name = {"125m": "gpt2", "350m": "gpt2-medium", "1b": "gpt2-large"}[args.model_scale]
    tokenizer = get_tokenizer(model_name)
    vocab_size = tokenizer.get_vocab_size()

    # UPDATE_1 Model (Fixed TTT baseline)
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
            model_state = unwrap_state(ckpt["state"]["model"])
            # Only update fast_layer and fast_norm
            if "fast_layer" in model_state:
                nnx.update(update1_ttt_model.fast_layer, model_state["fast_layer"])
                print("  ✓ Loaded fast_layer from checkpoint")
            if "fast_norm" in model_state:
                nnx.update(update1_ttt_model.fast_norm, model_state["fast_norm"])
                print("  ✓ Loaded fast_norm from checkpoint")
        else:
            print("Warning: Could not find 'state.model' in UPDATE_1 checkpoint.")

    all_results = []

    # Convert budget to target update rate (cost = 1 + 2 * update_rate)
    target_update_rate = max(0.0, min(1.0, (args.budget - 1.0) / 2.0))
    print(f"\nTarget update rate from budget={args.budget:.2f}: {target_update_rate*100:.1f}%")

    # 1. Evaluate SKIP Baseline
    df_skip = evaluate_model(
        "SKIP (Baseline)",
        args.model_scale,
        0.0,
        args.num_eval_batches,
        args.batch_size,
        args.seed,
        model=None,
        fixed_action="SKIP",
        language=args.language,
        split=args.split,
        skip_examples=args.skip_examples,
        num_workers=args.num_workers,
    )
    all_results.append(df_skip)

    # 2. Evaluate UPDATE_1 (Fixed)
    if update1_ttt_model is not None:
        df_update1 = evaluate_model(
            "UPDATE_1 (Fixed)",
            args.model_scale,
            0.0,
            args.num_eval_batches,
            args.batch_size,
            args.seed,
            model=update1_ttt_model,
            fixed_action="UPDATE_1",
            language=args.language,
            split=args.split,
            skip_examples=args.skip_examples,
            num_workers=args.num_workers,
        )
        all_results.append(df_update1)

        # 3. Evaluate Random Skip Baseline
        print(f"\n=== Running Random Skip Baseline (update_rate={target_update_rate:.2%}) ===")
        df_random = evaluate_model(
            f"Random Skip ({target_update_rate:.0%} update)",
            args.model_scale,
            args.budget,
            args.num_eval_batches,
            args.batch_size,
            args.seed + 1,  # Different seed for random decisions
            model=update1_ttt_model,
            language=args.language,
            split=args.split,
            skip_examples=args.skip_examples,
            num_workers=args.num_workers,
            random_update_rate=target_update_rate,
        )
        all_results.append(df_random)

        # 4. Evaluate Oracle Baseline (upper bound)
        print(f"\n=== Running Oracle Baseline (update_rate={target_update_rate:.2%}) ===")
        df_oracle = evaluate_oracle(
            f"Oracle ({target_update_rate:.0%} update)",
            args.model_scale,
            update_rate=target_update_rate,
            num_batches=args.num_eval_batches,
            batch_size=args.batch_size,
            seed=args.seed,
            model=update1_ttt_model,
            language=args.language,
            split=args.split,
            skip_examples=args.skip_examples,
            num_workers=args.num_workers,
        )
        all_results.append(df_oracle)

    # 5. Evaluate TTT Improvement Gating (training-free, measurement-based)
    if args.eval_ttt_improvement:
        model_source = "UPDATE_1" if update1_ttt_model else "fresh"
        print(f"\n=== Running TTT Improvement Gating (update_rate={target_update_rate:.2%}) ===")
        print(f"    (Using {model_source} TTT weights)")
        df_ttt_improvement = evaluate_ttt_improvement_gating(
            f"TTT Improvement ({target_update_rate:.0%} update)",
            args.model_scale,
            update_rate=target_update_rate,
            num_batches=args.num_eval_batches,
            batch_size=args.batch_size,
            seed=args.seed,
            model=update1_ttt_model,
            language=args.language,
            split=args.split,
            skip_examples=args.skip_examples,
            num_workers=args.num_workers,
        )
        all_results.append(df_ttt_improvement)

    # 6. Evaluate Threshold Gating (Walk phase - online decision making)
    if args.eval_threshold:
        model_source = "UPDATE_1" if update1_ttt_model else "fresh"
        print(f"\n=== Running Threshold Gating (mode={args.threshold_mode}, update_rate={target_update_rate:.2%}) ===")
        print(f"    (Using {model_source} TTT weights)")
        df_threshold = evaluate_threshold_gating(
            f"Threshold-{args.threshold_mode} ({target_update_rate:.0%} update)",
            args.model_scale,
            update_rate=target_update_rate,
            num_batches=args.num_eval_batches,
            batch_size=args.batch_size,
            seed=args.seed,
            model=update1_ttt_model,
            language=args.language,
            split=args.split,
            skip_examples=args.skip_examples,
            num_workers=args.num_workers,
            threshold_mode=args.threshold_mode,
            initial_threshold=args.initial_threshold,
        )
        all_results.append(df_threshold)

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

    # TTT Improvement vs Random Skip vs Oracle comparison
    random_method = [m for m in summary.index if m.startswith("Random Skip")]
    oracle_method = [m for m in summary.index if m.startswith("Oracle")]
    ttt_imp_method = [m for m in summary.index if m.startswith("TTT Improvement")]

    if random_method and oracle_method:
        random_name = random_method[0]
        oracle_name = oracle_method[0]
        random_loss = float(summary.loc[random_name, "loss"])
        oracle_loss = float(summary.loc[oracle_name, "loss"])
        random_cost = float(summary.loc[random_name, "cost"])
        oracle_cost = float(summary.loc[oracle_name, "cost"])

        improvement_oracle = (random_loss - oracle_loss) / random_loss * 100

        print("\n" + "=" * 70)
        print("ABLATION: Random Skip vs TTT Improvement vs Oracle")
        print("=" * 70)
        print(f"{'Method':<30} {'Loss':>10} {'Cost':>10} {'vs Random':>12}")
        print("-" * 70)
        print(f"{'Random Skip':<30} {random_loss:>10.4f} {random_cost:>10.2f}x {'baseline':>12}")

        if ttt_imp_method:
            ttt_imp_name = ttt_imp_method[0]
            ttt_imp_loss = float(summary.loc[ttt_imp_name, "loss"])
            ttt_imp_cost = float(summary.loc[ttt_imp_name, "cost"])
            improvement_ttt = (random_loss - ttt_imp_loss) / random_loss * 100
            print(f"{'TTT Improvement (Training-free)':<30} {ttt_imp_loss:>10.4f} {ttt_imp_cost:>10.2f}x {improvement_ttt:>+11.2f}%")

        print(f"{'Oracle (Upper Bound)':<30} {oracle_loss:>10.4f} {oracle_cost:>10.2f}x {improvement_oracle:>+11.2f}%")
        print("-" * 70)

        # How much of the Oracle improvement does TTT Improvement capture?
        if ttt_imp_method:
            gap_random_to_oracle = random_loss - oracle_loss
            gap_random_to_ttt = random_loss - ttt_imp_loss
            capture_rate = (gap_random_to_ttt / gap_random_to_oracle * 100) if gap_random_to_oracle > 0 else 0

            print(f"\nOracle Gap:         Random → Oracle = {gap_random_to_oracle:.4f}")
            print(f"TTT Improvement:    Random → TTT    = {gap_random_to_ttt:.4f}")
            print(f"Capture Rate:       {capture_rate:.1f}% of Oracle improvement")

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
