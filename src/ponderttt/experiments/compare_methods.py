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
    binary_eval_stochastic: bool = True,  # Use stochastic sampling for BinaryGatingNetwork
    random_update_rate: Optional[float] = None,  # For Random Skip baseline (0.0-1.0)
):
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

    # RNG key for stochastic evaluation
    rng_key = jax.random.PRNGKey(seed)

    # Evaluation Loop
    for i, batch in enumerate(tqdm(data_iter, total=num_batches, desc=method_name)):
        if i >= num_batches:
            break

        chunks = batch["chunks"]
        masks = batch["chunk_attention_mask"]
        num_chunks = chunks.shape[1]
        total_budget = budget_target * num_chunks
        current_spend = 0.0

        feature_extractor.reset_history()

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

            # Budget Feature
            budget_rem = (total_budget - current_spend) / total_budget if total_budget > 0 else 0.0

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
                # Binary decision - use __call__ directly instead of get_decision
                # to avoid JIT caching issues with updated gating_net weights
                gating_scale, _, soft_probs, decision = gating_net(features, train=False)
                decision_val = int(decision[0])

                if decision_val == 0:
                    # SKIP
                    cost = 1.0
                    loss = float(jit_loss_from_logits(
                        logits_base,
                        chunk_batch["input_ids"],
                        chunk_batch["attention_mask"]
                    ))
                    decision_str = "SKIP"
                else:
                    # UPDATE: run TTT with scale (JIT)
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

            # Determine if this is a real code chunk (>10% valid tokens)
            valid_ratio = float(jnp.sum(chunk_batch["attention_mask"])) / chunk_len
            is_real_code = valid_ratio > 0.1

            results["loss"].append(loss)
            results["cost"].append(cost)
            results["method"].append(method_name)
            results["decision"].append(decision_str)
            results["text"].append(text)
            results["is_real_code"].append(is_real_code)

            current_spend += cost
            feature_extractor.update_history(loss, cost)

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

        class TrainableSystem(nnx.Module):
            def __init__(self, ttt_model, gating_net):
                self.fast_layer = ttt_model.fast_layer
                self.fast_norm = ttt_model.fast_norm
                self.gating_net = gating_net
                if hasattr(ttt_model, 'lm_head'):
                    self.lm_head = ttt_model.lm_head
                else:
                    self.lm_head = None

        trainable_system = TrainableSystem(diff_ttt_model, diff_net)
        ckpt = load_checkpoint(args.diff_checkpoint, target=None)

        if "state" in ckpt and "model" in ckpt["state"]:
            model_state = unwrap_state(ckpt["state"]["model"])
            nnx.update(trainable_system, model_state)
            # IMPORTANT: Reassign all updated components
            # nnx.update may replace Variable objects, breaking original references
            diff_net = trainable_system.gating_net
            diff_ttt_model.fast_layer = trainable_system.fast_layer
            diff_ttt_model.fast_norm = trainable_system.fast_norm
            if trainable_system.lm_head is not None:
                diff_ttt_model.lm_head = trainable_system.lm_head
            print("Differentiable Gating and TTT weights loaded.")
        else:
            print("Warning: Could not find 'state.model' in checkpoint.")
    else:
        print("No Differentiable checkpoint supplied; skipping differentiable evaluation.")

    # Binary Gating Network (Hard Skip, trained with Gumbel-Softmax)
    binary_net: Optional[BinaryGatingNetwork] = None
    binary_ttt_model = None

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

        class TrainableSystemBinary(nnx.Module):
            def __init__(self, ttt_model, gating_net):
                self.fast_layer = ttt_model.fast_layer
                self.fast_norm = ttt_model.fast_norm
                self.gating_net = gating_net
                if hasattr(ttt_model, 'lm_head'):
                    self.lm_head = ttt_model.lm_head
                else:
                    self.lm_head = None

        trainable_system_binary = TrainableSystemBinary(binary_ttt_model, binary_net)
        ckpt = load_checkpoint(args.binary_gating_checkpoint, target=None)

        if "state" in ckpt and "model" in ckpt["state"]:
            model_state = unwrap_state(ckpt["state"]["model"])
            nnx.update(trainable_system_binary, model_state)
            # IMPORTANT: Reassign all updated components
            # nnx.update may replace Variable objects, breaking original references
            binary_net = trainable_system_binary.gating_net
            binary_ttt_model.fast_layer = trainable_system_binary.fast_layer
            binary_ttt_model.fast_norm = trainable_system_binary.fast_norm
            if trainable_system_binary.lm_head is not None:
                binary_ttt_model.lm_head = trainable_system_binary.lm_head
            print("Binary Gating (Hard Skip) and TTT weights loaded.")
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
        )
        all_results.append(df_binary)

        # Calculate actual update rate from Binary Gating
        binary_update_rate = (df_binary["decision"] == "UPDATE").mean()
        print(f"\n=== Binary Gating Update Rate: {binary_update_rate:.2%} ===")

    # 6. Evaluate Random Skip Baseline (same update rate as Binary Gating)
    # IMPORTANT: Use binary_ttt_model (same TTT weights as Binary Gating) for fair comparison
    if binary_update_rate is not None and binary_ttt_model is not None:
        print(f"\n=== Running Random Skip Baseline (update_rate={binary_update_rate:.2%}) ===")
        print("    (Using same TTT weights as Binary Gating for fair comparison)")
        df_random = evaluate_model(
            f"Random Skip ({binary_update_rate:.0%} update)",
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
            random_update_rate=binary_update_rate,
        )
        all_results.append(df_random)

        # 6b. Evaluate Oracle Baseline (upper bound)
        print(f"\n=== Running Oracle Baseline (update_rate={binary_update_rate:.2%}) ===")
        print("    (Using same TTT weights as Binary Gating for fair comparison)")
        df_oracle = evaluate_oracle(
            f"Oracle ({binary_update_rate:.0%} update)",
            args.model_scale,
            update_rate=binary_update_rate,
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
            improvement_binary_vs_oracle = (binary_loss - oracle_loss) / (random_loss - oracle_loss) * 100 if random_loss != oracle_loss else 0

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