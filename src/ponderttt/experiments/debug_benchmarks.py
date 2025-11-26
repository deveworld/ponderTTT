"""
Run code generation benchmarks with debugging features.
"""

import argparse
import json
import os
import traceback
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from tqdm import tqdm

from ..data import get_tokenizer
from ..evaluation.benchmarks import BenchmarkSuite, _check_solution
from ..models import TTTTransformerLM, load_ttt_model, TTTConfig
from ..models.gating_nnx import GatingConfig, GatingNetwork
from ..utils import FeatureExtractor, extract_features
from ..utils.checkpointing import load_checkpoint


@nnx.jit(static_argnames=("use_ttt",))
def _model_forward_jit(model, input_ids, use_ttt, gating_scale):
    return model(input_ids, use_ttt=use_ttt, gating_scale=gating_scale)


@nnx.jit(static_argnames=("deterministic",))
def _policy_forward_jit(net, x, deterministic):
    return net(x, deterministic=deterministic)


@nnx.jit(static_argnames=("train",))
def _gating_forward_jit(net, x, train):
    return net(x, train=train)


def parse_args():
    parser = argparse.ArgumentParser(description="Run benchmarks (Debug Mode)")
    parser.add_argument(
        "--benchmark", 
        type=str, 
        choices=["humaneval", "mbpp", "all"], 
        default="humaneval"
    )
    parser.add_argument("--model_scale", type=str, default="125m")
    parser.add_argument("--checkpoint", type=str, help="Path to trained TTT checkpoint (optional)")
    parser.add_argument("--baseline", type=str, help="Path to baseline model checkpoint (optional)")
    parser.add_argument("--gating_checkpoint", type=str, help="Path to gating network checkpoint (optional)")
    parser.add_argument("--output_dir", type=str, default="outputs/benchmarks_debug")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of problems (for testing)")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for generation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow_unsafe", action="store_true", help="Allow unsafe code execution")
    parser.add_argument("--force_scale", type=float, default=None, help="Force TTT scale (0.0 to disable, 1.0 to max)")
    return parser.parse_args()


class SimpleGenerator:
    """Simple autoregressive generator for NNX models."""
    
    def __init__(self, model: TTTTransformerLM, tokenizer, gating_net=None, force_scale=None):
        self.model = model
        self.tokenizer = tokenizer
        self.gating_net = gating_net
        self.force_scale = force_scale
        self.pad_token_id = tokenizer.token_to_id("<|pad|>")
        self.eos_token_id = tokenizer.token_to_id("<|endoftext|>")
        
        base_config = getattr(model, "gpt2_config", None)
        if base_config is None and hasattr(model, "base_model"):
            base_config = getattr(model.base_model, "config", None)
        self.max_seq_len = getattr(base_config, "n_positions", 1024)
        self._warned_truncation = False
        
        self.feature_extractor = FeatureExtractor(
            vocab_size=tokenizer.get_vocab_size(),
            pad_token_id=self.pad_token_id,
            seq_length_norm=512,
        )

        vocab_size = tokenizer.get_vocab_size()
        pad_id = self.pad_token_id
        
        @jax.jit
        def extract_features_jit(input_ids, logits, attention_mask):
            return extract_features(
                input_ids=input_ids,
                logits=logits,
                vocab_size=vocab_size,
                pad_token_id=pad_id,
                seq_length_norm=512.0,
                attention_mask=attention_mask,
                budget_remaining=1.0
            )
        self.extract_features_jit = extract_features_jit

    def _call_model(self, input_tensor: jax.Array, use_ttt: bool, gating_scale: jax.Array | None):
        batch_size = input_tensor.shape[0]
        if gating_scale is None:
            gating_scale = jnp.zeros((batch_size, 1), dtype=jnp.float32)
        return _model_forward_jit(self.model, input_tensor, use_ttt=use_ttt, gating_scale=gating_scale)

    def _truncate_context(self, token_ids: list[int]) -> list[int]:
        if len(token_ids) <= self.max_seq_len:
            return token_ids
        if not self._warned_truncation:
            print(
                f"WARNING: Prompt length {len(token_ids)} exceeds model limit {self.max_seq_len}. "
                "Keeping the most recent tokens."
            )
            self._warned_truncation = True
        return token_ids[-self.max_seq_len:]

    def generate_batch(
        self, 
        prompts: list[str], 
        max_new_tokens: int = 128, 
        temperature: float = 0.0
    ) -> list[str]:
        input_ids_list = [self._truncate_context(list(self.tokenizer.encode(p).ids)) for p in prompts]
        batch_size = len(prompts)
        generated_tokens = [[] for _ in range(batch_size)]
        finished = [False] * batch_size
        
        for _ in range(max_new_tokens):
            if all(finished):
                break
                
            for idx in range(batch_size):
                input_ids_list[idx] = self._truncate_context(input_ids_list[idx])

            current_lens = [len(ids) for ids in input_ids_list]
            max_len = max(current_lens)
            
            mini_batch_size = 16
            remat_group_size = 1
            if hasattr(self.model, "fast_layer") and hasattr(self.model.fast_layer, "config"):
                if isinstance(self.model.fast_layer.config, TTTConfig):
                    mini_batch_size = self.model.fast_layer.config.mini_batch_size
                    remat_group_size = self.model.fast_layer.config.remat_mini_batch_group_size
            
            alignment = mini_batch_size * remat_group_size
            target_len = ((max_len + alignment - 1) // alignment) * alignment
            target_len = min(target_len, self.max_seq_len)
            
            padded_input_ids = []
            attention_masks = []
            
            for i, ids in enumerate(input_ids_list):
                pad_len = target_len - len(ids)
                padded_ids = ids + [self.pad_token_id] * pad_len
                mask = [1] * len(ids) + [0] * pad_len
                padded_input_ids.append(padded_ids)
                attention_masks.append(mask)
                
            input_tensor = jnp.array(padded_input_ids, dtype=jnp.int32)
            mask_tensor = jnp.array(attention_masks, dtype=jnp.int32)

            gating_scale = None            
            if self.force_scale is not None:
                scale = self.force_scale
                if scale > 0.001:
                    gating_scale = jnp.full((batch_size, 1), scale, dtype=jnp.float32)
                    outputs = self._call_model(input_tensor, use_ttt=True, gating_scale=gating_scale)
                else:
                    outputs = self._call_model(input_tensor, use_ttt=False, gating_scale=None)
            elif self.gating_net is not None:
                out_base = self._call_model(input_tensor, use_ttt=False, gating_scale=None)
                features = self.extract_features_jit(
                    input_ids=input_tensor,
                    logits=out_base["logits"],
                    attention_mask=mask_tensor
                )
                if hasattr(self.gating_net, "evaluate_actions"):
                    policy_out = _policy_forward_jit(self.gating_net, features, deterministic=True)
                    actions = policy_out["action"]
                    step_map = jnp.array([0.0, 1.0, 2.0, 4.0])
                    scales = step_map[actions.astype(int)]
                else:
                    scales = _gating_forward_jit(self.gating_net, features, train=False)[:, 0]

                # Fix: Always use TTT to match training "Soft Skip" behavior
                gating_scale = scales[:, None]
                outputs = self._call_model(input_tensor, use_ttt=True, gating_scale=gating_scale)
            else:
                outputs = self._call_model(input_tensor, use_ttt=False, gating_scale=None)
            
            indices = jnp.array([length - 1 for length in current_lens])
            next_token_logits = outputs["logits"][jnp.arange(batch_size), indices, :]
            
            if temperature > 0:
                probs = jax.nn.softmax(next_token_logits / temperature, axis=-1)
                next_tokens = jax.random.categorical(jax.random.PRNGKey(0), jnp.log(probs)) 
            else:
                next_tokens = jnp.argmax(next_token_logits, axis=-1)
            
            for i, token in enumerate(next_tokens):
                if not finished[i]:
                    token_val = int(token)
                    if token_val != self.eos_token_id:
                        generated_tokens[i].append(token_val)
                    if token_val == self.eos_token_id:
                        finished[i] = True
                    else:
                        input_ids_list[i].append(token_val)
                        if len(input_ids_list[i]) > self.max_seq_len:
                            input_ids_list[i] = input_ids_list[i][-self.max_seq_len:]
                        
        completions = []
        for i in range(batch_size):
            completions.append(self.tokenizer.decode(generated_tokens[i]))
            
        return completions


def unwrap_state(state):
    if isinstance(state, dict):
        if "value" in state and len(state) == 1:
            return state["value"]
        return {k: unwrap_state(v) for k, v in state.items()}
    return state


def print_stats(name, arr):
    arr = np.array(arr)
    print(f"  {name}: shape={arr.shape}, mean={arr.mean():.6f}, std={arr.std():.6f}, min={arr.min():.6f}, max={arr.max():.6f}")
    if np.allclose(arr, 0):
        print(f"  WARNING: {name} is all zeros!")


def main():
    args = parse_args()

    if args.checkpoint and args.baseline:
        raise ValueError("Please provide only one of --checkpoint or --baseline")

    if args.allow_unsafe:
        os.environ["PONDER_TTT_ALLOW_UNSAFE_BENCHMARKS"] = "1"
    elif os.environ.get("PONDER_TTT_ALLOW_UNSAFE_BENCHMARKS") != "1":
        print("\nERROR: Benchmarks require executing generated code.")
        return
    
    tokenizer = get_tokenizer({"125m": "gpt2", "350m": "gpt2-medium", "1b": "gpt2-large"}[args.model_scale])
    pad_token_id = tokenizer.token_to_id("<|pad|>")
    vocab_size = tokenizer.get_vocab_size()

    print(f"Loading model {args.model_scale}...")
    model, _ = load_ttt_model(
        model_name={"125m": "gpt2", "350m": "gpt2-medium", "1b": "gpt2-large"}[args.model_scale],
        fast_weight_type="ttt",
        load_pretrained=True,
        vocab_size=vocab_size,
        pad_token_id=pad_token_id
    )
    
    print("\n[DEBUG] Initial Model Stats (Pretrained):")
    print_stats("base_model.wte", model.base_model.wte.embedding[...])
    print_stats("fast_layer.wo", model.fast_layer.wo.kernel[...]) # type: ignore
    
    gating_net = None

    if args.checkpoint:
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
        
        dummy_gating = GatingNetwork(GatingConfig(), nnx.Rngs(0))
        trainable_sys = TrainableSystem(model, dummy_gating)
        
        try:
            print("Attempting to load with strict structure matching...")
            target = {"state": {"model": nnx.state(trainable_sys)}}
            ckpt = load_checkpoint(args.checkpoint, target=target)
            if "state" in ckpt and "model" in ckpt["state"]:
                nnx.update(trainable_sys, ckpt["state"]["model"])
                print("Loaded as Differentiable Training checkpoint")
            else:
                raise ValueError("Checkpoint does not contain 'state.model'")
            if args.gating_checkpoint is None:
                 gating_net = trainable_sys.gating_net
        except Exception as e_diff:
            print(f"Strict loading failed ({e_diff}), trying loose loading...")
            try:
                ckpt = load_checkpoint(args.checkpoint, target=None)
                if "state" in ckpt and "model" in ckpt["state"]:
                    model_state = unwrap_state(ckpt["state"]["model"])
                    nnx.update(trainable_sys, model_state)
                    print("Loaded as Differentiable Training checkpoint (loose)")
                    if args.gating_checkpoint is None:
                        gating_net = trainable_sys.gating_net
                else:
                    raise ValueError("Checkpoint does not contain 'state.model'")
            except Exception as e_loose:
                print(f"Not a Differentiable Training checkpoint ({e_loose}), trying Baseline...")
                try:
                    ckpt = load_checkpoint(args.checkpoint, target=None)
                    if "state" in ckpt and "model" in ckpt["state"]:
                        nnx.update(model, ckpt["state"]["model"])
                        print("Loaded as Baseline checkpoint")
                    else:
                        raise ValueError("Checkpoint does not contain 'state.model'")
                except Exception as e_base:
                     print(f"Failed to load checkpoint: {e_base}")
                     raise ValueError("Could not load checkpoint")

    print("\n[DEBUG] Post-Load Model Stats:")
    print_stats("base_model.wte", model.base_model.wte.embedding[...])
    print_stats("fast_layer.wo", model.fast_layer.wo.kernel[...]) # type: ignore
    print_stats("fast_norm.scale", model.fast_norm.scale.value) # type: ignore

    generator = SimpleGenerator(model, tokenizer, gating_net, force_scale=args.force_scale)
    
    suite = BenchmarkSuite(
        include_humaneval=(args.benchmark in ["humaneval", "all"]),
        include_mbpp=(args.benchmark in ["mbpp", "all"]),
    )
    
    results = {}
    
    for name, benchmark in suite.benchmarks.items():
        print(f"\nRunning {name} ({len(benchmark)} problems)...")
        problems = benchmark.problems
        if args.limit:
            problems = problems[:args.limit]
            print(f"  Limiting to {args.limit} problems")
            
        scores = []
        for i in tqdm(range(0, len(problems), args.batch_size), desc=f"Running {name}"):
            batch_problems = problems[i : i + args.batch_size]
            prompts = [p.prompt for p in batch_problems]
            try:
                completions = generator.generate_batch(prompts, args.max_new_tokens, args.temperature)
                for problem, completion in zip(batch_problems, completions):
                    is_correct = _check_solution(problem, completion)
                    scores.append(1.0 if is_correct else 0.0)
            except Exception as e:
                print(f"  Batch failed: {e}")
                traceback.print_exc()
                for _ in batch_problems:
                    scores.append(0.0)

        final_score = sum(scores) / len(scores) if scores else 0.0
        results[name] = {"pass@1": final_score}
        print(f"  Result: {final_score}")

    os.makedirs(args.output_dir, exist_ok=True)
    with open(Path(args.output_dir) / "results_debug.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
