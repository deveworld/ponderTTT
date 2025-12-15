
import argparse
import jax
import jax.numpy as jnp
import pandas as pd
import numpy as np
from flax import nnx
from tqdm import tqdm
from scipy import stats

from ..data import create_data_iterator, get_tokenizer
from ..models import load_ttt_model, TTTTransformerLM
from ..utils import cross_entropy_loss

@nnx.jit
def fit_forward(model, input_ids, attention_mask, position_ids):
    # 1. Standard Forward (No TTT) -> Get Entropy & CE Loss (SKIP)
    # We need logits for entropy.
    out_skip = model(
        input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_ttt=False
    )
    logits = out_skip["logits"]
    
    # Calculate Entropy
    # prob = softmax(logits)
    # entropy = -sum(prob * log(prob))
    # We compute this on the *last* token prediction or average? 
    # Gating is per-chunk. Usually we care about the *average* entropy of the chunk 
    # or the entropy of the *first* token if we decide before processing?
    # PonderTTT decides *per chunk*. 
    # "Prediction Entropy" usually means: how unsure is the model about this chunk?
    # We'll compute average entropy over the chunk.
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    probs = jnp.exp(log_probs)
    entropy = -jnp.sum(probs * log_probs, axis=-1) # [B, L]
    # Mask out padding
    # attention_mask is [B, L]
    # We want average entropy over valid tokens
    # Note: logits are for predicting next token. 
    # logits[:, i] predicts input_ids[:, i+1]
    # So we align with input_ids[:, 1:] 
    
    valid_mask = attention_mask[:, 1:]
    avg_entropy = jnp.sum(entropy[:, :-1] * valid_mask) / (jnp.sum(valid_mask) + 1e-9)
    
    loss_skip = cross_entropy_loss(logits[:, :-1], input_ids[:, 1:], valid_mask)
    
    # 2. TTT Internal Stats (TTT Reconstruction Loss)
    # To get TTT stats, we run with use_ttt=True, but we want the *Initial* loss before update.
    # TTTLayer returns ttt_stats.
    out_update = model(
        input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_ttt=True
    )
    loss_update = cross_entropy_loss(out_update["logits"][:, :-1], input_ids[:, 1:], valid_mask)
    
    ttt_stats = out_update.get("ttt_stats", {})
    # ttt_loss_step_0 is the reconstruction loss with initial weights (theta_0)
    # ttt_loss_step_1 is with theta_1
    ttt_loss_initial = ttt_stats.get("ttt_loss_step_0", jnp.array(0.0))
    ttt_loss_final = ttt_stats.get("ttt_loss_step_1", jnp.array(0.0))
    
    # Ensure all are scalars (take mean over batch if necessary, though input is usually single batch or reduced)
    # cross_entropy_loss returns a scalar (mean over batch).
    # avg_entropy calculation above resulted in a scalar.
    # ttt_loss_step_0 might be [B] or scalar depending on model. Let's force mean.
    
    return {
        "loss_skip": jnp.mean(loss_skip),
        "loss_update": jnp.mean(loss_update),
        "avg_entropy": jnp.mean(avg_entropy),
        "ttt_loss_initial": jnp.mean(ttt_loss_initial),
        "ttt_loss_final": jnp.mean(ttt_loss_final)
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_scale", type=str, default="125m")
    parser.add_argument("--num_batches", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()
    
    print(f"Analyzing signals for {args.model_scale} model...")
    
    model_name = {"125m": "gpt2", "350m": "gpt2-medium", "1b": "gpt2-large"}[args.model_scale]
    tokenizer = get_tokenizer(model_name)
    
    model, _ = load_ttt_model(
        model_name=model_name,
        fast_weight_type="ttt",
        load_pretrained=True,
        vocab_size=tokenizer.get_vocab_size()
    )
    
    data_iter = create_data_iterator(
        tokenizer=tokenizer,
        split="train", # Use train or validation? Paper used held-out. Let's use train for quick check or 'test' implies skipping?
        # The create_data_iterator defaults to The Stack v2 train (since it only has train). 
        # We can use skip_examples to simulate held-out.
        language="Python",
        batch_size=args.batch_size,
        seq_length=1024,
        chunk_size=512,
        skip_examples=160000, # Held-out
        max_examples=args.num_batches * args.batch_size * 2
    )
    
    results = []
    
    for i, batch in enumerate(tqdm(data_iter, total=args.num_batches)):
        if i >= args.num_batches:
            break
            
        chunks = batch["chunks"]
        masks = batch["chunk_attention_mask"]
        num_chunks = chunks.shape[1]
        
        for c_idx in range(num_chunks):
            chunk_input = chunks[:, c_idx]
            chunk_mask = masks[:, c_idx]
            
            chunk_len = chunk_input.shape[-1]
            position_ids = jnp.arange(chunk_len, dtype=jnp.int32) + c_idx * chunk_len
            position_ids = jnp.broadcast_to(position_ids, chunk_input.shape)
            
            # Skip empty chunks
            if jnp.sum(chunk_mask) == 0:
                continue
                
            metrics = fit_forward(model, chunk_input, chunk_mask, position_ids)
            
            # Convert to python floats
            metrics = {k: float(v) for k, v in metrics.items()}
            
            metrics["advantage"] = metrics["loss_skip"] - metrics["loss_update"]
            metrics["ttt_improvement"] = metrics["ttt_loss_initial"] - metrics["ttt_loss_final"]
            
            results.append(metrics)
            
    df = pd.DataFrame(results)
    
    print("\n=== Signal Correlation Analysis (Pearson r) ===")
    print(f"Samples: {len(df)}")
    
    signals = ["loss_skip", "avg_entropy", "ttt_loss_initial", "ttt_improvement"]
    target = "advantage"
    
    for sig in signals:
        r, p = stats.pearsonr(df[sig], df[target])
        print(f"{sig:<20} vs Advantage: r={r:.4f} (p={p:.4e})")
        
    print("\n=== Correlation Matrix ===")
    print(df[signals + [target]].corr())
    
    output_file = f"outputs/signal_analysis_{args.model_scale}.csv"
    df.to_csv(output_file)
    print(f"\nDetailed results saved to {output_file}")

if __name__ == "__main__":
    main()
