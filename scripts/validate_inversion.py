import argparse
import jax
import jax.numpy as jnp
import pandas as pd
from flax import nnx
from tqdm import tqdm
import optax
from ponderttt.models import load_ttt_model
from ponderttt.data import get_tokenizer, create_data_iterator


def run_validation(
    model_scale="xl",
    max_grad_norm=1.0,
    num_chunks=50,
    batch_size=1,
    seed=42,
    language="Python",
    checkpoint_path=None,
):
    print(
        f"Validating Inversion Hypothesis on {model_scale} (Gradient Clipping = {max_grad_norm})..."
    )
    if checkpoint_path:
        print(f"Target Checkpoint: {checkpoint_path}")

    model_name = {
        "125m": "gpt2",
        "350m": "gpt2-medium",
        "1b": "gpt2-large",
        "xl": "gpt2-xl",
    }[model_scale]

    print(f"Loading {model_name}...")
    tokenizer = get_tokenizer(model_name)
    model, _ = load_ttt_model(
        model_name,
        fast_weight_type="ttt",
        load_pretrained=True,
        vocab_size=tokenizer.get_vocab_size(),
        checkpoint_path=checkpoint_path,
    )

    data_iter = create_data_iterator(
        tokenizer=tokenizer,
        split="train",
        language=language,
        batch_size=batch_size,
        seq_length=1024,
        chunk_size=512,
        max_examples=num_chunks * 2,
    )

    # Initialize fast_layer.wo to small random values to unmask TTT updates
    if hasattr(model, "fast_layer") and hasattr(model.fast_layer, "wo"):
        print("Initializing fast_layer.wo to small random values...")
        key = jax.random.PRNGKey(seed)
        wo_param = model.fast_layer.wo.kernel
        wo_param.value = jax.random.normal(key, wo_param.shape) * 0.02

    results = []

    @nnx.jit
    def forward_no_clip(model, input_ids):
        """UPDATE_1 without gradient clipping."""
        if hasattr(model.fast_layer, "config"):
            model.fast_layer.config.max_grad_norm = None

        out = model(input_ids, use_ttt=True)
        logits = out["logits"][:, :-1]
        labels = input_ids[:, 1:]
        mask = jnp.ones_like(labels)

        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
        loss = (loss * mask).sum() / mask.sum()
        recon_loss = out["ttt_stats"]["ttt_loss_step_0"].mean()

        return loss, recon_loss

    @nnx.jit
    def forward_with_clip(model, input_ids):
        """UPDATE_1 with gradient clipping."""
        if hasattr(model.fast_layer, "config"):
            model.fast_layer.config.max_grad_norm = max_grad_norm

        out = model(input_ids, use_ttt=True)
        logits = out["logits"][:, :-1]
        labels = input_ids[:, 1:]
        mask = jnp.ones_like(labels)

        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
        loss = (loss * mask).sum() / mask.sum()

        return loss

    @nnx.jit
    def forward_skip(model, input_ids):
        out = model(input_ids, use_ttt=False)
        logits = out["logits"][:, :-1]
        labels = input_ids[:, 1:]
        mask = jnp.ones_like(labels)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
        loss = (loss * mask).sum() / mask.sum()
        return loss

    print("Running inference...")
    chunk_counter = 0
    for i, batch in enumerate(tqdm(data_iter)):
        if chunk_counter >= num_chunks:
            break

        chunks = batch["chunks"]
        for c_idx in range(chunks.shape[1]):
            input_ids = chunks[:, c_idx]

            loss_skip = float(forward_skip(model, input_ids))
            loss_update_nc, recon_loss = forward_no_clip(model, input_ids)
            loss_update_nc = float(loss_update_nc)
            recon_loss = float(recon_loss)
            loss_update_c = float(forward_with_clip(model, input_ids))

            results.append(
                {
                    "recon_loss": recon_loss,
                    "loss_skip": loss_skip,
                    "loss_update_no_clip": loss_update_nc,
                    "loss_update_clip": loss_update_c,
                    "improvement_no_clip": loss_skip - loss_update_nc,
                    "improvement_clip": loss_skip - loss_update_c,
                }
            )

            chunk_counter += 1
            if chunk_counter >= num_chunks:
                break

    df = pd.DataFrame(results)
    df_sorted = df.sort_values("recon_loss", ascending=False)
    k = max(1, len(df) // 5)  # Top 20%
    top_k = df_sorted.head(k)

    print("\n" + "=" * 60)
    print("INVERSION HYPOTHESIS VALIDATION RESULTS")
    print(f"Model: {model_name}, Clipping Norm: {max_grad_norm}")
    print("=" * 60)

    print(f"\n[Analysing Top {k} Chunks with HIGHEST Reconstruction Loss (Top 20%)]")
    avg_skip = top_k["loss_skip"].mean()
    avg_nc = top_k["loss_update_no_clip"].mean()
    avg_c = top_k["loss_update_clip"].mean()

    print(f"  Avg Loss (SKIP):            {avg_skip:.4f}")
    print(f"  Avg Loss (UPDATE_1 NoClip): {avg_nc:.4f} (Imp: {avg_skip - avg_nc:.4f})")
    print(f"  Avg Loss (UPDATE_1 Clip):   {avg_c:.4f} (Imp: {avg_skip - avg_c:.4f})")

    print("-" * 60)

    if avg_c < avg_nc and (avg_skip - avg_c) > 0 and (avg_skip - avg_nc) < 0:
        print("RESULT: Clipping FIXED the degradation! -> Inversion Hypothesis WEAKENED.")
        print("        (High loss was due to gradient instability)")
    elif avg_c >= avg_skip or (avg_skip - avg_c) < (avg_skip - avg_nc) + 0.01:
        print("RESULT: Clipping did NOT fix degradation! -> Inversion Hypothesis STRENGTHENED.")
        print("        (High reconstruction loss indicates 'updating is harmful')")
    else:
        print("RESULT: Mixed/Inconclusive. See details.")

    print("-" * 60)
    print("Top 5 Chunks Detail:")
    print(
        top_k[["recon_loss", "loss_skip", "loss_update_no_clip", "loss_update_clip"]]
        .head(5)
        .to_string()
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_scale", type=str, default="xl")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_chunks", type=int, default=50)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    args = parser.parse_args()

    run_validation(
        model_scale=args.model_scale,
        max_grad_norm=args.max_grad_norm,
        num_chunks=args.num_chunks,
        checkpoint_path=args.checkpoint_path,
    )
