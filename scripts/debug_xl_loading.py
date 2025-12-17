import jax
import jax.numpy as jnp
from flax import nnx
import time

from ponderttt.models.base_model_nnx import load_ttt_model
from ponderttt.models.gpt2_nnx import GPT2Config

from ponderttt.data.tokenization import get_tokenizer


def check_model(model_name):
    print(f"\n[{model_name}] Loading...")
    start = time.time()
    try:
        # Load model using the project's loader
        # Use simple "gpt2" for tokenizer as they share vocab usually
        tokenizer = get_tokenizer("gpt2")
        model, config = load_ttt_model(
            model_name=model_name, fast_weight_type="ttt", load_pretrained=True
        )
        print(f"[{model_name}] Loaded in {time.time() - start:.2f}s")

        # Verify config
        print(
            f"[{model_name}] Config: layers={config.n_layer}, dim={config.n_embd}, heads={config.n_head}"
        )

        # Run inference
        text = "def factorial(n):"
        encoding = tokenizer.encode(text)
        input_ids = jnp.array([encoding.ids], dtype=jnp.int32)

        print(f"[{model_name}] Running inference on input: '{text}'")
        # Forward pass (SKIP path)
        output = model(input_ids, use_ttt=False)
        logits = output["logits"]

        # Calculate loss manually for last token
        # Predict "n"-like token or something reasonable?
        # Actually just check perplexity on a common phrase
        text_eval = "The quick brown fox jumps over the lazy dog."
        encoding_eval = tokenizer.encode(text_eval)
        input_ids_eval = jnp.array([encoding_eval.ids], dtype=jnp.int32)

        output_eval = model(input_ids_eval, use_ttt=False)
        logits_eval = output_eval["logits"]

        # Compute CE loss
        # Shift logits and labels
        shift_logits = logits_eval[..., :-1, :]
        shift_labels = input_ids_eval[..., 1:]

        loss_fn = lambda logits, labels: -jnp.take_along_axis(
            jax.nn.log_softmax(logits, axis=-1), labels[..., None], axis=-1
        ).mean()

        loss = loss_fn(shift_logits, shift_labels)
        ppl = jnp.exp(loss)

        print(f"[{model_name}] Loss: {loss:.4f}, Perplexity: {ppl:.2f}")

        # Basic sanity check on weights
        # Check if weights are non-zero and reasonable stats
        wte = model.base_model.wte.embedding[...]
        print(f"[{model_name}] WTE mean: {wte.mean():.6f}, std: {wte.std():.6f}")

        if abs(wte.mean()) < 1e-6 and wte.std() < 1e-6:
            print(f"⚠️ [{model_name}] WARNING: WTE seems to be all zeros!")

        if model_name == "gpt2-xl" and loss > 5.0:
            print(
                f"⚠️ [{model_name}] WARNING: Loss is unexpectedly high (>5.0). Typical is < 4.0."
            )

    except Exception as e:
        print(f"[{model_name}] ERROR: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    # Use CPU to avoid OOM or interference
    jax.config.update("jax_platform_name", "cpu")

    print("Checking GPT-2 (124M) as baseline...")
    check_model("gpt2")

    print("\nChecking GPT-2 XL (1.5B)...")
    check_model("gpt2-xl")
