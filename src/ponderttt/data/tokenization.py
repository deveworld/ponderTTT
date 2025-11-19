"""
Tokenization utilities using HuggingFace tokenizers library directly.

Uses the tokenizers library without transformers dependency.
"""

from tokenizers import Tokenizer


def get_tokenizer(
    model_name: str = "gpt2",
    padding_side: str = "right",
    add_special_tokens: bool = True,
) -> Tokenizer:
    """
    Load and configure tokenizer using tokenizers library.

    Args:
        model_name: HuggingFace model name (e.g., "gpt2", "gpt2-medium")
        padding_side: Where to add padding tokens (not directly supported in tokenizers)
        add_special_tokens: Whether to add special tokens

    Returns:
        Configured tokenizer from tokenizers library

    Note:
        This uses the tokenizers library directly without transformers.
        The padding_side parameter is accepted for API compatibility but
        padding behavior should be controlled via enable_padding() and pad() methods.
    """
    # Load tokenizer from HuggingFace Hub
    tokenizer = Tokenizer.from_pretrained(model_name)

    # Ensure dedicated padding token exists
    pad_token = "<|pad|>"
    if tokenizer.token_to_id(pad_token) is None:
        tokenizer.add_special_tokens([pad_token])

    pad_token_id = tokenizer.token_to_id(pad_token)
    tokenizer.enable_padding(
        pad_id=pad_token_id,
        pad_token=pad_token,
    )

    return tokenizer
