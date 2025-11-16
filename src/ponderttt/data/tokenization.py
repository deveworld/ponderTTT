"""
Tokenization utilities.
"""

from transformers import AutoTokenizer
from typing import Optional


def get_tokenizer(
    model_name: str = "gpt2",
    padding_side: str = "right",
    add_special_tokens: bool = True,
) -> AutoTokenizer:
    """
    Load and configure tokenizer.

    Args:
        model_name: HuggingFace model name
        padding_side: Where to add padding tokens
        add_special_tokens: Whether to add special tokens

    Returns:
        Configured tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = padding_side

    return tokenizer
