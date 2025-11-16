"""
Dataset implementation for code data with multi-host sharding support.
"""

import numpy as np
import jax
import jax.numpy as jnp
from typing import Iterator, Optional, Dict
from datasets import load_dataset
from transformers import PreTrainedTokenizer


class CodeDataset:
    """
    Streaming dataset for code from The Stack v2 with multi-host sharding.

    Args:
        tokenizer: HuggingFace tokenizer
        split: Dataset split ('train', 'validation', 'test')
        language: Programming language (e.g., 'python', 'javascript')
        seq_length: Maximum sequence length
        chunk_size: Chunk size for TTT
        shard_across_hosts: If True, shard data across hosts for distributed training
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        split: str = "train",
        language: str = "python",
        seq_length: int = 8192,
        chunk_size: int = 4096,
        shard_across_hosts: bool = True,
    ):
        self.tokenizer = tokenizer
        self.split = split
        self.language = language
        self.seq_length = seq_length
        self.chunk_size = chunk_size
        self.shard_across_hosts = shard_across_hosts

        # Load dataset in streaming mode (The Stack v2)
        # Note: language should be capitalized (e.g., "Python", "Java")
        self.dataset = load_dataset(
            "bigcode/the-stack-v2",
            language.capitalize(),
            split=split,
            streaming=True,
        )

        # Shard across hosts for distributed training
        if self.shard_across_hosts:
            try:
                num_hosts = jax.process_count()
                host_id = jax.process_index()

                if num_hosts > 1:
                    self.dataset = self.dataset.shard(
                        num_shards=num_hosts,
                        index=host_id,
                    )
                    if host_id == 0:
                        print(f" Dataset sharded across {num_hosts} hosts")
                        print(f"  Host {host_id} processing shard {host_id}/{num_hosts}")
            except (RuntimeError, ValueError):
                # JAX distributed not initialized, single host mode
                pass

    def __iter__(self) -> Iterator[Dict[str, np.ndarray]]:
        """
        Iterate over tokenized code examples.

        Yields:
            Dictionary with:
                - input_ids: [seq_len] int array
                - attention_mask: [seq_len] bool array
                - chunks: [num_chunks, chunk_size] int array
        """
        for example in self.dataset:
            text = example["content"]

            # Tokenize
            encoded = self.tokenizer(
                text,
                truncation=True,
                max_length=self.seq_length,
                padding="max_length",
                return_tensors="np",
            )

            input_ids = encoded["input_ids"][0]
            attention_mask = encoded["attention_mask"][0].astype(bool)

            # Create chunks
            num_chunks = self.seq_length // self.chunk_size
            chunks = input_ids[:num_chunks * self.chunk_size].reshape(
                num_chunks, self.chunk_size
            )

            yield {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "chunks": chunks,
            }


def create_data_iterator(
    tokenizer: PreTrainedTokenizer,
    split: str = "train",
    batch_size: int = 8,
    seq_length: int = 8192,
    chunk_size: int = 4096,
    max_examples: Optional[int] = None,
) -> Iterator[Dict[str, jnp.ndarray]]:
    """
    Create batched data iterator that yields JAX arrays.

    Args:
        tokenizer: Tokenizer
        split: Dataset split
        batch_size: Batch size
        seq_length: Sequence length
        chunk_size: Chunk size for TTT
        max_examples: Maximum number of examples to load

    Yields:
        Dictionary with batched JAX arrays:
            - input_ids: [batch, seq_len]
            - attention_mask: [batch, seq_len]
            - chunks: [batch, num_chunks, chunk_size]
    """
    dataset = CodeDataset(
        tokenizer=tokenizer,
        split=split,
        seq_length=seq_length,
        chunk_size=chunk_size,
    )

    batch = {
        "input_ids": [],
        "attention_mask": [],
        "chunks": [],
    }

    count = 0
    for example in dataset:
        batch["input_ids"].append(example["input_ids"])
        batch["attention_mask"].append(example["attention_mask"])
        batch["chunks"].append(example["chunks"])

        if len(batch["input_ids"]) == batch_size:
            # Convert to JAX arrays
            yield {
                "input_ids": jnp.array(batch["input_ids"]),
                "attention_mask": jnp.array(batch["attention_mask"]),
                "chunks": jnp.array(batch["chunks"]),
            }

            # Reset batch
            batch = {"input_ids": [], "attention_mask": [], "chunks": []}

            count += batch_size
            if max_examples and count >= max_examples:
                break
