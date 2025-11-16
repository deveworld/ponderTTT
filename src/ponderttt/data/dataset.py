"""
Dataset implementation for code data with multi-host sharding support.
"""

import numpy as np
import jax
import jax.numpy as jnp
from typing import Iterator, Optional, Dict
from datasets import load_dataset
from transformers import PreTrainedTokenizer
import boto3
from botocore import UNSIGNED
from botocore.client import Config
from smart_open import open as smart_open


class CodeDataset:
    """
    Streaming dataset for code from The Stack v2 (dedup) with multi-host sharding.

    Downloads actual code content from Software Heritage S3 bucket using unsigned requests.

    Args:
        tokenizer: HuggingFace tokenizer
        split: Dataset split ('train', 'validation', 'test')
        language: Programming language (e.g., 'Python', 'JavaScript')
        seq_length: Maximum sequence length
        chunk_size: Chunk size for TTT
        shard_across_hosts: If True, shard data across hosts for distributed training
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        split: str = "train",
        language: str = "Python",
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

        # Setup S3 client for unsigned requests (no AWS credentials needed)
        self.s3_client = boto3.client(
            's3',
            region_name='us-east-1',
            config=Config(signature_version=UNSIGNED)
        )

        # Load dataset in streaming mode (The Stack v2 dedup)
        self.dataset = load_dataset(
            "bigcode/the-stack-v2-dedup",
            language,
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
                        print(f"Dataset sharded across {num_hosts} hosts")
                        print(f"  Host {host_id} processing shard {host_id}/{num_hosts}")
            except (RuntimeError, ValueError):
                # JAX distributed not initialized, single host mode
                pass

    def _download_content(self, blob_id: str, src_encoding: str) -> str:
        """
        Download actual code content from Software Heritage S3 bucket.

        Args:
            blob_id: Software Heritage blob ID
            src_encoding: Source file encoding

        Returns:
            Decoded file content as string
        """
        s3_url = f"s3://softwareheritage/content/{blob_id}"
        try:
            with smart_open(
                s3_url,
                "rb",
                compression=".gz",
                transport_params={"client": self.s3_client}
            ) as f:
                content = f.read().decode(src_encoding)
            return content
        except Exception as e:
            # Skip files that fail to download
            return ""

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
            # Download actual content from S3
            text = self._download_content(
                example["blob_id"],
                example["src_encoding"]
            )

            # Skip empty or failed downloads
            if not text or len(text.strip()) == 0:
                continue

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
