"""
Dataset implementation for code data with multi-host sharding support.
"""

import hashlib
import logging
import pickle
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import closing
from pathlib import Path
from typing import BinaryIO, cast

import boto3
import jax
import jax.numpy as jnp
import numpy as np
from botocore import UNSIGNED
from botocore.client import Config
from datasets import load_dataset
from smart_open import open as smart_open
from tokenizers import Tokenizer


class CodeDataset:
    """
    Streaming dataset for code from The Stack v2 (train-full-ids) with multi-host sharding.

    Downloads actual code content from Software Heritage S3 bucket using unsigned requests.

    Args:
        tokenizer: Tokenizer from tokenizers library
        split: Dataset split ('train', 'validation', 'test')
        language: Programming language (e.g., 'Python', 'JavaScript')
        seq_length: Maximum sequence length
        chunk_size: Chunk size for TTT
        shard_across_hosts: If True, shard data across hosts for distributed training
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
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
        self.pad_token_id = self.tokenizer.token_to_id("<|pad|>")
        if self.pad_token_id is None:
            raise ValueError(
                "Tokenizer is missing <|pad|> token; please add it before constructing the dataset."
            )

        if self.seq_length % self.chunk_size != 0:
            raise ValueError(
                f"seq_length ({self.seq_length}) must be divisible by chunk_size ({self.chunk_size})"
            )

        # Setup S3 client for unsigned requests (no AWS credentials needed)
        # Add timeouts to prevent hanging on slow downloads
        self.s3_client = boto3.client(
            "s3",
            region_name="us-east-1",
            config=Config(
                signature_version=UNSIGNED,
                connect_timeout=5,
                read_timeout=10,
                max_pool_connections=50,
            ),
        )

        # Load dataset in streaming mode (The Stack v2 train-full-ids)
        self.dataset = load_dataset(
            "bigcode/the-stack-v2-train-full-ids",
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
                    if hasattr(self.dataset, "shard") and not isinstance(
                        self.dataset, (dict,)
                    ):
                        self.dataset = self.dataset.shard(
                            num_shards=num_hosts,
                            index=host_id,
                        )
                        if host_id == 0:
                            print(f"Dataset sharded across {num_hosts} hosts")
                            print(
                                f"  Host {host_id} processing shard {host_id}/{num_hosts}"
                            )
                    elif host_id == 0:
                        print(
                            "Dataset implementation does not support sharding; proceeding without host sharding."
                        )
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
            with closing(
                cast(
                    BinaryIO,
                    smart_open(
                        s3_url,
                        "rb",
                        compression=".gz",
                        transport_params={"client": self.s3_client},
                    ),
                )
            ) as f:
                content = f.read().decode(src_encoding)
            return content
        except Exception as exc:
            logging.warning("Failed to download blob %s (%s): %s", blob_id, src_encoding, exc)
            # Skip files that fail to download or timeout
            return ""

    def _process_example(self, example: dict) -> dict[str, np.ndarray] | None:
        """
        Process a single example: download and tokenize.

        Args:
            example: Dataset example with blob_id and src_encoding

        Returns:
            Processed example dict or None if failed
        """
        # Download actual content from S3
        text = self._download_content(example["blob_id"], example["src_encoding"])

        # Skip empty or failed downloads
        if not text or len(text.strip()) == 0:
            return None

        # Tokenize using tokenizers library (returns encoding with attention mask)
        encoded = self.tokenizer.encode(text)

        input_ids = np.array(encoded.ids, dtype=np.int32)
        attention_mask = np.array(encoded.attention_mask, dtype=bool)

        # Truncate or pad to seq_length while keeping attention mask aligned
        if len(input_ids) > self.seq_length:
            input_ids = input_ids[: self.seq_length]
            attention_mask = attention_mask[: self.seq_length]
        elif len(input_ids) < self.seq_length:
            pad_length = self.seq_length - len(input_ids)
            pad_ids = np.full(pad_length, self.pad_token_id, dtype=np.int32)
            pad_mask = np.zeros(pad_length, dtype=bool)
            input_ids = np.concatenate([input_ids, pad_ids])
            attention_mask = np.concatenate([attention_mask, pad_mask])

        # Create chunks + per-chunk attention masks
        num_chunks = self.seq_length // self.chunk_size
        seq_len = num_chunks * self.chunk_size
        input_ids = input_ids[:seq_len]
        attention_mask = attention_mask[:seq_len]

        chunks = input_ids.reshape(num_chunks, self.chunk_size)
        chunk_attention = attention_mask.reshape(num_chunks, self.chunk_size)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "chunks": chunks,
            "chunk_attention_mask": chunk_attention,
        }

    def __iter__(self) -> Iterator[dict[str, np.ndarray]]:
        """
        Iterate over tokenized code examples.

        Yields:
            Dictionary with:
                - input_ids: [seq_len] int array
                - attention_mask: [seq_len] bool array
                - chunks: [num_chunks, chunk_size] int array
        """
        for example in self.dataset:
            if isinstance(example, dict):
                processed = self._process_example(example)
                if processed is not None:
                    yield processed


def create_data_iterator(
    tokenizer: Tokenizer,
    split: str = "train",
    language: str = "Python",
    batch_size: int = 8,
    seq_length: int = 8192,
    chunk_size: int = 4096,
    max_examples: int | None = None,
    cache_data: bool = True,
    num_workers: int = 8,
    cache_dir: str = ".cache/ponderttt",
) -> Iterator[dict[str, jnp.ndarray]]:
    """
    Create batched data iterator that yields JAX arrays.

    Args:
        tokenizer: Tokenizer
        split: Dataset split
        language: Programming language shard
        batch_size: Batch size
        seq_length: Sequence length
        chunk_size: Chunk size for TTT
        max_examples: Maximum number of examples to load
        cache_data: If True, download all data before training (recommended for GPU)
        num_workers: Number of parallel workers for downloading (default: 8)
        cache_dir: Directory to cache downloaded data (default: .cache/ponderttt)

    Yields:
        Dictionary with batched JAX arrays:
            - input_ids: [batch, seq_len]
            - attention_mask: [batch, seq_len]
            - chunks: [batch, num_chunks, chunk_size]
    """
    dataset = CodeDataset(
        tokenizer=tokenizer,
        split=split,
        language=language,
        seq_length=seq_length,
        chunk_size=chunk_size,
    )

    def _batch_generator():
        """Generator that creates batches from dataset."""
        batch = {
            "input_ids": [],
            "attention_mask": [],
            "chunks": [],
            "chunk_attention_mask": [],
        }

        count = 0
        for example in dataset:
            batch["input_ids"].append(example["input_ids"])
            batch["attention_mask"].append(example["attention_mask"])
            batch["chunks"].append(example["chunks"])
            batch["chunk_attention_mask"].append(example["chunk_attention_mask"])

            if len(batch["input_ids"]) == batch_size:
                # Convert to JAX arrays
                yield {
                    "input_ids": jnp.array(batch["input_ids"]),
                    "attention_mask": jnp.array(batch["attention_mask"]),
                    "chunks": jnp.array(batch["chunks"]),
                    "chunk_attention_mask": jnp.array(batch["chunk_attention_mask"]),
                }

                # Reset batch
                batch = {
                    "input_ids": [],
                    "attention_mask": [],
                    "chunks": [],
                    "chunk_attention_mask": [],
                }

                count += batch_size
                if max_examples and count >= max_examples:
                    break

        if batch["input_ids"]:
            yield {
                "input_ids": jnp.array(batch["input_ids"]),
                "attention_mask": jnp.array(batch["attention_mask"]),
                "chunks": jnp.array(batch["chunks"]),
                "chunk_attention_mask": jnp.array(batch["chunk_attention_mask"]),
            }

    # Cache all data upfront if enabled with parallel downloading
    if cache_data:
        # Create cache key based on parameters
        try:
            tokenizer_serialized = tokenizer.to_str()
        except Exception:
            tokenizer_serialized = repr(tokenizer)
        tokenizer_hash = hashlib.md5(tokenizer_serialized.encode()).hexdigest()
        tokenizer_id = getattr(tokenizer, "model", None)
        tokenizer_id_str = tokenizer_id.__class__.__name__ if tokenizer_id is not None else "unknown"

        cache_key = hashlib.md5(
            f"{split}_{batch_size}_{seq_length}_{chunk_size}_{max_examples}_"
            f"{language}_vocab{tokenizer.get_vocab_size()}_{tokenizer_hash}_{tokenizer_id_str}".encode()
        ).hexdigest()
        cache_path = Path(cache_dir) / f"{cache_key}.pkl"

        # Check if cache exists
        if cache_path.exists():
            print(f"Loading cached data from {cache_path}...")
            with open(cache_path, "rb") as f:
                cached_batches = pickle.load(f)
            print(f"Loaded {len(cached_batches)} batches from cache")
            return iter(cached_batches)

        # Download and cache if not exists
        print(f"Cache not found. Downloading with {num_workers} parallel workers...")
        from tqdm import tqdm

        # Collect raw examples first
        total_examples = max_examples if max_examples else batch_size * 100
        raw_examples = []

        for example in dataset.dataset:
            raw_examples.append(example)
            if len(raw_examples) >= total_examples:
                break

        print(f"Processing {len(raw_examples)} examples in parallel...")

        # Process examples in parallel
        processed_examples: list[dict[str, np.ndarray] | None] = [None] * len(
            raw_examples
        )
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all processing jobs
            futures = {
                executor.submit(dataset._process_example, example): i
                for i, example in enumerate(raw_examples)
            }

            # Collect results with progress bar
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Downloading"
            ):
                idx = futures[future]
                result = future.result()
                if result is not None:
                    processed_examples[idx] = result

        print(f"Successfully processed {len(processed_examples)} examples")

        # Create batches from processed examples
        cached_batches = []
        batch: dict[str, list] = {
            "input_ids": [],
            "attention_mask": [],
            "chunks": [],
            "chunk_attention_mask": [],
        }

        for example in processed_examples:
            if example is None:
                continue
            batch["input_ids"].append(example["input_ids"])
            batch["attention_mask"].append(example["attention_mask"])
            batch["chunks"].append(example["chunks"])
            batch["chunk_attention_mask"].append(example["chunk_attention_mask"])

            if len(batch["input_ids"]) == batch_size:
                cached_batches.append(
                    {
                        "input_ids": jnp.array(batch["input_ids"]),
                        "attention_mask": jnp.array(batch["attention_mask"]),
                        "chunks": jnp.array(batch["chunks"]),
                        "chunk_attention_mask": jnp.array(batch["chunk_attention_mask"]),
                    }
                )
                batch = {"input_ids": [], "attention_mask": [], "chunks": [], "chunk_attention_mask": []}

        if batch["input_ids"]:
            cached_batches.append(
                {
                    "input_ids": jnp.array(batch["input_ids"]),
                    "attention_mask": jnp.array(batch["attention_mask"]),
                    "chunks": jnp.array(batch["chunks"]),
                    "chunk_attention_mask": jnp.array(batch["chunk_attention_mask"]),
                }
            )

        print(f"Created {len(cached_batches)} batches ready for training")

        # Save to cache
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f_write:
            pickle.dump(cached_batches, f_write)
        print(f"Saved cache to {cache_path}")

        # Return iterator over cached data
        return iter(cached_batches)
    else:
        return _batch_generator()
