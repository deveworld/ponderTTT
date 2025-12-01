"""
Dataset implementation for code data with multi-host sharding support.
"""

import gzip
import hashlib
import logging
import pickle
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import boto3
import jax
import jax.numpy as jnp
import numpy as np
from botocore import UNSIGNED
from botocore.client import Config
from datasets import load_dataset
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
        exclude_benchmarks: bool = True,
        max_connections: int = 50,
        num_workers: int = 32,
    ):
        self.tokenizer = tokenizer
        self.split = split
        self.language = language
        self.seq_length = seq_length
        self.chunk_size = chunk_size
        self.shard_across_hosts = shard_across_hosts
        self.exclude_benchmarks = exclude_benchmarks
        self.num_workers = num_workers
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
                max_pool_connections=max_connections,
            ),
        )

        # Initialize contamination filters if requested
        self.forbidden_strings = []
        if self.exclude_benchmarks:
            self._init_contamination_filters()

        # Load dataset in streaming mode (The Stack v2 train-full-ids)
        self.dataset = load_dataset(
            "bigcode/the-stack-v2-train-full-ids",
            "default",
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

    def _init_contamination_filters(self):
        """Load benchmark solutions to filter out from training data."""
        try:
            # Import here to avoid potential circular dependencies or setup issues
            from ..evaluation.benchmarks import HumanEvalBenchmark

            if jax.process_index() == 0:
                print("Loading HumanEval for decontamination...")
            
            he = HumanEvalBenchmark()
            count = 0
            for problem in he.problems:
                # Filter out canonical solutions if they are long enough to be unique
                if problem.canonical_solution and len(problem.canonical_solution.strip()) > 40:
                    self.forbidden_strings.append(problem.canonical_solution.strip())
                    count += 1
            
            if jax.process_index() == 0:
                print(f"Loaded {count} forbidden strings for decontamination.")
            
        except Exception as e:
            logging.warning(f"Failed to load benchmarks for decontamination: {e}")

    def _download_content(self, blob_id: str, src_encoding: str) -> str:
        """
        Download actual code content from Software Heritage S3 bucket.

        Args:
            blob_id: Software Heritage blob ID
            src_encoding: Source file encoding

        Returns:
            Decoded file content as string
        """
        try:
            # Direct boto3 access is faster than smart_open
            response = self.s3_client.get_object(
                Bucket="softwareheritage", 
                Key=f"content/{blob_id}"
            )
            content_gz = response["Body"].read()
            return gzip.decompress(content_gz).decode(src_encoding)
        except Exception:
            # logging.warning("Failed to download blob %s: %s", blob_id, exc)
            return ""

    def _download_and_tokenize(self, example: dict) -> tuple[np.ndarray, np.ndarray] | None:
        """
        Download and tokenize a single file.

        Args:
            example: Dataset example with blob_id and src_encoding

        Returns:
            Tuple of (input_ids, attention_mask) arrays or None if failed
        """
        # Download actual content from S3
        text = self._download_content(example["blob_id"], example["src_encoding"])

        # Skip empty or failed downloads
        if not text or len(text.strip()) == 0:
            return None

        # Contamination check
        if self.exclude_benchmarks and self.forbidden_strings:
            for forbidden in self.forbidden_strings:
                if forbidden in text:
                    return None

        # Tokenize using tokenizers library
        encoded = self.tokenizer.encode(text)

        input_ids = np.array(encoded.ids, dtype=np.int32)
        attention_mask = np.array(encoded.attention_mask, dtype=bool)

        return input_ids, attention_mask

    def _create_sequence_from_buffer(
        self, token_buffer: np.ndarray, mask_buffer: np.ndarray
    ) -> dict[str, np.ndarray]:
        """
        Create a training sequence from the token buffer.

        Args:
            token_buffer: Buffer of token IDs (at least seq_length tokens)
            mask_buffer: Buffer of attention masks

        Returns:
            Processed example dict with chunks
        """
        input_ids = token_buffer[: self.seq_length]
        attention_mask = mask_buffer[: self.seq_length]

        # Create chunks + per-chunk attention masks
        num_chunks = self.seq_length // self.chunk_size
        chunks = input_ids.reshape(num_chunks, self.chunk_size)
        chunk_attention = attention_mask.reshape(num_chunks, self.chunk_size)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "chunks": chunks,
            "chunk_attention_mask": chunk_attention,
        }

    def _process_example(self, example: dict) -> dict[str, np.ndarray] | None:
        """
        Process a single example (legacy method for compatibility).
        Now uses concatenation internally but still processes one file at a time.

        Args:
            example: Dataset example with blob_id and src_encoding

        Returns:
            Processed example dict or None if failed
        """
        result = self._download_and_tokenize(example)
        if result is None:
            return None

        input_ids, attention_mask = result

        # Truncate if longer than seq_length
        if len(input_ids) > self.seq_length:
            input_ids = input_ids[: self.seq_length]
            attention_mask = attention_mask[: self.seq_length]
        elif len(input_ids) < self.seq_length:
            # Pad if shorter (this will be avoided in concatenated mode)
            pad_length = self.seq_length - len(input_ids)
            pad_ids = np.full(pad_length, self.pad_token_id, dtype=np.int32)
            pad_mask = np.zeros(pad_length, dtype=bool)
            input_ids = np.concatenate([input_ids, pad_ids])
            attention_mask = np.concatenate([attention_mask, pad_mask])

        return self._create_sequence_from_buffer(input_ids, attention_mask)

    def __iter__(self) -> Iterator[dict[str, np.ndarray]]:
        """
        Iterate over tokenized code examples with parallel prefetching.

        Yields:
            Dictionary with:
                - input_ids: [seq_len] int array
                - attention_mask: [seq_len] bool array
                - chunks: [num_chunks, chunk_size] int array
        """
        def example_generator():
            for repo in self.dataset:
                if isinstance(repo, dict) and "files" in repo:
                    for file_info in repo["files"]:
                        if file_info.get("language") == self.language:
                            yield file_info

        # Use ThreadPoolExecutor to prefetch and process examples in parallel
        # This significantly speeds up streaming when cache_data=False
        # We use a bounded buffer to prevent memory exhaustion while keeping workers busy
        
        iterator = example_generator()
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            
            # Initial fill of the pipeline
            for _ in range(self.num_workers * 2):
                try:
                    item = next(iterator)
                    futures.append(executor.submit(self._process_example, item))
                except StopIteration:
                    break
            
            while futures:
                # Wait for the first completed future
                # Note: as_completed yields futures as they complete
                # We use a small batch approach to replenish the pool

                # Check for completed futures without blocking too long on any single one
                # But we need to yield results in order or out of order?
                # Out of order is fine for training data.
                
                # Simple strategy: wait for at least one, then collect all currently done
                # and replenish.
                
                # Using as_completed on the current set of futures
                # We need to be careful not to create a new as_completed iterator every loop
                # if we are modifying the list.
                
                # Better approach: Use a list of futures and check them, or use a queue.
                # Since we want to use ThreadPoolExecutor, let's use a simple loop with wait.
                from concurrent.futures import wait, FIRST_COMPLETED
                
                done, not_done = wait(futures, return_when=FIRST_COMPLETED)
                
                for future in done:
                    result = future.result()
                    if result is not None:
                        yield result
                    
                    # Submit a new task for each completed one to keep pool full
                    try:
                        item = next(iterator)
                        not_done.add(executor.submit(self._process_example, item))
                    except StopIteration:
                        pass
                
                futures = list(not_done)


def create_data_iterator(
    tokenizer: Tokenizer,
    split: str = "train",
    language: str = "Python",
    batch_size: int = 8,
    seq_length: int = 8192,
    chunk_size: int = 4096,
    max_examples: int | None = None,
    skip_examples: int = 0,
    cache_data: bool = True,
    num_workers: int = 8,
    cache_dir: str = ".cache/ponderttt",
    exclude_benchmarks: bool = True,
    concatenate_documents: bool = True,
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
        max_examples: Maximum number of examples to load (after skipping)
        skip_examples: Number of examples to skip before loading (for train/eval split)
        cache_data: If True, download all data before training (recommended for GPU)
        num_workers: Number of parallel workers for downloading (default: 8)
        cache_dir: Directory to cache downloaded data (default: .cache/ponderttt)
        exclude_benchmarks: If True, filter out examples containing HumanEval solutions
        concatenate_documents: If True, concatenate multiple files to fill seq_length
                              without padding (standard LM pretraining approach).
                              Files are separated by <|endoftext|> token.

    Yields:
        Dictionary with batched JAX arrays:
            - input_ids: [batch, seq_len]
            - attention_mask: [batch, seq_len]
            - chunks: [batch, num_chunks, chunk_size]

    Example:
        # For training: use first 10000 examples
        train_iter = create_data_iterator(..., max_examples=10000, skip_examples=0)

        # For evaluation: skip first 10000, use next 2000
        eval_iter = create_data_iterator(..., max_examples=2000, skip_examples=10000)
    """
    dataset = CodeDataset(
        tokenizer=tokenizer,
        split=split,
        language=language,
        seq_length=seq_length,
        chunk_size=chunk_size,
        exclude_benchmarks=exclude_benchmarks,
        max_connections=max(num_workers, 50),
        num_workers=num_workers,
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
            f"skip{skip_examples}_{language}_vocab{tokenizer.get_vocab_size()}_{tokenizer_hash}_{tokenizer_id_str}_"
            f"exclude{exclude_benchmarks}_concat{concatenate_documents}".encode()
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
        if concatenate_documents:
            print("Using document concatenation (no padding)")
        from tqdm import tqdm

        # Collect and process in parallel
        total_examples = max_examples if max_examples else batch_size * 100
        # For concatenation mode, we need more raw files to fill the buffer
        files_to_download = total_examples * 3 if concatenate_documents else total_examples
        total_to_scan = files_to_download + skip_examples

        if skip_examples > 0:
            print(f"Skipping first {skip_examples} files, then downloading ~{files_to_download} files...")
        else:
            print(f"Downloading ~{files_to_download} files in parallel...")
        from threading import Lock
        pbar_lock = Lock()

        # Get separator token for concatenation
        eos_token_id = tokenizer.token_to_id("<|endoftext|>")
        if eos_token_id is None:
            eos_token_id = tokenizer.token_to_id("</s>")
        if eos_token_id is None:
            eos_token_id = dataset.pad_token_id  # fallback

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            skipped_count = 0

            with tqdm(total=total_to_scan, desc="Scanning" if skip_examples > 0 else "Downloading") as pbar:
                def update_pbar(f):
                    with pbar_lock:
                        pbar.update(1)

                for repo in dataset.dataset:
                    if isinstance(repo, dict) and "files" in repo:
                        for file_info in repo["files"]:
                            if file_info.get("language") == dataset.language:
                                # Skip examples first
                                if skipped_count < skip_examples:
                                    skipped_count += 1
                                    pbar.update(1)
                                    continue

                                # Use _download_and_tokenize for concatenation mode
                                if concatenate_documents:
                                    future = executor.submit(dataset._download_and_tokenize, file_info)
                                else:
                                    future = executor.submit(dataset._process_example, file_info)
                                future.add_done_callback(update_pbar)
                                futures.append(future)

                                if len(futures) >= files_to_download:
                                    break
                    if len(futures) >= files_to_download:
                        break

            # Collect results
            print("Waiting for remaining downloads...")
            raw_tokens = []
            for f in futures:
                res = f.result()
                if res is not None:
                    raw_tokens.append(res)

        print(f"Successfully downloaded {len(raw_tokens)} files")

        # Process into examples
        if concatenate_documents:
            # Concatenate all tokens with separator
            print("Concatenating documents...")
            token_buffer = []
            mask_buffer = []

            for item in raw_tokens:
                if item is None:
                    continue
                input_ids, attention_mask = item

                # Add separator between documents
                if token_buffer:
                    token_buffer.append(np.array([eos_token_id], dtype=np.int32))
                    mask_buffer.append(np.array([True], dtype=bool))

                token_buffer.append(input_ids)
                mask_buffer.append(attention_mask)

            # Concatenate all buffers
            if token_buffer:
                all_tokens = np.concatenate(token_buffer)
                all_masks = np.concatenate(mask_buffer)
            else:
                all_tokens = np.array([], dtype=np.int32)
                all_masks = np.array([], dtype=bool)

            print(f"Total tokens after concatenation: {len(all_tokens)}")

            # Create sequences from concatenated buffer
            processed_examples = []
            num_chunks_per_seq = seq_length // chunk_size

            for start_idx in range(0, len(all_tokens) - seq_length + 1, seq_length):
                input_ids = all_tokens[start_idx : start_idx + seq_length]
                attention_mask = all_masks[start_idx : start_idx + seq_length]

                chunks = input_ids.reshape(num_chunks_per_seq, chunk_size)
                chunk_attention = attention_mask.reshape(num_chunks_per_seq, chunk_size)

                processed_examples.append({
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "chunks": chunks,
                    "chunk_attention_mask": chunk_attention,
                })

                if max_examples and len(processed_examples) >= max_examples:
                    break

            print(f"Created {len(processed_examples)} sequences (no padding)")
        else:
            # Legacy mode: each file is a separate example (may have padding)
            processed_examples = [r for r in raw_tokens if r is not None]

        print(f"Total sequences: {len(processed_examples)}")

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
