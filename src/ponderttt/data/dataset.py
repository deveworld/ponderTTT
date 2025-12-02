"""
Dataset implementation for code data with multi-host sharding support.
"""

import gzip
import hashlib
import json
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


def _get_cache_base_key(
    tokenizer: Tokenizer,
    split: str,
    language: str,
    seq_length: int,
    chunk_size: int,
    exclude_benchmarks: bool,
    concatenate_documents: bool,
) -> str:
    """Generate a short cache key based on data format parameters.

    Format: {language_abbrev}_{params_hash}
    Example: py_a1b2c3d4 (for Python with specific params)
    """
    # Language abbreviation (lowercase, max 4 chars)
    lang_abbrev = language.lower()[:4]

    # Hash all parameters for uniqueness
    try:
        tokenizer_serialized = tokenizer.to_str()
    except Exception:
        tokenizer_serialized = repr(tokenizer)

    params_str = (
        f"{split}_{language}_{seq_length}_{chunk_size}_"
        f"{tokenizer.get_vocab_size()}_{tokenizer_serialized}_"
        f"{exclude_benchmarks}_{concatenate_documents}"
    )
    params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]

    return f"{lang_abbrev}_{params_hash}"


def _find_compatible_cache(cache_dir: Path, base_key: str) -> tuple[Path | None, dict | None]:
    """
    Find existing cache files that are compatible with the base key.

    Returns:
        Tuple of (cache_path, metadata) or (None, None) if not found
    """
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        return None, None

    # Look for cache files with this base key
    pattern = f"{base_key}_*.pkl"
    cache_files = list(cache_dir.glob(pattern))

    if not cache_files:
        return None, None

    # Find the cache with the most data
    best_cache = None
    best_metadata = None
    best_count = 0

    for cache_file in cache_files:
        metadata_file = cache_file.with_suffix(".meta.json")
        if metadata_file.exists():
            try:
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                if metadata.get("total_sequences", 0) > best_count:
                    best_count = metadata["total_sequences"]
                    best_cache = cache_file
                    best_metadata = metadata
            except Exception:
                continue

    return best_cache, best_metadata


def _save_cache_with_metadata(
    cache_path: Path,
    data: dict,
    metadata: dict,
):
    """Save cache data and metadata."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Save data
    with open(cache_path, "wb") as f:
        pickle.dump(data, f)

    # Save metadata
    metadata_path = cache_path.with_suffix(".meta.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


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
        cache_dir_path = Path(cache_dir)

        # Generate base cache key (without max_examples)
        base_key = _get_cache_base_key(
            tokenizer, split, language, seq_length, chunk_size,
            exclude_benchmarks, concatenate_documents
        )

        # Include skip_examples in the key since different skip values need different data
        # Use 'k' suffix for thousands to keep filename short (e.g., s160k instead of s160000)
        if skip_examples >= 1000 and skip_examples % 1000 == 0:
            skip_str = f"s{skip_examples // 1000}k"
        else:
            skip_str = f"s{skip_examples}"
        cache_key = f"{base_key}_{skip_str}"

        # Target number of sequences needed
        total_needed = max_examples if max_examples else batch_size * 100

        # Find existing compatible cache
        existing_cache, existing_metadata = _find_compatible_cache(cache_dir_path, cache_key)

        # Check if existing cache has enough data
        if existing_cache and existing_metadata:
            cached_sequences = existing_metadata.get("total_sequences", 0)
            cached_files_scanned = existing_metadata.get("files_scanned", 0)

            if cached_sequences >= total_needed:
                # Cache has enough data - load and slice
                print(f"Loading cache with {cached_sequences} sequences (need {total_needed})...")
                with open(existing_cache, "rb") as f:
                    cache_data_dict = pickle.load(f)

                all_tokens = cache_data_dict["all_tokens"]
                all_masks = cache_data_dict["all_masks"]

                # Create batches from the required portion
                num_chunks_per_seq = seq_length // chunk_size
                cached_batches = []
                batch: dict[str, list] = {
                    "input_ids": [], "attention_mask": [],
                    "chunks": [], "chunk_attention_mask": [],
                }

                sequences_created = 0
                for start_idx in range(0, len(all_tokens) - seq_length + 1, seq_length):
                    if sequences_created >= total_needed:
                        break

                    input_ids = all_tokens[start_idx : start_idx + seq_length]
                    attention_mask = all_masks[start_idx : start_idx + seq_length]
                    chunks = input_ids.reshape(num_chunks_per_seq, chunk_size)
                    chunk_attention = attention_mask.reshape(num_chunks_per_seq, chunk_size)

                    batch["input_ids"].append(input_ids)
                    batch["attention_mask"].append(attention_mask)
                    batch["chunks"].append(chunks)
                    batch["chunk_attention_mask"].append(chunk_attention)
                    sequences_created += 1

                    if len(batch["input_ids"]) == batch_size:
                        cached_batches.append({
                            "input_ids": jnp.array(batch["input_ids"]),
                            "attention_mask": jnp.array(batch["attention_mask"]),
                            "chunks": jnp.array(batch["chunks"]),
                            "chunk_attention_mask": jnp.array(batch["chunk_attention_mask"]),
                        })
                        batch = {"input_ids": [], "attention_mask": [], "chunks": [], "chunk_attention_mask": []}

                if batch["input_ids"]:
                    cached_batches.append({
                        "input_ids": jnp.array(batch["input_ids"]),
                        "attention_mask": jnp.array(batch["attention_mask"]),
                        "chunks": jnp.array(batch["chunks"]),
                        "chunk_attention_mask": jnp.array(batch["chunk_attention_mask"]),
                    })

                print(f"Using {sequences_created} sequences from cache ({len(cached_batches)} batches)")
                return iter(cached_batches)
            else:
                # Cache exists but needs more data - will extend it
                print(f"Cache has {cached_sequences} sequences, need {total_needed}. Extending...")
                with open(existing_cache, "rb") as f:
                    cache_data_dict = pickle.load(f)

                existing_tokens = list(cache_data_dict.get("raw_tokens", []))
                start_from_file = cached_files_scanned
                print(f"Will continue downloading from file {start_from_file}...")
        else:
            # No existing cache
            existing_tokens = []
            start_from_file = 0
            cached_files_scanned = 0

        # Download data (either fresh or extending existing cache)
        if not existing_cache or existing_metadata.get("total_sequences", 0) < total_needed:
            print(f"Downloading with {num_workers} parallel workers...")
            if concatenate_documents:
                print("Using document concatenation (no padding)")
            from tqdm import tqdm
            from threading import Lock

            # Calculate how many more files we need
            # For concatenation mode, we need more raw files to fill the buffer
            if existing_metadata:
                # Extending existing cache - use metadata for accurate count
                cached_sequences = existing_metadata.get("total_sequences", 0)
                sequences_still_needed = total_needed - cached_sequences
            else:
                # Fresh download
                sequences_still_needed = total_needed
            # With concatenation, we need many more files because:
            # 1. Many code files are short (< seq_length tokens)
            # 2. Decontamination filters out files containing benchmark solutions
            # 3. Some files may be mostly whitespace/comments
            # Using 30x multiplier to be safe (empirically determined)
            files_to_download = sequences_still_needed * 30 if concatenate_documents else sequences_still_needed
            files_to_download = max(files_to_download, 10000)  # Download at least 10k files

            total_to_scan = start_from_file + files_to_download + skip_examples

            if skip_examples > 0 and start_from_file == 0:
                print(f"Skipping first {skip_examples} files, then downloading ~{files_to_download} files...")
            else:
                print(f"Downloading ~{files_to_download} more files starting from file {start_from_file}...")

            pbar_lock = Lock()

            # Get separator token for concatenation
            eos_token_id = tokenizer.token_to_id("<|endoftext|>")
            if eos_token_id is None:
                eos_token_id = tokenizer.token_to_id("</s>")
            if eos_token_id is None:
                eos_token_id = dataset.pad_token_id  # fallback

            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = []
                file_count = 0
                files_after_skip = 0

                desc = "Scanning" if skip_examples > 0 and start_from_file == 0 else "Downloading"
                with tqdm(total=total_to_scan, desc=desc) as pbar:
                    def update_pbar(f):
                        with pbar_lock:
                            pbar.update(1)

                    for repo in dataset.dataset:
                        if isinstance(repo, dict) and "files" in repo:
                            for file_info in repo["files"]:
                                if file_info.get("language") == dataset.language:
                                    file_count += 1

                                    # Skip examples first (for held-out evaluation)
                                    if file_count <= skip_examples:
                                        pbar.update(1)
                                        continue

                                    files_after_skip += 1

                                    # Skip already cached files
                                    if files_after_skip <= start_from_file:
                                        pbar.update(1)
                                        continue

                                    # Download new files
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
                new_raw_tokens = []
                for f in futures:
                    res = f.result()
                    if res is not None:
                        new_raw_tokens.append(res)

            print(f"Successfully downloaded {len(new_raw_tokens)} new files")
            total_files_scanned = start_from_file + len(futures)

            # Combine with existing tokens
            all_raw_tokens = existing_tokens + new_raw_tokens

            # Process into concatenated tokens
            if concatenate_documents:
                print("Concatenating documents...")
                token_buffer = []
                mask_buffer = []

                for item in all_raw_tokens:
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

                total_sequences = (len(all_tokens) - seq_length + 1) // seq_length if len(all_tokens) >= seq_length else 0
                print(f"Total tokens: {len(all_tokens)}, can create {total_sequences} sequences")

                # Save cache with raw tokens for future extension
                # Use 'k' suffix for thousands to keep filename short
                if total_sequences >= 1000 and total_sequences % 1000 == 0:
                    seq_str = f"{total_sequences // 1000}k"
                else:
                    seq_str = str(total_sequences)
                cache_path = cache_dir_path / f"{cache_key}_{seq_str}.pkl"
                cache_data_to_save = {
                    "all_tokens": all_tokens,
                    "all_masks": all_masks,
                    "raw_tokens": all_raw_tokens,  # Keep raw for extension
                }
                metadata = {
                    "total_sequences": total_sequences,
                    "total_tokens": len(all_tokens),
                    "files_scanned": total_files_scanned,
                    "raw_files_count": len(all_raw_tokens),
                    "seq_length": seq_length,
                    "chunk_size": chunk_size,
                    "batch_size": batch_size,
                    "skip_examples": skip_examples,
                }

                _save_cache_with_metadata(cache_path, cache_data_to_save, metadata)
                print(f"Saved cache to {cache_path}")

                # Remove old cache if we extended it
                if existing_cache and existing_cache != cache_path:
                    try:
                        existing_cache.unlink()
                        existing_cache.with_suffix(".meta.json").unlink()
                        print(f"Removed old cache {existing_cache}")
                    except Exception:
                        pass

                # Create batches
                num_chunks_per_seq = seq_length // chunk_size
                cached_batches = []
                batch_dict: dict[str, list] = {
                    "input_ids": [], "attention_mask": [],
                    "chunks": [], "chunk_attention_mask": [],
                }

                sequences_created = 0
                for start_idx in range(0, len(all_tokens) - seq_length + 1, seq_length):
                    if max_examples and sequences_created >= max_examples:
                        break

                    input_ids = all_tokens[start_idx : start_idx + seq_length]
                    attention_mask = all_masks[start_idx : start_idx + seq_length]
                    chunks = input_ids.reshape(num_chunks_per_seq, chunk_size)
                    chunk_attention = attention_mask.reshape(num_chunks_per_seq, chunk_size)

                    batch_dict["input_ids"].append(input_ids)
                    batch_dict["attention_mask"].append(attention_mask)
                    batch_dict["chunks"].append(chunks)
                    batch_dict["chunk_attention_mask"].append(chunk_attention)
                    sequences_created += 1

                    if len(batch_dict["input_ids"]) == batch_size:
                        cached_batches.append({
                            "input_ids": jnp.array(batch_dict["input_ids"]),
                            "attention_mask": jnp.array(batch_dict["attention_mask"]),
                            "chunks": jnp.array(batch_dict["chunks"]),
                            "chunk_attention_mask": jnp.array(batch_dict["chunk_attention_mask"]),
                        })
                        batch_dict = {"input_ids": [], "attention_mask": [], "chunks": [], "chunk_attention_mask": []}

                if batch_dict["input_ids"]:
                    cached_batches.append({
                        "input_ids": jnp.array(batch_dict["input_ids"]),
                        "attention_mask": jnp.array(batch_dict["attention_mask"]),
                        "chunks": jnp.array(batch_dict["chunks"]),
                        "chunk_attention_mask": jnp.array(batch_dict["chunk_attention_mask"]),
                    })

                print(f"Created {len(cached_batches)} batches ({sequences_created} sequences)")
                return iter(cached_batches)
            else:
                # Legacy mode: each file is a separate example (may have padding)
                processed_examples = [r for r in all_raw_tokens if r is not None]
                print(f"Total sequences: {len(processed_examples)}")

                # Create batches
                cached_batches = []
                batch_dict: dict[str, list] = {
                    "input_ids": [], "attention_mask": [],
                    "chunks": [], "chunk_attention_mask": [],
                }

                for example in processed_examples:
                    if example is None:
                        continue
                    batch_dict["input_ids"].append(example["input_ids"])
                    batch_dict["attention_mask"].append(example["attention_mask"])
                    batch_dict["chunks"].append(example["chunks"])
                    batch_dict["chunk_attention_mask"].append(example["chunk_attention_mask"])

                    if len(batch_dict["input_ids"]) == batch_size:
                        cached_batches.append({
                            "input_ids": jnp.array(batch_dict["input_ids"]),
                            "attention_mask": jnp.array(batch_dict["attention_mask"]),
                            "chunks": jnp.array(batch_dict["chunks"]),
                            "chunk_attention_mask": jnp.array(batch_dict["chunk_attention_mask"]),
                        })
                        batch_dict = {"input_ids": [], "attention_mask": [], "chunks": [], "chunk_attention_mask": []}

                if batch_dict["input_ids"]:
                    cached_batches.append({
                        "input_ids": jnp.array(batch_dict["input_ids"]),
                        "attention_mask": jnp.array(batch_dict["attention_mask"]),
                        "chunks": jnp.array(batch_dict["chunks"]),
                        "chunk_attention_mask": jnp.array(batch_dict["chunk_attention_mask"]),
                    })

                print(f"Created {len(cached_batches)} batches ready for training")
                return iter(cached_batches)
    else:
        return _batch_generator()
