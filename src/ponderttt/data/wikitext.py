"""
WikiText-2 and WikiText-103 dataset loading utilities.

Handles tokenization, batching, and data loading for language modeling.
"""

from typing import Dict, Optional, Tuple

import torch
from datasets import load_dataset  # type: ignore[import-untyped]
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


class WikiTextDataset(Dataset):
    """
    WikiText dataset for language modeling.

    Args:
        split: Dataset split ('train', 'validation', 'test')
        tokenizer_name: Name of HuggingFace tokenizer (default: 'gpt2')
        max_length: Maximum sequence length (default: 512)
        dataset_name: Which WikiText to use ('wikitext-2-raw-v1' or 'wikitext-103-raw-v1')
        cache_dir: Directory to cache dataset
        drop_last: Whether to drop incomplete sequences at the end (default: True)
    """

    def __init__(
        self,
        split: str = "train",
        tokenizer_name: str = "gpt2",
        max_length: int = 512,
        dataset_name: str = "wikitext-2-raw-v1",
        cache_dir: Optional[str] = None,
        drop_last: bool = True,
    ):
        self.split = split
        self.max_length = max_length
        self.dataset_name = dataset_name
        self.drop_last = drop_last

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # GPT-2 doesn't have a pad token by default
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load dataset
        print(f"Loading {dataset_name} {split} split...")
        dataset = load_dataset(
            "wikitext", dataset_name, split=split, cache_dir=cache_dir
        )

        # Tokenize
        print("Tokenizing dataset...")
        self.input_ids = self._tokenize_dataset(dataset)

        # Calculate number of complete sequences
        self.num_sequences = len(self.input_ids) // self.max_length
        if not self.drop_last and len(self.input_ids) % self.max_length != 0:
            self.num_sequences += 1

        print(f"Dataset loaded: {self.num_sequences} sequences ({len(self.input_ids)} tokens total)")
        if self.drop_last:
            dropped = len(self.input_ids) % self.max_length
            if dropped > 0:
                print(f"  (dropped {dropped} trailing tokens due to drop_last=True)")

    def _tokenize_dataset(self, dataset) -> torch.Tensor:
        """
        Tokenize dataset into single continuous sequence.

        Returns:
            Single 1D tensor of all token IDs (no chunking yet)
        """
        # Concatenate all texts
        all_text = "\n\n".join([item["text"] for item in dataset if item["text"].strip()])

        # Tokenize entire corpus
        tokenized = self.tokenizer(
            all_text,
            return_tensors="pt",
            truncation=False,
            add_special_tokens=False,
        )

        input_ids = tokenized["input_ids"][0]
        return input_ids

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Get a single item by slicing from continuous token stream.

        Args:
            index: Sequence index

        Returns:
            Dictionary with 'input_ids' and 'labels' (shifted for LM)
        """
        # Calculate start position for this sequence
        start_idx = index * self.max_length

        # Check if this is the last (potentially incomplete) sequence
        if index == self.num_sequences - 1 and not self.drop_last:
            # Last sequence: take remaining tokens
            seq_input_ids = self.input_ids[start_idx:]
        else:
            # Regular sequence: take exactly max_length tokens
            seq_input_ids = self.input_ids[start_idx : start_idx + self.max_length]

        # For language modeling, labels are shifted input_ids
        # input:  [w1, w2, w3, w4]
        # labels: [w2, w3, w4, w5]
        # But we'll handle the shift in the model/loss computation

        return {
            "input_ids": seq_input_ids,
            "labels": seq_input_ids.clone(),  # Model will handle the shift
        }


def collate_fn(batch: list) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader.

    Args:
        batch: List of samples from WikiTextDataset

    Returns:
        Batched tensors
    """
    input_ids = torch.stack([item["input_ids"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])

    return {
        "input_ids": input_ids,
        "labels": labels,
    }


def get_wikitext_dataloaders(
    dataset_name: str = "wikitext-2-raw-v1",
    tokenizer_name: str = "gpt2",
    max_length: int = 512,
    batch_size: int = 8,
    num_workers: int = 0,
    cache_dir: Optional[str] = None,
    drop_last: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders for WikiText.

    Args:
        dataset_name: Which WikiText dataset to use
        tokenizer_name: HuggingFace tokenizer name
        max_length: Maximum sequence length
        batch_size: Batch size
        num_workers: Number of workers for DataLoader
        cache_dir: Cache directory for datasets
        drop_last: Whether to drop incomplete sequences at the end (default: True)

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = WikiTextDataset(
        split="train",
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        dataset_name=dataset_name,
        cache_dir=cache_dir,
        drop_last=drop_last,
    )

    val_dataset = WikiTextDataset(
        split="validation",
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        dataset_name=dataset_name,
        cache_dir=cache_dir,
        drop_last=drop_last,
    )

    test_dataset = WikiTextDataset(
        split="test",
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        dataset_name=dataset_name,
        cache_dir=cache_dir,
        drop_last=drop_last,
    )

    # Use pin_memory only if CUDA is available
    pin_memory = torch.cuda.is_available()

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader


def get_wikitext2_dataloaders(
    batch_size: int = 8,
    max_length: int = 256,
    **kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Alias for get_wikitext_dataloaders with wikitext-2 as default.

    Args:
        batch_size: Batch size
        max_length: Maximum sequence length
        **kwargs: Additional arguments passed to get_wikitext_dataloaders

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    return get_wikitext_dataloaders(
        dataset_name="wikitext-2-raw-v1",
        batch_size=batch_size,
        max_length=max_length,
        **kwargs
    )
