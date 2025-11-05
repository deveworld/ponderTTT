"""
WikiText-2 and WikiText-103 dataset loading utilities.

Handles tokenization, batching, and data loading for language modeling.
"""

from typing import Dict, Optional, Tuple

import torch
from datasets import load_dataset
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
    """

    def __init__(
        self,
        split: str = "train",
        tokenizer_name: str = "gpt2",
        max_length: int = 512,
        dataset_name: str = "wikitext-2-raw-v1",
        cache_dir: Optional[str] = None,
    ):
        self.split = split
        self.max_length = max_length
        self.dataset_name = dataset_name

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # GPT-2 doesn't have a pad token by default
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load dataset
        print(f"Loading {dataset_name} {split} split...")
        dataset = load_dataset(
            "wikitext", dataset_name, split=split, cache_dir=cache_dir, trust_remote_code=True
        )

        # Tokenize
        print("Tokenizing dataset...")
        self.tokenized_data = self._tokenize_dataset(dataset)

        print(f"Dataset loaded: {len(self.tokenized_data)} sequences")

    def _tokenize_dataset(self, dataset) -> list:
        """Tokenize and chunk the dataset into fixed-length sequences."""
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

        # Chunk into sequences of max_length
        sequences = []
        for i in range(0, len(input_ids) - self.max_length, self.max_length):
            chunk = input_ids[i : i + self.max_length]
            if len(chunk) == self.max_length:
                sequences.append(chunk)

        return sequences

    def __len__(self) -> int:
        return len(self.tokenized_data)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Get a single item.

        Returns:
            Dictionary with 'input_ids' and 'labels' (shifted for LM)
        """
        input_ids = self.tokenized_data[index]

        # For language modeling, labels are shifted input_ids
        # input:  [w1, w2, w3, w4]
        # labels: [w2, w3, w4, w5]
        # But we'll handle the shift in the model/loss computation

        return {
            "input_ids": input_ids,
            "labels": input_ids.clone(),  # Model will handle the shift
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
    )

    val_dataset = WikiTextDataset(
        split="validation",
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        dataset_name=dataset_name,
        cache_dir=cache_dir,
    )

    test_dataset = WikiTextDataset(
        split="test",
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        dataset_name=dataset_name,
        cache_dir=cache_dir,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
