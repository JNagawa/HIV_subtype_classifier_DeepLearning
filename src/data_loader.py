"""
data_loader.py — FASTA Parsing, Metadata Extraction, and Dataset Creation
===========================================================================
Handles loading HIV-1 pol gene sequences from LANL FASTA files, extracting
metadata from sequence headers, and creating PyTorch datasets for model training.

Designed for the pol-gene-only, 4-class (A, B, C, D) classification task
using one-hot encoding of raw nucleotide sequences.

Key design decision: alignment gaps are stripped from sequences so the model
operates on raw, unaligned nucleotides — no multiple sequence alignment
required at inference time. This differentiates our approach from traditional
tools (REGA, COMET, jpHMM) that require pre-aligned input.
"""

import os
import re
import numpy as np
import pandas as pd
from collections import Counter
from Bio import SeqIO
from torch.utils.data import Dataset, DataLoader
import torch


# ============================================================================
# FASTA Parsing & Metadata Extraction
# ============================================================================


def parse_lanl_header(header):
    """
    Parse a LANL HIV database FASTA header to extract metadata.

    LANL headers typically follow the format:
    subtype.country.year.accession.patient_id

    But formats vary, so we handle multiple patterns.

    Parameters
    ----------
    header : str
        The FASTA header string (without '>')

    Returns
    -------
    dict
        Dictionary with extracted metadata fields
    """
    parts = header.split(".")

    metadata = {
        "raw_header": header,
        "subtype": parts[0] if len(parts) > 0 else "Unknown",
        "country": parts[1] if len(parts) > 1 else "Unknown",
        "year": parts[2] if len(parts) > 2 else "Unknown",
        "accession": parts[3] if len(parts) > 3 else "Unknown",
        "name": parts[4] if len(parts) > 4 else header,
    }

    return metadata


def load_fasta_file(filepath):
    """
    Load sequences from a FASTA file and extract metadata.

    Parameters
    ----------
    filepath : str
        Path to the FASTA file

    Returns
    -------
    list of dict
        List of dictionaries containing sequence and metadata
    """
    records = []

    for record in SeqIO.parse(filepath, "fasta"):
        seq_str = str(record.seq).upper()
        metadata = parse_lanl_header(record.description)
        metadata["sequence"] = seq_str
        metadata["seq_length"] = len(seq_str)
        metadata["accession_id"] = record.id

        records.append(metadata)

    print(f"Loaded {len(records)} sequences from {os.path.basename(filepath)}")
    return records


def load_all_fasta_files(data_dir, file_pattern=None):
    """
    Load all FASTA files from a directory.

    Parameters
    ----------
    data_dir : str
        Directory containing FASTA files
    file_pattern : str, optional
        Regex pattern to match filenames. If None, loads all .fasta/.fa files.

    Returns
    -------
    pd.DataFrame
        DataFrame with all sequences and metadata
    """
    all_records = []

    for filename in sorted(os.listdir(data_dir)):
        if not (
            filename.endswith(".fasta")
            or filename.endswith(".fa")
            or filename.endswith(".fas")
        ):
            continue

        if file_pattern and not re.search(file_pattern, filename):
            continue

        filepath = os.path.join(data_dir, filename)
        records = load_fasta_file(filepath)
        all_records.extend(records)

    df = pd.DataFrame(all_records)
    print(f"\nTotal sequences loaded: {len(df)}")

    if "subtype" in df.columns:
        print(f"Subtypes found: {df['subtype'].value_counts().to_dict()}")

    return df


# ============================================================================
# Sequence Preprocessing — Gap Stripping
# ============================================================================


def strip_gaps(sequence):
    """
    Remove alignment gap characters from a nucleotide sequence.

    LANL sequences are pre-aligned to a fixed length (4,259 bp for pol gene)
    using '-' gap characters. Stripping these produces the raw, unaligned
    sequence of variable length, enabling alignment-free classification.

    Parameters
    ----------
    sequence : str
        Aligned nucleotide sequence potentially containing '-' gaps

    Returns
    -------
    str
        Raw sequence with all gap characters removed
    """
    return sequence.replace("-", "").replace(".", "")


# ============================================================================
# Subtype Label Processing
# ============================================================================

# Map of subtype labels to standardized 4-class labels: A, B, C, D
SUBTYPE_MAPPING = {
    "A1": "A",
    "A": "A",
    "B": "B",
    "C": "C",
    "D": "D",
}


def standardize_subtype(subtype_str):
    """
    Standardize subtype labels to our 4 target classes: A, B, C, D.

    Sequences with subtypes not in {A, A1, B, C, D} are excluded (returns None).

    Parameters
    ----------
    subtype_str : str
        Raw subtype label from LANL

    Returns
    -------
    str or None
        Standardized label ('A', 'B', 'C', or 'D'), or None if excluded
    """
    subtype_str = subtype_str.strip()

    # Direct mapping
    if subtype_str in SUBTYPE_MAPPING:
        return SUBTYPE_MAPPING[subtype_str]

    # Unknown or not in our target — exclude
    return None


def create_label_mapping(df):
    """
    Create integer label mapping for classification.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'subtype_class' column

    Returns
    -------
    dict, dict
        label_to_idx and idx_to_label mappings
    """
    classes = sorted(df["subtype_class"].unique())
    label_to_idx = {label: idx for idx, label in enumerate(classes)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}

    print(f"Classes ({len(classes)}): {classes}")
    print(f"Label mapping: {label_to_idx}")

    return label_to_idx, idx_to_label


# ============================================================================
# PyTorch Dataset
# ============================================================================


class HIVSequenceDataset(Dataset):
    """
    PyTorch Dataset for HIV-1 pol gene sequences.

    Supports two encoding schemes:
    - 'onehot': One-hot encoding (4 channels × sequence length) — for CNNs/MLPs
    - 'label': Integer encoding (for embedding layers in LSTMs)

    Sequences are expected to be gap-stripped (raw, unaligned). Each sequence
    is encoded at its natural length; batching with dynamic padding is handled
    by the collate_fn, not by this dataset.

    Parameters
    ----------
    sequences : list of str
        Nucleotide sequences (gap-stripped, variable length)
    labels : list of int
        Integer class labels
    max_length : int or None
        Safety cap for truncation. If None, no truncation is applied.
    encoding : str
        Encoding method: 'onehot' or 'label'
    """

    # Nucleotide to index mapping — 4 channels, no gap
    NUC_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}
    NUM_CHANNELS = 4  # A, C, G, T

    def __init__(self, sequences, labels, max_length=None, encoding="onehot"):
        self.sequences = sequences
        self.labels = labels
        self.max_length = max_length
        self.encoding = encoding

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]

        # Apply safety cap truncation if set
        if self.max_length is not None:
            seq = seq[: self.max_length]

        if self.encoding == "onehot":
            encoded = self._onehot_encode(seq)
        elif self.encoding == "label":
            encoded = self._label_encode(seq)
        else:
            raise ValueError(f"Unknown encoding: {self.encoding}")

        return encoded, torch.tensor(label, dtype=torch.long)

    def _onehot_encode(self, sequence):
        """
        One-hot encode a nucleotide sequence.

        Returns tensor of shape (4, seq_len) — variable length per sequence.
        """
        seq_len = len(sequence)
        encoded = np.zeros((self.NUM_CHANNELS, seq_len), dtype=np.float32)

        for i, nuc in enumerate(sequence):
            idx = self.NUC_TO_IDX.get(nuc.upper(), -1)
            if idx >= 0:  # Encode A, C, G, T
                encoded[idx, i] = 1.0

        return torch.tensor(encoded, dtype=torch.float32)

    def _label_encode(self, sequence):
        """
        Label encode a nucleotide sequence.

        Returns tensor of shape (seq_len,) with values 0-3 (A,C,G,T)
        and 4 for unknown nucleotides.
        """
        seq_len = len(sequence)
        encoded = np.full(seq_len, 4, dtype=np.int64)  # 4 = unknown

        for i, nuc in enumerate(sequence):
            encoded[i] = self.NUC_TO_IDX.get(nuc.upper(), 4)

        return torch.tensor(encoded, dtype=torch.long)


def collate_variable_length(batch, encoding="onehot"):
    """
    Custom collate function for variable-length sequences.

    Pads each batch dynamically to the longest sequence in that batch,
    rather than using a fixed global maximum. This is more memory-efficient
    and avoids wasting computation on padding positions.

    Parameters
    ----------
    batch : list of (tensor, tensor)
        List of (encoded_sequence, label) tuples from HIVSequenceDataset
    encoding : str
        'onehot' or 'label' — determines padding strategy

    Returns
    -------
    tuple of (torch.Tensor, torch.Tensor)
        Padded sequences and labels
    """
    sequences, labels = zip(*batch)
    labels = torch.stack(labels)

    if encoding == "onehot":
        # sequences are (num_channels, seq_len) — pad along dim=1
        num_channels = sequences[0].shape[0]
        max_len = max(s.shape[1] for s in sequences)
        padded = torch.zeros(len(sequences), num_channels, max_len)
        for i, s in enumerate(sequences):
            padded[i, :, : s.shape[1]] = s
    else:
        # sequences are (seq_len,) — pad with 4 (padding token)
        max_len = max(s.shape[0] for s in sequences)
        padded = torch.full((len(sequences), max_len), 4, dtype=torch.long)
        for i, s in enumerate(sequences):
            padded[i, : s.shape[0]] = s

    return padded, labels


def collate_onehot(batch):
    """Collate function for one-hot encoded variable-length sequences."""
    return collate_variable_length(batch, encoding="onehot")


def collate_label(batch):
    """Collate function for label-encoded variable-length sequences."""
    return collate_variable_length(batch, encoding="label")


# ============================================================================
# Data Pipeline Functions
# ============================================================================


def prepare_data(
    df,
    label_to_idx,
    max_length=None,
    encoding="onehot",
    test_size=0.15,
    val_size=0.15,
    random_state=42,
    batch_size=32,
    strip_alignment_gaps=True,
):
    """
    Full data preparation pipeline: strip gaps, split, encode, create DataLoaders.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'sequence' and 'subtype_class' columns
    label_to_idx : dict
        Mapping of class labels to integer indices
    max_length : int or None
        Safety cap for truncation. None = no truncation.
    encoding : str
        Encoding method: 'onehot' or 'label'
    test_size : float
        Fraction for test set
    val_size : float
        Fraction for validation set
    random_state : int
        Random seed
    batch_size : int
        Batch size for DataLoaders
    strip_alignment_gaps : bool
        If True, remove alignment gap characters before encoding.
        This produces variable-length unaligned sequences.

    Returns
    -------
    dict
        Dictionary with 'train', 'val', 'test' DataLoaders and metadata
    """
    from sklearn.model_selection import train_test_split

    sequences = df["sequence"].tolist()

    # Strip alignment gaps to produce raw unaligned sequences
    if strip_alignment_gaps:
        sequences = [strip_gaps(seq) for seq in sequences]
        lengths = [len(s) for s in sequences]
        print(
            f"Gap-stripped sequences: length range {min(lengths)}–{max(lengths)} bp "
            f"(mean: {np.mean(lengths):.0f})"
        )

    labels = [label_to_idx[s] for s in df["subtype_class"]]

    # First split: train+val vs test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        sequences,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    # Second split: train vs val
    val_frac = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=val_frac,
        random_state=random_state,
        stratify=y_trainval,
    )

    print(f"Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    # Create datasets (variable-length, no fixed padding)
    train_dataset = HIVSequenceDataset(X_train, y_train, max_length, encoding)
    val_dataset = HIVSequenceDataset(X_val, y_val, max_length, encoding)
    test_dataset = HIVSequenceDataset(X_test, y_test, max_length, encoding)

    # Select collate function for dynamic per-batch padding
    collate_fn = collate_onehot if encoding == "onehot" else collate_label

    # Create DataLoaders with dynamic padding
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # Compute class weights for imbalanced data
    num_classes = len(label_to_idx)
    class_weights = torch.ones(num_classes, dtype=torch.float32)
    print(f"Class weights: {class_weights.tolist()}")

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "num_classes": num_classes,
        "max_length": max_length,
    }


def create_dataloaders(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    max_length=None,
    encoding="onehot",
    batch_size=32,
    strip_alignment_gaps=True,
):
    """
    Create DataLoaders from pre-split data (e.g., loaded from preprocessed_data.pkl).

    Use this when train/val/test splits already exist (from Notebook 02).
    For splitting from scratch, use prepare_data() instead.

    Parameters
    ----------
    X_train, X_val, X_test : list of str
        Nucleotide sequences for each split
    y_train, y_val, y_test : list of int
        Integer class labels for each split
    max_length : int or None
        Safety cap for truncation. None = no truncation.
    encoding : str
        Encoding method: 'onehot' or 'label'
    batch_size : int
        Batch size for DataLoaders
    strip_alignment_gaps : bool
        If True, remove alignment gap characters before encoding.

    Returns
    -------
    tuple of (DataLoader, DataLoader, DataLoader)
        train_loader, val_loader, test_loader
    """
    # Strip alignment gaps if requested
    if strip_alignment_gaps:
        X_train = [strip_gaps(s) for s in X_train]
        X_val = [strip_gaps(s) for s in X_val]
        X_test = [strip_gaps(s) for s in X_test]
        print(
            f"Gap-stripped: train lengths {min(len(s) for s in X_train)}–"
            f"{max(len(s) for s in X_train)} bp"
        )

    # Select collate function for dynamic per-batch padding
    collate_fn = collate_onehot if encoding == "onehot" else collate_label

    train_loader = DataLoader(
        HIVSequenceDataset(X_train, y_train, max_length, encoding),
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        HIVSequenceDataset(X_val, y_val, max_length, encoding),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        HIVSequenceDataset(X_test, y_test, max_length, encoding),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    print(
        f"DataLoaders ready: train={len(train_loader)}, "
        f"val={len(val_loader)}, test={len(test_loader)} batches"
    )

    return train_loader, val_loader, test_loader
