"""
utils.py — Helper Functions
============================
Seed setting, device detection, sequence statistics, and visualization helpers.
Designed for pol-gene-only, 4-class (A, B, C, D) classification.
"""

import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")


def get_device():
    """Detect best available device (GPU/CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB"
        )
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def sequence_stats(df, save_dir=None):
    """
    Print and plot comprehensive sequence statistics.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'sequence', 'subtype_class', 'seq_length' columns
    save_dir : str, optional
        Directory to save plots
    """
    print(f"\n{'='*50}")
    print(f"Dataset Statistics")
    print(f"{'='*50}")
    print(f"Total sequences: {len(df)}")
    print(
        f"Sequence length: {df['seq_length'].min()} - {df['seq_length'].max()} "
        f"(mean: {df['seq_length'].mean():.0f}, median: {df['seq_length'].median():.0f})"
    )

    if "subtype_class" in df.columns:
        print(f"\nSubtype distribution:")
        for subtype, count in df["subtype_class"].value_counts().items():
            pct = 100 * count / len(df)
            print(f"  {subtype}: {count} ({pct:.1f}%)")

    # Nucleotide composition
    print(f"\nNucleotide composition (first 100 sequences):")
    sample = df["sequence"].head(100)
    all_nucs = "".join(sample)
    total = len(all_nucs)
    for nuc in ["A", "C", "G", "T", "N"]:
        count = all_nucs.count(nuc)
        print(f"  {nuc}: {count/total*100:.1f}%")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        _plot_distributions(df, save_dir)


def _plot_distributions(df, save_dir):
    """Create distribution plots for EDA."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Subtype distribution
    if "subtype_class" in df.columns:
        counts = df["subtype_class"].value_counts()
        colors = sns.color_palette("husl", len(counts))
        axes[0].bar(counts.index, counts.values, color=colors)
        axes[0].set_title("Subtype Distribution", fontsize=13)
        axes[0].set_ylabel("Count")
        for i, (idx, v) in enumerate(counts.items()):
            axes[0].text(i, v + 5, str(v), ha="center", fontsize=9)

    # 2. Sequence length distribution
    axes[1].hist(
        df["seq_length"], bins=50, color="steelblue", edgecolor="white", alpha=0.8
    )
    axes[1].set_title("Sequence Length Distribution (pol)", fontsize=13)
    axes[1].set_xlabel("Length (bp)")
    axes[1].set_ylabel("Count")
    axes[1].axvline(
        df["seq_length"].median(),
        color="red",
        linestyle="--",
        label=f"Median: {df['seq_length'].median():.0f}",
    )
    axes[1].legend()

    # 3. Length by subtype
    if "subtype_class" in df.columns:
        df.boxplot(column="seq_length", by="subtype_class", ax=axes[2])
        axes[2].set_title("Sequence Length by Subtype", fontsize=13)
        axes[2].set_xlabel("Subtype")
        axes[2].set_ylabel("Length (bp)")
        plt.sca(axes[2])
        plt.xticks(rotation=0)

    plt.suptitle(
        "HIV-1 pol Gene Dataset — Exploratory Data Analysis", fontsize=15, y=1.02
    )
    plt.tight_layout()
    plt.savefig(f"{save_dir}/eda_distributions.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved EDA plots to {save_dir}/eda_distributions.png")


def count_parameters(model):
    """Count and display model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total:,} | Trainable: {trainable:,}")
    return total, trainable
