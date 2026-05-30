# HIV Subtype Classifier

![HIV-1 Subtype Classification](https://img.shields.io/badge/Task-Genomic%20Classification-blue)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-red)
![Models](https://img.shields.io/badge/Models-1D--CNN%20%7C%20BiLSTM%20%7C%20DNABERT-green)

A comprehensive deep learning pipeline for classifying HIV-1 *pol* gene sequences into their respective subtypes (A, B, C, D). This project serves as a final examination submission for **MSB7216: Deep Learning for Health Data**.

## Overview

The *pol* gene in the HIV-1 genome is highly conserved and is the primary target for antiretroviral therapies. Accurately classifying HIV subtypes from raw nucleotide sequences is critical for tracking epidemiological spread and drug resistance.

This repository explores multiple deep learning architectures to automatically extract genomic motifs without manual feature engineering (like k-mer counting). It rigorously addresses extreme class imbalance through Focal Loss optimization and utilizes Universal Saliency Maps for sequence-level explainability.

## Architectures Investigated
1. **MLP Baseline**: Simple feed-forward network on one-hot encoded sequences.
2. **1D-CNN**: Extracts spatial motifs and local nucleotide patterns.
3. **BiLSTM**: Captures bidirectional long-range genomic dependencies. *(Best Performing Model)*
4. **DNABERT (Transfer Learning)**: Fine-tunes a massive Transformer model pre-trained on the human genome to understand HIV sequences.

## Directory Structure
```
hiv-subtype-classifier/
├── data/
│   ├── raw/                 # Downloaded FASTA sequences (Git Ignored)
│   └── processed/           # Pickled tensors and labels (Git Ignored)
├── models/                  # Trained .pth checkpoints (Git Ignored)
├── notebooks/               # Jupyter notebooks (The Core Pipeline)
│   ├── 01_data_acquisition_and_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_baseline_model.ipynb
│   ├── 04_deep_learning_models.ipynb
│   ├── 05_model_evaluation_and_comparisons.ipynb
│   ├── 06_transfer_learning.ipynb
│   └── 07_deployment.ipynb
├── reports/                 # Evaluation figures, final reports, and slides
├── src/                     # Python source code for data loading and models
├── README.md                # Project landing page
├── requirements.txt         # Dependencies
└── instructions.txt         # Step-by-step execution guide
```

## Quick Start
Please see `instructions.txt` for step-by-step instructions on setting up the environment, downloading the data, and running the pipeline from scratch.

## Key Features
- **Pure Focal Loss Integration**: Elegantly handles extreme dataset imbalance (e.g., massive Subtype B overrepresentation) by dynamically weighting gradients instead of relying on destructive class weights.
- **Universal Saliency Explainability**: Automatically calculates Input x Gradient attention maps across the nucleotide sequence to highlight exactly which base-pairs drove the model's classification.
- **Dynamic Deployment**: A self-contained evaluation and deployment pipeline that automatically detects and loads the highest-performing architecture dynamically.

## Author
**MSB7216: Deep Learning for Health Data** – Final Project, May 2026
