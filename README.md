# Deep Learning Model for HIV-1 Subtype Classification Using the pol Gene

## 📌 Project Overview

This project develops and evaluates a deep learning model capable of accurately classifying HIV-1 sequences into their respective subtypes (**A, B, C, and D**) using raw, unaligned nucleotide data from the **pol gene**.

HIV-1 is characterized by extreme genetic diversity, with subtypes such as A and D playing a critical role in disease progression and treatment outcomes in East African clinical settings. While existing subtyping tools (REGA, COMET, jpHMM) require computationally intensive multiple sequence alignment as a prerequisite, and k-mer methods require manual feature engineering, this project strips alignment gaps from sequences and uses neural network architectures to **automatically learn biological motifs directly from raw, variable-length genomic data** — no alignment needed at inference time.

### Why This Matters

- **Epidemiological surveillance** — tracking subtype spread across regions
- **Treatment optimization** — differential drug efficacy across subtypes
- **Decentralized molecular surveillance** — near-instant classification without alignment or transmitting sensitive sequence data to remote servers
- **Resource-limited settings** — high-speed classification on standard hardware, no alignment software required

---

## 🧬 Dataset

**Source:** Subset of ~9,270 sequences from the dataset used by Solis-Reyes et al. (2018) in "An open-source k-mer based machine learning tool for fast and accurate subtyping of HIV-1 genomes" ([doi: 10.1371/journal.pone.0206409](https://doi.org/10.1371/journal.pone.0206409)).

Raw sequences downloaded from the [Los Alamos National Laboratory (LANL) HIV Sequence Database](https://www.hiv.lanl.gov/components/sequence/HIV/search/search.html).

| Parameter | Value |
|-----------|-------|
| Organism | HIV-1 |
| Subtypes | A, B, C, D (4-class classification) |
| Gene Region | pol (coding sequence) |
| Sequence Type | Nucleotide (DNA) |
| Problematic | Excluded |

> **Note:** This is NOT a Kaggle/Zindi dataset. LANL is the gold-standard curated database for HIV research.

**Experiment metadata:** [Kameris Experiments GitHub](https://github.com/stephensolis/kameris-experiments)

---

## 🏗️ Methodology

### Encoding: Gap-Stripped One-Hot (No K-mers)

Unlike Kameris (which uses k-mer frequency vectors), this project encodes raw nucleotide sequences directly as **one-hot tensors** of shape `(4, sequence_length)`. This preserves positional information and allows the neural network to learn discriminative motifs automatically. The 4 channels correspond to A, C, G, T. Ambiguous IUPAC bases and gaps are stripped or mapped to zero vectors.

### Models

| # | Model | Type | Purpose |
|---|-------|------|---------| 
| 1 | MLP | Baseline | Non-convolutional baseline on pooled one-hot features |
| 2 | 1D-CNN | Deep Learning (Primary) | Local motif detection via convolutional filters |
| 3 | BiLSTM | Deep Learning (Secondary) | Long-range dependency modeling |

### Pipeline

```
LANL FASTA → Parse & Clean → One-Hot Encode → Train/Val/Test Split
  → MLP Baseline
  → 1D-CNN (primary)
  → BiLSTM (secondary)
  → Evaluate (Accuracy, F1, ROC, Confusion Matrix)
```

---

## 📁 Repository Structure

```
hiv-subtype-classifier/
├── data/
│   ├── raw/                    # Raw FASTA files from LANL
│   └── processed/              # Encoded/preprocessed pickles
├── notebooks/
│   ├── 01_data_acquisition_and_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_baseline_model.ipynb
│   ├── 04_deep_learning_models.ipynb
│   └── 05_evaluation_and_analysis.ipynb
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # FASTA parsing, encoding, gap stripping
│   ├── models.py               # MLP, CNN, BiLSTM architectures (4-channel)
│   ├── train.py                # Training loop with early stopping
│   ├── evaluate.py             # Metrics, confusion matrix, ROC
│   └── utils.py                # Helper functions
├── models/                     # Saved model checkpoints
├── reports/
│   └── figures/                # Generated plots
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/jnagawa/hiv-subtype-classifier.git
cd hiv-subtype-classifier
pip install -r requirements.txt
```

### 2. Download Data
Follow the instructions in `notebooks/01_data_acquisition_and_exploration.ipynb` to download pol gene sequences from the LANL HIV database.

### 3. Run on Google Colab
Open the notebooks in Google Colab with a **T4 GPU** runtime:
- Upload the project to Google Drive
- Open notebooks from `notebooks/` folder
- Run cells sequentially (01 → 02 → 03 → 04 → 05)

---

## 📊 Evaluation Metrics

- Accuracy (overall and per-class)
- Precision, Recall, F1-score (macro-averaged)
- Confusion Matrix
- ROC Curves (one-vs-rest)
- AUC Scores

---

## ⚖️ Ethical Considerations

- All sequences are publicly available from LANL with no patient-identifiable information
- Geographic bias: subtype B is overrepresented (North America/Europe focus in sequencing)
- Class imbalance handled via Focal Loss with class weights
- Model limitations acknowledged for sequences outside the 4 target subtypes

---

## 📚 References

1. Solis-Reyes S, Avino M, Poon A, Kari L (2018). An open-source k-mer based machine learning tool for fast and accurate subtyping of HIV-1 genomes. *PLoS ONE* 13(11): e0206409. https://doi.org/10.1371/journal.pone.0206409
2. Los Alamos HIV Database: https://www.hiv.lanl.gov/
3. Kameris Experiments: https://github.com/stephensolis/kameris-experiments

---

## 👤 Author

**MSB7216: Deep Learning for Health Data** — Final Project, May 2026

---

## 📄 License

This project is for educational purposes. Sequence data is subject to LANL database usage terms.
