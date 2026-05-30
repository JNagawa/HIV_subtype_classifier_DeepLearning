"""
app.py — Gradio Deployment for HIV-1 Subtype Classifier
=========================================================
A web interface where users can paste or upload an HIV-1 nucleotide
sequence and get subtype predictions with confidence scores.

Uses Monte Carlo Dropout for uncertainty estimation — predictions with
high variance across multiple forward passes are flagged as indeterminate
rather than displaying a potentially false diagnosis.

No sequence alignment required — works on raw, unaligned nucleotide sequences.

Run: python app.py
Or in Colab: exec(open('app.py').read())
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gradio as gr
from src.models import HIV_CNN

# ============================================================
# Configuration — UPDATE THESE PATHS
# ============================================================

# For Colab: use Google Drive path
# MODEL_PATH = '/content/drive/MyDrive/Deep Learning/hiv-subtype-classifier/models/HIV_CNN_best.pth'

# For local: use relative path
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models',
                          'HIV_CNN_best.pth')

# Must match training configuration
NUM_CLASSES = 4
IDX_TO_LABEL = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
NUC_TO_IDX = {'A': 0, 'C': 1, 'G': 2, 'T': 3}  # 4-channel, no gap
NUM_CHANNELS = 4

# MC Dropout settings
MC_PASSES = 10            # Number of stochastic forward passes
UNCERTAINTY_STD = 0.15    # Max std deviation threshold to flag uncertainty
MIN_CONFIDENCE = 0.70     # Minimum confidence threshold

# ============================================================
# Load Model
# ============================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = HIV_CNN(NUM_CLASSES).to(device)

if os.path.exists(MODEL_PATH):
    ckpt = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt['state_dict'])
    print(f"Model loaded from {MODEL_PATH}")
else:
    print(f"WARNING: Model file not found at {MODEL_PATH}")
    print("The app will run but predictions will be random.")

# ============================================================
# Prediction Functions
# ============================================================

def encode_sequence(seq_text):
    """
    Clean and one-hot encode a nucleotide sequence.
    
    Works on raw, unaligned sequences — no alignment gaps needed.
    Uses 4-channel encoding (A, C, G, T).
    """
    # Clean: remove whitespace, numbers, FASTA header lines
    lines = seq_text.strip().split('\n')
    seq = ''
    for line in lines:
        line = line.strip()
        if line.startswith('>'):
            continue  # Skip FASTA headers
        seq += ''.join(c for c in line.upper() if c in 'ACGTNRYSWKMBDHV-')

    # Strip alignment gaps (if present) and replace ambiguous bases
    seq = seq.replace('-', '').replace('.', '')
    seq = ''.join(c if c in 'ACGT' else 'N' for c in seq)

    if len(seq) < 100:
        return None, "Sequence too short. Please provide at least 100 nucleotides."

    # One-hot encode at natural length (no fixed padding)
    seq_len = len(seq)
    encoded = np.zeros((NUM_CHANNELS, seq_len), dtype=np.float32)
    for i, nuc in enumerate(seq):
        idx = NUC_TO_IDX.get(nuc)
        if idx is not None:
            encoded[idx, i] = 1.0

    return torch.tensor(encoded).unsqueeze(0), None


def enable_mc_dropout(model):
    """
    Enable dropout layers during inference for Monte Carlo Dropout.
    
    Only activates Dropout layers — BatchNorm stays in eval mode
    to use running statistics (not batch statistics).
    """
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def mc_dropout_predict(encoded, n_passes=MC_PASSES):
    """
    Run multiple stochastic forward passes with dropout active.
    
    Returns the mean prediction, standard deviation across passes,
    and a flag indicating whether the prediction is uncertain.
    
    Parameters
    ----------
    encoded : torch.Tensor
        One-hot encoded sequence of shape (1, 4, seq_len)
    n_passes : int
        Number of stochastic forward passes
    
    Returns
    -------
    mean_probs : np.ndarray
        Mean class probabilities across passes
    std_probs : np.ndarray
        Standard deviation of class probabilities
    is_uncertain : bool
        True if prediction should be flagged as indeterminate
    """
    model.eval()             # Put BatchNorm in eval mode
    enable_mc_dropout(model)  # But keep Dropout active
    
    all_probs = []
    encoded = encoded.to(device)
    
    for _ in range(n_passes):
        with torch.no_grad():
            logits = model(encoded)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            all_probs.append(probs)
    
    all_probs = np.array(all_probs)  # (n_passes, num_classes)
    mean_probs = all_probs.mean(axis=0)
    std_probs = all_probs.std(axis=0)
    
    max_confidence = mean_probs.max()
    max_std = std_probs.max()
    
    is_uncertain = (max_std > UNCERTAINTY_STD) or (max_confidence < MIN_CONFIDENCE)
    
    return mean_probs, std_probs, is_uncertain


def classify_sequence(sequence_text):
    """
    Classify an HIV-1 sequence and return subtype predictions.
    
    Uses Monte Carlo Dropout to detect high-uncertainty predictions
    and flag them as indeterminate instead of a forced classification.
    
    Parameters
    ----------
    sequence_text : str
        Raw nucleotide sequence (FASTA format or plain text).
        No alignment required — gaps are automatically stripped.
    
    Returns
    -------
    dict
        Class label -> confidence score
    """
    if not sequence_text or len(sequence_text.strip()) < 10:
        return {"Error": "Please paste a valid HIV-1 nucleotide sequence."}

    encoded, error = encode_sequence(sequence_text)
    if error:
        return {"Error": error}

    # MC Dropout prediction with uncertainty estimation
    mean_probs, std_probs, is_uncertain = mc_dropout_predict(encoded)
    
    if is_uncertain:
        predicted_idx = mean_probs.argmax()
        predicted_label = IDX_TO_LABEL[predicted_idx]
        max_std = std_probs.max()
        return {
            f"⚠️ Indeterminate (leaning {predicted_label})": float(mean_probs.max()),
            "High uncertainty detected": float(max_std),
            "Recommendation": 0.0,
        }
    
    # Build result dict from mean predictions
    result = {}
    for idx in range(NUM_CLASSES):
        label = IDX_TO_LABEL[idx]
        conf = float(mean_probs[idx])
        result[label] = conf

    return result

# ============================================================
# Gradio Interface
# ============================================================

EXAMPLE_SEQ = """>Example_HIV1_sequence
ATGGGTGCGAGAGCGTCAGTATTAAGCGGGGGAGAATTAGATCGATGGGAAAAAATTCGG
TTAAGGCCAGGGGGAAAGAAAAAATATAAATTAAAACATATAGTATGGGCAAGCAGGGAGC
TAGAACGATTCGCAGTTAATCCTGGCCTGTTAGAAACATCAGAAGGCTGTAGACAAATACT
GGGACAGCTACAACCATCCCTTCAGACAGGATCAGAAGAACTTAGATCATTATATAATACA
"""

description = """
## 🧬 HIV-1 Subtype Classifier

Paste an HIV-1 nucleotide sequence (pol gene region) to predict
the subtype. Supports FASTA format or raw sequence.

**No alignment required** — works directly on raw nucleotide sequences.

**Subtypes:** A, B, C, D

**Model:** 1D-CNN with Monte Carlo Dropout uncertainty estimation,
trained on LANL HIV Database pol gene sequences using Focal Loss.

> ⚠️ This is a research/educational tool. For clinical subtype determination,
> use validated tools like REGA, COMET, or jpHMM. Predictions with high
> uncertainty are automatically flagged as indeterminate.
"""

iface = gr.Interface(
    fn=classify_sequence,
    inputs=gr.Textbox(
        label="HIV-1 Nucleotide Sequence",
        placeholder="Paste your sequence here (FASTA or plain text)...",
        lines=10,
        value=EXAMPLE_SEQ
    ),
    outputs=gr.Label(
        num_top_classes=NUM_CLASSES,
        label="Predicted Subtype"
    ),
    title="HIV-1 Subtype Classifier",
    description=description,
    examples=[[EXAMPLE_SEQ]],
    theme=gr.themes.Soft(),
    flagging_mode="never",
)

if __name__ == "__main__":
    iface.launch(share=True, debug=True)
