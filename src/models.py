"""
models.py — Deep Learning Model Architectures for HIV-1 Subtype Classification
================================================================================
Implements three models of increasing complexity:
1. MLP baseline — simple fully-connected network on flattened one-hot input
2. 1D-CNN — local motif detection via convolutional filters (primary model)
3. BiLSTM — long-range dependency modeling (secondary experiment)

All models designed for Google Colab T4 GPU (16GB VRAM).
Input: One-hot encoded pol gene sequences (variable-length, gap-stripped).
       4 channels (A, C, G, T) — no alignment gaps.
Output: 4-class classification (subtypes A, B, C, D).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================================
# Baseline: Multi-Layer Perceptron (MLP)
# ============================================================================

class HIV_MLP(nn.Module):
    """
    Simple MLP baseline for HIV subtype classification.
    
    Operates on flattened one-hot encoded sequences. This serves as a
    non-convolutional baseline to demonstrate the value of learned spatial
    features in the CNN.
    
    Input: One-hot encoded sequences of shape (batch, 4, seq_len)
    Output: Class logits of shape (batch, num_classes)
    
    Parameters
    ----------
    num_classes : int
        Number of output classes (4: A, B, C, D)
    hidden_dims : list of int
        Hidden layer dimensions
    dropout : float
        Dropout rate
    """
    
    def __init__(self, num_classes=4, hidden_dims=[512, 256], dropout=0.4):
        super(HIV_MLP, self).__init__()
        
        # Global average pooling to reduce dimensionality before FC layers
        # Since input lengths vary, we pool to 64 fixed positions
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Alternative: use windowed statistics
        self.feature_extractor = nn.Sequential(
            nn.AdaptiveAvgPool1d(64),  # Reduce to 64 positions
            nn.Flatten(),  # 4 * 64 = 256 features
        )
        
        layers = []
        in_dim = 4 * 64  # After pooling
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, num_classes))
        
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 4, seq_len)
        
        Returns
        -------
        torch.Tensor
            Logits of shape (batch, num_classes)
        """
        features = self.feature_extractor(x)  # (batch, 256)
        logits = self.classifier(features)
        return logits


# ============================================================================
# Primary Model: 1D Convolutional Neural Network
# ============================================================================

class HIV_CNN(nn.Module):
    """
    1D-CNN for HIV pol gene subtype classification.
    
    Learns local nucleotide patterns (motifs) at different scales via
    multiple conv layers with increasing receptive fields. This is the
    primary deep learning model — it automatically learns biological
    motifs directly from raw genomic data without manual feature
    engineering (e.g., k-mer frequencies).
    
    Input: One-hot encoded sequences of shape (batch, 4, seq_len)
    Output: Class logits of shape (batch, num_classes)
    
    Parameters
    ----------
    num_classes : int
        Number of output classes (4: A, B, C, D)
    dropout : float
        Dropout rate
    """
    
    def __init__(self, num_classes=4, dropout=0.3):
        super(HIV_CNN, self).__init__()
        
        self.conv_blocks = nn.Sequential(
            # Block 1: detect short motifs (7-mers)
            nn.Conv1d(4, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Dropout(dropout),
            
            # Block 2: detect medium motifs
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Dropout(dropout),
            
            # Block 3: detect larger patterns
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Dropout(dropout),
            
            # Block 4: high-level features
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # Global average pooling
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 4, seq_len)
        
        Returns
        -------
        torch.Tensor
            Logits of shape (batch, num_classes)
        """
        features = self.conv_blocks(x)
        logits = self.classifier(features)
        return logits


# ============================================================================
# Secondary Model: Bidirectional LSTM
# ============================================================================

class HIV_BiLSTM(nn.Module):
    """
    Bidirectional LSTM for HIV pol gene subtype classification.
    
    Uses an embedding layer to learn nucleotide representations,
    followed by a BiLSTM to capture long-range dependencies in both
    directions along the genome. Included as a secondary experiment
    to compare sequential vs. convolutional feature learning.
    
    Input: Label-encoded sequences of shape (batch, seq_len)
    Output: Class logits of shape (batch, num_classes)
    
    Parameters
    ----------
    num_classes : int
        Number of output classes (4: A, B, C, D)
    vocab_size : int
        Size of nucleotide vocabulary (A,C,G,T + padding = 5)
    embed_dim : int
        Embedding dimension
    hidden_dim : int
        LSTM hidden dimension
    num_layers : int
        Number of LSTM layers
    dropout : float
        Dropout rate
    """
    
    def __init__(self, num_classes=4, vocab_size=5,
                 embed_dim=128, hidden_dim=256, num_layers=2, dropout=0.3):
        super(HIV_BiLSTM, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=4)
        
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Label-encoded input of shape (batch, seq_len)
        
        Returns
        -------
        torch.Tensor
            Logits of shape (batch, num_classes)
        """
        embedded = self.embedding(x)           # (batch, seq_len, embed_dim)
        embedded = self.dropout(embedded)
        
        lstm_out, (hidden, _) = self.lstm(embedded)
        
        # Concatenate final hidden states from both directions
        forward_hidden = hidden[-2]   # Last layer, forward
        backward_hidden = hidden[-1]  # Last layer, backward
        combined = torch.cat([forward_hidden, backward_hidden], dim=1)
        
        combined = self.dropout(combined)
        logits = self.classifier(combined)
        return logits


# ============================================================================
# Model Factory
# ============================================================================

def get_model(model_name, num_classes=4, dropout=0.3, device='cpu'):
    """
    Factory function to create models by name.
    
    Parameters
    ----------
    model_name : str
        One of 'mlp', 'cnn', 'bilstm'
    num_classes : int
        Number of output classes (4: A, B, C, D)
    dropout : float
        Dropout rate
    device : str
        Device to move model to
    
    Returns
    -------
    nn.Module
        Instantiated model on the specified device
    """
    models = {
        'mlp': HIV_MLP,
        'cnn': HIV_CNN,
        'bilstm': HIV_BiLSTM,
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(models.keys())}")
    
    model = models[model_name](num_classes=num_classes, dropout=dropout)
    model = model.to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'='*50}")
    print(f"Model: {model_name}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"{'='*50}\n")
    
    return model
