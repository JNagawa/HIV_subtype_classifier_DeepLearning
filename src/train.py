"""
train.py — Training Loop with Early Stopping and Experiment Logging
====================================================================
Handles model training with:
- Focal Loss for class imbalance (default) or CrossEntropyLoss
- Learning rate scheduling (CosineAnnealingLR)
- Early stopping on validation loss
- Per-epoch logging of metrics
- Model checkpoint saving
"""

import os
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm


class EarlyStopping:
    """
    Early stopping to prevent overfitting.

    Monitors validation loss and stops training if no improvement
    for a specified number of epochs.

    Parameters
    ----------
    patience : int
        Number of epochs to wait for improvement
    min_delta : float
        Minimum change to qualify as improvement
    verbose : bool
        Print messages on each check
    """

    def __init__(self, patience=10, min_delta=0.001, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
        self.best_model_state = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_state = copy.deepcopy(model.state_dict())
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model_state = copy.deepcopy(model.state_dict())
            if self.verbose:
                print(f"  ↓ Val loss improved to {val_loss:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"  ⏳ No improvement for {self.counter}/{self.patience} epochs")
            if self.counter >= self.patience:
                self.should_stop = True
                if self.verbose:
                    print(f"  🛑 Early stopping triggered!")


# ============================================================================
# Focal Loss for Class Imbalance
# ============================================================================


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance (Lin et al., 2017).

    Down-weights the loss for well-classified (easy) examples and focuses
    training on hard, misclassified examples. This is especially useful for
    the HIV subtype D minority class, which tends to be misclassified as C.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Parameters
    ----------
    alpha : torch.Tensor or None
        Per-class weighting factors. If None, all classes weighted equally.
        Typically set to inverse class frequencies.
    gamma : float
        Focusing parameter. gamma=0 is standard CE. gamma=2 is common.
        Higher gamma increases focus on hard examples.
    reduction : str
        'mean', 'sum', or 'none'
    """

    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            if isinstance(alpha, (list, np.ndarray)):
                self.alpha = torch.tensor(alpha, dtype=torch.float32)
            else:
                self.alpha = alpha
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        """
        Compute focal loss.

        Parameters
        ----------
        inputs : torch.Tensor
            Raw logits of shape (batch, num_classes)
        targets : torch.Tensor
            Ground truth labels of shape (batch,)

        Returns
        -------
        torch.Tensor
            Scalar loss value
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)  # p_t = probability of correct class

        focal_weight = (1 - pt) ** self.gamma

        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            alpha_t = alpha[targets]
            focal_weight = alpha_t * focal_weight

        loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


# ============================================================================
# Training Functions
# ============================================================================


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train model for one epoch.

    Returns
    -------
    float
        Average training loss
    float
        Training accuracy
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(train_loader, desc="  Training", leave=False):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        # Gradient clipping to prevent exploding gradients (important for LSTMs)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """
    Validate model on validation set.

    Returns
    -------
    float
        Average validation loss
    float
        Validation accuracy
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="  Validating", leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def train_model(
    model,
    train_loader,
    val_loader,
    num_classes,
    class_weights=None,
    num_epochs=50,
    learning_rate=1e-3,
    patience=10,
    save_dir="models",
    model_name="model",
    device="cpu",
    loss_type="focal",
    focal_gamma=2.0,
):
    """
    Full training pipeline with early stopping and LR scheduling.

    Parameters
    ----------
    model : nn.Module
        The model to train
    train_loader : DataLoader
        Training data loader
    val_loader : DataLoader
        Validation data loader
    num_classes : int
        Number of output classes
    class_weights : torch.Tensor, optional
        Class weights for handling imbalance (used as alpha in Focal Loss)
    num_epochs : int
        Maximum number of training epochs
    learning_rate : float
        Initial learning rate
    patience : int
        Early stopping patience
    save_dir : str
        Directory to save model checkpoints
    model_name : str
        Name prefix for saved files
    device : str
        Device ('cuda' or 'cpu')
    loss_type : str
        'focal' for Focal Loss (default) or 'ce' for CrossEntropyLoss
    focal_gamma : float
        Focusing parameter for Focal Loss (default: 2.0)

    Returns
    -------
    dict
        Training history with per-epoch metrics
    """
    os.makedirs(save_dir, exist_ok=True)

    # Loss function selection
    if loss_type == "focal":
        alpha = class_weights.to(device) if class_weights is not None else None
        criterion = FocalLoss(alpha=alpha, gamma=focal_gamma)
        print(
            f"Loss: Focal Loss (gamma={focal_gamma}, alpha={'class_weights' if alpha is not None else 'None'})"
        )
    else:
        if class_weights is not None:
            criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        else:
            criterion = nn.CrossEntropyLoss()
        print(
            f"Loss: CrossEntropyLoss (weights={'yes' if class_weights is not None else 'no'})"
        )

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )

    # Early stopping
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    # Training history
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }

    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"Device: {device}")
    print(f"Epochs: {num_epochs} | LR: {learning_rate} | Patience: {patience}")
    print(f"{'='*60}\n")

    start_time = time.time()

    for epoch in range(num_epochs):
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch [{epoch+1}/{num_epochs}] (LR: {current_lr:.6f})")

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Step scheduler
        scheduler.step()

        # Log
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

        # Check early stopping
        early_stopping(val_loss, model)
        if early_stopping.should_stop:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    # Restore best model
    if early_stopping.best_model_state is not None:
        model.load_state_dict(early_stopping.best_model_state)
        print(f"Restored best model (val_loss: {early_stopping.best_loss:.4f})")

    # Save best model
    save_path = os.path.join(save_dir, f"{model_name}_best.pth")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "history": history,
            "num_classes": num_classes,
            "best_val_loss": early_stopping.best_loss,
        },
        save_path,
    )
    print(f"Model saved to {save_path}")

    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed/60:.1f} minutes")

    return history
