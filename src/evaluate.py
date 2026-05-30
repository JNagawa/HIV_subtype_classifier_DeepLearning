"""
evaluate.py — Model Evaluation, Error Analysis, and Visualization
==================================================================
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    f1_score,
    accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
import os


def get_predictions(model, data_loader, device="cpu"):
    """Get all predictions and true labels from a DataLoader."""
    model.eval()
    all_preds, all_labels, all_proba = [], [], []
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Predicting", leave=False):
            inputs = inputs.to(device)
            outputs = model(inputs)
            proba = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_proba.extend(proba.cpu().numpy())
    return np.array(all_labels), np.array(all_preds), np.array(all_proba)


def evaluate_model(
    model,
    test_loader,
    idx_to_label,
    device="cpu",
    save_dir="reports/figures",
    model_name="model",
):
    """Full evaluation pipeline with visualizations."""
    os.makedirs(save_dir, exist_ok=True)
    y_true, y_pred, y_proba = get_predictions(model, test_loader, device)
    num_classes = len(idx_to_label)
    class_names = [idx_to_label[i] for i in range(num_classes)]

    print(f"\n{'='*60}\nEvaluation Results: {model_name}\n{'='*60}")
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report)

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro"
    )
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Macro F1-Score:   {f1:.4f}")

    plot_confusion_matrix(y_true, y_pred, class_names, save_dir, model_name)
    plot_roc_curves(y_true, y_proba, class_names, save_dir, model_name)
    plot_per_class_metrics(y_true, y_pred, class_names, save_dir, model_name)

    return {
        "accuracy": accuracy,
        "macro_f1": f1,
        "macro_precision": precision,
        "macro_recall": recall,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "report": report,
    }


def plot_confusion_matrix(y_true, y_pred, class_names, save_dir, model_name):
    """Plot and save confusion matrix heatmaps (raw + normalized)."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[0],
    )
    axes[0].set_title(f"{model_name} — Confusion Matrix (Counts)")
    axes[0].set_ylabel("True Label")
    axes[0].set_xlabel("Predicted Label")
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[1],
    )
    axes[1].set_title(f"{model_name} — Confusion Matrix (Normalized)")
    axes[1].set_ylabel("True Label")
    axes[1].set_xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(
        f"{save_dir}/{model_name}_confusion_matrix.png", dpi=150, bbox_inches="tight"
    )
    plt.show()


def plot_roc_curves(y_true, y_proba, class_names, save_dir, model_name):
    """Plot ROC curves (one-vs-rest) for each class."""
    num_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=range(num_classes))
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.Set2(np.linspace(0, 1, num_classes))
    for i, (cn, color) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{cn} (AUC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{model_name} — ROC Curves")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{model_name}_roc_curves.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_per_class_metrics(y_true, y_pred, class_names, save_dir, model_name):
    """Plot per-class precision, recall, F1 as grouped bar chart."""
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(class_names))
    )
    x = np.arange(len(class_names))
    w = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - w, prec, w, label="Precision", color="#2196F3")
    ax.bar(x, rec, w, label="Recall", color="#4CAF50")
    ax.bar(x + w, f1, w, label="F1-Score", color="#FF9800")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_title(f"{model_name} — Per-Class Metrics")
    for i, s in enumerate(sup):
        ax.text(i, 1.02, f"n={s}", ha="center", fontsize=9, color="gray")
    plt.tight_layout()
    plt.savefig(
        f"{save_dir}/{model_name}_per_class_metrics.png", dpi=150, bbox_inches="tight"
    )
    plt.show()


def plot_training_history(history, save_dir="reports/figures", model_name="model"):
    """Plot training/validation loss, accuracy, and LR curves."""
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(epochs, history["train_loss"], "b-", label="Train", lw=2)
    axes[0].plot(epochs, history["val_loss"], "r-", label="Val", lw=2)
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(epochs, history["train_acc"], "b-", label="Train", lw=2)
    axes[1].plot(epochs, history["val_acc"], "r-", label="Val", lw=2)
    axes[1].set_title("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(epochs, history["lr"], "g-", lw=2)
    axes[2].set_title("Learning Rate")
    axes[2].grid(True, alpha=0.3)
    plt.suptitle(f"{model_name} — Training History", fontsize=16)
    plt.tight_layout()
    plt.savefig(
        f"{save_dir}/{model_name}_training_history.png", dpi=150, bbox_inches="tight"
    )
    plt.show()


def error_analysis(y_true, y_pred, y_proba, idx_to_label):
    """Analyze misclassified sequences."""
    class_names = [idx_to_label[i] for i in range(len(idx_to_label))]
    misclassified = y_true != y_pred
    n_err = misclassified.sum()
    print(f"\nMisclassified: {n_err}/{len(y_true)} ({100*n_err/len(y_true):.1f}%)")
    cm = confusion_matrix(y_true, y_pred)
    np.fill_diagonal(cm, 0)
    pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                pairs.append((class_names[i], class_names[j], cm[i, j]))
    pairs.sort(key=lambda x: x[2], reverse=True)
    print("Most confused pairs:")
    for t, p, c in pairs[:10]:
        print(f"  {t} → {p}: {c}")
    return {"n_errors": n_err, "confused_pairs": pairs}
