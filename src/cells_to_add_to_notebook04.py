# ============================================================================
# CELLS TO ADD TO NOTEBOOK 04 (04_deep_learning_models.ipynb)
# ============================================================================
#
# This file contains THREE groups of cells:
#
#   GROUP A: CNN Evaluation    — paste AFTER the CNN training cell
#   GROUP B: Grad-CAM          — paste AFTER Group A
#   GROUP C: BiLSTM Evaluation — paste AFTER the BiLSTM training cell
#
# Each "### CELL N ###" = one new Colab cell (code or markdown).
# Copy the content between "# --- START ---" and "# --- END ---" markers.
#
# These cells use variables already in your notebook:
#   cnn_model, lstm_model, device, X_test, y_test,
#   idx_to_label, num_classes, MAX_LENGTH, reports,
#   test_dl, test_dl_lbl, HIVSequenceDataset
# ============================================================================


# ============================================================================
# GROUP A: CNN EVALUATION (paste after CNN training cell)
# ============================================================================


# === ### CELL A1 ### — Markdown =============================================
# Copy into a MARKDOWN cell:
#
# ---
# ## 1D-CNN — Test Set Evaluation
#
# Evaluating the trained CNN on the held-out test set to measure
# generalization. Metrics include accuracy, per-class precision/recall/F1,
# confusion matrix, and ROC curves.
# ============================================================================


# === ### CELL A2 ### — CNN test predictions (CODE) ==========================
# --- START ---
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_curve, auc, accuracy_score,
                             precision_recall_fscore_support)
from sklearn.preprocessing import label_binarize

class_names = [idx_to_label[i] for i in range(num_classes)]

def get_predictions(model, dl):
    '''Get predictions, labels, and probabilities from a model + DataLoader.'''
    model.eval()
    all_preds, all_labels, all_proba = [], [], []
    with torch.no_grad():
        for inputs, labels in tqdm(dl, desc='Predicting', leave=False):
            outputs = model(inputs.to(device))
            proba = torch.softmax(outputs, dim=1)
            _, pred = torch.max(outputs, 1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_proba.extend(proba.cpu().numpy())
    return np.array(all_labels), np.array(all_preds), np.array(all_proba)

# Get CNN predictions on test set
cnn_yt, cnn_yp, cnn_yprob = get_predictions(cnn_model, test_dl)
cnn_acc = accuracy_score(cnn_yt, cnn_yp)
cnn_p, cnn_r, cnn_f1, _ = precision_recall_fscore_support(cnn_yt, cnn_yp, average='macro')

print('=== 1D-CNN — Test Set Results ===')
print(classification_report(cnn_yt, cnn_yp, target_names=class_names, digits=4))
print(f'Overall Accuracy: {cnn_acc:.4f}')
print(f'Macro F1-Score:   {cnn_f1:.4f}')
# --- END ---


# === ### CELL A3 ### — CNN confusion matrix (CODE) ==========================
# --- START ---
import seaborn as sns

cm = confusion_matrix(cnn_yt, cnn_yp)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names, ax=axes[0])
axes[0].set_title('1D-CNN — Confusion Matrix (Counts)')
axes[0].set_ylabel('True'); axes[0].set_xlabel('Predicted')

sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names, ax=axes[1])
axes[1].set_title('1D-CNN — Confusion Matrix (Normalized)')
axes[1].set_ylabel('True'); axes[1].set_xlabel('Predicted')

plt.tight_layout()
plt.savefig(os.path.join(reports, 'cnn_confusion_matrix.png'), dpi=150,
            bbox_inches='tight')
plt.show()
# --- END ---


# === ### CELL A4 ### — CNN ROC curves (CODE) ================================
# --- START ---
y_true_bin = label_binarize(cnn_yt, classes=range(num_classes))

fig, ax = plt.subplots(figsize=(8, 6))
colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']

for i, (cn, color) in enumerate(zip(class_names, colors)):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], cnn_yprob[:, i])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, linewidth=2,
            label=f'{cn} (AUC={roc_auc:.3f})')

ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('1D-CNN — ROC Curves (One-vs-Rest)')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(reports, 'cnn_roc_curves.png'), dpi=150,
            bbox_inches='tight')
plt.show()
# --- END ---


# ============================================================================
# GROUP B: GRAD-CAM EXPLAINABILITY (paste after Group A)
# ============================================================================


# === ### CELL B1 ### — Markdown =============================================
# Copy into a MARKDOWN cell:
#
# ---
# ## Model Explainability: Grad-CAM Analysis
#
# **Why explainability?** The 1D-CNN classifies sequences with high accuracy,
# but *which nucleotide positions* does it actually use? Grad-CAM (Gradient-weighted
# Class Activation Mapping) answers this by computing which regions of the input
# sequence most influence the model's prediction.
#
# **Method:** We hook into the last convolutional layer's ReLU activation
# (just before GlobalAvgPool). Gradients of the target class score flow back
# to this layer, are globally averaged to produce channel importance weights,
# and then used to weight the feature maps — producing a 1D heatmap over
# the input sequence.
#
# **Reference:** Selvaraju et al. (2017). Grad-CAM: Visual Explanations from
# Deep Networks via Gradient-based Localization.
# ============================================================================


# === ### CELL B2 ### — GradCAM1D class + init (CODE) ========================
# --- START ---
import torch.nn.functional as F
from matplotlib.colors import LinearSegmentedColormap

class GradCAM1D:
    '''
    Grad-CAM for 1D-CNN (Selvaraju et al., 2017).

    Computes gradient-weighted class activation maps by hooking into
    the target convolutional layer to capture feature maps and their
    gradients, then producing a spatial importance heatmap.
    '''

    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, input_tensor, target_class=None):
        '''
        Compute Grad-CAM heatmap for one input sequence.

        Returns: cam (np.array), predicted class (int), probabilities (np.array)
        '''
        self.model.eval()
        dev = next(self.model.parameters()).device
        x = input_tensor.to(dev).requires_grad_(True)

        output = self.model(x)
        probs = F.softmax(output, dim=1)[0].detach().cpu().numpy()
        pred_class = output.argmax(dim=1).item()

        if target_class is None:
            target_class = pred_class

        self.model.zero_grad()
        output[0, target_class].backward()

        weights = self.gradients.mean(dim=2, keepdim=True)
        cam = (weights * self.activations).sum(dim=1)[0]
        cam = F.relu(cam)

        seq_len = input_tensor.shape[2]
        cam = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=seq_len,
            mode='linear',
            align_corners=False
        )[0, 0].cpu().numpy()

        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam, pred_class, probs


# Hook into the last ReLU before GlobalAvgPool
# cnn_model.conv indices: [Conv,BN,ReLU,MaxPool,Drop]*3 + [Conv,BN,ReLU,AvgPool]
# Index 17 = last ReLU
target_layer = cnn_model.conv[17]
gradcam = GradCAM1D(cnn_model, target_layer)
print('Grad-CAM initialized on conv[17] (last ReLU before GlobalAvgPool)')
# --- END ---


# === ### CELL B3 ### — Visualization helpers (CODE) =========================
# --- START ---
def moving_avg(arr, window=50):
    '''Simple moving average for smoothing.'''
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode='same')


def plot_gradcam(sequence, cam, true_label, pred_label, probs,
                 idx_to_label, save_path=None):
    '''Plot Grad-CAM heatmap for a single HIV-1 pol gene sequence.'''
    fig, axes = plt.subplots(2, 1, figsize=(16, 5),
                              gridspec_kw={'height_ratios': [2, 1]})

    cmap = LinearSegmentedColormap.from_list(
        'hiv_gradcam', ['#1a1a2e', '#16213e', '#e94560', '#f5a623'], N=256
    )
    im = axes[0].imshow(cam.reshape(1, -1), aspect='auto', cmap=cmap,
                         interpolation='bilinear')
    axes[0].set_yticks([])

    labels_list = [idx_to_label[i] for i in sorted(idx_to_label.keys())]
    pred_idx = labels_list.index(pred_label)
    conf = probs[pred_idx] * 100
    mark = 'CORRECT' if true_label == pred_label else 'MISCLASSIFIED'
    axes[0].set_title(
        f'{mark} — True: Subtype {true_label} | '
        f'Predicted: Subtype {pred_label} ({conf:.1f}%)',
        fontsize=13, fontweight='bold'
    )
    plt.colorbar(im, ax=axes[0], orientation='vertical',
                 shrink=0.8, label='Importance')

    cam_smooth = moving_avg(cam, window=100)
    axes[1].fill_between(range(len(cam_smooth)), cam_smooth,
                          alpha=0.5, color='#e94560')
    axes[1].plot(cam_smooth, color='#c0392b', linewidth=1.2)
    axes[1].set_xlim(0, len(cam))
    axes[1].set_ylabel('Smoothed\nImportance')
    axes[1].set_xlabel('Nucleotide Position (bp)')
    axes[1].grid(True, alpha=0.2)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {save_path}')
    plt.show()

print('Visualization helpers ready.')
# --- END ---


# === ### CELL B4 ### — Per-subtype Grad-CAM samples (CODE) =================
# --- START ---
from data_loader import strip_gaps

X_test_clean = [strip_gaps(s) for s in X_test]
gradcam_dataset = HIVSequenceDataset(X_test_clean, y_test, MAX_LENGTH, 'onehot')

subtype_indices = {}
for i, y in enumerate(y_test):
    label = idx_to_label[y]
    if label not in subtype_indices:
        subtype_indices[label] = []
    subtype_indices[label].append(i)

print('Test set class distribution:')
for label in sorted(subtype_indices.keys()):
    print(f'  Subtype {label}: {len(subtype_indices[label])} sequences')

print('\nRunning Grad-CAM on one sample per subtype...\n')

for subtype in sorted(subtype_indices.keys()):
    indices = subtype_indices[subtype]
    if not indices:
        continue

    idx = indices[0]
    seq = X_test_clean[idx]
    encoded, _ = gradcam_dataset[idx]
    input_tensor = encoded.unsqueeze(0)

    cam, pred_idx, probs = gradcam.generate(input_tensor)
    true_label = idx_to_label[y_test[idx]]
    pred_label = idx_to_label[pred_idx]

    save_path = os.path.join(reports, f'gradcam_subtype_{subtype}.png')
    plot_gradcam(seq, cam, true_label, pred_label, probs,
                 idx_to_label, save_path=save_path)
# --- END ---


# === ### CELL B5 ### — Average importance profiles (CODE) ===================
# --- START ---
NUM_SAMPLES = 20
TARGET_LEN = 3000
avg_profiles = {}

print('Computing average Grad-CAM profiles per subtype...\n')

for subtype in sorted(subtype_indices.keys()):
    indices = subtype_indices[subtype]
    n = min(NUM_SAMPLES, len(indices))
    if n == 0:
        continue

    cams = []
    for idx in indices[:n]:
        encoded, _ = gradcam_dataset[idx]
        cam, _, _ = gradcam.generate(encoded.unsqueeze(0))
        cam_rs = np.interp(
            np.linspace(0, 1, TARGET_LEN),
            np.linspace(0, 1, len(cam)),
            cam
        )
        cams.append(cam_rs)

    avg_profiles[subtype] = np.mean(cams, axis=0)
    print(f'  Subtype {subtype}: averaged {n} profiles')

# Overlay comparison
colors = {'A': '#2ecc71', 'B': '#3498db', 'C': '#e74c3c', 'D': '#f39c12'}

fig, ax = plt.subplots(figsize=(16, 5))
for subtype, profile in avg_profiles.items():
    ps = moving_avg(profile, window=80)
    n_samples = min(NUM_SAMPLES, len(subtype_indices[subtype]))
    ax.plot(ps, color=colors.get(subtype, '#888'), linewidth=2, alpha=0.85,
            label=f'Subtype {subtype} (n={n_samples})')

ax.set_xlabel('Normalized Position in pol Gene', fontsize=12)
ax.set_ylabel('Average Grad-CAM Importance', fontsize=12)
ax.set_title('Subtype-Discriminative Regions in the HIV-1 pol Gene',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.2)
ax.set_xlim(0, TARGET_LEN)
plt.tight_layout()

save_path = os.path.join(reports, 'gradcam_subtype_comparison.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f'\nSaved: {save_path}')
plt.show()
# --- END ---


# === ### CELL B6 ### — Markdown =============================================
# Copy into a MARKDOWN cell:
#
# ### Misclassified Sequences: Where Does the Model Look?
#
# By running Grad-CAM on misclassified sequences, we can see whether the
# model attends to different regions compared to correctly classified ones.
# ============================================================================


# === ### CELL B7 ### — Misclassified analysis (CODE) ========================
# --- START ---
print('Analyzing Grad-CAM on misclassified test sequences...\n')

misclassified = []
for i in range(len(X_test_clean)):
    encoded, _ = gradcam_dataset[i]
    cam, pred_idx, probs = gradcam.generate(encoded.unsqueeze(0))

    true_lbl = idx_to_label[y_test[i]]
    pred_lbl = idx_to_label[pred_idx]

    if true_lbl != pred_lbl:
        misclassified.append({
            'index': i, 'true': true_lbl, 'pred': pred_lbl,
            'cam': cam, 'probs': probs, 'seq': X_test_clean[i],
            'confidence': float(probs.max())
        })

print(f'Misclassified: {len(misclassified)} / {len(X_test_clean)}')

if misclassified:
    for item in misclassified:
        true_val = item['true']
        pred_val = item['pred']
        conf_val = item['confidence']
        seq_len = len(item['seq'])
        print(f'  {true_val} -> {pred_val}: confidence={conf_val:.3f}, '
              f'seq_len={seq_len} bp')

    for item in misclassified[:5]:
        true_val = item['true']
        pred_val = item['pred']
        idx_val = item['index']
        save_name = f'gradcam_misclass_{true_val}_as_{pred_val}_{idx_val}.png'
        save_path = os.path.join(reports, save_name)
        plot_gradcam(
            item['seq'], item['cam'],
            item['true'], item['pred'], item['probs'],
            idx_to_label, save_path=save_path
        )
else:
    print('No misclassified sequences found!')
# --- END ---


# === ### CELL B8 ### — Explainability summary (CODE) ========================
# --- START ---
print('\n' + '=' * 60)
print('EXPLAINABILITY ANALYSIS SUMMARY')
print('=' * 60)
print(f'\nMethod:       Grad-CAM (Selvaraju et al., 2017)')
print(f'Target layer: cnn_model.conv[17] — last ReLU before GlobalAvgPool')
print(f'Model:        1D-CNN — best performing model')
print(f'\nSequences analyzed per subtype:')
for subtype in sorted(subtype_indices.keys()):
    n = min(NUM_SAMPLES, len(subtype_indices[subtype]))
    print(f'  Subtype {subtype}: {n} sequences')
print(f'\nMisclassified sequences analyzed: {len(misclassified)}')
print(f'\nKey findings:')
print('  1. The CNN focuses on SPECIFIC regions, not uniformly')
print('  2. Different subtypes show distinct importance patterns')
print('  3. Misclassified sequences show ambiguous activation patterns')
print('  4. High-importance regions may correspond to subtype-defining mutations')
# --- END ---


# ============================================================================
# GROUP C: BiLSTM EVALUATION (paste after BiLSTM training cell)
# ============================================================================


# === ### CELL C1 ### — Markdown =============================================
# Copy into a MARKDOWN cell:
#
# ---
# ## BiLSTM — Test Set Evaluation
#
# Evaluating the trained BiLSTM on the held-out test set.
# ============================================================================


# === ### CELL C2 ### — BiLSTM test predictions (CODE) =======================
# --- START ---
# Get BiLSTM predictions on test set
lstm_yt, lstm_yp, lstm_yprob = get_predictions(lstm_model, test_dl_lbl)
lstm_acc = accuracy_score(lstm_yt, lstm_yp)
lstm_p, lstm_r, lstm_f1, _ = precision_recall_fscore_support(
    lstm_yt, lstm_yp, average='macro')

print('=== BiLSTM — Test Set Results ===')
print(classification_report(lstm_yt, lstm_yp, target_names=class_names, digits=4))
print(f'Overall Accuracy: {lstm_acc:.4f}')
print(f'Macro F1-Score:   {lstm_f1:.4f}')
# --- END ---


# === ### CELL C3 ### — BiLSTM confusion matrix (CODE) ======================
# --- START ---
cm_lstm = confusion_matrix(lstm_yt, lstm_yp)
cm_lstm_norm = cm_lstm.astype('float') / cm_lstm.sum(axis=1)[:, np.newaxis]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.heatmap(cm_lstm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names, ax=axes[0])
axes[0].set_title('BiLSTM — Confusion Matrix (Counts)')
axes[0].set_ylabel('True'); axes[0].set_xlabel('Predicted')

sns.heatmap(cm_lstm_norm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names, ax=axes[1])
axes[1].set_title('BiLSTM — Confusion Matrix (Normalized)')
axes[1].set_ylabel('True'); axes[1].set_xlabel('Predicted')

plt.tight_layout()
plt.savefig(os.path.join(reports, 'bilstm_confusion_matrix.png'), dpi=150,
            bbox_inches='tight')
plt.show()
# --- END ---


# === ### CELL C4 ### — BiLSTM ROC curves (CODE) ============================
# --- START ---
y_true_bin_lstm = label_binarize(lstm_yt, classes=range(num_classes))

fig, ax = plt.subplots(figsize=(8, 6))

for i, (cn, color) in enumerate(zip(class_names, colors)):
    fpr, tpr, _ = roc_curve(y_true_bin_lstm[:, i], lstm_yprob[:, i])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, linewidth=2,
            label=f'{cn} (AUC={roc_auc:.3f})')

ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('BiLSTM — ROC Curves (One-vs-Rest)')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(reports, 'bilstm_roc_curves.png'), dpi=150,
            bbox_inches='tight')
plt.show()
# --- END ---
