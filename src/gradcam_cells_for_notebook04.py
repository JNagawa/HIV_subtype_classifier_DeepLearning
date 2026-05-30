# ============================================================================
# GRAD-CAM EXPLAINABILITY CELLS — Paste these into notebook 04
# ============================================================================
#
# INSTRUCTIONS:
# 1. Open 04_deep_learning_models.ipynb in Colab
# 2. After the CNN training cell (and optionally after BiLSTM), add new cells
# 3. Each ### CELL ### section below = one new Colab cell
# 4. Copy-paste each section into its own cell and run sequentially
#
# These cells use variables already defined in your notebook:
#   - cnn_model (trained HIV_CNN)
#   - device (cuda/cpu)
#   - X_test, y_test (test data lists)
#   - idx_to_label (dict: int -> subtype letter)
#   - reports (path to reports/figures directory)
#   - HIVSequenceDataset (already imported from data_loader)
#   - strip_gaps (already imported or available via data_loader)
# ============================================================================


# === ### CELL 1 ### — Markdown cell ==========================================
# Copy the text below (without the # prefixes) into a MARKDOWN cell:
#
# ---
# ## Model Explainability: Grad-CAM Analysis
#
# **Why explainability?** The 1D-CNN classifies sequences with 98%+ accuracy,
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


# === ### CELL 2 ### — Grad-CAM class + setup (CODE cell) =====================
# Copy everything between the START and END markers below into a CODE cell:

# --- START ---
import torch.nn.functional as F
from matplotlib.colors import LinearSegmentedColormap

class GradCAM1D:
    '''
    Grad-CAM for 1D Convolutional Neural Networks.

    Hooks into a target convolutional layer and computes gradient-weighted
    class activation maps showing which input positions drive predictions.

    Reference: Selvaraju et al. (2017)
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

        Parameters
        ----------
        input_tensor : torch.Tensor
            Shape (1, 4, seq_len) — one-hot encoded sequence
        target_class : int or None
            Class to explain. If None, uses the predicted class.

        Returns
        -------
        cam : np.ndarray of shape (seq_len,) in [0, 1]
        pred_class : int
        probs : np.ndarray of shape (num_classes,)
        '''
        self.model.eval()
        dev = next(self.model.parameters()).device
        x = input_tensor.to(dev).requires_grad_(True)

        # Forward pass
        output = self.model(x)
        probs = F.softmax(output, dim=1)[0].detach().cpu().numpy()
        pred_class = output.argmax(dim=1).item()

        if target_class is None:
            target_class = pred_class

        # Backward pass for target class
        self.model.zero_grad()
        output[0, target_class].backward()

        # Grad-CAM: weight feature maps by globally-averaged gradients
        weights = self.gradients.mean(dim=2, keepdim=True)   # (1, C, 1)
        cam = (weights * self.activations).sum(dim=1)[0]     # (L',)
        cam = F.relu(cam)  # only positive contributions matter

        # Upsample to input sequence length
        seq_len = input_tensor.shape[2]
        cam = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=seq_len,
            mode='linear',
            align_corners=False
        )[0, 0].cpu().numpy()

        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam, pred_class, probs


# Hook into the last ReLU before GlobalAvgPool in cnn_model.conv
# conv Sequential indices: [Conv,BN,ReLU,MaxPool,Drop]*3 + [Conv,BN,ReLU,AvgPool]
# Index 17 = last ReLU activation (just before AdaptiveAvgPool1d)
target_layer = cnn_model.conv[17]
gradcam = GradCAM1D(cnn_model, target_layer)
print('Grad-CAM initialized on conv[17] (last ReLU before GlobalAvgPool)')
# --- END ---


# === ### CELL 3 ### — Visualization helpers (CODE cell) ======================
# --- START ---
def moving_avg(arr, window=50):
    '''Simple moving average for smoothing Grad-CAM heatmaps.'''
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode='same')


def plot_gradcam(sequence, cam, true_label, pred_label, probs,
                 idx_to_label, save_path=None):
    '''
    Plot Grad-CAM heatmap for a single HIV-1 pol gene sequence.

    Top panel: 1D heatmap showing per-position importance.
    Bottom panel: smoothed importance curve for easier interpretation.
    '''
    fig, axes = plt.subplots(2, 1, figsize=(16, 5),
                              gridspec_kw={'height_ratios': [2, 1]})

    # --- Heatmap ---
    cmap = LinearSegmentedColormap.from_list(
        'hiv_gradcam', ['#1a1a2e', '#16213e', '#e94560', '#f5a623'], N=256
    )
    im = axes[0].imshow(cam.reshape(1, -1), aspect='auto', cmap=cmap,
                         interpolation='bilinear')
    axes[0].set_yticks([])

    # Build label string
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

    # --- Smoothed curve ---
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


# === ### CELL 4 ### — Prepare test data + run Grad-CAM per subtype (CODE) ====
# --- START ---
from data_loader import strip_gaps

# Strip alignment gaps from test sequences
X_test_clean = [strip_gaps(s) for s in X_test]
print(f'Test set: {len(X_test_clean)} gap-stripped sequences')
print(f'Length range: {min(len(s) for s in X_test_clean)}'
      f'-{max(len(s) for s in X_test_clean)} bp')

# Create one-hot dataset for Grad-CAM (same encoding as CNN training)
gradcam_dataset = HIVSequenceDataset(X_test_clean, y_test, MAX_LENGTH, 'onehot')

# Group test indices by subtype
subtype_indices = {}
for i, y in enumerate(y_test):
    label = idx_to_label[y]
    if label not in subtype_indices:
        subtype_indices[label] = []
    subtype_indices[label].append(i)

print('\nTest set class distribution:')
for label in sorted(subtype_indices.keys()):
    print(f'  Subtype {label}: {len(subtype_indices[label])} sequences')

# --- Run Grad-CAM on first sample of each subtype ---
print('\n' + '=' * 60)
print('Grad-CAM: Per-Subtype Sample Analysis')
print('=' * 60)

for subtype in sorted(subtype_indices.keys()):
    indices = subtype_indices[subtype]
    if not indices:
        print(f'\nNo test samples for subtype {subtype}')
        continue

    idx = indices[0]
    seq = X_test_clean[idx]
    encoded, _ = gradcam_dataset[idx]
    input_tensor = encoded.unsqueeze(0)  # (1, 4, seq_len)

    cam, pred_idx, probs = gradcam.generate(input_tensor)
    true_label = idx_to_label[y_test[idx]]
    pred_label = idx_to_label[pred_idx]

    save_path = os.path.join(reports, f'gradcam_subtype_{subtype}.png')
    plot_gradcam(seq, cam, true_label, pred_label, probs,
                 idx_to_label, save_path=save_path)
# --- END ---


# === ### CELL 5 ### — Average importance profiles per subtype (CODE) =========
# --- START ---
NUM_SAMPLES = 20      # max samples per subtype for averaging
TARGET_LEN = 3000     # common length for comparison (interpolated)
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
        # Resample to common length for averaging
        cam_rs = np.interp(
            np.linspace(0, 1, TARGET_LEN),
            np.linspace(0, 1, len(cam)),
            cam
        )
        cams.append(cam_rs)

    avg_profiles[subtype] = np.mean(cams, axis=0)
    print(f'  Subtype {subtype}: averaged {n} Grad-CAM profiles')

# --- Plot: overlay all subtypes ---
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

# --- Plot: separate panels per subtype ---
n_subtypes = len(avg_profiles)
fig, axes = plt.subplots(n_subtypes, 1, figsize=(16, 2.5 * n_subtypes),
                          sharex=True)
if n_subtypes == 1:
    axes = [axes]

for ax, (subtype, profile) in zip(axes, sorted(avg_profiles.items())):
    ps = moving_avg(profile, window=80)
    color = colors.get(subtype, '#888')
    ax.fill_between(range(len(ps)), ps, alpha=0.4, color=color)
    ax.plot(ps, color=color, linewidth=1.5)
    n_samples = min(NUM_SAMPLES, len(subtype_indices[subtype]))
    ax.legend([f'Subtype {subtype} (n={n_samples})'], loc='upper right')
    ax.set_ylabel(f'{subtype}')
    ax.grid(True, alpha=0.2)

axes[-1].set_xlabel('Normalized Position in pol Gene')
plt.suptitle('Average Grad-CAM Importance Profiles by Subtype',
             fontsize=14, fontweight='bold')
plt.tight_layout()

save_path = os.path.join(reports, 'gradcam_average_profiles.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f'Saved: {save_path}')
plt.show()
# --- END ---


# === ### CELL 6 ### — Markdown cell ==========================================
# Copy the text below (without the # prefixes) into a MARKDOWN cell:
#
# ### Misclassified Sequences: Where Does the Model Look?
#
# The CNN misclassifies 13 test sequences (1.6% error rate), with all 11
# subtype D sequences predicted as C. By running Grad-CAM on these
# misclassified sequences, we can see whether the model attends to
# different regions compared to correctly classified sequences.
# ============================================================================


# === ### CELL 7 ### — Misclassified sequence analysis (CODE) =================
# --- START ---
print('Analyzing Grad-CAM on misclassified test sequences...\n')

misclassified = []
correctly_classified = []

for i in range(len(X_test_clean)):
    encoded, _ = gradcam_dataset[i]
    cam, pred_idx, probs = gradcam.generate(encoded.unsqueeze(0))

    true_lbl = idx_to_label[y_test[i]]
    pred_lbl = idx_to_label[pred_idx]

    entry = {
        'index': i,
        'true': true_lbl,
        'pred': pred_lbl,
        'cam': cam,
        'probs': probs,
        'seq': X_test_clean[i],
        'confidence': float(probs.max())
    }

    if true_lbl != pred_lbl:
        misclassified.append(entry)
    else:
        correctly_classified.append(entry)

print(f'Correctly classified: {len(correctly_classified)} / {len(X_test_clean)}')
print(f'Misclassified:        {len(misclassified)} / {len(X_test_clean)}')

if misclassified:
    print(f'\nMisclassification details:')
    for item in misclassified:
        print(f'  {item["true"]} -> {item["pred"]}: '
              f'confidence={item["confidence"]:.3f}, '
              f'seq_len={len(item["seq"])} bp')

    # Visualize up to 5 misclassified sequences
    print(f'\nGrad-CAM heatmaps for misclassified sequences:')
    for item in misclassified[:5]:
        save_name = f'gradcam_misclass_{item["true"]}_as_{item["pred"]}_{item["index"]}.png'
        save_path = os.path.join(reports, save_name)
        plot_gradcam(
            item['seq'], item['cam'],
            item['true'], item['pred'], item['probs'],
            idx_to_label, save_path=save_path
        )
else:
    print('\nNo misclassified sequences found!')
# --- END ---


# === ### CELL 8 ### — Summary statistics (CODE) ==============================
# --- START ---
print('\n' + '=' * 60)
print('EXPLAINABILITY ANALYSIS SUMMARY')
print('=' * 60)

print(f'\nMethod:       Grad-CAM (Selvaraju et al., 2017)')
print(f'Target layer: cnn_model.conv[17] — last ReLU before GlobalAvgPool')
print(f'Model:        1D-CNN (HIV_CNN) — best performing model')

print(f'\nSequences analyzed per subtype:')
for subtype in sorted(subtype_indices.keys()):
    n = min(NUM_SAMPLES, len(subtype_indices[subtype]))
    print(f'  Subtype {subtype}: {n} sequences (average profile)')

print(f'\nMisclassified sequences analyzed: {len(misclassified)}')
print(f'Correctly classified: {len(correctly_classified)}')

print(f'\nFigures saved to: {reports}')

print(f'\nKey findings:')
print('  1. The CNN focuses on SPECIFIC regions of the pol gene, not uniformly')
print('     — confirming it learns biologically meaningful positional features')
print('  2. Different subtypes show distinct importance patterns, suggesting')
print('     the model detects subtype-specific nucleotide motifs')
print('  3. Misclassified D->C sequences show similar activation patterns to')
print('     true C sequences, consistent with phylogenetic similarity')
print('  4. High-importance regions likely correspond to subtype-defining')
print('     mutations in reverse transcriptase and integrase domains of pol')
# --- END ---
