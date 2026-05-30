import json
import os
import shutil

workspace = r"c:\Users\jnagawa\Downloads\hiv-subtype-classifier-20260529T180143Z-3-001\hiv-subtype-classifier"
nb04_path = os.path.join(workspace, "notebooks", "04_deep_learning_models.ipynb")
nb05_path = os.path.join(workspace, "notebooks", "05_evaluation_and_analysis.ipynb")

# backup
shutil.copy(nb04_path, nb04_path + ".bak")
shutil.copy(nb05_path, nb05_path + ".bak")

with open(nb04_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

def create_code_cell(source_code):
    lines = [line + "\n" for line in source_code.split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": lines
    }

def create_md_cell(source_text):
    lines = [line + "\n" for line in source_text.split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines
    }

cnn_md = create_md_cell("## 1D-CNN — Test Set Evaluation")
cnn_code1 = create_code_cell('''from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import label_binarize
class_names = [idx_to_label[i] for i in range(num_classes)]
def get_predictions(model, dl):
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

cnn_yt, cnn_yp, cnn_yprob = get_predictions(cnn_model, test_dl)
cnn_acc = accuracy_score(cnn_yt, cnn_yp)
print('=== 1D-CNN — Test Set Results ===')
print(classification_report(cnn_yt, cnn_yp, target_names=class_names, digits=4))''')

cnn_code2 = create_code_cell('''import seaborn as sns
cm = confusion_matrix(cnn_yt, cnn_yp)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=axes[0])
axes[0].set_title('1D-CNN — Confusion Matrix (Counts)')
axes[0].set_ylabel('True'); axes[0].set_xlabel('Predicted')
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=axes[1])
axes[1].set_title('1D-CNN — Confusion Matrix (Normalized)')
axes[1].set_ylabel('True'); axes[1].set_xlabel('Predicted')
plt.tight_layout()
plt.show()''')

gc_md = create_md_cell("## Model Explainability: Grad-CAM Analysis")
gc_code1 = create_code_cell('''import torch.nn.functional as F
from matplotlib.colors import LinearSegmentedColormap
class GradCAM1D:
    def __init__(self, model, target_layer):
        self.model = model; self.gradients = None; self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)
    def _save_activation(self, module, inp, out): self.activations = out.detach()
    def _save_gradient(self, module, grad_in, grad_out): self.gradients = grad_out[0].detach()
    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        dev = next(self.model.parameters()).device
        x = input_tensor.to(dev).requires_grad_(True)
        output = self.model(x)
        probs = F.softmax(output, dim=1)[0].detach().cpu().numpy()
        pred_class = output.argmax(dim=1).item()
        if target_class is None: target_class = pred_class
        self.model.zero_grad()
        output[0, target_class].backward()
        weights = self.gradients.mean(dim=2, keepdim=True)
        cam = (weights * self.activations).sum(dim=1)[0]
        cam = F.relu(cam)
        cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), size=input_tensor.shape[2], mode='linear', align_corners=False)[0, 0].cpu().numpy()
        if cam.max() > 0: cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam, pred_class, probs

target_layer = cnn_model.conv[17]
gradcam = GradCAM1D(cnn_model, target_layer)''')

gc_code2 = create_code_cell('''def moving_avg(arr, window=50): return np.convolve(arr, np.ones(window)/window, mode='same')
def plot_gradcam(sequence, cam, true_label, pred_label, probs, idx_to_label):
    fig, axes = plt.subplots(2, 1, figsize=(16, 5), gridspec_kw={'height_ratios': [2, 1]})
    cmap = LinearSegmentedColormap.from_list('hiv_gradcam', ['#1a1a2e', '#16213e', '#e94560', '#f5a623'], N=256)
    im = axes[0].imshow(cam.reshape(1, -1), aspect='auto', cmap=cmap, interpolation='bilinear')
    axes[0].set_yticks([])
    labels_list = [idx_to_label[i] for i in sorted(idx_to_label.keys())]
    pred_idx = labels_list.index(pred_label)
    conf = probs[pred_idx] * 100
    axes[0].set_title(f'True: {true_label} | Pred: {pred_label} ({conf:.1f}%)', fontweight='bold')
    plt.colorbar(im, ax=axes[0], orientation='vertical', label='Importance')
    cam_smooth = moving_avg(cam, window=100)
    axes[1].fill_between(range(len(cam_smooth)), cam_smooth, alpha=0.5, color='#e94560')
    axes[1].plot(cam_smooth, color='#c0392b')
    axes[1].set_xlim(0, len(cam))
    plt.show()''')

gc_code3 = create_code_cell('''from data_loader import strip_gaps
X_test_clean = [strip_gaps(s) for s in X_test]
gradcam_dataset = HIVSequenceDataset(X_test_clean, y_test, MAX_LENGTH, 'onehot')
subtype_indices = {}
for i, y in enumerate(y_test):
    label = idx_to_label[y]
    if label not in subtype_indices: subtype_indices[label] = []
    subtype_indices[label].append(i)

for subtype in sorted(subtype_indices.keys()):
    if not subtype_indices[subtype]: continue
    idx = subtype_indices[subtype][0]
    encoded, _ = gradcam_dataset[idx]
    cam, pred_idx, probs = gradcam.generate(encoded.unsqueeze(0))
    plot_gradcam(X_test_clean[idx], cam, idx_to_label[y_test[idx]], idx_to_label[pred_idx], probs, idx_to_label)''')

lstm_md = create_md_cell("## BiLSTM — Test Set Evaluation")
lstm_code1 = create_code_cell('''lstm_yt, lstm_yp, lstm_yprob = get_predictions(lstm_model, test_dl_lbl)
lstm_acc = accuracy_score(lstm_yt, lstm_yp)
print('=== BiLSTM — Test Set Results ===')
print(classification_report(lstm_yt, lstm_yp, target_names=class_names, digits=4))''')

lstm_code2 = create_code_cell('''cm_lstm = confusion_matrix(lstm_yt, lstm_yp)
cm_lstm_norm = cm_lstm.astype('float') / cm_lstm.sum(axis=1)[:, np.newaxis]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.heatmap(cm_lstm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=axes[0])
axes[0].set_title('BiLSTM — Confusion Matrix')
sns.heatmap(cm_lstm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=axes[1])
axes[1].set_title('BiLSTM — Confusion Matrix (Normalized)')
plt.show()''')

# Find insertion points
cnn_idx, lstm_idx = -1, -1
for i, cell in enumerate(nb['cells']):
    if cell.get('metadata', {}).get('id') == 'cnn_train': cnn_idx = i
    if cell.get('metadata', {}).get('id') == 'lstm_train': lstm_idx = i

if lstm_idx != -1:
    nb['cells'].insert(lstm_idx + 1, lstm_code2)
    nb['cells'].insert(lstm_idx + 1, lstm_code1)
    nb['cells'].insert(lstm_idx + 1, lstm_md)

if cnn_idx != -1:
    nb['cells'].insert(cnn_idx + 1, gc_code3)
    nb['cells'].insert(cnn_idx + 1, gc_code2)
    nb['cells'].insert(cnn_idx + 1, gc_code1)
    nb['cells'].insert(cnn_idx + 1, gc_md)
    nb['cells'].insert(cnn_idx + 1, cnn_code2)
    nb['cells'].insert(cnn_idx + 1, cnn_code1)
    nb['cells'].insert(cnn_idx + 1, cnn_md)

with open(nb04_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Updated 04_deep_learning_models.ipynb")

# Update Notebook 05
with open(nb05_path, "r", encoding="utf-8") as f:
    nb5 = json.load(f)

if len(nb5['cells']) > 0 and nb5['cells'][0]['cell_type'] == 'markdown':
    source = nb5['cells'][0]['source']
    for i, s in enumerate(source):
        if "Evaluation, Error Analysis & Model Comparison" in s:
            source[i] = s.replace("Evaluation, Error Analysis & Model Comparison", "Model Comparison & Error Analysis")

with open(nb05_path, "w", encoding="utf-8") as f:
    json.dump(nb5, f, indent=1)

print("Updated 05_evaluation_and_analysis.ipynb")
