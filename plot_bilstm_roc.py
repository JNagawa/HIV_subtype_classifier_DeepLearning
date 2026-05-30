import os
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader

import sys
sys.path.append('src')
from data_loader import HIVSequenceDataset, collate_label
from models import HIV_BiLSTM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths
processed_data = 'data/processed'
models_dir = 'models'
reports = 'reports/figures'

# Load Data
with open(os.path.join(processed_data, 'preprocessed_data.pkl'), 'rb') as f:
    data = pickle.load(f)

X_test, y_test = data['X_test'], data['y_test']
idx_to_label = data['idx_to_label']
num_classes = data['num_classes']
MAX_LENGTH = data['max_length']
class_names = [idx_to_label[i] for i in range(num_classes)]

# DataLoader
test_dl_lbl = DataLoader(HIVSequenceDataset(X_test, y_test, MAX_LENGTH, 'label'),
                         batch_size=32, num_workers=0, collate_fn=collate_label)

# Init Model
bilstm = HIV_BiLSTM(num_classes, vocab=6).to(device)

# Load Weights
bilstm.load_state_dict(torch.load(os.path.join(models_dir, 'HIV_BiLSTM_best.pth'), map_location=device)['state_dict'])

bilstm.eval()
all_probs = []
with torch.no_grad():
    for batch in test_dl_lbl:
        X, _ = batch
        X = X.to(device)
        logits = bilstm(X)
        probs = torch.softmax(logits, dim=1)
        all_probs.append(probs.cpu().numpy())

yprob = np.concatenate(all_probs, axis=0)
y_bin = label_binarize(y_test, classes=range(num_classes))

# Plot
fig, ax = plt.subplots(figsize=(10, 8))
colors = plt.cm.Set2(np.linspace(0, 1, num_classes))

for i, (cn, col) in enumerate(zip(class_names, colors)):
    fpr, tpr, _ = roc_curve(y_bin[:, i], yprob[:, i])
    ax.plot(fpr, tpr, color=col, lw=2, label=f'{cn} (AUC={auc(fpr,tpr):.3f})')
    
ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('BiLSTM — ROC Curves (One-vs-Rest)', fontsize=14)
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(reports, 'best_model_roc.png'), dpi=150, bbox_inches='tight')
plt.savefig(os.path.join('reports/latex_project/figures', 'best_model_roc.png'), dpi=150, bbox_inches='tight')
print("Successfully generated BiLSTM ROC curves as best_model_roc.png")
