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
from data_loader import HIVSequenceDataset, collate_onehot, collate_label
from models import HIV_MLP, HIV_CNN, HIV_BiLSTM

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

# DataLoaders
test_dl_oh = DataLoader(HIVSequenceDataset(X_test, y_test, MAX_LENGTH, 'onehot'),
                        batch_size=32, num_workers=0, collate_fn=collate_onehot)
test_dl_lbl = DataLoader(HIVSequenceDataset(X_test, y_test, MAX_LENGTH, 'label'),
                         batch_size=32, num_workers=0, collate_fn=collate_label)

# Init Models
mlp = HIV_MLP(num_classes).to(device)
cnn = HIV_CNN(num_classes).to(device)
bilstm = HIV_BiLSTM(num_classes, vocab=6).to(device)

# Load Weights
mlp.load_state_dict(torch.load(os.path.join(models_dir, 'HIV_MLP_best.pth'), map_location=device)['state_dict'])
cnn.load_state_dict(torch.load(os.path.join(models_dir, 'HIV_CNN_best.pth'), map_location=device)['state_dict'])
bilstm.load_state_dict(torch.load(os.path.join(models_dir, 'HIV_BiLSTM_best.pth'), map_location=device)['state_dict'])

models = {
    'Baseline MLP': (mlp, test_dl_oh),
    '1D-CNN': (cnn, test_dl_oh),
    'BiLSTM': (bilstm, test_dl_lbl)
}

def get_probs(model, dataloader):
    model.eval()
    all_probs = []
    with torch.no_grad():
        for batch in dataloader:
            X, _ = batch
            X = X.to(device)
            logits = model(X)
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())
    return np.concatenate(all_probs, axis=0)

y_bin = label_binarize(y_test, classes=range(num_classes))

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
colors = plt.cm.Set2(np.linspace(0, 1, num_classes))

for ax, (name, (model, dl)) in zip(axes, models.items()):
    yprob = get_probs(model, dl)
    
    for i, (cn, col) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], yprob[:, i])
        ax.plot(fpr, tpr, color=col, lw=2, label=f'{cn} (AUC={auc(fpr,tpr):.3f})')
        
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'{name} - ROC Curves')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(reports, 'all_models_roc.png'), dpi=150, bbox_inches='tight')
print("Successfully generated all_models_roc.png")
