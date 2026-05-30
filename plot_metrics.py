import pandas as pd
import os
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader

import sys
sys.path.append('src')
from data_loader import HIVSequenceDataset, collate_onehot, collate_label
from models import HIV_MLP, HIV_CNN, HIV_BiLSTM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

processed_data = 'data/processed'
models_dir = 'models'
reports = 'reports/figures'

with open(os.path.join(processed_data, 'preprocessed_data.pkl'), 'rb') as f:
    data = pickle.load(f)

X_test, y_test = data['X_test'], data['y_test']
idx_to_label = data['idx_to_label']
num_classes = data['num_classes']
MAX_LENGTH = data['max_length']
class_names = [idx_to_label[i] for i in range(num_classes)]

test_dl_oh = DataLoader(HIVSequenceDataset(X_test, y_test, MAX_LENGTH, 'onehot'),
                        batch_size=32, num_workers=0, collate_fn=collate_onehot)
test_dl_lbl = DataLoader(HIVSequenceDataset(X_test, y_test, MAX_LENGTH, 'label'),
                         batch_size=32, num_workers=0, collate_fn=collate_label)

mlp = HIV_MLP(num_classes).to(device)
cnn = HIV_CNN(num_classes).to(device)
bilstm = HIV_BiLSTM(num_classes, vocab=6).to(device)

mlp.load_state_dict(torch.load(os.path.join(models_dir, 'HIV_MLP_best.pth'), map_location=device)['state_dict'])
cnn.load_state_dict(torch.load(os.path.join(models_dir, 'HIV_CNN_best.pth'), map_location=device)['state_dict'])
bilstm.load_state_dict(torch.load(os.path.join(models_dir, 'HIV_BiLSTM_best.pth'), map_location=device)['state_dict'])

def get_preds(model, dataloader):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in dataloader:
            X, _ = batch
            X = X.to(device)
            logits = model(X)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
    return np.array(all_preds)

preds_mlp = get_preds(mlp, test_dl_oh)
preds_cnn = get_preds(cnn, test_dl_oh)
preds_bilstm = get_preds(bilstm, test_dl_lbl)

# Confusion Matrices
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
models_preds = [('Baseline MLP', preds_mlp), ('1D-CNN', preds_cnn), ('BiLSTM', preds_bilstm)]

for ax, (name, ypred) in zip(axes, models_preds):
    cm = confusion_matrix(y_test, ypred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=class_names, yticklabels=class_names)
    ax.set_title(f'{name} Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')

plt.tight_layout()
plt.savefig(os.path.join(reports, 'all_confusion_matrices.png'), dpi=150, bbox_inches='tight')
plt.savefig('reports/latex_project/figures/all_confusion_matrices.png', dpi=150, bbox_inches='tight')

# Per-Class Metrics for BiLSTM
p, r, f, _ = precision_recall_fscore_support(y_test, preds_bilstm)
metrics_df = pd.DataFrame({'Precision': p, 'Recall': r, 'F1-Score': f}, index=class_names)

fig, ax = plt.subplots(figsize=(10, 6))
metrics_df.plot(kind='bar', ax=ax, colormap='viridis', edgecolor='black')
ax.set_title('BiLSTM - Per-Class Performance Metrics', fontsize=14)
ax.set_ylabel('Score')
ax.set_ylim(0, 1.1)
plt.xticks(rotation=0)
plt.legend(loc='lower right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(reports, 'best_model_per_class_metrics.png'), dpi=150, bbox_inches='tight')
plt.savefig('reports/latex_project/figures/best_model_per_class_metrics.png', dpi=150, bbox_inches='tight')

print("Successfully generated all_confusion_matrices.png and best_model_per_class_metrics.png")
