import torch
import os

models = ['HIV_MLP_best.pth', 'HIV_CNN_best.pth', 'HIV_BiLSTM_best.pth']

for m in models:
    path = os.path.join('models', m)
    if os.path.exists(path):
        try:
            data = torch.load(path, map_location='cpu', weights_only=False)
            if 'history' in data:
                hist = data['history']
                train_acc = hist['train_acc'][-1]
                val_acc = hist['val_acc'][-1]
                print(f"{m}:")
                print(f"  Training Accuracy:   {train_acc:.4f}")
                print(f"  Validation Accuracy: {val_acc:.4f}")
                print(f"  Difference (Overfit): {train_acc - val_acc:.4f}")
            else:
                print(f"{m}: No history found.")
        except Exception as e:
            print(f"Error loading {m}: {e}")
