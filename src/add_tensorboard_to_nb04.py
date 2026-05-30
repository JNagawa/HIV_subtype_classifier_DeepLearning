import json
import os

workspace = r"c:\Users\jnagawa\Downloads\hiv-subtype-classifier-20260529T180143Z-3-001\hiv-subtype-classifier"
nb04_path = os.path.join(workspace, "notebooks", "04_deep_learning_models.ipynb")

with open(nb04_path, "r", encoding="utf-8") as f:
    nb4 = json.load(f)

# Find train_utils index
train_utils_idx = -1
for i, cell in enumerate(nb4['cells']):
    if cell.get('metadata', {}).get('id') == 'train_utils':
        train_utils_idx = i
        break

new_train_utils = """import copy
from torch.utils.tensorboard import SummaryWriter

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for inputs, labels in tqdm(loader, desc='  Train', leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        _, pred = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()
    return total_loss/total, correct/total

def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc='  Val', leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            _, pred = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
    return total_loss/total, correct/total

def train_model(model, train_dl, val_dl, model_name, epochs=50, lr=1e-3, patience=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Set up TensorBoard writer
    log_dir = os.path.join(projectdir, 'logs', model_name)
    writer = SummaryWriter(log_dir=log_dir)

    history = {'train_loss':[], 'train_acc':[], 'val_loss':[], 'val_acc':[], 'lr':[]}
    best_val_loss = float('inf')
    best_state = None
    no_improve = 0

    print(f'\\n{"="*50}\\nTraining: {model_name}\\n{"="*50}')
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Parameters: {total_params:,}')
    print(f'TensorBoard logging to: {log_dir}')

    for epoch in range(epochs):
        lr_now = optimizer.param_groups[0]['lr']
        t_loss, t_acc = train_epoch(model, train_dl, criterion, optimizer)
        v_loss, v_acc = eval_epoch(model, val_dl, criterion)
        scheduler.step()

        history['train_loss'].append(t_loss)
        history['train_acc'].append(t_acc)
        history['val_loss'].append(v_loss)
        history['val_acc'].append(v_acc)
        history['lr'].append(lr_now)
        
        # Log to TensorBoard
        writer.add_scalars('Loss', {'train': t_loss, 'val': v_loss}, epoch)
        writer.add_scalars('Accuracy', {'train': t_acc, 'val': v_acc}, epoch)
        writer.add_scalar('Learning Rate', lr_now, epoch)

        print(f'Epoch {epoch+1:2d}/{epochs} | '
              f'TL: {t_loss:.4f} TA: {t_acc:.4f} | '
              f'VL: {v_loss:.4f} VA: {v_acc:.4f} | LR: {lr_now:.6f}')

        if v_loss < best_val_loss - 0.001:
            best_val_loss = v_loss
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
                
    writer.close()

    if best_state:
        model.load_state_dict(best_state)
    torch.save({'state_dict': model.state_dict(), 'history': history},
               os.path.join(models_dir, f'{model_name}_best.pth'))
    print(f'Best val loss: {best_val_loss:.4f}')
    return history

print('Training utilities ready with TensorBoard Tracking.')
"""

tb_magic_code = """%load_ext tensorboard
import os
# Ensure logs directory exists
os.makedirs(os.path.join(projectdir, 'logs'), exist_ok=True)
%tensorboard --logdir "{os.path.join(projectdir, 'logs')}\""""

if train_utils_idx != -1:
    nb4['cells'][train_utils_idx]['source'] = [line + "\n" for line in new_train_utils.split("\n")][:-1]
    
    # Check if tensorboard cell already exists
    has_tb = any("%load_ext tensorboard" in "".join(c.get("source", [])) for c in nb4['cells'])
    if not has_tb:
        tb_cell = {
            "cell_type": "code",
            "metadata": {"id": "tensorboard_magic"},
            "execution_count": None,
            "outputs": [],
            "source": [line + "\n" for line in tb_magic_code.split("\n")][:-1]
        }
        nb4['cells'].insert(train_utils_idx + 1, tb_cell)

with open(nb04_path, "w", encoding="utf-8") as f:
    json.dump(nb4, f, indent=1)

print("TensorBoard integration added successfully!")
