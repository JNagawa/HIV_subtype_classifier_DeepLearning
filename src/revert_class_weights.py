import json
import os

workspace = r"c:\Users\jnagawa\Downloads\hiv-subtype-classifier-20260529T180143Z-3-001\hiv-subtype-classifier"
nb04_path = os.path.join(workspace, "notebooks", "04_deep_learning_models.ipynb")

with open(nb04_path, "r", encoding="utf-8") as f:
    nb4 = json.load(f)

train_utils_idx = -1
for i, cell in enumerate(nb4['cells']):
    if cell.get('metadata', {}).get('id') == 'train_utils':
        train_utils_idx = i
        break

if train_utils_idx != -1:
    source = "".join(nb4['cells'][train_utils_idx]['source'])
    # Replace the class weights logic back to simple CrossEntropyLoss
    old_logic = "try:\n        weights = torch.load(os.path.join(projectdir, 'data', 'processed', 'class_weights.pt')).to(device)\n        criterion = nn.CrossEntropyLoss(weight=weights)\n        print('Loaded balanced class weights for CrossEntropyLoss.')\n    except FileNotFoundError:\n        criterion = nn.CrossEntropyLoss()\n        print('Class weights not found, using standard CrossEntropyLoss.')"
    new_source = source.replace(old_logic, "criterion = nn.CrossEntropyLoss()")
    nb4['cells'][train_utils_idx]['source'] = [line + "\n" for line in new_source.split("\n")][:-1]

with open(nb04_path, "w", encoding="utf-8") as f:
    json.dump(nb4, f, indent=1)

print("Rolled back to standard CrossEntropyLoss!")
