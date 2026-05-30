import json
import os

workspace = r"c:\Users\jnagawa\Downloads\hiv-subtype-classifier-20260529T180143Z-3-001\hiv-subtype-classifier"
nb04b_path = os.path.join(workspace, "notebooks", "04b_transfer_learning.ipynb")

with open(nb04b_path, "r", encoding="utf-8") as f:
    nb4b = json.load(f)

for cell in nb4b['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell.get('source', []))
        if "criterion = nn.CrossEntropyLoss()" in source and "def train_epoch(" in source:
            old_logic = "criterion = nn.CrossEntropyLoss()"
            new_logic = "try:\n    weights = torch.load(os.path.join(projectdir, 'data', 'processed', 'class_weights.pt')).to(device)\n    criterion = nn.CrossEntropyLoss(weight=weights)\n    print('Loaded balanced class weights for CrossEntropyLoss.')\nexcept FileNotFoundError:\n    criterion = nn.CrossEntropyLoss()\n    print('Class weights not found, using standard CrossEntropyLoss.')"
            new_source = source.replace(old_logic, new_logic)
            cell['source'] = [line + "\n" for line in new_source.split("\n")][:-1]
            break

with open(nb04b_path, "w", encoding="utf-8") as f:
    json.dump(nb4b, f, indent=1)

print("Notebook 04b successfully updated!")
