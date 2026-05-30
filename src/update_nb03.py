import json
import os

workspace = r"c:\Users\jnagawa\Downloads\hiv-subtype-classifier-20260529T180143Z-3-001\hiv-subtype-classifier"
nb03_path = os.path.join(workspace, "notebooks", "03_baseline_model.ipynb")

with open(nb03_path, "r", encoding="utf-8") as f:
    nb3 = json.load(f)

for cell in nb3['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell.get('source', []))
        if "criterion = nn.CrossEntropyLoss()" in source and "def train_model(" in source:
            old_logic = "criterion = nn.CrossEntropyLoss()"
            new_logic = "try:\n        weights = torch.load(os.path.join(projectdir, 'data', 'processed', 'class_weights.pt')).to(device)\n        criterion = nn.CrossEntropyLoss(weight=weights)\n        print('Loaded balanced class weights for CrossEntropyLoss.')\n    except FileNotFoundError:\n        criterion = nn.CrossEntropyLoss()\n        print('Class weights not found, using standard CrossEntropyLoss.')"
            new_source = source.replace(old_logic, new_logic)
            cell['source'] = [line + "\n" for line in new_source.split("\n")][:-1]
            break

with open(nb03_path, "w", encoding="utf-8") as f:
    json.dump(nb3, f, indent=1)

print("Notebook 03 successfully updated!")
