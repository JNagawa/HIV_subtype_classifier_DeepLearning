import json
import os

workspace = r"c:\Users\jnagawa\Downloads\hiv-subtype-classifier-20260529T180143Z-3-001\hiv-subtype-classifier"
nb04_path = os.path.join(workspace, "notebooks", "04_deep_learning_models.ipynb")

with open(nb04_path, "r", encoding="utf-8") as f:
    nb4 = json.load(f)

for cell in nb4['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell.get('source', []))
        # Part 1: Remove WeightedRandomSampler
        if "WeightedRandomSampler" in source and "def make_loaders(" in source:
            old_sampler = """    # Implement WeightedRandomSampler for class balancing
    class_counts = Counter(y_train)
    weights_dict = {cls: 1.0 / count for cls, count in class_counts.items()}
    sample_weights = [weights_dict[y] for y in y_train]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(y_train), replacement=True)

    # Create loaders
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                          num_workers=2, pin_memory=True, collate_fn=collate_fn)"""
            new_sampler = """    # Create loaders
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=2, pin_memory=True, collate_fn=collate_fn)"""
            
            new_source = source.replace(old_sampler, new_sampler)
            # Remove the import just in case
            new_source = new_source.replace("from torch.utils.data import WeightedRandomSampler", "")
            cell['source'] = [line + "\n" for line in new_source.split("\n")][:-1]
            
        # Part 2: Add class weights to train_utils
        if "criterion = nn.CrossEntropyLoss()" in source and "def train_model(" in source:
            old_logic = "criterion = nn.CrossEntropyLoss()"
            new_logic = "try:\n        weights = torch.load(os.path.join(projectdir, 'data', 'processed', 'class_weights.pt')).to(device)\n        criterion = nn.CrossEntropyLoss(weight=weights)\n        print('Loaded balanced class weights for CrossEntropyLoss.')\n    except FileNotFoundError:\n        criterion = nn.CrossEntropyLoss()\n        print('Class weights not found, using standard CrossEntropyLoss.')"
            new_source = "".join(cell.get('source', [])).replace(old_logic, new_logic)
            cell['source'] = [line + "\n" for line in new_source.split("\n")][:-1]

with open(nb04_path, "w", encoding="utf-8") as f:
    json.dump(nb4, f, indent=1)

print("Notebook 04 successfully updated!")
