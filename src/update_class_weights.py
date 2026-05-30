import json
import os

workspace = r"c:\Users\jnagawa\Downloads\hiv-subtype-classifier-20260529T180143Z-3-001\hiv-subtype-classifier"
nb02_path = os.path.join(workspace, "notebooks", "02_preprocessing.ipynb")
nb04_path = os.path.join(workspace, "notebooks", "04_deep_learning_models.ipynb")

# Update NB02
with open(nb02_path, "r", encoding="utf-8") as f:
    nb2 = json.load(f)

has_weights = any("compute_class_weight" in "".join(c.get("source", [])) for c in nb2['cells'])
if not has_weights:
    weights_code = """# ===== Compute Class Weights =====
from sklearn.utils.class_weight import compute_class_weight

# Compute balanced class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

print("Class weights for [A, B, C, D]:")
print(class_weights)

# Save to processed data folder
weights_path = os.path.join(processed_data, 'class_weights.pt')
torch.save(class_weights_tensor, weights_path)
print(f"Class weights saved to {weights_path}")"""
    
    weights_cell = {
        "cell_type": "code",
        "metadata": {"id": "class_weights_compute"},
        "execution_count": None,
        "outputs": [],
        "source": [line + "\n" for line in weights_code.split("\n")][:-1]
    }
    
    nb2['cells'].append({
        "cell_type": "markdown",
        "metadata": {"id": "weights_header"},
        "source": ["## Step 5: Compute Class Weights\n", "To prevent overfitting to the majority classes (B and C), we compute balanced class weights and save them to be used by the loss function in our deep learning models."]
    })
    nb2['cells'].append(weights_cell)

with open(nb02_path, "w", encoding="utf-8") as f:
    json.dump(nb2, f, indent=1)

# Update NB04
with open(nb04_path, "r", encoding="utf-8") as f:
    nb4 = json.load(f)

train_utils_idx = -1
for i, cell in enumerate(nb4['cells']):
    if cell.get('metadata', {}).get('id') == 'train_utils':
        train_utils_idx = i
        break

if train_utils_idx != -1:
    source = "".join(nb4['cells'][train_utils_idx]['source'])
    if "class_weights.pt" not in source:
        new_source = source.replace(
            "criterion = nn.CrossEntropyLoss()",
            "try:\n        weights = torch.load(os.path.join(projectdir, 'data', 'processed', 'class_weights.pt')).to(device)\n        criterion = nn.CrossEntropyLoss(weight=weights)\n        print('Loaded balanced class weights for CrossEntropyLoss.')\n    except FileNotFoundError:\n        criterion = nn.CrossEntropyLoss()\n        print('Class weights not found, using standard CrossEntropyLoss.')"
        )
        nb4['cells'][train_utils_idx]['source'] = [line + "\n" for line in new_source.split("\n")][:-1]

with open(nb04_path, "w", encoding="utf-8") as f:
    json.dump(nb4, f, indent=1)

print("Notebook 02 and 04 successfully updated with Class Weights logic!")
