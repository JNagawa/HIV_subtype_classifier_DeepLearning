import json
import os

workspace = r"c:\Users\jnagawa\Downloads\hiv-subtype-classifier-20260529T180143Z-3-001\hiv-subtype-classifier"
nb04_path = os.path.join(workspace, "notebooks", "04_deep_learning_models.ipynb")

with open(nb04_path, "r", encoding="utf-8") as f:
    nb4 = json.load(f)

# Find indices again to be safe
to_delete = []
for i, cell in enumerate(nb4['cells']):
    source = "".join(cell.get('source', []))
    if 'Grad-CAM' in source or 'class GradCAM1D' in source:
        to_delete.append(i)

# Delete in reverse order to avoid shifting indices
for i in sorted(to_delete, reverse=True):
    del nb4['cells'][i]

with open(nb04_path, "w", encoding="utf-8") as f:
    json.dump(nb4, f, indent=1)

print("Grad-CAM successfully removed from Notebook 04!")
