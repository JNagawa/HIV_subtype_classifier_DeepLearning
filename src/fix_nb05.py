import json
import os

workspace = r"c:\Users\jnagawa\Downloads\hiv-subtype-classifier-20260529T180143Z-3-001\hiv-subtype-classifier"
nb05_path = os.path.join(workspace, "notebooks", "05_evaluation_and_analysis.ipynb")

with open(nb05_path, "r", encoding="utf-8") as f:
    nb5 = json.load(f)

# Find the indices of the cells to replace
cell1_idx = -1
for i, cell in enumerate(nb5['cells']):
    source = "".join(cell.get('source', []))
    if "class HIV_MLP" in source and "def load_model(" in source:
        cell1_idx = i
        break

if cell1_idx != -1:
    new_source = """# ===== Load model architectures from src/models.py =====
import sys
sys.path.append(os.path.join(projectdir, 'src'))
from models import get_model

# Load trained weights
def load_trained_model(name):
    # Map from 'HIV_CNN' to 'cnn'
    short_name = name.split('_')[1].lower()
    m = get_model(short_name, num_classes=num_classes, device=device)
    ckpt = torch.load(os.path.join(models_dir, f'{name}_best.pth'), map_location=device)
    m.load_state_dict(ckpt['state_dict'])
    m.eval()
    return m

mlp = load_trained_model('HIV_MLP')
cnn = load_trained_model('HIV_CNN')
bilstm = load_trained_model('HIV_BiLSTM')
print('All 3 models loaded successfully from src/models.py!')"""
    
    # Replace cell 1 with the new combined code
    nb5['cells'][cell1_idx]['source'] = [line + "\n" for line in new_source.split("\n")]
    nb5['cells'][cell1_idx]['source'][-1] = nb5['cells'][cell1_idx]['source'][-1].strip()
    
    with open(nb05_path, "w", encoding="utf-8") as f:
        json.dump(nb5, f, indent=1)
    print("Notebook 05 successfully fixed! Cell was fully replaced.")
else:
    print("Failed to find the combined cell.")
