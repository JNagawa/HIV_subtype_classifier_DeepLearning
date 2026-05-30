import json
import os

workspace = r"c:\Users\jnagawa\Downloads\hiv-subtype-classifier-20260529T180143Z-3-001\hiv-subtype-classifier"
nb_path = os.path.join(workspace, "notebooks", "06_deployment.ipynb")

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 🚀 Notebook 06: Model Deployment (Web App)\n",
            "\n",
            "This notebook launches the **Gradio** web interface for our HIV-1 Subtype Classifier.\n",
            "\n",
            "By running this notebook, you will instantly get a public URL (`https://...gradio.live`) that you can share with colleagues. They can use it to paste nucleotide sequences and get live predictions!"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "!pip install gradio -q"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "import numpy as np\n",
            "import torch\n",
            "import torch.nn as nn\n",
            "import gradio as gr\n",
            "\n",
            "# Mount Google Drive\n",
            "from google.colab import drive\n",
            "drive.mount('/content/drive')\n",
            "\n",
            "# Project paths\n",
            "projectdir = '/content/drive/MyDrive/Deep Learning/hiv-subtype-classifier'\n",
            "models_dir = os.path.join(projectdir, 'models')\n",
            "sys.path.append(os.path.join(projectdir, 'src'))\n",
            "\n",
            "# Import the model architecture dynamically (just like Notebook 05)\n",
            "from models import get_model"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
            "print(f'Using device: {device}')\n",
            "\n",
            "# Settings\n",
            "NUM_CLASSES = 4\n",
            "IDX_TO_LABEL = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}\n",
            "NUC_TO_IDX = {'A': 0, 'C': 1, 'G': 2, 'T': 3}\n",
            "NUM_CHANNELS = 4\n",
            "\n",
            "MC_PASSES = 10\n",
            "UNCERTAINTY_STD = 0.15\n",
            "MIN_CONFIDENCE = 0.70\n",
            "\n",
            "# Load the best 1D-CNN Model\n",
            "model_path = os.path.join(models_dir, 'HIV_CNN_best.pth')\n",
            "model = get_model('cnn', num_classes=NUM_CLASSES, device=device)\n",
            "\n",
            "if os.path.exists(model_path):\n",
            "    ckpt = torch.load(model_path, map_location=device)\n",
            "    model.load_state_dict(ckpt['state_dict'])\n",
            "    print(\"Model loaded successfully!\")\n",
            "else:\n",
            "    print(f\"WARNING: Model not found at {model_path}. Please run training first.\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "def encode_sequence(seq_text):\n",
            "    lines = seq_text.strip().split('\\n')\n",
            "    seq = ''\n",
            "    for line in lines:\n",
            "        line = line.strip()\n",
            "        if line.startswith('>'): continue\n",
            "        seq += ''.join(c for c in line.upper() if c in 'ACGTNRYSWKMBDHV-')\n",
            "\n",
            "    seq = seq.replace('-', '').replace('.', '')\n",
            "    seq = ''.join(c if c in 'ACGT' else 'N' for c in seq)\n",
            "\n",
            "    if len(seq) < 100:\n",
            "        return None, \"Sequence too short. Please provide at least 100 nucleotides.\"\n",
            "\n",
            "    seq_len = len(seq)\n",
            "    encoded = np.zeros((NUM_CHANNELS, seq_len), dtype=np.float32)\n",
            "    for i, nuc in enumerate(seq):\n",
            "        idx = NUC_TO_IDX.get(nuc)\n",
            "        if idx is not None:\n",
            "            encoded[idx, i] = 1.0\n",
            "    return torch.tensor(encoded).unsqueeze(0), None\n",
            "\n",
            "def enable_mc_dropout(model):\n",
            "    for module in model.modules():\n",
            "        if isinstance(module, nn.Dropout):\n",
            "            module.train()\n",
            "\n",
            "def mc_dropout_predict(encoded, n_passes=MC_PASSES):\n",
            "    model.eval()\n",
            "    enable_mc_dropout(model)\n",
            "    \n",
            "    all_probs = []\n",
            "    encoded = encoded.to(device)\n",
            "    \n",
            "    for _ in range(n_passes):\n",
            "        with torch.no_grad():\n",
            "            logits = model(encoded)\n",
            "            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()\n",
            "            all_probs.append(probs)\n",
            "    \n",
            "    all_probs = np.array(all_probs)\n",
            "    mean_probs = all_probs.mean(axis=0)\n",
            "    std_probs = all_probs.std(axis=0)\n",
            "    \n",
            "    is_uncertain = (std_probs.max() > UNCERTAINTY_STD) or (mean_probs.max() < MIN_CONFIDENCE)\n",
            "    return mean_probs, std_probs, is_uncertain\n",
            "\n",
            "def classify_sequence(sequence_text):\n",
            "    if not sequence_text or len(sequence_text.strip()) < 10:\n",
            "        return {\"Error\": \"Please paste a valid HIV-1 nucleotide sequence.\"}\n",
            "\n",
            "    encoded, error = encode_sequence(sequence_text)\n",
            "    if error: return {\"Error\": error}\n",
            "\n",
            "    mean_probs, std_probs, is_uncertain = mc_dropout_predict(encoded)\n",
            "    \n",
            "    if is_uncertain:\n",
            "        predicted_idx = mean_probs.argmax()\n",
            "        predicted_label = IDX_TO_LABEL[predicted_idx]\n",
            "        return {\n",
            "            f\"⚠️ Indeterminate (leaning {predicted_label})\": float(mean_probs.max()),\n",
            "            \"High uncertainty detected\": float(std_probs.max()),\n",
            "            \"Recommendation\": 0.0,\n",
            "        }\n",
            "    \n",
            "    result = {}\n",
            "    for idx in range(NUM_CLASSES):\n",
            "        result[IDX_TO_LABEL[idx]] = float(mean_probs[idx])\n",
            "    return result"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "EXAMPLE_SEQ = \"\"\">Example_HIV1_sequence\\nATGGGTGCGAGAGCGTCAGTATTAAGCGGGGGAGAATTAGATCGATGGGAAAAAATTCGGTTAAGGCCAGGGGGAAAGAAAAAATATAAATTAAAACATATAGTATGGGCAAGCAGGGAGCTAGAACGATTCGCAGTTAATCCTGGCCTGTTAGAAACATCAGAAGGCTGTAGACAAATACTGGGACAGCTACAACCATCCCTTCAGACAGGATCAGAAGAACTTAGATCATTATATAATACA\"\"\"\n",
            "\n",
            "description = \"\"\"\n",
            "## 🧬 HIV-1 Subtype Classifier\\n\n",
            "Paste an HIV-1 nucleotide sequence (pol gene region) to predict the subtype. Supports FASTA format or raw sequence.\\n\n",
            "**No alignment required** — works directly on raw nucleotide sequences.\\n\n",
            "**Subtypes:** A, B, C, D\\n\n",
            "**Model:** 1D-CNN with Monte Carlo Dropout uncertainty estimation, trained on LANL HIV Database pol gene sequences using Focal Loss.\n",
            "\"\"\"\n",
            "\n",
            "iface = gr.Interface(\n",
            "    fn=classify_sequence,\n",
            "    inputs=gr.Textbox(label=\"HIV-1 Nucleotide Sequence\", placeholder=\"Paste your sequence here...\", lines=10, value=EXAMPLE_SEQ),\n",
            "    outputs=gr.Label(num_top_classes=NUM_CLASSES, label=\"Predicted Subtype\"),\n",
            "    title=\"HIV-1 Subtype Classifier\",\n",
            "    description=description,\n",
            "    examples=[[EXAMPLE_SEQ]],\n",
            "    theme=gr.themes.Soft(),\n",
            "    flagging_mode=\"never\",\n",
            ")\n",
            "\n",
            "# Launch the app with share=True to generate a public link\n",
            "iface.launch(share=True, debug=True, inline=False)"
        ]
    }
]

notebook_json = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(notebook_json, f, indent=1)

print("Created 06_deployment.ipynb successfully!")
