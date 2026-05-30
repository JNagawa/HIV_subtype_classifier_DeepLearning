import json
import os

workspace = r"c:\Users\jnagawa\Downloads\hiv-subtype-classifier-20260529T180143Z-3-001\hiv-subtype-classifier"
nb05_path = os.path.join(workspace, "notebooks", "05_evaluation_and_analysis.ipynb")

with open(nb05_path, "r", encoding="utf-8") as f:
    nb5 = json.load(f)

# Find cell indices
evaluate_idx = -1
cm_idx = -1
for i, cell in enumerate(nb5['cells']):
    if cell.get('metadata', {}).get('id') == 'evaluate_all':
        evaluate_idx = i
    if cell.get('metadata', {}).get('id') == 'confusion_all':
        cm_idx = i

bert_setup_code = """# ===== DNABERT Setup =====
from transformers import BertModel, AutoTokenizer
from torch.utils.data import Dataset, DataLoader

MODEL_NAME = "damlab/HIV_BERT"
MAX_TOKENS = 512
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def seq_to_kmer_string(sequence, k=6):
    seq = sequence.upper().replace('-', 'N')
    kmers = [seq[i:i+k] for i in range(len(seq) - k + 1)]
    return ' '.join(kmers)

class DNABERTDataset(Dataset):
    def __init__(self, sequences, labels, tokenizer, max_length=MAX_TOKENS, k=6):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.k = k
    def __len__(self): return len(self.sequences)
    def __getitem__(self, idx):
        kmer_str = seq_to_kmer_string(self.sequences[idx], self.k)
        encoding = self.tokenizer(kmer_str, truncation=True, max_length=self.max_length, padding='max_length', return_tensors='pt')
        return {'input_ids': encoding['input_ids'].squeeze(0), 'attention_mask': encoding['attention_mask'].squeeze(0), 'labels': torch.tensor(self.labels[idx], dtype=torch.long)}

dnabert_test_loader = DataLoader(DNABERTDataset(X_test, y_test, tokenizer), batch_size=16, shuffle=False, num_workers=2, pin_memory=True)

class HIVBERTClassifier(nn.Module):
    def __init__(self, model_name, num_classes, dropout=0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        self.classifier = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_size, num_classes))
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_output)

dnabert_model = HIVBERTClassifier(MODEL_NAME, num_classes).to(device)
try:
    dnabert_model.load_state_dict(torch.load(os.path.join(models_dir, 'dnabert_hiv_best.pth'), map_location=device))
    print("DNABERT loaded successfully!")
except Exception as e:
    print("Could not load DNABERT:", e)"""

new_evaluate_all_code = """# Prediction helper
def predict(model, dl, is_bert=False):
    model.eval()
    preds, labels, probas = [], [], []
    with torch.no_grad():
        for batch in tqdm(dl, leave=False):
            if is_bert:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                lbl = batch['labels'].numpy()
                out = model(input_ids, attention_mask)
            else:
                inp, lbl = batch
                out = model(inp.to(device))
                lbl = lbl.numpy()
            
            p = torch.softmax(out, 1)
            preds.extend(out.argmax(1).cpu().numpy())
            labels.extend(lbl)
            probas.extend(p.cpu().numpy())
    return np.array(labels), np.array(preds), np.array(probas)

# Evaluate all models
results = {}
for name, model, dl, is_bert in [
    ('MLP Baseline', mlp, test_dl_oh, False),
    ('1D-CNN', cnn, test_dl_oh, False),
    ('BiLSTM', bilstm, test_dl_lbl, False),
    ('DNABERT', dnabert_model, dnabert_test_loader, True)
]:
    yt, yp, yprob = predict(model, dl, is_bert=is_bert)
    acc = accuracy_score(yt, yp)
    p, r, f1, _ = precision_recall_fscore_support(yt, yp, average='macro', zero_division=0)
    results[name] = {'acc': acc, 'f1': f1, 'prec': p, 'rec': r,
                     'yt': yt, 'yp': yp, 'yprob': yprob}
    print(f'\\n{name}: Acc={acc:.4f}, Macro F1={f1:.4f}')
    print(classification_report(yt, yp, target_names=class_names, digits=4, zero_division=0))"""

new_cm_code = """# Confusion matrices for all models
n_models = len(results)
fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
# Make axes iterable if n_models == 1
if n_models == 1: axes = [axes]
for ax, (name, res) in zip(axes, results.items()):
    cm = confusion_matrix(res['yt'], res['yp'])
    # Avoid div by zero in normalized CM
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_n = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_n = np.nan_to_num(cm_n)
    sns.heatmap(cm_n, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title(name, fontsize=13)
    ax.set_ylabel('True'); ax.set_xlabel('Predicted')

plt.suptitle('Normalized Confusion Matrices — All Models', fontsize=15)
plt.tight_layout()
plt.savefig(f'{reports}/all_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.show()"""

if cm_idx != -1:
    nb5['cells'][cm_idx]['source'] = [line + "\n" for line in new_cm_code.split("\n")][:-1]

if evaluate_idx != -1:
    bert_setup_cell = {
        "cell_type": "code",
        "metadata": {"id": "bert_setup"},
        "execution_count": None,
        "outputs": [],
        "source": [line + "\n" for line in bert_setup_code.split("\n")][:-1]
    }
    nb5['cells'].insert(evaluate_idx, bert_setup_cell)
    # The index shifts by 1 because we inserted before it
    nb5['cells'][evaluate_idx+1]['source'] = [line + "\n" for line in new_evaluate_all_code.split("\n")][:-1]

# Insert !pip install transformers -q if not present
has_pip = any("!pip install transformers" in "".join(c.get("source", [])) for c in nb5['cells'])
if not has_pip:
    pip_cell = {
        "cell_type": "code",
        "metadata": {"id": "pip_transformers"},
        "execution_count": None,
        "outputs": [],
        "source": ["!pip install transformers -q\n"]
    }
    nb5['cells'].insert(1, pip_cell)

with open(nb05_path, "w", encoding="utf-8") as f:
    json.dump(nb5, f, indent=1)

print("DNABERT model added to Notebook 5 evaluation!")
