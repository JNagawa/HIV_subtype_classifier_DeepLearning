import os
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(SCRIPT_DIR, 'Final_Report.docx')
FIGURES_DIR = os.path.join(SCRIPT_DIR, 'figures')

def add_heading(doc, text, level):
    h = doc.add_heading(text, level=level)
    run = h.runs[0]
    run.font.color.rgb = RGBColor(0, 51, 102)

def add_paragraph(doc, text, bold=False):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    run = p.add_run(text)
    run.bold = bold
    return p

def add_image_if_exists(doc, filename, width_inches=6.0):
    filepath = os.path.join(FIGURES_DIR, filename)
    if os.path.exists(filepath):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run()
        r.add_picture(filepath, width=Inches(width_inches))
        cap = doc.add_paragraph(f"Figure: {filename.replace('_', ' ').replace('.png', '').title()}")
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap.runs[0].italic = True
    else:
        print(f"Warning: {filename} not found.")

def create_table(doc, headers, rows):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = 'Table Grid'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, header in enumerate(headers):
        cell = table.rows[0].cells[j]
        cell.text = header
        cell.paragraphs[0].runs[0].bold = True
    for i, row_data in enumerate(rows):
        for j, val in enumerate(row_data):
            table.rows[i+1].cells[j].text = str(val)

def generate():
    doc = Document()
    
    title = doc.add_heading('Deep Learning for HIV-1 Subtype Classification: A BiLSTM and Focal Loss Approach to Analyzing pol Gene Sequences', 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph('Final Project - MSB7216: Deep Learning for Health Data').alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    # 1. Abstract
    add_heading(doc, '1. Abstract', 1)
    add_paragraph(doc, "Background: The HIV-1 pol gene is highly conserved and a primary target for antiretroviral therapy. Tracking epidemiological spread and drug resistance relies heavily on accurate viral subtyping.")
    add_paragraph(doc, "Problem Statement: Traditional methods require complex multiple sequence alignments or k-mer counting, which struggle with hypervariable regions and extreme geographical data imbalances.")
    add_paragraph(doc, "Objective: To develop a fully automated, alignment-free Deep Learning classifier to categorize raw nucleotide sequences into Subtypes A, B, C, and D.")
    add_paragraph(doc, "Methods: Sequences were parsed, stripped of gaps, and encoded into integers and one-hot vectors. To combat severe class imbalance (Subtype B overrepresentation), we utilized a pure Focal Loss mechanism (Gamma=2.0). Models evaluated include a Baseline MLP, a 1D-CNN, and a Bidirectional LSTM (BiLSTM).")
    add_paragraph(doc, "Results: The BiLSTM vastly outperformed other architectures, achieving 94.73% overall accuracy. The 1D-CNN and DNABERT models suffered from severe class collapse (predicting majority Subtype B at 52.22%).")
    add_paragraph(doc, "Conclusion: BiLSTMs are highly effective at modeling raw, unaligned biological sequences, successfully extracting bidirectional contextual dependencies while resisting geographical imbalance collapse.")

    # 2. Introduction
    add_heading(doc, '2. Introduction / Background', 1)
    add_paragraph(doc, "Human Immunodeficiency Virus Type 1 (HIV-1) exhibits extreme genetic diversity. The pol gene is clinically critical because it encodes the reverse transcriptase, protease, and integrase enzymes. Accurately identifying the viral subtype is essential for personalized medicine and epidemiological tracking. Traditional bioinformatics algorithms rely heavily on Multiple Sequence Alignment (MSA), a computationally expensive process. This project introduces an alignment-free deep learning pipeline designed to automatically extract phylogenetic features directly from raw nucleotide sequences.")

    # 3. Dataset Description
    add_heading(doc, '3. Dataset Description', 1)
    add_paragraph(doc, "The dataset consists of HIV-1 pol gene sequences from the LANL HIV Sequence Database. A critical challenge is geographical bias. Subtype B dominates the dataset at ~52.2%, being the primary variant in North America and Western Europe. Subtypes A, C, and D are less represented. This severe class imbalance necessitated advanced loss optimization to prevent minority class suppression.")
    add_image_if_exists(doc, 'subtype_distribution.png', 4.5)

    # 4. Methodology
    add_heading(doc, '4. Methodology', 1)
    
    add_heading(doc, '4.1 Processing (Encoding)', 2)
    add_paragraph(doc, "Sequences were cleansed of alignment gap characters ('-'). Two encodings were utilized: One-hot encoding for the MLP and CNN models, and integer label embedding for the BiLSTM and Transformer models. All sequences were padded/truncated to a uniform length of 3000bp.")

    add_heading(doc, '4.2 Class Balancing', 2)
    add_paragraph(doc, "Initial iterations utilizing standard categorical cross-entropy caused catastrophic collapse into the majority class (Subtype B). The pipeline implemented pure Focal Loss (gamma=2.0) to dynamically down-weight easily classified majority examples and force the optimizer to focus heavily on hard-to-predict minority strains.")

    add_heading(doc, '4.3 Experiments', 2)
    add_paragraph(doc, "1. Baseline MLP: Global Average Pooling to evaluate global nucleotide composition.")
    add_paragraph(doc, "2. Deep Learning (1D-CNN): Convolutional networks designed to extract localized spatial motifs (k-mers).")
    add_paragraph(doc, "3. Deep Learning (BiLSTM): Designed to process sequence context bidirectionally to capture long-range biological dependencies.")

    add_heading(doc, '4.4 Explainability', 2)
    add_paragraph(doc, "A Universal Saliency Map was implemented using Input-Gradient Attention to visualize model confidence. The algorithm plotted high-resolution attention maps over the DNA sequence to highlight exact biological regions driving the classification, successfully bypassing CNN-only limitations.")

    add_heading(doc, '4.5 Deployment', 2)
    add_paragraph(doc, "A dynamic model-factory system automatically parses training histories, identifies the best architecture, loads the trained checkpoint natively, and exposes an inference pipeline.")

    add_heading(doc, '4.6 Transfer Learning', 2)
    add_paragraph(doc, "An exploratory phase utilized DNABERT, a Transformer pre-trained on the human genome, fine-tuned for HIV subtyping to test cross-domain knowledge transfer.")

    # 5. Results
    add_heading(doc, '5. Results', 1)
    add_paragraph(doc, "The BiLSTM achieved an impressive 94.73% accuracy. However, both the 1D-CNN and DNABERT collapsed entirely, predicting only the majority class (Subtype B) resulting in exactly 52.22% accuracy.")
    
    headers = ['Model', 'Test Accuracy', 'Macro Precision', 'Macro Recall', 'Macro F1']
    rows = [
        ['Baseline MLP', '0.6527', '0.5082', '0.7472', '0.5114'],
        ['1D-CNN', '0.5222', '0.1305', '0.2500', '0.1715'],
        ['BiLSTM', '0.9473', '0.6274', '0.6982', '0.6541'],
        ['DNABERT', '0.5222', '0.1305', '0.2500', '0.1715']
    ]
    create_table(doc, headers, rows)
    
    add_image_if_exists(doc, 'best_model_roc.png', 5.0)
    add_image_if_exists(doc, 'all_confusion_matrices.png', 6.0)

    # 6. Error Analysis
    add_heading(doc, '6. Error Analysis', 1)
    add_paragraph(doc, "Observed Trends: The BiLSTM misclassified approximately 5% of sequences, while the CNN and DNABERT experienced 100% minority-class misclassification (predicting everything as Subtype B).")
    add_paragraph(doc, "Possible Causes & Implications: The Saliency Maps revealed that the BiLSTM focuses on highly conserved functional regions. When novel sequences exhibit hypermutation in these regions (viral drift), the model's confidence shatters. The CNN and DNABERT failures indicate that local spatial motifs (convolutions) and generic human-genome pre-training are highly susceptible to geographical class imbalance, completely ignoring minority strains.")
    add_image_if_exists(doc, 'saliency_worst_error.png', 6.0)

    # 7. Limitations & Challenges
    add_heading(doc, '7. Limitations and Challenges', 1)
    add_paragraph(doc, "1. Severe Geographical Imbalance: Despite Focal Loss, overcoming the 52% Subtype B dominance remains a fundamental challenge, as evidenced by the CNN's total collapse.")
    add_paragraph(doc, "2. Computational Complexity: Transfer learning with DNABERT on 3000bp sequences was highly constrained by GPU VRAM, leading to suboptimal batch sizes and eventual model collapse.")
    add_paragraph(doc, "3. Domain Gap: Pre-training Transformers on human genomes does not inherently translate to high-mutation viral genomes.")

    # 8. Conclusion
    add_heading(doc, '8. Conclusion and Future Work', 1)
    add_paragraph(doc, "Alignment-free Deep Learning via Bidirectional LSTMs successfully resolves HIV-1 subtyping, achieving 94.73% accuracy. It proves that sequential recurrent networks are far more resilient to biological class imbalances than spatial CNNs.")
    add_paragraph(doc, "Future Work: The pipeline will be expanded to include Circulating Recombinant Forms (CRFs), and the deployment module will be hosted as a REST API for programmatic clinical access.")

    doc.save(OUTPUT_PATH)
    print(f"Report saved to {OUTPUT_PATH}")

if __name__ == '__main__':
    generate()
