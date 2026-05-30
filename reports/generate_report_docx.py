"""
generate_report_docx.py — Generate Final Report as Word Document
================================================================
Run: python reports/generate_report_docx.py
Output: reports/Final_Report.docx
"""

import os
from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.style import WD_STYLE_TYPE

# Output path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(SCRIPT_DIR, 'Final_Report.docx')
FIGURES_DIR = os.path.join(SCRIPT_DIR, 'figures')


def set_cell_shading(cell, color_hex):
    """Set background color for a table cell."""
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement
    shading = OxmlElement('w:shd')
    shading.set(qn('w:fill'), color_hex)
    shading.set(qn('w:val'), 'clear')
    cell._tc.get_or_add_tcPr().append(shading)


def create_table(doc, headers, rows, col_widths=None):
    """Create a formatted table."""
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = 'Table Grid'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    # Header row
    for j, header in enumerate(headers):
        cell = table.rows[0].cells[j]
        cell.text = header
        for paragraph in cell.paragraphs:
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in paragraph.runs:
                run.bold = True
                run.font.size = Pt(9)
                run.font.color.rgb = RGBColor(255, 255, 255)
        set_cell_shading(cell, '2E4057')

    # Data rows
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = table.rows[i + 1].cells[j]
            cell.text = str(val)
            for paragraph in cell.paragraphs:
                paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for run in paragraph.runs:
                    run.font.size = Pt(9)
            if i % 2 == 1:
                set_cell_shading(cell, 'F0F4F8')

    return table


def add_heading(doc, text, level):
    """Add a heading with custom formatting."""
    heading = doc.add_heading(text, level=level)
    return heading


def add_body(doc, text):
    """Add body paragraph."""
    p = doc.add_paragraph(text)
    p.style.font.size = Pt(11)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.15
    return p


def add_figure(doc, filename, caption, width=5.5):
    """Add a figure with caption if the file exists."""
    filepath = os.path.join(FIGURES_DIR, filename)
    if os.path.exists(filepath):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run()
        run.add_picture(filepath, width=Inches(width))
        cap = doc.add_paragraph(f'Figure: {caption}')
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap.runs[0].italic = True
        cap.runs[0].font.size = Pt(9)
        cap.runs[0].font.color.rgb = RGBColor(100, 100, 100)
        return True
    return False


def build_report():
    doc = Document()

    # ---- Page setup ----
    for section in doc.sections:
        section.top_margin = Cm(2.5)
        section.bottom_margin = Cm(2.5)
        section.left_margin = Cm(2.5)
        section.right_margin = Cm(2.5)

    # ---- Adjust default styles ----
    style = doc.styles['Normal']
    font = style.font
    font.name = 'Calibri'
    font.size = Pt(11)
    style.paragraph_format.line_spacing = 1.15

    for level in range(1, 4):
        h_style = doc.styles[f'Heading {level}']
        h_style.font.color.rgb = RGBColor(30, 60, 90)

    # ========================================================================
    # TITLE PAGE
    # ========================================================================
    for _ in range(6):
        doc.add_paragraph()

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run('Deep Learning Model for HIV-1 Subtype Classification\nUsing the pol Gene')
    run.bold = True
    run.font.size = Pt(22)
    run.font.color.rgb = RGBColor(30, 60, 90)

    doc.add_paragraph()

    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = subtitle.add_run('MSB7216: Deep Learning for Health Data\nFinal Project Report')
    run.font.size = Pt(14)
    run.font.color.rgb = RGBColor(80, 80, 80)

    doc.add_paragraph()

    author = doc.add_paragraph()
    author.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = author.add_run('Jovita Nagawa\nMay 2026')
    run.font.size = Pt(12)

    doc.add_page_break()

    # ========================================================================
    # 1. ABSTRACT
    # ========================================================================
    add_heading(doc, '1. Abstract', level=1)

    add_body(doc,
        'HIV-1 genetic diversity, organized into subtypes (clades), has direct implications for disease '
        'progression, drug resistance, and vaccine design. Accurate and rapid subtype classification is '
        'essential for epidemiological surveillance and clinical decision-making, particularly in sub-Saharan '
        'Africa where multiple subtypes co-circulate. This project develops and evaluates deep learning models '
        'for classifying HIV-1 sequences into four major subtypes (A, B, C, and D) using raw nucleotide data '
        'from the pol gene region.'
    )
    add_body(doc,
        'Unlike traditional subtyping tools (REGA, COMET, jpHMM) that require computationally expensive multiple '
        'sequence alignment as a prerequisite, and unlike k-mer approaches that require manual feature engineering, '
        'our approach strips alignment gaps from sequences and encodes raw nucleotides as one-hot tensors, training '
        'convolutional and recurrent neural networks to learn discriminative motifs directly from unaligned data. '
        'We address class imbalance using Focal Loss and benchmark three from-scratch architectures: a Multi-Layer '
        'Perceptron (MLP) baseline, a 1D Convolutional Neural Network (CNN), and a Bidirectional Long Short-Term '
        'Memory network (BiLSTM). We also attempt transfer learning with DNABERT-2.'
    )
    add_body(doc,
        'Using 5,564 curated sequences from the Los Alamos National Laboratory (LANL) HIV Sequence Database, '
        'our best model (1D-CNN) achieves 98.44% overall accuracy on the test set, with near-perfect '
        'classification for subtypes A, B, and C, though subtype D classification remains challenging due to '
        'severe class imbalance (only 75 sequences, 1.3% of the dataset). We deploy the trained model as an '
        'interactive web application using Gradio with Monte Carlo Dropout for uncertainty estimation, '
        'automatically flagging low-confidence predictions as indeterminate. All code, data '
        'pipelines, and experiments are publicly available on GitHub for full reproducibility.'
    )

    # ========================================================================
    # 2. INTRODUCTION
    # ========================================================================
    add_heading(doc, '2. Introduction', level=1)

    add_body(doc,
        'Human Immunodeficiency Virus type 1 (HIV-1) is among the most genetically diverse pathogens known, '
        'with its classification into groups, subtypes, and circulating recombinant forms (CRFs) reflecting '
        'distinct evolutionary lineages. The major subtypes (A, B, C, D, F, G, H, J, and K) exhibit geographic '
        'clustering: subtype B predominates in the Americas and Western Europe, subtype C accounts for roughly '
        'half of global infections (concentrated in Southern Africa and India), and subtypes A and D co-circulate '
        'in East Africa, where their interaction is associated with differential clinical outcomes '
        '(Solis-Reyes et al., 2018).'
    )
    add_body(doc,
        'Subtype identification is clinically important for several reasons. First, certain antiretroviral drugs '
        'show differential efficacy across subtypes. Second, subtype information informs vaccine design, as '
        'immune responses may be subtype-specific. Third, accurate subtyping is essential for molecular '
        'epidemiological surveillance — tracking the spread of subtypes across geographic regions and populations.'
    )
    add_body(doc,
        'Traditional subtyping relies on phylogenetic analysis, which requires multiple sequence alignment (MSA) '
        '— a computationally expensive O(n²) to O(n³) process. Tools like REGA, COMET, and jpHMM implement '
        'reference-based approaches but still depend on aligned input sequences. More recently, k-mer frequency '
        'vector methods have been proposed (Solis-Reyes et al., 2018), which avoid alignment but still require '
        'manual feature engineering decisions (e.g., choice of k).'
    )
    add_body(doc,
        'Deep learning offers an attractive alternative: neural networks can learn features directly from raw '
        'sequence data without requiring alignment or manual feature engineering. By stripping alignment gaps and '
        'working with the natural variable-length nucleotide sequences, deep learning models can operate on raw, '
        'unaligned input — eliminating the MSA prerequisite entirely.'
    )

    add_heading(doc, 'Objectives', level=2)
    add_body(doc, 'This project aims to:')
    objectives = [
        'Develop deep learning models that classify HIV-1 pol gene sequences into subtypes A, B, C, and D '
        'directly from raw, unaligned nucleotide sequences — eliminating the multiple sequence alignment prerequisite that all traditional tools require.',
        'Compare the performance of three architectures (MLP, 1D-CNN, BiLSTM) to understand the relative '
        'importance of local versus global sequence features for subtype discrimination.',
        'Analyze classification errors to understand which subtypes are most confusable and why.',
        'Deploy the best model as a web application for real-time sequence classification.'
    ]
    for i, obj in enumerate(objectives, 1):
        doc.add_paragraph(f'{obj}', style='List Number')

    # ========================================================================
    # 3. RELATED WORK
    # ========================================================================
    add_heading(doc, '3. Related Work', level=1)

    add_heading(doc, 'Alignment-Based Subtyping', level=2)
    add_body(doc,
        'The gold standard for HIV-1 subtyping involves phylogenetic analysis against reference sequences. '
        'Tools like REGA v3 (Pineda-Peña et al., 2013) and COMET (Struck et al., 2014) automate this process, '
        'but they require MSA and are computationally expensive for large-scale surveillance.'
    )

    add_heading(doc, 'K-mer Based Methods', level=2)
    add_body(doc,
        'Solis-Reyes et al. (2018) introduced Kameris, an open-source k-mer based machine learning tool that '
        'achieves high accuracy using Euclidean distances between k-mer frequency vectors and k-nearest neighbors '
        'classification. Their approach is fast and does not require alignment, but requires choosing a k-mer '
        'length (they used k=6) and discards positional information.'
    )

    add_heading(doc, 'Deep Learning for Genomic Classification', level=2)
    add_body(doc,
        'CNNs have been successfully applied to DNA sequence classification tasks including transcription factor '
        'binding site prediction (Alipanahi et al., 2015), variant effect prediction (Zhou & Troyanskaya, 2015), '
        'and pathogen classification. For HIV specifically, Fabris et al. (2019) explored deep learning for drug '
        'resistance prediction from pol sequences.'
    )

    add_heading(doc, 'Our Contribution', level=2)
    add_body(doc,
        'This project differs from both traditional alignment-dependent tools and k-mer methods. We strip '
        'alignment gaps from LANL sequences and work with raw, variable-length nucleotide sequences — no MSA '
        'required at inference time. We replace k-mer frequency vectors with direct one-hot encoding, preserving '
        'positional information and allowing the CNN to learn discriminative motifs automatically.'
    )

    # ========================================================================
    # 4. DATASET DESCRIPTION
    # ========================================================================
    add_heading(doc, '4. Dataset Description', level=1)

    add_heading(doc, 'Source', level=2)
    add_body(doc,
        'We use a subset of the dataset described by Solis-Reyes et al. (2018), obtained from the Los Alamos '
        'National Laboratory (LANL) HIV Sequence Database (https://www.hiv.lanl.gov/). LANL is the gold-standard '
        'curated database for HIV genomic research, maintained by the U.S. Department of Energy. This is '
        'explicitly not a Kaggle or Zindi dataset.'
    )

    add_heading(doc, 'Data Acquisition', level=2)
    add_body(doc,
        'We downloaded the exact accession IDs used in the Kameris hiv1-lanl-pol experiment (9,270 accessions) '
        'from the Kameris experiments metadata repository on GitHub. These accession IDs were uploaded to the '
        'LANL search interface to retrieve the corresponding pol coding sequences (CDS) in FASTA format. Of the '
        '9,270 accessions, 9,264 were successfully retrieved; the remaining 6 had been withdrawn or had '
        'incomplete pol CDS records.'
    )

    add_heading(doc, 'Data Characteristics', level=2)
    create_table(doc,
        ['Property', 'Value'],
        [
            ['Total sequences retrieved', '9,264'],
            ['Sequence length', '4,259 bp (aligned to reference)'],
            ['Gene region', 'pol (coding sequence)'],
            ['Subtypes present (raw)', '226 distinct labels'],
            ['After filtering to A, B, C, D', '5,588'],
            ['After removing duplicates', '5,564'],
            ['Final dataset size', '5,564 sequences'],
        ]
    )
    doc.add_paragraph()

    add_heading(doc, 'Subtype Distribution', level=2)
    add_body(doc, 'The final 4-class dataset exhibits significant class imbalance:')
    create_table(doc,
        ['Subtype', 'Count', 'Percentage'],
        [
            ['A', '278', '5.0%'],
            ['B', '2,908', '52.3%'],
            ['C', '2,303', '41.4%'],
            ['D', '75', '1.3%'],
        ]
    )
    doc.add_paragraph()
    add_body(doc,
        'Subtype B is overrepresented because most HIV sequencing historically occurred in North America and '
        'Western Europe. Subtype D is severely underrepresented (only 75 sequences), reflecting both its lower '
        'global prevalence and fewer sequencing efforts.'
    )

    # Add distribution figure if available
    add_figure(doc, 'subtype_distribution.png', 'Subtype distribution in the final modeling dataset', 4.5)

    add_heading(doc, 'Data Splitting', level=2)
    add_body(doc, 'We used stratified splitting to preserve class proportions across all splits:')
    create_table(doc,
        ['Split', 'Total', 'A', 'B', 'C', 'D'],
        [
            ['Train (70%)', '3,896', '194 (5.0%)', '2,037 (52.3%)', '1,612 (41.4%)', '53 (1.4%)'],
            ['Validation (15%)', '833', '42 (5.0%)', '435 (52.2%)', '345 (41.4%)', '11 (1.3%)'],
            ['Test (15%)', '835', '42 (5.0%)', '436 (52.2%)', '346 (41.4%)', '11 (1.3%)'],
        ]
    )

    # ========================================================================
    # 5. METHODOLOGY
    # ========================================================================
    add_heading(doc, '5. Methodology', level=1)

    add_heading(doc, '5.1 Gap Stripping and Sequence Encoding', level=2)
    add_body(doc,
        'The LANL sequences are distributed in pre-aligned form (4,259 bp, padded with gap characters). To '
        'remove the dependency on alignment, we strip all gap characters from each sequence, producing raw '
        'nucleotide sequences of variable length (~2,500–4,200 bp). This is a critical preprocessing step: '
        'traditional tools (REGA, COMET, jpHMM) require aligned input, whereas our model operates directly '
        'on the unaligned sequences.'
    )
    add_body(doc,
        'Each gap-stripped sequence is one-hot encoded as a 2D tensor of shape (4 × L), where L is the '
        'natural sequence length. The four channels correspond to nucleotides A, C, G, and T. A custom collate '
        'function dynamically pads each batch to the length of the longest sequence in that batch.'
    )
    add_body(doc,
        'For the BiLSTM model, we use label encoding (integer indices 0–3 for A, C, G, T, with 4 as padding) '
        'with a learned embedding layer.'
    )

    add_heading(doc, '5.2 Class Imbalance Handling: Focal Loss', level=2)
    add_body(doc,
        'Given the severe imbalance (subtype D has 53 training samples vs. 2,037 for subtype B), we use '
        'Focal Loss (Lin et al., 2017) combined with inverse-frequency class weights as the alpha parameter. '
        'Focal Loss down-weights easy-to-classify examples and focuses training on hard minority cases:'
    )
    create_table(doc,
        ['Class', 'Weight'],
        [['A', '5.02'], ['B', '0.48'], ['C', '0.60'], ['D', '18.38']]
    )
    doc.add_paragraph()

    add_heading(doc, '5.3 Model Architectures', level=2)

    add_heading(doc, 'Transfer Learning: DNABERT', level=3)
    add_body(doc,
        'To demonstrate transfer learning, we fine-tuned DNABERT (Ji et al., 2021), a BERT model pre-trained '
        'on the human reference genome using Masked Language Modeling with 6-mer tokenization. DNA sequences '
        'are converted to overlapping 6-mers (e.g., ATCGAT TCGATC ...) and processed by the standard BERT '
        'architecture (12 layers, 12 attention heads, 768-dim embeddings). We replaced the MLM head with a '
        '4-class classification head and used a two-phase fine-tuning strategy: (1) freeze the pre-trained '
        'encoder and train only the classification head for 3 epochs, then (2) unfreeze all layers and '
        'fine-tune end-to-end with differential learning rates (2×10⁻⁵ for the encoder, 1×10⁻⁴ for the head).'
    )

    add_heading(doc, 'K-mer Feature Engineering', level=3)
    add_body(doc,
        'As an alternative feature engineering approach inspired by Kameris (Solis-Reyes et al., 2018), we '
        'extracted 6-mer frequency vectors from each sequence. Each sequence is represented as a 4,096-dimensional '
        'vector (4^6 possible 6-mers), where each element is the normalized frequency of that k-mer in the '
        'sequence. This discards positional information but captures overall sequence composition. A feedforward '
        'neural network (4096 → 512 → 256 → 128 → 4) was trained on these features.'
    )

    add_heading(doc, 'Baseline: Multi-Layer Perceptron (MLP)', level=3)
    add_body(doc,
        'The MLP serves as a non-convolutional baseline. It first applies adaptive average pooling to reduce '
        'the variable-length sequence to 64 summary positions, then flattens to a 256-dimensional feature '
        'vector (4 channels × 64 positions). Two fully-connected hidden layers (512 → 256) with batch '
        'normalization, ReLU activation, and 40% dropout produce the final 4-class logits. Total parameters: '
        '196,868.'
    )

    add_heading(doc, 'Primary Model: 1D Convolutional Neural Network (CNN)', level=3)
    add_body(doc,
        'The 1D-CNN consists of four convolutional blocks with increasing receptive fields:'
    )
    create_table(doc,
        ['Block', 'Filters', 'Kernel Size', 'Pooling', 'Purpose'],
        [
            ['1', '64', '7', 'MaxPool(4)', 'Short motifs (7-mers)'],
            ['2', '128', '5', 'MaxPool(4)', 'Medium motifs'],
            ['3', '256', '3', 'MaxPool(4)', 'Larger patterns'],
            ['4', '256', '3', 'GlobalAvgPool', 'High-level features'],
        ]
    )
    doc.add_paragraph()
    add_body(doc,
        'Each block includes batch normalization, ReLU activation, and 30% dropout. The global average pooled '
        'features (256-dimensional) pass through a classifier head (256 → 128 → 4). Total parameters: 373,636.'
    )

    add_heading(doc, 'Secondary Model: Bidirectional LSTM (BiLSTM)', level=3)
    add_body(doc,
        'The BiLSTM uses a learned embedding layer (vocabulary size 5: A, C, G, T, padding; embedding dimension 128) followed by a '
        '2-layer bidirectional LSTM with hidden dimension 256. The final hidden states from both directions are '
        'concatenated (512-dimensional) and passed through a classifier head (512 → 128 → 4). Total parameters: '
        '2,434,436.'
    )

    add_heading(doc, '5.4 Training Configuration', level=2)
    create_table(doc,
        ['Hyperparameter', 'MLP', 'CNN', 'BiLSTM'],
        [
            ['Optimizer', 'AdamW', 'AdamW', 'AdamW'],
            ['Learning rate', '3×10⁻³', '1×10⁻³', '5×10⁻⁴'],
            ['Weight decay', '1×10⁻⁴', '1×10⁻⁴', '1×10⁻⁴'],
            ['LR scheduler', 'CosineAnnealing', 'CosineAnnealing', 'CosineAnnealing'],
            ['Max epochs', '5', '10', '30'],
            ['Early stopping', '—', 'Patience 5', 'Patience 5'],
            ['Batch size', '32', '32', '32'],
            ['Gradient clipping', '1.0', '1.0', '1.0'],
            ['Random seed', '42', '42', '42'],
        ]
    )

    # ========================================================================
    # 6. EXPERIMENTS
    # ========================================================================
    add_heading(doc, '6. Experiments', level=1)

    add_heading(doc, 'Experiment 1: MLP Baseline', level=2)
    add_body(doc,
        'The MLP was trained for 5 epochs. Validation loss decreased from 1.30 to 0.72 across epochs, with '
        'validation accuracy reaching 79.83% at epoch 4. The model exhibited noticeable instability in validation '
        'metrics, with a spike to 3.94 validation loss at epoch 2, suggesting the pooled features lack sufficient '
        'discriminative power. Final best validation loss: 0.7214.'
    )

    add_heading(doc, 'Experiment 2: 1D-CNN Training', level=2)
    add_body(doc,
        'The CNN was trained for up to 10 epochs with early stopping (patience 5). The model converged rapidly '
        '— achieving 97.74% training accuracy by epoch 2 and best validation loss of 0.3531 at epoch 2. '
        'Training was halted at epoch 7 by early stopping. The rapid convergence suggests the convolutional '
        'filters quickly learn discriminative nucleotide motifs.'
    )

    add_heading(doc, 'Experiment 3: BiLSTM Training', level=2)
    add_body(doc,
        'The BiLSTM was trained for up to 30 epochs with early stopping (patience 5). It converged more slowly '
        'than the CNN, reaching best validation loss of 0.5059 at epoch 12. Early stopping triggered at epoch '
        '17. The slower convergence and higher parameter count (2.43M vs. 374K for CNN) suggest recurrent '
        'processing of variable-length sequences is less efficient than local convolutional feature extraction '
        'for this task.'
    )

    # Add training curves figure
    add_figure(doc, 'model_comparison_curves.png', 'Training and validation curves for CNN and BiLSTM models', 5.5)

    # ========================================================================
    # 7. RESULTS
    # ========================================================================
    add_heading(doc, '7. Results', level=1)

    add_heading(doc, 'Overall Performance Comparison', level=2)
    create_table(doc,
        ['Model', 'Accuracy', 'Macro F1', 'Macro Precision', 'Macro Recall'],
        [
            ['MLP Baseline', '72.81%', '0.6095', '0.6025', '0.7410'],
            ['1D-CNN', '98.44%', '0.7421', '0.7409', '0.7435'],
            ['BiLSTM', '93.53%', '0.6468', '0.6173', '0.7035'],
            ['DNABERT (transfer learning)', '41.44%', '0.1036', '0.2500', '0.1465']
        ]
    )
    doc.add_paragraph()
    add_body(doc,
        'The 1D-CNN achieves the best overall accuracy (98.44%), representing a 25.63 percentage point '
        'improvement over the MLP baseline and a 4.91 percentage point improvement over the BiLSTM.'
    )
    add_body(doc,
        'DNABERT Transfer Learning Limitations: We attempted to fine-tune DNABERT (a pre-trained '
        'genomic language model) for subtype classification. However, DNABERT relies on 6-mer tokenization and limits input lengths to 512 tokens. '
        'With pol gene sequences ranging from 2,500–4,200 bp, only the first ~517 bp of each sequence could be utilized. '
        'Consequently, the model achieved only 41.44% accuracy (and effectively 0 F1-score on minority classes), '
        'demonstrating that extreme sequence truncation discards the critical distal motifs necessary for accurate '
        'HIV-1 subtype discrimination. This highlights the necessity for our custom CNN architecture that can process '
        'full-length sequences via global average pooling.',
        bold=True
    )

    add_heading(doc, 'Per-Class Performance (1D-CNN)', level=2)
    create_table(doc,
        ['Subtype', 'Precision', 'Recall', 'F1-Score', 'Support'],
        [
            ['A', '1.0000', '0.9762', '0.9880', '42'],
            ['B', '1.0000', '0.9977', '0.9989', '436'],
            ['C', '0.9638', '1.0000', '0.9816', '346'],
            ['D', '0.0000', '0.0000', '0.0000', '11'],
        ]
    )
    doc.add_paragraph()
    add_body(doc,
        'The CNN achieves near-perfect classification for subtypes A, B, and C, but completely fails on subtype '
        'D (zero precision, recall, and F1). This is directly attributable to the extreme class imbalance — with '
        'only 53 training samples for subtype D, the model cannot learn sufficiently robust features despite the '
        '18.38× class weight.'
    )

    # Add confusion matrix and ROC figures
    add_figure(doc, 'all_confusion_matrices.png', 'Confusion matrices for all three models (counts and normalized)', 5.5)
    add_figure(doc, 'best_model_roc.png', 'ROC curves for the best model (1D-CNN) with per-class AUC scores', 4.5)

    add_heading(doc, 'Key Observations', level=2)
    observations = [
        'CNN dominates: The 98.44% accuracy demonstrates that local convolutional motifs are highly '
        'discriminative for HIV-1 subtype classification, consistent with the biology — subtype-defining '
        'mutations tend to cluster in specific pol gene regions.',
        'BiLSTM underperforms CNN: Despite modeling long-range dependencies with 6.5× more parameters, the '
        'BiLSTM (93.53%) cannot match the CNN. This suggests subtype-discriminative signal is primarily local '
        '(motif-level) rather than dependent on long-range sequence context.',
        'Subtype D is a failure case: All three models struggle with subtype D. The MLP actually performs best '
        'on D (F1=0.52) because its simpler decision boundaries avoid overfitting to the majority classes.',
        'Macro F1 vs. Accuracy discrepancy: The CNN\'s macro F1 (0.7421) is much lower than its accuracy '
        '(98.44%) because subtype D\'s zero F1 drags down the macro average. The weighted F1 (0.9780) more '
        'accurately reflects practical performance.'
    ]
    for obs in observations:
        doc.add_paragraph(obs, style='List Bullet')

    # ========================================================================
    # 8. ERROR ANALYSIS
    # ========================================================================
    add_heading(doc, '8. Error Analysis', level=1)

    add_body(doc,
        'The 1D-CNN misclassified only 13 out of 835 test sequences (1.6% error rate). Detailed error analysis '
        'reveals the following misclassification patterns:'
    )

    create_table(doc,
        ['True Label', 'Predicted Label', 'Count'],
        [
            ['D', 'C', '11'],
            ['A', 'C', '1'],
            ['B', 'C', '1'],
        ]
    )
    doc.add_paragraph()

    add_heading(doc, 'Key Findings', level=2)
    findings = [
        'D → C confusion dominates: All 11 subtype D test sequences were misclassified as subtype C. This is '
        'biologically plausible: subtypes C and D share significant phylogenetic similarity in the pol gene, '
        'particularly in the reverse transcriptase and integrase regions.',
        'High-confidence errors: The mean prediction confidence for misclassified sequences was 0.709 (median '
        '0.786), indicating these are not low-confidence borderline cases but systematic misclassifications.',
        'Geographic confound: Subtypes C and D co-circulate in East Africa. Some sequences labeled as subtype D '
        'may contain C-like pol regions due to recombination events not captured by pure subtype labels.',
        'Sample size is the bottleneck: With only 53 training samples for subtype D, the model has insufficient '
        'examples to learn D-specific motifs that distinguish it from the closely related subtype C (1,612 '
        'training samples — a 30:1 ratio).'
    ]
    for finding in findings:
        doc.add_paragraph(finding, style='List Bullet')

    # ========================================================================
    # 9. ETHICAL CONSIDERATIONS
    # ========================================================================
    add_heading(doc, '9. Ethical Considerations', level=1)

    add_heading(doc, 'Data Privacy', level=2)
    add_body(doc,
        'All sequences used in this project are publicly available from the LANL HIV Database. Sequences are '
        'identified by GenBank accession numbers only — no patient names, clinical data, or personally '
        'identifiable information is associated with the sequences.'
    )

    add_heading(doc, 'Geographic and Sampling Bias', level=2)
    add_body(doc,
        'The dataset reflects historical biases in HIV sequencing. Subtype B is overrepresented (52.3%) because '
        'most sequencing has been conducted in North America and Western Europe. Subtypes A and D are severely '
        'underrepresented (5.0% and 1.3%) despite their clinical importance in East Africa. This bias directly '
        'impacts model performance: the model achieves perfect classification for well-sampled subtypes but '
        'completely fails on the rare subtype D.'
    )

    add_heading(doc, 'Fairness and Deployment Risks', level=2)
    add_body(doc,
        'Deploying this model in East Africa — where subtype D classification matters most — would produce '
        'misleading results. The Gradio deployment interface includes an explicit disclaimer that this is a '
        'research/educational tool and should not be used for clinical subtype determination. Clinical decisions '
        'should rely on validated tools (REGA, COMET, jpHMM) with established regulatory approval.'
    )

    add_heading(doc, 'Broader Impact', level=2)
    add_body(doc,
        'If expanded and validated, deep learning subtyping that operates on raw, unaligned sequences could '
        'enable rapid, decentralized molecular surveillance without transmitting sensitive genomic data to '
        'remote servers — an important consideration for data sovereignty in resource-limited settings.'
    )

    # ========================================================================
    # 10. LIMITATIONS
    # ========================================================================
    add_heading(doc, '10. Limitations', level=1)

    limitations = [
        'Severe class imbalance: The 30:1 ratio between subtype B (2,037 training samples) and subtype D (53 '
        'training samples) prevents effective learning of D-specific features. Class-weighted loss mitigates but '
        'does not solve this problem.',
        'Only 4 subtypes: HIV-1 has 9+ subtypes and numerous CRFs. Our model cannot classify subtypes F, G, H, '
        'J, K, or any recombinant forms.',
        'Gap-stripping as proxy for unaligned data: While we strip alignment gaps to produce variable-length '
        'sequences, the underlying data was originally downloaded from LANL in pre-aligned form. True unaligned sequences obtained '
        'directly from sequencing instruments may contain additional artifacts (e.g., varying start/end positions) not '
        'present in our gap-stripped data.',
        'Transformer context limits: Fine-tuning DNABERT failed to generalize (41.44% accuracy) because its 512-token limit '
        'required severe truncation of the ~2,500–4,200 bp pol sequences. This indicates that HIV-1 subtype features '
        'are distributed throughout the gene, not localized to the first 500 bp.',
        'No explainability: We do not implement gradient-based visualization (e.g., Grad-CAM) or attention '
        'mechanisms to identify which nucleotide positions drive classification decisions. This limits biological '
        'interpretability.',
        'Geographical bias: The LANL database, and consequently our dataset, is heavily biased toward Subtype B '
        'due to historically disproportionate sequencing efforts in North America and Europe. True global prevalence '
        'is not reflected in the training data distribution.'
    ]
    for lim in limitations:
        doc.add_paragraph(lim, style='List Number')

    # ========================================================================
    # 11. FUTURE WORK
    # ========================================================================
    add_heading(doc, '11. Future Work', level=1)

    future = [
        'Data augmentation for rare subtypes: Synthetic sequence generation (e.g., using evolutionary models '
        'or GANs) could increase subtype D training data.',
        'Multi-gene classification: Extending the model to accept concatenated sequences from gag, pol, and '
        'env genes simultaneously, which would improve classification robustness and allow for recombinant detection.',
        'Explainability: Implementing Grad-CAM or learned attention mechanisms to identify which nucleotide '
        'positions are most discriminative for each subtype. This could reveal biologically meaningful motifs.',
        'Long-context transformers: Utilizing long-context transformer architectures (like DNABERT-2 with FlashAttention '
        'or HyenaDNA) to process full 4,000+ bp sequences, potentially improving performance on rare subtypes through pre-trained genomic representations.',
        'True unaligned evaluation: Testing the model on sequences obtained directly from sequencing pipelines '
        '(e.g., consensus sequences from NGS) without any prior alignment to a reference.',
        'Clinical validation: Testing the model on prospective clinical samples, particularly from East African '
        'settings where subtype classification has direct treatment implications.',
    ]
    for item in future:
        doc.add_paragraph(item, style='List Number')

    # ========================================================================
    # 12. CONCLUSION
    # ========================================================================
    add_heading(doc, '12. Conclusion', level=1)

    add_body(doc,
        'This project demonstrates that deep learning can effectively classify HIV-1 pol gene sequences into '
        'major subtypes directly from raw, unaligned nucleotide data — without requiring the multiple sequence alignment that all traditional tools depend on. By stripping alignment gaps and using dynamic per-batch padding, our '
        '1D-CNN achieves 98.44% overall accuracy — a substantial improvement over both the MLP baseline (72.81%) '
        'and the BiLSTM (93.53%). The CNN\'s dominance suggests that HIV-1 subtype discrimination is primarily '
        'driven by local nucleotide motifs rather than long-range sequence dependencies, which is consistent '
        'with the known biology of subtype-defining mutations in the pol gene.'
    )
    add_body(doc,
        'However, the complete failure on subtype D (0% F1) highlights a critical limitation: deep learning '
        'cannot overcome fundamental data scarcity. With only 53 training examples for subtype D, even aggressive '
        'class weighting is insufficient. This underscores the importance of equitable genomic sampling — the '
        'same communities most affected by HIV-1 subtype D (East Africa) are those least represented in sequence '
        'databases.'
    )
    add_body(doc,
        'The deployed Gradio web application incorporates Monte Carlo Dropout for uncertainty estimation, '
        'automatically flagging low-confidence predictions as indeterminate rather than forcing a potentially '
        'incorrect classification. Combined with explicit disclaimers, this ensures responsible use as a '
        'research tool rather than a clinical diagnostic.'
    )

    # ========================================================================
    # 13. REFERENCES
    # ========================================================================
    add_heading(doc, '13. References', level=1)

    references = [
        'Solis-Reyes, S., Avino, M., Poon, A., & Kari, L. (2018). An open-source k-mer based machine learning '
        'tool for fast and accurate subtyping of HIV-1 genomes. PLoS ONE, 13(11), e0206409.',
        'Los Alamos National Laboratory HIV Sequence Database. https://www.hiv.lanl.gov/',
        'Kameris Experiments Repository. https://github.com/stephensolis/kameris-experiments',
        'Pineda-Peña, A.C., et al. (2013). Automated subtyping of HIV-1 genetic sequences for clinical and '
        'surveillance purposes: performance evaluation of the new REGA version 3. Infection, Genetics and '
        'Evolution, 19, 337-348.',
        'Struck, D., et al. (2014). COMET: adaptive context-based modeling for ultrafast HIV-1 subtype '
        'identification. Nucleic Acids Research, 42(18), e144.',
        'Alipanahi, B., Delong, A., Weirauch, M.T., & Frey, B.J. (2015). Predicting the sequence specificities '
        'of DNA- and RNA-binding proteins by deep learning. Nature Biotechnology, 33(8), 831-838.',
        'Zhou, J., & Troyanskaya, O.G. (2015). Predicting effects of noncoding variants with deep learning-based '
        'sequence model. Nature Methods, 12(10), 931-934.',
        'Paszke, A., et al. (2019). PyTorch: An imperative style, high-performance deep learning library. '
        'NeurIPS, 8024-8035.',
        'Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural Computation, 9(8), 1735-1780.',
        'LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. Nature, 521(7553), 436-444.',
    ]
    for ref in references:
        doc.add_paragraph(ref, style='List Number')

    # ---- Save ----
    doc.save(OUTPUT_PATH)
    print(f'Report saved to: {OUTPUT_PATH}')


if __name__ == '__main__':
    build_report()
