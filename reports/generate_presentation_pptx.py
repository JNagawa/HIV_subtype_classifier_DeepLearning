"""
generate_presentation_pptx.py — Generate Presentation as PowerPoint
====================================================================
Run: python reports/generate_presentation_pptx.py
Output: reports/presentation.pptx
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(SCRIPT_DIR, 'presentation.pptx')
FIGURES_DIR = os.path.join(SCRIPT_DIR, 'figures')

# Color palette
BG_DARK = RGBColor(15, 15, 26)
BG_CARD = RGBColor(30, 41, 59)
BLUE = RGBColor(96, 165, 250)
LIGHT_BLUE = RGBColor(147, 197, 253)
PURPLE = RGBColor(167, 139, 250)
GREEN = RGBColor(52, 211, 153)
ORANGE = RGBColor(251, 191, 36)
RED = RGBColor(248, 113, 113)
WHITE = RGBColor(224, 224, 224)
GRAY = RGBColor(148, 163, 184)
DARK_TEXT = RGBColor(30, 41, 59)


def set_slide_bg(slide, color=BG_DARK):
    """Set slide background color."""
    background = slide.background
    fill = background.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_textbox(slide, left, top, width, height, text, font_size=18,
                color=WHITE, bold=False, alignment=PP_ALIGN.LEFT, font_name='Calibri'):
    """Add a text box to a slide."""
    txBox = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.color.rgb = color
    p.font.bold = bold
    p.font.name = font_name
    p.alignment = alignment
    return txBox, tf


def add_bullet_frame(slide, left, top, width, height, items, font_size=16,
                     color=WHITE, bullet_color=BLUE):
    """Add a bulleted list."""
    txBox = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = txBox.text_frame
    tf.word_wrap = True

    for i, item in enumerate(items):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.text = f"▸ {item}"
        p.font.size = Pt(font_size)
        p.font.color.rgb = color
        p.font.name = 'Calibri'
        p.space_after = Pt(6)

    return txBox, tf


def add_table(slide, left, top, width, height, headers, rows):
    """Add a formatted table to a slide."""
    n_rows = len(rows) + 1
    n_cols = len(headers)
    table_shape = slide.shapes.add_table(n_rows, n_cols, Inches(left), Inches(top),
                                          Inches(width), Inches(height))
    table = table_shape.table

    # Header row
    for j, header in enumerate(headers):
        cell = table.cell(0, j)
        cell.text = header
        for paragraph in cell.text_frame.paragraphs:
            paragraph.font.size = Pt(12)
            paragraph.font.bold = True
            paragraph.font.color.rgb = WHITE
            paragraph.font.name = 'Calibri'
            paragraph.alignment = PP_ALIGN.CENTER
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(46, 64, 87)

    # Data rows
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = table.cell(i + 1, j)
            cell.text = str(val)
            for paragraph in cell.text_frame.paragraphs:
                paragraph.font.size = Pt(11)
                paragraph.font.color.rgb = DARK_TEXT
                paragraph.font.name = 'Calibri'
                paragraph.alignment = PP_ALIGN.CENTER
            if i % 2 == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = RGBColor(241, 245, 249)
            else:
                cell.fill.solid()
                cell.fill.fore_color.rgb = RGBColor(255, 255, 255)

    return table_shape


def add_card(slide, left, top, width, height, color=BG_CARD):
    """Add a rounded rectangle card background."""
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top),
                                    Inches(width), Inches(height))
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()
    shape.shadow.inherit = False
    return shape


def add_slide_number(slide, number, total):
    """Add slide number."""
    add_textbox(slide, 8.5, 7.0, 1.5, 0.4, f"{number} / {total}",
                font_size=10, color=GRAY, alignment=PP_ALIGN.RIGHT)


def add_figure_if_exists(slide, filename, left, top, width):
    """Add a figure from the figures directory if it exists."""
    filepath = os.path.join(FIGURES_DIR, filename)
    if os.path.exists(filepath):
        slide.shapes.add_picture(filepath, Inches(left), Inches(top), Inches(width))
        return True
    return False


def build_presentation():
    prs = Presentation()
    prs.slide_width = Inches(10)
    prs.slide_height = Inches(7.5)
    total_slides = 15

    # ================================================================
    # SLIDE 1: TITLE
    # ================================================================
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # Blank layout
    set_slide_bg(slide, RGBColor(10, 10, 46))

    add_textbox(slide, 0.8, 1.5, 8.4, 1.5,
                'Deep Learning for HIV-1\nSubtype Classification',
                font_size=36, color=LIGHT_BLUE, bold=True, alignment=PP_ALIGN.LEFT)

    add_textbox(slide, 0.8, 3.2, 8.4, 0.8,
                'Using Raw Nucleotide Sequences from the pol Gene',
                font_size=20, color=GRAY, alignment=PP_ALIGN.LEFT)

    add_textbox(slide, 0.8, 4.5, 8.4, 1.5,
                'Jovita Nagawa\nMSB7216: Deep Learning for Health Data\nFinal Project Presentation — May 2026',
                font_size=14, color=RGBColor(100, 116, 139), alignment=PP_ALIGN.LEFT)

    add_textbox(slide, 0.8, 6.2, 8.4, 0.5,
                'PyTorch  •  1D-CNN  •  BiLSTM  •  DNABERT  •  K-mers  •  Gradio',
                font_size=12, color=BLUE, alignment=PP_ALIGN.LEFT)

    add_slide_number(slide, 1, total_slides)

    # ================================================================
    # SLIDE 2: WHY IT MATTERS
    # ================================================================
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)

    add_textbox(slide, 0.5, 0.3, 9, 0.7, '🧬  Why HIV-1 Subtype Classification Matters',
                font_size=28, color=LIGHT_BLUE, bold=True)

    add_bullet_frame(slide, 0.5, 1.2, 4.5, 2.5, [
        'HIV-1 has extreme genetic diversity across 9+ subtypes',
        'Subtypes A and D co-circulate in East Africa',
        'Subtype influences drug efficacy & disease progression',
        'Traditional subtyping requires slow sequence alignment',
    ], font_size=14)

    add_textbox(slide, 0.5, 3.8, 4.5, 0.5, 'Our Goal', font_size=18, color=PURPLE, bold=True)
    add_bullet_frame(slide, 0.5, 4.3, 4.5, 2.0, [
        'Classify pol gene → subtypes A, B, C, D',
        'Use raw nucleotide data — no alignment required',
        'Enable rapid, unaligned-sequence surveillance',
    ], font_size=14)

    add_card(slide, 5.3, 1.2, 4.2, 4.8)
    add_textbox(slide, 5.5, 1.4, 3.8, 0.5, 'Clinical Relevance', font_size=16, color=PURPLE, bold=True)
    add_bullet_frame(slide, 5.5, 1.9, 3.8, 4.0, [
        'Epidemiological surveillance',
        'Treatment optimization',
        'Decentralized molecular testing',
        'Resource-limited settings',
    ], font_size=14, color=WHITE)

    add_slide_number(slide, 2, total_slides)

    # ================================================================
    # SLIDE 3: DATASET
    # ================================================================
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)

    add_textbox(slide, 0.5, 0.3, 9, 0.7, '📊  Dataset: LANL HIV Sequence Database',
                font_size=28, color=LIGHT_BLUE, bold=True)

    add_bullet_frame(slide, 0.5, 1.2, 4.5, 2.5, [
        'Source: Los Alamos National Lab (LANL) — gold standard',
        'Accessions from Solis-Reyes et al. (2018) Kameris paper',
        'NOT from Kaggle or Zindi',
        '9,270 accessions → 9,264 retrieved → 5,564 final',
        'All sequences: originally 4,259 bp (gap-stripped to ~2,500-4,200)',
    ], font_size=13)

    add_table(slide, 5.2, 1.2, 4.3, 2.0,
              ['Subtype', 'Count', '%'],
              [['B', '2,908', '52.3%'],
               ['C', '2,303', '41.4%'],
               ['A', '278', '5.0%'],
               ['D', '75', '1.3%']])

    add_card(slide, 5.2, 3.8, 4.3, 0.8, RGBColor(60, 40, 20))
    add_textbox(slide, 5.4, 3.9, 3.9, 0.6,
                '⚠️ Severe imbalance: Subtype D has only 75 sequences (30:1 ratio vs B)',
                font_size=11, color=ORANGE)

    # Add distribution figure if available
    if add_figure_if_exists(slide, 'subtype_distribution.png', 0.5, 4.5, 4.5):
        pass

    add_slide_number(slide, 3, total_slides)

    # ================================================================
    # SLIDE 4: METHODOLOGY
    # ================================================================
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)

    add_textbox(slide, 0.5, 0.3, 9, 0.7, '🏗️  Methodology Overview',
                font_size=28, color=LIGHT_BLUE, bold=True)

    # Pipeline flow
    steps = ['LANL\nFASTA', 'Parse &\nClean', 'One-Hot\nEncode', 'Stratified\nSplit', 'Train\nModels', 'Evaluate\n& Deploy']
    for i, step in enumerate(steps):
        x = 0.3 + i * 1.6
        add_card(slide, x, 1.3, 1.3, 0.9, RGBColor(30, 30, 60))
        add_textbox(slide, x + 0.05, 1.35, 1.2, 0.8, step,
                    font_size=11, color=LIGHT_BLUE, alignment=PP_ALIGN.CENTER)
        if i < len(steps) - 1:
            add_textbox(slide, x + 1.3, 1.5, 0.3, 0.5, '→', font_size=18, color=BLUE)

    add_textbox(slide, 0.5, 2.6, 4.5, 0.5, 'Encoding: Gap-Stripped One-Hot',
                font_size=16, color=PURPLE, bold=True)
    add_bullet_frame(slide, 0.5, 3.1, 4.5, 2.5, [
        'Strip alignment gaps → variable-length',
        'Each nucleotide → 4D vector (A, C, G, T)',
        'Dynamic per-batch padding (collate_fn)',
        'No alignment required at inference',
    ], font_size=13)

    add_textbox(slide, 5.3, 2.6, 4.2, 0.5, 'Focal Loss (γ=2.0) + Class Weights',
                font_size=16, color=PURPLE, bold=True)
    add_table(slide, 5.3, 3.1, 4.0, 2.0,
              ['Class', 'Weight'],
              [['A', '5.02×'], ['B', '0.48×'], ['C', '0.60×'], ['D', '18.38×']])

    add_slide_number(slide, 4, total_slides)

    # ================================================================
    # SLIDE 5: MODEL ARCHITECTURES
    # ================================================================
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)

    add_textbox(slide, 0.5, 0.3, 9, 0.7, '🧠  Model Architectures',
                font_size=28, color=LIGHT_BLUE, bold=True)

    # MLP Card
    add_card(slide, 0.3, 1.3, 3.0, 4.5)
    add_textbox(slide, 0.5, 1.4, 2.6, 0.4, 'Baseline: MLP', font_size=16, color=GRAY, bold=True)
    add_textbox(slide, 0.5, 1.9, 2.6, 2.5,
                'AdaptiveAvgPool1d(64)\nFlatten → 256\nFC(256→512) + BN + ReLU\nFC(512→256) + BN + ReLU\nFC(256→4)',
                font_size=12, color=RGBColor(180, 180, 200))
    add_textbox(slide, 0.5, 4.8, 2.6, 0.4, '196K params  •  5 epochs',
                font_size=11, color=GRAY, alignment=PP_ALIGN.CENTER)

    # CNN Card
    add_card(slide, 3.5, 1.3, 3.0, 4.5, RGBColor(30, 41, 70))
    add_textbox(slide, 3.7, 1.4, 2.6, 0.4, '⭐ Primary: 1D-CNN', font_size=16, color=BLUE, bold=True)
    add_textbox(slide, 3.7, 1.9, 2.6, 2.5,
                'Conv1d(4→64, k=7) + Pool(4)\nConv1d(64→128, k=5) + Pool(4)\nConv1d(128→256, k=3) + Pool(4)\nConv1d(256→256, k=3) + GAP\nFC(256→128→4)',
                font_size=12, color=RGBColor(180, 200, 220))
    add_textbox(slide, 3.7, 4.8, 2.6, 0.4, '374K params  •  7 epochs',
                font_size=11, color=GREEN, alignment=PP_ALIGN.CENTER)

    # BiLSTM Card
    add_card(slide, 6.7, 1.3, 3.0, 4.5)
    add_textbox(slide, 6.9, 1.4, 2.6, 0.4, 'Secondary: BiLSTM', font_size=16, color=PURPLE, bold=True)
    add_textbox(slide, 6.9, 1.9, 2.6, 2.5,
                'Embedding(5→128)\nBiLSTM(128→256, 2 layers)\nConcat fwd+bwd → 512\nFC(512→128→4)',
                font_size=12, color=RGBColor(180, 180, 200))
    add_textbox(slide, 6.9, 4.8, 2.6, 0.4, '2.4M params  •  17 epochs',
                font_size=11, color=ORANGE, alignment=PP_ALIGN.CENTER)

    add_textbox(slide, 0.5, 6.2, 9, 0.5,
                'All models: AdamW • CosineAnnealing LR • Focal Loss (γ=2.0) • Early Stopping • Grad Clipping',
                font_size=11, color=GRAY, alignment=PP_ALIGN.CENTER)

    add_slide_number(slide, 5, total_slides)

    # ================================================================
    # SLIDE 6: TRANSFER LEARNING
    # ================================================================
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)

    add_textbox(slide, 0.5, 0.3, 9, 0.7, '🚀  Transfer Learning & K-mer Experiments',
                font_size=28, color=LIGHT_BLUE, bold=True)

    add_textbox(slide, 0.5, 1.2, 4.5, 0.5, 'DNABERT (Transfer Learning Limitations)', font_size=18, color=PURPLE, bold=True)
    add_bullet_frame(slide, 0.5, 1.7, 4.5, 3.0, [
        'Pre-trained on human genome (Ji et al., 2021)',
        'Standard BERT: 12 layers, 12 heads, 768-dim',
        'DNABERT relies on 6-mer tokens (max 512 length)',
        'Pol sequences (4,259bp) severely truncated to ~517bp',
        'Discarded distal motifs critical for subtype discrimination',
        'Achieved only 41.44% accuracy (Failed to generalize)',
    ], font_size=13)

    add_card(slide, 5.3, 1.2, 4.2, 4.5)
    add_textbox(slide, 5.5, 1.3, 3.8, 0.4, 'K-mer DNN (Alt. Features)', font_size=18, color=GREEN, bold=True)
    add_bullet_frame(slide, 5.5, 1.8, 3.8, 2.0, [
        '6-mer frequency vectors (4⁶ = 4,096 features)',
        'Inspired by Kameris (Solis-Reyes 2018)',
        'Discards positional info → composition only',
        'DNN: 4096→512→256→128→4',
    ], font_size=13)
    add_textbox(slide, 5.5, 4.2, 3.8, 1.0,
                'Key question: Does positional info\n(CNN) outperform composition (k-mers)?',
                font_size=12, color=ORANGE)

    add_slide_number(slide, 6, total_slides)

    # ================================================================
    # SLIDE 7: TRAINING DYNAMICS
    # ================================================================
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)

    add_textbox(slide, 0.5, 0.3, 9, 0.7, '📈  Training Dynamics',
                font_size=28, color=LIGHT_BLUE, bold=True)

    add_textbox(slide, 0.5, 1.2, 4.5, 0.5, '1D-CNN (Primary Model)', font_size=18, color=BLUE, bold=True)
    add_table(slide, 0.5, 1.8, 4.3, 2.0,
              ['Epoch', 'Train Acc', 'Val Loss', 'Val Acc'],
              [['1', '60.4%', '1.4777', '46.5%'],
               ['2 ★', '97.7%', '0.3531', '98.7%'],
               ['7', '99.7%', '—', 'Early stop']])

    add_textbox(slide, 5.3, 1.2, 4.5, 0.5, 'BiLSTM (Secondary Model)', font_size=18, color=PURPLE, bold=True)
    add_table(slide, 5.3, 1.8, 4.3, 2.0,
              ['Epoch', 'Train Acc', 'Val Loss', 'Val Acc'],
              [['1', '57.5%', '1.1462', '76.2%'],
               ['12 ★', '93.3%', '0.5059', '95.1%'],
               ['17', '94.7%', '—', 'Early stop']])

    # Add training curves figure
    add_figure_if_exists(slide, 'model_comparison_curves.png', 0.5, 4.2, 9.0)

    add_slide_number(slide, 7, total_slides)

    # ================================================================
    # SLIDE 8: RESULTS OVERVIEW
    # ================================================================
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, RGBColor(15, 26, 15))

    add_textbox(slide, 0.5, 0.3, 9, 0.7, '🏆  Results: Model Comparison',
                font_size=28, color=LIGHT_BLUE, bold=True)

    add_table(slide, 0.5, 1.3, 9.0, 2.0,
              ['Model', 'Accuracy', 'Macro F1', 'Macro Precision', 'Macro Recall'],
              [['MLP Baseline', '72.81%', '0.6095', '0.6025', '0.7410'],
               ['1D-CNN ⭐', '98.44%', '0.7421', '0.7409', '0.7435'],
               ['BiLSTM', '93.53%', '0.6468', '0.6173', '0.7035']])

    # Metric cards
    for i, (val, label, col) in enumerate([
        ('98.4%', 'Best Accuracy (CNN)', GREEN),
        ('+25.6pp', 'vs. Baseline', GREEN),
        ('1.6%', 'Error Rate (13/835)', RED),
    ]):
        x = 0.5 + i * 3.2
        add_card(slide, x, 4.0, 2.8, 1.8)
        add_textbox(slide, x, 4.1, 2.8, 1.0, val,
                    font_size=36, color=col, bold=True, alignment=PP_ALIGN.CENTER)
        add_textbox(slide, x, 5.0, 2.8, 0.5, label,
                    font_size=12, color=GRAY, alignment=PP_ALIGN.CENTER)

    add_slide_number(slide, 8, total_slides)

    # ================================================================
    # SLIDE 9: PER-CLASS PERFORMANCE
    # ================================================================
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)

    add_textbox(slide, 0.5, 0.3, 9, 0.7, '📋  Per-Class Performance (1D-CNN)',
                font_size=28, color=LIGHT_BLUE, bold=True)

    add_table(slide, 0.5, 1.2, 4.5, 2.5,
              ['Subtype', 'Precision', 'Recall', 'F1', 'n'],
              [['A ✅', '1.000', '0.976', '0.988', '42'],
               ['B ✅', '1.000', '0.998', '0.999', '436'],
               ['C ✅', '0.964', '1.000', '0.982', '346'],
               ['D ❌', '0.000', '0.000', '0.000', '11']])

    add_textbox(slide, 5.3, 1.2, 4.2, 0.5, 'Key Insights', font_size=18, color=PURPLE, bold=True)
    add_bullet_frame(slide, 5.3, 1.8, 4.2, 2.5, [
        'Accuracy (98.4%) reflects weighted performance',
        'Macro F1 (0.74) penalizes equally per class',
        'D\'s 0.00 F1 drags macro average down',
        'Weighted F1 = 0.978 (practical measure)',
        'Excellent for A, B, C — unusable for D',
        'Root cause: data scarcity, not architecture',
    ], font_size=13)

    # Add per-class metrics figure
    add_figure_if_exists(slide, 'best_model_per_class_metrics.png', 0.5, 4.5, 9.0)

    add_slide_number(slide, 9, total_slides)

    # ================================================================
    # SLIDE 10: ERROR ANALYSIS
    # ================================================================
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)

    add_textbox(slide, 0.5, 0.3, 9, 0.7, '🔍  Error Analysis',
                font_size=28, color=LIGHT_BLUE, bold=True)

    add_textbox(slide, 0.5, 1.1, 4.5, 0.5,
                'Only 13/835 test sequences misclassified (1.6%)',
                font_size=15, color=WHITE, bold=True)

    add_table(slide, 0.5, 1.7, 4.3, 1.5,
              ['True → Predicted', 'Count', 'Notes'],
              [['D → C', '11', 'Phylogenetically close'],
               ['A → C', '1', 'Rare'],
               ['B → C', '1', 'Rare']])

    add_textbox(slide, 0.5, 3.5, 4.3, 0.4, 'Error Confidence', font_size=14, color=PURPLE, bold=True)
    add_bullet_frame(slide, 0.5, 3.9, 4.3, 1.5, [
        'Mean confidence: 0.709',
        'Median confidence: 0.786',
        'Systematic, not borderline errors',
    ], font_size=13)

    add_card(slide, 5.3, 1.2, 4.2, 2.5)
    add_textbox(slide, 5.5, 1.3, 3.8, 0.4, 'Biological Explanation', font_size=14, color=PURPLE, bold=True)
    add_bullet_frame(slide, 5.5, 1.7, 3.8, 2.0, [
        'C and D share phylogenetic similarity in pol',
        'Both co-circulate in East Africa',
        'Possible recombination events',
        '30:1 ratio (C vs D training samples)',
    ], font_size=12)

    add_card(slide, 5.3, 4.0, 4.2, 2.0)
    add_textbox(slide, 5.5, 4.1, 3.8, 0.4, 'Why CNN > BiLSTM', font_size=14, color=BLUE, bold=True)
    add_bullet_frame(slide, 5.5, 4.5, 3.8, 1.5, [
        'Subtype signal is local (motif-level)',
        'CNN filters naturally detect local motifs',
        'BiLSTM adds 6.5× params without benefit',
    ], font_size=12)

    add_slide_number(slide, 10, total_slides)

    # ================================================================
    # SLIDE 11: CONFUSION MATRICES
    # ================================================================
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)

    add_textbox(slide, 0.5, 0.3, 9, 0.7, '📊  Confusion Matrices & ROC Curves',
                font_size=28, color=LIGHT_BLUE, bold=True)

    add_figure_if_exists(slide, 'all_confusion_matrices.png', 0.3, 1.2, 9.4)
    add_figure_if_exists(slide, 'best_model_roc.png', 2.5, 4.0, 5.0)

    add_slide_number(slide, 11, total_slides)

    # ================================================================
    # SLIDE 12: DEPLOYMENT
    # ================================================================
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)

    add_textbox(slide, 0.5, 0.3, 9, 0.7, '🚀  Deployment: Gradio Web App',
                font_size=28, color=LIGHT_BLUE, bold=True)

    add_textbox(slide, 0.5, 1.2, 4.5, 0.5, 'Features', font_size=18, color=PURPLE, bold=True)
    add_bullet_frame(slide, 0.5, 1.7, 4.5, 3.0, [
        'Paste or upload HIV-1 nucleotide sequences',
        'Supports FASTA format and plain text',
        'Real-time subtype prediction with confidence',
        'Handles ambiguous IUPAC bases',
        'One-click shareable link',
        'Works locally or on Google Colab',
    ], font_size=13)

    add_card(slide, 5.3, 1.2, 4.2, 3.0)
    add_textbox(slide, 5.5, 1.3, 3.8, 0.4, 'How to Run', font_size=16, color=GREEN, bold=True)
    add_textbox(slide, 5.5, 1.8, 3.8, 1.5,
                'python app.py\n\n# Or in Google Colab:\nexec(open(\'app.py\').read())\n\nInput: HIV-1 nucleotide sequence\nOutput: Subtype + confidence scores',
                font_size=12, color=RGBColor(180, 200, 220))

    add_card(slide, 0.5, 5.2, 9.0, 0.8, RGBColor(60, 40, 20))
    add_textbox(slide, 0.7, 5.3, 8.6, 0.6,
                '⚠️  Research/educational tool only. For clinical subtype determination, use REGA, COMET, or jpHMM.',
                font_size=12, color=ORANGE)

    add_slide_number(slide, 12, total_slides)

    # ================================================================
    # SLIDE 13: ETHICAL CONSIDERATIONS
    # ================================================================
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)

    add_textbox(slide, 0.5, 0.3, 9, 0.7, '⚖️  Ethical Considerations',
                font_size=28, color=LIGHT_BLUE, bold=True)

    add_textbox(slide, 0.5, 1.2, 4.5, 0.5, 'Data Privacy', font_size=16, color=PURPLE, bold=True)
    add_bullet_frame(slide, 0.5, 1.7, 4.5, 1.5, [
        'All sequences from public LANL database',
        'No patient-identifiable information',
        'GenBank accession numbers only',
    ], font_size=13)

    add_textbox(slide, 0.5, 3.3, 4.5, 0.5, 'Sampling Bias', font_size=16, color=PURPLE, bold=True)
    add_bullet_frame(slide, 0.5, 3.8, 4.5, 2.0, [
        'Subtype B overrepresented (Western sequencing bias)',
        'A & D underrepresented despite East African importance',
        'Model performance mirrors this inequity',
    ], font_size=13)

    add_card(slide, 5.3, 1.2, 4.2, 3.5)
    add_textbox(slide, 5.5, 1.3, 3.8, 0.4, 'Fairness Implications', font_size=16, color=ORANGE, bold=True)
    add_bullet_frame(slide, 5.5, 1.8, 3.8, 3.0, [
        'Near-perfect for well-sampled subtypes (B, C)',
        'Complete failure for subtype D',
        'D is most relevant to East African settings',
        'Communities most affected → least represented',
        'Explicit disclaimers in deployment',
    ], font_size=12)

    add_slide_number(slide, 13, total_slides)

    # ================================================================
    # SLIDE 14: LIMITATIONS & FUTURE WORK
    # ================================================================
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)

    add_textbox(slide, 0.5, 0.3, 9, 0.7, '⚠️  Limitations & Future Work',
                font_size=28, color=LIGHT_BLUE, bold=True)

    add_textbox(slide, 0.5, 1.2, 4.5, 0.5, 'Current Limitations', font_size=16, color=RED, bold=True)
    add_bullet_frame(slide, 0.5, 1.7, 4.5, 4.5, [
        'Severe class imbalance (D: 75 sequences)',
        'Only 4 of 9+ subtypes classified',
        'No CRF/recombinant detection',
        'Gap-stripped ≠ truly unaligned data',
        'Transformer context limited to 512 tokens',
        'No explainability (Grad-CAM, attention)',
    ], font_size=13)

    add_textbox(slide, 5.3, 1.2, 4.2, 0.5, 'Future Directions', font_size=16, color=GREEN, bold=True)
    add_bullet_frame(slide, 5.3, 1.7, 4.2, 4.5, [
        'Data augmentation for rare subtypes',
        'Multi-gene classification (gag + pol + env)',
        'Extend to all subtypes and CRFs',
        'Grad-CAM filter visualization',
        'Efficient transformers (FlashAttention)',
        'True unaligned sequence evaluation',
    ], font_size=13)

    add_slide_number(slide, 14, total_slides)

    # ================================================================
    # SLIDE 15: THANK YOU
    # ================================================================
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, RGBColor(26, 10, 46))

    add_textbox(slide, 0.5, 2.0, 9, 1.5, 'Thank You',
                font_size=44, color=LIGHT_BLUE, bold=True, alignment=PP_ALIGN.CENTER)
    add_textbox(slide, 0.5, 3.3, 9, 0.8, 'Questions & Discussion',
                font_size=24, color=GRAY, alignment=PP_ALIGN.CENTER)

    add_card(slide, 2.5, 4.5, 5.0, 1.8)
    add_textbox(slide, 2.7, 4.6, 4.6, 0.4, 'Resources', font_size=14, color=PURPLE, bold=True)
    add_bullet_frame(slide, 2.7, 5.0, 4.6, 1.2, [
        'GitHub: github.com/jnagawa/hiv-subtype-classifier',
        'Dataset: Los Alamos HIV Database (hiv.lanl.gov)',
        'Framework: PyTorch + Gradio',
    ], font_size=12)

    add_textbox(slide, 0.5, 6.5, 9, 0.5,
                'Jovita Nagawa — MSB7216: Deep Learning for Health Data — May 2026',
                font_size=12, color=GRAY, alignment=PP_ALIGN.CENTER)

    add_slide_number(slide, 15, total_slides)

    # ---- Save ----
    prs.save(OUTPUT_PATH)
    print(f'Presentation saved to: {OUTPUT_PATH}')


if __name__ == '__main__':
    build_presentation()
