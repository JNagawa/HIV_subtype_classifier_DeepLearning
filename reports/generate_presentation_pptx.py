import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(SCRIPT_DIR, 'presentation.pptx')
FIGURES_DIR = os.path.join(SCRIPT_DIR, 'figures')

def add_slide(prs, title, content=[], image_filename=None, image_width=5.0, image_left=4.5, image_top=1.5):
    slide_layout = prs.slide_layouts[1]
    slide = prs.slides.add_slide(slide_layout)
    title_box = slide.shapes.title
    title_box.text = title
    
    if content:
        content_box = slide.placeholders[1]
        tf = content_box.text_frame
        tf.clear()
        for i, point in enumerate(content):
            p = tf.add_paragraph()
            p.text = point
            p.level = 0
            if image_filename:
                p.font.size = Pt(14)
            else:
                p.font.size = Pt(20)
                
    if image_filename:
        filepath = os.path.join(FIGURES_DIR, image_filename)
        if os.path.exists(filepath):
            slide.shapes.add_picture(filepath, Inches(image_left), Inches(image_top), width=Inches(image_width))
        else:
            print(f"Warning: {image_filename} not found.")

def generate():
    prs = Presentation()
    
    # 1. Title
    slide_layout = prs.slide_layouts[0]
    slide = prs.slides.add_slide(slide_layout)
    slide.shapes.title.text = "Deep Learning for HIV-1 Subtype Classification"
    slide.placeholders[1].text = "A BiLSTM and Focal Loss Approach on unaligned pol Gene Sequences\nMSB7216: Deep Learning for Health Data"
    
    # 2. Abstract & Objective
    add_slide(prs, "Abstract & Objective", [
        "Background: HIV-1 subtyping is critical for tracking drug resistance.",
        "Problem: Traditional MSA methods struggle with hypermutations.",
        "Objective: Alignment-free Deep Learning pipeline for Subtypes A, B, C, D.",
        "Methods: CNN and BiLSTM networks using Focal Loss.",
        "Results: BiLSTM achieved 94.73% accuracy, resisting majority-class collapse."
    ])

    # 3. The Dataset
    add_slide(prs, "The Dataset & Geographical Bias", [
        "Data sourced from Los Alamos National Laboratory (LANL).",
        "Severe Bias: Subtype B dominates North America & Europe (52.2% of data).",
        "Sequence Length Variability: Shown in the plot, padded to 3000bp.",
    ], image_filename="sequence_lengths.png", image_left=4.5)

    # 4. Processing Pipeline
    add_slide(prs, "Methodology: Processing & Encoding", [
        "Alignment gaps ('-') completely stripped.",
        "1. One-Hot Encoding: Used for MLP and 1D-CNN.",
        "2. Label Embedding: Integers used for BiLSTM inputs."
    ], image_filename="subtype_distribution.png", image_left=4.5)

    # 5. Class Balancing Strategy
    add_slide(prs, "Methodology: Class Balancing", [
        "Standard categorical cross-entropy caused precision degradation.",
        "Implemented pure Focal Loss (Gamma=2.0).",
        "Forces the gradient to focus heavily on hard-to-predict minority strains (A, C, D)."
    ])

    # 6. Experiments: Models & Architecture
    add_slide(prs, "Methodology: Architectures", [
        "1. Baseline MLP (73.8k Params): GlobalAvgPool -> Dense(16) -> Classifier.",
        "2. 1D-CNN (200k Params): 4 Conv1D layers (filters: 32 to 128) alternating with MaxPool & Dropout (0.5).",
        "3. BiLSTM (2.1M Params): 128-dim Embedding -> 2-layer Bidirectional LSTM (256 hidden) -> Dense(128)."
    ])

    # 7. Transfer Learning
    add_slide(prs, "Methodology: Transfer Learning", [
        "Exploratory phase utilizing DNABERT.",
        "Pre-trained on the human genome via Masked Language Modeling.",
        "Classification head fine-tuned to test cross-domain transfer learning to viral genomes."
    ], image_filename="dnabert_training_curves.png", image_left=4.5)

    # 8. Explainability
    add_slide(prs, "Methodology: Explainability", [
        "Universal Saliency Map using Input-Gradient Attention.",
        "Highlights the exact nucleotide regions driving the BiLSTM's classifications."
    ], image_filename="saliency_worst_error.png")

    # 9. Results: Table
    slide_layout = prs.slide_layouts[5]
    slide = prs.slides.add_slide(slide_layout)
    slide.shapes.title.text = "Results: Model Comparison"
    
    table_shape = slide.shapes.add_table(5, 5, Inches(1), Inches(2), Inches(8), Inches(2))
    table = table_shape.table
    data = [
        ['Model', 'Test Accuracy', 'Macro Precision', 'Macro Recall', 'Macro F1'],
        ['Baseline MLP', '0.6527', '0.5082', '0.7472', '0.5114'],
        ['1D-CNN', '0.5222', '0.1305', '0.2500', '0.1715'],
        ['BiLSTM', '0.9473', '0.6274', '0.6982', '0.6541'],
        ['DNABERT', '0.5222', '0.1305', '0.2500', '0.1715']
    ]
    for i, row in enumerate(data):
        for j, val in enumerate(row):
            table.cell(i, j).text = val

    # 10. Results: Visuals
    add_slide(prs, "Results: Confusion Matrices", [
        "BiLSTM accurately maps minority classes.",
        "CNN and DNABERT collapsed entirely (52.22% Subtype B baseline)."
    ], image_filename="all_confusion_matrices.png", image_width=5.5, image_left=4.0)

    # 10b. Results: ROC Curves
    add_slide(prs, "Results: ROC Curves", [
        "The Receiver Operating Characteristic confirms the BiLSTM's high true positive rate across all four classes.",
    ], image_filename="best_model_roc.png", image_width=5.0, image_left=4.5)

    # 11. Error Analysis
    add_slide(prs, "Error Analysis", [
        "Causes: CNNs and DNABERT collapsed due to geographic imbalance.",
        "Implications: Saliency Maps show BiLSTMs rely on conserved regions. When 'viral drift' mutates these domains, confidence drops."
    ])

    # 12. Limitations
    add_slide(prs, "Limitations & Challenges", [
        "1. Extreme Geographical Imbalance: Hard for spatial models (CNN) to overcome.",
        "2. Computational Limits: VRAM restricted DNABERT's ability to learn 3000bp sequences.",
        "3. Domain Gap: Human genome pre-training doesn't translate perfectly to viruses."
    ])

    # 13. Deployment
    add_slide(prs, "Deployment Strategy", [
        "Decoupled production deployment module.",
        "Automatically loads the highest-performing architecture (BiLSTM).",
        "Ready for real-time FASTA inference."
    ])

    # 14. Conclusion
    add_slide(prs, "Conclusion & Future Work", [
        "Conclusion: Alignment-free Deep Learning (BiLSTM) handles viral genomic imbalances far better than CNNs, achieving 94.73% accuracy.",
        "Future Work: Expand to Circulating Recombinant Forms (CRFs) and launch via FastAPI."
    ])
    
    prs.save(OUTPUT_PATH)

if __name__ == '__main__':
    generate()
