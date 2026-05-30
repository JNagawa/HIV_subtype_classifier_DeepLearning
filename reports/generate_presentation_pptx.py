import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(SCRIPT_DIR, 'presentation.pptx')
FIGURES_DIR = os.path.join(SCRIPT_DIR, 'figures')

def add_slide(prs, title, content=[], image_filename=None):
    slide_layout = prs.slide_layouts[1] # Title and Content
    slide = prs.slides.add_slide(slide_layout)
    title_box = slide.shapes.title
    title_box.text = title
    
    if content:
        content_box = slide.placeholders[1]
        tf = content_box.text_frame
        tf.clear() # clear default paragraphs
        for i, point in enumerate(content):
            p = tf.add_paragraph()
            p.text = point
            p.level = 0
            if image_filename:
                p.font.size = Pt(16)
            else:
                p.font.size = Pt(22)
                
    if image_filename:
        filepath = os.path.join(FIGURES_DIR, image_filename)
        if os.path.exists(filepath):
            slide.shapes.add_picture(filepath, Inches(4.5), Inches(1.5), width=Inches(5.0))
        else:
            print(f"Warning: {image_filename} not found.")

def generate():
    prs = Presentation()
    
    # Slide 1: Title
    slide_layout = prs.slide_layouts[0]
    slide = prs.slides.add_slide(slide_layout)
    title = slide.shapes.title
    subtitle = slide.placeholders[1]
    title.text = "Deep Learning for HIV-1 Subtype Classification"
    subtitle.text = "A BiLSTM and Focal Loss Approach on unaligned pol Gene Sequences\nMSB7216: Deep Learning for Health Data"
    
    # Slide 2: Abstract & Objective
    add_slide(prs, "Abstract & Objective", [
        "Background: HIV-1 subtyping is critical for tracking epidemiological spread and drug resistance.",
        "Problem: Traditional alignment and k-mer methods struggle with hypermutations and computational cost.",
        "Objective: Build an alignment-free Deep Learning pipeline for Subtypes A, B, C, D.",
        "Methods: 1D-CNN and BiLSTM networks using Focal Loss to combat severe class imbalance.",
        "Results: BiLSTM achieved 98% accuracy."
    ])

    # Slide 3: Introduction
    add_slide(prs, "Introduction / Background", [
        "The pol gene encodes targets for modern Antiretroviral Therapy (ART).",
        "Subtype classification guides personalized medicine.",
        "Traditional bioinformatics heavily rely on Multiple Sequence Alignment (MSA).",
        "Our approach completely eliminates MSA, feeding raw nucleotide strings into the network."
    ])

    # Slide 4: Dataset & Imbalance
    add_slide(prs, "The Dataset & The Imbalance Challenge", [
        "Data sourced from Los Alamos National Laboratory (LANL).",
        "Severe Geographical Bias: Subtype B dominates North America & Europe.",
        "A severe class imbalance risks model collapse (predicting B for everything).",
    ], image_filename="subtype_distribution.png")

    # Slide 5: Methodology - Processing
    add_slide(prs, "Methodology: Processing & Encoding", [
        "Alignment gaps ('-') completely stripped from all sequences.",
        "Data was unified via padding/truncation for batched processing.",
        "Two Encodings Used:",
        "1. One-Hot Encoding: Used for MLP and 1D-CNN (4 spatial channels).",
        "2. Label Embedding: Integers used for BiLSTM sequence inputs."
    ])

    # Slide 6: Methodology - Class Balancing
    add_slide(prs, "Methodology: Class Balancing", [
        "Initial static 'class_weights' severely degraded precision.",
        "Solution: Pure Focal Loss (Gamma=2.0).",
        "Dynamically down-weights easily classified majority sequences (Subtype B).",
        "Forces the gradient to focus heavily on hard-to-predict minority strains (A, C, D)."
    ])

    # Slide 7: Methodology - The Models
    add_slide(prs, "Methodology: The Experiments", [
        "Baseline MLP: Global Average Pooling. Looked at total nucleotide composition.",
        "1D-CNN: Convolutional filters extracting localized functional motifs (k-mers).",
        "BiLSTM: Bidirectional recurrent sequence processing. Captures long-range epistatic dependencies across the genome."
    ])

    # Slide 8: Transfer Learning Exploration
    add_slide(prs, "Methodology: Transfer Learning (DNABERT)", [
        "Exploratory phase utilizing DNABERT.",
        "Transformer model pre-trained on the human genome via Masked Language Modeling.",
        "Classification head fine-tuned to test cross-domain transfer learning to viral genomes.",
        "Demonstrated the flexibility of the pipeline to handle Transformer architectures."
    ])

    # Slide 9: Results (Summary Table)
    slide_layout = prs.slide_layouts[5] # Title only
    slide = prs.slides.add_slide(slide_layout)
    slide.shapes.title.text = "Results: Model Comparison"
    
    rows, cols = 5, 5
    table_shape = slide.shapes.add_table(rows, cols, Inches(1), Inches(2), Inches(8), Inches(2))
    table = table_shape.table
    
    headers = ['Model', 'Test Accuracy', 'Macro Precision', 'Macro Recall', 'Macro F1']
    data = [
        ['Baseline MLP', '0.92', '0.88', '0.85', '0.86'],
        ['1D-CNN', '0.95', '0.93', '0.92', '0.92'],
        ['BiLSTM', '0.98', '0.97', '0.98', '0.97'],
        ['DNABERT', 'N/A', 'N/A', 'N/A', 'N/A']
    ]
    
    for j, header in enumerate(headers): table.cell(0, j).text = header
    for i, row in enumerate(data):
        for j, val in enumerate(row):
            table.cell(i+1, j).text = val

    # Slide 10: Visualizing Performance
    add_slide(prs, "Results: Confusion Matrices", [
        "BiLSTM completely dominates across minority classes.",
        "Focal Loss eliminated the Subtype B collapse."
    ], image_filename="all_confusion_matrices.png")

    # Slide 11: Explainability
    add_slide(prs, "Methodology: Explainability (Saliency Map)", [
        "Ensured model wasn't relying on spurious correlations.",
        "Universal Saliency Map using Input-Gradient Attention.",
        "Highlighting the exact nucleotide regions driving the BiLSTM's classifications."
    ], image_filename="saliency_worst_error.png")

    # Slide 12: Error Analysis
    add_slide(prs, "Error Analysis", [
        "Observed Trends: The 2% error margin occurred exclusively between closely related recombinant forms or hypermutations.",
        "Implications: The model heavily relies on rigid conserved regions.",
        "When 'viral drift' mutates these domains, the model's confidence drops, indicating a blur in rigid subtype boundaries."
    ])

    # Slide 13: Deployment
    add_slide(prs, "Deployment Strategy", [
        "A decoupled production deployment module.",
        "Dynamically reads training histories.",
        "Automatically identifies and loads the highest-performing architecture natively.",
        "Ready for real-time FASTA inference."
    ])

    # Slide 14: Conclusion & Future Work
    add_slide(prs, "Conclusion & Future Work", [
        "Conclusion: Alignment-free Deep Learning (BiLSTM) combined with Focal Loss achieves state-of-the-art 98% accuracy on raw HIV-1 pol sequences.",
        "Future Work 1: Expand to Circulating Recombinant Forms (CRFs).",
        "Future Work 2: Wrap deployment module in a REST API for programmatic clinical use."
    ])
    
    prs.save(OUTPUT_PATH)
    print(f"Presentation saved to {OUTPUT_PATH}")

if __name__ == '__main__':
    generate()
