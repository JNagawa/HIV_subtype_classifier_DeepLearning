import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(SCRIPT_DIR, 'presentation.pptx')
FIGURES_DIR = os.path.join(SCRIPT_DIR, 'figures')

def style_title(slide):
    if slide.shapes.title:
        title = slide.shapes.title
        title.fill.solid()
        title.fill.fore_color.rgb = RGBColor(20, 50, 90)
        for p in title.text_frame.paragraphs:
            p.font.color.rgb = RGBColor(255, 255, 255)
            p.font.name = 'Segoe UI'
            p.font.bold = True

def add_slide(prs, title, content=[], image_filename=None, image_width=5.0, image_left=4.5, image_top=1.5):
    slide_layout = prs.slide_layouts[1]
    slide = prs.slides.add_slide(slide_layout)
    slide.shapes.title.text = title
    style_title(slide)
    
    if content:
        content_box = slide.placeholders[1]
        tf = content_box.text_frame
        tf.clear()
        for point in content:
            p = tf.add_paragraph()
            p.text = f"• {point}"
            p.level = 0
            p.font.name = 'Segoe UI'
            p.font.size = Pt(18)
                
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
    title = slide.shapes.title
    title.text = "Deep Learning for HIV-1 Subtype Classification"
    title.text_frame.paragraphs[0].font.name = 'Segoe UI'
    title.text_frame.paragraphs[0].font.bold = True
    title.text_frame.paragraphs[0].font.color.rgb = RGBColor(20, 50, 90)
    
    subtitle = slide.placeholders[1]
    subtitle.text = "A BiLSTM and Focal Loss Approach on unaligned pol Gene Sequences\nMSB7216: Deep Learning for Health Data"
    for p in subtitle.text_frame.paragraphs:
        p.font.name = 'Segoe UI'
        p.font.color.rgb = RGBColor(80, 80, 80)
    
    # 2. Abstract & Overview
    add_slide(prs, "Abstract & Overview", [
        "Background: HIV-1 subtyping is essential for guiding antiretroviral therapy and tracking epidemiological spread.",
        "Previous Work: Existing tools heavily rely on computationally expensive Multiple Sequence Alignment (MSA) or k-mer counting.",
        "Problem Statement: MSA struggles with hypermutations, and genomic databases suffer from severe geographical class imbalance.",
        "Objective: To develop an automated, alignment-free Deep Learning classifier to categorize raw nucleotide sequences into Subtypes A, B, C, and D.",
        "Significance: Eliminating the MSA bottleneck enables faster, more scalable, and highly accurate subtyping for clinical deployment."
    ])

    # 3. The Dataset
    add_slide(prs, "Data Acquisition (Kameris Experiment)", [
        "Data was acquired by parsing 9,270 HIV-1 pol gene accessions utilized in the Kameris et al. experiment.",
        "A JSON file was pulled from the Kameris GitHub repository to extract specific LANL sequence IDs.",
        "These IDs were fed into the LANL HIV Sequence Database to batch download the raw FASTA sequences.",
        "Sequences were then strictly filtered to our 4 target classes: Subtypes A, B, C, and D.",
        "Severe Bias: Subtype B heavily dominates the final dataset (52.2%)."
    ])

    # 4. Processing & Encoding
    slide_layout = prs.slide_layouts[1]
    slide = prs.slides.add_slide(slide_layout)
    slide.shapes.title.text = "Methodology: Processing & Encoding"
    style_title(slide)
    
    content_box = slide.placeholders[1]
    tf = content_box.text_frame
    for text in [
        "Gap Stripping: Alignment gaps ('-') completely removed.",
        "Padding: Unaligned sequences padded/truncated to uniform 3000bp.",
        "One-Hot Encoding: Matrix [L x 4] for spatial models (CNN, MLP).",
        "Label Embedding: Integer mapping for sequential models (BiLSTM)."
    ]:
        p = tf.add_paragraph()
        p.text = f"• {text}"
        p.font.size = Pt(16)
        
    # Table illustrating encoding
    table_shape = slide.shapes.add_table(3, 3, Inches(0.5), Inches(3.0), Inches(4.5), Inches(1.5))
    table = table_shape.table
    table.cell(0, 0).text, table.cell(0, 1).text, table.cell(0, 2).text = "Raw Sequence", "Cleaned Sequence", "Integer Encoding"
    table.cell(1, 0).text, table.cell(1, 1).text, table.cell(1, 2).text = "A-T-C--G", "ATCG", "[1, 4, 2, 3]"
    table.cell(2, 0).text, table.cell(2, 1).text, table.cell(2, 2).text = "T--C-A-N", "TCAN", "[4, 2, 1, 0]"
    for row in table.rows:
        for cell in row.cells:
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(14)
                p.font.bold = True
                
    filepath_len = os.path.join(FIGURES_DIR, "sequence_lengths.png")
    if os.path.exists(filepath_len):
        slide.shapes.add_picture(filepath_len, Inches(5.2), Inches(2.0), width=Inches(4.5))

    # 5. Data Splitting & Class Balancing
    slide_layout = prs.slide_layouts[1]
    slide = prs.slides.add_slide(slide_layout)
    slide.shapes.title.text = "Methodology: Data Splitting & Class Balancing"
    style_title(slide)
    
    content_box = slide.placeholders[1]
    tf = content_box.text_frame
    for text in [
        "Stratified Split: Data divided into 70% Train, 15% Val, and 15% Test.",
        "Class Imbalance: Subtype B severely dominates the data (>52%).",
        "Standard cross-entropy caused precision collapse in minority classes.",
        "Solution: Pure Focal Loss (Gamma=2.0) applied natively in PyTorch.",
        "Forces gradient to focus on hard-to-predict minority strains (A, C, D)."
    ]:
        p = tf.add_paragraph()
        p.text = f"• {text}"
        p.font.size = Pt(16)
        
    filepath_dist = os.path.join(FIGURES_DIR, "subtype_distribution.png")
    if os.path.exists(filepath_dist):
        slide.shapes.add_picture(filepath_dist, Inches(5.2), Inches(3.5), width=Inches(4.5))

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
    style_title(slide)
    
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
