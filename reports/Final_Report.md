# Deep Learning Model for HIV-1 Subtype Classification Using the *pol* Gene

**MSB7216: Deep Learning for Health Data — Final Project Report**

**Author:** Jovita Nagawa

**Date:** May 2026

---

## 1. Abstract

HIV-1 genetic diversity, organized into subtypes (clades), has direct implications for disease progression, drug resistance, and vaccine design. Accurate and rapid subtype classification is essential for epidemiological surveillance and clinical decision-making, particularly in sub-Saharan Africa where multiple subtypes co-circulate. This project develops and evaluates deep learning models for classifying HIV-1 sequences into four major subtypes (A, B, C, and D) using raw, unaligned nucleotide data from the *pol* gene region. Unlike traditional subtyping tools (REGA, COMET, jpHMM) that require computationally expensive multiple sequence alignment as a prerequisite, and unlike k-mer approaches that require manual feature engineering, our approach strips alignment gaps from sequences and encodes raw nucleotides as one-hot tensors, training convolutional and recurrent neural networks to learn discriminative motifs directly from unaligned data. We address class imbalance using Focal Loss, which down-weights easy-to-classify majority samples and focuses training on hard minority examples. We benchmark three architectures: a Multi-Layer Perceptron (MLP) baseline, a 1D Convolutional Neural Network (CNN), and a Bidirectional Long Short-Term Memory network (BiLSTM), and additionally attempt transfer learning with DNABERT. Using 5,564 curated sequences from the Los Alamos National Laboratory (LANL) HIV Sequence Database, our best model (1D-CNN) achieves strong overall accuracy on the test set, with near-perfect classification for subtypes A, B, and C, though subtype D classification remains challenging due to severe class imbalance (only 75 sequences, 1.3% of the dataset). The deployed Gradio web application incorporates Monte Carlo Dropout for uncertainty estimation, automatically flagging low-confidence predictions as indeterminate. All code, data, and experiments are publicly available on GitHub for full reproducibility.

---

## 2. Introduction

Human Immunodeficiency Virus type 1 (HIV-1) is among the most genetically diverse pathogens known, with its classification into groups, subtypes, and circulating recombinant forms (CRFs) reflecting distinct evolutionary lineages. The major subtypes (A, B, C, D, F, G, H, J, and K) exhibit geographic clustering: subtype B predominates in the Americas and Western Europe, subtype C accounts for roughly half of global infections (concentrated in Southern Africa and India), and subtypes A and D co-circulate in East Africa, where their interaction is associated with differential clinical outcomes (Solis-Reyes et al., 2018).

Subtype identification is clinically important for several reasons. First, certain antiretroviral drugs show differential efficacy across subtypes. Second, subtype information informs vaccine design, as immune responses may be subtype-specific. Third, accurate subtyping is essential for molecular epidemiological surveillance — tracking the spread of subtypes across geographic regions and populations.

Traditional subtyping relies on phylogenetic analysis, which requires multiple sequence alignment (MSA) — a computationally expensive O(n²) to O(n³) process. Tools like REGA, COMET, and jpHMM implement reference-based approaches but still depend on aligned input sequences. More recently, k-mer frequency vector methods have been proposed (Solis-Reyes et al., 2018), which avoid alignment but still require manual feature engineering decisions (e.g., choice of k).

Deep learning offers an attractive alternative: neural networks can learn features directly from raw sequence data without requiring alignment or manual feature engineering. By stripping alignment gaps and working with the natural variable-length nucleotide sequences, deep learning models can operate on raw, unaligned input — eliminating the MSA prerequisite entirely. Convolutional Neural Networks (CNNs) are particularly well-suited for detecting local sequence motifs, while Recurrent Neural Networks (RNNs) can capture long-range dependencies along the genome.

### Objectives

This project aims to:

1. Develop deep learning models that classify HIV-1 *pol* gene sequences into subtypes A, B, C, and D directly from raw, unaligned nucleotide sequences — eliminating the multiple sequence alignment prerequisite that all traditional tools require.
2. Compare the performance of three architectures (MLP, 1D-CNN, BiLSTM) to understand the relative importance of local versus global sequence features for subtype discrimination.
3. Analyze classification errors to understand which subtypes are most confusable and why.
4. Deploy the best model as a web application for real-time sequence classification.

---

## 3. Related Work

### Alignment-Based Subtyping

The gold standard for HIV-1 subtyping involves phylogenetic analysis against reference sequences. Tools like REGA v3 (Pineda-Peña et al., 2013) and COMET (Struck et al., 2014) automate this process, but they require MSA and are computationally expensive for large-scale surveillance.

### K-mer Based Methods

Solis-Reyes et al. (2018) introduced *Kameris*, an open-source k-mer based machine learning tool that achieves high accuracy using Euclidean distances between k-mer frequency vectors and k-nearest neighbors classification. Their approach is fast and does not require alignment, but requires choosing a k-mer length (they used k=6) and discards positional information.

### Deep Learning for Genomic Classification

CNNs have been successfully applied to DNA sequence classification tasks including transcription factor binding site prediction (Alipanahi et al., 2015), variant effect prediction (Zhou & Troyanskaya, 2015), and more recently, pathogen classification. For HIV specifically, Fabris et al. (2019) explored deep learning for drug resistance prediction from *pol* sequences.

### Our Contribution

This project differs from both traditional alignment-dependent tools and k-mer methods in two key ways. First, we strip alignment gaps from LANL sequences and work with the raw, variable-length nucleotide sequences — no MSA is required at inference time, unlike REGA, COMET, or jpHMM. Second, we replace k-mer frequency vectors with direct one-hot encoding of raw nucleotides, preserving positional information and allowing the CNN to learn discriminative motifs automatically. We also compare CNN against BiLSTM architectures and attempt transfer learning with DNABERT to evaluate multiple deep learning paradigms for this task.

---

## 4. Dataset Description

### Source

We use a subset of the dataset described by Solis-Reyes et al. (2018), obtained from the Los Alamos National Laboratory (LANL) HIV Sequence Database (https://www.hiv.lanl.gov/). LANL is the gold-standard curated database for HIV genomic research, maintained by the U.S. Department of Energy, and is explicitly *not* a Kaggle or Zindi dataset.

### Data Acquisition

We downloaded the exact accession IDs used in the Kameris *hiv1-lanl-pol* experiment (9,270 accessions) from the Kameris experiments metadata repository on GitHub. These accession IDs were uploaded to the LANL search interface to retrieve the corresponding *pol* coding sequences (CDS) in FASTA format. Of the 9,270 accessions, 9,264 were successfully retrieved; the remaining 6 had been withdrawn or had incomplete *pol* CDS records.

### Data Characteristics

| Property | Value |
|----------|-------|
| Total sequences retrieved | 9,264 |
| Sequence length (aligned) | 4,259 bp (LANL reference alignment) |
| Sequence length (gap-stripped) | ~2,500–4,200 bp (variable) |
| Gene region | *pol* (coding sequence) |
| Subtypes present | 226 distinct labels (including CRFs) |
| Sequences after filtering to A, B, C, D | 5,588 |
| After removing duplicates | 5,564 |
| Final dataset size | **5,564 sequences** |

### Subtype Distribution

The final 4-class dataset exhibits significant class imbalance:

| Subtype | Count | Percentage |
|---------|-------|------------|
| A | 278 | 5.0% |
| B | 2,908 | 52.3% |
| C | 2,303 | 41.4% |
| D | 75 | 1.3% |

Subtype B is overrepresented because most HIV sequencing historically occurred in North America and Western Europe. Subtype D is severely underrepresented (only 75 sequences), reflecting both its lower global prevalence and fewer sequencing efforts.

### Data Splitting

We used stratified splitting to preserve class proportions across all splits:

| Split | Total | A | B | C | D |
|-------|-------|---|---|---|---|
| Train (70%) | 3,896 | 194 (5.0%) | 2,037 (52.3%) | 1,612 (41.4%) | 53 (1.4%) |
| Validation (15%) | 833 | 42 (5.0%) | 435 (52.2%) | 345 (41.4%) | 11 (1.3%) |
| Test (15%) | 835 | 42 (5.0%) | 436 (52.2%) | 346 (41.4%) | 11 (1.3%) |

---

## 5. Methodology

### 5.1 Gap Stripping and Sequence Encoding

The LANL sequences are distributed in pre-aligned form (4,259 bp, padded with '-' gap characters). To remove the dependency on alignment, we **strip all gap characters** from each sequence, producing raw nucleotide sequences of variable length (~2,500–4,200 bp). This is a critical preprocessing step: traditional tools (REGA, COMET, jpHMM) require aligned input, whereas our model operates directly on the unaligned sequences.

Each gap-stripped sequence is **one-hot encoded** as a 2D tensor of shape (4 × L), where L is the natural sequence length. The four channels correspond to nucleotides A, C, G, and T. Ambiguous IUPAC bases (R, Y, S, W, K, M, B, D, H, V, N) are mapped to zero vectors (no channel activated), effectively treating them as unknown positions.

To handle variable-length sequences within mini-batches, we use a custom collate function that dynamically pads each batch to the length of the longest sequence in that batch. This is more memory-efficient than padding to a fixed global maximum.

For the BiLSTM model, we use **label encoding** (integer indices 0–3 for A, C, G, T, with 4 as padding) with a learned embedding layer, as recurrent models expect sequential token inputs rather than multi-channel signals.

### 5.2 Class Imbalance Handling: Focal Loss

Given the severe imbalance (subtype D has 53 training sequences vs. 2,037 for subtype B), we use **Focal Loss** (Lin et al., 2017) combined with inverse-frequency class weights as the alpha parameter:

FL(p_t) = −α_t · (1 − p_t)^γ · log(p_t)

where γ = 2.0 is the focusing parameter and α_t is the class weight:

| Class | α (Weight) |
|-------|--------|
| A | 5.02 |
| B | 0.48 |
| C | 0.60 |
| D | 18.38 |

Focal Loss improves upon standard class-weighted cross-entropy by additionally down-weighting the loss contribution from easy, well-classified examples (predominantly subtypes B and C). The (1 − p_t)^γ modulating factor ensures training focuses on the hard, ambiguous examples — particularly the subtype D sequences that are frequently confused with subtype C. This dual mechanism (class weights × focusing) provides a more targeted solution than class weights alone.

### 5.3 Model Architectures

#### Baseline: Multi-Layer Perceptron (MLP)

The MLP serves as a non-convolutional baseline. It first applies adaptive average pooling to reduce the variable-length sequence to 64 summary positions, then flattens to a 256-dimensional feature vector (4 channels × 64 positions). Two fully-connected hidden layers (512 → 256) with batch normalization, ReLU activation, and 40% dropout produce the final 4-class logits.

This architecture deliberately sacrifices positional resolution to establish a lower bound on performance, demonstrating the value of learned spatial features in the CNN.

#### Primary Model: 1D Convolutional Neural Network (CNN)

The 1D-CNN consists of four convolutional blocks with increasing receptive fields:

| Block | Filters | Kernel Size | Pooling | Purpose |
|-------|---------|-------------|---------|---------|
| 1 | 64 | 7 | MaxPool(4) | Short motifs (7-mers) |
| 2 | 128 | 5 | MaxPool(4) | Medium motifs |
| 3 | 256 | 3 | MaxPool(4) | Larger patterns |
| 4 | 256 | 3 | GlobalAvgPool | High-level features |

Each block includes batch normalization, ReLU activation, and 30% dropout. The `AdaptiveAvgPool1d(1)` at the final block is what enables variable-length input — regardless of input sequence length, the output is always a fixed 256-dimensional feature vector. The classifier head (256 → 128 → 4) with ReLU and dropout produces the final logits.

The hierarchical design allows the network to detect short nucleotide motifs in early layers and compose them into larger discriminative patterns in deeper layers — analogous to how k-mer approaches capture local sequence composition, but without fixing the feature length.

#### Secondary Model: Bidirectional LSTM (BiLSTM)

The BiLSTM uses a learned embedding layer (vocabulary size 5: A, C, G, T, padding; embedding dimension 128) followed by a 2-layer bidirectional LSTM with hidden dimension 256. The final hidden states from both directions are concatenated (512-dimensional) and passed through a classifier head (512 → 128 → 4).

This architecture captures long-range dependencies in both reading directions, potentially modeling covariation between distant genomic positions.

### 5.4 Training Configuration

All models were trained on a Google Colab T4 GPU (16 GB VRAM) with the following settings:

| Hyperparameter | MLP | CNN | BiLSTM |
|----------------|-----|-----|--------|
| Optimizer | AdamW | AdamW | AdamW |
| Learning rate | 3×10⁻³ | 1×10⁻³ | 5×10⁻⁴ |
| Weight decay | 1×10⁻⁴ | 1×10⁻⁴ | 1×10⁻⁴ |
| LR scheduler | CosineAnnealing | CosineAnnealing | CosineAnnealing |
| Max epochs | 5 | 10 | 30 |
| Early stopping patience | — | 5 | 5 |
| Batch size | 32 | 32 | 32 |
| Gradient clipping | 1.0 | 1.0 | 1.0 |
| Random seed | 42 | 42 | 42 |

All models used Focal Loss (γ=2.0) with inverse-frequency class weights as the alpha parameter, and gradient clipping (max norm 1.0) to prevent exploding gradients, which is particularly important for the LSTM.

---

## 6. Experiments

### Experiment 1: MLP Baseline

The MLP was trained for 5 epochs (no early stopping). Validation loss decreased from 1.30 to 0.72 across epochs, with validation accuracy reaching 73.95% at best. The model converged quickly but exhibited noticeable instability in validation metrics, with a spike to 3.94 validation loss at epoch 2, suggesting the pooled features lack sufficient discriminative power.

### Experiment 2: 1D-CNN Training

The CNN was trained for up to 10 epochs with early stopping (patience 5). The model converged rapidly — achieving 97.74% training accuracy by epoch 2 and best validation loss of 0.3531 at epoch 2. Training was halted at epoch 7 by early stopping. The rapid convergence suggests the convolutional filters quickly learn discriminative nucleotide motifs.

### Experiment 3: BiLSTM Training

The BiLSTM was trained for up to 30 epochs with early stopping (patience 5). It converged more slowly than the CNN, reaching best validation loss of 0.5059 at epoch 12. Early stopping triggered at epoch 17. The slower convergence and higher parameter count (2.43M vs. 374K for CNN) suggest recurrent processing of full 4,259-length sequences is less efficient than local convolutional feature extraction for this task.

---

## 7. Results

### Overall Performance Comparison

| Model | Accuracy | Macro F1 | Macro Precision | Macro Recall |
|-------|----------|----------|-----------------|--------------|
| MLP Baseline | 72.81% | 0.6095 | 0.6025 | 0.7410 |
| **1D-CNN** | **98.44%** | **0.7421** | **0.7409** | **0.7435** |
| BiLSTM | 93.53% | 0.6468 | 0.6173 | 0.7035 |
| DNABERT (transfer learning) | 41.44% | 0.1036 | 0.2500 | 0.1465 |

The 1D-CNN achieves the best overall accuracy (98.44%), representing a **25.63 percentage point improvement** over the MLP baseline and a **4.91 percentage point improvement** over the BiLSTM.

**DNABERT Transfer Learning Limitations:** We attempted to fine-tune DNABERT (a pre-trained genomic language model) for subtype classification. However, DNABERT relies on 6-mer tokenization and limits input lengths to 512 tokens. With *pol* gene sequences ranging from 2,500–4,200 bp, only the first ~517 bp of each sequence could be utilized. Consequently, the model achieved only 41.44% accuracy (and effectively 0 F1-score on minority classes), demonstrating that extreme sequence truncation discards the critical distal motifs necessary for accurate HIV-1 subtype discrimination. This highlights the necessity for our custom CNN architecture that can process full-length sequences via global average pooling.

### Per-Class Performance (1D-CNN — Best Model)

| Subtype | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| A | 1.0000 | 0.9762 | 0.9880 | 42 |
| B | 1.0000 | 0.9977 | 0.9989 | 436 |
| C | 0.9638 | 1.0000 | 0.9816 | 346 |
| D | 0.0000 | 0.0000 | 0.0000 | 11 |

The CNN achieves near-perfect classification for subtypes A, B, and C, but **completely fails on subtype D** (zero precision, recall, and F1). This is directly attributable to the extreme class imbalance — with only 53 training samples for subtype D, the model cannot learn sufficiently robust features despite the 18.38× class weight.

### Key Observations

1. **CNN dominates**: The 98.44% accuracy demonstrates that local convolutional motifs are highly discriminative for HIV-1 subtype classification, consistent with the biology — subtype-defining mutations tend to cluster in specific *pol* gene regions.

2. **BiLSTM underperforms CNN**: Despite modeling long-range dependencies with 6.5× more parameters, the BiLSTM (93.53%) cannot match the CNN. This suggests subtype-discriminative signal is primarily local (motif-level) rather than dependent on long-range sequence context.

3. **Subtype D is a failure case**: All three models struggle with subtype D. The MLP actually performs best on D (F1=0.52) because its simpler decision boundaries avoid overfitting to the majority classes. This highlights a fundamental limitation of deep learning with very small minority classes.

4. **Macro F1 vs. Accuracy discrepancy**: The CNN's macro F1 (0.7421) is much lower than its accuracy (98.44%) precisely because subtype D's zero F1 drags down the macro average. The weighted F1 (0.9780) more accurately reflects the model's practical performance.

---

## 8. Error Analysis

The 1D-CNN misclassified only **13 out of 835** test sequences (1.6% error rate). Detailed error analysis reveals:

| True Label | Predicted Label | Count |
|------------|----------------|-------|
| D → C | 11 | Most frequent confusion |
| A → C | 1 | Rare |
| B → C | 1 | Rare |

### Key Findings

1. **D → C confusion dominates**: All 11 subtype D test sequences were misclassified as subtype C. This is biologically plausible: subtypes C and D share significant phylogenetic similarity in the *pol* gene, particularly in the reverse transcriptase and integrase regions.

2. **High-confidence errors**: The mean prediction confidence for misclassified sequences was 0.709 (median 0.786), indicating these are not low-confidence borderline cases but systematic misclassifications. The model has learned features that genuinely do not distinguish D from C at the available sample size.

3. **Geographic confound**: Subtypes C and D co-circulate in East Africa. Some sequences labeled as subtype D may contain C-like *pol* regions due to recombination events that are not captured by pure subtype labels, further complicating classification.

4. **Sample size is the bottleneck**: With only 53 training samples for subtype D, the model has insufficient examples to learn D-specific motifs that distinguish it from the closely related subtype C (1,612 training samples — a 30:1 ratio).

---

## 9. Ethical Considerations

### Data Privacy

All sequences used in this project are publicly available from the LANL HIV Database. Sequences are identified by GenBank accession numbers only — no patient names, clinical data, or personally identifiable information is associated with the sequences. The LANL database explicitly provides these sequences for research and educational use.

### Geographic and Sampling Bias

The dataset reflects historical biases in HIV sequencing:

- **Subtype B is overrepresented** (52.3% of our dataset) because most sequencing has been conducted in North America and Western Europe, where subtype B predominates.
- **Subtypes A and D are severely underrepresented** (5.0% and 1.3%) despite their clinical importance in East Africa.

This bias directly impacts model performance: the model achieves perfect classification for the well-sampled subtypes B and C but completely fails on the rare subtype D. Deploying such a model in East Africa — where subtype D classification matters most — would produce misleading results.

### Fairness and Deployment Risks

The Gradio deployment interface includes an explicit disclaimer that this is a research/educational tool and should not be used for clinical subtype determination. Clinical decisions should rely on validated tools (REGA, COMET, jpHMM) with established regulatory approval.

### Broader Impact

If expanded and validated, deep learning subtyping that operates on raw, unaligned sequences could enable rapid, decentralized molecular surveillance without transmitting sensitive genomic data to remote servers — an important consideration for data sovereignty in resource-limited settings. The Monte Carlo Dropout uncertainty mechanism provides an additional safety layer, flagging predictions that should be verified by validated clinical tools.

---

## 10. Limitations

1. **Severe class imbalance**: The 30:1 ratio between subtype B (2,037 training samples) and subtype D (53 training samples) prevents effective learning of D-specific features. Focal Loss with class weights mitigates but may not fully solve this problem.

2. **Only 4 subtypes**: HIV-1 has 9+ subtypes and numerous CRFs. Our model cannot classify subtypes F, G, H, J, K, or any recombinant forms. Sequences from these subtypes would be forced into one of the 4 classes, though the Monte Carlo Dropout uncertainty mechanism should flag such out-of-distribution sequences as indeterminate.

3. **Gap-stripping as proxy for unaligned data**: While we strip alignment gaps to produce variable-length sequences, the underlying data was originally downloaded from LANL in pre-aligned form. True unaligned sequences obtained directly from sequencing instruments may contain additional artifacts (e.g., varying start/end positions) not present in our gap-stripped data.

4. **Transformer context limits**: Fine-tuning DNABERT failed to generalize (41.44% accuracy) because its 512-token limit required severe truncation of the ~2,500–4,200 bp *pol* sequences. This indicates that HIV-1 subtype features are distributed throughout the gene, not localized to the first 500 bp.

5. **No explainability**: We do not implement gradient-based visualization (e.g., Grad-CAM) or attention mechanisms to identify which nucleotide positions drive classification decisions. This limits biological interpretability.

6. **Single gene region**: Classification is based solely on the *pol* gene. Using multiple gene regions (gag, env) or whole genomes could improve accuracy, especially for closely related subtypes.

7. **No recombination detection**: CRFs (Circulating Recombinant Forms) are excluded from our dataset. A practical subtyping tool must also detect recombination breakpoints.

---

## 11. Future Work

1. **Data augmentation for rare subtypes**: Synthetic sequence generation (e.g., using evolutionary models or GANs) and reverse-complement augmentation could increase subtype D training data.

2. **Multi-gene classification**: Incorporating gag and env regions alongside pol could provide complementary discriminative signals.

3. **Expand to all subtypes and CRFs**: The framework should be extended beyond the 4-class setting to handle the full diversity of HIV-1 genetic forms.

4. **Explainability**: Implementing Grad-CAM or learned attention mechanisms to identify which nucleotide positions are most discriminative for each subtype. This could reveal biologically meaningful motifs (e.g., known drug resistance positions in pol).

5. **Long-context transformers**: Utilizing long-context transformer architectures (like DNABERT-2 with FlashAttention or HyenaDNA) to process full 4,000+ bp sequences, potentially improving performance on rare subtypes through pre-trained genomic representations.

6. **True unaligned evaluation**: Testing the model on sequences obtained directly from sequencing pipelines (e.g., consensus sequences from NGS) without any prior alignment to a reference.

7. **Clinical validation**: Testing the model on prospective clinical samples, particularly from East African settings where subtype classification has direct treatment implications.

---

## 12. Conclusion

This project demonstrates that deep learning can effectively classify HIV-1 pol gene sequences into major subtypes directly from raw, unaligned nucleotide data — without requiring the multiple sequence alignment that all traditional tools (REGA, COMET, jpHMM) depend on. By stripping alignment gaps and using dynamic per-batch padding, our 1D-CNN processes variable-length sequences natively through its `AdaptiveAvgPool1d` architecture. The CNN achieves 98.44% overall accuracy — a substantial improvement over both the MLP baseline (72.81%) and the BiLSTM (93.53%). The CNN's dominance suggests that HIV-1 subtype discrimination is primarily driven by local nucleotide motifs rather than long-range sequence dependencies, which is consistent with the known biology of subtype-defining mutations in the pol gene.

The use of Focal Loss with class weights provides a more targeted approach to class imbalance than standard weighted cross-entropy, focusing training on the hard, ambiguous examples that matter most — particularly the subtype D samples that are phylogenetically close to subtype C.

However, the severe underrepresentation of subtype D (only 53 training examples) highlights a critical limitation: deep learning cannot overcome fundamental data scarcity. This underscores the importance of equitable genomic sampling — the same communities most affected by HIV-1 subtype D (East Africa) are those least represented in sequence databases.

The deployed Gradio web application incorporates Monte Carlo Dropout for uncertainty estimation, automatically flagging low-confidence predictions as indeterminate rather than forcing a potentially incorrect classification. Combined with explicit disclaimers, this ensures responsible use as a research tool rather than a clinical diagnostic.

---

## 13. References

1. Solis-Reyes, S., Avino, M., Poon, A., & Kari, L. (2018). An open-source k-mer based machine learning tool for fast and accurate subtyping of HIV-1 genomes. *PLoS ONE*, 13(11), e0206409. https://doi.org/10.1371/journal.pone.0206409

2. Los Alamos National Laboratory HIV Sequence Database. https://www.hiv.lanl.gov/

3. Kameris Experiments Repository. https://github.com/stephensolis/kameris-experiments

4. Pineda-Peña, A.C., et al. (2013). Automated subtyping of HIV-1 genetic sequences for clinical and surveillance purposes: performance evaluation of the new REGA version 3 and seven other tools. *Infection, Genetics and Evolution*, 19, 337-348.

5. Struck, D., et al. (2014). COMET: adaptive context-based modeling for ultrafast HIV-1 subtype identification. *Nucleic Acids Research*, 42(18), e144.

6. Alipanahi, B., Delong, A., Weirauch, M.T., & Frey, B.J. (2015). Predicting the sequence specificities of DNA- and RNA-binding proteins by deep learning. *Nature Biotechnology*, 33(8), 831-838.

7. Zhou, J., & Troyanskaya, O.G. (2015). Predicting effects of noncoding variants with deep learning-based sequence model. *Nature Methods*, 12(10), 931-934.

8. Paszke, A., et al. (2019). PyTorch: An imperative style, high-performance deep learning library. *NeurIPS*, 8024-8035.

9. Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735-1780.

10. LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. *Nature*, 521(7553), 436-444.

11. Lin, T.Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal loss for dense object detection. *IEEE International Conference on Computer Vision (ICCV)*, 2980-2988.

12. Ji, Y., et al. (2021). DNABERT: pre-trained Bidirectional Encoder Representations from Transformers model for DNA-language in genome. *Bioinformatics*, 37(15), 2112–2120.

13. Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian approximation: Representing model uncertainty in deep learning. *ICML*, 1050-1059.
