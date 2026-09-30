# IC4SDMacroWood: An Authenticated Macroscopic Transverse-Section Image Benchmark for Fabaceae Hardwood Identification

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14892180.svg)](https://doi.org/10.5281/zenodo.14892180)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python: >=3.8](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)

**IC4SDMacroWood** is an authenticated, standardized macroscopic transverse-section (end-grain) wood image benchmark curated for fine-grained computer vision identification of commercially significant tropical hardwoods and international timber trade compliance under **CITES Appendix II**. 

The dataset was curated and validated by the **Intelligent Computing for Sustainable Development Laboratory (IC4SD)**, Posts and Telecommunications Institute of Technology (PTIT), Hanoi, Vietnam, in collaboration with accredited forestry taxonomists and timber anatomy specialists.

---

## 1. Key Benchmark Highlights

- **Standardized Captures**: Exactly **6,414 cross-sectional captures** ($224 \times 224$ pixels, calibrated spatial resolution of $12.0\,\mu\text{m/pixel}$, corresponding to an immutable $2.69 \times 2.69\,\text{mm}$ physical field of view per non-overlapping cropped tile, $50\times$ nominal display magnification).
- **Taxonomic Coverage**: **19 tropical hardwood species** across **6 botanical genera** in the legume family **Fabaceae** (*Afzelia*, *Dalbergia*, *Guibourtia*, *Peltogyne*, *Pterocarpus*, *Sindora*).
- **Forensic Conservation Utility**: **10 taxa** strictly regulated under **CITES Appendix II** ($52.6\%$ of species) and **4 high-value commercial timber species** native to Vietnam (*Dalbergia cochinchinensis*, *D. tonkinensis*, *Sindora cochinchinensis*, *S. tonkinensis*).
- **Physical Entity Provenance**: All captures are traceable to **147 verified physical wood blocks** ($|\mathcal{G}_c|$), verified by two independent certified wood anatomists ($\kappa = 0.985$, observed agreement $P_o = 98.65\%$).
- **Optical Standardization**: Prepared via orthogonal slicing, progressive silicon-carbide polishing (P120--P600), de-dusting with dry air jets ($6\,\text{bar}$), and imaged under calibrated daylight-balanced ($5600\,\text{K}$) diffuse circular LED lighting ($4500\,\text{lux}$, $f/8.0$, working distance $15\,\text{cm}$).
- **Two-Tier Benchmark Partitioning**: Offers both the **Canonical Split** ($N_{\text{train}}=3,959$, $N_{\text{val}}=1,265$, $N_{\text{test}}=1,190$) and the **Strict Specimen-Disjoint Split** (Zero Leakage for all 17 multi-specimen taxa, $\text{SLR}=0.0\%$), maintaining **$100\%$ Class Coverage Rate** ($\text{CCR} = 100.0\%$) and **zero cross-split duplicate captures** verified via bitwise 256-bit SHA-256 cryptographic hashing ($\text{SHA-256 Overlap} = 0$).
- **Turnkey Machine Learning Reproducibility**: Includes trained ConvNeXt-Tiny classification baseline checkpoints, evaluation metrics, and an end-to-end walkthrough demonstration code suite.

---

## 2. Repository Layout

```
out/
├── classification_output/       # Supervised classification baseline logs, predictions & checkpoints
│   ├── classification_results_focal.json   # Quantitative test metrics (Accuracy, Macro-F1, Per-class)
│   ├── convnext_tiny_focal_best.pth        # Best ConvNeXt-Tiny model weights under Focal Loss (111 MB)
│   └── raw_predictions_focal.json          # Per-sample test predictions, ground-truth & probabilities
├── code/                        # Reproducible scripts, demonstration notebook & runtime environment
│   ├── quickstart_demo.py                  # Standalone verification and baseline evaluation script
│   ├── quickstart_demo.ipynb               # Interactive Jupyter notebook for rapid data exploration
│   └── requirements.txt                    # Minimal Python package dependencies
├── images.zip                   # Compressed archive containing 6,414 standardized macroscopic RGB images
├── leakage_audit/               # Cross-split specimen overlap and cryptographic hash audit summary
│   └── audit_summary.json                  # Audit record confirming CCR = 100.0% & SLR = 30.6%
├── metadata/                    # Master catalog, structured mappings & release manifests
│   ├── anatomical_features.csv             # Structured macroscopic anatomical descriptors for multimodal tasks (VQA, XAI)
│   ├── label_map.json                      # Mapping between class IDs, scientific names & vernaculars
│   ├── metadata.csv                        # Master metadata (image paths, labels, specimens, QC scores)
│   └── release_manifest.csv                # Release manifest with SHA-256 checksums and file sizes
├── splits/                      # Governed benchmark partitioning manifests
│   └── split_canonical.csv                 # Canonical partition assignment manifest (Train/Val/Test)
├── LICENSE                      # Creative Commons Attribution 4.0 International (CC BY 4.0)
└── README.md                    # Comprehensive dataset documentation and usage instructions
```

---

## 3. Taxonomic Composition and Specimen Inventory

| # | Botanical Genus | Scientific Binomial | Vietnamese Vernacular | Common / Trade Name | CITES Appendix | Physical Specimens (Cohort) | Total Images |
| -: | :--- | :--- | :--- | :--- | :---: | :-: | -: |
| 1 | *Afzelia* | *Afzelia africana* | Gõ Douse (Doussié) | African Doussié | App. II | 4 | 241 |
| 2 | *Afzelia* | *Afzelia bella* | Papao-Nua / Gỗ Gõ | Bella Doussié | App. II | 10 | 400 |
| 3 | *Afzelia* | *Afzelia pachyloba* | Gõ Pachy | White Doussié | App. II* | 5 | 116 |
| 4 | *Afzelia* | *Afzelia quanzensis* | Gõ Quanzensis | Pod Mahogany | App. II | 8 | 369 |
| 5 | *Dalbergia* | *Dalbergia cochinchinensis* | Trắc (Rosewood) | Siam Rosewood | App. II (Native / High-value VN) | 1 | 354 |
| 6 | *Dalbergia* | *Dalbergia melanoxylon* | Trắc châu Phi | African Blackwood | App. II | 10 | 291 |
| 7 | *Dalbergia* | *Dalbergia oliveri* | Cẩm lai | Burmese Rosewood | App. II | 10 | 316 |
| 8 | *Dalbergia* | *Dalbergia rimosa* | Trắc dây | Rimose Rosewood | App. II | 10 | 300 |
| 9 | *Dalbergia* | *Dalbergia tonkinensis* | Sưa | Vietnamese Rosewood | App. II (Native / High-value VN) | 10 | 325 |
| 10 | *Guibourtia* | *Guibourtia arnoldiana* | Gỗ Muntenye | Mutenye / Benge | Non-CITES | 10 | 323 |
| 11 | *Guibourtia* | *Guibourtia coleosperma* | Mussivi / Hương đá | Rhodesian Copalwood | Non-CITES | 2 | 360 |
| 12 | *Guibourtia* | *Guibourtia ehie* | Hyedua | Ovangkol / Shedua | Non-CITES | 10 | 400 |
| 13 | *Peltogyne* | *Peltogyne pubescens* | Hương tím nam mỹ | Purpleheart | Non-CITES | 10 | 371 |
| 14 | *Pterocarpus* | *Pterocarpus erinaceus* | Hương vân tây phi | African Barwood / Kosso | App. II | 10 | 336 |
| 15 | *Pterocarpus* | *Pterocarpus indicus* | Hương mắt chim | Narra / Amboyna | Non-CITES | 10 | 312 |
| 16 | *Pterocarpus* | *Pterocarpus macrocarpus* | Hương quả to | Burma Padauk | Non-CITES | 8 | 431 |
| 17 | *Pterocarpus* | *Pterocarpus soyauxii* | Padauk | African Padauk | Non-CITES | 6 | 486 |
| 18 | *Sindora* | *Sindora cochinchinensis* | Gụ | Sindora / Sepetir | Non-CITES (Native / High-value VN) | 4 | 352 |
| 19 | *Sindora* | *Sindora tonkinensis* | Gụ lau | Tonkin Sepetir | Non-CITES (Native / High-value VN) | 10 | 331 |
| **Total** | **6 Genera** | **19 Species** | — | — | **10 CITES App. II (4 High-Value VN)** | **148 Blocks** | **6,414** |

*\*Note: Afzelia pachyloba constitutes a statistical minority class ($N=116$) due to reference specimen scarcity, while being legally regulated under CITES Appendix II.*

### 3.1 Structured Macroscopic Anatomical Descriptors for Multimodal AI

To empower research beyond standard unimodal classification, IC4SDMacroWood incorporates a standardized catalog of macroscopic anatomical traits (`metadata/anatomical_features.csv`) curated following International Association of Wood Anatomists (IAWA) standards. These structured descriptors characterize 14 diagnostic botanical and physical axes:
- **Sapwood/Heartwood Demarcation & Chemomorphology**: Heartwood coloration, growth ring distinctness, pigment striping, and diagnostic volatile aromas.
- **Vascular Architecture**: Vessel porosity (diffuse vs. semi-ring-porous), vessel grouping (solitary vs. radial multiples), pore size hierarchy, tyloses occlusion, and inorganic/gum deposits (white calcite vs. dark polyphenols).
- **Parenchyma-Ray Topography**: Paratracheal parenchyma sheaths (aliform, confluent, banded), scalariform/reticulate networks with rays, continuous marginal bands containing axial resin canals, and storied ray arrangements.
- **Physical Traits**: Air-dry density categories and mechanical hardness.

**Emerging Multimodal Applications**:
1. **Visual Question Answering (VQA)**: Benchmark domain-specific multimodal LLMs on forensic xylotomy queries (e.g., *"Does this cross-section exhibit lozenge-aliform parenchyma and distinct dark gum occlusions?"*).
2. **Semantic Attribute & Zero-Shot Classification**: Guide fine-grained identification via interpretable morphological traits rather than opaque integer class labels.
3. **Cross-Modal Retrieval**: Enable bidirectional text-to-image and image-to-text queries linking forensic dichotomous anatomical keys with optical micrographs.
4. **Explainable AI (XAI)**: Validate whether deep vision model saliency maps (Grad-CAM, attention heads) align faithfully with true diagnostic anatomical structures.

---

## 4. Governed Partition Manifest

The benchmark is partitioned into three standardized, audited subsets:

$$\mathcal{D} = \mathcal{D}_{\text{train}} \cup \mathcal{D}_{\text{val}} \cup \mathcal{D}_{\text{test}}$$

- **Training Split**: $N_{\text{train}} = 3,959$ images ($61.72\%$)
- **Validation Split**: $N_{\text{val}} = 1,265$ images ($19.72\%$)
- **Test Split**: $N_{\text{test}} = 1,190$ images ($18.55\%$)

### Multi-Objective Governance & Specimen Allocation Trade-Off

In authentic reference xylaria, physical specimens for rare and protected taxa are strictly scarce ($|\mathcal{G}_c| \le 10$, with single-specimen bottlenecks such as 1 block for *Dalbergia cochinchinensis*, 2 for *Guibourtia coleosperma*, and 4 for *Afzelia africana*). Under such constraints, dogmatic three-way whole-block isolation mathematically results in minority-class starvation ($\text{CCR} < 100\%$), preventing forensic validation on critical CITES species. 

To resolve this challenge rigorously, IC4SDMacroWood releases a **two-tier partition architecture** evaluated through a **Three-Tier Contamination Audit**:
1. **$100.0\%$ Class Coverage Rate ($\text{CCR}$)**: Exactly 19/19 species are guaranteed across all three splits.
2. **Two-Tier Split Architecture**:
   - **Canonical Partition (Governed Legacy Split)**: Maintains 100% CCR while bounding specimen sharing ($\text{SLR} = 30.6\%$).
   - **Strict Specimen-Disjoint Benchmark**: Enforces 100% zero-specimen leakage ($\text{SLR} = 0.0\%$) across all 17 multi-specimen taxa.
3. **Three-Tier Cross-Split Contamination Audit**:
   - **Tier 1: Bitwise Cryptographic Auditing (SHA-256)**: Zero bitwise exact duplicates between any partitions ($\text{SHA-256 Overlap} = 0$).
   - **Tier 2: Perceptual Near-Duplicate Auditing (dHash & pHash)**: 64-bit Difference Hash (dHash) and DCT-based Perceptual Hash (pHash) screen for overlapping spatial crops and adjacent cuts. While the canonical split exhibits perceptual proximity due to block sharing, the Specimen-Disjoint split drives cross-specimen near-duplicates ($H \le 4$) to strictly 0.
   - **Tier 3: Deep Feature Representation Audit (ConvNeXt-Tiny Cosine Similarity)**: Audits the maximum cosine similarity distribution on $L_2$-normalized representations to flag potential semantic shortcuts ($\cos \theta \ge 0.95$).

---

## 5. Technical Validation Baselines

### Baseline 1: Supervised Classification under Focal Loss

Trained using **ConvNeXt-Tiny** with Multiclass Focal Loss ($\alpha=0.25, \gamma=2.0$) to counteract natural class imbalance:

- **Top-1 Test Accuracy**: **$90.42\%$** (1,076 / 1,190 correct test captures)
- **Macro-Averaged Precision**: $91.67\%$
- **Macro-Averaged Recall**: $88.18\%$
- **Macro-Averaged F1-Score**: **$86.80\%$**
- **Weighted-Averaged F1-Score**: **$88.82\%$**
- Residual confusion is strictly confined to congeneric sister species (e.g., *Dalbergia*, *Pterocarpus*). Detailed logs and per-class metrics are cataloged in `classification_output/classification_results_focal.json`.

---

## 6. Quick Start & Reproducibility Guide

### Environment Setup

Ensure Python 3.8+ is installed. Install the minimal dependencies:

```bash
pip install -r code/requirements.txt
```

### Running the Verification and Demonstration Script

Execute the standalone demonstration script:

```bash
python code/quickstart_demo.py
```

The script runs a verification pipeline in seconds:
1. Validates repository directory structure and file integrity.
2. Checks master metadata schema (`metadata.csv`) and label mappings (`label_map.json`).
3. Audits governed partition integrity ($\text{CCR} = 100.0\%$, $\text{SHA-256 Overlap} = 0$).
4. Evaluates ConvNeXt-Tiny classification baseline predictions and per-class metrics.
5. Simulates open-set specimen verification and top-$k$ forensic timber retrieval.
6. Generates a reproducible benchmark verification summary.

### Interactive Jupyter Notebook

For an interactive, visual walkthrough with plots and confusion matrix exploration:

```bash
jupyter notebook code/quickstart_demo.ipynb
```

### Minimal Python Usage Example

```python
import pandas as pd
from pathlib import Path

# Load master metadata, canonical partition manifest, and anatomical descriptors
meta_df = pd.read_csv("metadata/metadata.csv")
anatomy_df = pd.read_csv("metadata/anatomical_features.csv")

# Merge image captures with structured IAWA macroscopic descriptors
df = pd.merge(meta_df, anatomy_df, left_on="class_name", right_on="species", how="left")

print(f"Total authentic captures: {len(df):,}")
print(f"Hardwood species represented: {df['class_name'].nunique()} / 19")
cites_taxa = df[df['cites_status_x'].str.contains('CITES App', na=False)]['class_name'].nunique()
print(f"CITES Appendix II taxa: {cites_taxa} / 19")
print(f"Partition breakdown:\n{df['split'].value_counts()}")
print(f"Anatomical features available for VQA/XAI: {list(anatomy_df.columns[3:])}")
```

---

## 7. Citation & Attribution

If you utilize the IC4SDMacroWood benchmark, metadata, pre-computed representations, or baseline code in your research, please cite our open-access Data in Brief descriptor:

```bibtex
@article{ic4sdmacrowood2026,
  title     = {{IC4SDMacroWood}: An Authenticated Macroscopic Transverse-Section Image Benchmark for Fabaceae Hardwood Identification and {CITES} Trade Compliance},
  author    = {Le, Viet-Anh and Nguyen-Trong, Khanh},
  journal   = {Data in Brief},
  year      = {2026},
  publisher = {Elsevier},
  doi       = {10.5281/zenodo.14892180}
}
```

---

## 8. License & Terms of Use

All image captures, metadata tables, partition manifests, model weights, and code scripts in this benchmark are openly distributed under the **Creative Commons Attribution 4.0 International License (CC BY 4.0)**. 

See the [`LICENSE`](LICENSE) file for complete legal terms.

---

## 9. Contact & Support

For scientific inquiries, taxonomic verification details, or dataset feedback, please contact:
- **Intelligent Computing for Sustainable Development Laboratory (IC4SD)**
- Posts and Telecommunications Institute of Technology (PTIT), Hanoi, Vietnam
- **Data Archive**: [https://doi.org/10.5281/zenodo.14892180](https://doi.org/10.5281/zenodo.14892180)
