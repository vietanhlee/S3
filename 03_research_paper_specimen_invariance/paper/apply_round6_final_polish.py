import sys
import re

tex_path = "03_research_paper_specimen_invariance/paper/main.tex"

with open(tex_path, "r", encoding="utf-8") as f:
    text = f.read()

# Backup
with open(tex_path + ".bak_r6", "w", encoding="utf-8") as f:
    f.write(text)

# 1. Eq. (3) & Displayed Eq. (4) for class-balanced weight with explicit Nc exponent
old_cb = """\\begin{equation}
    \\mathcal{L}_{\\text{species}}(\\theta, \\phi) = -\\frac{1}{B} \\sum_{i=1}^B \\alpha_{y_i} (1 - p_{i, y_i})^\\gamma \\log(p_{i, y_i}),
    \\label{eq:loss_species}
\\end{equation}
where $p_{i, y_i} = \\frac{\\exp(\\hat{y}_{i, y_i})}{\\sum_{j=1}^C \\exp(\\hat{y}_{i, j})}$ is the softmax probability assigned to true species $y_i$, $\\gamma = 2.0$ is the focusing parameter that down-weights easy, well-classified examples, and $\\alpha_c = \\frac{1 - \\beta_{\\text{cb}}}{1 - \\beta_{\\text{cb}}^{N_c}}$ represents the class-balancing weight computed from the effective number of samples $N_c$~\\cite{cui2019} with hyperparameter $\\beta_{\\text{cb}} = 0.999$."""

new_cb = """\\begin{equation}
    \\mathcal{L}_{\\text{species}}(\\theta, \\phi) = -\\frac{1}{B} \\sum_{i=1}^B \\alpha_{y_i} (1 - p_{i, y_i})^\\gamma \\log(p_{i, y_i}),
    \\label{eq:loss_species}
\\end{equation}
where $p_{i, y_i} = \\frac{\\exp(\\hat{y}_{i, y_i})}{\\sum_{j=1}^C \\exp(\\hat{y}_{i, j})}$ is the softmax probability assigned to true taxon $y_i$, $\\gamma = 2.0$ is the focusing parameter that down-weights easy, well-classified examples, and the class-balancing weight $\\alpha_c$ is formulated explicitly following Cui et al.~\\cite{cui2019} based on the effective number of samples:
\\begin{equation}
    \\alpha_c = \\frac{1 - \\beta_{\\text{cb}}}{1 - \\beta_{\\text{cb}}^{N_c}},
    \\label{eq:cb_weight}
\\end{equation}
with hyperparameter $\\beta_{\\text{cb}} = 0.999$. In Eq.~\\eqref{eq:cb_weight}, $N_c$ denotes the total number of training image samples available for taxon $c$, and the denominator features $\\beta_{\\text{cb}}$ raised explicitly to the exponent $N_c$ (i.e., $\\beta_{\\text{cb}}^{N_c}$)."""

if old_cb in text:
    text = text.replace(old_cb, new_cb, 1)
    print("Eq. (3) & Displayed Eq. (4) updated successfully!")
else:
    print("WARNING: old_cb not found!")

# 2. Eq. (7) on Page 8: Replace \\begin{cases} with clean indicator formulation to prevent 1.1.1.1... glyphs
old_mask = """\\begin{equation}
    M_c = \\mathbb{I}[|\\mathcal{G}_c| \\ge 2] = \\begin{cases}
        1, & \\text{if } |\\mathcal{G}_c| \\ge 2, \\\\
        0, & \\text{if } |\\mathcal{G}_c| = 1.
    \\end{cases}
    \\label{eq:mask}
\\end{equation}"""

new_mask = """\\begin{equation}
    M_c = \\mathbf{1}(|\\mathcal{G}_c| \\ge 2) = 
    \\begin{cases}
        1, & \\text{if } |\\mathcal{G}_c| \\ge 2, \\\\
        0, & \\text{if } |\\mathcal{G}_c| = 1,
    \\end{cases}
    \\label{eq:mask}
\\end{equation}"""

if old_mask in text:
    text = text.replace(old_mask, new_mask, 1)
    print("Eq. (7) mask updated successfully!")
else:
    print("WARNING: old_mask not found!")

# 3. Algorithm 1, Line 2: Ensure clean indicator notation
old_alg_mask = "\\State Compute species validity mask $M_c = \\mathbb{I}[|\\mathcal{G}_c| \\ge 2]$ for each $c \\in \\{1,\\dots,C\\}$."
new_alg_mask = "\\State Compute binary species validity mask $M_c = \\mathbf{1}(|\\mathcal{G}_c| \\ge 2)$ for each taxon $c \\in \\{1,\\dots,C\\}$."

if old_alg_mask in text:
    text = text.replace(old_alg_mask, new_alg_mask, 1)
    print("Algorithm 1 line 2 updated successfully!")
else:
    print("WARNING: old_alg_mask not found!")

# 4. Table 4: Clean up typography and ensure pristine citations & labels
old_forestry_table = """\\begin{table*}[t]
\\centering
\\footnotesize
\\renewcommand{\\arraystretch}{1.05}
\\caption{Direct Benchmark against Published Macroscopic Timber Forensics Pipelines on the S3 Wood Benchmark under identical 5-Fold Round-Robin Leave-One-Specimen-Out (LOSO) Cross-Validation. Comparison between canonical forestry architectures and the proposed Specimen-Invariant framework across ResNet-50, ConvNeXt-Tiny, and Swin-T. $\\text{LOSO}_{\\text{multi}}$ reports out-of-specimen accuracy on the 17 multi-specimen taxa evaluated with strictly disjoint physical block holdouts ($\\mathrm{SLR} \\equiv 0.0\\%$). Overall Top-1 reflects the aggregate operational profile across all 19 taxa. Bold denotes superior performance.}
\\label{tab:forestry_sota_benchmark}
\\resizebox{\\textwidth}{!}{%
\\begin{tabular}{llcccccc}
\\toprule
\\textbf{Forestry Pipeline / Study} & \\textbf{Base Neural Backbone} & \\textbf{Training Regime \\& Loss} & \\makecell{\\textbf{LOSO$_{\\text{multi}}$}\\\\\\textbf{Top-1 (\\%)}\\\\\\scriptsize (Primary)} & \\makecell{\\textbf{Overall}\\\\\\textbf{Top-1 (\\%)}\\\\\\scriptsize (Operational)} & \\makecell{\\textbf{Macro-F1}\\\\\\textbf{(\\%)}\\\\\\scriptsize (Overall)} & \\makecell{\\textbf{SRI$_{\\text{lin}}$}\\\\\\textbf{$\\downarrow$}} & \\makecell{\\textbf{ECE (\\%)}\\\\\\textbf{$\\downarrow$}} \\\\
\\midrule
XyloTron Pipeline (USDA FPL)~\\cite{ravindran2020,wiedenhoeft2020xylotron} & ResNet-34 & Cross-Entropy (ERM) & 82.40 $\\pm$ 2.10 & 82.15 $\\pm$ 2.05 & 79.80 $\\pm$ 2.25 & 0.868 & 14.15 \\\\
Macroscopic ResNet Pipeline (Wu et al.~\\cite{wu2021}) & ResNet-50 & Cross-Entropy (ERM) & 82.10 $\\pm$ 2.05 & 81.85 $\\pm$ 1.95 & 79.50 $\\pm$ 2.15 & 0.875 & 13.80 \\\\
Deep Residual Wood Pipeline (Fabijanska et al.~\\cite{fabijanska2021}) & ResNet-50 & Cross-Entropy (ERM) & 82.55 $\\pm$ 1.90 & 82.30 $\\pm$ 1.85 & 80.15 $\\pm$ 2.05 & 0.860 & 13.50 \\\\
\\midrule
\\textbf{Proposed Specimen-Invariant (Ours - ResNet-50)} & ResNet-50 & GRL + vCLUB (Ours) & \\textbf{88.70 $\\pm$ 1.20} & \\textbf{88.40 $\\pm$ 1.15} & \\textbf{87.10 $\\pm$ 1.28} & \\textbf{0.142} & \\textbf{5.95} \\\\
\\textbf{Proposed Specimen-Invariant (Ours - ConvNeXt-Tiny)} & ConvNeXt-Tiny & GRL + vCLUB (Ours) & \\textbf{90.45 $\\pm$ 0.98} & \\textbf{90.15 $\\pm$ 0.95} & \\textbf{89.25 $\\pm$ 1.05} & \\textbf{0.118} & \\textbf{5.10} \\\\
\\textbf{Proposed Specimen-Invariant (Ours - Swin-T)} & Swin-T & GRL + vCLUB (Ours) & \\textbf{91.35 $\\pm$ 0.92} & \\textbf{91.05 $\\pm$ 0.88} & \\textbf{90.20 $\\pm$ 0.98} & \\textbf{0.095} & \\textbf{4.60} \\\\
\\bottomrule
\\end{tabular}%
}
\\end{table*}"""

new_forestry_table = """\\begin{table*}[t]
\\centering
\\footnotesize
\\renewcommand{\\arraystretch}{1.08}
\\caption{Direct Benchmark against Published Macroscopic Timber Forensics Pipelines (XyloTron~\\cite{ravindran2020,wiedenhoeft2020xylotron}, Wu et al.~\\cite{wu2021}, and Fabijanska et al.~\\cite{fabijanska2021}) on the S3 Wood Benchmark under identical 5-Fold Round-Robin Leave-One-Specimen-Out (LOSO) Cross-Validation. Comparison between canonical forestry architectures and the proposed Specimen-Invariant framework across ResNet-50, ConvNeXt-Tiny, and Swin-T. $\\text{LOSO}_{\\text{multi}}$ reports out-of-specimen accuracy on the 17 multi-specimen taxa evaluated with strictly disjoint physical block holdouts ($\\mathrm{SLR} \\equiv 0.0\\%$). Overall Top-1 reflects the aggregate operational profile across all 19 taxa. Bold denotes superior performance.}
\\label{tab:forestry_sota_benchmark}
\\resizebox{\\textwidth}{!}{%
\\begin{tabular}{llcccccc}
\\toprule
\\textbf{Published Pipeline / Approach} & \\textbf{Base Neural Backbone} & \\textbf{Training Regime \\& Loss} & \\makecell{\\textbf{LOSO$_{\\text{multi}}$}\\\\\\textbf{Top-1 (\\%)}\\\\\\scriptsize (Primary)} & \\makecell{\\textbf{Overall}\\\\\\textbf{Top-1 (\\%)}\\\\\\scriptsize (Operational)} & \\makecell{\\textbf{Macro-F1}\\\\\\textbf{(\\%)}\\\\\\scriptsize (Overall)} & \\makecell{\\textbf{SRI$_{\\text{lin}}$}\\\\\\textbf{$\\downarrow$}} & \\makecell{\\textbf{ECE (\\%)}\\\\\\textbf{$\\downarrow$}} \\\\
\\midrule
XyloTron Pipeline (USDA-FPL)~\\cite{ravindran2020,wiedenhoeft2020xylotron} & ResNet-34 & Cross-Entropy (ERM) & 82.40 $\\pm$ 2.10 & 82.15 $\\pm$ 2.05 & 79.80 $\\pm$ 2.25 & 0.868 & 14.15 \\\\
Macroscopic ResNet (Wu et al.~\\cite{wu2021}) & ResNet-50 & Cross-Entropy (ERM) & 82.10 $\\pm$ 2.05 & 81.85 $\\pm$ 1.95 & 79.50 $\\pm$ 2.15 & 0.875 & 13.80 \\\\
Deep Residual Pipeline (Fabijanska et al.~\\cite{fabijanska2021}) & ResNet-50 & Cross-Entropy (ERM) & 82.55 $\\pm$ 1.90 & 82.30 $\\pm$ 1.85 & 80.15 $\\pm$ 2.05 & 0.860 & 13.50 \\\\
\\midrule
\\textbf{Proposed Framework (Ours: ResNet-50)} & ResNet-50 & GRL + vCLUB (Proposed) & \\textbf{88.70 $\\pm$ 1.20} & \\textbf{88.40 $\\pm$ 1.15} & \\textbf{87.10 $\\pm$ 1.28} & \\textbf{0.142} & \\textbf{5.95} \\\\
\\textbf{Proposed Framework (Ours: ConvNeXt-Tiny)} & ConvNeXt-Tiny & GRL + vCLUB (Proposed) & \\textbf{90.45 $\\pm$ 0.98} & \\textbf{90.15 $\\pm$ 0.95} & \\textbf{89.25 $\\pm$ 1.05} & \\textbf{0.118} & \\textbf{5.10} \\\\
\\textbf{Proposed Framework (Ours: Swin-T)} & Swin-T & GRL + vCLUB (Proposed) & \\textbf{91.35 $\\pm$ 0.92} & \\textbf{91.05 $\\pm$ 0.88} & \\textbf{90.20 $\\pm$ 0.98} & \\textbf{0.095} & \\textbf{4.60} \\\\
\\bottomrule
\\end{tabular}%
}
\\end{table*}"""

if old_forestry_table in text:
    text = text.replace(old_forestry_table, new_forestry_table, 1)
    print("Table 4 (Forestry SOTA Benchmark) cleaned up successfully!")
else:
    print("WARNING: old_forestry_table not found!")

# 5. Table 2: Clean CDAN row to prevent OCR confusing \\otimes with ^2
old_cdan_row = "& CDAN~\\cite{long2018cdan} & Multilinear conditioned discriminator: $\\min_\\theta \\max_D \\mathcal{L}_y - \\lambda \\mathcal{L}_d(D(z \\otimes \\hat{y}), s)$ \\\\"
new_cdan_row = "& CDAN~\\cite{long2018cdan} & Multilinear conditioned discriminator: $\\min_\\theta \\max_D \\mathcal{L}_{\\text{species}} - \\lambda \\mathcal{L}_{\\text{domain}}(D(\\mathbf{h}), s)$, where $\\mathbf{h} = z \\otimes \\hat{y}$ \\\\"

if old_cdan_row in text:
    text = text.replace(old_cdan_row, new_cdan_row, 1)
    print("Table 2 CDAN row updated successfully!")
else:
    print("WARNING: old_cdan_row not found!")

# 6. Table 7: Format Blocks ($S_c$) column to prevent unformatted text dumps from outputting "1 1 1 1..."
# In Table 7, replace single digits like "4", "10", "1" with "4 vouchers", "10 vouchers", "1 voucher" or "4 blocks", etc.
old_table7_body = """\\multirow{4}{*}{\\textit{Afzelia}} 
& \\textit{Afzelia africana} & Appendix II & 4 & 84.6 & 83.2 & 0.852 & \\textbf{91.7} & \\textbf{90.5} & \\textbf{0.114} & \\textbf{0.130} & +7.3 \\\\
& \\textit{Afzelia bella} & Appendix II & 10 & 85.0 & 84.1 & 0.835 & \\textbf{92.5} & \\textbf{91.8} & \\textbf{0.108} & \\textbf{0.125} & +7.7 \\\\
& \\textit{Afzelia pachyloba} & Appendix II & 5 & 80.2 & 78.5 & 0.880 & \\textbf{88.8} & \\textbf{87.4} & \\textbf{0.128} & \\textbf{0.145} & +8.9 \\\\
& \\textit{Afzelia quanzensis} & Appendix II & 8 & 83.7 & 82.4 & 0.840 & \\textbf{90.8} & \\textbf{89.9} & \\textbf{0.112} & \\textbf{0.128} & +7.5 \\\\
\\midrule
\\multirow{5}{*}{\\textit{Dalbergia}} 
& \\textit{Dalbergia cochinchinensis} & Appendix II & 1 & 78.4 & 76.8 & N/A & \\textbf{87.2} & \\textbf{86.1} & N/A & N/A & +9.3 \\\\
& \\textit{Dalbergia melanoxylon} & Appendix II & 10 & 88.5 & 87.6 & 0.795 & \\textbf{94.8} & \\textbf{94.2} & \\textbf{0.085} & \\textbf{0.098} & +6.6 \\\\
& \\textit{Dalbergia oliveri} & Appendix II & 10 & 81.6 & 80.2 & 0.865 & \\textbf{89.2} & \\textbf{88.5} & \\textbf{0.122} & \\textbf{0.138} & +8.3 \\\\
& \\textit{Dalbergia rimosa} & Appendix II & 10 & 86.0 & 84.8 & 0.810 & \\textbf{93.0} & \\textbf{92.4} & \\textbf{0.096} & \\textbf{0.110} & +7.6 \\\\
& \\textit{Dalbergia tonkinensis} & Appendix II & 10 & 82.5 & 81.0 & 0.855 & \\textbf{89.5} & \\textbf{88.8} & \\textbf{0.118} & \\textbf{0.135} & +7.8 \\\\
\\midrule
\\multirow{3}{*}{\\textit{Guibourtia}} 
& \\textit{Guibourtia arnoldiana} & Non-CITES & 10 & 85.8 & 84.5 & 0.825 & \\textbf{92.3} & \\textbf{91.6} & \\textbf{0.102} & \\textbf{0.118} & +7.1 \\\\
& \\textit{Guibourtia coleosperma} & Non-CITES & 1 & 80.0 & 78.2 & N/A & \\textbf{88.1} & \\textbf{87.3} & N/A & N/A & +9.1 \\\\
& \\textit{Guibourtia ehie} & Non-CITES & 10 & 87.2 & 86.0 & 0.805 & \\textbf{93.5} & \\textbf{92.8} & \\textbf{0.092} & \\textbf{0.105} & +6.8 \\\\
\\midrule
\\multirow{1}{*}{\\textit{Peltogyne}} 
& \\textit{Peltogyne pubescens} & Non-CITES & 10 & 88.0 & 86.9 & 0.790 & \\textbf{94.1} & \\textbf{93.5} & \\textbf{0.088} & \\textbf{0.102} & +6.6 \\\\
\\midrule
\\multirow{4}{*}{\\textit{Pterocarpus}} 
& \\textit{Pterocarpus erinaceus} & Appendix II & 10 & 82.1 & 80.8 & 0.860 & \\textbf{89.6} & \\textbf{88.9} & \\textbf{0.120} & \\textbf{0.136} & +8.1 \\\\
& \\textit{Pterocarpus indicus} & Non-CITES & 10 & 84.3 & 83.0 & 0.830 & \\textbf{91.2} & \\textbf{90.4} & \\textbf{0.110} & \\textbf{0.126} & +7.4 \\\\
& \\textit{Pterocarpus macrocarpus} & Non-CITES & 8 & 81.0 & 79.4 & 0.870 & \\textbf{88.5} & \\textbf{87.6} & \\textbf{0.125} & \\textbf{0.142} & +8.2 \\\\
& \\textit{Pterocarpus soyauxii} & Appendix II & 6 & 83.5 & 82.1 & 0.845 & \\textbf{90.4} & \\textbf{89.7} & \\textbf{0.115} & \\textbf{0.131} & +7.6 \\\\
\\midrule
\\multirow{2}{*}{\\textit{Sindora}} 
& \\textit{Sindora cochinchinensis} & Non-CITES & 4 & 79.5 & 77.8 & 0.895 & \\textbf{87.9} & \\textbf{86.8} & \\textbf{0.135} & \\textbf{0.152} & +9.0 \\\\
& \\textit{Sindora siamensis} & Non-CITES & 11 & 86.5 & 85.2 & 0.815 & \\textbf{93.2} & \\textbf{92.5} & \\textbf{0.098} & \\textbf{0.112} & +7.3 \\\\"""

new_table7_body = """\\multirow{4}{*}{\\textit{Afzelia}} 
& \\textit{Afzelia africana} & Appendix II & 4 vouchers & 84.6 & 83.2 & 0.852 & \\textbf{91.7} & \\textbf{90.5} & \\textbf{0.114} & \\textbf{0.130} & +7.3 \\\\
& \\textit{Afzelia bella} & Appendix II & 10 vouchers & 85.0 & 84.1 & 0.835 & \\textbf{92.5} & \\textbf{91.8} & \\textbf{0.108} & \\textbf{0.125} & +7.7 \\\\
& \\textit{Afzelia pachyloba} & Appendix II & 5 vouchers & 80.2 & 78.5 & 0.880 & \\textbf{88.8} & \\textbf{87.4} & \\textbf{0.128} & \\textbf{0.145} & +8.9 \\\\
& \\textit{Afzelia quanzensis} & Appendix II & 8 vouchers & 83.7 & 82.4 & 0.840 & \\textbf{90.8} & \\textbf{89.9} & \\textbf{0.112} & \\textbf{0.128} & +7.5 \\\\
\\midrule
\\multirow{5}{*}{\\textit{Dalbergia}} 
& \\textit{Dalbergia cochinchinensis} & Appendix II & 1 voucher & 78.4 & 76.8 & N/A & \\textbf{87.2} & \\textbf{86.1} & N/A & N/A & +9.3 \\\\
& \\textit{Dalbergia melanoxylon} & Appendix II & 10 vouchers & 88.5 & 87.6 & 0.795 & \\textbf{94.8} & \\textbf{94.2} & \\textbf{0.085} & \\textbf{0.098} & +6.6 \\\\
& \\textit{Dalbergia oliveri} & Appendix II & 10 vouchers & 81.6 & 80.2 & 0.865 & \\textbf{89.2} & \\textbf{88.5} & \\textbf{0.122} & \\textbf{0.138} & +8.3 \\\\
& \\textit{Dalbergia rimosa} & Appendix II & 10 vouchers & 86.0 & 84.8 & 0.810 & \\textbf{93.0} & \\textbf{92.4} & \\textbf{0.096} & \\textbf{0.110} & +7.6 \\\\
& \\textit{Dalbergia tonkinensis} & Appendix II & 10 vouchers & 82.5 & 81.0 & 0.855 & \\textbf{89.5} & \\textbf{88.8} & \\textbf{0.118} & \\textbf{0.135} & +7.8 \\\\
\\midrule
\\multirow{3}{*}{\\textit{Guibourtia}} 
& \\textit{Guibourtia arnoldiana} & Non-CITES & 10 vouchers & 85.8 & 84.5 & 0.825 & \\textbf{92.3} & \\textbf{91.6} & \\textbf{0.102} & \\textbf{0.118} & +7.1 \\\\
& \\textit{Guibourtia coleosperma} & Non-CITES & 1 voucher & 80.0 & 78.2 & N/A & \\textbf{88.1} & \\textbf{87.3} & N/A & N/A & +9.1 \\\\
& \\textit{Guibourtia ehie} & Non-CITES & 10 vouchers & 87.2 & 86.0 & 0.805 & \\textbf{93.5} & \\textbf{92.8} & \\textbf{0.092} & \\textbf{0.105} & +6.8 \\\\
\\midrule
\\multirow{1}{*}{\\textit{Peltogyne}} 
& \\textit{Peltogyne pubescens} & Non-CITES & 10 vouchers & 88.0 & 86.9 & 0.790 & \\textbf{94.1} & \\textbf{93.5} & \\textbf{0.088} & \\textbf{0.102} & +6.6 \\\\
\\midrule
\\multirow{4}{*}{\\textit{Pterocarpus}} 
& \\textit{Pterocarpus erinaceus} & Appendix II & 10 vouchers & 82.1 & 80.8 & 0.860 & \\textbf{89.6} & \\textbf{88.9} & \\textbf{0.120} & \\textbf{0.136} & +8.1 \\\\
& \\textit{Pterocarpus indicus} & Non-CITES & 10 vouchers & 84.3 & 83.0 & 0.830 & \\textbf{91.2} & \\textbf{90.4} & \\textbf{0.110} & \\textbf{0.126} & +7.4 \\\\
& \\textit{Pterocarpus macrocarpus} & Non-CITES & 8 vouchers & 81.0 & 79.4 & 0.870 & \\textbf{88.5} & \\textbf{87.6} & \\textbf{0.125} & \\textbf{0.142} & +8.2 \\\\
& \\textit{Pterocarpus soyauxii} & Appendix II & 6 vouchers & 83.5 & 82.1 & 0.845 & \\textbf{90.4} & \\textbf{89.7} & \\textbf{0.115} & \\textbf{0.131} & +7.6 \\\\
\\midrule
\\multirow{2}{*}{\\textit{Sindora}} 
& \\textit{Sindora cochinchinensis} & Non-CITES & 4 vouchers & 79.5 & 77.8 & 0.895 & \\textbf{87.9} & \\textbf{86.8} & \\textbf{0.135} & \\textbf{0.152} & +9.0 \\\\
& \\textit{Sindora siamensis} & Non-CITES & 11 vouchers & 86.5 & 85.2 & 0.815 & \\textbf{93.2} & \\textbf{92.5} & \\textbf{0.098} & \\textbf{0.112} & +7.3 \\\\"""

if old_table7_body in text:
    text = text.replace(old_table7_body, new_table7_body, 1)
    print("Table 7 vouchers formatting updated successfully!")
else:
    print("WARNING: old_table7_body not found!")

with open(tex_path, "w", encoding="utf-8") as f:
    f.write(text)

print("Final Polish script completed successfully!")
