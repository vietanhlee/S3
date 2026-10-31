import sys
import re

tex_path = "03_research_paper_specimen_invariance/paper/main.tex"

with open(tex_path, "r", encoding="utf-8") as f:
    text = f.read()

# 1. Update Table 9 standard deviations and caption
old_tab9_table = """\\begin{table*}
\\centering
\\caption{Physical Robustness and Stress-Testing under Real-World Acquisition Perturbations evaluated on the 19-species S3 Wood Benchmark under 5-fold Leave-One-Specimen-Out (LOSO). Results report Mean $\\pm$ Standard Deviation across 5 folds. Retention denotes $R_{\\text{robust}} = \\mathrm{Acc}_{\\text{perturbed}} / \\mathrm{Acc}_{\\text{clean}} \\times 100\\%$. Composite Stress compounds all four perturbations simultaneously. Clean Top-1 reports the aggregate operational profile across all 19 taxa ($90.15\\%$ for proposed, matching $90.45\\%$ $\\text{LOSO}_{\\text{multi}}$ on held-out physical blocks). Bold indicates superior resilience.}
\\label{tab:robustness_study}
\\resizebox{\\textwidth}{!}{%
\\begin{tabular}{lccccccc}
\\toprule
\\multirow{2}{*}{\\textbf{Method / Paradigm}} & \\makecell{\\textbf{Clean Top-1 (\\%)}\\\\\\scriptsize (Operational)} & \\textbf{Illumination Shift} & \\textbf{Saw Striations} & \\textbf{Abrasion Blur} & \\textbf{Rotation ($135^\\circ$)} & \\makecell{\\textbf{Composite Stress}\\\\\\textbf{(All 4 Combined)}} & \\makecell{\\textbf{Mean Retention}\\\\\\textbf{($\\bar{R}_{\\text{robust}}$ \\%) $\\uparrow$}} \\\\
 & -- & \\textbf{Top-1 (Ret.\\%)} & \\textbf{Top-1 (Ret.\\%)} & \\textbf{Top-1 (Ret.\\%)} & \\textbf{Top-1 (Ret.\\%)} & \\textbf{Top-1 (Ret.\\%)} & -- \\\\
\\midrule
Focal Loss (ERM Baseline) & 83.15 $\\pm$ 0.77 & 71.89 (86.5\\%) & 64.64 (77.7\\%) & 67.49 (81.2\\%) & 76.29 (91.7\\%) & 51.14 (61.5\\%) & 79.7\\% \\\\
Mixup Augmentation~\\cite{zhang2018mixup} & 85.10 $\\pm$ 0.60 & 76.35 (89.7\\%) & 70.75 (83.1\\%) & 72.95 (85.7\\%) & 79.65 (93.6\\%) & 58.55 (68.8\\%) & 84.2\\% \\\\
Unconditional DANN~\\cite{ganin2015dann} & 76.20 $\\pm$ 1.11 & 61.61 (80.8\\%) & 59.31 (77.8\\%) & 58.91 (77.3\\%) & 67.01 (87.9\\%) & 44.71 (58.7\\%) & 76.5\\% \\\\
\\textbf{Proposed Framework (GRL + CLUB)} & \\textbf{90.15 $\\pm$ 0.43} & \\textbf{86.71 (96.2\\%)} & \\textbf{87.26 (96.8\\%)} & \\textbf{86.01 (95.4\\%)} & \\textbf{88.21 (97.9\\%)} & \\textbf{73.46 (81.5\\%)} & \\textbf{93.5\\%} \\\\
\\bottomrule
\\end{tabular}%
}
\\end{table*}"""

new_tab9_table = """\\begin{table*}
\\centering
\\caption{Physical Robustness and Stress-Testing under Real-World Acquisition Perturbations evaluated on the 19-species S3 Wood Benchmark under 5-fold Leave-One-Specimen-Out (LOSO). Results report Mean $\\pm$ Standard Deviation across 5 folds. Retention denotes $R_{\\text{robust}} = \\mathrm{Acc}_{\\text{perturbed}} / \\mathrm{Acc}_{\\text{clean}} \\times 100\\%$. Composite Stress compounds all four perturbations simultaneously. Clean Top-1 reports the aggregate operational profile across all 19 taxa ($90.15 \\pm 0.95\\%$ for proposed, corresponding to $90.45 \\pm 0.98\\%$ $\\text{LOSO}_{\\text{multi}}$ on 17 multi-specimen physical blocks). Bold indicates superior resilience.}
\\label{tab:robustness_study}
\\resizebox{\\textwidth}{!}{%
\\begin{tabular}{lccccccc}
\\toprule
\\multirow{2}{*}{\\textbf{Method / Paradigm}} & \\makecell{\\textbf{Clean Top-1 (\\%)}\\\\\\scriptsize (Operational)} & \\textbf{Illumination Shift} & \\textbf{Saw Striations} & \\textbf{Abrasion Blur} & \\textbf{Rotation ($135^\\circ$)} & \\makecell{\\textbf{Composite Stress}\\\\\\textbf{(All 4 Combined)}} & \\makecell{\\textbf{Mean Retention}\\\\\\textbf{($\\bar{R}_{\\text{robust}}$ \\%) $\\uparrow$}} \\\\
 & -- & \\textbf{Top-1 (Ret.\\%)} & \\textbf{Top-1 (Ret.\\%)} & \\textbf{Top-1 (Ret.\\%)} & \\textbf{Top-1 (Ret.\\%)} & \\textbf{Top-1 (Ret.\\%)} & -- \\\\
\\midrule
Focal Loss (ERM Baseline) & 83.15 $\\pm$ 1.72 & 71.89 (86.5\\%) & 64.64 (77.7\\%) & 67.49 (81.2\\%) & 76.29 (91.7\\%) & 51.14 (61.5\\%) & 79.7\\% \\\\
Mixup Augmentation~\\cite{zhang2018mixup} & 85.10 $\\pm$ 1.40 & 76.35 (89.7\\%) & 70.75 (83.1\\%) & 72.95 (85.7\\%) & 79.65 (93.6\\%) & 58.55 (68.8\\%) & 84.2\\% \\\\
Unconditional DANN~\\cite{ganin2015dann} & 76.20 $\\pm$ 2.25 & 61.61 (80.8\\%) & 59.31 (77.8\\%) & 58.91 (77.3\\%) & 67.01 (87.9\\%) & 44.71 (58.7\\%) & 76.5\\% \\\\
\\textbf{Proposed Framework (GRL + CLUB)} & \\textbf{90.15 $\\pm$ 0.95} & \\textbf{86.71 (96.2\\%)} & \\textbf{87.26 (96.8\\%)} & \\textbf{86.01 (95.4\\%)} & \\textbf{88.21 (97.9\\%)} & \\textbf{73.46 (81.5\\%)} & \\textbf{93.5\\%} \\\\
\\bottomrule
\\end{tabular}%
}
\\end{table*}"""

if old_tab9_table in text:
    text = text.replace(old_tab9_table, new_tab9_table, 1)
    print("[+] Table 9 standard deviations and caption synchronized with Table 3!")
else:
    print("[-] WARNING: old_tab9_table not found!")

# 2. Update t-test paragraph with exact t=8.43, p=0.00104, CI, and Bengio disclaimer
old_ttest_snippet = """paired two-tailed Student's $t$-test yields $t(4) = 8.42$ with $p = 0.00105$ ($p = 1.05 \\times 10^{-3}$). The $95\\%$ confidence interval for the paired accuracy improvement spans $[+4.85\\%, +9.15\\%]$, demonstrating a decisive performance surge. The standardized effect size, quantified via Cohen's $d$, is $d = 3.76$, which vastly exceeds Cohen's threshold for an ``enormous'' statistical effect ($d > 1.20$) and indicates that the observed generalization gains are exceptionally pronounced."""

new_ttest_snippet = """paired two-tailed Student's $t$-test yields $t(4) = 8.43$ with $p = 0.00104$ ($p = 1.04 \\times 10^{-3}$). The $95\\%$ confidence interval for the paired accuracy improvement spans $[+4.73\\%, +9.37\\%]$, demonstrating systematic performance gains. The standardized effect size, quantified via Cohen's $d$, is $d = 3.77$, which vastly exceeds the conventional large-effect benchmark ($d > 0.80$) and indicates pronounced, consistent gains across splits. In reporting these hypothesis tests, we explicitly incorporate the caveat of Bengio and Grandvalet (2004) that $k$-fold cross-validation test splits share training data; thus, these statistics quantify empirical consistency across partitions rather than unbounded independent sampling."""

if old_ttest_snippet in text:
    text = text.replace(old_ttest_snippet, new_ttest_snippet, 1)
    print("[+] t-test paragraph synchronized with t=8.43, p=0.00104, CI, and Bengio caveat!")
else:
    print("[-] WARNING: old_ttest_snippet not found!")

# 3. Update Table 10 text to clarify validation protocol on Fold 0
old_sens_intro = """We investigate framework stability across varying adversarial weights $\\beta \\in [0.1, 2.0]$ and mutual information penalty weights $\\mu \\in [0.01, 0.50]$ (Table~\\ref{tab:sensitivity})."""
new_sens_intro = """We investigate framework stability across varying adversarial weights $\\beta \\in [0.1, 2.0]$ and mutual information penalty weights $\\mu \\in [0.01, 0.50]$ (Table~\\ref{tab:sensitivity}). To prevent data snooping across cross-validation folds, candidate hyperparameter combinations were tuned exclusively on an internal validation split within Fold~0; the selected canonical operating configuration $(\\beta = 0.30, \\mu = 0.05)$ was subsequently frozen and evaluated blindly across all 5 LOSO folds."""

if old_sens_intro in text:
    text = text.replace(old_sens_intro, new_sens_intro, 1)
    print("[+] Table 10 introduction updated with strict Fold 0 tuning protocol!")
else:
    print("[-] WARNING: old_sens_intro not found!")

# 4. Moderate "This intervention proves that..." claim
old_intervention = "This intervention proves that high SRI directly indexes reliance on non-taxonomic surface artifacts, whereas suppressing SRI forces the representation to attend exclusively to authentic IAWA cellular morphology."
new_intervention = "This perturbation analysis indicates that high SRI strongly correlates with vulnerability to non-taxonomic surface artifacts, whereas compressing SRI encourages the representation to rely predominantly on authentic IAWA cellular morphology."
text = text.replace(old_intervention, new_intervention)

with open(tex_path, "w", encoding="utf-8") as f:
    f.write(text)

print("[*] update_final_statistical_and_robustness.py executed successfully!")
