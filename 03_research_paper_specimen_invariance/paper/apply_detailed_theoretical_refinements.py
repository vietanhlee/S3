import sys
import re

tex_path = "03_research_paper_specimen_invariance/paper/main.tex"

with open(tex_path, "r", encoding="utf-8") as f:
    text = f.read()

# 1. Update Proposition 1 & Proof to address reviewer's theoretical concern with DPI
old_prop1_block = """\\begin{proposition}[Semantic Collapse under Global Invariance]
\\label{prop:collapse}
Let physical specimens $S$ be strictly nested within botanical taxa $Y$ such that $S_i \\in \\mathcal{G}_{y_i}$ and $\\mathcal{G}_c \\cap \\mathcal{G}_{c'} = \\emptyset$ for all $c \\ne c'$. If an unconditioned adversarial discriminator enforces global specimen invariance $I(Z; S) \\to 0$, then for any single-specimen taxon $c^*$ where $|\\mathcal{G}_{c^*}| = 1$, the mutual information between the representation and species label indicator is strictly bounded:
\\begin{equation}
    I(Z; \\mathbb{I}[Y = c^*]) \\le I(Z; \\mathbb{I}[S = s^*]) \\le I(Z; S) \\to 0,
\\end{equation}
which implies that the representation $Z$ is stripped of all discriminative features necessary to identify species $c^*$.
\\end{proposition}

\\begin{proof}
For a singleton taxon $c^*$, there exists exactly one physical specimen block $s^* \\in \\mathcal{G}_{c^*}$. Therefore, the event $\\{Y = c^*\\}$ is completely identical to the event $\\{S = s^*\\}$: the indicator random variables satisfy $\\mathbb{I}[Y = c^*] \\equiv \\mathbb{I}[S = s^*]$. By the data processing inequality and the definition of mutual information:
\\begin{equation}
    I(Z; \\mathbb{I}[Y = c^*]) = I(Z; \\mathbb{I}[S = s^*]) \\le I(Z; S).
\\end{equation}
When the global discriminator forces $I(Z; S) \\to 0$, it directly drives $I(Z; \\mathbb{I}[Y = c^*]) \\to 0$. Consequently, the representation $Z$ becomes statistically independent of the indicator for species $c^*$, causing complete classification failure on that taxon.
\\end{proof}"""

new_prop1_block = """\\begin{proposition}[Taxonomic Confounding and Singleton Semantic Collapse under Global Invariance]
\\label{prop:collapse}
Let physical wood specimen index $S$ determine taxon label $Y$ via the biological nesting $Y = g(S)$ (each physical wood block belongs exclusively to exactly one species, $H(Y \\mid S) = 0$). By the Data Processing Inequality, the global taxonomic mutual information is bounded by specimen mutual information:
\\begin{equation}
    I(Z; Y) \\le I(Z; S).
\\end{equation}
Consequently, an unconditioned adversarial discriminator enforcing global specimen invariance $I(Z; S) \\to 0$ theoretically drives all taxonomic mutual information toward zero ($I(Z; Y) \\to 0$). In finite training regimes, models reconcile this tension on multi-specimen taxa ($|\\mathcal{G}_c| \\ge 2$) by aligning intra-class voucher distributions while preserving inter-species separation. However, for any singleton taxon $c^*$ where $|\\mathcal{G}_{c^*}| = 1$, specimen and class indicators are isomorphic on support $\\mathcal{X}_{c^*}$ ($\\mathbb{I}[Y = c^*] \\equiv \\mathbb{I}[S = s^*]$); thus, unconditioned opposition forces $I(Z; \\mathbb{I}[Y = c^*]) \\le I(Z; S) \\to 0$, destroying taxonomic features and causing catastrophic classification failure.
\\end{proposition}

\\begin{proof}
Because $Y = g(S)$ is a deterministic function of specimen identity, the Markov chain $Y \\to S \\to X \\to Z$ holds. By the Data Processing Inequality, $I(Z; Y) \\le I(Z; S)$. Under global unconditioned invariance, driving $I(Z; S) \\to 0$ imposes an asymptotic upper bound of zero on $I(Z; Y)$. For any singleton taxon $c^*$, the specimen label $s^*$ is uniquely identified with class $c^*$, yielding identical indicator random variables $\\mathbb{I}[Y = c^*] \\equiv \\mathbb{I}[S = s^*]$. Hence:
\\begin{equation}
    I(Z; \\mathbb{I}[Y = c^*]) = I(Z; \\mathbb{I}[S = s^*]) \\le I(Z; S).
\\end{equation}
Because no intra-class specimen variance exists for taxon $c^*$, the global adversarial gradient opposes the only features predictive of $s^*$, which are precisely the diagnostic anatomical features of species $c^*$. In contrast, species-conditioned adversarial alignment decomposes $I(Z; S) = I(Z; Y) + I(Z; S \\mid Y)$ and optimizes $I(Z; S \\mid Y) \\to 0$, selectively eliminating intra-taxon specimen shortcuts while unconditionally preserving $I(Z; Y) > 0$.
\\end{proof}"""

if old_prop1_block in text:
    text = text.replace(old_prop1_block, new_prop1_block, 1)
    print("[+] Proposition 1 and Proof successfully refined with DPI and intra/inter decomposition!")
else:
    print("[-] WARNING: old_prop1_block not found!")

# 2. Update Eq. (club) and Lemma 1 (vCLUB intra-taxon negative permutation)
old_club_block = """The conditional sample-based CLUB estimator over a mini-batch of $B_m$ multi-specimen samples is formulated as:
\\begin{equation}
    \\mathcal{L}_{\\text{CLUB}}(\\theta, \\xi) = \\frac{1}{B_m} \\sum_{i=1}^{B_m} \\left[ \\log q_\\xi(s_i \\mid z_i, y_i) - \\frac{1}{B_m} \\sum_{j=1}^{B_m} \\log q_\\xi(s_j \\mid z_i, y_i) \\right].
    \\label{eq:club}
\\end{equation}

\\begin{lemma}[Upper Bound Property of Conditional CLUB~\\cite{cheng2020club}]
\\label{lem:club}
For any variational distribution $q_\\xi(s \\mid z, y)$, the expected conditional CLUB estimator serves as a valid upper bound on the true conditional mutual information:
\\begin{equation}
    I(Z; S \\mid Y) \\le \\mathbb{E} \\left[ \\mathcal{L}_{\\text{CLUB}} \\right],
\\end{equation}
with equality holding if and only if $q_\\xi(s \\mid z, y) \\equiv P(s \\mid z, y)$ and $Z$ is independent of $S$ given $Y$.
\\end{lemma}"""

new_club_block = """The conditional sample-based CLUB estimator over a mini-batch of $B_m$ multi-specimen samples is formulated with intra-taxon negative permutation as:
\\begin{equation}
    \\mathcal{L}_{\\text{CLUB}}(\\theta, \\xi) = \\frac{1}{B_m} \\sum_{i=1}^{B_m} \\left[ \\log q_\\xi(s_i \\mid z_i, y_i) - \\frac{1}{|\\mathcal{B}_{y_i}|} \\sum_{j \\in \\mathcal{B}_{y_i}} \\log q_\\xi(s_j \\mid z_i, y_i) \\right],
    \\label{eq:club}
\\end{equation}
where $\\mathcal{B}_{y_i} = \\{k \\in \\{1,\\dots,B_m\\} \\mid y_k = y_i, k \\ne i\\}$ denotes the set of co-specific negative samples in the mini-batch. Because the support of local specimen distribution $q_\\xi(s \\mid z, y)$ is restricted to $\\mathcal{G}_y$, evaluating negative specimens from foreign taxa ($y_j \\ne y_i$) is mathematically undefined; intra-taxon shuffling guarantees valid negative contrast exclusively among vouchers of the same botanical species.

\\begin{lemma}[Variational Upper Bound Property of Conditional CLUB~\\cite{cheng2020club}]
\\label{lem:club}
When the variational distribution $q_\\xi(s \\mid z, y)$ is trained via maximum likelihood to approximate $P(S \\mid Z, Y)$, the expected conditional CLUB estimator serves as a variational upper bound on the conditional mutual information:
\\begin{equation}
    I(Z; S \\mid Y) \\le \\mathbb{E} \\left[ \\mathcal{L}_{\\text{CLUB}} \\right] + \\Delta_{\\text{var}}(\\xi),
\\end{equation}
where $\\Delta_{\\text{var}}(\\xi) = \\mathbb{E}_{P(Z,Y)}[D_{\\text{KL}}(P(S \\mid Z,Y) \\parallel q_\\xi(S \\mid Z,Y))] - D_{\\text{KL}}(P(S \\mid Y) \\parallel q_\\xi(S \\mid Y))$ denotes the variational gap. Minimizing $\\mathcal{L}_{\\text{CLUB}}$ compresses $I(Z; S \\mid Y)$ toward the variational threshold $\\varepsilon_{\\text{var}}$.
\\end{lemma}"""

if old_club_block in text:
    text = text.replace(old_club_block, new_club_block, 1)
    print("[+] Eq. (club) and Lemma 1 updated with intra-taxon negative permutation and variational gap!")
else:
    print("[-] WARNING: old_club_block not found!")

# 3. Update t-test text and Daubert text
old_t_test = """paired two-tailed Student's $t$-test yields $t(4) = 8.42$ with $p = 0.00105$ ($p = 1.05 \\times 10^{-3}$). The $95\\%$ confidence interval for the paired accuracy improvement spans $[+4.85\\%, +9.15\\%]$, demonstrating a decisive performance surge. The standardized effect size, quantified via Cohen's $d$, reaches $d = 3.76$ (classified as an exceptionally large effect size, where $d > 0.80$ is conventionally considered large~\cite{cohen1988}). Furthermore, an exact non-parametric Wilcoxon signed-rank test across all $n = 5$ evaluation folds yields a test statistic of $W = 0$ (reflecting strictly positive gains $\\Delta > 0$ across every evaluated fold), corresponding to an exact two-tailed $p$-value of $p = 0.0625$ (and an exact one-tailed $p$-value of $p = 0.03125$)."""

new_t_test = """paired two-tailed Student's $t$-test yields $t(4) = 8.43$ with $p = 0.00104$ ($p = 1.04 \\times 10^{-3}$). The $95\\%$ confidence interval for the paired accuracy improvement spans $[+4.73\\%, +9.37\\%]$, demonstrating consistent performance gains. The standardized effect size, quantified via Cohen's $d$, reaches $d = 3.77$ (classified as a very large effect size where $d > 0.80$ is conventionally considered large~\cite{cohen1988}). Furthermore, an exact non-parametric Wilcoxon signed-rank test across all $n = 5$ evaluation folds yields a test statistic of $W = 0$ (reflecting strictly positive gains across every fold), corresponding to an exact two-tailed $p$-value of $p = 0.0625$ (and an exact one-tailed $p$-value of $p = 0.03125$). We explicitly acknowledge that for $n = 5$ pairs, $p = 0.0625$ is the mathematical floor for two-tailed Wilcoxon tests ($2^{-4} = 0.0625$), approaching nominal significance; parametric tests and effect size ($d = 3.77$) corroborate systematic advantage. As cautioned by Bengio and Grandvalet (2004), cross-validation folds share training data; these statistics evaluate empirical consistency across splits rather than independent sampling."""

if old_t_test in text:
    text = text.replace(old_t_test, new_t_test, 1)
    print("[+] t-test text and Wilcoxon disclaimer updated successfully!")
else:
    print("[-] WARNING: old_t_test not found!")

# 4. Update Daubert paragraph to objective forensic screening standards
old_daubert_para = """In international judicial proceedings regarding timber confiscation, forensic evidence must meet stringent standards of scientific validity (e.g., the evidentiary reliability standard in United States federal courts). An AI system that relies on mechanical surface scratches rather than botanical anatomy is inherently vulnerable to legal challenge and dismissal. By mathematically proving specimen invariance, bounding residual mutual information, providing empirical verification via the Specimen Recoverability Index ($\mathrm{SRI}_{\text{lin}} \to 0.118$, $\mathrm{SRI}_{\text{mlp}} \to 0.134$), and demonstrating anatomical alignment via Grad-CAM, our framework provides the interpretability and evidentiary rigor required for legal admissibility."""

# Let's search by prefix
daubert_prefix = "In international judicial proceedings regarding timber confiscation"
idx_d = text.find(daubert_prefix)
if idx_d != -1:
    end_d = text.find("\n\n", idx_d)
    matched_daubert = text[idx_d:end_d]
    new_daubert_para = """In international administrative and judicial proceedings regarding timber consignment seizure, automated forensic screening tools must adhere to objective evidentiary standards of scientific reliability and transparency (e.g., ISO/IEC 17025 laboratory quality criteria and EUDR due-diligence verification). An AI system that relies opportunistically on specimen-specific machining scratches rather than diagnostic wood anatomy introduces intolerable risks of spurious classification and arbitrary confiscations. By actively regularizing against specimen shortcut encoding, bounding residual mutual information, providing empirical verification via the Specimen Recoverability Index ($\mathrm{SRI}_{\\text{lin}} \\to 0.118$, $\mathrm{SRI}_{\\text{mlp}} \\to 0.134$), and confirming anatomical alignment via expert-validated saliency audits, our framework substantially strengthens the diagnostic credibility and evidentiary defensibility of automated timber screening."""
    text = text.replace(matched_daubert, new_daubert_para, 1)
    print("[+] Daubert paragraph replaced with forensic compliance standards!")
else:
    print("[-] WARNING: daubert_prefix not found!")

# 5. Causal SCM paragraph refinement: acknowledge biological S -> X variation
scm_search = "\\noindent \\textbf{Structural Causal Formulation of Timber Shortcut Learning}:"
idx_scm = text.find(scm_search)
if idx_scm != -1:
    end_scm = text.find("\n\n", idx_scm)
    old_scm_text = text[idx_scm:end_scm]
    new_scm_text = """\\noindent \\textbf{Structural Causal Formulation of Timber Shortcut Learning}: Formally, we conceptualize macroscopic timber image generation through a Structural Causal Model (SCM). Ground-truth botanical taxon $Y$ governs macro-anatomical xylem structures $A$ (e.g., vessel diameter, axial parenchyma patterns, ray spacing). However, physical specimen voucher $S$ not only carries intrinsic biological tree-level morphological variations ($S \\to X$), but also acts as an opportunistic confounder introducing non-taxonomic surface artifacts $E$ (e.g., circular saw striations, sanding scratches, directional lighting gradients, moisture discolouration). The observed macroscopic image patch is generated via $X = g(A, E, S)$. When an unregularized deep neural network $f_\\theta(X)$ is trained via ERM, it exploits high-frequency correlations between $E$ and $Y$, learning opportunistic shortcuts $S \\to Z \\to \\hat{Y}$ rather than true anatomical invariants $A \\to Z \\to \\hat{Y}$. Our objective is to suppress specimen-specific idiosyncrasies $I(Z; S \\mid Y) \\to 0$ while preserving authentic taxonomic morphology $I(Z; Y) > 0$."""
    text = text.replace(old_scm_text, new_scm_text, 1)
    print("[+] SCM paragraph updated with explicit biological S -> X variation!")

with open(tex_path, "w", encoding="utf-8") as f:
    f.write(text)

print("[*] apply_detailed_theoretical_refinements.py completed successfully!")
