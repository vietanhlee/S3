# Original User Request

## Initial Request — 2026-09-29T12:32:04Z

Conduct an exhaustive, independent, and adversarial peer review across the trilogy of timber forensic research manuscripts (Paper Data, Paper 01 on Specimen Leakage Governance, and Paper 02 on Specimen Invariance & Information Bottlenecks), with deep scrutiny on mathematical soundness, algorithmic correctness, and code-to-paper alignment.

Working directory: G:/S3_paper
Integrity mode: development

## Reference Manuscripts
1. **Paper Data**: `01_data_paper_forensic_cites/paper_data/main.tex` — Targeted for *Elsevier Data in Brief*.
2. **Paper 01**: `02_research_paper_specimen_leakage/paper/main.tex` — Data-centric paper on PECVC, SSPB leakage, and combinatorial partition governance (CEGS-Split).
3. **Paper 02**: `03_research_paper_specimen_invariance/paper/main.tex` — Model-centric paper on Species-Conditioned Masked Softmax GRL and CLUB variational mutual information bottlenecks.

## Requirements

### R1. Journal Scope & Formatting Audit for Paper Data (Elsevier Data in Brief)
Audit formatting conformity to Elsevier Data in Brief specifications: verify removal of non-compliant sections (`\section{Background}`), integration of `Specifications Table` mandatory rows (`Data format`, `Related research article`), condensation of `Value of the Data` to 4-6 concise bullets, and verification of CITES CoP19 regulatory status (specifically correcting *Pterocarpus soyauxii* from Non-CITES to Appendix II with Annotation #17).

### R2. Mathematical & Algorithmic Rigor Audit for Paper 01 (Partition Governance)
Perform a rigorous mathematical and algorithmic audit of:
- **Lemma 1**: Mathematical necessity proof for discrete block disjointness ($|\mathcal{G}_c| \ge 3$) under full class coverage ($\mathrm{CCR}=100\%$).
- **Observation 1 (Knapsack Friction & Fallback Leakage)**: Formal justification of the empirical $\mathrm{SLR} = 6.0\%$ floor (7 boundary fallback blocks out of 116 canonical blocks).
- **Combinatorial Meta-Selector (CEGS-Split)**: Formulation of multi-objective Simulated Annealing over the $11^{18}$ search space balancing DataSAIL loss, MMD, and Hardest-Class F1.
- **Fairness of 1-NN Benchmarking**: Justification of frozen representation evaluation and its correlation with end-to-end deep fine-tuning.

### R3. Representation Learning & Proof Verification for Paper 02 (Specimen Invariance)
Perform an adversarial mathematical and code-alignment review of:
- **Proposition 1 & Proof (Semantic Collapse on Singleton Taxa)**: Verification of information-theoretic derivation bounding $I(Z; Y = c^*) \le I(Z; S = s^*) \to 0$.
- **Species-Conditioned Masked Softmax GRL (Theorem 1)**: Mathematical correctness of masking singleton classes ($M_c = 0$) in the adversarial specimen discriminator.
- **Variational CLUB Mutual Information Bottleneck**: Derive and verify the conditional CLUB estimator and detect the discrepancy between manuscript claim (joint GRL + CLUB) and trainer implementation (`trainer_adversarial.py` dispatching GRL and CLUB as separate modes).
- **Specimen Recoverability Index (SRI) & LOSO Protocol**: Audit experimental integrity and baseline comparisons across 13 learning paradigms.

## Acceptance Criteria

### Audit Depth & Critical Rigor
- [ ] Explicit mathematical verification for all definitions, lemmas, propositions, and theorems in Paper 01 and Paper 02.
- [ ] Code-to-manuscript audit identifying any implementation discrepancies between LaTeX text and Python code.
- [ ] Ready-to-apply LaTeX diff snippets and formula corrections for every flagged issue.
- [ ] Distinct, actionable reviewer reports for Paper Data, Paper 01, and Paper 02 with clear prioritization (Major vs. Minor revisions).

## Follow-up — 2026-09-29T12:37:24Z

Đã cài đặt thành công pypdfium2 (v5.13.0) và Pillow vào cả môi trường ảo .venv lẫn môi trường Python hệ thống. 
Lệnh xác minh `G:\S3_paper\.venv\Scripts\python.exe -c "import pypdfium2, PIL; print('READY')"` đã trả về READY.
Vui lòng tiếp tục tiến trình tiền kiểm và bắt đầu phiên thẩm định chuyên sâu cho 3 bài báo.
