# Document Review Analysis Partition

input_format: latex
text_map_path: G:/S3_paper/.agents/teamwork/document_orchestrator_1/DOCUMENT_TEXT_MAP.md
total_sections: 65

## Segments

### Segment 1: paper_data_scope_compliance
- **category**: Methodology
- **section_ranges**: ["Specifications Table", "Value of the Data", "Data Description", "Experimental Design, Materials and Methods", "Ethics Statement", "Data Availability", "Code Availability"]
- **target_manuscript**: 01_data_paper_forensic_cites/paper_data/main.tex
- **context_instruction**: >
  Auditing Paper Data (Targeted for Elsevier Data in Brief).
  Rigorously verify compliance against Elsevier Data in Brief specifications:
  1. Journal Scope & Non-compliant sections: Verify removal of any narrative/background sections (specifically check if \section{Background} exists and must be removed; ensure paper remains strictly a data descriptor).
  2. Specifications Table mandatory rows: Verify presence and exact formatting of all required rows: Subject, Specific subject area, Type of data, Data format, Data collection, Data source location, Data accessibility, and Related research article. Check for missing or improper fields.
  3. Value of the Data: Condense and audit Value of the Data to ensure it contains exactly 4-6 concise, impactful bullet points highlighting forensic compliance, rapid non-destructive screening, multimodal ground truth, and dual-split architecture.
  4. CITES Regulatory Status & Taxonomic Audit: Audit Table 1 (taxonomic inventory) and verify CITES CoP19 regulatory status. Specifically check Pterocarpus soyauxii — verify whether it is currently marked Non-CITES or CITES App. II and provide the precise regulatory correction to Appendix II with Annotation #17 under CoP19. Also verify status of all other 18 taxa (Afzelia, Dalbergia, Guibourtia, Peltogyne, Sindora).
  5. Technical validation: Inspect baseline ConvNeXt-Tiny classification performance and dataset curation scripts.
  6. Output concrete LaTeX diff patches for all identified issues.

### Segment 2: paper01_foundations_theory
- **category**: Theory / Math
- **section_ranges**: ["Introduction", "Related Work", "Specimen-Centric Data Governance and Proposed Framework - Notation and Specimen Leakage Risk (SLR)", "Taxonomy of Partitioning Protocols and Candidate Solver Pool", "The Three Pillars Evaluation Framework"]
- **target_manuscript**: 02_research_paper_specimen_leakage/paper/main.tex
- **context_instruction**: >
  Auditing Paper 01 theoretical foundations and mathematical derivations.
  Rigorously verify:
  1. Lemma 1: Mathematical necessity proof for discrete block disjointness (|G_c| >= 3) under full class coverage (CCR=100%). Check validity of proof steps, edge cases (|G_c| < 3), pigeonhole principle / partition mechanics across Train/Val/Test splits.
  2. Observation 1 (Knapsack Friction & Fallback Leakage): Verify formal justification of the empirical SLR = 6.0% floor (7 boundary fallback blocks out of 116 canonical blocks). Check integer programming knapsack capacity constraints and boundary leakage mechanics.
  3. Notation and Specimen Leakage Risk (SLR): Check mathematical formalization of SLR, Same-Specimen-Picture Bias (SSPB), Pairwise External C-index/Variance (PECVC), and Three Pillars evaluation metrics.
  4. Provide exact mathematical proof corrections and LaTeX diff snippets.

### Segment 3: paper01_combinatorial_optimization_and_benchmarks
- **category**: Experiments
- **section_ranges**: ["Combinatorial Meta-Selector (CEGS-Split)", "Experimental Setup", "Experimental Results and Analysis", "Discussion and Limitations", "Conclusion", "Appendices"]
- **target_manuscript**: 02_research_paper_specimen_leakage/paper/main.tex
- **context_instruction**: >
  Auditing Paper 01 optimization algorithms, empirical evaluations, and code alignment.
  Rigorously verify:
  1. Combinatorial Meta-Selector (CEGS-Split): Audit the formulation of multi-objective Simulated Annealing over the 11^18 search space balancing DataSAIL loss, MMD, and Hardest-Class F1. Verify cooling schedules, transition probabilities, energy functions, and Pareto optimality.
  2. Code-to-manuscript audit: Cross-check LaTeX algorithms and equations against implementation in 02_research_paper_specimen_leakage/partitioning/governed_split.py, graph_splits.py, split_protocols.py, and datasail_benchmark/. Identify any parameter or algorithmic discrepancies.
  3. Fairness of 1-NN Benchmarking: Evaluate the justification of frozen representation evaluation and its correlation with end-to-end deep fine-tuning (multi-backbone validation). Check whether 1-NN evaluation is fair across baselines or introduces representation bias.
  4. Provide concrete LaTeX diff snippets and algorithmic revisions.

### Segment 4: paper02_causal_theory_proofs
- **category**: Theory / Math
- **section_ranges**: ["Introduction", "Related Work", "Specimen-Invariant Learning Methodology - Causal Formulation and Problem Setup", "System Architecture Overview", "Primary Diagnostic Objective: Class-Balanced Focal Loss", "Species-Conditioned Specimen Discriminator with Masked Softmax GRL"]
- **target_manuscript**: 03_research_paper_specimen_invariance/paper/main.tex
- **context_instruction**: >
  Auditing Paper 02 causal formulation, information-theoretic bounds, and GRL theory.
  Rigorously verify:
  1. Proposition 1 & Proof (Semantic Collapse on Singleton Taxa): Deep scrutiny of the information-theoretic derivation bounding I(Z; Y = c*) <= I(Z; S = s*) -> 0 when specimen and species are collinear. Verify whether data processing inequality, conditioning, or mutual information identities hold strictly, and identify any hidden assumptions or flaws in the proof.
  2. Species-Conditioned Masked Softmax GRL (Theorem 1): Audit mathematical correctness of masking singleton classes (M_c = 0) in the adversarial specimen discriminator. Verify gradient flow under Gradient Reversal Layer (GRL), minimax equilibrium conditions, and proof of non-degradation on singleton species.
  3. Cross-reference code in 03_research_paper_specimen_invariance/models/grl.py and models/full_model.py.
  4. Provide rigorous mathematical corrections and ready-to-apply LaTeX diffs.

### Segment 5: paper02_information_bottleneck_and_code_alignment
- **category**: Theory / Math
- **section_ranges**: ["Variational Mutual Information Bottleneck via CLUB", "Joint Objective, Minimax Optimization, and Convergence Dynamics"]
- **target_manuscript**: 03_research_paper_specimen_invariance/paper/main.tex
- **context_instruction**: >
  Auditing Paper 02 CLUB estimator, minimax dynamics, and code alignment.
  Rigorously verify:
  1. Variational CLUB Mutual Information Bottleneck: Derive and verify the conditional CLUB estimator I(Z; S | Y) <= I_CLUB(Z; S | Y). Check sample estimators, log-ratio formulations, and variational distribution parameterization q_theta(s|z, y).
  2. Code-to-manuscript discrepancy check: Audit 03_research_paper_specimen_invariance/trainers/trainer_adversarial.py and models/club.py against manuscript claims in Section 3.5 & 3.6. Specifically verify whether the manuscript claims a unified joint objective (GRL + CLUB trained simultaneously) while the trainer implementation dispatches GRL and CLUB as separate disjoint training modes (mode == 'adversarial' vs mode == 'club'). Formulate clear criticism and correction instructions for authors.
  3. Minimax optimization and convergence dynamics: Check Lipschitz continuity assumptions, learning rate scheduling, and stability guarantees.
  4. Provide ready-to-apply LaTeX diff snippets and code reconciliation guidance.

### Segment 6: paper02_experimental_evaluation_and_loso
- **category**: Experiments
- **section_ranges**: ["Experimental Protocol and Diagnostic Metrics", "Experimental Results and In-Depth Analysis", "Discussion, Anatomical Saliency, and Operational Viability", "Conclusion"]
- **target_manuscript**: 03_research_paper_specimen_invariance/paper/main.tex
- **context_instruction**: >
  Auditing Paper 02 empirical protocols, evaluation metrics, and experimental integrity.
  Rigorously verify:
  1. Specimen Recoverability Index (SRI): Verify mathematical definition, linear probe / k-NN probe formulation, and its behavioral correlation with deployment generalization gap (GGSL).
  2. Round-Robin Leave-One-Specimen-Out (LOSO) Protocol: Audit protocol rigor, computational tractability, and variance estimation across 13 learning paradigms (ERM, Focal, GroupDRO, CORAL, DANN, CDAN, CLUB, etc.).
  3. Master Invariance Benchmark across 13 Baselines: Check baseline implementations in 03_research_paper_specimen_invariance/trainers/ and losses/. Verify fairness of comparison, hyperparameter tuning parity, calibration metrics (ECE, MCE), and ablation validity.
  4. Discussion and practical impact: Evaluate Grad-CAM anatomical saliency claims against IAWA feature conventions and legal admissibility under CITES CoP19 border screening.
  5. Provide concrete LaTeX diff snippets and actionable reviewer comments.
