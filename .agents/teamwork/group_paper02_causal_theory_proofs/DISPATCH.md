## 2026-09-29T12:44:46Z
Sender: 162469b4-bf2d-4563-9d8c-663bdc3dbf92
Content:
You are the Group Orchestrator for Segment 4: paper02_causal_theory_proofs.
Working Directory: G:/S3_paper/.agents/teamwork/group_paper02_causal_theory_proofs
Segment Name: paper02_causal_theory_proofs
Input Format: latex
Analysis Partition Path: G:/S3_paper/.agents/teamwork/document_orchestrator_1/ANALYSIS_PARTITION.md
Text Map Path: G:/S3_paper/.agents/teamwork/document_orchestrator_1/DOCUMENT_TEXT_MAP.md
Target Manuscript: G:/S3_paper/03_research_paper_specimen_invariance/paper/main.tex (Sections 1, 2, 3.1, 3.2, 3.3, 3.4)
Associated Code:
- G:/S3_paper/03_research_paper_specimen_invariance/models/grl.py
- G:/S3_paper/03_research_paper_specimen_invariance/models/full_model.py
- G:/S3_paper/03_research_paper_specimen_invariance/losses/

Your mission:
Execute the tournament review tree [4, 2, 1] with sample size 2 for this segment to conduct an adversarial mathematical and causal representation review of Paper 02.
1. Level 0: Dispatch 4 parallel Analysts (teamwork_preview_worker).
2. Level 1: Dispatch 2 Review Aggregators (teamwork_preview_worker), each sampling 2 Level 0 candidate reviews.
3. Level 2: Dispatch 1 Final Aggregator (teamwork_preview_worker) to produce the definitive segment unit report: G:/S3_paper/.agents/teamwork/group_paper02_causal_theory_proofs/unit_report_paper02_causal_theory_proofs.md.

Mandatory Segment Audit Directives (R3 Part 1):
- Proposition 1 & Proof (Semantic Collapse on Singleton Taxa): Adversarially review the information-theoretic derivation bounding I(Z; Y = c*) <= I(Z; S = s*) -> 0. Verify the causal DAG, conditioning mechanics, mutual information chain rule, and data processing inequalities. Scrutinize whether forcing specimen invariance under unmasked adversarial training mathematically forces erasure of species identity for singleton taxa (where specimen S and class Y are collinear). Detect any leaps, unjustified steps, or notation ambiguities.
- Species-Conditioned Masked Softmax GRL (Theorem 1): Audit mathematical correctness of Theorem 1 and the masking mechanism (M_c = 0 for singleton classes) in the adversarial specimen discriminator. Verify gradient reversal dynamics, minimax game formulation, stability at Nash equilibrium, and proof that masking prevents semantic collapse while retaining specimen invariance on multi-specimen classes.
- Code-to-Manuscript Verification: Cross-check the mathematical masking formula in Theorem 1 with the PyTorch implementation in models/grl.py and models/full_model.py. Confirm whether the mask is applied before or after softmax, and whether loss weighting is consistent.
- Deliver rigorous mathematical derivations, formula corrections, and ready-to-apply LaTeX diff patches.

When complete, write unit_report_paper02_causal_theory_proofs.md and send a completion message to parent with its absolute path.
