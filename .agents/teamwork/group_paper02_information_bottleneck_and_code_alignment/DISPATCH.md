## 2026-09-29T12:44:46Z
You are the Group Orchestrator for Segment 5: paper02_information_bottleneck_and_code_alignment.
Working Directory: G:/S3_paper/.agents/teamwork/group_paper02_information_bottleneck_and_code_alignment
Segment Name: paper02_information_bottleneck_and_code_alignment
Input Format: latex
Analysis Partition Path: G:/S3_paper/.agents/teamwork/document_orchestrator_1/ANALYSIS_PARTITION.md
Text Map Path: G:/S3_paper/.agents/teamwork/document_orchestrator_1/DOCUMENT_TEXT_MAP.md
Target Manuscript: G:/S3_paper/03_research_paper_specimen_invariance/paper/main.tex (Sections 3.5, 3.6)
Associated Code:
- G:/S3_paper/03_research_paper_specimen_invariance/trainers/trainer_adversarial.py
- G:/S3_paper/03_research_paper_specimen_invariance/models/club.py
- G:/S3_paper/03_research_paper_specimen_invariance/trainers/base_trainer.py

Your mission:
Execute the tournament review tree [4, 2, 1] with sample size 2 for this segment to conduct an adversarial mathematical and code-alignment review of Paper 02.
1. Level 0: Dispatch 4 parallel Analysts (teamwork_preview_worker).
2. Level 1: Dispatch 2 Review Aggregators (teamwork_preview_worker), each sampling 2 Level 0 candidate reviews.
3. Level 2: Dispatch 1 Final Aggregator (teamwork_preview_worker) to produce the definitive segment unit report: G:/S3_paper/.agents/teamwork/group_paper02_information_bottleneck_and_code_alignment/unit_report_paper02_information_bottleneck_and_code_alignment.md.

Mandatory Segment Audit Directives (R3 Part 2):
- Variational CLUB Mutual Information Bottleneck: Scrutinize the theoretical derivation of the conditional Contrastive Log-ratio Upper Bound (CLUB) estimator I(Z; S | Y) <= I_CLUB(Z; S | Y). Verify sample approximation equations, variational distribution parameterization q_theta(s|z, y), and unbiasedness/upper-bound guarantees.
- Code-to-Manuscript Discrepancy Audit (CRITICAL): Conduct a strict line-by-line audit between Section 3.5/3.6 in main.tex and trainer_adversarial.py / models/club.py. Specifically evaluate the discrepancy between manuscript claims (presenting a unified joint objective simultaneously combining Masked GRL + Variational CLUB) versus the actual implementation in trainer_adversarial.py (which dispatches training via discrete mutually exclusive modes: e.g. mode == 'adversarial' vs mode == 'club', or separate training steps). Identify whether this constitutes an architectural overclaim, implementation divergence, or undocumented ablation.
- Minimax Optimization & Convergence: Check convergence dynamics, gradient balancing, learning rate schedules, and empirical stability claims.
- Deliver clear, actionable reviewer critiques, exact code reconciliation recommendations, and ready-to-apply LaTeX diff patches.

When complete, write unit_report_paper02_information_bottleneck_and_code_alignment.md and send a completion message to parent with its absolute path.
