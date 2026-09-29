## 2026-09-29T12:44:46Z

You are the Group Orchestrator for Segment 2: paper01_foundations_theory.
Working Directory: G:/S3_paper/.agents/teamwork/group_paper01_foundations_theory
Segment Name: paper01_foundations_theory
Input Format: latex
Analysis Partition Path: G:/S3_paper/.agents/teamwork/document_orchestrator_1/ANALYSIS_PARTITION.md
Text Map Path: G:/S3_paper/.agents/teamwork/document_orchestrator_1/DOCUMENT_TEXT_MAP.md
Target Manuscript: G:/S3_paper/02_research_paper_specimen_leakage/paper/main.tex (Sections 1, 2, 3.1, 3.2, 3.3)
Associated Code:
- G:/S3_paper/02_research_paper_specimen_leakage/partitioning/
- G:/S3_paper/02_research_paper_specimen_leakage/split_protocols.py

Your mission:
Execute the tournament review tree [4, 2, 1] with sample size 2 for this segment to conduct an adversarial mathematical and theoretical audit of Paper 01.
1. Level 0: Dispatch 4 parallel Analysts (teamwork_preview_worker).
2. Level 1: Dispatch 2 Review Aggregators (teamwork_preview_worker), each sampling 2 Level 0 candidate reviews.
3. Level 2: Dispatch 1 Final Aggregator (teamwork_preview_worker) to produce the definitive segment unit report: G:/S3_paper/.agents/teamwork/group_paper01_foundations_theory/unit_report_paper01_foundations_theory.md.

Mandatory Segment Audit Directives (R2 Part 1):
- Lemma 1 Audit: Perform an exhaustive mathematical audit of Lemma 1 (mathematical necessity proof for discrete block disjointness under full class coverage CCR = 100%). Verify the necessity condition |G_c| >= 3 for each class c across train, val, and test splits. Scrutinize the proof steps, edge cases (|G_c| < 3, e.g., singleton/doubleton taxa), pigeonhole constraints, and whether the claim holds strictly or requires additional regularity conditions.
- Observation 1 (Knapsack Friction & Fallback Leakage): Rigorously audit the formal justification of the empirical Specimen Leakage Risk floor SLR = 6.0% (attributed to 7 boundary fallback blocks out of 116 canonical blocks). Check the knapsack formulation, capacity constraints, integer relaxation, and whether the 6.0% leakage floor is theoretically unavoidable or an artifact of the greedy solver heuristic.
- Notation & Metric Rigor: Verify mathematical formalization of Specimen Leakage Risk (SLR), Same-Specimen-Picture Bias (SSPB), Pairwise External C-index/Variance (PECVC), and Three Pillars Evaluation Framework metrics.
- Provide explicit mathematical formula corrections and ready-to-apply LaTeX diff snippets.

When complete, write unit_report_paper01_foundations_theory.md and send a completion message to parent with its absolute path.
