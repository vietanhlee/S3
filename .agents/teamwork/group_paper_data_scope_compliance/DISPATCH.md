## 2026-09-29T12:44:46Z

You are the Group Orchestrator for Segment 1: paper_data_scope_compliance.
Working Directory: G:/S3_paper/.agents/teamwork/group_paper_data_scope_compliance
Segment Name: paper_data_scope_compliance
Input Format: latex
Analysis Partition Path: G:/S3_paper/.agents/teamwork/document_orchestrator_1/ANALYSIS_PARTITION.md
Text Map Path: G:/S3_paper/.agents/teamwork/document_orchestrator_1/DOCUMENT_TEXT_MAP.md
Target Manuscript: G:/S3_paper/01_data_paper_forensic_cites/paper_data/main.tex
Associated Code & Assets:
- G:/S3_paper/01_data_paper_forensic_cites/paper_data/scripts/
- G:/S3_paper/01_data_paper_forensic_cites/modules/

Your mission:
Execute the tournament review tree [4, 2, 1] with sample size 2 for this segment to conduct an adversarial, exhaustive peer review of Paper Data (Elsevier Data in Brief).
1. Level 0: Dispatch 4 parallel Analysts (teamwork_preview_worker). Each analyst reads the target manuscript, checks compliance, audits specifications, verifies CITES regulatory status, and inspects baseline classification.
2. Level 1: Dispatch 2 Review Aggregators (teamwork_preview_worker), each sampling 2 Level 0 candidate reviews to reconcile findings, eliminate false positives, and refine evidence.
3. Level 2: Dispatch 1 Final Aggregator (teamwork_preview_worker) combining Level 1 outputs to produce the definitive segment unit report: G:/S3_paper/.agents/teamwork/group_paper_data_scope_compliance/unit_report_paper_data_scope_compliance.md.

Mandatory Segment Audit Directives (R1):
- Journal Scope & Formatting Audit: Verify strict adherence to Elsevier Data in Brief guidelines. Confirm whether any background or narrative sections (e.g., \section{Background}) exist and must be excised. Ensure the manuscript remains strictly an empirical data descriptor.
- Specifications Table Mandatory Rows: Verify presence and exact structure of all required rows: Subject, Specific subject area, Type of data, Data format, Data collection, Data source location, Data accessibility, and Related research article. Flag any missing, malformed, or ambiguous rows.
- Value of the Data: Condense and audit \section{Value of the Data} to ensure it presents exactly 4-6 concise, high-impact bullet points focusing on forensic timber compliance, rapid non-destructive screening, multimodal ground truth, and dual-split architecture.
- CITES CoP19 Regulatory Status Audit: Deeply audit Table 1 (Taxonomic inventory). Specifically audit Pterocarpus soyauxii: verify its current listing in the manuscript (often erroneously labeled Non-CITES or missing Annotation #17) and provide the rigorous correction to CITES Appendix II with Annotation #17 under CoP19 decisions. Verify all other 18 taxa (Afzelia, Dalbergia, Guibourtia, Peltogyne, Sindora).
- Technical Validation: Verify the supervised baseline using ConvNeXt-Tiny (accuracy 90.42% +/- 0.38%), multi-seed report statistical_baseline_5seeds_report.md, and dataset curation scripts.
- Deliver ready-to-apply LaTeX diff snippets for all identified corrections.

When complete, write unit_report_paper_data_scope_compliance.md and send a completion message to parent with its absolute path.
