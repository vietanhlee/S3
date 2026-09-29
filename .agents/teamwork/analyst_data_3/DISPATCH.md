## 2026-09-29T12:46:50Z
You are Analyst 3 (Focus: Technical Validation & ML Benchmark Audit) for Segment 1: paper_data_scope_compliance in the adversarial document review of Paper Data (Elsevier Data in Brief).

Your input contract:
- Target Manuscript: G:/S3_paper/01_data_paper_forensic_cites/paper_data/main.tex
- Analysis Partition: G:/S3_paper/.agents/teamwork/document_orchestrator_1/ANALYSIS_PARTITION.md (input_format: latex)
- Text Map Path: G:/S3_paper/.agents/teamwork/document_orchestrator_1/DOCUMENT_TEXT_MAP.md
- Baseline Report: G:/S3_paper/01_data_paper_forensic_cites/paper_data/statistical_baseline_5seeds_report.md
- Scripts Directory: G:/S3_paper/01_data_paper_forensic_cites/paper_data/scripts/
- Modules Directory: G:/S3_paper/01_data_paper_forensic_cites/modules/

Your mission:
Perform an adversarial, comprehensive technical validation and code-to-manuscript audit and write your candidate review to:
G:/S3_paper/.agents/teamwork/segment_paper_data_scope_compliance/handoff_3.md

Mandatory Directives to address with rigorous scrutiny:
1. Technical Validation: Verify supervised baseline using ConvNeXt-Tiny (accuracy 90.42% +/- 0.38%). Cross-check numbers between main.tex and statistical_baseline_5seeds_report.md (precision, recall, F1, seed variance). Audit dataset curation and split deduplication scripts in paper_data/scripts/ and modules/.
2. Journal Scope & Non-compliant sections: Check if \section{Background} or inappropriate narrative sections exist in main.tex. Data in Brief requires manuscripts to be purely data descriptors. Provide excision diffs.
3. Specifications Table mandatory rows: Check all mandatory rows (Subject, Specific subject area, Type of data, Data format, Data collection, Data source location, Data accessibility, Related research article).
4. Value of the Data: Audit and condense \section{Value of the Data} into 4-6 concise, impactful bullet points.
5. CITES CoP19 Regulatory Status Audit: Deeply audit Table 1 (Taxonomic inventory), especially Pterocarpus soyauxii (verify and correct to CITES Appendix II with Annotation #17 under CoP19) and all other 18 taxa.
6. Provide ready-to-apply LaTeX diff snippets for all identified issues.

Write your exhaustive report to G:/S3_paper/.agents/teamwork/segment_paper_data_scope_compliance/handoff_3.md. When finished, send a brief message to parent notifying completion.
