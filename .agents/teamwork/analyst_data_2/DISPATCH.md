## 2026-09-29T12:46:50Z
You are Analyst 2 (Focus: CITES Taxonomy & Regulatory Audit) for Segment 1: paper_data_scope_compliance in the adversarial document review of Paper Data (Elsevier Data in Brief).

Your input contract:
- Target Manuscript: G:/S3_paper/01_data_paper_forensic_cites/paper_data/main.tex
- Analysis Partition: G:/S3_paper/.agents/teamwork/document_orchestrator_1/ANALYSIS_PARTITION.md (input_format: latex)
- Text Map Path: G:/S3_paper/.agents/teamwork/document_orchestrator_1/DOCUMENT_TEXT_MAP.md
- Baseline Report: G:/S3_paper/01_data_paper_forensic_cites/paper_data/statistical_baseline_5seeds_report.md
- Scripts Directory: G:/S3_paper/01_data_paper_forensic_cites/paper_data/scripts/
- Modules Directory: G:/S3_paper/01_data_paper_forensic_cites/modules/

Your mission:
Perform an adversarial, comprehensive botanical and regulatory audit of the manuscript and write your candidate review to:
G:/S3_paper/.agents/teamwork/segment_paper_data_scope_compliance/handoff_2.md

Mandatory Directives to address with rigorous scrutiny:
1. CITES CoP19 Regulatory Status & Taxonomic Audit: Conduct an exhaustive audit of Table 1 (Taxonomic inventory) and throughout the manuscript. Deeply verify Pterocarpus soyauxii: check its current listing in main.tex (verify whether it is mislabeled as Non-CITES or lacks Annotation #17) and provide the rigorous correction to CITES Appendix II with Annotation #17 (CoP19 Panama, effective Feb 2023). Verify all other 18 taxa across Afzelia, Dalbergia, Guibourtia, Peltogyne, Sindora, checking trade regulations, IUCN statuses, and scientific nomenclature.
2. Journal Scope & Non-compliant sections: Check for \section{Background} or literature review sections that violate Data in Brief formatting guidelines, providing excision diffs.
3. Specifications Table mandatory rows: Audit presence and formatting of Subject, Specific subject area, Type of data, Data format, Data collection, Data source location, Data accessibility, and Related research article.
4. Value of the Data: Audit \section{Value of the Data} to ensure exactly 4-6 concise, impactful bullets highlighting forensic timber compliance, rapid non-destructive screening, multimodal ground truth, and dual-split architecture.
5. Technical Validation & Scripts: Verify ConvNeXt-Tiny baseline (90.42% +/- 0.38%), check statistical_baseline_5seeds_report.md, and audit data preparation scripts.
6. Provide ready-to-apply LaTeX diff snippets for all identified issues.
