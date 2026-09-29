# BRIEFING — 2026-09-29T12:47:00Z

## Mission
Perform an adversarial, comprehensive technical validation and code-to-manuscript audit for Paper Data (Elsevier Data in Brief) covering scope compliance, specifications table, value of the data, CITES CoP19 status, and baseline benchmark numbers.

## 🔒 My Identity
- Archetype: analyst_data_3
- Roles: implementer, qa, specialist@document_review
- Working directory: G:/S3_paper/.agents/teamwork/analyst_data_3
- Original parent: 2b14dc54-d627-44cf-8770-ff2c3a2e1347
- Milestone: Segment 1 (paper_data_scope_compliance) Technical Validation & Scope Review

## 🔒 Key Constraints
- Target manuscript: G:/S3_paper/01_data_paper_forensic_cites/paper_data/main.tex
- Text map: G:/S3_paper/.agents/teamwork/document_orchestrator_1/DOCUMENT_TEXT_MAP.md
- Baseline report: G:/S3_paper/01_data_paper_forensic_cites/paper_data/statistical_baseline_5seeds_report.md
- Scripts: G:/S3_paper/01_data_paper_forensic_cites/paper_data/scripts/
- Modules: G:/S3_paper/01_data_paper_forensic_cites/modules/
- Output path: G:/S3_paper/.agents/teamwork/segment_paper_data_scope_compliance/handoff_3.md
- Output format: Segment Report Output Structure with 3-tier academic severity scale (Critical, Major, Minor), no Summary section, actionable diff snippets, Vietnamese response to user.
- Strictly adhere to Elsevier Data in Brief format and guidelines.

## Current Parent
- Conversation ID: 2b14dc54-d627-44cf-8770-ff2c3a2e1347
- Updated: not yet

## Task Summary
- **What to build**: Comprehensive adversarial audit report `handoff_3.md` covering all 6 mandatory directives:
  1. Technical validation of supervised baseline (ConvNeXt-Tiny 90.42% +/- 0.38%), cross-checking main.tex vs statistical_baseline_5seeds_report.md, and dataset curation/split scripts.
  2. Elsevier Data in Brief scope compliance: check narrative sections (e.g., \section{Background}) and provide excision diffs.
  3. Specifications Table: check all mandatory rows (Subject, Specific subject area, Type of data, Data format, Data collection, Data source location, Data accessibility, Related research article).
  4. Value of the Data: audit and condense into 4-6 concise, impactful bullet points.
  5. CITES CoP19 regulatory status audit: Table 1 audit, specifically Pterocarpus soyauxii (correct to Appendix II with Annotation #17 under CoP19) and all 18 taxa.
  6. Ready-to-apply LaTeX diff snippets for all identified issues.
- **Success criteria**: Exhaustive, mathematically and empirically validated findings with exact LaTeX patches.
- **Interface contracts**: G:/S3_paper/.agents/teamwork/document_orchestrator_1/ANALYSIS_PARTITION.md
- **Code layout**: G:/S3_paper/01_data_paper_forensic_cites/

## Key Decisions Made
- Will inspect main.tex, statistical_baseline_5seeds_report.md, paper_data/scripts, and modules thoroughly before drafting findings.

## Artifact Index
- G:/S3_paper/.agents/teamwork/segment_paper_data_scope_compliance/handoff_3.md — Candidate review handoff report

## Change Tracker
- **Files modified**: None yet
- **Build status**: N/A (document review)
- **Pending issues**: Investigation in progress

## Quality Status
- **Build/test result**: In progress
- **Lint status**: N/A
- **Tests added/modified**: N/A

## Loaded Skills
- None loaded yet.
