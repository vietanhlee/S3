# BRIEFING — 2026-09-29T12:48:00Z

## Mission
Perform an adversarial, comprehensive botanical and regulatory audit of Paper Data (Elsevier Data in Brief) with deep scrutiny on CITES CoP19 status, taxonomic correctness, DiB specifications table, Value of the Data, and non-compliant sections, producing handoff_2.md.

## 🔒 My Identity
- Archetype: analyst / implementer / qa / specialist@document_review
- Roles: implementer, qa, specialist@document_review
- Working directory: G:\S3_paper\.agents\teamwork\analyst_data_2
- Original parent: 2b14dc54-d627-44cf-8770-ff2c3a2e1347
- Milestone: Segment 1 (paper_data_scope_compliance) Adversarial Audit

## 🔒 Key Constraints
- Target manuscript: G:/S3_paper/01_data_paper_forensic_cites/paper_data/main.tex
- Target report: G:/S3_paper/.agents/teamwork/segment_paper_data_scope_compliance/handoff_2.md
- CITES CoP19 Panama verification (effective 23 Feb 2023), especially Pterocarpus soyauxii (App. II, #17) and 18 other taxa across Afzelia, Dalbergia, Guibourtia, Peltogyne, Sindora.
- Elsevier Data in Brief specifications compliance: check/excise \section{Background}, audit Specifications Table mandatory rows, condense Value of the Data to 4-6 concise bullets.
- Technical validation: ConvNeXt-Tiny baseline (90.42% +/- 0.38%), check statistical_baseline_5seeds_report.md and scripts.
- Output ready-to-apply LaTeX diff snippets for all identified issues.
- Inter-Agent Communication Hygiene: do NOT send detailed report in send_message, only brief confirmation + file path.
- User rule: Code chuẩn production, trả lời rõ ràng bằng tiếng Việt.

## Current Parent
- Conversation ID: 2b14dc54-d627-44cf-8770-ff2c3a2e1347
- Updated: not yet

## Task Summary
- **What to build**: Comprehensive adversarial audit report (handoff_2.md) covering CITES taxonomy, regulatory status, DiB scope compliance, Specifications Table, Value of the Data, and technical validation with exact LaTeX diffs.
- **Success criteria**: Full verification of all 19 taxa, exact identification of Pterocarpus soyauxii regulatory misclassification, exact diffs for DiB scope compliance, specifications table, and baseline metrics verification.
- **Interface contracts**: G:/S3_paper/.agents/teamwork/document_orchestrator_1/ANALYSIS_PARTITION.md
- **Code layout**: Target files in 01_data_paper_forensic_cites/paper_data/

## Key Decisions Made
- Initializing audit plan across 5 core pillars: (1) CITES & Taxonomic Inventory, (2) DiB Scope Compliance, (3) Specifications Table, (4) Value of the Data, (5) Technical Validation & Code Alignment.

## Artifact Index
- G:/S3_paper/.agents/teamwork/segment_paper_data_scope_compliance/handoff_2.md — Final candidate review report
