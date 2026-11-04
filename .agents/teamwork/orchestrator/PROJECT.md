# Project: Specimen Leakage Audit and Honest Negative Result Reporting

## Architecture & Code Layout
- Target directory: `g:/S3_paper/03_research_paper_specimen_invariance/`
- Paper: `03_research_paper_specimen_invariance/paper/main.tex` and accompanying figures/tables
- Experiments & Source: `03_research_paper_specimen_invariance/src/` or related experimental scripts

## Feature Inventory
| # | Feature / Objective | Description | Milestone | Source |
|---|---------------------|-------------|-----------|--------|
| 1 | Scientific Re-framing (R1) | Title, Abstract, Intro, Contributions, Method, Discussion, Conclusion reframed to audit & negative result | M2 | ORIGINAL_REQUEST.md |
| 2 | Contradiction Resolution (R2) | Resolve 8 empirical contradictions (GGSL, Train SRI, ImageNet anchor, Baseline 97.07%, ECE 1.58-1.93%, 11 Fabaceae species only, unbold non-stat-sig rows, tone down 3rd-party claims) | M2 | ORIGINAL_REQUEST.md |
| 3 | Supplementary Experiments (R3) | 9 seeds random-split CI, specimen 50 & 53 failure analysis, GRL discriminator loss & adv_weight dynamics, non-independent split notice | M1 | ORIGINAL_REQUEST.md |
| 4 | LaTeX Clean Compilation | Compile cleanly to main.pdf without fatal errors or broken citations/references | M2 | ORIGINAL_REQUEST.md |
| 5 | Rigorous Verification & Audit | Verify against all acceptance criteria, check numbers and claims, audit integrity | M3 | ORIGINAL_REQUEST.md |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| 1 | Investigation & Analysis (Explorer) | Deep survey of codebase, 9 seeds data, specimen 50/53 images, discriminator logs, paper text contradictions | none | DONE |
| 2 | Code Experiments & Paper Refactoring (Worker) | Run 9-seed evaluation, extract CI, diagnose specimen 50 & 53, log GRL, rewrite main.tex, compile main.pdf | M1 | DONE |
| 3 | Comprehensive Review & Audit (Reviewer/Auditor) | Audit paper against R1, R2, R3, verify LaTeX build, verify statistical rigor, verify zero overclaims | M2 | DONE |
