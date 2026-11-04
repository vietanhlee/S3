# Task Assignment for Forensic Auditor M3

**Assigned Agent**: teamwork_auditor
**Working Directory**: g:/S3_paper/.agents/teamwork/auditor_m3/
**Context & Scope**: g:/S3_paper/.agents/teamwork/ORIGINAL_REQUEST.md
**Explorer Handoff**: g:/S3_paper/.agents/teamwork/explorer_m1/handoff.md
**Worker Handoff**: g:/S3_paper/.agents/teamwork/worker_m2/handoff.md
**Project Document**: g:/S3_paper/.agents/teamwork/orchestrator/PROJECT.md

## Audit Objectives & Verification Checklist
You must independently verify and audit all deliverables against the Acceptance Criteria in `ORIGINAL_REQUEST.md`:

### 1. LaTeX Compilation & Document Structure Audit
- Independently verify that `03_research_paper_specimen_invariance/paper/main.tex` compiles cleanly to `paper/main.pdf`.
- Verify the compilation log: ensure 0 fatal errors, 0 broken citation marks `[?]`, and 0 broken cross-references `??`.
- Audit document title, abstract, contributions, method, discussion, and conclusion: verify that the scientific reframing (R1) is 100% faithful to "Specimen Leakage Audit and Honest Negative Results Reporting". Ensure no overclaiming statements ("mitigating", "superior", "provably invariant", etc.) remain.

### 2. Empirical Consistency & 8 Contradictions Audit (R2)
Examine `paper/main.tex`, tables, and generated reports to verify each of the 8 contradiction resolutions:
1. GGSL: Confirm Section 4.3 accurately reports Baseline 2.74 pp vs GAN 2.82 pp (does not claim GAN eliminated GGSL or reduced it).
2. Train SRI: Confirm Section 4.3 reports GAN 0.538 does not decrease compared to Baseline 0.526.
3. ImageNet anchor: Confirm no contradictory logic with ImageNet anchor 0.5667.
4. Baseline Accuracy: Confirm Section 4.2 reports exact 97.07% instead of ~98.5-99.0%.
5. ECE: Confirm reported as 1.58% - 1.93% (not "< 1%").
6. Botanical Species: Audit entire manuscript to ensure *Dialium cochinchinense* and *Pterocarpus macrocarpus* do NOT appear; confirm all anatomical and forensic analyses strictly refer to the 11 Fabaceae species of S3.
7. Tone and Overclaiming: Confirm all claims are restrained, balanced, and evidence-based.
8. Third-party Studies: Confirm claims regarding prior literature are objective, reporting the leakage gap on S3 (+2.48 pp [1.73, 3.23 pp]) without sweeping accusations.
9. Table Unbolding: Verify that entire rows for GAN/adversarial methods in benchmark tables are NOT bolded unless they have statistically significant superiority.

### 3. Supplementary Experiments & Code Integrity Audit (R3)
- Verify `03_research_paper_specimen_invariance/run_benchmark.py`, `generate_report.py`, and `outputs/multi_seed_results.json`:
  - Verify the 9-seeds random-split evaluation and the 95% CI calculation of the leakage inflation gap (+2.48 pp, 95% CI [+1.73, +3.23 pp], p < 0.001).
  - Verify that the case study of specimen 50 and 53 in seed 456 (*Guibourtia arnoldiana* `.1` and `.3`, low Laplacian variance 113.8 & 152.1, misclassification into sister species *Guibourtia ehie*) is thoroughly and transparently discussed.
  - Verify that GRL discriminator loss logging is implemented and analyzed.
  - Verify that cross-split dependency (50.5% voucher overlap, Bengio & Grandvalet 2004) is properly acknowledged.

### 4. Integrity Forensic Checks
- Check for any hardcoding, dummy results, fake proofs, or integrity circumvention.
- Issue a definitive verdict: `CLEAN` or `INTEGRITY VIOLATION`.
- Save full evidence and audit report to `g:/S3_paper/.agents/teamwork/auditor_m3/handoff.md`.


## 2026-11-04T05:03:49Z
Received dispatch message from parent `ed8102ac-b450-41bf-b593-d0c904022dcd` to perform forensic integrity audit on `03_research_paper_specimen_invariance/` and `paper/main.tex` against Acceptance Criteria.


## 2026-11-04T05:24:45Z
Received re-audit directive for Gate Iteration 2 from orchestrator `ed8102ac-b450-41bf-b593-d0c904022dcd`.
Worker M2 reports 100% completion of remediation actions on `generate_figures.py`, `fig1_concept_invariant.pdf`, and `paper/main.pdf`.
Independent re-audit and empirical verification required.
