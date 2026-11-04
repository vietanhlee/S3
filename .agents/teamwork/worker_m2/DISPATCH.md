# Task Assignment for Worker M2

**Assigned Agent**: teamwork_worker
**Working Directory**: g:/S3_paper/.agents/teamwork/worker_m2/
**Context & Scope**: g:/S3_paper/.agents/teamwork/ORIGINAL_REQUEST.md
**Explorer Handoff Report**: g:/S3_paper/.agents/teamwork/explorer_m1/handoff.md
**Project Document**: g:/S3_paper/.agents/teamwork/orchestrator/PROJECT.md

## MANDATORY INTEGRITY WARNING
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

## 2026-11-04T04:19:22Z
[Message] timestamp=2026-11-04T04:19:22Z sender=ed8102ac-b450-41bf-b593-d0c904022dcd priority=MESSAGE_PRIORITY_HIGH content=You are assigned as the Scientific Code & Paper Worker (teamwork_worker).
Your working directory is: `g:/S3_paper/.agents/teamwork/worker_m2/`

You MUST read the following files before starting your work:
- `g:/S3_paper/.agents/teamwork/ORIGINAL_REQUEST.md`
- `g:/S3_paper/.agents/teamwork/worker_m2/DISPATCH.md`
- `g:/S3_paper/.agents/teamwork/explorer_m1/handoff.md`
- `g:/S3_paper/.agents/teamwork/orchestrator/PROJECT.md`

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

Your mission is to execute all code additions and paper refactoring for `03_research_paper_specimen_invariance/`:

1. Code Experiments & Diagnostics (R3):
   - In `03_research_paper_specimen_invariance/`:
     - Implement/run evaluation across all 9 seeds `[42, 123, 456, 789, 101, 202, 303, 404, 505]` for naive random split to compute the 95% Confidence Interval for the leakage inflation gap ($\Delta_{\text{inflation}} = \text{Acc}_{\text{naive}} - \text{Acc}_{\text{unseen}}$).
     - Incorporate the concrete analysis of specimen 50 and 53 in seed 456 (*Guibourtia arnoldiana* `.1` and `.3`, low Laplacian variance 113.8 & 152.1, misclassification into sister species *Guibourtia ehie*).
     - Implement logging of discriminator loss (`loss_specimen`), `loss_club`, `loss_species` over training epochs and analyze the optimization dynamics under `adv_weight`.
     - Explicitly note in the manuscript the partial dependency among the 9 splits (50.5% voucher overlap, Bengio & Grandvalet 2004) causing the cross-split CI to be optimistic.

2. Paper Refactoring & Contradiction Resolution (R1 & R2):
   - In `03_research_paper_specimen_invariance/paper/main.tex`:
     - Scientific Re-framing (R1):
       - Update Title, Abstract, Introduction, Contributions, Method, Discussion, and Conclusion to frame the paper as "Auditing Specimen Leakage and Evaluating Invariance Methods in Macroscopic Timber Forensics: An Empirical Benchmark and Honest Negative Results".
       - Define the 3 core contributions: (1) Quantifying specimen leakage in macroscopic timber forensics; (2) Voucher-level holdout protocol with cluster bootstrap; (3) Empirical evaluation showing GRL+vCLUB does not provide statistically significant improvement over ERM baseline, dissecting why (clear Fabaceae boundaries, accuracy ceiling ~97%, optimization trade-offs in GRL).
       - Report honest negative results: Acc diff -0.19 pp, 95% CI [-1.16, +0.78], p = 0.66 t-test, p = 0.91 Wilcoxon; SRI, GGSL, ECE CIs contain 0.
       - Unbold all table rows for GAN/adversarial where there is no statistically significant improvement.
     - Contradiction Resolution (R2):
       1. GGSL in Section 4.3: Baseline 2.74 pp vs GAN 2.82 pp (GRL does not eliminate GGSL).
       2. Train SRI: GAN 0.538 does not decrease compared to Baseline 0.526.
       3. ImageNet anchor: Remove self-contradictory logic between 0.5667, 0.5261, 0.5380.
       4. Baseline Acc in Section 4.2: Change "~98.5-99.0%" to exact table figure 97.07%.
       5. ECE: Change "< 1%" to 1.58% - 1.93%.
       6. Plant species: Remove *Dialium cochinchinense* and *Pterocarpus macrocarpus*; only mention the official 11 Fabaceae species of S3.
       7. Tone and Title: Remove all overclaiming terms.
       8. Third-party studies: Tone down claims, specify leakage inflation gap on S3 dataset without generalizing allegations.

3. LaTeX Compilation & Verification:
   - Compile `main.tex` in `03_research_paper_specimen_invariance/paper/`:
     ```powershell
     pdflatex -interaction=nonstopmode -halt-on-error main.tex
     bibtex main
     pdflatex -interaction=nonstopmode -halt-on-error main.tex
     pdflatex -interaction=nonstopmode -halt-on-error main.tex
     ```
   - Verify that `main.pdf` is generated cleanly with 0 fatal errors, 0 broken citations `[?]`, and 0 broken references `??`.

4. Output:
   - Write a detailed handoff report to `g:/S3_paper/.agents/teamwork/worker_m2/handoff.md`.
   - Send completion message to orchestrator (conversation ID: ed8102ac-b450-41bf-b593-d0c904022dcd).


## 2026-11-04T04:36:14Z
[Message] timestamp=2026-11-04T04:36:14Z sender=ed8102ac-b450-41bf-b593-d0c904022dcd priority=MESSAGE_PRIORITY_HIGH content=**Context**: Checking Worker M2 execution progress
**Content**: Hi Worker M2, please provide a brief status update on your progress: (1) 9-seeds random-split evaluation and 95% CI calculation, (2) specimen 50 & 53 diagnosis, (3) GRL loss logging, (4) refactoring paper/main.tex, and (5) LaTeX compilation. Please update your progress.md with the latest visited timestamp.
**Action**: Reply with your current status and update progress.md.

## 2026-11-04T04:56:32Z
[Message] timestamp=2026-11-04T04:56:32Z sender=ed8102ac-b450-41bf-b593-d0c904022dcd priority=MESSAGE_PRIORITY_HIGH content=**Context**: Checking Worker M2 progress
**Content**: Hi Worker M2, please give a quick status update on Bước 4 (benchmark script update / logging), Bước 5 (paper/main.tex refactoring), and Bước 6 (LaTeX compilation). Remember to update your progress.md timestamp.
**Action**: Reply with your current status.

## 2026-11-04T05:16:59Z
[Message] timestamp=2026-11-04T05:16:59Z sender=ed8102ac-b450-41bf-b593-d0c904022dcd priority=MESSAGE_PRIORITY_HIGH content=**Context**: Urgent Remediation Directive from Forensic Audit (Gate Iteration 1)
**Content**: The Forensic Auditor has issued an INTEGRITY VIOLATION verdict on the work product.
Full audit report: `g:/S3_paper/.agents/teamwork/auditor_m3/handoff.md`

Violations Identified:
1. In `paper/figures/fig1_concept_invariant.pdf` (embedded on page 4 of `main.pdf`):
   - Panel (b) Box 3 still displays old overclaim: "Specimen recoverability SRI -> 0.118 (Residual artifacts erased)" which directly contradicts table results (Train SRI is 0.538 vs baseline 0.526 and anchor 0.567).
   - Panel (b) Box 4 displays: "Closes Specimen Leakage Gap: Robust 89.25% Macro-F1 (90.45% LOSO_multi)" and "Evidence-grade, specimen-invariant representations for customs triage", which directly contradicts the honest negative result (GRL does not close the leakage gap, unseen Macro-F1 is 96.64%).
2. In `paper/generate_figures.py` (lines 341 and 348), the excluded species `Pterocarpus macrocarpus` is still present in the Taxa definitions.

Remediation Required Immediately:
1. Edit `03_research_paper_specimen_invariance/paper/generate_figures.py`:
   - In `generate_figure1()` Panel (b):
     - Update title: "Evaluated Specimen-Invariant Framework (Invariance Stress-Test)".
     - Box Z text: Change to reflect actual Train SRI (0.538 GRL vs 0.526 ERM, ImageNet Anchor 0.567; specimen info not purged).
     - Box Outcome text: Change to reflect honest negative result (Out-of-Specimen Acc 96.88% vs 97.07% Baseline; minimax conflict & Fabaceae ceiling).
   - In `generate_figure3()`: Remove `Pterocarpus macrocarpus` (replace with standard Fabaceae species like `Pterocarpus erinaceus`).
2. Run `python paper/generate_figures.py` to regenerate `paper/figures/fig1_concept_invariant.pdf` (and check that `0.118`, `89.25%`, and `LOSO_multi` are completely absent).
3. Re-run clean LaTeX compilation: `pdflatex` + `bibtex` + `pdflatex` x 2 to rebuild `paper/main.pdf`.
4. Update `progress.md` and reply when complete so Auditor M3 can perform the re-audit.
**Action**: Execute remediation immediately and reply with status.
