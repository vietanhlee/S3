import sys
from pathlib import Path

# Script thực hiện chuẩn hóa văn phong học thuật, giảm nhẹ tông giọng quảng bá/thậm xưng
# trên bản thảo paper_data/main.tex theo chuẩn Q1 Data in Brief / Scientific Data.

REPLACEMENTS = [
    (
        r"an operational canonical benchmark guaranteeing 100\% class coverage alongside an audited specimen-disjoint split",
        r"an operational canonical benchmark providing 100\% class coverage alongside an audited specimen-disjoint split"
    ),
    (
        r"an operational canonical benchmark guaranteeing 100\% Class Coverage Rate across all subsets for frontline inspection,",
        r"an operational canonical benchmark providing 100\% Class Coverage Rate across all subsets for frontline inspection,"
    ),
    (
        r"alongside rigorous multi-seed classification baselines, establishing empirical performance standards and modular code pipelines to ensure immediate adoption across forestry informatics and computer vision communities.",
        r"alongside systematic multi-seed classification baselines, establishing empirical performance standards and modular code pipelines to facilitate adoption across forestry informatics and computer vision communities."
    ),
    (
        r"established rigorous image curation but focused exclusively on plantation hardwoods",
        r"established systematic image curation but focused exclusively on plantation hardwoods"
    ),
    (
        r"Crucially, all 19 species achieve a 100\% Class Coverage Rate",
        r"All 19 species achieve a 100\% Class Coverage Rate"
    ),
    (
        r"enforcing complete block isolation on a single-specimen class mathematically renders validation or test coverage impossible.",
        r"enforcing complete block isolation on a single-specimen class structurally precludes validation or test coverage."
    ),
    (
        r"\item \textbf{Evaluation Specimen Leakage Rate ($\text{SLR}_{\text{eval}}$) and Image-Level Test Contamination ($\text{ILR}_{\text{test}}$)}: Crucially, the metric governing test-set generalization integrity is the train-to-test leakage:",
        r"\item \textbf{Evaluation Specimen Leakage Rate ($\text{SLR}_{\text{eval}}$) and Image-Level Test Contamination ($\text{ILR}_{\text{test}}$)}: In evaluating test-set generalization, the primary metric governing specimen separation is the train-to-test leakage:"
    ),
    (
        r"guaranteeing $\text{SLR}_{\text{eval}} = 0.0\%$ and $\text{ILR}_{\text{test}} = 0.0\%$.",
        r"yielding $\text{SLR}_{\text{eval}} = 0.0\%$ and $\text{ILR}_{\text{test}} = 0.0\%$."
    ),
    (
        r"operational customs screening benchmark guaranteeing 100\% taxonomic coverage",
        r"operational customs screening benchmark maintaining 100\% taxonomic coverage"
    ),
    (
        r"Allocations guarantee 100\% Class Coverage Rate ($\text{CCR}=100.0\%$)",
        r"Allocations provide 100\% Class Coverage Rate ($\text{CCR}=100.0\%$)"
    ),
    (
        r"to ensure dense operational class representation;",
        r"to provide dense operational class representation;"
    ),
    (
        r"establishing a definitive diagnostic morphological key that prevents cross-genus confusion",
        r"providing a distinct diagnostic morphological key that prevents cross-genus confusion"
    ),
    (
        r"informed by IAWA macroscopic and microscopic criteria~\cite{iawa1989} and authoritative xylotomical references~\cite{wiedenhoeft2011,insidewood}:",
        r"informed by IAWA macroscopic and microscopic criteria~\cite{iawa1989} and established xylotomical references~\cite{wiedenhoeft2011,insidewood}:"
    ),
    (
        r"are included as multimodal domain knowledge from authoritative references.",
        r"are included as multimodal domain knowledge from reference literature."
    ),
    (
        r"Conventional histological microtomy---the authoritative diagnostic standard---requires prolonged softening",
        r"Conventional histological microtomy---the established reference diagnostic standard---requires prolonged softening"
    ),
    (
        r"Our experimental workflow was engineered to guarantee optical fidelity and taxonomic authenticity",
        r"Our experimental workflow was engineered to maintain optical fidelity and taxonomic authenticity"
    ),
    (
        r"Crucially, dataset tiles were created by direct spatial cropping \textbf{without downsampling}, thereby preserving the physical scale of $(224 \times 12.0\,\mu\text{m}) \times (224 \times 12.0\,\mu\text{m}) \approx 2.69 \times 2.69\,\text{mm}$ ($7.23\,\text{mm}^2$). This ensures vessel pore diameters, ray widths, and parenchyma band intervals directly match physical millimeter dimensions under IAWA feature criteria.",
        r"Dataset tiles were created by direct spatial cropping \textbf{without downsampling}, thereby preserving the physical scale of $(224 \times 12.0\,\mu\text{m}) \times (224 \times 12.0\,\mu\text{m}) \approx 2.69 \times 2.69\,\text{mm}$ ($7.23\,\text{mm}^2$). This allows vessel pore diameters, ray widths, and parenchyma band intervals to directly match physical millimeter dimensions under IAWA feature criteria."
    ),
    (
        r"To establish rigorous ground-truth authenticity, all physical wood specimens",
        r"To verify ground-truth authenticity, all physical wood specimens"
    ),
    (
        r"Crucially, all initial ambiguities ($1.36\%$, exactly 87 captures across 3 physical blocks) were concentrated exclusively",
        r"All initial ambiguities ($1.36\%$, exactly 87 captures across 3 physical blocks) were concentrated exclusively"
    ),
    (
        r"To guarantee benchmark integrity and reproducibility, a rigorous quality control and governed partitioning pipeline was executed:",
        r"To verify benchmark integrity and reproducibility, a systematic quality control and governed partitioning pipeline was executed:"
    ),
    (
        r"guaranteeing total absence of bitwise image contamination.",
        r"confirming the absence of bitwise image contamination."
    ),
    (
        r"Enforcing rigid whole-specimen isolation across 3-way splits would mathematically exclude rare single-specimen taxa",
        r"Enforcing rigid whole-specimen isolation across 3-way splits would necessarily exclude rare single-specimen taxa"
    ),
    (
        r"guaranteeing 100\% Class Coverage Rate ($\text{CCR} = 100.0\%$, 19/19 taxa) across all subsets",
        r"providing 100\% Class Coverage Rate ($\text{CCR} = 100.0\%$, 19/19 taxa) across all subsets"
    ),
    (
        r"To rigorously confirm technical reproducibility and evaluate sensitivity to stochastic initialization",
        r"To evaluate technical reproducibility and assess sensitivity to stochastic initialization"
    ),
    (
        r"with perfect discrimination consistently maintained on \textit{Dalbergia melanoxylon}, \textit{Peltogyne pubescens}, and \textit{Sindora cochinchinensis}.",
        r"with complete discrimination ($100\%$ F1) consistently maintained on \textit{Dalbergia melanoxylon}, \textit{Peltogyne pubescens}, and \textit{Sindora cochinchinensis}."
    ),
    (
        r"Crucially, the lowest-performing classes in fine-grained multi-class classification are legally regulated under CITES Appendix~II:",
        r"In multi-class evaluation, the lowest-performing classes are legally regulated under CITES Appendix~II:"
    ),
    (
        r"Rather than an opaque statistical defect, this alignment proves that network predictions directly reflect genuine morphological convergence across Fabaceae clades.",
        r"Rather than an arbitrary classification artifact, this alignment indicates that network predictions reflect genuine morphological convergence across Fabaceae clades."
    ),
    (
        r"To ensure statistical rigor and eliminate pseudoreplication",
        r"To maintain statistical validity and eliminate pseudoreplication"
    ),
    (
        r"Consignments triggering border flags can subsequently undergo definitive confirmatory testing via laboratory microtomy or mass spectrometry.",
        r"Consignments triggering border flags can subsequently undergo formal confirmatory testing via laboratory microtomy or mass spectrometry."
    ),
    (
        r"guaranteeing complete out-of-distribution physical isolation across all remaining multi-specimen taxa",
        r"maintaining complete out-of-distribution physical isolation across all remaining multi-specimen taxa"
    ),
    (
        r"(ii) rigorous out-of-distribution forensic deployment across completely novel physical logs and harvest accessions",
        r"(ii) specimen-disjoint forensic evaluation across novel physical logs and harvest accessions"
    ),
    (
        r"proving high baseline representation separability for researchers without GPU fine-tuning resources;",
        r"demonstrating baseline representation separability for researchers without GPU fine-tuning resources;"
    ),
    (
        r"Crucially, this empirical limitation is explained by our inter-specimen biological variance audit across the 147 physical blocks",
        r"This empirical limitation is reflected in our inter-specimen biological variance audit across the 147 physical blocks"
    ),
    (
        r"(i) \textbf{Transverse Rotational Symmetry is Crucial}:",
        r"(i) \textbf{Transverse Rotational Invariance is Essential}:"
    ),
    (
        r"we emphasize a vital domain-specific methodological boundary:",
        r"we highlight a domain-specific methodological boundary:"
    ),
    (
        r"Genus-level F1-scores demonstrate exceptional reliability across all regulated and commercial groups:",
        r"Genus-level F1-scores demonstrate high reliability across all regulated and commercial groups:"
    ),
    (
        r"Crucially, resolving intra-genus confusion (where all 21 misclassified captures between \textit{A. pachyloba} and \textit{A. bella} remain correctly within the protected genus \textit{Afzelia}) elevates classification accuracy",
        r"Resolving intra-genus confusion (where all 21 misclassified captures between \textit{A. pachyloba} and \textit{A. bella} remain correctly within the protected genus \textit{Afzelia}) elevates classification accuracy"
    ),
    (
        r"Expanding coverage to these taxa remains a vital objective for community benchmarking.",
        r"Expanding coverage to these taxa remains an important objective for community benchmarking."
    ),
    (
        r"Crucially, the disjoint benchmark reveals an empirical generalization gap of $\Delta\text{Acc} = -8.52\%$",
        r"The disjoint benchmark reveals an empirical generalization gap of $\Delta\text{Acc} = -8.52\%$"
    )
]

def main():
    tex_path = Path(__file__).resolve().parent.parent / "main.tex"
    if not tex_path.exists():
        print(f"Error: {tex_path} does not exist!")
        sys.exit(1)

    with open(tex_path, "r", encoding="utf-8") as f:
        content = f.read()

    print(f"Loaded {tex_path} ({len(content)} characters)")
    missing_count = 0
    for idx, (target, repl) in enumerate(REPLACEMENTS, 1):
        if target not in content:
            print(f"[MISSING #{idx}] Could not find: {target[:70]}...")
            missing_count += 1

    if missing_count > 0:
        print(f"Aborting: {missing_count} targets missing.")
        sys.exit(1)

    print(f"All {len(REPLACEMENTS)} targets found! Applying replacements...")
    modified_content = content
    for idx, (target, repl) in enumerate(REPLACEMENTS, 1):
        modified_content = modified_content.replace(target, repl, 1)

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(modified_content)

    print(f"Successfully updated {tex_path}!")

if __name__ == "__main__":
    main()
