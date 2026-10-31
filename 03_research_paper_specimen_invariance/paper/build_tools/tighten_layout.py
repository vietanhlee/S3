# -*- coding: utf-8 -*-
with open('main.tex', 'r', encoding='utf-8') as f:
    text = f.read()

# Normalize line endings
text = text.replace('\r\n', '\n')

# 1. Figure 3 scale
text = text.replace('width=0.80\\textwidth]{figures/fig3_gradcam_comparison.pdf}', 'width=0.68\\textwidth]{figures/fig3_gradcam_comparison.pdf}')

# 2. Section 6.4 p1
old_p1 = r'''A defining operational challenge in automated botanical machine learning is the extreme scarcity of vouchered physical reference wood blocks for threatened tropical taxa. Reference timber specimens curated in accredited scientific xylaria (such as MADw in the United States, SJRw, and BFH in Germany) constitute irreplaceable, legally protected cultural and botanical heritage. For species listed under CITES Appendix~I and II, international cross-border exchange of physical reference wood blocks is tightly restricted under \textbf{CITES Article VII(6)} (Exemption for Scientific Exchange between registered scientific institutions). Securing Article VII(6) transfer certificates and bilateral import/export authorizations is a protracted administrative process that can span several years. Consequently, individual scientific xylaria worldwide curate only 1 to 10 physical vouchered blocks per rare tropical taxon (e.g., \textit{Dalbergia cochinchinensis} or \textit{Afzelia pachyloba}), which directly explains the singleton voucher constraints present in scientific collections.'''
new_p1 = r'''A defining operational challenge in botanical machine learning is the scarcity of vouchered physical reference wood blocks for threatened tropical taxa. Reference timber specimens curated in accredited scientific xylaria (e.g., MADw, SJRw, BFH) represent irreplaceable botanical heritage. For CITES Appendix~I and II taxa, international cross-border transfer of wood blocks is tightly restricted under \textbf{CITES Article VII(6)} (Scientific Exchange Exemption), requiring multi-year administrative authorization. Consequently, accredited xylaria worldwide typically hold only 1 to 10 physical vouchered blocks per rare taxon (e.g., \textit{Dalbergia cochinchinensis}, \textit{Afzelia pachyloba}), directly explaining the singleton voucher constraint in biological collections.'''
if old_p1 in text:
    text = text.replace(old_p1, new_p1)
    print("Replaced p1")

# 3. Section 6.4 synthetic holdout
old_synth = r'''\noindent \textbf{Synthetic Domain Shift Holdout Benchmark}: In the absence of multi-institutional physical specimen exchanges due to CITES Article VII(6) transfer constraints, we explicitly simulated severe cross-collection domain shifts through our synthetic perturbation stress-test benchmark (Section~\ref{sec:experiments}.5, Table~\ref{tab:robustness_study}). When evaluated under compound multi-factor field degradation (simultaneous extreme shadow illumination, directional circular saw ridges, abrasion blur, and arbitrary grain orientation), the standard empirical risk minimization baseline collapses to $51.14\%$ ($61.5\%$ retention) and Unconditional DANN to $44.71\%$. In contrast, our Specimen-Invariant framework maintains $\mathbf{73.46\%}$ Top-1 accuracy ($\mathbf{81.5\%}$ retention, $+22.32$~pp higher than baseline). This empirical holdout demonstrates that by actively purging specimen-level surface artifacts during training, the network acquires fundamental resilience against the primary optical and physical domain shifts encountered across distinct collections.'''
new_synth = r'''\noindent \textbf{Synthetic Domain Shift Holdout Benchmark}: In the absence of physical multi-institution specimen exchanges under CITES Article VII(6), we simulated severe cross-collection domain shifts via our synthetic stress benchmark (Section~\ref{sec:experiments}.5, Table~\ref{tab:robustness_study}). Under compound multi-factor field stress (simultaneous shadow illumination, circular saw ridges, abrasion blur, and grain rotation), standard ERM collapses to $51.14\%$ ($61.5\%$ retention) and Unconditional DANN to $44.71\%$. In contrast, our Specimen-Invariant framework maintains $\mathbf{73.46\%}$ Top-1 accuracy ($\mathbf{81.5\%}$ retention, $+22.32$~pp over baseline), demonstrating that purging specimen-level surface shortcuts confers resilience against optical and physical domain shifts.'''
if old_synth in text:
    text = text.replace(old_synth, new_synth)
    print("Replaced synth")

# 4. Section 6.4 scope
old_scope = r'''\noindent \textbf{Scope and Contextualization of Generalization Claims}: While our curated benchmark of 147 physical blocks across 19 taxa represents the most comprehensive macroscopic tropical timber dataset evaluated under rigorous 5-fold Leave-One-Specimen-Out cross-validation ($\mathrm{SLR} \equiv 0.0\%$) to date, we candidly contextualize our findings: evaluations within a single reference archival collection cannot fully substitute for comprehensive cross-continental, multi-institution xylarium validation. Evaluating models across distinct international xylaria (e.g., training on Asian reference collections and evaluating on European or North American vouchered collections) remains an essential and highly anticipated frontier of empirical validation as international scientific transfer permits under CITES Article VII(6) are secured.'''
new_scope = r'''\noindent \textbf{Scope and Contextualization of Generalization Claims}: While our curated benchmark of 147 physical blocks across 19 taxa is the most comprehensive macroscopic timber dataset evaluated under strict 5-fold Leave-One-Specimen-Out cross-validation ($\mathrm{SLR} \equiv 0.0\%$) to date, we candidly emphasize that evaluations within a single reference collection cannot fully substitute for cross-continental, multi-institution xylarium validation. Evaluating models across distinct international xylaria remains an essential long-term frontier as bilateral scientific transfer permits under CITES Article VII(6) are secured.'''
if old_scope in text:
    text = text.replace(old_scope, new_scope)
    print("Replaced scope")

# 5. Section 6.5 limitations
old_limit = r'''While our framework demonstrates robust specimen invariance across cross-sectional macroscopic imagery, several avenues warrant future inquiry. First, timber products frequently transit international borders as finished veneers, acoustic guitars, or charcoal, where transverse end-grain surfaces are inaccessible. Extending invariant representation learning to longitudinal anatomical planes (radial and tangential surfaces)~\cite{rosadasilva2022} and integrating non-destructive spectroscopy (Near-Infrared Spectroscopy, NIRS, or Direct Analysis in Real Time Time-of-Flight Mass Spectrometry, DART-TOFMS)~\cite{dormontt2015} represents a promising multi-modal frontier. Second, exploring open-set out-of-distribution detection to identify uncatalogued tropical wood species will further enhance deployment safety in frontier ports.'''
new_limit = r'''While our framework demonstrates robust specimen invariance on transverse cross-sections, several frontiers warrant future inquiry: (1)~extending invariant representation learning to longitudinal anatomical planes (radial and tangential surfaces)~\cite{rosadasilva2022} and integrating non-destructive spectroscopy (NIRS or DART-TOFMS)~\cite{dormontt2015} for finished timber products; and (2)~exploring open-set out-of-distribution detection to identify uncatalogued tropical timbers in frontier ports.'''
if old_limit in text:
    text = text.replace(old_limit, new_limit)
    print("Replaced limit")

# 6. Data and code availability
old_data = r'''\section*{Data and Code Availability Statement}
To ensure complete scientific reproducibility and support international customs enforcement efforts, all source code, model configurations, training pipelines, baseline implementations, and evaluation probe scripts are publicly available on GitHub at \url{https://github.com/IC4SD-Lab/Specimen-Invariant-Wood-Forensics}. The curated 19-species S3 Wood Benchmark metadata, split indices, and sample imagery are permanently archived on Zenodo (DOI: 10.5281/zenodo.14889210). Pretrained model checkpoints and training logs are hosted under open-access scientific licenses.'''
new_data = r'''\section*{Data and Code Availability}
Source code, model configurations, training pipelines, baseline implementations, and probe scripts are available on GitHub at \url{https://github.com/IC4SD-Lab/Specimen-Invariant-Wood-Forensics}. Benchmark metadata, split indices, and sample imagery are archived on Zenodo (DOI: 10.5281/zenodo.14889210). Pretrained model checkpoints and training logs are openly accessible.'''
if old_data in text:
    text = text.replace(old_data, new_data)
    print("Replaced data")

with open('main.tex', 'w', encoding='utf-8') as f:
    f.write(text)
print("Finished!")
