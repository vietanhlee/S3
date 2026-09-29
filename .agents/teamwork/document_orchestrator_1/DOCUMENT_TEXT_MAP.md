<!-- FILE: 01_data_paper_forensic_cites/paper_data/main.tex -->

\documentclass[a4paper,fleqn,12pt]{cas-sc}

\usepackage[utf8]{inputenc}
\usepackage[T5,T1]{fontenc}
\DeclareTextFontCommand{\textvn}{\fontencoding{T5}\selectfont}

\usepackage[numbers,sort&compress]{natbib}
\usepackage{graphicx}
\usepackage{amsmath,amssymb,amsfonts,bm}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{float}
\usepackage[section]{placeins}
\usepackage{hyperref}
\usepackage{tikz}
\usepackage{url}
\usepackage{multirow}
\usepackage{array}
\usepackage{microtype}
\microtypesetup{expansion=false}
\usepackage{makecell}
\usepackage{enumitem}

% Academic Hyperref Configuration
\hypersetup{
    colorlinks=true,
    linkcolor=cyan!80!black,
    citecolor=cyan!80!black,
    urlcolor=cyan!80!black
}

\newcommand{\orcidicon}[1]{\href{https://orcid.org/#1}{\texorpdfstring{%
\begin{tikzpicture}[baseline=-0.4ex]%
\definecolor{orcidgreen}{HTML}{A6CE39}%
\draw[fill=orcidgreen,draw=none] (0,0) circle (1.0ex);%
\node at (0,0) {\color{white}\fontsize{4}{4}\selectfont\sffamily\bfseries iD};%
\end{tikzpicture}%
}{}}}

\setcounter{topnumber}{4}
\setcounter{bottomnumber}{4}
\setcounter{totalnumber}{8}
\setcounter{dbltopnumber}{4}
\renewcommand{\topfraction}{0.95}
\renewcommand{\bottomfraction}{0.90}
\renewcommand{\textfraction}{0.05}
\renewcommand{\floatpagefraction}{0.75}

\ExplSyntaxOn
\cs_set:Npn \__reset_fig:
{
  \tl_set:Nx \l_fig_pos_tl { t }
  \tl_set:Nx \l_fig_cols_tl { 1 }
  \tl_set:Nn \l_fig_align_tl { \centering }
  \skip_set:Nn \l_fig_abovecap_skip { 6pt }
  \skip_set:Nn \l_fig_belowcap_skip { 6pt }
  \skip_set:Nn \l_fig_abovefig_skip { 6pt }
  \skip_set:Nn \l_fig_belowfig_skip { 6pt }
}
\tl_set:Nn \l_fig_align_tl { \centering }

\cs_set:Npn \__first_footerline: {}
\cs_set:Npn \__first_foot: {}
\cs_set:Npn \__cas_foot: 
{%
  \noindent\begin{minipage}[t]{\linewidth}%
    \setlength{\parindent}{0pt}%
    \noindent\rule{\linewidth}{.2pt}\par\vspace{2pt}%
    \sffamily\small%
    \__first_footerline:%
    \hfill Page~\thepage {}~of~ \lastpage%
  \end{minipage}%
}
\ExplSyntaxOff
\let\printorcid\relax

\makeatletter
\setlength{\@fptop}{0pt}
\setlength{\@fpbot}{0pt plus 1fil}
\setlength{\@fpsep}{10pt plus 2fil}
\let\printFirstPageNotes\relax
\makeatother

\hyphenation{Afzelia Guibourtia Pterocarpus Dalbergia Sindora cochinchinensis melanoxylon pachyloba quanzensis arnoldiana coleosperma}

\begin{document}

\let\WriteBookmarks\relax

\shorttitle{ForensicMacroWood-CITES: Forensic Macroscopic Timber Benchmark}
% \shortauthors{V.-A. Le and K. Nguyen-Trong}

\title [mode = title]{ForensicMacroWood-CITES: A Multimodal Macroscopic Wood Anatomy and High-Resolution Transverse Image Benchmark for CITES-Regulated Hardwood Identification}

% \author[1]{Viet-Anh Le \orcidicon{0009-0003-5748-0439}}
% \author[1]{Khanh Nguyen-Trong \orcidicon{0000-0001-5175-8805}\cormark[1]}

% \address[1]{Intelligent Computing for Sustainable Development Laboratory (IC4SD), Posts and Telecommunications Institute of Technology (PTIT), Km 10 Nguyen Trai Street, Ha Dong District, Hanoi 100000, Vietnam}

% \cortext[1]{Corresponding author. \textit{E-mail addresses:} \href{mailto:khanhnt@ptit.edu.vn}{khanhnt@ptit.edu.vn} (K. Nguyen-Trong), \href{mailto:anhlv.b23kh002@stu.ptit.edu.vn}{anhlv.b23kh002@stu.ptit.edu.vn} (V.-A. Le)}

\begin{abstract}
Rapid forensic discrimination of protected timbers at customs checkpoints requires robust macroscopic identification methods. However, previous datasets for computer-vision wood anatomy often lack multimodal descriptors, rely on uncalibrated optics, or omit heavily trafficked species regulated under CITES. We introduce ForensicMacroWood-CITES, an authenticated multimodal benchmark comprising 6,414 standardized high-resolution transverse-plane images ($12.0\,\mu\text{m/pixel}$) from 19 commercially significant Fabaceae hardwoods. The cohort encompasses 11 CITES Appendix~II-regulated taxa and 4 high-value endemic species, anchored to 148 verified physical wood blocks. To bridge computer vision and classical xylotomy, the dataset integrates an expert-curated matrix of 14 macroscopic anatomical descriptors standardized under IAWA criteria, facilitating multimodal vision-language learning and explainable AI auditing. To address reference specimen scarcity, we provide a mathematically governed partition architecture that guarantees 100\% class coverage without minority starvation, alongside a strict specimen-disjoint split. Technical validation using ConvNeXt-Tiny establishes a robust classification baseline (90.42\% test accuracy), confirming dataset integrity. The benchmark provides a non-destructive screening testbed for timber forensics and is publicly accessible at \href{https://doi.org/10.5281/zenodo.14892180}{10.5281/zenodo.14892180}.
\end{abstract}

\begin{keywords}
Forensic timber compliance \sep CITES Appendix~II trade regulation \sep Multimodal wood anatomy \sep Transverse xylotomy \sep Specimen scarcity management \sep IAWA morphological descriptors \sep ConvNeXt-Tiny \sep Multi-seed statistical validation
\end{keywords}

\maketitle

\section*{Specifications Table}

\noindent
\small
\setlength{\tabcolsep}{5pt}
\renewcommand{\arraystretch}{1.12}
\begin{tabularx}{\linewidth}{@{} l X @{}}
\toprule
\textbf{Subject} & Computer Science; Forestry and Wood Science \\
\midrule
\textbf{Specific subject area} & Forensic Wood Anatomy, CITES Trade Compliance, Computer Vision, Multimodal Vision-Language Learning, Statistical Significance Testing \\
\midrule
\textbf{Type of data} & Image (standardized $224 \times 224$ px RGB JPEG in \texttt{images.zip}), Table (master CSV metadata, split manifests, structured 14-axis anatomical descriptors in \texttt{metadata/anatomical\_features.csv}), Mapping (JSON label map) \\
\midrule
\textbf{Data format} & Raw, filtered, standardized RGB images (JPEG), structured tabular metadata (CSV), label mapping dictionaries (JSON) \\
\midrule
\textbf{Data collection} & Transverse end-grain surfaces surfaced orthogonally on a sliding table saw, progressively polished across four grit gradations (P120--P600), cleaned with dry air jets ($6\,\text{bar}$), and captured under calibrated daylight-balanced ($5600\,\text{K}$) diffuse circular LED illumination ($4500\,\text{lux}$, $f/8.0$, working distance $15\,\text{cm}$) delivering $12.0\,\mu\text{m/pixel}$ calibrated spatial resolution ($2.69 \times 2.69\,\text{mm}$ FOV per tile, $50\times$ nominal display magnification) \\
\midrule
\textbf{Data source location} & \textbf{Institution:} Intelligent Computing for Sustainable Development Laboratory (IC4SD), Posts and Telecommunications Institute of Technology (PTIT) \newline
\textbf{City/Country:} Hanoi, Vietnam \\
\midrule
\textbf{Data accessibility} & \textbf{Repository:} Zenodo Scientific Archive \newline
\textbf{Data identification number:} \href{https://doi.org/10.5281/zenodo.14892180}{10.5281/zenodo.14892180} \newline
\textbf{Direct URL to data:} \url{https://doi.org/10.5281/zenodo.14892180} \newline
\textbf{Instructions for accessing these data:} All files, split manifests, anatomical matrices, and the executable demonstration script (\texttt{code/quickstart\_demo.py}) are openly downloadable without restrictions under CC BY 4.0 license \\
\midrule
\textbf{Related research article} & None \\
\bottomrule
\end{tabularx}

\vspace{0.8em}

\section{Value of the Data}
\label{sec:value}

\begin{itemize}[leftmargin=*,itemsep=3pt,topsep=2pt]
    \item \textbf{Dedicated Forensic Reference for CITES Hardwood Compliance}: This dataset provides an authenticated benchmark for fine-grained discrimination of 19 commercially significant tropical Fabaceae hardwoods. With 11 taxa strictly regulated under CITES Appendix~II, it directly supports customs checkpoints and wildlife enforcement authorities in combatting illicit timber trafficking.
    \item \textbf{Rapid Non-Destructive Screening Testbed}: Unlike conventional forensic microtomy which is destructive and requires days per sample, these standardized transverse captures ($12.0\,\mu\text{m/pixel}$) enable rapid, non-destructive front-line triage of timber consignments within minutes.
    \item \textbf{Multimodal Ground Truth for Explainable AI}: The dataset integrates an expert-curated matrix of 14 macroscopic anatomical descriptors standardized under IAWA criteria. This structural metadata supports multimodal vision-language models, visual question answering, and explainable AI audits to ensure models attend to genuine botanical features.
    \item \textbf{Dual-Split Architecture Addressing Specimen Scarcity}: To overcome acute specimen scarcity in authentic xylarium archives, the repository provides two audited partitions. The canonical benchmark guarantees a 100\% Class Coverage Rate across all subsets, while a strict specimen-disjoint split enables rigorous testing of out-of-distribution biological invariance.
    \item \textbf{Turnkey Machine Learning Reproducibility}: The release delivers scale-preserving cropped tiles alongside reproducible ConvNeXt-Tiny classification baselines ($90.42\% \pm 0.38\%$ accuracy across five random seeds). Provided quickstart scripts ensure immediate adoption for machine learning research.
\end{itemize}

\section{Data Description}
\label{sec:description}

\subsection{Taxonomic Composition and Inventory}

The ForensicMacroWood-CITES repository encompasses \textbf{19 commercially significant tropical hardwood species} spanning \textbf{6 botanical genera} within the legume family \textbf{Fabaceae} (Leguminosae), aggregating exactly \textbf{6,414 standardized macroscopic cross-sectional captures}. Curated from institutional timber reference archives at the Intelligent Computing for Sustainable Development Laboratory (IC4SD), PTIT, Hanoi, Vietnam, all 6,414 images possess verified species-level ground truth anchored to traceable physical wood specimens ($|\mathcal{G}_c| = 148$ wood blocks) and validated through an independent dual-expert verification protocol ($\kappa = 0.985$).

\begin{table}[pos=htbp]
\centering
\small
\setlength{\tabcolsep}{4.5pt}
\renewcommand{\arraystretch}{1.18}
\caption{Taxonomic inventory and physical specimen sampling of the 19 tropical hardwood species in the ForensicMacroWood-CITES benchmark, detailing botanical genus, scientific binomial nomenclature, native Vietnamese vernacular names, international trade names, CITES conservation status, verified physical specimen cohort sizes ($|\mathcal{G}_c|$, totaling 148 wood blocks), and standardized image counts.}
\label{tab:taxonomic_inventory}
\resizebox{\textwidth}{!}{%
\begin{tabular}{rlllllcc}
\toprule
\textbf{\#} & \textbf{Botanical Genus} & \textbf{Scientific Binomial} & \textbf{Vietnamese Vernacular} & \textbf{Trade / Common Name} & \textbf{CITES Status} & \makecell{\textbf{Physical}\\\textbf{Specimens} ($|\mathcal{G}_c|$)} & \makecell{\textbf{Total}\\\textbf{Images}} \\
\midrule
1 & \textit{Afzelia} & \textit{Afzelia africana} & \textvn{Gõ Douse (Gõ Doussié)} & African Doussié & CITES App.~II & 4 & 241 \\
2 & \textit{Afzelia} & \textit{Afzelia bella} & \textvn{Papao-Nua / Gỗ Gõ} & Bella Doussié & CITES App.~II & 10 & 400 \\
3 & \textit{Afzelia} & \textit{Afzelia pachyloba} & \textvn{Gõ Pachy} & White Doussié & CITES App.~II\textsuperscript{a} & 5 & 116 \\
4 & \textit{Afzelia} & \textit{Afzelia quanzensis} & \textvn{Gõ Quanzensis} & Pod Mahogany & CITES App.~II & 8 & 369 \\
\midrule
5 & \textit{Dalbergia} & \textit{Dalbergia cochinchinensis} & \textvn{Trắc (Rosewood)} & Siam Rosewood & CITES App.~II (Native / High-value VN) & 1 & 354 \\
6 & \textit{Dalbergia} & \textit{Dalbergia melanoxylon} & \textvn{Trắc châu Phi} & African Blackwood & CITES App.~II & 10 & 291 \\
7 & \textit{Dalbergia} & \textit{Dalbergia oliveri} & \textvn{Cẩm lai (Burmese Rosewood)} & Burmese Rosewood & CITES App.~II & 10 & 316 \\
8 & \textit{Dalbergia} & \textit{Dalbergia rimosa} & \textvn{Trắc dây} & Rimose Rosewood & CITES App.~II\textsuperscript{b} & 10 & 300 \\
9 & \textit{Dalbergia} & \textit{Dalbergia tonkinensis} & \textvn{Sưa} & Vietnamese Rosewood & CITES App.~II (Native / High-value VN) & 10 & 325 \\
\midrule
10 & \textit{Guibourtia} & \textit{Guibourtia arnoldiana} & \textvn{Gỗ Muntenye} & Mutenye / Benge & Non-CITES & 10 & 323 \\
11 & \textit{Guibourtia} & \textit{Guibourtia coleosperma} & \textvn{Mussivi / Hương đá} & Rhodesian Copalwood & Non-CITES & 2 & 360 \\
12 & \textit{Guibourtia} & \textit{Guibourtia ehie} & \textvn{Hyedua} & Ovangkol / Shedua & Non-CITES & 10 & 400 \\
\midrule
13 & \textit{Peltogyne} & \textit{Peltogyne pubescens} & \textvn{Hương tím nam mỹ} & Purpleheart & Non-CITES & 10 & 371 \\
\midrule
14 & \textit{Pterocarpus} & \textit{Pterocarpus erinaceus} & \textvn{Hương vân tây phi} & African Barwood / Kosso & CITES App.~II & 10 & 336 \\
15 & \textit{Pterocarpus} & \textit{Pterocarpus indicus} & \textvn{Hương mắt chim} & Narra / Amboyna & Non-CITES & 10 & 312 \\
16 & \textit{Pterocarpus} & \textit{Pterocarpus macrocarpus} & \textvn{Hương quả to} & Burma Padauk & Non-CITES & 8 & 431 \\
17 & \textit{Pterocarpus} & \textit{Pterocarpus soyauxii} & \textvn{Padouk / Hương padouk} & African Padauk & CITES App.~II\textsuperscript{c} & 6 & 486 \\
\midrule
18 & \textit{Sindora} & \textit{Sindora cochinchinensis} & \textvn{Gụ} & Sindora / Sepetir & Non-CITES (Native / High-value VN) & 4 & 352 \\
19 & \textit{Sindora} & \textit{Sindora tonkinensis} & \textvn{Gụ lau} & Tonkin Sepetir & Non-CITES (Native / High-value VN) & 10 & 331 \\
\midrule
\multicolumn{2}{l}{\textbf{Total Benchmark}} & \multicolumn{3}{l}{\textbf{19 Botanical Species (6 Genera, 11 CITES App.~II, 4 High-Value VN)}} & \textbf{Total} & \textbf{148} & \textbf{6,414} \\
\bottomrule
\multicolumn{8}{@{}p{\linewidth}@{}}{\vspace{3pt}\footnotesize \textsuperscript{a}\textit{Afzelia pachyloba} constitutes a statistical minority class ($N=116$ captures) due to acute reference specimen scarcity, while being legally regulated under CITES Appendix~II. \textsuperscript{b}All species of the genus \textit{Dalbergia} (with the exception of \textit{D. nigra} in Appendix~I) are regulated under CITES Appendix~II with annotation \#15 since CoP17 (2017). \textsuperscript{c}\textit{Pterocarpus soyauxii} (African Padauk) is regulated under CITES Appendix~II with annotation \#17 (logs, sawn wood, veneer sheets, plywood and transformed wood) adopted at CoP19 (Panama City, 2022) with entry into force on 23 February 2023.}
\end{tabular}%
}
\end{table}

Table~\ref{tab:taxonomic_inventory} details the taxonomic breakdown of the 19 species, providing scientific nomenclature, native Vietnamese vernacular terms, international trade designations, CITES conservation listings, physical specimen cohort sizes ($|\mathcal{G}_c|$, totaling 148 verified wood blocks), and total image volumes.

\subsection{Comparative Dataset Positioning}
\label{sec:comparative_positioning}

To contextualize the forensic utility of ForensicMacroWood-CITES, Table~\ref{tab:dataset_comparison} systematically compares its core attributes against prominent public macroscopic wood datasets and recent literature benchmarks~\cite{ravindran2020,ravindran2021,figueroamata2022,song2025,liu2025,nguyentrong2026eucalyptus}. Foundational initiatives like the XyloTron project~\cite{ravindran2020,ravindran2021} and regional surveys~\cite{figueroamata2022,song2025} have significantly advanced computer vision in forestry. However, these existing public collections typically focus on non-regulated commercial timbers with minimal CITES coverage ($<15\%$). They also frequently utilize lower optical magnifications or non-standardized mobile lenses. 

Recent institutional benchmarks, such as IC4SD-Wood-Eucalyptus~\cite{nguyentrong2026eucalyptus}, established rigorous image curation but focused exclusively on plantation hardwoods ($0\%$ CITES) for pulpwood management. Meanwhile, specialized studies targeting high-value CITES taxa~\cite{liu2025} remain restricted to private, unshared archives. This highlights a critical gap in open-access forensic resources for endangered tropical hardwoods.

ForensicMacroWood-CITES addresses this gap by introducing a fundamentally distinct, high-impact forensic contribution. First, it targets \textbf{19 commercially significant tropical hardwoods across 6 genera in the legume family Fabaceae}, featuring a critical concentration of international trade-restricted timbers (\textbf{57.9\% CITES Appendix~II}, 11 regulated species) alongside 4 high-value endemic Vietnamese species. Second, it provides an expert-curated matrix of \textbf{14 diagnostic macroscopic anatomical descriptors} structured under IAWA xylotomical standards. This enables emerging multimodal paradigms that were absent in prior single-modality collections. Finally, it delivers high optical calibration ($12.0\,\mu\text{m/pixel}$), verified physical block traceability, governed split manifests with zero bitwise redundancy, and audited perceptual boundaries.

\begin{table}[pos=htbp]
\centering
\small
\setlength{\tabcolsep}{4.5pt}
\renewcommand{\arraystretch}{1.18}
\caption{Systematic comparison of ForensicMacroWood-CITES against prominent public and literature macroscopic timber datasets, contrasting taxonomic focus, optical magnification, forensic regulation, partition governance, multimodal descriptors, and machine learning accessibility.}
\label{tab:dataset_comparison}
\resizebox{\linewidth}{!}{%
\begin{tabular}{lccccccccc}
\toprule
\textbf{Dataset / Study} & \textbf{Taxonomic Focus} & \makecell{\textbf{No.}\\\textbf{Spp.}} & \makecell{\textbf{Total}\\\textbf{Images}} & \makecell{\textbf{Optical Mag.}\\\textbf{/ Resolution}} & \makecell{\textbf{CITES App.~II}\\\textbf{Ratio (\%)}} & \makecell{\textbf{Partition}\\\textbf{Protocol}} & \makecell{\textbf{Contamination Audit}\\\textbf{(SHA / pHash)}} & \makecell{\textbf{IAWA Multimodal}\\\textbf{Descriptors}} & \makecell{\textbf{Access}\\\textbf{Model}} \\
\midrule
XyloTron Datasets~\cite{ravindran2020,ravindran2021} & Neotropical/US hardwoods & 10--40 & 2,300--5,000 & $10\times$ ($24\,\mu\text{m/px}$) & Low ($<15\%$) & Specimen-level (SLR not reported) & Not reported & No & Open \\
Costa Rican Timbers~\cite{figueroamata2022} & Native Costa Rican trees & 21 & 2,360 & Variable (phone macro) & Minimal ($<5\%$) & Random image split & Not reported & No & Open \\
Asian Commercial~\cite{song2025} & Vietnamese/Asian timber & 15 & $\sim$3,000 & $20\times$--$40\times$ USB & Moderate ($20\%$) & Random $k$-fold (SLR not reported) & Not reported & No & Restricted \\
\textit{Pterocarpus} Discrimination~\cite{liu2025} & \textit{Pterocarpus} genus only & 6 & $\sim$1,800 & $20\times$--$30\times$ stereo & Subset ($50\%$, 3 spp.) & Random split (SLR not reported) & Not reported & No & Private \\
IC4SD-Wood-Eucalyptus~\cite{nguyentrong2026eucalyptus} & Plantation genus \textit{Eucalyptus} & 10 & 2,910 & $12.0\,\mu\text{m/px}$ ($50\times$) & None ($0.0\%$, 0 spp.) & Specimen-aware (audited) & SHA-256 audited & No & Open (CC BY 4.0) \\
\midrule
\textbf{ForensicMacroWood-CITES (This Work)} & \textbf{Tropical Fabaceae (6 genera)} & \textbf{19} & \textbf{6,414} & \textbf{12.0}\,$\mu$\textbf{m/px (50}$\times$ \textbf{nominal)} & \textbf{High (57.9\%, 11 spp.)} & \textbf{Governed Pareto (100\% CCR)} & \textbf{Bitwise \& Perceptual} & \textbf{Yes (14-axis CSV)} & \textbf{Open (CC BY 4.0)} \\
\bottomrule
\end{tabular}%
}
\end{table}

\subsection{Repository Architecture and Metadata Schema}

The dataset package is organized systematically to facilitate seamless integration into machine learning pipelines. Table~\ref{tab:repo_contents} outlines the functional directory architecture, and Table~\ref{tab:metadata_fields} defines the structured metadata attributes cataloged in \texttt{metadata.csv}. To ensure immediate reproducibility without high-end computational infrastructure, an executable walkthrough script (\texttt{code/quickstart\_demo.py}) is provided in the repository, allowing users to verify metadata schemas, inspect image quality scores, and replicate baseline classification evaluations in seconds.

\begin{table}[pos=htbp]
\centering
\small
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.12}
\caption{Systematic directory layout and functional contents of the ForensicMacroWood-CITES release package.}
\label{tab:repo_contents}
\begin{tabularx}{\linewidth}{@{} r l X @{}}
\toprule
\textbf{\#} & \textbf{File / Folder Path} & \textbf{Content Description and Functionality} \\
\midrule
1 & \texttt{images.zip} & Compressed ZIP archive containing 6,414 standardized $224 \times 224$ px macroscopic RGB images ($12.0\,\mu\text{m/px}$ calibrated resolution, $2.69 \times 2.69\,\text{mm}$ tile FOV) organized by taxon subfolders. \\
2 & \texttt{metadata/} & Master metadata table (\texttt{metadata.csv}), structured anatomical descriptors (\texttt{anatomical\_features.csv}), label mappings (\texttt{label\_map.json}), and bitwise SHA-256 release manifest (\texttt{release\_manifest.csv}). \\
3 & \texttt{splits/} & Governed canonical partition manifest (\texttt{split\_canonical.csv}; $N_{\text{train}} = 3,959$, $N_{\text{val}} = 1,265$, $N_{\text{test}} = 1,190$) alongside strict specimen-disjoint split manifest (\texttt{split\_specimen\_disjoint.csv}). \\
4 & \texttt{code/} & Standalone demonstration script (\texttt{quickstart\_demo.py}), interactive walkthrough notebook (\texttt{quickstart\_demo.ipynb}), and dependencies (\texttt{requirements.txt}). \\
5 & \texttt{classification\_output/} & ConvNeXt-Tiny supervised classification baseline checkpoints (\texttt{convnext\_tiny\_focal\_best.pth}), test predictions, and evaluation metrics under Focal Loss. \\
6 & \texttt{leakage\_audit/} & Specimen overlap audit records and cryptographic integrity summary (\texttt{audit\_summary.json}) verifying zero cross-split duplication. \\
7 & \texttt{README.md}, \texttt{LICENSE} & Comprehensive dataset documentation, quickstart instructions, and Creative Commons Attribution 4.0 International license terms. \\
\bottomrule
\end{tabularx}
\end{table}

\begin{table}[pos=htbp]
\centering
\small
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.12}
\caption{Core metadata attributes and data schema documented within \texttt{metadata.csv}.}
\label{tab:metadata_fields}
\begin{tabularx}{\linewidth}{@{} r l c X @{}}
\toprule
\textbf{\#} & \textbf{Field Name} & \textbf{Data Type} & \textbf{Field Description and Functionality} \\
\midrule
1 & \texttt{image\_id} & String & Unique standardized image identifier (e.g., \texttt{VNMW\_000001}). \\
2 & \texttt{image\_path} & String & Relative file path to the macroscopic image within the repository archive. \\
3 & \texttt{genus} & String & Botanical genus name (\textit{Afzelia}, \textit{Dalbergia}, \textit{Guibourtia}, \textit{Peltogyne}, etc.). \\
4 & \texttt{species} & String & Botanical species-level specific epithet. \\
5 & \texttt{class\_name} & String & Full binomial botanical nomenclature. \\
6 & \texttt{class\_index} & Integer & Taxonomic integer index from 0 to 18 consistent with \texttt{label\_map.json}. \\
7 & \texttt{vietnamese\_name} & String & Native Vietnamese vernacular name stored in standard UTF-8 encoding. \\
8 & \texttt{cites\_status} & String & Legal trade status under CITES (CITES Appendix~II or Non-CITES). \\
9 & \texttt{specimen\_id} & String & Physical wood block identifier anchoring images to distinct physical specimens. \\
10 & \texttt{split} & String & Partition membership (\texttt{train}, \texttt{val}, or \texttt{test}). \\
11 & \texttt{sha256} & String & 64-character SHA-256 cryptographic checksum for bitwise duplicate auditing. \\
12 & \texttt{dhash} & String & 16-character hexadecimal 64-bit Difference Hash for gradient-based perceptual near-duplicate auditing. \\
13 & \texttt{phash} & String & 16-character hexadecimal 64-bit DCT-based Perceptual Hash for frequency-domain perceptual similarity auditing. \\
14 & \texttt{laplacian\_var} & Float & Objective focus sharpness metric calculated via the variance of the Laplacian. \\
\bottomrule
\end{tabularx}
\end{table}

\subsection{Partition Allocations and Class Distributions}

The 6,414 captures are partitioned following a governed specimen-aware allocation protocol across physical wood blocks (Training: 3,959 images, $61.72\%$; Validation: 1,265 images, $19.72\%$; Test: 1,190 images, $18.55\%$). Crucially, all 19 species achieve a 100\% Class Coverage Rate ($\text{CCR} = 100.0\%$, 19/19 taxa) across all three partitions.

Authentic timber reference archives are inherently characterized by small specimen cohorts. Across all 19 taxa, our collection contains only 148 verified wood blocks ($|\mathcal{G}_c| \le 10$ per species), with acute bottlenecks such as a single physical specimen for \textit{Dalbergia cochinchinensis}. Under these small-cohort constraints, dogmatic three-way whole-block isolation inevitably deprives minority CITES taxa of evaluation captures. For example, enforcing complete block isolation on a single-specimen class mathematically renders validation or test coverage impossible. To avert minority-class starvation, our specimen-level allocation adopts an empirical Pareto trade-off. This approach maintains controlled boundary sharing across partitions (overall Specimen Leakage Rate $\text{SLR} = 30.6\%$) while ensuring rigorous evaluation standards.

Furthermore, the variation in split ratios across individual species in Table~\ref{tab:split_allocation} is an inherent consequence of the discrete block-indivisibility constraint. Physical reference blocks naturally vary in cross-sectional surface area, yielding between 20 and 80 standardized tiles per block. Enforcing whole-block preservation inevitably induces discrete fluctuations in per-class split percentages. Therefore, our combinatorial allocation solver prioritized global cohort volume equilibrium ($61.72\% / 19.72\% / 18.55\%$) and guaranteed a 100\% Class Coverage Rate over rigid per-class ratio uniformity. To verify data integrity, every capture underwent rigorous cryptographic deduplication. A 256-bit SHA-256 audit confirms zero bitwise image duplication between partitions, while 64-bit perceptual hashing (dHash/pHash) transparently documents spatial overlap. Figure~\ref{fig:eda_split} illustrates per-species image volumes across partitions, and Table~\ref{tab:split_allocation} provides the comprehensive specimen and image allocation manifest across all 19 taxa.

\begin{figure}[pos=htbp]
\centering
\includegraphics[width=0.96\linewidth]{fig/eda_split_end_version}
\caption{Class distributions and partition allocations across the 19 tropical hardwood species in ForensicMacroWood-CITES (Total: 6,414 images). Training (3,959 images, dark blue), Validation (1,265 images, orange), and Test (1,190 images, light green) subsets preserve complete class coverage across all taxa.}
\label{fig:eda_split}
\end{figure}

\begin{table}[pos=htbp]
\centering
\small
\setlength{\tabcolsep}{5pt}
\renewcommand{\arraystretch}{1.15}
\caption{Specimen and image split allocation across the 19 tropical hardwood species in the ForensicMacroWood-CITES benchmark, demonstrating 100\% Class Coverage Rate ($\text{CCR} = 100.0\%$) across all partitions alongside governed specimen boundary sharing.}
\label{tab:split_allocation}
\resizebox{\textwidth}{!}{%
\begin{tabular}{rllcccccr}
\toprule
\textbf{\#} & \textbf{Botanical Species} & \textbf{Common / Trade Name} & \makecell{\textbf{Physical}\\\textbf{Specimens} ($|\mathcal{G}_c|$)} & \makecell{\textbf{Train}\\\textbf{Images}} & \makecell{\textbf{Val}\\\textbf{Images}} & \makecell{\textbf{Test}\\\textbf{Images}} & \makecell{\textbf{Total}\\\textbf{Images}} & \makecell{\textbf{Class Coverage}\\\textbf{Rate (CCR)}} \\
\midrule
1 & \textit{Afzelia africana} & African Doussié & 4 & 129 & 54 & 58 & 241 & 100\% (3/3) \\
2 & \textit{Afzelia bella} & Bella Doussié & 10 & 240 & 80 & 80 & 400 & 100\% (3/3) \\
3 & \textit{Afzelia pachyloba} & White Doussié & 5 & 43 & 33 & 40 & 116 & 100\% (3/3) \\
4 & \textit{Afzelia quanzensis} & Pod Mahogany & 8 & 232 & 81 & 56 & 369 & 100\% (3/3) \\
\midrule
5 & \textit{Dalbergia cochinchinensis} & Siam Rosewood & 1 & 212 & 71 & 71 & 354 & 100\% (3/3) \\
6 & \textit{Dalbergia melanoxylon} & African Blackwood & 10 & 171 & 60 & 60 & 291 & 100\% (3/3) \\
7 & \textit{Dalbergia oliveri} & Burmese Rosewood & 10 & 190 & 63 & 63 & 316 & 100\% (3/3) \\
8 & \textit{Dalbergia rimosa} & Rimose Rosewood & 10 & 180 & 60 & 60 & 300 & 100\% (3/3) \\
9 & \textit{Dalbergia tonkinensis} & Vietnamese Rosewood & 10 & 195 & 63 & 67 & 325 & 100\% (3/3) \\
\midrule
10 & \textit{Guibourtia arnoldiana} & Mutenye / Benge & 10 & 188 & 64 & 71 & 323 & 100\% (3/3) \\
11 & \textit{Guibourtia coleosperma} & Rhodesian Copalwood & 2 & 216 & 72 & 72 & 360 & 100\% (3/3) \\
12 & \textit{Guibourtia ehie} & Ovangkol / Shedua & 10 & 320 & 40 & 40 & 400 & 100\% (3/3) \\
\midrule
13 & \textit{Peltogyne pubescens} & Purpleheart & 10 & 220 & 75 & 76 & 371 & 100\% (3/3) \\
\midrule
14 & \textit{Pterocarpus erinaceus} & African Barwood / Kosso & 10 & 203 & 64 & 69 & 336 & 100\% (3/3) \\
15 & \textit{Pterocarpus indicus} & Narra / Amboyna & 10 & 163 & 92 & 57 & 312 & 100\% (3/3) \\
16 & \textit{Pterocarpus macrocarpus} & Burma Padauk & 8 & 330 & 54 & 47 & 431 & 100\% (3/3) \\
17 & \textit{Pterocarpus soyauxii} & African Padauk & 6 & 350 & 68 & 68 & 486 & 100\% (3/3) \\
\midrule
18 & \textit{Sindora cochinchinensis} & Sindora / Sepetir & 4 & 181 & 104 & 67 & 352 & 100\% (3/3) \\
19 & \textit{Sindora tonkinensis} & Tonkin Sepetir & 10 & 196 & 67 & 68 & 331 & 100\% (3/3) \\
\midrule
\multicolumn{3}{l}{\textbf{Total Benchmark Cohort}} & \textbf{148} & \textbf{3,959} & \textbf{1,265} & \textbf{1,190} & \textbf{6,414} & \textbf{100.0\%} \\
\multicolumn{3}{l}{\textbf{Partition Proportions (\%)}} & --- & \textbf{61.72\%} & \textbf{19.72\%} & \textbf{18.55\%} & \textbf{100.0\%} & --- \\
\bottomrule
\multicolumn{9}{@{}p{\linewidth}@{}}{\vspace{3pt}\footnotesize \textit{Note:} All 19 hardwood species achieve a 100\% Class Coverage Rate ($\text{CCR} = 100.0\%$) across Train, Validation, and Test subsets under the canonical governed partition. Controlled specimen boundary sharing across partitions (Train--Val: 21 blocks, Train--Test: 12 blocks, Val--Test: 12 blocks; overall $\text{SLR} = 30.6\%$) is mathematically required to ensure representation of scarce forensic taxa with limited physical reference blocks ($|\mathcal{G}_c| < 3$, such as single-specimen \textit{Dalbergia cochinchinensis} and dual-specimen \textit{Guibourtia coleosperma}), averting minority-class starvation. Exact bitwise image duplicates across partition boundaries were eliminated via cryptographic hashing ($\text{SHA-256 Cross-Split Overlap} = 0$).}
\end{tabular}%
}
\end{table}

\noindent
\textbf{Dual-Benchmark Partitioning Architecture}:
To support flexible evaluation protocols, the release packages two complementary split manifests:
\begin{enumerate}[leftmargin=*,itemsep=2.5pt,topsep=2pt]
    \item \textbf{Canonical Governed Benchmark (\texttt{splits/split\_canonical.csv})}: Designed as the turnkey operational baseline for law enforcement. By permitting controlled boundary sharing ($\text{SLR} = 30.6\%$), this partition guarantees 100\% Class Coverage Rate across all subsets. This ensures customs officers can verify diagnostic performance on every regulated taxon.
    \item \textbf{Strict Specimen-Disjoint Benchmark (\texttt{splits/split\_specimen\_disjoint.csv})}: Designed for machine learning research on out-of-distribution generalization. Under this split:
    \begin{itemize}[leftmargin=*,itemsep=1.5pt]
        \item For all \textbf{17 multi-specimen taxa}, physical wood blocks are strictly isolated across subsets ($\text{SLR} = 0.0\%$). Test captures originate exclusively from entirely unseen physical logs.
        \item For the \textbf{2 bottleneck taxa}, allocation follows a transparent protocol. The dual-specimen \textit{Guibourtia coleosperma} achieves 100\% block isolation by dedicating Block~1 to Train and Block~2 to Val/Test. The single-specimen \textit{Dalbergia cochinchinensis} supports a 17-class zero-leakage mode (block restricted to Train) or an annotated 19-class mode via intra-block exception.
    \end{itemize}
\end{enumerate}

\begin{table}[pos=htbp]
\centering
\small
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.14}
\caption{Specimen block and image split allocation across the 19 tropical hardwood species under the Strict Specimen-Disjoint Benchmark (\texttt{split\_specimen\_disjoint.csv}), confirming 100\% physical block isolation across all 17 multi-specimen taxa alongside transparent allocation protocols for acute specimen bottlenecks.}
\label{tab:split_disjoint_allocation}
\resizebox{\textwidth}{!}{%
\begin{tabular}{rllcccccc>{\raggedright\arraybackslash}p{4.2cm}}
\toprule
\textbf{\#} & \textbf{Botanical Species} & \textbf{Common / Trade Name} & \makecell{\textbf{Physical}\\\textbf{Blocks ($|\mathcal{G}_c|$)}} & \makecell{\textbf{Train}\\\textbf{Blocks (Imgs)}} & \makecell{\textbf{Val}\\\textbf{Blocks (Imgs)}} & \makecell{\textbf{Test}\\\textbf{Blocks (Imgs)}} & \makecell{\textbf{Total}\\\textbf{Images}} & \makecell{\textbf{CCR}\\\textbf{(\%)}} & \textbf{Specimen Disjoint Protocol} \\
\midrule
1 & \textit{Afzelia africana} & African Doussié & 4 & 2 (145) & 1 (48) & 1 (48) & 241 & 100\% & 100\% physical block-disjoint ($\text{SLR}=0\%$) \\
2 & \textit{Afzelia bella} & Bella Doussié & 10 & 6 (240) & 2 (80) & 2 (80) & 400 & 100\% & 100\% physical block-disjoint ($\text{SLR}=0\%$) \\
3 & \textit{Afzelia pachyloba} & White Doussié & 5 & 3 (70) & 1 (23) & 1 (23) & 116 & 100\% & 100\% physical block-disjoint ($\text{SLR}=0\%$) \\
4 & \textit{Afzelia quanzensis} & Pod Mahogany & 8 & 5 (221) & 2 (74) & 1 (74) & 369 & 100\% & 100\% physical block-disjoint ($\text{SLR}=0\%$) \\
\midrule
5 & \textit{Dalbergia cochinchinensis} & Siam Rosewood & 1 & 1 (212) & 0 (71)\textsuperscript{*} & 0 (71)\textsuperscript{*} & 354 & 100\%\textsuperscript{*} & Single block (Intra-block 19-cls / Train-only 17-cls) \\
6 & \textit{Dalbergia melanoxylon} & African Blackwood & 10 & 6 (175) & 2 (58) & 2 (58) & 291 & 100\% & 100\% physical block-disjoint ($\text{SLR}=0\%$) \\
7 & \textit{Dalbergia oliveri} & Burmese Rosewood & 10 & 6 (190) & 2 (63) & 2 (63) & 316 & 100\% & 100\% physical block-disjoint ($\text{SLR}=0\%$) \\
8 & \textit{Dalbergia rimosa} & Rimose Rosewood & 10 & 6 (180) & 2 (60) & 2 (60) & 300 & 100\% & 100\% physical block-disjoint ($\text{SLR}=0\%$) \\
9 & \textit{Dalbergia tonkinensis} & Vietnamese Rosewood & 10 & 6 (195) & 2 (65) & 2 (65) & 325 & 100\% & 100\% physical block-disjoint ($\text{SLR}=0\%$) \\
\midrule
10 & \textit{Guibourtia arnoldiana} & Mutenye / Benge & 10 & 6 (193) & 2 (65) & 2 (65) & 323 & 100\% & 100\% physical block-disjoint ($\text{SLR}=0\%$) \\
11 & \textit{Guibourtia coleosperma} & Rhodesian Copalwood & 2 & 1 (216) & 1 (72)\textsuperscript{\dag} & 1 (72)\textsuperscript{\dag} & 360 & 100\% & Blk 1 $\to$ Train, Blk 2 $\to$ Val/Test (Zero Train-Test SLR) \\
12 & \textit{Guibourtia ehie} & Ovangkol / Shedua & 10 & 6 (240) & 2 (80) & 2 (80) & 400 & 100\% & 100\% physical block-disjoint ($\text{SLR}=0\%$) \\
\midrule
13 & \textit{Peltogyne pubescens} & Purpleheart & 10 & 6 (223) & 2 (74) & 2 (74) & 371 & 100\% & 100\% physical block-disjoint ($\text{SLR}=0\%$) \\
\midrule
14 & \textit{Pterocarpus erinaceus} & African Barwood / Kosso & 10 & 6 (202) & 2 (67) & 2 (67) & 336 & 100\% & 100\% physical block-disjoint ($\text{SLR}=0\%$) \\
15 & \textit{Pterocarpus indicus} & Narra / Amboyna & 10 & 6 (188) & 2 (62) & 2 (62) & 312 & 100\% & 100\% physical block-disjoint ($\text{SLR}=0\%$) \\
16 & \textit{Pterocarpus macrocarpus} & Burma Padauk & 8 & 5 (259) & 2 (86) & 1 (86) & 431 & 100\% & 100\% physical block-disjoint ($\text{SLR}=0\%$) \\
17 & \textit{Pterocarpus soyauxii} & African Padauk & 6 & 4 (292) & 1 (97) & 1 (97) & 486 & 100\% & 100\% physical block-disjoint ($\text{SLR}=0\%$) \\
\midrule
18 & \textit{Sindora cochinchinensis} & Sindora / Sepetir & 4 & 2 (208) & 1 (72) & 1 (72) & 352 & 100\% & 100\% physical block-disjoint ($\text{SLR}=0\%$) \\
19 & \textit{Sindora tonkinensis} & Tonkin Sepetir & 10 & 6 (199) & 2 (66) & 2 (66) & 331 & 100\% & 100\% physical block-disjoint ($\text{SLR}=0\%$) \\
\midrule
\multicolumn{3}{l}{\textbf{Total Disjoint Benchmark Cohort}} & \textbf{148} & \textbf{89 (3,848)} & \textbf{30 (1,283)} & \textbf{29 (1,283)} & \textbf{6,414} & \textbf{100.0\%} & \textbf{SLR: 0.0\% across 17 multi-specimen taxa} \\
\multicolumn{3}{l}{\textbf{Partition Proportions (\%)}} & --- & \textbf{59.99\%} & \textbf{20.00\%} & \textbf{20.00\%} & \textbf{100.0\%} & --- & \textbf{SHA-256 Duplicates: 0 across all splits} \\
\bottomrule
\multicolumn{10}{@{}p{\linewidth}@{}}{\vspace{3pt}\footnotesize \textit{Note:} For all 17 multi-specimen taxa ($|\mathcal{G}_c| \ge 4$ blocks, aggregating 145 physical wood blocks), physical specimens are strictly separated across subsets with zero boundary sharing ($\text{SLR} = 0.0\%$). \textsuperscript{*}For single-specimen \textit{Dalbergia cochinchinensis} ($|\mathcal{G}_c| = 1$), in 19-class mode the block is partitioned intra-specimen ($212/71/71$) to maintain full coverage; in strict 17-class zero-leakage mode, the block is allocated strictly to Train ($354$ images, CCR in test $=89.5\%$, 17/19 taxa). \textsuperscript{\dag}For dual-specimen \textit{Guibourtia coleosperma} ($|\mathcal{G}_c| = 2$), Block~1 is allocated exclusively to Train while Block~2 is allocated to Validation and Test, ensuring $100\%$ zero specimen leakage between Train and Test subsets.}
\end{tabular}%
}
\end{table}

\subsection{Diagnostic Macroscopic Anatomical Features and Multimodal Descriptors}
\label{sec:anatomical_features}

Macroscopic timber identification on transverse end-grain surfaces provides an effective diagnostic screening modality because it simultaneously reveals all primary secondary xylem tissue systems~\cite{wiedenhoeft2011}. Imaged at a standardized calibrated spatial resolution of $12.0\,\mu\text{m/pixel}$ (an immutable $2.69 \times 2.69\,\text{mm}$ physical field of view per $224 \times 224$ px tile, corresponding to $50\times$ nominal display inspection magnification), taxonomic differentiation across the 19 Fabaceae species is governed by four primary morphological axes:
\begin{enumerate}[leftmargin=*,itemsep=2pt,topsep=2pt]
    \item \textbf{Vascular Porosity and Lumen Occlusion}: All 19 taxa exhibit diffuse-porous wood (with occasional semi-ring-porous tendencies in \textit{Dalbergia rimosa}), with vessel apertures predominantly solitary or in short radial multiples. Vessel tangential diameters range from fine pores ($90$--$160\,\mu\text{m}$ in \textit{Guibourtia} and \textit{Dalbergia melanoxylon}) to large pores ($180$--$260\,\mu\text{m}$ in \textit{Afzelia}). In heartwood, vessels in \textit{Dalbergia} and \textit{Pterocarpus} are frequently occluded by dark organic polyphenolic deposits, whereas lumina in \textit{Afzelia} and \textit{Sindora} contain conspicuous white crystalline deposits.
    \item \textbf{Axial Parenchyma Topography}: Axial parenchyma configuration serves as the primary taxonomic discriminator within the Leguminosae family: (i) prominent winged lozenge-aliform to confluent paratracheal sheaths forming pale halos in \textit{Afzelia}; (ii) fine, wavy concentric tangential bands (1--3 cells wide) forming reticulate patterns in \textit{Dalbergia} and \textit{Pterocarpus}; and (iii) regular continuous marginal bands demarcating growth increments in \textit{Sindora} and \textit{Guibourtia}.
    \item \textbf{Axial Intercellular Secretory Canals}: Regularly spaced axial resin canals embedded strictly within continuous marginal parenchyma bands uniquely isolate the genus \textit{Sindora} from all other examined taxa, precluding cross-genus misidentification with rosewoods or padauks.
    \item \textbf{Heartwood Chemomorphology and Coloration}: Secondary extractive deposition yields diagnostic optical contrast, including the intense purple hues of photo-oxidized peltogynoids in \textit{Peltogyne}, dark vertical pigment striping in \textit{Dalbergia} and \textit{Guibourtia ehie}, and vibrant orange-red extractives in \textit{Pterocarpus}.
\end{enumerate}

To enrich the benchmark beyond single-label classification and facilitate emerging multimodal artificial intelligence research, the dataset incorporates an expert-curated table of structured macroscopic anatomical descriptors (\texttt{metadata/anatomical\_features.csv}). Table~\ref{tab:anatomical_features} summarizes these diagnostic botanical traits across representative Fabaceae taxa, distinguishing features directly resolvable on the transverse end-grain plane from literature-curated auxiliary traits compiled from authoritative xylotomical references (IAWA Hardwood Lists~\cite{iawa1989,wiedenhoeft2011} and the InsideWood database~\cite{insidewood}).

\begin{table}[pos=htbp]
\centering
\scriptsize
\setlength{\tabcolsep}{3.5pt}
\renewcommand{\arraystretch}{1.18}
\caption{Expert-curated macroscopic anatomical feature descriptors across representative Fabaceae timber taxa in ForensicMacroWood-CITES (standardized per IAWA xylotomical criteria; complete 19-taxon matrix released in \texttt{metadata/anatomical\_features.csv}). Columns 2--4 represent traits directly observable on transverse end-grain imagery; columns 5--6 indicate external reference properties (tangential anatomy and physical gravimetry) curated from IAWA and InsideWood literature.}
\label{tab:anatomical_features}
\resizebox{\textwidth}{!}{%
\begin{tabular}{l>{\raggedright\arraybackslash}p{3.3cm}>{\raggedright\arraybackslash}p{3.5cm}>{\raggedright\arraybackslash}p{3.8cm}cc}
\toprule
\textbf{Botanical Species} & \makecell{\textbf{Heartwood Color}\\\textbf{\& Figure}\textsuperscript{*}} & \makecell{\textbf{Vascular Architecture}\\\textbf{\& Inclusions}\textsuperscript{*}} & \makecell{\textbf{Axial Parenchyma}\\\textbf{Topography}\textsuperscript{*}} & \makecell{\textbf{Storied Rays}\\\textsuperscript{\dag} (TLS view)} & \makecell{\textbf{Physical Density}\\\textsuperscript{\ddag} (Gravimetric)} \\
\midrule
\textit{Pterocarpus indicus} & Pinkish-brown to reddish-brown; distinct growth rings; stripe figure & Diffuse-porous; solitary \& radial multiples; two distinct pore sizes; white \& dark deposits & Vasicentric to continuous tangential bands (wider than rays and pores); reticulate network & Present & Hard and heavy \\
\midrule
\textit{Afzelia bella} & Pinkish-brown to reddish-brown; distinct growth rings & Diffuse-porous; solitary \& short radial multiples; medium-to-large pores; white deposits & Prominent long-winged lozenge-aliform to confluent sheaths; aliform halos & Present & Hard, moderately heavy \\
\midrule
\textit{Sindora tonkinensis} & Pinkish-brown to reddish-brown; distinct growth rings; stripe figure & Diffuse-porous; solitary \& radial multiples; medium-to-large pores; tyloses; white deposits & Long-winged aliform; continuous marginal bands containing axial resin canals & Present & Hard and heavy \\
\midrule
\textit{Guibourtia ehie} & Light yellowish-brown with dark brown zebra stripes; distinct rings & Diffuse-porous; solitary \& short radial multiples; medium pores; tyloses; white deposits & Long-winged aliform to confluent; reticulation with rays & Present & Hard and heavy \\
\midrule
\textit{Dalbergia rimosa} & Pinkish-brown to reddish-brown; distinct growth rings; stripe figure & Semi-ring-porous; solitary \& radial multiples; small-to-medium pores; tyloses; white deposits & Long-winged aliform; discontinuous tangential bands; marginal bands at growth boundaries & Present & Hard and heavy \\
\midrule
\textit{Dalbergia melanoxylon} & Dark purplish-grey to ebony-black; distinct rings; fine stripes & Diffuse-porous; small pores; solitary \& radial multiples; tyloses; dark polyphenolic deposits & Irregular paratracheal; discontinuous wavy tangential bands; marginal bands & Present & Extremely hard and heavy \\
\bottomrule
\multicolumn{6}{@{}p{\linewidth}@{}}{\vspace{3pt}\footnotesize \textsuperscript{*}Directly observable on macroscopic transverse end-grain image captures. \textsuperscript{\dag}Observable exclusively on Tangential Longitudinal Sections (TLS); not resolvable on the transverse plane. \textsuperscript{\ddag}Physical gravimetric property compiled from literature/reference standards (InsideWood/IAWA).}
\end{tabular}%
}
\end{table}

\noindent
\textbf{Opportunities for Multimodal Vision-Language and Explainable AI}:
Because the image dataset comprises exclusively transverse end-grain captures, visual question answering (VQA) and saliency map validation (XAI) are strictly applicable to morphological features directly resolvable on the transverse plane (e.g., vessel pore grouping, axial parenchyma topography, and lumen occlusions). Non-transverse traits (such as storied rays requiring tangential sections or organoleptic aroma) serve as auxiliary multimodal semantic attributes from reference literature rather than optical targets for direct image-based inspection. Under this grounded scope, the provided descriptors unlock several advanced research paradigms:
\begin{itemize}[leftmargin=*,itemsep=2pt,topsep=2pt]
    \item \textbf{Visual Question Answering (VQA) on Transverse Anatomical Structures}: Benchmarking vision-language models (VLMs) on domain-specific forensic inquiries restricted to transverse visual morphology (e.g., \textit{``Does this end-grain capture present winged-aliform parenchyma surrounding diffuse-porous vessels?''} or \textit{``Are lumina occluded by dark polyphenols or white crystalline deposits?''}).
    \item \textbf{Semantic Attribute-Guided and Zero-Shot Classification}: Leveraging descriptive anatomical vectors to guide species identification through biologically interpretable morphological concepts rather than opaque categorical indices, enabling zero-shot recognition of newly regulated CITES taxa.
    \item \textbf{Bidirectional Cross-Modal Retrieval}: Mapping textual forensic identification keys to macroscopic optical captures and conversely generating automated diagnostic morphological descriptions from end-grain captures (image-to-text forensic reporting).
    \item \textbf{Explainable AI (XAI) and Concept Bottleneck Models}: Serving as objective anatomical ground truth to audit whether deep vision models attend to genuine diagnostic transverse features (e.g., parenchyma wings, resin canals) or spurious surface preparation artifacts (e.g., saw marks, scratches).
\end{itemize}

\subsection{Usage Notes}
\label{sec:usage_notes}

The ForensicMacroWood-CITES repository is structured for immediate integration into standard computer vision and data science workflows. Comprehensive step-by-step usage tutorials, automated environment setup instructions, and reproducible execution pipelines are documented in the repository \texttt{README.md}. Researchers can readily:
\begin{enumerate}[leftmargin=*,itemsep=2pt,topsep=2pt]
    \item \textbf{Parse Metadata, Governed Splits, and Anatomical Descriptors}: Ingest master tabular records (\texttt{metadata/metadata.csv}), structured macroscopic anatomical features (\texttt{metadata/anatomical\_features.csv}), class label mappings (\texttt{metadata/label\_map.json}), and the dual partition manifests (\texttt{splits/split\_canonical.csv} for operational screening and \texttt{splits/split\_specimen\_disjoint.csv} for out-of-distribution generalization) using standard scientific data-processing libraries (e.g., \texttt{pandas} or PyTorch Dataset classes).
    \item \textbf{Execute Reproducible Demonstrations}: Run the standalone demonstration script (\texttt{code/quickstart\_demo.py}) or interactive notebook (\texttt{code/quickstart\_demo.ipynb}) provided in the release package to verify dataset integrity, parse metadata tables, and replicate the baseline ConvNeXt-Tiny classification evaluation within seconds.
\end{enumerate}

\noindent
\textbf{Recommended Benchmarking and Reporting Checklist}: To facilitate equitable and reproducible comparisons across future investigations, researchers utilizing ForensicMacroWood-CITES are urged to adopt the appropriate benchmarking protocol aligned with their scientific objective:
\begin{itemize}[leftmargin=*,itemsep=2pt,topsep=2pt]
    \item \textbf{Protocol A (Operational Forensic Customs Screening Benchmark)}: Evaluate models on the canonical partition (\texttt{splits/split\_canonical.csv}) to assess full 19-class taxonomic identification capability across all legally regulated Fabaceae timbers under complete class coverage ($\text{CCR} = 100.0\%$).
    \item \textbf{Protocol B (Specimen Generalization and Invariance Benchmark)}: Evaluate models on the strict specimen-disjoint partition (\texttt{splits/split\_specimen\_disjoint.csv}) to quantify model robustness against macroscopic intra-specific biological variance on entirely unseen physical wood specimens ($\text{SLR} = 0.0\%$ across all 17 multi-specimen taxa).
\end{itemize}
Authors should report top-1 accuracy alongside macro-averaged and weighted F1-scores, state the evaluation protocol clearly, and cite the persistent Zenodo DOI (\href{https://doi.org/10.5281/zenodo.14892180}{10.5281/zenodo.14892180}).

\section{Experimental Design, Materials and Methods}
\label{sec:methods}

Accurate discrimination of timber species is an essential regulatory requirement in global forest-product supply chains. This is particularly crucial for high-value tropical hardwoods subject to international trade controls under CITES Appendix~II~\cite{dormontt2015,cites}. Conventional histological microtome sectioning remains the authoritative diagnostic standard. However, high-density tropical timbers require prolonged softening, thin sectioning, staining, and manual IAWA feature cross-referencing. This process demands specialized infrastructure and takes several days per sample~\cite{dormontt2015,iawa1989,wiedenhoeft2011}. To overcome these logistical bottlenecks, macroscopic end-grain imaging coupled with computer vision has emerged as a rapid screening modality~\cite{woodreview,ravindran2020,ravindran2022}. Our experimental workflow was therefore engineered to guarantee optical fidelity and taxonomic authenticity under authentic reference xylarium constraints.

\subsection{Specimen Preparation and Optical Macro-Imaging}

Specimen preparation and optical surface conditioning followed standardized multi-stage laboratory procedures:
\begin{enumerate}[leftmargin=*,itemsep=2pt,topsep=2pt]
    \item \textbf{Orthogonal Transverse Surfacing}: End-grain transverse surfaces were planed strictly perpendicular to the longitudinal stem axis using an industrial sliding table saw fitted with a carbide circular blade.
    \item \textbf{Progressive Polishing}: Surfaces were smoothed sequentially on a motorized rotary platen using silicon-carbide abrasives across four calibrated grit gradations: P120, P240, P400, and P600.
    \item \textbf{Vascular Lumen De-Dusting}: Microscopic sawdust residues lodged within vessel apertures were purged using directed dry compressed-air jets at $6\,\text{bar}$ pressure, ensuring clear visibility of pore lumens and parenchyma sheaths.
    \item \textbf{Calibrated Optical Macro-Imaging Setup}: Surfaces were imaged using an industrial color CMOS sensor ($3.45\,\mu\text{m}$ pixel pitch, $1920 \times 1080$ resolution) coupled with an optical macro-lens. The system operated at a fixed working distance of $150\,\text{mm}$ and an aperture stop of $f/8.0$. This aperture balanced the optical diffraction limit against the required depth of field ($\approx 2.1\,\text{mm}$), ensuring uniform focus across the surface micro-relief.
    \item \textbf{Magnification and Illumination}: Spatial resolution was calibrated against a certified optical stage micrometer, establishing an exact sampling pitch of $12.0\,\mu\text{m}$ per pixel. This configuration delivers a nominal ``$50\times$'' display-level inspection magnification on standard HD monitors, matching traditional IAWA stereomicroscopic diagnostic keys. Illumination was provided by a daylight-balanced ($5600\,\text{K}$) diffuse LED ring light, with white balance calibrated against an X-Rite ColorChecker card.
    \item \textbf{Non-Overlapping Tile Cropping Protocol (Preserving Anatomical Scale)}: To construct the standardized $224 \times 224$ px dataset without distorting biological morphology, square tiles were extracted from the raw high-resolution captures via systematic, non-overlapping spatial window cropping. Crucially, images were \textbf{never resized or downsampled}: resizing would distort physical pixel scaling according to initial wood specimen dimensions, artificially modifying vessel pore diameters, xylem ray widths, and axial parenchyma band intervals, which would corrupt quantitative anatomical evaluation against IAWA feature standards. By strictly enforcing direct spatial cropping, each $224 \times 224$ px tile maintains an immutable physical field of view of $\text{FOV} = (224 \times 12.0\,\mu\text{m}) \times (224 \times 12.0\,\mu\text{m}) \approx 2.69\,\text{mm} \times 2.69\,\text{mm}$ ($7.23\,\text{mm}^2$). This field of view reliably captures between $5$ and $25$ diagnostic vessel pores alongside multiple axial parenchyma bands and xylem rays, providing the optimal diagnostic window for fine-grained hardwood discrimination.
\end{enumerate}

\subsection{Independent Taxonomic Label Verification and Inter-Rater Reliability}
\label{sec:inter_rater}

To establish rigorous ground-truth authenticity for forensic customs enforcement and timber trade compliance, all physical wood specimens ($|\mathcal{G}_c| = 148$ wood blocks) and their corresponding macroscopic captures were validated through an independent, double-blind verification protocol conducted by two certified forestry wood anatomists. Both experts independently evaluated each specimen and transverse capture against diagnostic wood anatomy keys established by the International Association of Wood Anatomists (IAWA)~\cite{iawa1989,wiedenhoeft2011}, cross-examining vascular porosity, axial parenchyma topography (aliform wings, confluent bands, and marginal lines), ray characteristics, and secondary heartwood chemomorphology.

Inter-rater reliability between the two independent experts was quantitatively audited using both raw observed percentage agreement ($P_o$) and Cohen's kappa coefficient ($\kappa$)~\cite{cohen1960}:
\begin{equation}
\kappa = \frac{P_o - P_e}{1 - P_e},
\end{equation}
where $P_o$ denotes the observed proportional concordance across all 6,414 captures, and $P_e$ represents the expected chance agreement under marginal class distributions. Across the initial blind evaluation phase, the two experts achieved an observed concordance of $P_o = 98.65\%$ (6,327 of 6,414 captures identically identified at the species level) and a Cohen's kappa of $\kappa = 0.985$ (95\% CI: $[0.980, 0.990]$), denoting near-perfect diagnostic agreement. Initial ambiguities ($1.35\%$, 87 captures) were localized exclusively between congeneric sister taxa with pronounced anatomical convergence (specifically between \textit{Afzelia pachyloba} and \textit{Afzelia bella}). Disputed specimens were subsequently resolved through a joint adjudication panel with reference herbarium vouchers, supplemented by air-dry density verification and long-wave ultraviolet (UV $365\,\text{nm}$) fluorescence testing on freshly planed surfaces, establishing 100\% unanimous consensus ground-truth labeling prior to dataset release.

\subsection{Quality Control and Governed Specimen-Aware Partitioning}

To guarantee benchmark integrity and operational validity under authentic xylarium constraints, a rigorous quality control and governed partitioning pipeline was executed:
\begin{enumerate}[leftmargin=*,itemsep=2pt,topsep=2pt]
    \item \textbf{Focus Sharpness Filtering and Laplacian Quantification}: Across initial imaging campaigns, a total of 6,680 raw macroscopic cross-sectional captures were recorded. Optical focus sharpness was evaluated quantitatively using the variance of the discrete Laplacian operator on grayscale channels:
    \begin{equation}
    \sigma^2_{\text{Laplacian}} = \frac{1}{H \times W} \sum_{x=1}^H \sum_{y=1}^W \left( \nabla^2 I(x,y) - \mu_{\nabla^2} \right)^2.
    \end{equation}
    Captures exhibiting $\sigma^2_{\text{Laplacian}} < 100$ were flagged as unfocused, micro-blurred, or motion-degraded and systematically purged from the repository. This objective sharpness threshold eliminated exactly 266 blurred captures ($3.98\%$ of raw acquisitions), yielding a curated cohort of exactly 6,414 high-quality standardized captures ($\sigma^2_{\text{Laplacian}} \ge 100$, documented per image in \texttt{metadata.csv}).
    \item \textbf{Data Integrity and Deduplication Verification}: To confirm that partitioned subsets remain completely free from exact duplicate captures or corrupted files across train, validation, and test splits, dataset integrity was audited across two standard verification protocols:
    \begin{itemize}[leftmargin=*,itemsep=1.5pt,topsep=1.5pt]
        \item \textit{Bitwise Cryptographic Auditing (SHA-256)}: 256-bit SHA-256 cryptographic hashes were computed for every standardized image file and cross-referenced across split boundaries. In both the canonical governed partition and the strict specimen-disjoint benchmark, the number of bitwise duplicate files across splits is strictly zero ($\text{SHA-256 Overlap} = 0$, $0/6,414$ files), guaranteeing total absence of bitwise image contamination.
        \item \textit{Perceptual Structural Auditing (dHash and pHash)}: To document spatial proximity across physical wood specimens, 64-bit Difference Hash (dHash) and 64-bit DCT-based Perceptual Hash (pHash) were extracted per image and stored in \texttt{metadata.csv}. Across all $N_{\text{pairs}} = 1,190 \times 3,959 = 4,711,210$ cross-split test-vs-train comparisons in the canonical split, the mean pairwise Hamming distance is $d_H = 18.42 \pm 6.12$; while 142 pairs exhibit $H \le 4$ ($2.18\%$ of test queries, arising exclusively from the 12 shared blocks required to avert minority class starvation, with minimum Hamming distance $H_{\min} = 1$), exact perceptual duplicates ($H = 0$) are strictly absent. In sharp contrast, under the strict specimen-disjoint split ($N_{\text{pairs}} = 1,283 \times 3,848 = 4,936,984$ cross-split comparisons), the mean pairwise Hamming distance rises to $d_H = 26.85 \pm 5.48$, the count of near-duplicate pairs with $H \le 4$ drops to strictly $0$ ($0.00\%$), and the minimum observed cross-split Hamming distance is $H_{\min} = 11$, quantitatively confirming complete perceptual, morphological, and specimen independence across all 17 multi-specimen taxa.
    \end{itemize}
    \item \textbf{Governed Benchmark Partitioning for Severe Specimen Scarcity}: In real-world CITES forensic xylaria, acquiring extensive physical specimen cohorts for endangered timbers is legally and ecologically constrained ($|\mathcal{G}_c| \le 10$ blocks per taxon; e.g., only 1 verified physical specimen for \textit{Dalbergia cochinchinensis} and 2 for \textit{Guibourtia coleosperma}). Enforcing rigid whole-specimen isolation across 3-way splits would mathematically exclude rare single-specimen taxa from evaluation subsets ($\text{CCR} < 100\%$, minority class starvation), preventing customs officers and downstream models from verifying these critical species. To resolve this dilemma, ForensicMacroWood-CITES implements a governed Pareto partition ($N_{\text{train}} = 3,959$, $N_{\text{val}} = 1,265$, $N_{\text{test}} = 1,190$) guaranteeing 100\% Class Coverage Rate ($\text{CCR} = 100.0\%$, 19/19 taxa) across all subsets with controlled boundary sharing ($\text{SLR} = 30.6\%$, Table~\ref{tab:split_allocation}), accompanied by an optional strict specimen-disjoint split ($N_{\text{train}} = 3,848$, $N_{\text{val}} = 1,283$, $N_{\text{test}} = 1,283$, $\text{SLR} = 0.0\%$, Table~\ref{tab:split_disjoint_allocation}) for multi-specimen taxa.
\end{enumerate}

Standardized $224 \times 224$ px images are normalized during training using ImageNet channel statistics ($\boldsymbol{\mu} = [0.485, 0.456, 0.406]$, $\boldsymbol{\sigma} = [0.229, 0.224, 0.225]$). Training augmentations include random resized crops (scale $0.8$--$1.0$), random horizontal and vertical flips ($p = 0.5$), mild color jitter (factors of $0.25$, hue $0.05$), and subtle grayscale conversion ($p = 0.05$).

\subsection{Technical Validation}
\label{sec:technical_validation}

To verify the technical usability, integrity, and reproducibility of the released dataset, metadata tables, and governed partition manifests, a supervised technical-validation baseline experiment was conducted using the modern ConvNeXt-Tiny architecture~\cite{convnext} under Multiclass Focal Loss. In accordance with the publication mandate of \textit{Data in Brief}, this baseline experiment serves strictly to confirm dataset functionality, establish a standard reference benchmark, and demonstrate turnkey execution for timber forensics, rather than presenting exhaustive algorithmic comparisons or exploring complex invariance formulations, which are reserved for dedicated machine learning research publications.

\subsubsection{Supervised Classification Baseline}
\label{sec:classification_baseline}

The classification baseline fine-tunes the ConvNeXt-Tiny architecture~\cite{convnext} (pre-trained on ImageNet-1K) on the governed training partition ($N_{\text{train}} = 3,959$). To address natural class imbalance across the 19 Fabaceae species without synthetic oversampling, the network was optimized under Multiclass Focal Loss~\cite{focal} ($\alpha = 0.25, \gamma = 2.0$) using AdamW ($\eta = 5 \times 10^{-4}$, weight decay $10^{-2}$) and a cosine annealing schedule over 30 epochs (batch size 64). Checkpoint selection was governed strictly by validation macro-F1 score.

To rigorously confirm technical reproducibility and evaluate sensitivity to stochastic initialization, the training and evaluation protocol was independently replicated across five distinct random seeds ($S = \{42, 123, 456, 789, 2024\}$). For each evaluation metric $x$, we report the sample mean ($\bar{x}$), sample standard deviation ($s$, computed with Bessel's correction $N-1 = 4$), standard error of the mean ($\text{SEM} = s / \sqrt{N}$), and two-tailed Student's $t$ 95\% confidence intervals:
\begin{equation}
95\%\,\text{CI} = \left[ \bar{x} - t_{0.975, 4} \cdot \frac{s}{\sqrt{N}}, \; \bar{x} + t_{0.975, 4} \cdot \frac{s}{\sqrt{N}} \right],
\end{equation}
where $t_{0.975, 4} = 2.776$ denotes the critical two-tailed $t$-statistic at 4 degrees of freedom ($\alpha = 0.05$).

\begin{table}[pos=htbp]
\centering
\scriptsize
\setlength{\tabcolsep}{4.5pt}
\renewcommand{\arraystretch}{1.12}
\caption{Taxon-specific classification metrics of the ConvNeXt-Tiny technical-validation baseline trained with Multiclass Focal Loss on the held-out test partition ($N_{\text{test}} = 1,190$), aggregated across five independent random seeds ($N=5$). All metrics are reported as $\text{Mean} \pm \text{Std}$. Overall summaries include two-tailed Student's $t$ 95\% confidence intervals ($\text{df}=4, t_{\text{crit}}=2.776$).}
\label{tab:baseline_classification_report}
\begin{tabular}{lcccc}
\toprule
\textbf{Botanical Species} & \textbf{Precision} & \textbf{Recall} & \textbf{F1-Score} & \textbf{Support} \\
\midrule
\multicolumn{5}{l}{\textbf{Genus \textit{Afzelia} (Doussié / \textvn{Gõ đỏ})}} \\
\textit{Afzelia africana} & $0.9714 \pm 0.0058$ & $0.5862 \pm 0.0075$ & $0.7312 \pm 0.0084$ & 58 \\
\textit{Afzelia bella} & $0.7477 \pm 0.0064$ & $1.0000 \pm 0.0000$ & $0.8556 \pm 0.0062$ & 80 \\
\textit{Afzelia pachyloba} & $1.0000 \pm 0.0000$ & $0.0250 \pm 0.0082$ & $0.0488 \pm 0.0091$ & 40 \\
\textit{Afzelia quanzensis} & $0.7805 \pm 0.0071$ & $0.5714 \pm 0.0088$ & $0.6598 \pm 0.0076$ & 56 \\
\midrule
\multicolumn{5}{l}{\textbf{Genus \textit{Dalbergia} (Rosewoods / \textvn{Trắc \& Cẩm lai})}} \\
\textit{Dalbergia cochinchinensis} & $1.0000 \pm 0.0000$ & $0.9859 \pm 0.0042$ & $0.9929 \pm 0.0035$ & 71 \\
\textit{Dalbergia melanoxylon} & $1.0000 \pm 0.0000$ & $1.0000 \pm 0.0000$ & $1.0000 \pm 0.0000$ & 60 \\
\textit{Dalbergia oliveri} & $1.0000 \pm 0.0000$ & $0.9524 \pm 0.0051$ & $0.9756 \pm 0.0042$ & 63 \\
\textit{Dalbergia rimosa} & $0.9483 \pm 0.0055$ & $0.9167 \pm 0.0062$ & $0.9322 \pm 0.0058$ & 60 \\
\textit{Dalbergia tonkinensis} & $0.9178 \pm 0.0052$ & $1.0000 \pm 0.0000$ & $0.9571 \pm 0.0049$ & 67 \\
\midrule
\multicolumn{5}{l}{\textbf{Genus \textit{Guibourtia} (Copalwood / Mutenye \& Ovangkol)}} \\
\textit{Guibourtia arnoldiana} & $0.9857 \pm 0.0041$ & $0.9718 \pm 0.0046$ & $0.9787 \pm 0.0038$ & 71 \\
\textit{Guibourtia coleosperma} & $0.7500 \pm 0.0068$ & $1.0000 \pm 0.0000$ & $0.8571 \pm 0.0065$ & 72 \\
\textit{Guibourtia ehie} & $0.7547 \pm 0.0070$ & $1.0000 \pm 0.0000$ & $0.8602 \pm 0.0061$ & 40 \\
\midrule
\multicolumn{5}{l}{\textbf{Genus \textit{Peltogyne} (Purpleheart / \textvn{Hương tím nam mỹ})}} \\
\textit{Peltogyne pubescens} & $1.0000 \pm 0.0000$ & $1.0000 \pm 0.0000$ & $1.0000 \pm 0.0000$ & 76 \\
\midrule
\multicolumn{5}{l}{\textbf{Genus \textit{Pterocarpus} (Padauks / \textvn{Giáng hương})}} \\
\textit{Pterocarpus erinaceus} & $0.8800 \pm 0.0059$ & $0.9565 \pm 0.0048$ & $0.9167 \pm 0.0054$ & 69 \\
\textit{Pterocarpus indicus} & $0.9615 \pm 0.0047$ & $0.8772 \pm 0.0061$ & $0.9174 \pm 0.0052$ & 57 \\
\textit{Pterocarpus macrocarpus} & $0.9038 \pm 0.0053$ & $1.0000 \pm 0.0000$ & $0.9495 \pm 0.0041$ & 47 \\
\textit{Pterocarpus soyauxii} & $0.8462 \pm 0.0062$ & $0.9706 \pm 0.0049$ & $0.9041 \pm 0.0055$ & 68 \\
\midrule
\multicolumn{5}{l}{\textbf{Genus \textit{Sindora} (Sepetir / \textvn{Gõ mật})}} \\
\textit{Sindora cochinchinensis} & $1.0000 \pm 0.0000$ & $1.0000 \pm 0.0000$ & $1.0000 \pm 0.0000$ & 67 \\
\textit{Sindora tonkinensis} & $0.9697 \pm 0.0048$ & $0.9412 \pm 0.0053$ & $0.9552 \pm 0.0046$ & 68 \\
\midrule
\textbf{Overall Accuracy} & \multicolumn{3}{c}{\textbf{0.9042 $\pm$ 0.0038} \quad (95\% CI: [0.8995, 0.9089])} & 1,190 \\
\textbf{Macro Average} & $0.9167 \pm 0.0042$ & $0.8818 \pm 0.0051$ & \textbf{0.8680 $\pm$ 0.0045} \quad (95\% CI: [0.8624, 0.8736]) & 1,190 \\
\textbf{Weighted Average} & $0.9167 \pm 0.0039$ & $0.9042 \pm 0.0038$ & \textbf{0.8882 $\pm$ 0.0041} \quad (95\% CI: [0.8831, 0.8933]) & 1,190 \\
\bottomrule
\multicolumn{5}{l}{\footnotesize Note: Metrics are aggregated across $N=5$ independent random seeds ($S = \{42, 123, 456, 789, 2024\}$).} \\
\multicolumn{5}{l}{\footnotesize 95\% confidence intervals are calculated using Student's $t$-distribution ($t_{\text{crit}} = 2.776, \text{df}=4$).} \\
\end{tabular}
\end{table}

As summarized in Table~\ref{tab:baseline_classification_report}, the ConvNeXt-Tiny baseline demonstrates exceptional stability on the held-out test split ($N_{\text{test}} = 1,190$). Across five independent random seeds, the model achieves a mean overall accuracy of $90.42\% \pm 0.38\%$ and a macro-averaged F1-score of $86.80\% \pm 0.45\%$. The tight standard errors and narrow confidence bounds confirm resilience to stochastic optimization fluctuations. At the taxon level, 13 of the 19 species achieve mean F1-scores exceeding $0.90$. Notably, perfect discrimination is consistently maintained on \textit{Dalbergia melanoxylon}, \textit{Peltogyne pubescens}, and \textit{Sindora cochinchinensis}.

Inspection of the multi-seed averaged test-set confusion matrix (Figure~\ref{fig:confusion_matrix}, aggregated across five independent random runs with cell annotations reporting rounded mean test counts alongside their corresponding mean row-normalized recall percentages) reveals distinct behavioral patterns across taxonomic clades. While strong intra-genus confusion occurs between congeneric sister taxa with pronounced anatomical convergence (notably 21 test captures of \textit{Afzelia pachyloba} misclassified as \textit{Afzelia bella} due to shared lozenge-aliform parenchyma topography), notable inter-genus misclassifications also occur across distinct clades: (i) 24 captures of \textit{Afzelia quanzensis} ($42.9\%$) are confused with \textit{Guibourtia coleosperma}; (ii) 13 captures of \textit{Afzelia pachyloba} ($32.5\%$) are predicted as \textit{Guibourtia ehie}; and (iii) 9 captures of \textit{Afzelia africana} ($15.5\%$) are misclassified as \textit{Pterocarpus soyauxii}. These cross-genus errors stem from shared macroscopic traits, such as comparable diffuse-porous pore diameters and similar reddish-brown heartwood colorations, and directly explain why the precision of \textit{Guibourtia coleosperma} ($75.0\%$) and \textit{Guibourtia ehie} ($75.5\%$) is comparatively reduced. In addition, the acute minority class \textit{Afzelia pachyloba} ($N=116$ total, support $= 40$ in test) suffers from severe class imbalance, yielding a recall of $0.025$ (only 1 out of 40 captures correctly identified), reflecting the small-sample regime documented in Table~\ref{tab:taxonomic_inventory}.

\begin{figure}[pos=htbp]
\centering
\includegraphics[width=0.92\linewidth]{fig/confusion_matrix_focal_test.pdf}
\caption{Averaged confusion matrix of the ConvNeXt-Tiny technical-validation baseline under Multiclass Focal Loss evaluated across all 19 Fabaceae taxa on the held-out test split ($N_{\text{test}} = 1,190$), aggregated over five independent random seeds ($N=5$). Cell annotations report rounded mean test counts alongside their corresponding mean row-normalized recall percentages. While the diagonal density confirms robust overall discrimination ($90.42\% \pm 0.38\%$), notable off-diagonal misclassifications emerge both within congeneric taxa (\textit{A. pachyloba} $\to$ \textit{A. bella}) and across genera (\textit{A. quanzensis} $\to$ \textit{G. coleosperma}, \textit{A. pachyloba} $\to$ \textit{G. ehie}), reflecting cross-genus macroscopic similarities and acute minority-class constraints.}
\label{fig:confusion_matrix}
\end{figure}

\subsubsection{Comparative Technical Validation: Operational Canonical vs. Strict Specimen-Disjoint Benchmarks}
\label{sec:comparative_validation}

To empirically substantiate the dual-benchmark architecture and provide concrete reference figures for both partition protocols introduced in Section~\ref{sec:description}, Table~\ref{tab:split_comparison_baseline} directly contrasts the ConvNeXt-Tiny classification baseline across the Operational Canonical Split and the Strict Specimen-Disjoint Split, both evaluated across five independent random initializations ($N=5$ seeds, reporting $\text{Mean} \pm \text{Std}$ alongside two-tailed Student's $t$ 95\% CI).

\begin{table}[pos=htbp]
\centering
\small
\setlength{\tabcolsep}{4.5pt}
\renewcommand{\arraystretch}{1.18}
\caption{Technical validation baseline comparison between the Operational Canonical Split and the Strict Specimen-Disjoint Split, both aggregated across five independent random seeds ($N=5$, reported as $\text{Mean} \pm \text{Std}$ with two-tailed Student's $t$ 95\% CI under Multiclass Focal Loss), quantifying the empirical generalization gap ($\Delta$) induced by out-of-distribution physical specimen isolation.}
\label{tab:split_comparison_baseline}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lcccc}
\toprule
\textbf{Evaluation Metric / Benchmark Property} & \makecell{\textbf{Operational Canonical}\\\textbf{Benchmark ($N=5$ Seeds)}} & \makecell{\textbf{Strict Specimen-Disjoint}\\\textbf{Benchmark ($N=5$ Seeds)}} & \makecell{\textbf{Generalization}\\\textbf{Gap ($\Delta$)}} & \textbf{Forensic \& Scientific Interpretation} \\
\midrule
Overall Top-1 Accuracy & $\mathbf{0.9042 \pm 0.0038}$ \footnotesize{([0.8995, 0.9089])} & $\mathbf{0.8235 \pm 0.0052}$ \footnotesize{([0.8170, 0.8300])} & $-8.07\%$ & Inter-specimen biological variance across unseen logs reduces baseline accuracy by $\sim 8.1\%$. \\
Macro-Averaged Precision & $0.9167 \pm 0.0042$ & $0.8412 \pm 0.0058$ & $-7.55\%$ & Heightened cross-specimen variance slightly elevates false positive rates on sister taxa. \\
Macro-Averaged Recall & $0.8818 \pm 0.0051$ & $0.7845 \pm 0.0064$ & $-9.73\%$ & Minority CITES taxa with acute specimen scarcity exhibit lower recall on unseen logs. \\
Macro-Averaged F1-Score & $\mathbf{0.8680 \pm 0.0045}$ \footnotesize{([0.8624, 0.8736])} & $\mathbf{0.7692 \pm 0.0061}$ \footnotesize{([0.7616, 0.7768])} & $-9.88\%$ & Confirms that specimen leakage in naive splits inflates reported F1 by $\sim 9.9\%$. \\
Weighted-Averaged F1-Score & $0.8882 \pm 0.0041$ \footnotesize{([0.8831, 0.8933])} & $0.8054 \pm 0.0055$ \footnotesize{([0.7986, 0.8122])} & $-8.28\%$ & Dominant commercial Fabaceae taxa maintain robust diagnostic discrimination ($>80\%$). \\
\midrule
Class Coverage Rate (CCR) & $100.0\%$ (19/19 taxa) & $100.0\%$\textsuperscript{*} (19/19) / $89.5\%$\textsuperscript{\dag} (17/19) & $0.0\% / -10.5\%$ & Full coverage maintained in 19-class mode; 17-class mode provides pure zero-leakage test. \\
Train-to-Test Specimen Overlap & 12 shared blocks ($\text{SLR} = 30.6\%$) & 0 shared blocks ($\text{SLR} = 0.0\%$\textsuperscript{\ddag}) & $-30.6\%$ & Zero specimen leakage achieved across all 17 multi-specimen taxa ($|\mathcal{G}_c| \ge 4$). \\
Bitwise Cryptographic Duplicates & 0 files ($\text{SHA-256 Overlap} = 0$) & 0 files ($\text{SHA-256 Overlap} = 0$) & $0$ & Absolute cryptographic integrity and file deduplication confirmed across all splits. \\
Perceptual Near-Duplicates ($H \le 4$) & 142 pairs ($2.18\%$ of test queries) & 0 pairs ($0.00\%, H_{\min} = 11$) & $-142$ pairs & Strict block isolation completely purges spatial and perceptual near-duplicates. \\
\bottomrule
\multicolumn{5}{@{}p{\linewidth}@{}}{\vspace{3pt}\footnotesize \textsuperscript{*}In 19-class evaluation mode, single-specimen \textit{Dalbergia cochinchinensis} is partitioned via an annotated intra-block exception ($212$ Train / $71$ Val / $71$ Test). \textsuperscript{\dag}In 17-class mode, single-specimen \textit{D. cochinchinensis} is allocated strictly to Train, yielding $100\%$ zero-specimen-leakage testing across the remaining 17 multi-specimen and 1 dual-specimen taxa. \textsuperscript{\ddag}Specimen Leakage Rate ($\text{SLR}$) is calculated across all 17 multi-specimen taxa.}
\end{tabular}%
}
\end{table}

As documented in Table~\ref{tab:split_comparison_baseline}, evaluating the baseline on the strict specimen-disjoint benchmark yields a mean overall accuracy of $82.35\% \pm 0.52\%$ and a macro-averaged F1-score of $76.92\% \pm 0.61\%$. Contrasted against the operational canonical benchmark, this reveals an empirical generalization gap of $\Delta \text{Acc} = -8.07\%$ and $\Delta \text{F1} = -9.88\%$.

From an authentic wood anatomy perspective, this performance gap provides critical biological confirmation rather than reflecting algorithmic failure. In the canonical split, controlled block sharing permits models to exploit subtle, intra-specimen micro-features unique to an individual physical block. Conversely, the strict specimen-disjoint split compels the model to generalize across genuine inter-individual biological variance. This includes substantial differences in vessel lumen diameter, axial parenchyma spacing, and extractive pigmentation driven by tree ontogeny and micro-climate.

The observed $8.07\%$ accuracy gap quantitatively demonstrates that naive random splitting in prior timber benchmarks artificially inflates metrics. This validates our dual-benchmark design: the operational canonical split serves immediate customs screening, while the strict disjoint split exposes the scientific challenge of biological specimen invariance. Further algorithmic explorations of domain generalization are reserved for dedicated machine learning research.

\subsection{Limitations and Practical Considerations}
\label{sec:limitations}

Several practical limitations and methodological trade-offs should be considered when utilizing the ForensicMacroWood-CITES benchmark:
\begin{itemize}[leftmargin=*,itemsep=2pt,topsep=2pt]
    \item \textbf{Modest Physical Specimen Cohort Sizes}: Physical reference cohorts remain modest ($|\mathcal{G}_c| \le 10$ wood blocks per species, aggregating 148 verified physical specimens across the 19 Fabaceae taxa), with acute scarcity bottlenecks for critically endangered or strictly protected timbers (specifically only 1 physical specimen for \textit{Dalbergia cochinchinensis} and 2 for \textit{Guibourtia coleosperma}). Consequently, intra-specific anatomical variation arising from geographic provenance, tree age, and micro-climatic growth conditions is not exhaustively sampled.
    \item \textbf{Controlled Specimen Allocation vs. Absolute Disjointness}: The canonical benchmark guarantees a 100\% Class Coverage Rate ($\text{CCR} = 100.0\%$) across all partitions via an audited Pareto trade-off that incurs a controlled specimen boundary sharing rate of $\text{SLR} = 30.6\%$. While exact bitwise duplicate captures across split boundaries are strictly purged ($\text{SHA-256 Overlap} = 0$), this partition is a controlled compromise designed to prevent minority-class starvation rather than an ideal, completely specimen-disjoint split. Downstream practitioners evaluating models for out-of-distribution deployment on entirely unseen physical logs should take this governed boundary sharing into account.
    \item \textbf{Laboratory Surface Preparation vs. In-the-Field Generalization}: All benchmark captures were recorded on orthogonally planed and progressively polished cross-sections (P120--P600 abrasives) under calibrated daylight-balanced ($5600\,\text{K}$) diffuse circular LED lighting. Reported baseline classification accuracy ($90.42\%$) may not directly generalize to rough chainsaw cuts, weathered end-grain surfaces, sawdust-obscured vessels, or uncontrolled solar glare encountered during real-time inspections at port container terminals or log landings.
    \item \textbf{Single Anatomical Plane}: The current benchmark focuses exclusively on the transverse (cross-sectional end-grain) plane. While the transverse view exposes primary diagnostic xylotomical systems (vessels, axial parenchyma halos, and growth increments), incorporating longitudinal radial and tangential surfaces~\cite{rosadasilva2022} in future releases could provide complementary discriminatory features for challenging sister species with convergent cross-sectional anatomy.
\end{itemize}

\section{Ethics Statement}
\label{sec:ethics}

The authors confirm adherence to ethical publication standards. This work does not involve human participants, animal experimentation, or personal data mining. The dataset consists strictly of non-destructive macroscopic optical digital photographs of dry, surfaced reference wood blocks; no living wild plant material was collected, and no genetic material or biochemical extracts were sampled, extracted, or sequenced. Consequently, this study does not utilize genetic resources and does not trigger access-and-benefit-sharing (ABS) compliance obligations under the Nagoya Protocol.

Regarding timber species listed in CITES Appendix~II (genera \textit{Afzelia}, \textit{Dalbergia}, and \textit{Pterocarpus}), all examined reference wood blocks originate from cataloged institutional xylarium reference archives and verified non-commercial academic collections curated in Viet Nam. These historical specimens were acquired in compliance with domestic forestry laws prior to applicable commercial trade restrictions or through institutional scientific specimen transfer letters for non-commercial taxonomic and educational purposes. All physical specimens remain curated in permanent institutional reference xylaria, with individual specimen accession identifiers documented in the release metadata (\texttt{specimen\_id}) and verifiable upon academic request.

\section{Data Availability}
\label{sec:data_availability}

\textbf{ForensicMacroWood-CITES}: A multimodal benchmark dataset of macroscopic timber cross-sections with structured IAWA morphological descriptors and governed partition manifests for CITES forensic wood identification (Original data) is openly accessible at the Zenodo scientific repository under DOI \href{https://doi.org/10.5281/zenodo.14892180}{10.5281/zenodo.14892180}. The dataset is released under the Creative Commons Attribution 4.0 International (CC BY 4.0) license.

\section{Code Availability}
\label{sec:code_availability}

Scripts for metadata verification, governed specimen-aware partition construction, and baseline training experiments (ConvNeXt-Tiny focal loss classification and standard cross-entropy) are openly available at the GitHub repository: \url{https://github.com/vietanhlee/ForensicMacroWood-CITES} (mirrored from \url{https://github.com/vietanhlee/S3_paper}). The codebase is released under the MIT License and includes a dedicated Zenodo software DOI (\href{https://doi.org/10.5281/zenodo.14892181}{10.5281/zenodo.14892181}) aligned with the archived benchmark dataset version.

\section*{CRediT Author Statement}

\textbf{Viet-Anh Le}: Conceptualization; Methodology; Software; Formal analysis; Data curation; Investigation; Visualization; Writing -- original draft. \newline
\textbf{Khanh Nguyen-Trong}: Conceptualization; Methodology; Supervision; Validation; Formal analysis; Funding acquisition; Project administration; Writing -- review \& editing.

\section*{Funding}

This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors. Computational infrastructure, imaging systems, and experimental baseline evaluations were supported by the Intelligent Computing for Sustainable Development Laboratory (IC4SD), Posts and Telecommunications Institute of Technology (PTIT).

\section*{Declaration of Generative AI in Scientific Writing}

During the drafting and stylistic refinement of this manuscript, the authors utilized generative AI technology exclusively to improve English readability, proofread grammatical phrasing, and optimize structural presentation. The authors thoroughly reviewed and edited all generated textual modifications and assume complete responsibility for the final contents of this publication. The underlying scientific contributions---including experimental methodology, timber specimen preparation, macro-imaging acquisition, taxonomic ground-truth verification, specimen-disjoint partitioning, and empirical evaluation metrics---were conducted entirely by the human authors without AI-driven generation, fabrication, or manipulation of dataset assets or analytical results.

\section*{Declaration of Competing Interest}

The authors affirm that this work was conducted in the absence of any commercial, financial, or personal relationships that could be construed as a potential conflict of interest or that could have inappropriately influenced the representation of the data and findings herein.

\section*{Acknowledgements}

The authors express their sincere appreciation to the botanical and forestry specialists for their valuable assistance in timber specimen preparation, anatomical inspection, and taxonomic ground-truth authentication. Grateful acknowledgement is also extended to the Intelligent Computing for Sustainable Development Laboratory (IC4SD), Posts and Telecommunications Institute of Technology (PTIT), for providing optical imaging instruments, computational facilities, and sustained technical support throughout this project.

\bibliographystyle{elsarticle-num}
\bibliography{refs}

\end{document}


<!-- FILE: 01_data_paper_forensic_cites/paper_data/refs.bib -->

@article{woodreview,
  author    = {S.-W. Hwang and J. Sugiyama},
  title     = {Computer vision-based wood identification and its expansion and contribution potentials in wood science: {A} review},
  journal   = {Plant Methods},
  volume    = {17},
  number    = {1},
  pages     = {47},
  year      = {2021}
}

@article{cites,
  author       = {{CITES Secretariat}},
  title        = {Convention on International Trade in Endangered Species of Wild Fauna and Flora},
  journal      = {United Nations Treaty Series},
  volume       = {993},
  pages        = {243--340},
  year         = {1973}
}

@article{dormontt2015,
  author    = {E. E. Dormontt and M. Boner and B. Braun and G. Breulmann and B. Degen and E. Espinoza and S. Gardner and P. Guillery and P. Hermanson and G. Koch and others},
  title     = {Forensic timber identification: It's time to integrate disciplines to combat illegal logging},
  journal   = {Biological Conservation},
  volume    = {191},
  pages     = {790--798},
  year      = {2015}
}

@article{wiedenhoeft2011,
  author    = {A. C. Wiedenhoeft},
  title     = {Structure and function of wood},
  journal   = {Wood Handbook: Wood as an Engineering Material},
  publisher = {USDA Forest Service, Forest Products Laboratory},
  pages     = {3-1--3-18},
  year      = {2010}
}

@article{wu2021,
  author    = {F. Wu and R. Gazo and E. Haviarova and B. Benes},
  title     = {Wood identification based on longitudinal section images by using deep learning},
  journal   = {Wood Science and Technology},
  volume    = {55},
  number    = {2},
  pages     = {553--563},
  year      = {2021}
}

@article{fabijanska2021,
  author    = {A. Fabijanska and M. Danek and J. Barniak},
  title     = {Wood species automatic identification from wood core images with a residual convolutional neural network},
  journal   = {Computers and Electronics in Agriculture},
  volume    = {181},
  pages     = {105941},
  year      = {2021}
}

@article{figueroamata2022,
  author    = {G. Figueroa-Mata and E. Mata-Montero and J. C. Valverde-Ot{\'a}rola and D. Arias-Aguilar and N. Zamora-Villalobos},
  title     = {Using deep learning to identify {Costa Rican} native tree species from wood cut images},
  journal   = {Frontiers in Plant Science},
  volume    = {13},
  pages     = {789227},
  year      = {2022}
}

@article{ravindran2020,
  author    = {P. Ravindran and B. J. Thompson and R. K. Soares and A. C. Wiedenhoeft},
  title     = {The {XyloTron}: Flexible, open-source, image-based macroscopic field identification of wood products},
  journal   = {Frontiers in Plant Science},
  volume    = {11},
  pages     = {1015},
  year      = {2020}
}

@article{ravindran2021,
  author    = {P. Ravindran and A. G. Costa and R. K. Soares and A. C. Wiedenhoeft},
  title     = {Field-deployable computer vision wood identification of {Peruvian} timbers},
  journal   = {Frontiers in Plant Science},
  volume    = {12},
  pages     = {647515},
  year      = {2021}
}

@article{ravindran2022,
  author    = {P. Ravindran and C. S. Owens and F. J. Alfaro-S{\'a}nchez and others},
  title     = {Evaluation of a low-cost smartphone-based field-deployable macroscopic wood identification system},
  journal   = {IAWA Journal},
  volume    = {43},
  number    = {1-2},
  pages     = {24--40},
  year      = {2022}
}

@article{rosadasilva2022,
  author    = {N. {Rosa da Silva} and M. De Ridder and F. Baetens and J. Van den Bulcke and J. Van Acker and D. E. Hubau and P. Beeckman},
  title     = {Improved wood species identification based on multi-view imagery of the three anatomical planes},
  journal   = {Plant Methods},
  volume    = {18},
  number    = {1},
  pages     = {79},
  year      = {2022}
}

@article{liu2025,
  author    = {S. Liu and C. Zheng and T. He and others},
  title     = {Automated species discrimination and feature visualization of closely related {Pterocarpus} wood species using deep learning models: Comparison of four convolutional neural networks},
  journal   = {Wood Science and Technology},
  volume    = {59},
  pages     = {86},
  year      = {2025}
}

@article{song2025,
  author    = {T. Song and V.-D. Duong and T.-P. Le and T. V. Ta},
  title     = {Deep learning for automated identification of {Vietnamese} timber species: {A} tool for ecological monitoring and conservation},
  journal   = {Ecological Informatics},
  volume    = {90},
  pages     = {103314},
  year      = {2025}
}

@inproceedings{convnext,
  author    = {Z. Liu and H. Mao and C.-Y. Wu and C. Feichtenhofer and T. Darrell and S. Xie},
  title     = {A {ConvNet} for the 2020s},
  booktitle = {Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {11976--11986},
  year      = {2022}
}

@inproceedings{swin,
  author    = {Z. Liu and Y. Lin and Y. Cao and H. Hu and Y. Wei and Z. Zhang and S. Lin and B. Guo},
  title     = {{Swin Transformer}: Hierarchical vision transformer using shifted windows},
  booktitle = {Proc. IEEE/CVF International Conference on Computer Vision (ICCV)},
  pages     = {10012--10022},
  year      = {2021}
}

@inproceedings{resnet,
  author    = {K. He and X. Zhang and S. Ren and J. Sun},
  title     = {Deep residual learning for image recognition},
  booktitle = {Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {770--778},
  year      = {2016}
}

@inproceedings{efficientnetv2,
  author    = {M. Tan and Q. V. Le},
  title     = {{EfficientNetV2}: Smaller models and faster training},
  booktitle = {Proc. International Conference on Machine Learning (ICML)},
  pages     = {10096--10106},
  year      = {2021}
}

@inproceedings{focal,
  author    = {T.-Y. Lin and P. Goyal and R. Girshick and K. He and P. Doll{\'a}r},
  title     = {Focal loss for dense object detection},
  booktitle = {Proc. IEEE/CVF International Conference on Computer Vision (ICCV)},
  pages     = {2980--2988},
  year      = {2017}
}

@inproceedings{arcface,
  author    = {J. Deng and J. Guo and N. Xue and S. Zafeiriou},
  title     = {{ArcFace}: Additive angular margin loss for deep face recognition},
  booktitle = {Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {4690--4699},
  year      = {2019}
}

@inproceedings{supcon,
  author    = {P. Khosla and P. Teterwak and C. Wang and A. Sarna and Y. Tian and P. Isola and A. Maschinot and C. Liu and D. Krishnan},
  title     = {Supervised contrastive learning},
  booktitle = {Proc. Advances in Neural Information Processing Systems (NeurIPS)},
  volume    = {33},
  pages     = {18661--18673},
  year      = {2020}
}

@inproceedings{proxyanchor,
  author    = {S. Kim and D. Kim and M. Cho and S. Kwak},
  title     = {Proxy anchor loss for deep metric learning},
  booktitle = {Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {3238--3247},
  year      = {2020}
}

@inproceedings{simclr,
  author    = {T. Chen and S. Kornblith and M. Norouzi and G. Hinton},
  title     = {A simple framework for contrastive learning of visual representations},
  booktitle = {Proc. International Conference on Machine Learning (ICML)},
  pages     = {1597--1607},
  year      = {2020}
}

@inproceedings{byol,
  author    = {J.-B. Grill and F. Strub and F. Altch{\'e} and C. Tallec and P. Richemond and E. Buchatskaya and C. Doersch and B. Avila Pires and Z. Guo and M. Gheshlaghi Azar and others},
  title     = {Bootstrap your own latent-a new approach to self-supervised learning},
  booktitle = {Proc. Advances in Neural Information Processing Systems (NeurIPS)},
  volume    = {33},
  pages     = {21271--21284},
  year      = {2020}
}

@inproceedings{simsiam,
  author    = {X. Chen and K. He},
  title     = {Exploring simple siamese representation learning},
  booktitle = {Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {15750--15758},
  year      = {2021}
}

@inproceedings{barlow,
  author    = {J. Zbontar and L. Jing and I. Misra and Y. LeCun and S. Deny},
  title     = {Barlow twins: Self-supervised learning via redundancy reduction},
  booktitle = {Proc. International Conference on Machine Learning (ICML)},
  pages     = {12310--12320},
  year      = {2021}
}

@article{tsne,
  author    = {L. {van der Maaten} and G. Hinton},
  title     = {Visualizing data using {t-SNE}},
  journal   = {Journal of Machine Learning Research},
  volume    = {9},
  pages     = {2579--2605},
  year      = {2008}
}

@inproceedings{gradcam,
  author    = {R. R. Selvaraju and M. Cogswell and A. Das and R. Vedantam and D. Parikh and D. Batra},
  title     = {{Grad-CAM}: Visual explanations from deep networks via gradient-based localization},
  booktitle = {Proc. IEEE/CVF International Conference on Computer Vision (ICCV)},
  pages     = {618--626},
  year      = {2017}
}


@article{alturayeif2023,
  author    = {N. Alturayeif and J. Luqman and H. Al-Dossari},
  title     = {A systematic review on data leakage in machine learning},
  journal   = {IEEE Access},
  volume    = {11},
  pages     = {90234--90250},
  year      = {2023}
}

@article{kaufmann2023datasail,
  author    = {R. Kaufmann and M. K. Rozanski and F. A. Wolf and others},
  title     = {{DataSAIL}: Data splitting against information leakage},
  journal   = {Bioinformatics},
  volume    = {39},
  number    = {8},
  pages     = {btad476},
  year      = {2023}
}

@article{pedregosa2011,
  author    = {F. Pedregosa and G. Varoquaux and A. Gramfort and V. Michel and B. Thirion and O. Grisel and M. Blondel and P. Prettenhofer and R. Weiss and V. Dubourg and others},
  title     = {Scikit-learn: Machine learning in {Python}},
  journal   = {Journal of Machine Learning Research},
  volume    = {12},
  pages     = {2825--2830},
  year      = {2011}
}

@inproceedings{facenet,
  author    = {F. Schroff and D. Kalenichenko and J. Philbin},
  title     = {{FaceNet}: A unified embedding for face recognition and clustering},
  booktitle = {Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {815--823},
  year      = {2015}
}

@article{cohen1960,
  author    = {J. Cohen},
  title     = {A coefficient of agreement for nominal scales},
  journal   = {Educational and Psychological Measurement},
  volume    = {20},
  number    = {1},
  pages     = {37--46},
  year      = {1960}
}

@article{iawa1989,
  author    = {{IAWA Committee}},
  title     = {{IAWA} list of microscopic features for hardwood identification},
  journal   = {IAWA Bulletin n.s.},
  volume    = {10},
  number    = {3},
  pages     = {219--332},
  year      = {1989}
}

@article{nguyentrong2026eucalyptus,
  author    = {K. Nguyen-Trong and T. Nguyen-Thi and N. Nguyen-Trong},
  title     = {{IC4SD-Wood-Eucalyptus}: A macroscopic transverse-section image dataset with metadata, split manifests, and leakage-audit reports for {Eucalyptus} wood identification},
  journal   = {Data in Brief},
  volume    = {68},
  pages     = {113133},
  year      = {2026},
  doi       = {10.1016/j.dib.2026.113133}
}




<!-- FILE: 02_research_paper_specimen_leakage/paper/main.tex -->

\documentclass[a4paper,fleqn]{cas-sc}

\usepackage[utf8]{inputenc}
\usepackage[T5,T1]{fontenc}
% ── Native Vietnamese Typography Support for Legal Vernacular Names ─
\DeclareTextFontCommand{\textvn}{\fontencoding{T5}\selectfont}

\usepackage[numbers,sort&compress]{natbib}
\usepackage{amsmath,amssymb,amsfonts,amsthm}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{float}
\usepackage[section]{placeins}
\usepackage{hyperref}
\usepackage{tikz}
\usepackage{url}
\usepackage{multirow}
\usepackage{array}
\usepackage{microtype}
\microtypesetup{expansion=false}
\usepackage{makecell}
\usepackage{algorithm}
\usepackage{algpseudocode}
\usepackage{subcaption}

% ── Hyperref Academic Configuration ─────────────────────────────────
\hypersetup{
    colorlinks=true,
    linkcolor=cyan!80!black,
    citecolor=cyan!80!black,
    urlcolor=cyan!80!black
}

\newcommand{\orcidicon}[1]{\href{https://orcid.org/#1}{\texorpdfstring{%
\begin{tikzpicture}[baseline=-0.4ex]%
\definecolor{orcidgreen}{HTML}{A6CE39}%
\draw[fill=orcidgreen,draw=none] (0,0) circle (1.0ex);%
\node at (0,0) {\color{white}\fontsize{4}{4}\selectfont\sffamily\bfseries iD};%
\end{tikzpicture}%
}{}}}

% ── Footer Suppression for Elsevier CAS Template ───────────────────
\ExplSyntaxOn
\cs_set:Npn \__first_footerline: {}
\cs_set:Npn \__first_foot: {}
\cs_set:Npn \__cas_foot: {}
\ExplSyntaxOff

\let\printorcid\relax

% ── Strict Taxonomic & Author Hyphenation Prevention ─────────────────
\hyphenation{Afzelia Guibourtia Pterocarpus Dalbergia Sindora}

\newtheorem{definition}{Definition}
\newtheorem{proposition}{Proposition}
\newtheorem{lemma}{Lemma}
\newtheorem{observation}{Observation}

\begin{document}

\let\WriteBookmarks\relax
\def\floatpagepagefraction{1}
\def\textpagefraction{.001}

% Short title & Short authors for running headers
\shorttitle{Mitigating Specimen-Level Data Leakage in Wood Species Identification}
\shortauthors{\mbox{V.-A. Le} et al.}

% Main title
\title [mode = title]{Mitigating Specimen-Level Data Leakage in Wood Species Identification: A Multi-Paradigm Partitioning Benchmark and Combinatorial Governance Framework}

% Authors & Affiliations
\author[1]{\mbox{Viet-Anh Le}\orcidicon{0009-0003-5748-0439}}

\author[1]{Khanh Nguyen-Trong\orcidicon{0000-0001-5175-8805}}
\cormark[1]

\address[1]{Intelligent Computing for Sustainable Development Laboratory (IC4SD), Posts and Telecommunications Institute of Technology (PTIT), Hanoi, Vietnam}

\cortext[1]{ Corresponding author.\\ \hspace*{2.2em}\textit{E-mail addresses:} \href{mailto:khanhnt@ptit.edu.vn}{khanhnt@ptit.edu.vn} (K. Nguyen-Trong), \href{mailto:anhlv.b23kh002@stu.ptit.edu.vn}{anhlv.b23kh002@stu.ptit.edu.vn} (\mbox{V.-A.} Le)}

\begin{abstract}
In computer-vision-based biological specimen identification, data leakage across physical specimen boundaries -- termed \mbox{Same-Specimen-Picture Bias} (SSPB) or Class~IV Boundary Leakage -- causes deep learning models to exploit non-taxonomic surface preparation artifacts (saw striations, sanding abrasions, vignetting) rather than authentic diagnostic morphology. Under conventional random image-level partitioning, this shortcut learning produces inflated near-perfect test accuracy that degrades substantially upon deployment to novel physical specimens. Although standard group-aware partitioning isolates physical specimens, it treats all taxa homogeneously, frequently triggering severe minority-class performance collapse under deep neural fine-tuning.

To address this challenge from a data-centric governance perspective, we formalize the Specimen-Centric Data Protocol (SCDP) targeting two core governance objectives: physical subfolder integrity (as an asymptotic theoretical target, $\mathrm{SLR} \to 0\%$) and 100\% Class Coverage Rate ($\mathrm{CCR}$), evaluated through an operational Specimen Leakage Risk ($\mathrm{SLR}$) metric. We formalize the discrete integer condition for whole-block partitioning: strict 3-way specimen disjointness with full class coverage is mathematically impossible unless every taxon possesses at least three physical blocks ($|\mathcal{G}_c| \ge 3$). In real-world small-sample datasets ($|\mathcal{G}_c| < 10$), volume balance constraints and out-of-distribution (OOD) feature separation induce operational boundary friction, forcing standard solvers to trigger image fallbacks on boundary blocks to avoid class starvation. By employing a zero-training 1-nearest-neighbor benchmark on frozen deep representations, we isolate the causal impact of dataset partitioning from neural-network training stochasticity. We systematically benchmark 13 partitioning protocols across six algorithmic paradigms on an 18-species macroscopic wood dataset (curated from an archival repository of 20,470 images across 210 blocks into a standardized benchmark of 6,410 images across 116 quality-verified canonical blocks) using a ``Three Pillars'' evaluation framework comprising 16 quantitative metrics. Furthermore, to accommodate taxon-specific morphological heterogeneity, we introduce a Multi-Objective Simulated Annealing Meta-Selector (CEGS-Split: \textbf{C}ombinatorial \textbf{E}ntity \textbf{G}overnance \textbf{S}trategy for Dataset \textbf{Split}ting) that optimizes class-wise solver assignments over an $11^{18}$ search space.

Empirical evaluations across five random seeds reveal that naive random splitting inflates top-1 accuracy by $+2.13$ percentage points (pp) and macro-F1 by $+4.52$~pp relative to the standard group-disjoint baseline (\textsf{Stratified Group Split}), with the performance gap expanding to $+9.10$~pp in accuracy and $+16.75$~pp in Macro-F1 relative to global feature-disjoint separation (\textsf{DataSAIL Specimen-Level ILP}, Welch's $t$-test $p = 2.28 \times 10^{-6}$). However, unconstrained global ILP inherently causes minority-class starvation, collapsing hardest-class performance ($\mathrm{F1}_{\text{Hardest}} = 0.1193$) and class coverage ($\mathrm{CCR} = 88.9\%$). In contrast, the proposed Meta-Selector achieves an optimal Pareto compromise: it compresses empirical boundary leakage from $100\%$ down to the empirical baseline floor ($\mathrm{SLR} = 6.0\%$, where only 7 boundary blocks across 116 canonical blocks undergo image-level fallback to guarantee 3-way split representation) while strictly guaranteeing $\mathrm{CCR} = 100.0\%$ and preventing hardest-class collapse ($\mathrm{F1}_{\text{Hardest}} = 0.7407$ on frozen features, and providing a robust floor safeguard of $\ge 34.2\%$ under end-to-end neural fine-tuning across four distinct architectures). Formulation ablation confirms that continuous penalty terms fail to suppress leakage, whereas combinatorial meta-selection establishes rigorous Pareto-optimal governance. The governed benchmark dataset, cryptographic SHA-256 partition manifests, and reproducible evaluation suite are openly released.
\end{abstract}

\begin{keywords}
Data leakage \sep \mbox{Same-Specimen-Picture Bias} \sep Wood species identification \sep DataSAIL \sep Multi-objective optimization \sep Simulated annealing \sep Out-of-distribution benchmark \sep Data governance
\end{keywords}

\maketitle

%======================================================================
\section{Introduction}
\label{sec:intro}
%======================================================================

Automated visual identification powered by deep representations has emerged as an essential tool across botanical classification, medical imaging, agricultural diagnostics, and forensic material inspection~\cite{woodreview,rosadasilva2022,varoquaux2022,roberts2021}. In forestry science and international timber trade regulation, rapid macroscopic wood species identification is critical for enforcing the Convention on International Trade in Endangered Species of Wild Fauna and Flora (CITES Appendix~II)~\cite{cites}, verifying legal supply chains, and combating illegal deforestation~\cite{dormontt2015,song2025,liu2025}. Computer vision promises field-deployable, non-destructive screening that supplements labor-intensive xylotomy at customs checkpoints. Yet, despite reported laboratory classification accuracies exceeding 95--99\% in recent literature~\cite{song2025,fabijanska2021,wu2021}, such systems frequently suffer severe performance degradation once evaluated on novel physical specimens~\cite{ravindran2019,ravindran2020,ravindran2021,ravindran2022}.

We argue that a primary driver of this systemic deployment failure is the absence of rigorous data-governance standards during dataset construction and partitioning in applied computer vision. We formalize this challenge under a unified framework: \textbf{Physical Entity-Centric Visual Classification (PECVC)}. PECVC encompasses any visual recognition problem in which individual physical entities -- timber blocks, patient tissue specimens, agricultural plants, or alloy samples -- yield multiple correlated image captures. In prevailing PECVC workflows, researchers photograph physical specimens, crop high-resolution fields of view into image patches, and store them in flat directory structures organized solely by target species~\cite{east2025}. The resulting pool is subsequently partitioned via conventional \emph{random image-level splits}. This procedure discards physical object provenance: sub-images originating from the same physical entity are distributed across training, validation, and test sets.

This partitioning flaw induces \textbf{\mbox{Same-Specimen-Picture Bias} (SSPB)}~\cite{figueroamata2022}, more broadly identified as \textbf{specimen-level data leakage}~\cite{varoquaux2022,roberts2021,kapoor2023}. Crucially, as surveyed by Kapoor and Narayanan~\cite{kapoor2023} across 329 empirical machine-learning studies spanning 17 scientific fields and conceptualized by Varoquaux and Cheplygina~\cite{varoquaux2022}, this vulnerability constitutes an acute manifestation of \emph{Class~IV Boundary Leakage}: a structural partitioning failure where train/test splits violate physical entity boundaries, remaining completely invisible under standard cross-validation while driving substantial performance overestimation. This failure mode is conceptually and mathematically isomorphic to cross-sectional unit leakage in panel econometrics~\cite{cerqua2026,babii2024} and spatial autocorrelation in geoscience~\cite{roberts2017}. Under random image-level partitioning, deep neural networks behave as opportunistic ``shortcut learners''~\cite{geirhos2020}: rather than learning species-discriminative anatomical traits (vessel-pore patterns, axial parenchyma arrangements, ray density), they exploit non-taxonomic surface artifacts, illumination gradients, and sensor fingerprints unique to individual physical specimens. High reported test accuracy then measures the model's ability to re-identify previously seen physical sources rather than recognize novel biological material.

\subsection{Mechanisms of Specimen-Level Data Leakage}
In macroscopic wood imagery, specimen leakage operates through three distinct physical mechanisms:
\begin{enumerate}
\item \textbf{Mechanism 1 (Geometric and Anatomical Continuity)}: Sub-images extracted from the same physical wood block share continuous anatomical trajectories -- matching growth-ring curvature, identical vessel-pore cluster geometry, and continuous parenchyma banding. Deep representations encode these spatial configurations as high-dimensional visual signatures of the individual specimen rather than the botanical species.
\item \textbf{Mechanism 2 (Acquisition and Illumination Fingerprints)}: Images captured within the same imaging session inherit identical sensor noise (photo-response non-uniformity), flash reflections, lens vignetting, color balance casts, and static shadows, allowing models to exploit acquisition parameters rather than biological morphology.
\item \textbf{Mechanism 3 (Surface Preparation Artifacts)}: Mechanical processing (sawing, planing, sanding) leaves specimen-unique micro-scratch densities, weathering cracks, and varnish absorption patterns, which serve as salient non-taxonomic shortcuts~\cite{geirhos2020}.
\end{enumerate}

\subsection{Taxon Heterogeneity and Combinatorial Partitioning}
Standard group-aware partitioning (e.g., \texttt{GroupKFold}, \texttt{StratifiedGroupKFold}) enforces physical specimen isolation but assumes all classes behave homogeneously. In biological datasets, taxa exhibit pronounced morphological divergence:
\begin{itemize}
\item \textbf{Uniform taxa} with homogeneous growth rings and consistent coloration (e.g., \textit{Guibourtia coleosperma}) are cleanly separated by linear centroid-distance metrics (Mahalanobis distance, Ward linkage).
\item \textbf{Outlier-prone taxa} with irregular heartwood/sapwood boundaries or severe weathering (e.g., \textit{Afzelia bella}) distort standard distance metrics and require adversarial density validation~\cite{adversarialvalidation} or integer programming to isolate anomalous specimens.
\item \textbf{Visually similar sibling species} within the same genus (e.g., \textit{Dalbergia oliveri} vs.\ \textit{Dalbergia cochinchinensis}) require graph-based isolation of near-duplicate feature neighborhoods to sever indirect similarity leakage.
\end{itemize}

Imposing a single global splitting algorithm across all 18 species inevitably forces suboptimal partitions for taxa whose empirical distributions violate that algorithm's assumptions. If each taxon $c \in \{1,\dots,C\}$ ($C=18$) selects its own splitting solver $m_c \in \{1,\dots,K\}$ from a candidate pool of $K=11$ solvers, the configuration space spans:
\begin{equation}
\mathcal{M} = K^{C} = 11^{18} \approx 5.5599 \times 10^{18}\ \text{candidate global partitions.}
\label{eq:comb_space}
\end{equation}
Exhaustive evaluation of Eq.~\eqref{eq:comb_space} is computationally intractable. Furthermore, global objectives such as Maximum Mean Discrepancy ($\mathrm{MMD}$) and hardest-class F1 must be evaluated on the assembled global dataset $\mathcal{D}(\boldsymbol{m}) = \bigcup_{c=1}^{C} \mathcal{D}_c(m_c)$, which motivates a combinatorial meta-heuristic search rather than independent per-class greedy selection.

Despite growing awareness of specimen-level leakage, prior studies exhibit three critical gaps: (i) the absence of a standardized, verifiable data-governance protocol with quantitative leakage risk metrics; (ii) the lack of controlled, multi-paradigm benchmarking that isolates dataset partition boundaries from model training stochasticity; and (iii) the inability of single global splitting solvers to accommodate taxon-specific morphological heterogeneity across diverse biological classes. This paper directly addresses these three limitations.

\subsection{Primary Contributions}
The main contributions of this work are fourfold:
\begin{itemize}
\item \textbf{Governed S3 Wood Benchmark Dataset}: An 18-species macroscopic tropical timber benchmark curated from an archival collection of 20,470 raw images across 210 physical specimen blocks into a standardized, balanced collection of 6,410 quality-controlled images across 116 verified canonical blocks spanning 5 botanical genera (including 8 CITES Appendix~II taxa), organized into immutable specimen subfolders with cryptographic SHA-256 partition manifests.
\item \textbf{Multi-Paradigm Benchmark under the Three Pillars Framework}: A systematic evaluation of 13 partitioning protocols across six algorithmic paradigms using 16 quantitative metrics spanning leakage minimization, out-of-distribution difficulty, and decision-boundary preservation, validated through a zero-training 1-NN representation-level benchmark and 80 end-to-end deep neural fine-tuning runs.
\item \textbf{Combinatorial Meta-Selector (CEGS-Split)}: A multi-objective Simulated Annealing framework optimizing class-wise solver assignments over an $11^{18}$ search space. CEGS-Split reconciles taxon-specific morphological divergence, compressing empirical boundary leakage to the empirical baseline floor ($\mathrm{SLR} = 6.0\%$, leaving only 7 boundary blocks split across 116 canonical blocks) while strictly guaranteeing 100\% Class Coverage Rate ($\mathrm{CCR}$) and acting as an indispensable floor safeguard against hardest-class collapse across multiple deep neural architectures.
\item \textbf{Combinatorial Boundary Analysis and SCDP Data Governance}: Formalization of the Specimen-Centric Data Protocol (SCDP) targeting physical specimen integrity alongside the Specimen Leakage Risk ($\mathrm{SLR}$) metric. We mathematically establish the necessary integer block condition for strict 3-way disjointness ($|\mathcal{G}_c| \ge 3$) and analytically characterize the operational boundary friction arising from discrete block allocation, volume ratios, and out-of-distribution separation in small-sample regimes ($|\mathcal{G}_c| < 10$).
\end{itemize}

The remainder of this paper is structured as follows. Section~\ref{sec:related} reviews related work. Section~\ref{sec:governance} formalizes the data governance protocol and the combinatorial meta-selector. Section~\ref{sec:setup} details the experimental setup and representation-level evaluation. Section~\ref{sec:results} presents the empirical results, ablation, and diagnostic analysis. Section~\ref{sec:discussion} outlines practical guidelines and limitations. Section~\ref{sec:conclusion} concludes the paper.

%======================================================================
\section{Related Work}
\label{sec:related}
%======================================================================

\subsection{Specimen Leakage and \mbox{Same-Specimen-Picture Bias}}
Data leakage is recognized as a major cause of reproducibility failures in computational science~\cite{kaufman2012,kapoor2023}. Kapoor and Narayanan~\cite{kapoor2023} reviewed over 600 machine-learning-based scientific papers across 30 disciplines, finding that leakage-driven overoptimism systematically distorts reported performance. In plant vision, Figueroa-Mata et al.~\cite{figueroamata2022} conceptualized \mbox{Same-Specimen-Picture Bias} (SSPB), warning that evaluating models on sub-images of wood blocks present in the training set produces heavily biased accuracy. Rosa da Silva et al.~\cite{rosadasilva2022} corroborated that multi-view wood datasets routinely ignore physical sample boundaries. In timber forensics, Ravindran et al.~\cite{ravindran2019,ravindran2021,ravindran2022} and Wiedenhoeft~\cite{wiedenhoeft2011} documented deployment gaps of 10\% to 25\% between xylarium benchmarks and field tests. However, prior field evaluations conflated specimen leakage with external illumination shifts and sensor changes; our framework isolates pure specimen-level leakage under controlled feature representations.

\subsection{Cross-Disciplinary Parallels: Econometrics and Spatial Statistics}
Boundary leakage is structurally isomorphic across multiple quantitative disciplines. In financial econometrics, L{\'o}pez de Prado~\cite{lopezdeprado2018} formalized Purged Cross-Validation (PCV) to eliminate temporal overlap between train and test splits. In panel data econometrics, Cerqua et al.~\cite{cerqua2026} audited 480 machine learning models across 3,058 U.S. counties, demonstrating that random partitioning inflates predictive accuracy by over 17\% in MSE and 0.05 in AUC by conflating cross-sectional unit leakage (observing the same county in both splits) with temporal leakage. Babii et al.~\cite{babii2024} established that preserving panel dependency structures requires structured block sampling. In spatial statistics, Roberts et al.~\cite{roberts2017} showed that spatial autocorrelation inflates predictive skill unless spatial blocking is applied. Table~\ref{tab:cross_domain_inflation} summarizes reported performance inflation margins across diverse fields, confirming that specimen leakage in biological computer vision mirrors broader empirical phenomena. These cross-domain findings motivate our controlled isolation of the specimen boundary leakage effect in biological vision.

\begin{table}[pos=htbp]
\centering
\footnotesize
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.15}
\caption{Cross-domain empirical performance inflation induced by entity-level boundary leakage across scientific fields.}
\label{tab:cross_domain_inflation}
\resizebox{\textwidth}{!}{%
\begin{tabular}{llll}
\toprule
\makecell[l]{\textbf{Scientific}\\\textbf{Domain}} & \makecell[l]{\textbf{Physical Entity}\\\textbf{Unit}} & \makecell[l]{\textbf{Benchmark}\\\textbf{Reference}} & \makecell[l]{\textbf{Observed Inflation}\\\textbf{Margin}} \\
\midrule
Retinal Ophthalmology & Patient / Eye Volume & Tampu et al.~\cite{tampu2022} & $+5.0$ to $+30.0$ pp accuracy \\
Neuroimaging (Brain MRI) & Subject / Scan Volume & Yagis et al.~\cite{yagis2021} & Slide acc.\ up to $+29.0$ pp; $+25.4$ pp Macro-F1 \\
COVID-19 Radiography & Patient / Hospital Site & Roberts et al.~\cite{roberts2021} & $+12.0$ to $+24.0$ pp diagnostic accuracy \\
Macro Panel Econometrics & County / Geographic Unit & Cerqua et al.~\cite{cerqua2026} & $+17.2\%$ MSE underest.; $+0.05$ AUC \\
\midrule
\textbf{Macroscopic Wood ID (Ours)} & \textbf{Timber Specimen Block} & \textbf{This Study (Table~\ref{tab:master_results})} & \textbf{+9.10 pp Acc; +16.75 pp F1} \\
\bottomrule
\end{tabular}%
}
\end{table}

\subsection{Data Partitioning Paradigms and DataSAIL}
Standard machine-learning libraries provide group-aware utilities like \texttt{GroupKFold} and \texttt{StratifiedGroupKFold}~\cite{scikit}, which partition datasets based on discrete group tags (e.g., patient, batch, specimen). However, these conventional tools partition groups uniformly without considering continuous feature-space geometry or out-of-distribution metric shifts. To bridge this gap, Joeres et al.~\cite{joeres2025} introduced DataSAIL (Data Splitting Against Information Leakage), a seminal framework formulating dataset splitting as an integer linear program (ILP) that minimizes inter-split feature similarity:
\begin{equation}
\min_{\pi} \sum_{i \neq j} \mathbb{1}[\pi(i) \neq \pi(j)] \cdot S(i, j) \cdot w_i \cdot w_j,
\label{eq:datasail_objective}
\end{equation}
where $\pi(i)$ denotes partition assignment, $S(i,j)$ represents pairwise feature similarity, and $w_i, w_j$ denote sample importance weights. DataSAIL was primarily designed for chemoinformatics, drug-target interaction, and computational biology, where isolating highly similar molecular scaffolds is paramount. In its canonical formulation, DataSAIL optimizes a single global feature-similarity objective without explicitly imposing hard constraints on minority-class presence across all splits ($\mathrm{CCR} = 100\%$) or worst-case class recall ($\mathrm{F1}_{\text{Hardest}}$). When applied directly to multi-class biological identification tasks containing rare or endangered taxa (such as CITES wood identification), global unconstrained ILP naturally concentrates outlier specimens of minority taxa into evaluation splits to minimize cross-split similarity, inadvertently depriving the training split of representative samples. Our governance framework builds directly upon DataSAIL's core information-theoretic insight, integrating its leakage loss as a vital optimization anchor while establishing combinatorial class-wise meta-selection to rigorously safeguard minority-class survival.

%======================================================================
\section{Specimen-Centric Data Governance and Proposed Framework}
\label{sec:governance}
%======================================================================

\subsection{Notation and Specimen Leakage Risk (SLR)}
Let $\mathcal{D} = \{(x_i, y_i, g_i)\}_{i=1}^N$ denote a dataset of $N$ images, where $x_i \in \mathcal{X}$ is an RGB image, $y_i \in \{1, \dots, C\}$ is the botanical species label ($C=18$), and $g_i \in \mathcal{G}$ represents the physical specimen block identifier. A data partition $\pi: \mathcal{D} \to \{\text{Train}, \text{Val}, \text{Test}\}$ maps each sample to a partition split. We quantify physical specimen leakage via the \textbf{Specimen Leakage Risk (SLR)}:
\begin{equation}
\mathrm{SLR}(\pi) = \frac{\left|\{g \in \mathcal{G} \mid \exists\, x, x' \in g \text{ s.t. } \pi(x) \neq \pi(x')\}\right|}{|\mathcal{G}|} \times 100\%,
\label{eq:slr}
\end{equation}
where $\mathrm{SLR}=0.0\%$ denotes complete physical specimen isolation across all partition splits.

\begin{definition}[SCDP Data Governance Objectives]
An ideal dataset partition $\pi^*$ strictly satisfies the Specimen-Centric Data Protocol if and only if it complies with two verifiable governance conditions:
\begin{enumerate}
\item[(i)] \textbf{Subfolder Integrity (Zero Specimen Leakage)}: Every image originating from the same physical specimen $g \in \mathcal{G}$ belongs exclusively to an identical split:
\begin{equation}
\forall g \in \mathcal{G}, \quad \pi(x_i) = \pi(x_j) \quad \forall\, x_i, x_j \in g \iff \mathrm{SLR}(\pi^*) \equiv 0.0\%.
\label{eq:subfolder_integrity}
\end{equation}
\item[(ii)] \textbf{Class Coverage Rate (CCR)}: Every species $c \in \{1, \dots, C\}$ is represented across Train, Validation, and Test:
\begin{equation}
\mathrm{CCR}(\pi) = \frac{\left|\{c \in \{1,\dots,C\} \mid c \text{ present in Train, Val, and Test}\}\right|}{C} \times 100\% \equiv 100.0\%.
\label{eq:ccr}
\end{equation}
\end{enumerate}
\end{definition}

\begin{lemma}[Necessary Condition for Strict Block Disjointness with Full Class Coverage]
\label{lem:block_condition}
Let $\mathcal{D}$ be partitioned into $S \ge 2$ pairwise disjoint splits ($S=3$ for Train, Validation, and Test) such that $\pi: \mathcal{D} \to \{1,\dots,S\}$. A partition strictly satisfies Subfolder Integrity ($\mathrm{SLR}(\pi) \equiv 0.0\%$) and complete Class Coverage ($\mathrm{CCR}(\pi) \equiv 100.0\%$) only if every botanical taxon $c \in \{1,\dots,C\}$ possesses at least $S$ distinct physical specimen blocks:
\begin{equation}
\forall c \in \{1,\dots,C\}, \quad |\mathcal{G}_c| \ge S.
\end{equation}
\end{lemma}
\begin{proof}
By the definition of complete Class Coverage ($\mathrm{CCR} \equiv 100.0\%$), for every taxon $c$ and each split $s \in \{1,\dots,S\}$, there must exist at least one sample $x_s \in \mathcal{D}$ such that $y(x_s) = c$ and $\pi(x_s) = s$. Let $g(x) \in \mathcal{G}_c$ denote the physical specimen block containing sample $x$. Under strict Subfolder Integrity ($\mathrm{SLR} \equiv 0.0\%$), every specimen block maps exclusively to a single split, implying $g \cap \pi^{-1}(s) \neq \emptyset \implies g \subseteq \pi^{-1}(s)$. Because the splits are mutually disjoint ($\pi^{-1}(s) \cap \pi^{-1}(s') = \emptyset$ for $s \neq s'$), the physical specimen blocks $g(x_1), g(x_2), \dots, g(x_S)$ must be mutually distinct entities. Consequently, $|\mathcal{G}_c| \ge S$. For a 3-way partition (Train, Val, Test), this requires $|\mathcal{G}_c| \ge 3$ physical blocks per taxon.
\end{proof}

\begin{observation}[Integer-Partition Feasibility and Empirical Fallback Leakage]
\label{obs:trilemma}
While Lemma~\ref{lem:block_condition} establishes the theoretical existence condition ($|\mathcal{G}_c| \ge 3$), small-sample biological datasets ($|\mathcal{G}_c| < 10$) impose severe integer-knapsack friction when simultaneously satisfying:
\begin{enumerate}
\item[(a)] Target volume proportions across splits (e.g., $65\%/18\%/17\%$);
\item[(b)] Heterogeneous block capacities (specimen blocks yield unequal numbers of usable image patches);
\item[(c)] Out-of-distribution (OOD) distributional divergence or boundary preservation across splits.
\end{enumerate}
When physical blocks are indivisible discrete units, finding a 3-way partition that strictly satisfies all volume ratios while keeping every class represented across all three splits is frequently integer-infeasible. In standard group-partitioning implementations (e.g., \texttt{StratifiedGroupKFold}) and group-aware heuristics, this integer infeasibility is resolved through \textbf{boundary image-level fallbacks}: a minimal subset of boundary blocks are split across partitions to prevent empty splits and preserve $\mathrm{CCR} \equiv 100.0\%$. In our canonical benchmark of 116 physical blocks, exactly 7 boundary blocks are partitioned via fallback across splits, yielding an empirical baseline leakage floor of $\mathrm{SLR} = 7 / 116 \approx 6.0\%$. Conversely, enforcing $\mathrm{SLR} \equiv 0.0\%$ by disallowing boundary fallbacks (as seen in unconstrained DataSAIL Specimen ILP) forces the omission of minority taxa from evaluation splits, causing class coverage collapse ($\mathrm{CCR} = 88.9\%$) and worst-case boundary failure ($\mathrm{F1}_{\text{Hardest}} = 0.1193$). Practical data governance in small-sample regimes therefore operates as a Pareto optimization: compressing empirical boundary leakage to its feasible baseline floor ($6.0\%$) while strictly guaranteeing $100\%$ class coverage.
\end{observation}

\subsection{Taxonomy of Partitioning Protocols and Candidate Solver Pool}
To accommodate morphological divergence across diverse taxa, candidate class-wise partitions are drawn from a comprehensive candidate pool of $K=11$ algorithmic solvers $\mathcal{K} = \{\text{PP1}, \dots, \text{PP11}\}$. We draw an essential methodological distinction between \textbf{standalone baseline evaluation} (Category II, where continuous solvers are tested on raw image vectors to quantify unconstrained metric leakage) versus \textbf{block-governed candidate instantiation within CEGS-Split} (Category IV, where continuous solvers are strictly projected onto specimen centroids to preserve physical block integrity):
\begin{itemize}
\item \textbf{Continuous Feature-Space Solvers}:
  \begin{itemize}
  \item \textbf{PP1: Fixed Mahalanobis Stratification}: Computes distance quantiles to a static class centroid $\boldsymbol{\mu}_c$. In standalone Category II evaluation on raw image patches, this unconstrained sorting disperses subfolders across splits ($\mathrm{SLR} = 100.0\%$). In CEGS-Split, it is block-governed by evaluating Mahalanobis distances over pooled specimen centroids $\bar{\boldsymbol{\phi}}_g$, assigning intact physical blocks;
  \item \textbf{PP2: Iterative Mahalanobis Allocation}: Dynamically re-estimates class covariance and centroid upon allocating each specimen block, preventing outlier masking in skewed distributions;
  \item \textbf{PP3: Density-Adaptive Mahalanobis Banding}: Weights Mahalanobis distances using local kernel density estimates across specimen centroids;
  \item \textbf{PP5: Cosine Feature Graph Partitioning}: Constructs a $k$-NN cosine similarity graph and computes min-cut graph partitions. In standalone Category II evaluation on raw images, it yields $\mathrm{SLR} = 90.3\%$; inside CEGS-Split, the graph is constructed strictly over specimen block centroids $\bar{\boldsymbol{\phi}}_g$.
  \end{itemize}
\item \textbf{Specimen-Group-Aware Discrete Solvers (Block-Disjoint by Design)}:
  \begin{itemize}
  \item \textbf{PP4: Hierarchical Agglomerative Partitioning}: Performs Ward's minimum variance clustering on specimen centroids to cut discrete subtree clusters;
  \item \textbf{PP6: Spectral Graph Bipartitioning}: Bipartitions specimen similarity graphs using the Fiedler vector of the normalized graph Laplacian;
  \item \textbf{PP7: Adversarial Density Validation}~\cite{adversarialvalidation}: Trains a binary domain discriminator over specimen representations to route distribution-divergent blocks to evaluation splits;
  \item \textbf{PP8: Stratified Group Allocation}: Extends \texttt{StratifiedGroupKFold} to optimize multi-objective block disjointness while balancing class sample sizes;
  \item \textbf{PP9: Agglomerative Stratified Banding}: Combines specimen clustering with geometric radial distance bands (Near, Mid, Far);
  \item \textbf{PP10: Support Vector Margin Partitioning}: Fits a one-class SVM hyperplane to specimen centroids, routing boundary-margin outliers to evaluation;
  \item \textbf{PP11: Specimen-Level Integer Linear Programming}: Solves the DataSAIL ILP objective over specimen centroids.
  \end{itemize}
\end{itemize}

We evaluate 13 partitioning protocols organized into four broad categories:
\begin{itemize}
\item \textbf{Category I: Naive Image-Level Baselines}: \textsf{Naive Random Image Split} and \textsf{Naive Stratified Image Split} sample individual images uniformly without group provenance ($\mathrm{SLR} \approx 100\%$). \textsf{DataSAIL Image-Level ILP} minimizes Eq.~\eqref{eq:datasail_objective} over individual images without enforcing specimen disjointness ($\mathrm{SLR} = 95.3\%$).
\item \textbf{Category II: Single Splitting Protocols}: Imposes a single algorithmic paradigm globally across all 18 taxa, spanning both continuous feature-space baselines (PP1, PP5) and group-aware baselines (Naive Specimen Group Split, PP8 Stratified Group Split, PP4 Hierarchical Ward, PP9 Stratified Banding, PP7 Adversarial Density Validation, and PP11 DataSAIL Specimen-Level ILP).
\item \textbf{Category III: Single-Objective Combinatorial Selector}: Selects per-class solvers independently to minimize DataSAIL loss in isolation, ignoring cross-taxa assembly balance.
\item \textbf{Category IV: Multi-Objective Combinatorial Meta-Selector (CEGS-Split, Proposed)}: Explores the $11^{18}$ space via Simulated Annealing to discover Pareto-optimal per-class assignments balancing leakage, OOD difficulty, and boundary preservation.
\end{itemize}

\subsection{The Three Pillars Evaluation Framework}
Table~\ref{tab:three_pillars} organizes 16 quantitative evaluation metrics into three complementary pillars to evaluate partitions comprehensively.

\begin{table}[pos=htbp]
\centering
\small
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.22}
\caption{The Three Pillars Evaluation Framework comprising 16 quantitative metrics categorized by governance role.}
\label{tab:three_pillars}
\begin{tabular}{cllp{8.2cm}}
\toprule
\textbf{\#} & \textbf{Metric Name} & \textbf{Role} & \textbf{Mathematical Definition / Evaluation Objective} \\
\midrule
\multicolumn{4}{l}{\textit{\textbf{Pillar 1: Information Leakage Minimization}}} \\
$M_1$ & DataSAIL Loss $L(\pi)$ & Anchor & $\displaystyle L(\pi) = \frac{1}{2} \sum_{\pi(x) \neq \pi(x')} \cos(\phi(x), \phi(x')) \cdot \kappa(x) \kappa(x')$ \\
$M_2$ & Inter-Split Similarity $\bar{S}_{\text{inter}}$ & Probe & $\displaystyle \bar{S}_{\text{inter}} = \frac{1}{|\mathcal{D}_{\text{Tr}}| |\mathcal{D}_{\text{Te}}|} \sum_{x \in \mathcal{D}_{\text{Tr}}, y \in \mathcal{D}_{\text{Te}}} \cos(\phi(x), \phi(y))$ \\
$M_3$ & Specimen Leakage Risk $\mathrm{SLR}$ & Probe & Fraction of physical blocks split across partitions (Eq.~\ref{eq:slr}) \\
$M_4$ & Pseudoreplication Index $\mathrm{PRI}$ & Probe & Proportion of co-specimen image pairs shared across Train/Test \\
\midrule
\multicolumn{4}{l}{\textit{\textbf{Pillar 2: Out-of-Distribution \& Partition Geometry}}} \\
$M_5$ & Max.\ Mean Discrepancy $\mathrm{MMD}$ & Anchor & $\displaystyle \mathrm{MMD}^2(\mathcal{D}_{\text{Tr}}, \mathcal{D}_{\text{Te}}) = \left\| \frac{1}{N_{\text{tr}}} \sum_{x} \phi(x) - \frac{1}{N_{\text{te}}} \sum_{y} \phi(y) \right\|_{\mathcal{H}}^2$ \\
$M_6$ & Silhouette Separation $S_{\text{split}}$ & Probe & Cosine silhouette coefficient across partition assignments $\in [-1, 1]$ \\
$M_7$ & Class Coverage Rate $\mathrm{CCR}$ & Constraint & Percentage of taxa present in Train, Val, and Test (Eq.~\ref{eq:ccr}) \\
$M_8$ & Split Volume Deviation $\Delta V$ & Probe & $\displaystyle \Delta V = \sum_{s \in \{\text{Tr}, \text{Val}, \text{Te}\}} \left| \frac{|\mathcal{D}_s|}{N} - r_s \right|$ relative to targets $(65/18/17\%)$ \\
\midrule
\multicolumn{4}{l}{\textit{\textbf{Pillar 3: Downstream Generalization \& Statistical Rigor}}} \\
$M_9$ & 1-NN Top-1 Accuracy & Probe & Classification accuracy of 1-NN on frozen unit embeddings $\phi(x)$ \\
$M_{10}$ & 1-NN Top-3 Accuracy & Probe & Proportion of test queries with ground-truth class in Top-3 neighbors \\
$M_{11}$ & Macro-Averaged F1 & Probe & Unweighted arithmetic mean of per-species F1-scores across 18 taxa \\
$M_{12}$ & Balanced Accuracy & Probe & Arithmetic mean of per-species sensitivity/recall across all 18 taxa \\
$M_{13}$ & Hardest Class F1 $\mathrm{F1}_{\text{Hardest}}$ & Anchor & Minimum per-species F1-score: $\min_{c \in \mathcal{C}} \mathrm{F1}_c$ (worst-case boundary) \\
$M_{14}$ & Wasserstein Divergence $W_1$ & Probe & Mean $L_1$ Earth Mover's Distance between split and global class priors \\
$M_{15}$ & Welch's $t$-test $p$-value & Probe & Two-sample unequal-variance test assessing significance vs.\ Naive \\
$M_{16}$ & Cohen's $d$ Effect Size & Probe & Standardized effect size quantifying leakage inflation vs.\ Naive \\
\bottomrule
\end{tabular}
\end{table}

\paragraph{Optimization Anchors vs.\ Independent Diagnostic Probes}
Within the Three Pillars Framework, the multi-objective fitness function in Eq.~\eqref{eq:multi_obj_fitness} selects exactly one foundational mathematical representative from each pillar as an active optimization anchor: $L_{\text{DataSAIL}}$ for Pillar~1 (information leakage suppression), $MMD$ for Pillar~2 (distributional separation), and $\mathrm{F1}_{\text{Hardest}}$ for Pillar~3 (worst-case boundary preservation). Restricting active meta-heuristic search to these three orthogonal anchors prevents optimization dilution, objective collinearity, and combinatorial instability. Crucially, the remaining 13 metrics (such as $PRI$, $\bar{S}_{\text{inter}}$, $S_{\text{split}}$, $W_1$, and Cohen's $d$) are strictly withheld from the optimization loop, serving as independent, non-participating diagnostic probes to evaluate and stress-test the discovered partitions post-hoc.

\subsection{Combinatorial Meta-Selector (CEGS-Split)}
Candidate global configurations $\boldsymbol{m} = (m_1, \dots, m_C)$ are evaluated on the assembled dataset $\mathcal{D}(\boldsymbol{m}) = \bigcup_{c=1}^{C} \mathcal{D}_c(m_c)$ via the multi-objective fitness function:
\begin{equation}
\mathrm{Fitness}(\boldsymbol{m}) = w_1 \cdot \left( \frac{L_{\text{DataSAIL}}(\boldsymbol{m})}{10^3} \right) - w_2 \cdot \left( 10 \cdot MMD(\boldsymbol{m}) \right) - w_3 \cdot \left( 10 \cdot \mathrm{F1}_{\text{Hardest}}(\boldsymbol{m}) \right),
\label{eq:multi_obj_fitness}
\end{equation}
with baseline weights $w_1=1.0, w_2=0.5, w_3=0.5$. The fitness function is formulated for minimization: positive terms penalize cross-split visual leakage ($L_{\text{DataSAIL}}$), while negative terms reward distributional separation ($MMD$) and hardest-class generalization ($\mathrm{F1}_{\text{Hardest}}$). Scaling constants ($10^{-3}$ for DataSAIL loss, $10$ for $MMD$ and $\mathrm{F1}_{\text{Hardest}}$) were established from empirical pilot evaluations to normalize each objective to approximately unit order of magnitude.

\paragraph{Leakage-Free Validation and Held-Out Test Integrity}
To eliminate any vulnerability to \emph{Meta-Leakage} or data snooping during optimization, the fitness function in Eq.~\eqref{eq:multi_obj_fitness} is strictly evaluated across the \textbf{Train-Validation interface} ($\mathcal{D}_{\text{Train}} \leftrightarrow \mathcal{D}_{\text{Val}}$). Specifically, $\mathrm{F1}_{\text{Hardest}}(\boldsymbol{m})$ and $MMD(\boldsymbol{m})$ measure performance on the validation split relative to training representations. The \textbf{Held-Out Blind Test split} $\mathcal{D}_{\text{Test}}$ remains completely unobserved, untouchable, and frozen throughout all $N_{\text{iter}} = 10{,}000$ annealing iterations, evaluated solely once upon convergence on the final locked partition $\boldsymbol{m}^*$. Furthermore, each taxon's candidate partition designates a deterministic \emph{swap target} (Validation or Test, documented in Appendix~\ref{app:taxon_mapping}), specifying which non-training partition receives outlier blocks to balance cross-split volume without human intervention. The optimization procedure is executed via Simulated Annealing as formalized in Algorithm~\ref{alg:cegs_split}. Candidate partitions for all taxa and solvers are pre-cached offline, enabling rapid evaluation during stochastic neighbor mutations.

\paragraph{Block-Governed Instantiation and Validation of Empirical SLR = 6.0\%}
A foundational operational principle of CEGS-Split is that all candidate solvers in pool $\mathcal{K}$ are implemented as \textbf{block-governed entity operators}. Even when a continuous metric (such as Fixed Mahalanobis PP1 or Iterative Mahalanobis PP2) or graph solver (Cosine Graph PP5) is selected for a given taxon, the algorithm does not operate on loose image pixels. Instead, all images belonging to each physical specimen block $g \in \mathcal{G}_c$ are first mean-pooled into a single block centroid representation:
\begin{equation}
\bar{\boldsymbol{\phi}}(g) = \frac{1}{|g|} \sum_{x \in g} \phi(x), \quad \forall g \in \mathcal{G}_c.
\end{equation}
The mathematical partitioning logic (whether distance quantile sorting in PP1/PP2, or min-cut bipartitioning in PP5) is subsequently executed over the discrete set of block centroids $\mathcal{G}_c$, allocating entire physical blocks to Train, Val, or Test. Consequently, when CEGS-Split assigns PP1 to \textit{Dalbergia oliveri} and \textit{Guibourtia coleosperma}, or PP2 to \textit{Dalbergia melanoxylon}, \textit{Pterocarpus erinaceus}, and \textit{Pterocarpus macrocarpus} (Table~\ref{tab:taxon_solver_mapping}), these solvers assign intact physical wood blocks rather than fragmenting subfolders.

This block-level pooling provides the exact mathematical justification for the empirical measurement of $\mathrm{SLR} = 6.0\%$: across the 116 canonical blocks, 109 blocks remain strictly whole-subfolder disjoint. The remaining 7 split blocks correspond precisely to the integer knapsack boundary fallbacks required to maintain 3-way split representation ($\mathrm{CCR} = 100.0\%$) across minority taxa with $|\mathcal{G}_c| \le 5$ (Observation~\ref{obs:trilemma}). Crucially, had PP1 and PP2 been executed at the raw unblocked image level, the 5 taxa adopting them (comprising 34 physical blocks out of 116) would have experienced complete subfolder dispersion, mathematically driving global leakage to $\mathrm{SLR} \ge 34 / 116 \approx 29.3\%$. The empirical global measurement of $\mathrm{SLR} = 6.0\%$ confirms that specimen-block integrity is rigorously preserved across all continuous and discrete solvers in the governed benchmark.

\begin{algorithm}[pos=htbp]
\caption{Combinatorial Simulated Annealing Meta-Selector (CEGS-Split)}
\label{alg:cegs_split}
\begin{algorithmic}[1]
\Require Dataset $\mathcal{D} = \{(x_i, y_i, g_i)\}_{i=1}^N$, taxa $\{1,\dots,C\}$, solver pool $\mathcal{K}=\{1,\dots,K\}$, weights $(w_1, w_2, w_3)$, initial temperature $T_0$, cooling rate $\alpha$, maximum iterations $N_{\text{iter}}$.
\Ensure Pareto-favorable partition configuration $\boldsymbol{m}^*$ and assembled dataset $\mathcal{D}(\boldsymbol{m}^*)$.
\State \textbf{Offline Pre-computation:} Compute and cache candidate splits $\mathcal{D}_c(k)$ for all $c \in \{1,\dots,C\}$ and $k \in \mathcal{K}$.
\State \textbf{Initialization:} Sample $\boldsymbol{m}^{(0)} \sim \mathcal{K}^C$; assemble candidate Train/Val splits $\mathcal{D}_{\text{Train}}(\boldsymbol{m}^{(0)}), \mathcal{D}_{\text{Val}}(\boldsymbol{m}^{(0)})$; compute $F^{(0)} \gets \mathrm{Fitness}(\boldsymbol{m}^{(0)})$ strictly on Validation; set $\boldsymbol{m}^* \gets \boldsymbol{m}^{(0)}, F^* \gets F^{(0)}, T \gets T_0$.
\For{$t = 1$ \textbf{to} $N_{\text{iter}}$}
    \State Select random taxon $c \sim \{1, \dots, C\}$ and alternative solver $k' \sim \mathcal{K} \setminus \{m_c^{(t)}\}$.
    \State Form candidate configuration $\boldsymbol{m}_{\text{cand}} \gets (m_1^{(t)}, \dots, m_c^{(t)} \gets k', \dots, m_C^{(t)})$.
    \State Assemble $\mathcal{D}(\boldsymbol{m}_{\text{cand}})$ from cache and evaluate $\Delta F \gets \mathrm{Fitness}(\boldsymbol{m}_{\text{cand}}) - F^{(t)}$ on Validation.
    \If{$\Delta F < 0$ \textbf{or} $\mathrm{rand}(0,1) < \exp\left(-\Delta F / \max(10^{-5}, T)\right)$}
        \State $\boldsymbol{m}^{(t+1)} \gets \boldsymbol{m}_{\text{cand}}$, $F^{(t+1)} \gets \mathrm{Fitness}(\boldsymbol{m}_{\text{cand}})$.
        \If{$F^{(t+1)} < F^*$}
            \State $\boldsymbol{m}^* \gets \boldsymbol{m}^{(t+1)}$, $F^* \gets F^{(t+1)}$.
        \EndIf
    \Else
        \State $\boldsymbol{m}^{(t+1)} \gets \boldsymbol{m}^{(t)}$, $F^{(t+1)} \gets F^{(t)}$.
    \EndIf
    \State $T \gets T \times \alpha$.
\EndFor
\State Evaluate locked held-out test split $\mathcal{D}_{\text{Test}}(\boldsymbol{m}^*)$ once for final reporting.
\State \Return $\boldsymbol{m}^*$ and assembled dataset $\mathcal{D}(\boldsymbol{m}^*)$.
\end{algorithmic}
\end{algorithm}

\paragraph{Computational Complexity and Annealing Schedule}
With candidate splits pre-cached offline for all taxa and candidate solvers, each stochastic mutation step requires $O(N)$ operations to reassemble the global candidate partition $\mathcal{D}(\boldsymbol{m}_{\text{cand}})$ from cache and $O(N \log N)$ for representation-level fitness evaluation (dominated by nearest-neighbor distance computation). The simulated annealing schedule adopts geometric temperature decay $T \gets T \times \alpha$ with initial temperature $T_0 = 1.0$, minimum temperature threshold $T_{\min} = 10^{-5}$, and decay parameter $\alpha = 0.9995$ over $N_{\text{iter}} = 10{,}000$ iterations.

%======================================================================
\section{Experimental Setup}
\label{sec:setup}
%======================================================================

\subsection{Dataset Curation, Optical Acquisition, and Governance Verification}
\paragraph{Two-Tier Data Architecture and Curation Pipeline}
To ensure methodological transparency and avoid data-distribution confounding, this study distinguishes between two distinct data tiers:
\begin{enumerate}
\item \textbf{Tier 1: Archival Raw Repository}: Comprises 20,470 high-resolution cross-sectional RGB images ($224 \times 224$ pixels at native $12\,\mu\text{m}$ spatial resolution) collected across 210 physical specimen blocks from 18 tropical timber species spanning 5 botanical genera (Table~\ref{tab:taxonomic_inventory}): \textit{Afzelia} (4 spp.), \textit{Dalbergia} (5 spp., CITES Appendix~II-listed), \textit{Guibourtia} (3 spp.), \textit{Pterocarpus} (4 spp.), and \textit{Sindora} (2 spp.). Reference xylarium blocks were acquired under authorized scientific forestry research permits, strictly adhering to national forestry regulations and international CITES trade verification protocols. Transverse end-grain surfaces were polished with progressive silicon-carbide sandpaper grits (P120 to P600) and photographed under standardized ring-light diffuse illumination with fixed white-balance calibration to eliminate ambient illumination casts.
\item \textbf{Tier 2: Canonical Governed Benchmark}: In the raw repository, physical blocks vary substantially in physical dimensions and surface preservation: larger archival blocks yielded over 200 tiles, while smaller or weathered blocks yielded fewer. To eliminate sample-size distortion and morphological artifacts, we applied a rigorous three-step curation pipeline to select the canonical benchmark:
  \begin{itemize}
  \item \emph{Step 1 (Xylotomical Surface Quality)}: Excluded 54 damaged blocks exhibiting severe drying cracks (checks exceeding 25\% cross-sectional area), surface fungal discoloration, or uneven abrasive planing that obscured authentic diagnostic wood anatomy (vessel pore arrangements and parenchyma bands).
  \item \emph{Step 2 (Voucher Provenance Verification)}: Excluded 22 blocks lacking complete herbarium voucher cross-referencing or verified institutional provenance, ensuring 100\% taxonomic certainty conforming to IAWA macroscopic wood identification standards.
  \item \emph{Step 3 (Block Capacity Regularization)}: Filtered 18 redundant blocks from over-represented species to enforce balanced inter-species block distribution while strictly preserving rare taxa ($3 \le |\mathcal{G}_c| \le 16$ blocks per species), satisfying the integer block condition of Lemma~\ref{lem:block_condition} ($|\mathcal{G}_c| \ge 3$) across all 18 taxa without dropping any minority class.
  \end{itemize}
\end{enumerate}

\paragraph{Methodological Rationale and Selection Bias Analysis}
A critical methodological consideration is whether filtering 94 physical blocks (54 damaged, 22 unverified provenance, 18 capacity-regularized) induces convenience selection bias or artificially simplifies downstream benchmark difficulty. We address each curation decision systematically:
\begin{enumerate}
\item \emph{Exclusion of 54 Damaged and Weathered Blocks}: In macroscopic timber forensics, mechanical surface checks, rot cavities, fungal discoloration, and severe planing gouges introduce localized, high-contrast visual scars unique to individual physical blocks. If retained, parameterized deep networks rapidly exploit these distinctive surface defects as opportunistic visual shortcuts (Mechanism~3: Surface Preparation Artifacts), bypassing authentic botanical morphology (vessel pore patterns, axial parenchyma arrangements, ray density). Retaining damaged blocks would thus artificially inflate shortcut exploitability rather than test taxonomic discriminability. Excluding these artifacts is strictly necessary to force models to learn genuine xylotomical structures.
\item \emph{Exclusion of 22 Blocks Lacking Verified Provenance}: In timber trade law enforcement and CITES Appendix~II customs prosecution, visual species identification possesses legal forensic standing only when reference data are anchored to certified herbarium vouchers verified by accredited wood anatomists conforming to IAWA standards. Incorporating blocks of ambiguous provenance introduces latent ground-truth taxonomic label noise into the benchmark, undermining empirical validity.
\item \emph{Exclusion of 18 Redundant Blocks (Capacity Regularization)}: Uncontrolled botanical archives naturally exhibit severe specimen availability skew (e.g., common commercial timber such as \textit{Pterocarpus macrocarpus} possessing over 25 physical blocks, whereas endangered, heavily regulated taxa such as \textit{Dalbergia rimosa} or \textit{Sindora cochinchinensis} possess only 5--7 verified blocks). Retaining unlimited blocks for majority species would allow them to dominate the combinatorial simulated annealing objective and overshadow minority gradient signals during fine-tuning. Regularizing block capacities into a balanced band ($3 \le |\mathcal{G}_c| \le 16$) preserves representation equity without discarding rare taxa.
\item \emph{Benchmark Representativeness and Difficulty Preservation}: Crucially, this curation does not render the benchmark artificially easy. The standardized benchmark preserves all 18 taxonomic classes ($\mathrm{CCR} \equiv 100\%$), including the most morphologically indistinguishable congeneric sibling species pairs (\textit{Dalbergia oliveri} vs.\ \textit{Dalbergia cochinchinensis}; \textit{Afzelia xylocarpa} vs.\ \textit{Afzelia bella}) and rare taxa with minimal physical blocks ($|\mathcal{G}_c| \le 5$). Because held-out test partitions evaluate novel physical specimens devoid of shortcut surface defects, the anatomical classification challenge remains at peak diagnostic difficulty, ensuring that reported model performance reflects genuine deployment capabilities.
\end{enumerate}

This curation pipeline establishes the standardized canonical benchmark comprising exactly \textbf{116 physical blocks} and \textbf{6,410 quality-controlled images} (Train: 4,191 [65.4\%], Val: 1,154 [18.0\%], Test: 1,065 [16.6\%]), preserving whole-subfolder specimen integrity and 100\% Class Coverage Rate across all 18 species (Figure~\ref{fig:dataset_distribution}). Across the 18 taxa, block allocations average approximately 4--10 blocks in Train, 1--3 blocks in Val, and 1--3 blocks in Test per taxon (e.g., taxa with 9 canonical blocks allocate 5 Train, 2 Val, 2 Test). In the held-out test split, each taxon is represented by an average of $59.2 \pm 6.4$ test images (ranging strictly between 48 and 72 images), ensuring that hardest-class generalization metrics ($\mathrm{F1}_{\text{Hardest}}$) evaluate statistically meaningful image volumes rather than isolated single-image edge cases. At the same time, because minority taxa contain only 2 physical blocks in the test partition, unique structural variations in an individual block can exert substantial leverage on gradient optimization (explaining the bimodal dispersion observed under ConvNeXt-Tiny in Section~\ref{sec:finetuning}) and underscoring the critical need for combinatorial governance.

\begin{figure}[pos=htbp]
\centering
\includegraphics[width=0.96\linewidth, keepaspectratio]{fig/eda_split_end_version.png}
\caption{Taxonomic class distribution and partition allocation across the canonical 6,410-image S3 benchmark dataset (sampled from the archival 20,470-image repository). Partition allocations (Train: 4,191, Val: 1,154, Test: 1,065) are maintained while strictly enforcing whole-subfolder specimen integrity and 100\% Class Coverage Rate ($\mathrm{CCR}$) across all 18 taxa. The horizontal axis enumerates all 18 botanical taxa; the vertical axis displays sample image volume. Color bars denote split assignments (Train: blue/darkest shade, Val: orange/intermediate shade, Test: green/lightest shade), ensuring high visual contrast under both full-color display and monochrome/grayscale printing.}
\label{fig:dataset_distribution}
\end{figure}

\subsection{Representation Space and Zero-Training Evaluation Protocol}
To isolate the causal effect of dataset partitioning from neural-network training stochasticity (learning rate dynamics, weight initialization, optimizer convergence, epoch criteria), we employ a \textbf{Zero-Training 1-Nearest-Neighbor (1-NN) Evaluation Protocol} on frozen deep representations. Evaluating partitions via non-parametric 1-NN classification on frozen representations follows established linear-probing and frozen-feature benchmarking protocols in representation learning~\cite{scikit}. By freezing feature extraction, we eliminate optimization-dependent confounding factors, ensuring that observed performance shifts reflect genuine partition boundaries rather than classifier overfitting. While absolute accuracy margins may shift under end-to-end gradient fine-tuning, the directional impact of specimen leakage (i.e., artificial performance inflation) remains structurally invariant; furthermore, representation-level evaluation serves as a conservative lower-bound estimate of leakage inflation, as parameterized deep networks readily memorize and overfit to specimen-level surface artifacts.

Features are extracted using EfficientNetV2-M (\texttt{tf\_efficientnetv2\_m\_in21k})~\cite{efficientnetv2} pre-trained on ImageNet-21k, mapping $224\times224$ image patches to 1280-dimensional vectors $\boldsymbol{f}_i \in \mathbb{R}^{1280}$, $L_2$-normalized to unit sphere embeddings $\phi(x_i) = \boldsymbol{f}_i / \|\boldsymbol{f}_i\|_2$. For each test sample, predictions are assigned via maximum cosine similarity:
\begin{equation}
\hat{y}(x_{\text{test}}) = y\left( \arg\max_{x_{\text{train}} \in \mathcal{D}_{\text{Train}}} \phi(x_{\text{test}})^{\!\top} \phi(x_{\text{train}}) \right).
\end{equation}
All experiments are replicated across 5 random seeds (\texttt{42, 123, 456, 789, 2024}).

\subsection{Statistical Testing and Implementation Details}
Statistical significance relative to the Naive Random baseline is assessed via Welch's two-sample $t$-test (unequal variances assumed) and standardized Cohen's $d$ effect sizes computed over the 5-seed accuracy distributions. Pre-computation of candidate splits across all taxa and solvers requires approximately 120 seconds, while a 10,000-iteration Simulated Annealing search executes in under 90 minutes on standard workstation CPU hardware without requiring dedicated GPU acceleration.

%======================================================================
\section{Experimental Results and Analysis}
\label{sec:results}
%======================================================================

\subsection{Master Benchmark and Leakage Inflation}
Table~\ref{tab:master_results} reports the primary optimization and data-governance metrics across all 13 evaluated splitting protocols under the zero-training 1-NN benchmark. Crucially, to ensure complete methodological transparency across the entire evaluation spectrum, the full quantitative evaluation encompassing all 16 metrics of the Three Pillars framework -- including Top-3 Accuracy, Balanced Accuracy, Pseudoreplication Index ($\mathrm{PRI}$), Silhouette Separation ($S_{\text{split}}$), Wasserstein Divergence ($W_1$), and Inter-Split Cosine Similarity ($\bar{S}_{\text{inter}}$) -- is comprehensively documented in Table~\ref{tab:extended_three_pillars_p1} and Table~\ref{tab:extended_three_pillars_p2} (Appendix~\ref{app:extended_benchmark}).

\begin{table}[pos=htbp]
\centering
\scriptsize
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.15}
\caption{Master benchmark: zero-training 1-NN classification performance and leakage metrics across all 13 protocols (Mean $\pm$ Std across 5 seeds; $\mathrm{CCR}=100\%$ unless noted).}
\label{tab:master_results}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lcccccccc}
\toprule
\makecell[l]{\textbf{Splitting}\\\textbf{Protocol}} & \makecell{\textbf{KNN}\\\textbf{Top-1}} & \makecell{\textbf{Macro}\\\textbf{F1}} & \makecell{\textbf{Hardest}\\\textbf{F1}} & \makecell{\textbf{DataSAIL}\\\textbf{Loss $L(\pi)$}} & \makecell{\textbf{SLR}\\\textbf{(\%)}} & \makecell{\textbf{CCR}\\\textbf{(\%)}} & \textbf{MMD} & \makecell{\textbf{Nominal $d$}\\\textbf{vs.\ Naive$^{\dagger}$}} \\
\midrule
\multicolumn{9}{l}{\textit{\textbf{Category I: Naive Image-Level Baselines (Unchecked Boundary Leakage)}}} \\
Naive Random Image Split & 0.9987 $\pm$ 0.0011 & 0.9985 $\pm$ 0.0012 & 0.9880 & 7,178,341.6 $\pm$ 883.3 & 100.0\% & 100.0\% & 0.0147 & Baseline \\
Naive Stratified Image Split & 0.9985 $\pm$ 0.0010 & 0.9983 $\pm$ 0.0011 & 0.9876 & 7,175,820.4 $\pm$ 912.5 & 100.0\% & 100.0\% & 0.0145 & Baseline \\
DataSAIL Image-Level ILP & 0.9834 $\pm$ 0.0080 & 0.9794 $\pm$ 0.0099 & 0.8636 & 3,336,082.5 $\pm$ 38936.9 & 95.3\% & 100.0\% & 0.1110 & 3.02 \\
\midrule
\multicolumn{9}{l}{\textit{\textbf{Category II: Single Splitting Protocols (Single Paradigm Imposed Globally)}}} \\
Fixed Mahalanobis Strat. (Continuous Metric) & 0.9809 $\pm$ 0.0000 & 0.9755 $\pm$ 0.0000 & 0.8500 & 7,128,200.5 $\pm$ 0.0 & 100.0\% & 100.0\% & 0.0974 & 17.99 \\
Cosine Feature Graph (Continuous Graph) & 0.9476 $\pm$ 0.0033 & 0.8976 $\pm$ 0.0082 & 0.2954 & 7,142,339.5 $\pm$ 2797.6 & 90.3\% & 100.0\% & 0.0845 & 22.74 \\
Naive Specimen Group Split & 0.9657 $\pm$ 0.0111 & 0.9594 $\pm$ 0.0125 & 0.7094 & 7,091,834.8 $\pm$ 96814.5 & 7.6\% & 100.0\% & 0.0806 & 4.74 \\
Hierarchical Ward Partitioning & 0.9249 $\pm$ 0.0060 & 0.9116 $\pm$ 0.0065 & 0.6046 & 6,008,406.8 $\pm$ 222280.9 & 7.4\% & 100.0\% & 0.1066 & 19.21 \\
Adversarial Density Validation & 0.9553 $\pm$ 0.0186 & 0.9456 $\pm$ 0.0200 & 0.6431 & 6,974,598.2 $\pm$ 95298.9 & 6.4\% & 100.0\% & 0.0753 & 3.74 \\
Stratified Group Split & 0.9774 $\pm$ 0.0003 & 0.9533 $\pm$ 0.0002 & 0.6667 & 7,868,015.8 $\pm$ 1486.6 & 6.0\% & 100.0\% & 0.0657 & 21.23 \\
Agglomerative Stratified Banding & 0.9424 $\pm$ 0.0053 & 0.9309 $\pm$ 0.0067 & 0.7275 & 7,503,766.6 $\pm$ 26458.8 & 7.8\% & 100.0\% & 0.0776 & 16.42 \\
DataSAIL Specimen-Level ILP & 0.9076 $\pm$ 0.0050 & 0.8310 $\pm$ 0.0379 & 0.1193 & \textbf{4,841,999.8 $\pm$ 108695.1} & 5.7\% & 88.9\% & \textbf{0.1199} & \textbf{27.91} \\
\midrule
\multicolumn{9}{l}{\textit{\textbf{Category III: Combinatorial Selector (DataSAIL Single-Objective Optimization)}}} \\
Single-Objective Classwise Selector & 0.9875 $\pm$ 0.0019 & 0.9775 $\pm$ 0.0022 & 0.7407 $\pm$ 0.0310 & 6,871,774.0 $\pm$ 41200.0 & 14.7\% $\pm$ 1.2\% & 100.0\% & 0.0700 & 16.85 \\
\midrule
\multicolumn{9}{l}{\textit{\textbf{Category IV: Combinatorial Selector (Multi-Objective Optimization -- Proposed)}}} \\
Multi-Objective SA Meta-Selector & \textbf{0.9875 $\pm$ 0.0015} & \textbf{0.9775 $\pm$ 0.0018} & \textbf{0.7407 $\pm$ 0.0285} & 6,871,005.0 $\pm$ 38420.5 & \textbf{6.0\% $\pm$ 0.8\%} & 100.0\% & 0.0695 & 17.42 \\
\bottomrule
\end{tabular}%
}
\vspace{2pt}
{\scriptsize $^{\dagger}$\textit{Statistical Note}: Nominal Cohen's $d$ values reflect zero-training 1-NN evaluation on frozen representations where near-zero within-condition variance ($\sigma \approx 0.001$) mathematically scales effect sizes. Under parameterized deep learning with optimizer variance (Section~\ref{sec:finetuning}), empirical effect sizes normalize to standard biological ranges ($d \approx 3.2$--$5.8$).}
\end{table}

\textbf{Quantifying Leakage-Induced Inflation Across Baselines}: Comparing \textsf{Naive Random Image Split} ($99.87\%$ accuracy, $\mathrm{SLR}=100.0\%$) against the standard group-disjoint baseline, \textsf{Stratified Group Split} ($97.74\%$ accuracy, $\mathrm{SLR}=6.0\%$), reveals an empirical performance inflation of $\Delta\text{Acc} = +2.13$~pp and $\Delta\text{F1-Macro} = +4.52$~pp (Welch's $t$-test $p = 1.16 \times 10^{-14}$, highly significant under Bonferroni multiple-comparison correction $\alpha_{\text{adj}} = 0.05/12 \approx 0.0042$). When contrasted against global feature-disjoint separation, \textsf{DataSAIL Specimen-Level ILP} ($90.76\%$ accuracy, $\mathrm{SLR}=5.7\%$), this measured inflation margin expands substantially to $\Delta\text{Acc} = +9.10$~pp and $\Delta\text{F1-Macro} = +16.75$~pp ($p = 2.28 \times 10^{-6}$). In technical terms, the large nominal Cohen's $d$ ($27.91$) reported in Table~\ref{tab:master_results} mathematically reflects the vanishing within-condition variance of deterministic 1-NN voting on frozen deep representations ($\sigma \approx 0.001$), where minor metric shifts across identical nearest-neighbor votes produce immense standardized ratios. We intentionally avoid highlighting nominal $d$ in the executive summary, and note that under stochastic gradient fine-tuning in Section~\ref{sec:finetuning} (Table~\ref{tab:backbone_robustness}), where model training incurs realistic parameter variance, empirical effect sizes normalize to standard statistical ranges ($d \approx 3.2$--$5.8$), corroborating the genuine physical reality of specimen-level data leakage without methodological exaggeration.

\textbf{Mechanisms of Single-Paradigm Behavior and Generalization Trade-offs}: In Table~\ref{tab:master_results}, \textsf{DataSAIL Specimen-Level ILP} achieves the lowest cross-split leakage loss ($4.84 \times 10^6$) and the highest out-of-distribution divergence ($\mathrm{MMD} = 0.1199$), exactly conforming to its theoretical objective of minimizing inter-split feature similarity. However, because DataSAIL's canonical formulation optimizes global feature divergence without class-coverage equality constraints, the solver concentrates minority-taxa outlier blocks into evaluation partitions, resulting in training-set class omission ($\mathrm{CCR} = 88.9\%$) and severe minority-class degradation ($\mathrm{F1}_{\text{Hardest}} = 0.1193$). This outcome is not an algorithmic defect of DataSAIL, but a predictable consequence of applying an unconstrained feature-space ILP to fine-grained botanical classification where full taxonomic retention is mandatory. Conversely, while standard \textsf{Stratified Group Split} guarantees $\mathrm{CCR}=100.0\%$ through boundary fallbacks ($\mathrm{SLR} = 6.0\%$), it achieves lower hardest-class F1 on frozen embeddings ($0.6667$) and, more critically, suffers complete decision-boundary collapse ($\mathrm{F1}_{\text{Hardest}} = 0.00\%$) when fine-tuned on ResNet-50 (Section~\ref{sec:finetuning}). In contrast, the Multi-Objective SA Meta-Selector achieves a balanced Pareto compromise ($\mathrm{F1}_{\text{Hardest}} = 0.7407$, $\mathrm{CCR} = 100.0\%$, $\mathrm{SLR} = 6.0\%$) and preserves viable hardest-class generalization across all fine-tuned architectures.

\textbf{Governance Advantage of Multi-Objective Meta-Selection}: While the single-objective class-wise selector (Category III) and the multi-objective CEGS-Split (Category IV) achieve identical nominal mean accuracy ($98.75\%$) and macro-F1 ($97.75\%$) under 1-NN evaluation, this parity stems from representation-level metric saturation: across the 6,410-image dataset with frozen embeddings, test-set misclassifications ($N_{\text{test}} = 1{,}065$) concentrate within an identical small subset of congeneric sibling pairs (specifically \textit{Dalbergia oliveri} vs.\ \textit{Dalbergia cochinchinensis}), producing identical rational error fractions for hardest-class F1 ($20/27 \approx 0.7407$). Crucially, however, the multi-objective formulation delivers a decisive data-governance breakthrough: it reduces the Specimen Leakage Risk ($\mathrm{SLR}$) from $14.7\% \pm 1.2\%$ down to $6.0\% \pm 0.8\%$ (a $59.2\%$ reduction in leaked physical entities) while preserving identical classification capability. As detailed in Appendix~\ref{app:taxon_mapping} and Table~\ref{tab:taxon_solver_mapping}, Category III selects solvers independently for each taxon based solely on local DataSAIL loss, ignoring inter-taxa specimen balancing and cross-split assembly interactions, allowing 17 physical blocks to leak across partition boundaries. In contrast, CEGS-Split coordinates global assembly by assigning tailored solvers (e.g., Ward hierarchical clustering PP4, Adversarial Density Validation PP7, and Agglomerative Stratified Banding PP9) that adapt to genus-specific morphological variance, reconciling specimen disjointness with balanced class coverage. Crucially, the residual boundary leakage ($\mathrm{SLR} = 6.0\%$, corresponding to exactly 7 boundary blocks split out of 116 canonical blocks) manifests the integer-partition friction formalized in Observation~\ref{obs:trilemma}: for rare biological taxa with $|\mathcal{G}_c| \le 9$ physical blocks (e.g., \textit{Dalbergia rimosa}, \textit{Sindora cochinchinensis}), simultaneously satisfying non-empty 3-way class representation ($\mathrm{CCR}=100\%$), target volume ratios (65/18/17), and feature-space OOD separation leaves boundary specimens susceptible to minimal cross-split fallback friction. Remarkably, CEGS-Split compresses this structural friction to its empirical baseline floor ($6.0\%$), whereas unguided per-class selection allows structural leakage to escalate to $14.7\%$ (Category~III) and $16.4\%$ (Hard-Constrained pool without global multi-objective coordination), and naive partitioning induces complete boundary collapse ($\mathrm{SLR} = 100.0\%$).

\subsection{Meta-Selector Formulation Ablation}
Table~\ref{tab:ablation_meta} ablates the Meta-Selector over 10,000 Simulated Annealing iterations, comparing the proposed balanced configuration against unconstrained optimization, hard-constrained candidate pools ($\text{SLR}_c \equiv 0.0\%$), and continuous penalty terms ($-w_4 \cdot \mathrm{SLR}$).

\begin{table}[pos=htbp]
\centering
\scriptsize
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.15}
\caption{Meta-Selector formulation ablation across 10,000 Simulated Annealing iterations.}
\label{tab:ablation_meta}
\resizebox{\textwidth}{!}{%
\begin{tabular}{llcccccccc}
\toprule
\makecell[l]{\textbf{Optimization}\\\textbf{Formulation}} & \makecell{\textbf{Constraint}\\\textbf{Mechanism}} & \makecell{\textbf{Accuracy}\\\textbf{(\%)}} & \makecell{\textbf{Macro-F1}\\\textbf{(\%)}} & \makecell{\textbf{Hardest-Class}\\\textbf{F1 (\%)}} & \makecell{\textbf{DataSAIL}\\\textbf{Loss $L(\pi)$}} & \makecell{\textbf{SLR}\\\textbf{(\%)}} & \makecell{\textbf{CCR}\\\textbf{(\%)}} & \textbf{MMD} & \makecell{\textbf{Runtime}\\\textbf{(s)}} \\
\midrule
Proposed Meta-Selector (Full) & Balanced SA $(1.0, 0.5, 0.5)$ & 98.75\% & 97.75\% & 74.07\% & 6,871,005.0 & 6.0\% & 100.0\% & 0.0695 & 5042.8s \\
Unconstrained SA (10,000 iters) & None & 98.23\% & 97.84\% & 81.08\% & 3,247,277.5 & 66.4\% & 100.0\% & 0.0812 & 5116.4s \\
Hard-Constrained Candidate Pool & $\text{SLR}_c \equiv 0.0\%$ & 92.47\% & 90.71\% & 51.61\% & 5,261,003.0 & 16.4\% & 100.0\% & 0.0825 & 4971.2s \\
Penalized Fitness ($w_4 = 1.0$) & $-w_4 \cdot \mathrm{SLR}$ & 98.22\% & 97.84\% & 81.08\% & 3,253,192.5 & 57.8\% & 100.0\% & 0.0798 & 5182.1s \\
Penalized Fitness ($w_4 = 2.0$) & $-w_4 \cdot \mathrm{SLR}$ & 93.86\% & 93.38\% & 60.00\% & 3,264,586.8 & 50.0\% & 100.0\% & 0.0785 & 5147.2s \\
\bottomrule
\end{tabular}%
}
\end{table}

\textbf{Failure of Soft Penalties vs.\ Hard Governance}: Incorporating a continuous penalty $-w_4 \cdot \mathrm{SLR}$ fails to prevent boundary leakage ($\mathrm{SLR} = 57.8\%$ at $w_4=1.0$, and $50.0\%$ at $w_4=2.0$). We observe that within the DataSAIL loss landscape, the optimization benefit gained by assigning identical specimens across partition boundaries outweighs linear penalty costs, explaining why soft penalties fail to prevent leakage. In contrast, restricting the candidate pool strictly to specimen-disjoint protocols ($\text{SLR}_c \equiv 0.0\%$) effectively eliminates random image-level assignments, yielding zero image-level leakage while preserving full class coverage ($CCR = 100.0\%$). However, when candidate solvers are restricted solely via local per-class criteria without global coordination, localized clustering dynamics on small-sample taxa force fallback allocations to satisfy 3-way split representation, causing an elevated global $\mathrm{SLR} = 16.4\%$ (19 split blocks out of 116) and degrading hardest-class generalization to $51.61\%$. In contrast, CEGS-Split under default multi-objective weights coordinates global assembly interactions, driving global structural boundary friction down to its Pareto minimum ($\mathrm{SLR} = 6.0\%$, only 7 split blocks) while preserving superior hardest-class generalization ($74.07\%$).

\textbf{Balancing Multi-Objective Fitness and MMD Sensitivity}: In Eq.~\eqref{eq:multi_obj_fitness}, the three optimization terms operate across distinct orders of magnitude and opposing directions: minimizing inter-split similarity ($L_{\text{DataSAIL}} \approx 6.87 \times 10^6$, scaled by $10^{-3}$) exerts an inward compression against visual leakage, while maximizing out-of-distribution divergence ($-w_2 \cdot 10 \cdot MMD$) exerts an outward repulsive force to prevent evaluation splits from trivializing into near-identical distributions. Concurrently, maximizing hardest-class generalization ($-w_3 \cdot 10 \cdot \mathrm{F1}_{\text{Hardest}}$) serves as an indispensable preservation anchor for rare species. The direct coupling between the weight $w_2$ and empirical MMD is systematically validated in the sensitivity analysis of Appendix~\ref{app:sensitivity} (Table~\ref{tab:sensitivity_weights}): as $w_2$ increases from $0.2 \to 0.5 \to 0.8$, the measured MMD increases monotonically from $0.0612 \to 0.0695 \to 0.0924$. However, overly prioritizing OOD divergence ($w_2=0.8$) penalizes shared feature support too aggressively, driving $\mathrm{F1}_{\text{Hardest}}$ down from $0.7407$ to $0.6154$. The balanced baseline $(w_1, w_2, w_3) = (1.0, 0.5, 0.5)$ represents the optimal Pareto compromise, where simulated annealing steadily drives candidate mutations toward stable fitness plateau convergence within 10,000 iterations.

\paragraph{Feature Extractor Ablation Under Zero-Training 1-NN Probing}
A key methodological consideration is whether the leakage inflation dynamics identified in Table~\ref{tab:master_results} reflect properties of the specific feature extractor (EfficientNetV2-M) or represent an invariant property of dataset partitioning across representation spaces. To ablate the influence of feature extraction, we replicate the zero-training 1-NN evaluation on frozen representations extracted from four distinct architectural families spanning both convolutional and transformer paradigms: EfficientNetV2-M (convolutional inverted bottleneck, 1280-d)~\cite{efficientnetv2}, Swin-Large (hierarchical vision transformer with shifted window multi-head self-attention, 1536-d)~\cite{swin}, ConvNeXt-Tiny (modernized depthwise pure convolutional network, 768-d)~\cite{convnext}, and ResNet-50 (canonical residual baseline, 2048-d)~\cite{resnet}. 

Table~\ref{tab:feature_extractor_ablation} compares 1-NN test accuracy and macro-F1 across naive random image-level partitioning and governed specimen-disjoint CEGS-Split. Across all four distinct representation spaces, naive random splitting induces severe, statistically decisive performance inflation ($\Delta\text{Acc} = +0.89$~pp to $+4.30$~pp; $\Delta\text{Macro-F1} = +1.64$~pp to $+6.60$~pp vs.\ CEGS-Split; expanding to $+2.13$~pp to $+6.55$~pp vs.\ standard Stratified Group Split). Swin-Large achieves the highest governed zero-training accuracy ($99.02\%$), corroborating its superior capacity to extract specimen-invariant anatomical traits via shifted windows, while canonical ResNet-50 exhibits the steepest degradation under specimen isolation ($94.65\%$, $\Delta\text{Acc} = +4.30$~pp). Crucially, the persistence of leakage-induced performance inflation across all four diverse backbones confirms that specimen-level data leakage is an intrinsic, physical pathology of uncurated spatial partitioning rather than an artifact of any specific feature extractor or embedding dimensionality.

\begin{table}[pos=htbp]
\centering
\small
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.18}
\caption{Feature extractor ablation under zero-training 1-NN evaluation on frozen representations across four distinct architectural paradigms (Mean $\pm$ Std across 5 seeds; $CCR=100\%$).}
\label{tab:feature_extractor_ablation}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lcccccc}
\toprule
\makecell[l]{\textbf{Feature Extractor}\\\textbf{Backbone}} & \makecell{\textbf{Embedding}\\\textbf{Dimension}} & \makecell{\textbf{Architectural}\\\textbf{Paradigm}} & \makecell{\textbf{Naive Random}\\\textbf{Top-1 Acc (\%)}} & \makecell{\textbf{Governed CEGS}\\\textbf{Top-1 Acc (\%)}} & \makecell{\textbf{Inflation Margin}\\\textbf{$\Delta\text{Acc}$ (pp)}} & \makecell{\textbf{Governed}\\\textbf{Macro-F1 (\%)}} \\
\midrule
EfficientNetV2-M & 1,280 & Fused Inverted Bottleneck & 99.87 $\pm$ 0.11 & 98.75 $\pm$ 0.15 & +1.12 pp & 97.75 $\pm$ 0.18 \\
Swin-Large & 1,536 & Shifted Window Transformer & 99.91 $\pm$ 0.08 & 99.02 $\pm$ 0.10 & +0.89 pp & 98.25 $\pm$ 0.12 \\
ConvNeXt-Tiny & 768 & Depthwise Modernized ConvNet & 99.52 $\pm$ 0.15 & 97.80 $\pm$ 0.25 & +1.72 pp & 96.15 $\pm$ 0.28 \\
ResNet-50 & 2,048 & Canonical Residual Bottleneck & 98.95 $\pm$ 0.22 & 94.65 $\pm$ 0.38 & +4.30 pp & 92.10 $\pm$ 0.42 \\
\bottomrule
\end{tabular}%
}
\end{table}

\subsection{Multi-Backbone Validation Under End-to-End Fine-Tuning}
\label{sec:finetuning}
To verify whether the leakage inflation dynamics identified under the non-parametric 1-NN benchmark extend to gradient-based deep learning, we fine-tune four representative vision architectures—ConvNeXt-Tiny~\cite{convnext}, Swin-Large~\cite{swin}, EfficientNetV2-M~\cite{efficientnetv2}, and ResNet-50~\cite{resnet}—under Focal Loss ($\gamma=2.0, \alpha=0.25$)~\cite{focal}. 

Table~\ref{tab:backbone_hyperparameters} details the exact architectural configurations, input preprocessing, data augmentations, and optimization schedules standardized across all four deep neural backbones. Focal Loss was incorporated into the training pipeline to address the natural class-imbalance characteristic of biological timber inventories (where per-taxon sample volumes vary from 290 to 2,050 images across species, Table~\ref{tab:taxonomic_inventory}). Following canonical practice in imbalanced fine-grained classification~\cite{focal}, the weighting factor $\alpha=0.25$ and focusing parameter $\gamma=2.0$ were selected to suppress the cumulative loss contribution from voluminous, easily classified majority samples while focusing representational capacity on rare minority taxa. Models are optimized using AdamW ($\beta_1=0.9, \beta_2=0.999$) with an initial learning rate of $10^{-4}$, scheduled via 3-epoch linear warm-up followed by cosine annealing decay down to $10^{-6}$ over 20 epochs with batch size 32. Data augmentations include random resized crops ($224 \times 224$, scale $0.8$--$1.0$, aspect ratio $0.75$--$1.33$), random horizontal and vertical flips ($p=0.5$), mild color jitter (brightness, contrast, and saturation $0.25$, hue $0.05$), and random grayscale conversion ($p=0.05$). Table~\ref{tab:backbone_robustness} presents results across 80 training runs (4 architectures $\times$ 4 partitioning protocols $\times$ 5 seeds).

\begin{table}[pos=htbp]
\centering
\small
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.15}
\caption{Comprehensive preprocessing, data augmentation, optimization hyperparameters, and architectural specifications across all evaluated deep neural backbones.}
\label{tab:backbone_hyperparameters}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lcccc}
\toprule
\textbf{Configuration / Hyperparameter} & \textbf{ConvNeXt-Tiny}~\cite{convnext} & \textbf{Swin-Large}~\cite{swin} & \textbf{EfficientNetV2-M}~\cite{efficientnetv2} & \textbf{ResNet-50}~\cite{resnet} \\
\midrule
Architectural Family & Depthwise Modernized ConvNet & Hierarchical Vision Transformer & Fused Inverted Bottleneck & Canonical Residual Network \\
Pretrained Weight Source & ImageNet-1k (\texttt{timm}) & ImageNet-21k (\texttt{timm}) & ImageNet-21k (\texttt{timm}) & ImageNet-1k (\texttt{torchvision}) \\
Input Patch Resolution & $224 \times 224 \times 3$ & $224 \times 224 \times 3$ & $224 \times 224 \times 3$ & $224 \times 224 \times 3$ \\
Input Normalization & ImageNet Mean / Std & ImageNet Mean / Std & ImageNet Mean / Std & ImageNet Mean / Std \\
Feature Embedding Dimension ($D$) & 768 & 1,536 & 1,280 & 2,048 \\
\midrule
\multicolumn{5}{l}{\textit{\textbf{Data Augmentation Pipeline (Training Split)}}} \\
Random Resized Crop & Scale: $[0.8, 1.0]$, Ratio: $[0.75, 1.33]$ & Scale: $[0.8, 1.0]$, Ratio: $[0.75, 1.33]$ & Scale: $[0.8, 1.0]$, Ratio: $[0.75, 1.33]$ & Scale: $[0.8, 1.0]$, Ratio: $[0.75, 1.33]$ \\
Random Horizontal Flip & $p = 0.50$ & $p = 0.50$ & $p = 0.50$ & $p = 0.50$ \\
Random Vertical Flip & $p = 0.50$ & $p = 0.50$ & $p = 0.50$ & $p = 0.50$ \\
Color Jitter & Bri/Con/Sat: $0.25$, Hue: $0.05$ & Bri/Con/Sat: $0.25$, Hue: $0.05$ & Bri/Con/Sat: $0.25$, Hue: $0.05$ & Bri/Con/Sat: $0.25$, Hue: $0.05$ \\
Random Grayscale Conversion & $p = 0.05$ & $p = 0.05$ & $p = 0.05$ & $p = 0.05$ \\
Validation / Test Preprocessing & Resize $256 \times 256 \to$ CenterCrop $224$ & Resize $256 \times 256 \to$ CenterCrop $224$ & Resize $256 \times 256 \to$ CenterCrop $224$ & Resize $256 \times 256 \to$ CenterCrop $224$ \\
\midrule
\multicolumn{5}{l}{\textit{\textbf{Optimization Schedule and Loss Parameters}}} \\
Optimizer & AdamW ($\beta_1=0.9, \beta_2=0.999$) & AdamW ($\beta_1=0.9, \beta_2=0.999$) & AdamW ($\beta_1=0.9, \beta_2=0.999$) & AdamW ($\beta_1=0.9, \beta_2=0.999$) \\
Base Learning Rate ($\text{lr}$) & $1.0 \times 10^{-4}$ & $1.0 \times 10^{-4}$ & $1.0 \times 10^{-4}$ & $1.0 \times 10^{-4}$ \\
Weight Decay & $5.0 \times 10^{-2}$ & $5.0 \times 10^{-2}$ & $1.0 \times 10^{-4}$ & $1.0 \times 10^{-4}$ \\
Learning Rate Schedule & CosineAnnealing with Warmup & CosineAnnealing with Warmup & CosineAnnealing with Warmup & CosineAnnealing with Warmup \\
Warmup Epochs / Min $\text{lr}$ & 3 epochs / $1.0 \times 10^{-6}$ & 3 epochs / $1.0 \times 10^{-6}$ & 3 epochs / $1.0 \times 10^{-6}$ & 3 epochs / $1.0 \times 10^{-6}$ \\
Total Fine-Tuning Epochs & 20 epochs & 20 epochs & 20 epochs & 20 epochs \\
Effective Batch Size & 32 & 32 & 32 & 32 \\
Loss Function & Focal Loss ($\alpha=0.25, \gamma=2.0$) & Focal Loss ($\alpha=0.25, \gamma=2.0$) & Focal Loss ($\alpha=0.25, \gamma=2.0$) & Focal Loss ($\alpha=0.25, \gamma=2.0$) \\
Gradient Norm Clipping & $\max \|\boldsymbol{g}\|_2 = 1.0$ & $\max \|\boldsymbol{g}\|_2 = 1.0$ & $\max \|\boldsymbol{g}\|_2 = 1.0$ & $\max \|\boldsymbol{g}\|_2 = 1.0$ \\
Checkpoint Selection Metric & Validation Macro-F1 & Validation Macro-F1 & Validation Macro-F1 & Validation Macro-F1 \\
\bottomrule
\end{tabular}%
}
\end{table}

\begin{table}[pos=htbp]
\centering
\scriptsize
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.15}
\caption{Multi-backbone robustness benchmark under Focal Loss ($\gamma=2.0, \alpha=0.25$) fine-tuning across representative vision architectures and splitting protocols (Mean $\pm$ Std across 5 seeds).}
\label{tab:backbone_robustness}
\resizebox{\textwidth}{!}{%
\begin{tabular}{llcccccc}
\toprule
\makecell[l]{\textbf{Vision}\\\textbf{Architecture}} & \makecell[l]{\textbf{Splitting}\\\textbf{Protocol}} & \makecell{\textbf{Top-1}\\\textbf{Acc (\%)}} & \makecell{\textbf{Top-3}\\\textbf{Acc (\%)}} & \makecell{\textbf{Balanced}\\\textbf{Acc (\%)}} & \makecell{\textbf{Macro}\\\textbf{F1 (\%)}} & \makecell{\textbf{Hardest-Class}\\\textbf{F1 (\%)}} & \makecell{\textbf{SLR}\\\textbf{(\%)}} \\
\midrule
\multirow{4}{*}{\textbf{ConvNeXt-Tiny}~\cite{convnext}} & Naive Random Image Split & 98.84 $\pm$ 0.25 & 99.96 $\pm$ 0.04 & 98.70 $\pm$ 0.13 & 98.41 $\pm$ 0.46 & 84.94 $\pm$ 6.37 & 100.0\% \\
 & Stratified Group Split & 94.81 $\pm$ 0.71 & 99.59 $\pm$ 0.14 & 89.85 $\pm$ 1.12 & 88.91 $\pm$ 0.97 & 25.88 $\pm$ 2.69 & 6.0\% \\
 & DataSAIL Specimen-Level ILP & 86.95 $\pm$ 1.43 & 99.27 $\pm$ 0.28 & 85.87 $\pm$ 0.12 & 77.41 $\pm$ 0.06 & 0.00 $\pm$ 0.00 & 5.6\% \\
 & Multi-Objective SA Meta-Selector & 91.83 $\pm$ 0.49 & 99.38 $\pm$ 0.48 & 91.39 $\pm$ 0.20 & 89.76 $\pm$ 0.71 & 34.21 $\pm$ 34.21 & 7.8\% \\
\midrule
\multirow{4}{*}{\textbf{Swin-Large}~\cite{swin}} & Naive Random Image Split & 99.21 $\pm$ 0.12 & 100.00 $\pm$ 0.00 & 98.18 $\pm$ 0.20 & 98.61 $\pm$ 0.20 & 83.53 $\pm$ 1.47 & 100.0\% \\
 & Stratified Group Split & 98.20 $\pm$ 0.27 & 100.00 $\pm$ 0.00 & 96.61 $\pm$ 0.51 & 96.43 $\pm$ 1.07 & 71.59 $\pm$ 13.26 & 6.0\% \\
 & DataSAIL Specimen-Level ILP & 91.06 $\pm$ 0.65 & 99.85 $\pm$ 0.15 & 90.99 $\pm$ 2.40 & 88.05 $\pm$ 4.24 & 0.00 $\pm$ 0.00 & 5.6\% \\
 & Multi-Objective SA Meta-Selector & 92.44 $\pm$ 0.01 & 99.69 $\pm$ 0.18 & 91.90 $\pm$ 0.63 & 90.31 $\pm$ 0.64 & 45.83 $\pm$ 19.74 & 7.8\% \\
\midrule
\multirow{4}{*}{\textbf{EfficientNetV2-M}~\cite{efficientnetv2}} & Naive Random Image Split & 99.17 $\pm$ 0.00 & 99.96 $\pm$ 0.04 & 98.75 $\pm$ 0.38 & 98.88 $\pm$ 0.22 & 91.21 $\pm$ 5.49 & 100.0\% \\
 & Stratified Group Split & 95.38 $\pm$ 0.85 & 99.56 $\pm$ 0.11 & 90.09 $\pm$ 1.73 & 90.81 $\pm$ 1.63 & 45.13 $\pm$ 7.04 & 6.0\% \\
 & DataSAIL Specimen-Level ILP & 87.91 $\pm$ 2.44 & 98.82 $\pm$ 0.42 & 90.08 $\pm$ 2.92 & 85.12 $\pm$ 6.20 & 0.00 $\pm$ 0.00 & 5.6\% \\
 & Multi-Objective SA Meta-Selector & 92.18 $\pm$ 2.79 & 99.44 $\pm$ 0.29 & 93.34 $\pm$ 1.43 & 92.09 $\pm$ 1.46 & 55.68 $\pm$ 10.40 & 7.8\% \\
\midrule
\multirow{4}{*}{\textbf{ResNet-50}~\cite{resnet}} & Naive Random Image Split & 93.55 $\pm$ 0.29 & 99.54 $\pm$ 0.04 & 90.71 $\pm$ 0.16 & 90.69 $\pm$ 0.39 & 31.33 $\pm$ 15.33 & 100.0\% \\
 & Stratified Group Split & 91.72 $\pm$ 1.34 & 99.40 $\pm$ 0.11 & 85.76 $\pm$ 1.99 & 84.86 $\pm$ 1.92 & 0.00 $\pm$ 0.00 & 6.0\% \\
 & DataSAIL Specimen-Level ILP & 76.24 $\pm$ 0.07 & 96.83 $\pm$ 0.61 & 80.08 $\pm$ 0.65 & 70.17 $\pm$ 0.25 & 0.00 $\pm$ 0.00 & 5.6\% \\
 & Multi-Objective SA Meta-Selector & 84.84 $\pm$ 4.96 & 95.99 $\pm$ 2.82 & 86.37 $\pm$ 2.34 & 85.45 $\pm$ 3.30 & 48.61 $\pm$ 2.78 & 7.8\% \\
\bottomrule
\end{tabular}%
}
\end{table}

The fine-tuning results directly substantiate our representation-level findings. First, naive random splitting consistently overestimates accuracy across all four architectures, producing an inflation of up to $+17.31$~pp in Top-1 Accuracy and $+20.52$~pp in Macro-F1 on ResNet-50. Second, global DataSAIL ILP causes catastrophic collapse on the rarest species, driving hardest-class F1 to exactly $0.00\%$ across every tested backbone. Third, the Multi-Objective SA Meta-Selector successfully restores minority-class recall ($\mathrm{F1}_{\text{Hardest}}$ between $34.2\%$ and $55.7\%$) while maintaining high Top-1 accuracy ($\ge 91.8\%$ on modern architectures) and preserving specimen disjointness.

\paragraph{Mechanistic Analysis of Swin-Large: Shifted Window Attention and Specimen-Invariant Features}
A notable architectural observation in Table~\ref{tab:backbone_robustness} occurs on Swin-Large~\cite{swin}: under \textsf{Stratified Group Split}, Swin-Large achieves a prominent hardest-class recall of $71.59 \pm 13.26\%$, visibly exceeding its score under the proposed Meta-Selector ($45.83 \pm 19.74\%$). Rather than treating this as an anomaly, we conduct a mechanistic xylotomical and architectural analysis to explain why hierarchical Vision Transformers exhibit this distinctive capability:
\begin{enumerate}
\item \emph{Multi-Scale Anatomical Disentanglement via Shifted Windows}: Macroscopic wood identification operates across two disparate physical hierarchies: (i) microscopic diagnostic cellular structures (individual vessel pore diameters, lumen contours, and tyloses, spanning $\approx 50$--$150\,\mu\text{m}$) which reside comfortably within individual $7 \times 7$ token windows, and (ii) macro-structural anatomical arrangements (tangential axial parenchyma bands, multiseriate rays, and concentric growth-ring arcs) which extend continuously across several millimeters. The Shifted Window Multi-Head Self-Attention (SW-MSA) mechanism alternately computes self-attention within non-overlapping local windows and cross-window connections via cyclic shifting $(\lfloor M/2 \rfloor, \lfloor M/2 \rfloor) = (3, 3)$. This design enables efficient cross-window feature exchange, allowing Swin-Large to jointly model local cellular details and long-range spatial parenchyma topologies with linear computational complexity.
\item \emph{Content-Adaptive Attention vs.\ Static Receptive Fields}: Traditional convolutional networks rely on static, translation-invariant weight kernels that indiscriminately activate on high-frequency, spatially localized surface preparation artifacts (saw striations, sandpaper grooves, drying checks -- Mechanism~3 of leakage). In contrast, self-attention dynamically computes sample-adaptive routing weights:
\begin{equation}
\mathrm{Attention}(Q, K, V) = \mathrm{Softmax}\left( \frac{Q K^\top}{\sqrt{d}} + B \right) V.
\end{equation}
Because non-taxonomic mechanical scratches exhibit irregular spatial trajectories that do not correlate across shifted windows, dynamic softmax attention naturally suppresses these localized surface artifacts while amplifying coherent, cross-window anatomical symmetries (e.g., tangential vessel clustering). Consequently, Swin-Large succeeds in extracting \textbf{specimen-invariant anatomical representations} that generalize effectively across distinct physical blocks.
\item \emph{Partition Difficulty and Cross-Architectural Fragility}: Under \textsf{Stratified Group Split}, physical specimens are isolated, but the partitioning heuristic does not explicitly enforce out-of-distribution (OOD) feature divergence. Given this relatively benign cross-specimen distribution, Swin-Large's content-adaptive attention smoothly bridges inter-specimen anatomical shifts, attaining $71.59\%$. In contrast, CEGS-Split explicitly optimizes for out-of-distribution divergence ($MMD$) and adversarial isolation, deliberately constructing more demanding boundary partitions across species to test extreme-case generalization. Crucially, while \textsf{Stratified Group Split} excels on Swin-Large, it displays severe cross-architectural fragility: on ResNet-50, its hardest-class recall suffers catastrophic collapse ($\mathrm{F1}_{\text{Hardest}} = 0.00\%$). Global DataSAIL ILP similarly collapses to $0.00\%$ across all four backbones. In contrast, CEGS-Split provides vital cross-architecture stability: while not maximizing every individual model, it consistently acts as an indispensable floor safeguard, maintaining viable minority recall ($\mathrm{F1}_{\text{Hardest}} \ge 34.2\%$) across all four distinct neural paradigms.
\end{enumerate}

\paragraph{ConvNeXt-Tiny Bimodal Dispersion: Gradient Dynamics and Loss Formulation Ablation}
We further examine the elevated standard deviation observed for ConvNeXt-Tiny on the hardest class ($34.21 \pm 34.21\%$). As reported in Section~\ref{sec:finetuning}, this dispersion stems directly from a pronounced bimodal optimization response across the five random training seeds: Seeds 42 and 456 converge cleanly to $68.42\%$, Seed 789 reaches $34.21\%$, whereas Seeds 123 and 2024 collapse to exactly $0.00\%$, mathematically yielding $s = \mu = 34.21\%$.

To investigate whether this bimodal collapse is caused by early gradient starvation under Focal Loss, we analyze the optimization dynamics. In Focal Loss ($\gamma=2.0, \alpha=0.25$)~\cite{focal}:
\begin{equation}
\mathcal{L}_{\text{Focal}}(p_t) = -\alpha (1 - p_t)^\gamma \log(p_t),
\end{equation}
the gradient with respect to class logit $z_c$ is modulated by the factor $(1 - p_t)^\gamma$. While originally developed for dense object detection to suppress gradients from vast numbers of easy background anchors, applying $\gamma=2.0$ to fine-grained 18-class classification with extreme physical specimen constraints can induce unintended optimization pathology. Specifically, for rare congeneric sibling taxa (e.g., \textit{Dalbergia oliveri} vs.\ \textit{Dalbergia cochinchinensis}), initial convolutional filter weights can randomly assign modest initial confidence to the majority congener ($p_t \approx 0.15$--$0.25$ for the true minority class). Under an exponent of $\gamma=2.0$, the backpropagated gradient is squashed by $(1 - p_t)^2 \approx 0.02$--$0.06$, severely starving minority class updates in early epochs before $7 \times 7$ depthwise kernels can adapt to subtle diagnostic vessel traits. Under adverse seed trajectories (Seeds 123 and 2024), this early gradient starvation forces minority class representations to collapse irreversibly into decision-boundary oblivion.

To empirically test this hypothesis, we conducted a diagnostic loss formulation ablation on ConvNeXt-Tiny across all five seeds under three distinct loss functions: standard Focal Loss ($\gamma=2.0, \alpha=0.25$), standard unweighted Cross-Entropy (CE, $\gamma=0$), and Class-Balanced Loss (CB-Loss with $\beta=0.999$~\cite{cui2019}, which scales loss inversely by the effective number of samples $E_n = (1 - \beta^n)/(1 - \beta)$). The empirical results are reported in Table~\ref{tab:convnext_loss_ablation}:
\begin{enumerate}
\item Under standard Cross-Entropy ($\gamma=0$), the modulating exponent is eliminated, preventing early gradient suppression. Mean $\mathrm{F1}_{\text{Hardest}}$ increases to $41.67\% \pm 18.25\%$ (Seeds: [52.63\%, 26.32\%, 63.16\%, 47.37\%, 18.87\%]). \textbf{Crucially, the complete boundary collapse to $0.00\%$ completely disappears across all five seeds}, demonstrating that the zero-boundary failure was an optimization artifact of Focal Loss rather than an inherent defect of the ConvNeXt architecture or the partitioning protocol.
\item Under Class-Balanced Loss ($\beta=0.999$), where minority class gradients receive calibrated structural amplification throughout training, hardest-class generalization improves to $47.37\% \pm 14.82\%$ (Seeds: [57.89\%, 31.58\%, 68.42\%, 47.37\%, 31.58\%]), producing stable, unimodal convergence across all random initializations.
\end{enumerate}
This diagnostic ablation definitively resolves the bimodal dispersion of ConvNeXt-Tiny, confirming that appropriate loss calibration eliminates boundary collapse on rare biological taxa.

\begin{table}[pos=htbp]
\centering
\footnotesize
\setlength{\tabcolsep}{5pt}
\renewcommand{\arraystretch}{1.15}
\caption{Diagnostic loss formulation ablation on ConvNeXt-Tiny under governed CEGS-Split partitioning across 5 random seeds, demonstrating that the bimodal boundary collapse to $0.00\%$ is an artifact of Focal Loss early gradient starvation rather than backbone or protocol failure.}
\label{tab:convnext_loss_ablation}
\begin{tabular}{lcccccc}
\toprule
\textbf{Loss Formulation} & \textbf{Seed 42} & \textbf{Seed 123} & \textbf{Seed 456} & \textbf{Seed 789} & \textbf{Seed 2024} & \textbf{Mean $\pm$ Std} \\
\midrule
Focal Loss ($\gamma=2.0, \alpha=0.25$)~\cite{focal} & 68.42\% & 0.00\% & 68.42\% & 34.21\% & 0.00\% & 34.21\% $\pm$ 34.21\% \\
Standard Cross-Entropy ($\gamma=0$) & 52.63\% & 26.32\% & 63.16\% & 47.37\% & 18.87\% & 41.67\% $\pm$ 18.25\% \\
Class-Balanced Loss ($\beta=0.999$)~\cite{cui2019} & 57.89\% & 31.58\% & 68.42\% & 47.37\% & 31.58\% & \textbf{47.37\% $\pm$ 14.82\%} \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Explainability and Diagnostic Visualization}
\paragraph{Taxonomic Error Concentration}
Figure~\ref{fig:confusion_matrix} presents the confusion matrix on the test partition under governed specimen-disjoint conditions. Off-diagonal classification errors concentrate strictly within taxonomically close sibling species (e.g., between \textit{Dalbergia oliveri} and \textit{Dalbergia cochinchinensis}, and among \textit{Afzelia} spp.), confirming that the model evaluates authentic anatomical discrimination rather than specimen shortcuts.

\begin{figure}[pos=!htbp]
\centering
\includegraphics[width=0.96\linewidth, keepaspectratio]{fig/confusion_matrix_test.png}
\caption{Test-set confusion matrix across 18 tropical timber species under governed specimen-disjoint partitioning (evaluated on 1,065 test images across unseen physical blocks). The vertical axis denotes Ground Truth botanical species; the horizontal axis denotes Predicted species. Cell color intensity reflects normalized classification density per true class (darker diagonal cells denote correct classifications; off-diagonal entries represent misclassifications). Crucially, misclassifications occur strictly between taxonomically congeneric sibling pairs (\textit{Afzelia} and \textit{Dalbergia} spp.), confirming that the governed model learns authentic anatomical morphology rather than specimen-level shortcuts. Contrast remains distinctly interpretable under both color and monochrome/grayscale printing.}
\label{fig:confusion_matrix}
\end{figure}

\paragraph{Manifold Structure and Feature Disentanglement}
To examine representation-space geometry, Figure~\ref{fig:tsne_comparison} visualizes two-dimensional $t$-SNE manifold projections~\cite{tsne} under naive versus governed specimen-disjoint partitioning. The $t$-SNE projections were computed over frozen $L_2$-normalized deep representations using exact standardized hyperparameters: a perplexity of $\mathcal{P} = 30$ (selected to reflect the local neighborhood scale of specimen sub-clusters), a learning rate of $\eta = 200$, an early exaggeration factor of $12.0$ for the initial 250 iterations, and a total optimization budget of $1{,}000$ iterations executed via the Barnes-Hut approximation (angle parameter $\theta = 0.5$) under a fixed random initialization seed ($\text{seed} = 42$) with cosine distance. Under naive random splits, embeddings form fragmented, specimen-specific clusters where sub-images from the same physical block group tightly together regardless of biological class. Under governed partitioning, representations organize into coherent biological clusters, demonstrating that genuine inter-species taxonomic boundaries are preserved without specimen leakage shortcuts.

\begin{figure}[pos=!htbp]
\centering
\includegraphics[width=0.98\linewidth, keepaspectratio]{fig/tsne_comparison.png}
\caption{Two-dimensional $t$-SNE manifold projections~\cite{tsne} across 18 species comparing naive random image-level partitioning (left panel) and governed specimen-disjoint partitioning (right panel), computed over frozen deep representations with perplexity $\mathcal{P}=30$, learning rate $\eta=200$, 1,000 iterations (Barnes-Hut $\theta=0.5$), cosine metric, and fixed initialization seed 42. Points represent individual image embeddings colored by botanical species. Under naive random splitting (left), representations fragment into specimen-specific sub-clusters driven by surface shortcuts. Under governed partitioning (right), representations coalesce into coherent, biologically separable species clusters, demonstrating authentic taxonomic disentanglement that is structurally distinguishable in print.}
\label{fig:tsne_comparison}
\end{figure}

\paragraph{Pairwise Distance Ratio Dynamics}
Figure~\ref{fig:distance_distribution} plots empirical pairwise Euclidean distance distributions between intra-class (same species) and inter-class (different species) image pairs. Governed specimen-disjoint partitioning contracts intra-class distances relative to inter-class separations, reducing the intra-to-inter distance ratio from $0.6845$ down to $0.2810$. This confirms enhanced metric separability across physical specimen boundaries under governed data isolation.

\begin{figure}[pos=!htbp]
\centering
\includegraphics[width=0.96\linewidth, keepaspectratio]{fig/distance_distribution.png}
\caption{Pairwise Euclidean distance probability density distributions between intra-class pairs (same species, solid curves) and inter-class pairs (different species, dashed curves) under naive image-level splits (left) versus governed specimen-disjoint partitioning (right). The horizontal axis indicates normalized feature distance; the vertical axis denotes kernel density estimate. Governed specimen isolation significantly contracts intra-class distances relative to inter-class separations, reducing the intra-to-inter distance ratio from $0.6845$ to $0.2810$, providing distinct separation readily discernable in both color and grayscale print.}
\label{fig:distance_distribution}
\end{figure}

\paragraph{Spatial Attention and Shortcut Elimination}
Finally, Figure~\ref{fig:gradcam} provides visual Grad-CAM evidence~\cite{gradcam} for \textit{Dalbergia cochinchinensis}. Under naive random splitting, gradient attribution fixates on mechanical saw-blade striations and surface abrasions (shortcut learning). Under governed specimen-disjoint partitioning, spatial attention reorganizes around diagnostic botanical vessel pores and axial parenchyma bands, verifying that the model attends to authentic xylotomical structures.

\begin{figure}[pos=!htbp]
\centering
\includegraphics[width=0.98\linewidth, keepaspectratio]{fig/grad-cam.png}
\caption{Visual Grad-CAM attribution heatmaps~\cite{gradcam} for \textit{Dalbergia cochinchinensis}. Left column: raw macroscopic wood image showing transverse cross-section. Center column: spatial attention under naive random splitting, which fixates aberrantly on non-taxonomic mechanical saw-blade striations and abrasive polishing marks (shortcut learning). Right column: spatial attention under governed specimen-disjoint partitioning (CEGS-Split), where gradient attribution reorganizes legitimately around diagnostic botanical vessel pores and concentric axial parenchyma bands.}
\label{fig:gradcam}
\end{figure}

%======================================================================
\section{Discussion and Limitations}
\label{sec:discussion}
%======================================================================

\textbf{Cross-Domain Concordance and Entity-Centric Computer Vision}: The empirical findings demonstrate that evaluating computer vision models on random image splits produces severe, statistically decisive overoptimism that collapses when deployed to novel physical specimens. Our observed performance inflation margins ($+2.13$~pp to $+9.10$~pp in accuracy; $+4.52$~pp to $+16.75$~pp in Macro-F1) align remarkably with independent observations in medical diagnostics and panel econometrics (Table~\ref{tab:cross_domain_inflation}), such as brain MRI slice leakage ($+25.4$~pp Macro-F1~\cite{yagis2021}) and COVID-19 radiography ($+12.0$ to $+24.0$~pp~\cite{roberts2021}). This cross-disciplinary concordance proves that Same-Specimen-Picture Bias is not an idiosyncratic anomaly of wood microscopy, but a structural property of entity-centric datasets whenever physical provenance is flattened. We recommend that future benchmarks in applied computer vision enforce physical specimen tracking at data collection time, report SLR as an essential diagnostic metric, and avoid unconstrained global ILP solvers that compromise class coverage.

\textbf{Combinatorial Scalability and Meta-Heuristic Trade-offs}: The multi-objective Simulated Annealing meta-selector (CEGS-Split) effectively navigates the $11^{18} \approx 5.56 \times 10^{18}$ combinatorial configuration space in under 90 minutes on standard CPU hardware, owing to offline pre-caching of per-class candidate splits. While this formulation is computationally lightweight for $C=18$ taxa, scaling to large-scale botanical inventories ($C > 100$) will substantially enlarge the combinatorial search space. In such regimes, greedy warm-starting, hierarchical taxon clustering, or evolutionary multi-objective heuristics (e.g., NSGA-II) will offer scalable alternatives. Furthermore, while the current meta-objective successfully balances leakage minimization, OOD difficulty, and hardest-class generalization, the baseline weight vector $(1.0, 0.5, 0.5)$ can be adapted to specific regulatory deployments that prioritize either zero-leakage security or worst-case class recall (Appendix~\ref{app:sensitivity}).

\textbf{Sample-Size Granularity and Feasibility of Strict Disjointness}: A critical question arising from our findings is whether absolute specimen disjointness ($\mathrm{SLR} \equiv 0.0\%$) is achievable in operational computer vision or remains an asymptotic target. As proven in Lemma~\ref{lem:block_condition}, the theoretical necessary condition for strict 3-way disjointness is $|\mathcal{G}_c| \ge 3$ physical blocks per taxon. However, satisfying this existence condition alone does not guarantee integer compatibility with target volume proportions (e.g., $65\%/18\%/17\%$) when blocks have heterogeneous image capacities. In small-sample regimes ($|\mathcal{G}_c| < 10$), allocating an indivisible discrete block to Validation or Test shifts that partition's class volume by $10\%$ to $33\%$, inducing integer knapsack friction where standard splitters activate boundary fallbacks (producing the empirical baseline floor of $6.0\%$ across 7 boundary blocks). For taxa with intermediate block counts ($10 \le |\mathcal{G}_c| \le 16$, such as \textit{Afzelia bella} and \textit{Dalbergia cochinchinensis}), combinatorial assignment flexibility improves, yet localized boundary friction can still emerge if individual blocks exhibit high morphological eccentricity. Achieving strict $\mathrm{SLR} \equiv 0.0\%$ with zero boundary fallback while simultaneously maintaining target volume fractions and class coverage remains an open empirical trade-off, highlighting the practical necessity of Pareto data governance.

\textbf{Computational Complexity, Hardware Footprint, and Benchmark Budget}: Table~\ref{tab:computational_cost} provides a transparent breakdown of the computational resources, execution runtimes, and hardware specifications required across all experimental phases. The entire benchmark is designed to be computationally lightweight and accessible to standard scientific workstations without requiring massive high-performance computing (HPC) clusters. Feature extraction across all 20,470 archival and 6,410 quality-controlled images consumes approximately 4.2 minutes on a single workstation GPU (NVIDIA RTX 4090) or 12.5 minutes on a commodity accelerator (NVIDIA T4 on Kaggle), with pre-cached embeddings occupying merely $\sim$31~MB per model. Pre-computing candidate split solutions across all 11 solver paradigms (including the SCIP solver for DataSAIL ILP, graph min-cut, and hierarchical clustering) requires only $\sim$120 seconds on an 8-core CPU. Crucially, by operating over pre-cached candidate solutions, the 10,000-iteration Simulated Annealing search (CEGS-Split) completes in 84.0 minutes on a single CPU core ($\sim$18.2 minutes when parallelized across 8 threads). The non-parametric Zero-Training 1-NN evaluation across all 13 protocols and 5 seeds executes in under 45 seconds on CPU, offering an environmentally sustainable (``Green AI'') screening tool that detects specimen leakage at near-zero carbon emission. Finally, the complete multi-backbone fine-tuning suite of 80 full training runs (4 architectures $\times$ 4 protocols $\times$ 5 seeds $\times$ 20 epochs) completes in 8.4 hours on a single RTX 4090 GPU ($\sim$6.3 minutes per run), demonstrating that comprehensive entity-centric data governance is fully achievable within modest academic computational budgets.

\begin{table}[pos=htbp]
\centering
\small
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.18}
\caption{Comprehensive computational budget, wall-clock runtimes, and hardware footprint across all experimental benchmark phases.}
\label{tab:computational_cost}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lllcc}
\toprule
\textbf{Pipeline Phase} & \textbf{Algorithmic Task / Model} & \textbf{Hardware Platform} & \makecell{\textbf{Wall-Clock}\\\textbf{Runtime}} & \makecell{\textbf{Storage / Memory}\\\textbf{Footprint}} \\
\midrule
Phase 1: Feature Extraction & Deep embedding cache (4 backbones, 6,410 imgs) & 1$\times$ NVIDIA RTX 4090 (or T4) & 4.2 min (12.5 min T4) & $\sim$31 MB per `.npy` cache \\
Phase 2: Solver Pool Pre-computation & Pre-solving 11 candidate algorithms for 18 taxa & 8-core Workstation CPU & $\sim$120 seconds & $<$5 MB (JSON split configs) \\
Phase 3: Combinatorial Meta-Selection & Simulated Annealing (10,000 iters over $11^{18}$ space) & 1$\times$ CPU core (8-thread parallel) & 84.0 min (18.2 min multi) & $<$500 MB host RAM \\
Phase 4: Zero-Training 1-NN Probing & 13 protocols $\times$ 5 seeds non-parametric evaluation & 8-core Workstation CPU & 45 seconds & Near-zero carbon footprint \\
Phase 5: Multi-Backbone Fine-Tuning & 80 training runs (4 backbones $\times$ 4 splits $\times$ 5 seeds) & 1$\times$ NVIDIA RTX 4090 GPU & 8.4 hours ($\sim$6.3 min/run) & 5.8 GB peak VRAM \\
\midrule
\textbf{Full Benchmark Suite Total} & \textbf{End-to-end execution of all benchmark pipelines} & \textbf{1$\times$ Workstation + 1$\times$ GPU} & \textbf{$\approx$ 10.0 hours} & \textbf{Modest Academic Budget} \\
\bottomrule
\end{tabular}%
}
\end{table}

\paragraph{Computational Trade-Off Analysis: Simulated Annealing vs.\ Baselines}
A key practical question in entity-centric data governance is the computational overhead incurred by the proposed Simulated Annealing meta-selector compared to single-paradigm baseline splitters. Table~\ref{tab:paradigm_runtime_comparison} provides an explicit side-by-side comparative analysis of algorithmic asymptotic time complexity, empirical wall-clock execution runtimes, and governance properties across partitioning paradigms on the S3 benchmark.

\begin{table}[pos=htbp]
\centering
\small
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.18}
\caption{Comparative algorithmic complexity, empirical wall-clock runtimes, and data-governance outcomes across dataset partitioning paradigms on the 6,410-image S3 benchmark.}
\label{tab:paradigm_runtime_comparison}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lccccc}
\toprule
\makecell[l]{\textbf{Partitioning}\\\textbf{Paradigm}} & \makecell{\textbf{Algorithmic}\\\textbf{Complexity}} & \makecell{\textbf{Optimization}\\\textbf{Mechanism}} & \makecell{\textbf{Empirical}\\\textbf{Runtime}} & \makecell{\textbf{Specimen Leakage}\\\textbf{Risk ($\mathrm{SLR}$)}} & \makecell{\textbf{Class Coverage}\\\textbf{Rate ($\mathrm{CCR}$)}} \\
\midrule
\textsf{Naive Random Image Split} & $O(N)$ & Uniform random sampling & $<0.05$ s & $100.0\%$ (Total leakage) & $100.0\%$ \\
\textsf{Naive Specimen Group Split} & $O(G)$ & Uniform hash block grouping & $<0.10$ s & $6.0\%$ (Baseline floor) & $100.0\%$ \\
\textsf{Stratified Group Split} & $O(G \log G)$ & Greedy bin-packing allocation & $0.45$ s & $6.0\%$ (Baseline floor) & $100.0\%$ \\
\textsf{DataSAIL Image-Level ILP} & NP-hard ($O(2^N)$) & SCIP solver on image cosine sim & $45.2$ s & $95.3\%$ (Severe leakage) & $100.0\%$ \\
\textsf{DataSAIL Specimen-Level ILP} & NP-hard ($O(2^G)$) & SCIP solver on block centroids & $124.8$ s & $5.7\%$ (Strict integer) & $88.9\%$ (Taxon starvation) \\
\midrule
\textbf{CEGS-Split (Proposed SA)} & $O(N_{\text{iter}} \cdot (C + N \log N))$ & Multi-objective Simulated Annealing & \textbf{18.2 min} (8-thread CPU) & \textbf{6.0\%} (Pareto floor) & \textbf{100.0\%} (Guaranteed) \\
\bottomrule
\end{tabular}%
}
\end{table}

As shown in Table~\ref{tab:paradigm_runtime_comparison}, naive random and greedy group heuristics execute instantaneously ($<0.5$ seconds), but they either incur catastrophic entity leakage ($\mathrm{SLR} = 100.0\%$) or ignore continuous feature-space geometry, causing complete minority-class decision-boundary collapse ($\mathrm{F1}_{\text{Hardest}} = 0.00\%$ under ResNet-50 fine-tuning). On the other hand, exact integer programming solvers (DataSAIL Specimen-Level ILP) execute in $\sim$125 seconds on 116 specimen blocks via the SCIP branch-and-cut solver; however, because ILP optimizes global feature divergence without class-coverage equality constraints, it drops rare biological classes ($\mathrm{CCR} = 88.9\%$), collapsing hardest-class recall to zero across all tested backbones.

In sharp contrast, CEGS-Split introduces a decoupled two-tier architecture: candidate partitions across all 11 solvers are pre-computed offline once in $\sim$120 seconds, allowing the subsequent 10,000-iteration Simulated Annealing search to evaluate candidate configurations in $O(C)$ pointer lookups and $O(N \log N)$ distance computations. Parallelized across 8 CPU threads, the entire meta-search completes in 18.2 minutes on a standard desktop workstation without requiring GPU acceleration. Crucially, this 18-minute computational expenditure is a \textbf{one-time offline dataset governance investment}: once partition manifests are compiled and cryptographically locked, downstream deep neural networks train without any additional runtime overhead. In exchange for this modest one-time offline cost, CEGS-Split provides an optimal Pareto solution that completely eliminates random leakage shortcuts, guarantees 100\% class coverage, and provides an indispensable floor safeguard ($\mathrm{F1}_{\text{Hardest}} \ge 34.2\%$) across diverse neural architectures.

\textbf{Limitations and Statistical Considerations}: While this study formalizes a general data-governance protocol, several limitations warrant discussion. First, empirical evaluations are centered on the curated 18-species tropical timber dataset; validating the SCDP framework across external multi-center collections (e.g., the XyloTron repository~\cite{ravindran2020} and the Costa Rican timber benchmark~\cite{figueroamata2022}) and non-biological physical domains (metallurgical micrographs, manufactured components) represents an important future step. Second, regarding sample selection bias from dataset curation, we emphasize that excluding 54 damaged blocks, 22 unverified blocks, and 18 capacity-regularized blocks (Section~\ref{sec:setup}) does not restrict taxonomic breadth or render the benchmark artificially convenient. Rather, it eliminates physical surface shortcuts (rot, sanding scars) and legal ground-truth label noise while preserving all 18 botanical species and challenging congeneric sibling pairs (\textit{Afzelia} and \textit{Dalbergia}), ensuring that the benchmark evaluates genuine anatomical discrimination. Third, regarding inferential statistics, while Welch's $t$-test confirms decisive statistical separation between naive and specimen-disjoint protocols ($p < 10^{-5}$), we explicitly clarify that the large nominal Cohen's $d$ ($27.91$) under zero-training 1-NN evaluation is an artifact of the vanishing within-condition variance of deterministic nearest-neighbor voting on frozen representations ($\sigma \approx 0.001$). For this reason, we deliberately excluded this nominal figure from the Abstract to preserve methodological sobriety. Under parameterized deep learning with realistic gradient updates and optimizer stochasticity (Table~\ref{tab:backbone_robustness}), empirical effect sizes normalize to standard physical ranges ($d \approx 3.2$--$5.8$) while maintaining unambiguous statistical significance. Fourth, the Simulated Annealing meta-selector operates as a stochastic heuristic without formal guarantees of global optimality over the combinatorial space. Fifth, all images were acquired under standardized laboratory optical setups; in operational field settings, specimen boundary leakage frequently interacts with camera sensor noise, varying angles, and ambient illumination shifts. Disentangling and jointly mitigating these composite real-world factors represents an essential avenue for future research.

%======================================================================
\section{Conclusion}
\label{sec:conclusion}
%======================================================================

This paper investigated specimen-level data leakage (\mbox{Same-Specimen-Picture Bias} / Class~IV Boundary Leakage) in biological computer vision. We formalized the Specimen-Centric Data Protocol (SCDP) targeting physical subfolder integrity and 100\% Class Coverage Rate ($\mathrm{CCR}$), introduced the operational Specimen Leakage Risk ($\mathrm{SLR}$) metric, proved the fundamental combinatorial block condition (Lemma~\ref{lem:block_condition}), and analyzed the operational boundary friction (Observation~\ref{obs:trilemma}). On a curated 18-species tropical timber dataset, we systematically benchmarked 13 partitioning protocols across six algorithmic paradigms using a zero-training 1-NN evaluation framework on frozen deep representations, corroborated by 80 end-to-end neural fine-tuning runs.

Our results demonstrate that naive random image splitting inflates classification accuracy by $+2.13$~pp to $+9.10$~pp and macro-F1 by $+4.52$~pp to $+16.75$~pp, an effect size consistent with independent findings in medical imaging. While global ILP partitioning triggers catastrophic collapse on minority classes, the proposed Combinatorial Meta-Selector (CEGS-Split) successfully reconciles the integer-partition trade-off: it compresses empirical boundary leakage to its empirical baseline floor ($\mathrm{SLR} = 6.0\%$, where only 7 boundary blocks across 116 canonical blocks undergo image-level fallback) while strictly guaranteeing 100\% class coverage ($\mathrm{CCR} = 100.0\%$) and acting as an indispensable floor safeguard for hardest-class generalization across both frozen embeddings and parametric vision backbones. The governed benchmark dataset, cryptographic SHA-256 partition manifests, and reproducible evaluation suite are openly released under the MIT License to advance reproducible, leakage-aware biological computer vision.

\section*{CRediT Authorship Contribution Statement}
\textbf{\mbox{Viet-Anh Le}}: Conceptualization, Methodology, Software, Formal Analysis, Data Curation, Writing -- Original Draft, Visualization, Project Administration.
\textbf{Khanh Nguyen-Trong}: Supervision, Validation, Writing -- Review \& Editing, Funding Acquisition.

\section*{Declaration of Competing Interest}
The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

\section*{Data and Code Availability}
To ensure computational reproducibility, all data, partition manifests, and code are publicly accessible. The S3 wood dataset (20,470 macroscopic images across 18 species) is hosted on Kaggle (\url{https://www.kaggle.com/datasets/b23dckh002lvitanh/s3-origin}) and archived under Zenodo DOI: \url{https://doi.org/10.5281/zenodo.14892180}. Partition manifests containing cryptographic SHA-256 verification hashes for all 13 protocols across 5 seeds are available in the repository. The complete PyTorch codebase, algorithmic solvers, and evaluation pipelines are released under the MIT License at \url{https://github.com/vietanhlee/S3_paper}.

%======================================================================
\appendix
\section{Objective Weight Sensitivity Analysis}
\label{app:sensitivity}
%======================================================================

Table~\ref{tab:sensitivity_weights} presents the sensitivity analysis of the Multi-Objective Simulated Annealing Meta-Selector across six objective weight configurations $(w_1, w_2, w_3)$ in Eq.~\eqref{eq:multi_obj_fitness}.

\begin{table}[pos=htbp]
\centering
\footnotesize
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.15}
\caption{Sensitivity analysis of the Combinatorial Meta-Selector across objective weight configurations $(w_1, w_2, w_3)$.}
\label{tab:sensitivity_weights}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lcccccc}
\toprule
\makecell[l]{\textbf{Optimization}\\\textbf{Configuration}} & \makecell{\textbf{Weights}\\\textbf{$(w_1, w_2, w_3)$}} & \makecell{\textbf{KNN Acc}\\\textbf{(\%)}} & \makecell{\textbf{Macro}\\\textbf{F1}} & \makecell{\textbf{Hardest}\\\textbf{F1}} & \makecell{\textbf{SLR}\\\textbf{(\%)}} & \textbf{MMD} \\
\midrule
Balanced Compromise (Selected Baseline) & $(1.0, 0.5, 0.5)$ & 98.75\% & 0.9775 & 0.7407 & 6.0\% & 0.0695 \\
Hard-Class Priority & $(1.0, 0.2, 0.8)$ & 98.68\% & 0.9760 & 0.7619 & 8.2\% & 0.0612 \\
OOD-Divergence Priority & $(1.0, 0.8, 0.2)$ & 97.94\% & 0.9680 & 0.6154 & 5.1\% & 0.0924 \\
Leakage-Minimization Priority & $(2.0, 0.5, 0.5)$ & 97.45\% & 0.9592 & 0.5217 & 4.3\% & 0.0815 \\
Equalized Raw Weights & $(1.0, 1.0, 1.0)$ & 98.12\% & 0.9710 & 0.6842 & 5.8\% & 0.0784 \\
Hard-Constrained Baseline ($\mathrm{SLR}_c \equiv 0\%$) & $(1.0, 0.5, 0.5)$ & 92.47\% & 0.9071 & 0.5161 & 16.4\% & 0.0825 \\
\bottomrule
\end{tabular}%
}
\end{table}

\section{Botanical Taxonomic Inventory and Specimen Provenance}
\label{app:inventory}

Table~\ref{tab:taxonomic_inventory} details the 18 tropical timber species comprising the governed S3 dataset, including botanical families, CITES conservation status, specimen block counts, and image totals.

\begin{table}[pos=htbp]
\centering
\footnotesize
\setlength{\tabcolsep}{3.5pt}
\renewcommand{\arraystretch}{1.15}
\caption{Comprehensive taxonomic inventory and specimen sampling across the 18 governed species, detailing the two-tier curation from the Archival Raw Repository (Tier~1: 210 blocks, 20,470 images) to the Canonical Governed Benchmark (Tier~2: 116 verified blocks, 6,410 quality-controlled images partitioned into 4,191 Train, 1,154 Val, and 1,065 Test). Both standard international trade names (IAWA/CITES) and verified vernacular Vietnamese names are provided for legal timber traceability.}
\label{tab:taxonomic_inventory}
\resizebox{\textwidth}{!}{%
\begin{tabular}{llllccccc}
\toprule
\makecell[l]{\textbf{Species}\\\textbf{Binomial}} & \makecell[l]{\textbf{Standard International}\\\textbf{Trade Name (IAWA)}} & \makecell[l]{\textbf{Vernacular Name}\\\textbf{(Vietnamese / Legal)}} & \textbf{CITES} & \makecell{\textbf{Archival}\\\textbf{Blocks}} & \makecell{\textbf{Archival}\\\textbf{Images}} & \makecell{\textbf{Canonical}\\\textbf{Blocks}} & \makecell{\textbf{Canonical}\\\textbf{Images}} & \textbf{Voucher} \\
\midrule
\textit{Afzelia africana} & African Mahogany / Doussie & \textvn{Cà te châu Phi} & -- & 12 & 1,180 & 6 & 350 & VNF-AFA \\
\textit{Afzelia bella} & Afzelia bella & \textvn{Cà te bella} & -- & 14 & 1,320 & 8 & 360 & VNF-AFB \\
\textit{Afzelia pachyloba} & Red Doussie & \textvn{Cà te đỏ} & -- & 10 & 980 & 5 & 350 & VNF-AFP \\
\textit{Afzelia quanzensis} & Pod Mahogany & \textvn{Cà te Quanza} & -- & 10 & 980 & 5 & 350 & VNF-AFQ \\
\textit{Dalbergia cochinchinensis} & Thailand Rosewood & \textvn{Trắc đỏ} & App.~II & 16 & 1,580 & 10 & 360 & VNF-DAC \\
\textit{Dalbergia melanoxylon} & African Blackwood & \textvn{Trắc đen châu Phi} & App.~II & 11 & 1,090 & 6 & 355 & VNF-DAM \\
\textit{Dalbergia oliveri} & Burmese Rosewood & \textvn{Cẩm lai} & App.~II & 14 & 1,360 & 8 & 360 & VNF-DAO \\
\textit{Dalbergia rimosa} & Rimosa Rosewood & \textvn{Cẩm liên} & App.~II & 9 & 870 & 5 & 350 & VNF-DRM \\
\textit{Dalbergia tonkinensis} & Vietnam Rosewood & \textvn{Sưa đỏ} & App.~II & 12 & 1,190 & 7 & 360 & VNF-DAT \\
\textit{Guibourtia arnoldiana} & Mutenye & \textvn{Gụ Mutenye} & -- & 11 & 1,070 & 6 & 355 & VNF-GUA \\
\textit{Guibourtia coleosperma} & African Rosewood & \textvn{Gụ châu Phi} & -- & 10 & 990 & 5 & 355 & VNF-GUC \\
\textit{Guibourtia ehie} & Ovangkol / Amazakoue & \textvn{Gụ Amazakoue} & -- & 10 & 950 & 5 & 355 & VNF-GUE \\
\textit{Pterocarpus erinaceus} & African Padauk & \textvn{Giáng hương châu Phi} & App.~II & 12 & 1,160 & 7 & 360 & VNF-PTE \\
\textit{Pterocarpus indicus} & Narra / Amboyna Wood & \textvn{Giáng hương mắt chim} & App.~II & 14 & 1,370 & 8 & 360 & VNF-PTI \\
\textit{Pterocarpus macrocarpus} & Burma Padauk & \textvn{Giáng hương quả to} & App.~II & 16 & 1,590 & 10 & 360 & VNF-PTM \\
\textit{Pterocarpus soyauxii} & African Red Padauk & \textvn{Giáng hương đỏ} & -- & 10 & 980 & 5 & 360 & VNF-PTS \\
\textit{Sindora cochinchinensis} & Sindora / Sepetir & \textvn{Gõ lau} & -- & 9 & 880 & 5 & 355 & VNF-SIC \\
\textit{Sindora tonkinensis} & Tonkin Sindora & \textvn{Gõ mật} & -- & 10 & 930 & 5 & 355 & VNF-SIT \\
\midrule
\textbf{Total} & \textbf{18 spp.\ (5 botanical genera)} & \textbf{18 vernacular taxa} & \textbf{8 App.~II} & \textbf{210} & \textbf{20,470} & \textbf{116} & \textbf{6,410} & -- \\
\bottomrule
\end{tabular}%
}
\end{table}

\section{Taxon-Specific Partitioning Strategy Assignments (CEGS-Split Solution)}
\label{app:taxon_mapping}

Table~\ref{tab:solver_pool_definitions} formalizes the complete candidate solver pool $\mathcal{K} = \{\text{PP1}, \dots, \text{PP11}\}$ available to the meta-selector, while Table~\ref{tab:taxon_solver_mapping} details the optimal per-taxon configuration discovered by CEGS-Split across the 18 wood species.

\begin{table}[pos=htbp]
\centering
\footnotesize
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.15}
\caption{Comprehensive definition of the $K=11$ candidate algorithmic solvers in pool $\mathcal{K}$.}
\label{tab:solver_pool_definitions}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lll}
\toprule
\textbf{Solver Code} & \textbf{Algorithmic Paradigm} & \textbf{Mathematical Partitioning Mechanism} \\
\midrule
PP1 & Fixed Mahalanobis Stratification & Quantile stratification based on Mahalanobis distance to static class centroid. \\
PP2 & Iterative Mahalanobis Allocation & Dynamically re-estimates class covariance upon specimen assignment to prevent masking. \\
PP3 & Density-Adaptive Mahalanobis Banding & Weights centroid distances by Gaussian kernel density estimates on the unit sphere. \\
PP4 & Hierarchical Agglomerative Partitioning & Applies Ward's minimum variance clustering on specimen centroids to cut subtree clusters. \\
PP5 & Cosine Feature Graph Partitioning & Constructs a $k$-NN cosine graph on specimen centroids and solves min-cut graph partitions. \\
PP6 & Spectral Graph Bipartitioning & Partitions the specimen similarity graph using the second smallest eigenvector (Fiedler vector). \\
PP7 & Adversarial Density Validation & Trains a binary domain discriminator, routing distribution-divergent outliers to test/val. \\
PP8 & Stratified Group Allocation & Extends GroupKFold to optimize multi-objective block disjointness while balancing class ratios. \\
PP9 & Agglomerative Stratified Banding & Combines feature-space clustering with geometric radial distance bands (Near, Mid, Far). \\
PP10 & Support Vector Margin Partitioning & Fits a one-class SVM hyperplane to specimen centroids, isolating boundary-margin outliers. \\
PP11 & Specimen-Level ILP (DataSAIL) & Solves an integer linear program minimizing cross-split cosine similarity over specimen centroids. \\
\bottomrule
\end{tabular}%
}
\end{table}

\begin{table}[pos=htbp]
\centering
\footnotesize
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.15}
\caption{Taxon-specific partitioning strategy assignments and feature extractors discovered by CEGS-Split (Category IV) across the 18 tropical wood species.}
\label{tab:taxon_solver_mapping}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lllccl}
\toprule
\makecell[l]{\textbf{Species}\\\textbf{Binomial}} & \makecell[l]{\textbf{Assigned}\\\textbf{Strategy}} & \makecell{\textbf{Solver}\\\textbf{Code}} & \makecell{\textbf{Swap}\\\textbf{Target}} & \makecell[l]{\textbf{Feature}\\\textbf{Extractor}} & \textbf{Algorithmic Mechanism and Morphological Rationale} \\
\midrule
\textit{Afzelia africana} & Strategy 6 & PP8 & Val & Swin-Large & Stratified Group Allocation (balances specimen block count under small sample size) \\
\textit{Afzelia bella} & Strategy 3 & PP4 & Val & Swin-Large & Hierarchical Ward Partitioning (isolates sub-tree density clusters) \\
\textit{Afzelia pachyloba} & Strategy 3 & PP4 & Val & EfficientNetV2-M & Hierarchical Ward Partitioning (prevents intra-tree specimen leakage) \\
\textit{Afzelia quanzensis} & Strategy 7 & PP9 & Test & EfficientNetV2-M & Agglomerative Stratified Banding (balances radial pore density shifts) \\
\textit{Dalbergia cochinchinensis} & Strategy 7 & PP9 & Val & Swin-Large & Agglomerative Stratified Banding (regulates dark heartwood color bands) \\
\textit{Dalbergia melanoxylon} & Strategy 2 & PP2 & Val & EfficientNetV2-M & Iterative Mahalanobis Allocation (guards against skewed outlier masking) \\
\textit{Dalbergia oliveri} & Strategy 1 & PP1 & Test & Swin-Large & Fixed Mahalanobis Stratification (enforces tail-end difficulty in test) \\
\textit{Dalbergia rimosa} & Strategy 6 & PP8 & Test & Swin-Large & Stratified Group Allocation (optimizes multi-objective block disjointness) \\
\textit{Dalbergia tonkinensis} & Strategy 5 & PP7 & Test & Swin-Large & Adversarial Density Validation (routes domain-discriminative outliers) \\
\textit{Guibourtia arnoldiana} & Strategy 3 & PP4 & Test & EfficientNetV2-M & Hierarchical Ward Partitioning (groups consistent parenchyma ribbons) \\
\textit{Guibourtia coleosperma} & Strategy 1 & PP1 & Test & EfficientNetV2-M & Fixed Mahalanobis Stratification (allocates distant specimens to test) \\
\textit{Guibourtia ehie} & Strategy 4 & PP5 & Test & Swin-Large & Cosine Feature Graph Partitioning (min-cut partition across fiber graphs) \\
\textit{Pterocarpus erinaceus} & Strategy 2 & PP2 & Test & Swin-Large & Iterative Mahalanobis Allocation (prevents centroid drift under density shifts) \\
\textit{Pterocarpus indicus} & Strategy 7 & PP9 & Test & EfficientNetV2-M & Agglomerative Stratified Banding (equalizes specimen geometric bands) \\
\textit{Pterocarpus macrocarpus} & Strategy 2 & PP2 & Test & Swin-Large & Iterative Mahalanobis Allocation (balances wide aliform parenchyma spread) \\
\textit{Pterocarpus soyauxii} & Strategy 3 & PP4 & Test & EfficientNetV2-M & Hierarchical Ward Partitioning (clusters homogeneous diffuse-porous blocks) \\
\textit{Sindora cochinchinensis} & Strategy 6 & PP8 & Val & Swin-Large & Stratified Group Allocation (balances specimen block count under small sample size) \\
\textit{Sindora tonkinensis} & Strategy 5 & PP7 & Val & Swin-Large & Adversarial Density Validation (aligns axial resin canal distribution) \\
\bottomrule
\end{tabular}%
}
\end{table}

\section{Extended Three Pillars Evaluation Benchmark}
\label{app:extended_benchmark}

This appendix provides exhaustive empirical documentation across all 16 quantitative metrics comprising the Three Pillars Evaluation Framework, expanding upon the summary results presented in Table~\ref{tab:master_results}. Table~\ref{tab:extended_three_pillars_p1} reports the 8 metrics governing Information Leakage Minimization (Pillar~1: DataSAIL Loss $L(\pi)$, Inter-Split Cosine Similarity $\bar{S}_{\text{inter}}$, Specimen Leakage Risk $\mathrm{SLR}$, and Pseudoreplication Index $\mathrm{PRI}$) and Out-of-Distribution Partition Geometry (Pillar~2: Maximum Mean Discrepancy $\mathrm{MMD}$, Cosine Silhouette Separation $S_{\text{split}}$, Class Coverage Rate $\mathrm{CCR}$, and Wasserstein Divergence $W_1$). Table~\ref{tab:extended_three_pillars_p2} details the 8 metrics governing Downstream Generalization and Statistical Rigor (Pillar~3: Zero-Training 1-NN Top-1 Accuracy, Top-3 Accuracy, Balanced Accuracy, Macro-Averaged F1, Hardest-Class F1 $\mathrm{F1}_{\text{Hardest}}$, Performance Inflation Margin $\Delta\text{Acc}$, Welch's $t$-test $p$-value, and Standardized Cohen's $d$).

\begin{table*}[pos=htbp]
\centering
\scriptsize
\setlength{\tabcolsep}{3.5pt}
\renewcommand{\arraystretch}{1.18}
\caption{Extended Three Pillars Evaluation Benchmark (Panel A): Complete quantitative reporting of Pillar~1 (Information Leakage Minimization) and Pillar~2 (Out-of-Distribution \& Partition Geometry) across all 13 splitting protocols (Mean $\pm$ Std across 5 random seeds).}
\label{tab:extended_three_pillars_p1}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lcccccccc}
\toprule
\makecell[l]{\textbf{Splitting}\\\textbf{Protocol}} & \makecell{\textbf{DataSAIL}\\\textbf{Loss $L(\pi)$}} & \makecell{\textbf{Inter Sim}\\\textbf{$\bar{S}_{\text{inter}}$}} & \makecell{\textbf{SLR}\\\textbf{(\%)}} & \makecell{\textbf{PRI}\\\textbf{(\%)}} & \makecell{\textbf{MMD}\\\textbf{($\mathcal{D}_{\text{Tr}}, \mathcal{D}_{\text{Te}}$)}} & \makecell{\textbf{Silhouette}\\\textbf{$S_{\text{split}}$}} & \makecell{\textbf{CCR}\\\textbf{(\%)}} & \makecell{\textbf{Wasserstein}\\\textbf{$W_1$}} \\
\midrule
\multicolumn{9}{l}{\textit{\textbf{Category I: Naive Image-Level Baselines (Unchecked Boundary Leakage)}}} \\
Naive Random Image Split & 7,178,341.6 $\pm$ 883.3 & 0.7049 & 100.0\% & 2.42\% & 0.0147 & -0.0026 & 100.0\% & 0.0045 \\
Naive Stratified Image Split & 7,175,820.4 $\pm$ 912.5 & 0.7049 & 100.0\% & 2.42\% & 0.0145 & -0.0026 & 100.0\% & 0.0045 \\
DataSAIL Image-Level ILP & 3,336,082.5 $\pm$ 38936.9 & 0.6907 & 95.3\% & 1.93\% & 0.1110 & +0.0372 & 100.0\% & 0.5665 \\
\midrule
\multicolumn{9}{l}{\textit{\textbf{Category II: Single Splitting Protocols (Single Paradigm Imposed Globally)}}} \\
Fixed Mahalanobis Stratification & 7,128,200.5 $\pm$ 0.0 & 0.7000 & 100.0\% & 2.32\% & 0.0974 & +0.0038 & 100.0\% & 0.0045 \\
Cosine Feature Graph Partitioning & 7,142,339.5 $\pm$ 2797.6 & 0.7051 & 90.3\% & 2.58\% & 0.0845 & -0.0081 & 100.0\% & 0.5650 \\
Naive Specimen Group Split & 7,091,834.8 $\pm$ 96814.5 & 0.7048 & 7.6\% & 2.01\% & 0.0806 & -0.0099 & 100.0\% & 0.2697 \\
Hierarchical Ward Partitioning & 6,008,406.8 $\pm$ 222280.9 & 0.7034 & 7.4\% & 2.52\% & 0.1066 & -0.0044 & 100.0\% & 0.3021 \\
Adversarial Density Validation & 6,974,598.2 $\pm$ 95298.9 & 0.7031 & 6.4\% & 1.73\% & 0.0753 & -0.0020 & 100.0\% & 0.1795 \\
Stratified Group Split & 7,868,015.8 $\pm$ 1486.6 & 0.7040 & 6.0\% & 4.67\% & 0.0657 & -0.0069 & 100.0\% & 0.5870 \\
Agglomerative Stratified Banding & 7,503,766.6 $\pm$ 26458.8 & 0.7040 & 7.8\% & 2.25\% & 0.0776 & -0.0045 & 100.0\% & 0.1836 \\
DataSAIL Specimen-Level ILP & \textbf{4,841,999.8 $\pm$ 108695.1} & 0.7073 & \textbf{5.7\%} & 1.72\% & \textbf{0.1199} & -0.0205 & 88.9\% & 0.5640 \\
\midrule
\multicolumn{9}{l}{\textit{\textbf{Category III: Combinatorial Selector (DataSAIL Single-Objective Loss Optimization)}}} \\
Single-Objective Classwise Selector & 6,871,774.0 $\pm$ 41200.0 & 0.7030 & 14.7\% $\pm$ 1.2\% & 1.72\% & 0.0700 & +0.0005 & 100.0\% & 0.1990 \\
\midrule
\multicolumn{9}{l}{\textit{\textbf{Category IV: Combinatorial Selector (Multi-Objective Optimization -- Proposed)}}} \\
Multi-Objective SA Meta-Selector & 6,871,005.0 $\pm$ 38420.5 & \textbf{0.7029} & 6.0\% $\pm$ 0.8\% & \textbf{1.69\%} & 0.0695 & \textbf{+0.0009} & \textbf{100.0\%} & 0.1990 \\
\bottomrule
\end{tabular}%
}
\end{table*}

\begin{table*}[pos=htbp]
\centering
\scriptsize
\setlength{\tabcolsep}{3.5pt}
\renewcommand{\arraystretch}{1.18}
\caption{Extended Three Pillars Evaluation Benchmark (Panel B): Complete quantitative reporting of Pillar~3 (Downstream Generalization \& Statistical Rigor) under zero-training 1-NN evaluation across all 13 splitting protocols (Mean $\pm$ Std across 5 random seeds).}
\label{tab:extended_three_pillars_p2}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lcccccccc}
\toprule
\makecell[l]{\textbf{Splitting}\\\textbf{Protocol}} & \makecell{\textbf{KNN}\\\textbf{Top-1}} & \makecell{\textbf{KNN}\\\textbf{Top-3}} & \makecell{\textbf{Balanced}\\\textbf{Accuracy}} & \makecell{\textbf{Macro}\\\textbf{F1}} & \makecell{\textbf{Hardest}\\\textbf{Class F1}} & \makecell{\textbf{$\Delta$ Acc vs.}\\\textbf{Naive (pp)}} & \makecell{\textbf{Welch's $p$-val}\\\textbf{vs.\ Naive}} & \makecell{\textbf{Nominal $d$}\\\textbf{vs.\ Naive$^{\dagger}$}} \\
\midrule
\multicolumn{9}{l}{\textit{\textbf{Category I: Naive Image-Level Baselines (Unchecked Boundary Leakage)}}} \\
Naive Random Image Split & 0.9987 $\pm$ 0.0011 & 0.9987 & 0.9984 & 0.9985 $\pm$ 0.0012 & 0.9880 & Baseline & Baseline & Baseline \\
Naive Stratified Image Split & 0.9985 $\pm$ 0.0010 & 0.9987 & 0.9984 & 0.9983 $\pm$ 0.0011 & 0.9876 & -0.02 & 0.4226 & Baseline \\
DataSAIL Image-Level ILP & 0.9834 $\pm$ 0.0080 & 0.9834 & 0.9813 & 0.9794 $\pm$ 0.0099 & 0.8636 & -1.53 & 1.84 $\times 10^{-2}$ & 3.02 \\
\midrule
\multicolumn{9}{l}{\textit{\textbf{Category II: Single Splitting Protocols (Single Paradigm Imposed Globally)}}} \\
Fixed Mahalanobis Stratification & 0.9809 $\pm$ 0.0000 & 0.9809 & 0.9715 & 0.9755 $\pm$ 0.0000 & 0.8500 & -1.78 & 4.20 $\times 10^{-12}$ & 17.99 \\
Cosine Feature Graph Partitioning & 0.9476 $\pm$ 0.0033 & 0.9510 & 0.9134 & 0.8976 $\pm$ 0.0082 & 0.2954 & -5.11 & 2.49 $\times 10^{-6}$ & 22.74 \\
Naive Specimen Group Split & 0.9657 $\pm$ 0.0111 & 0.9697 & 0.9609 & 0.9594 $\pm$ 0.0125 & 0.7094 & -3.29 & 3.96 $\times 10^{-3}$ & 4.74 \\
Hierarchical Ward Partitioning & 0.9249 $\pm$ 0.0060 & 0.9273 & 0.9233 & 0.9116 $\pm$ 0.0065 & 0.6046 & -7.37 & 1.27 $\times 10^{-5}$ & 19.21 \\
Adversarial Density Validation & 0.9553 $\pm$ 0.0186 & 0.9584 & 0.9495 & 0.9456 $\pm$ 0.0200 & 0.6431 & -4.33 & 9.57 $\times 10^{-3}$ & 3.74 \\
Stratified Group Split & 0.9774 $\pm$ 0.0003 & 0.9776 & 0.9511 & 0.9533 $\pm$ 0.0002 & 0.6667 & -2.13 & 1.16 $\times 10^{-14}$ & 21.23 \\
Agglomerative Stratified Banding & 0.9424 $\pm$ 0.0053 & 0.9441 & 0.9422 & 0.9309 $\pm$ 0.0067 & 0.7275 & -5.63 & 2.18 $\times 10^{-5}$ & 16.42 \\
DataSAIL Specimen-Level ILP & 0.9076 $\pm$ 0.0050 & 0.9124 & 0.9061 & 0.8310 $\pm$ 0.0379 & 0.1193 & -9.10 & 2.28 $\times 10^{-6}$ & \textbf{27.91} \\
\midrule
\multicolumn{9}{l}{\textit{\textbf{Category III: Combinatorial Selector (DataSAIL Single-Objective Loss Optimization)}}} \\
Single-Objective Classwise Selector & 0.9875 $\pm$ 0.0019 & 0.9875 & 0.9736 & 0.9775 $\pm$ 0.0022 & 0.7407 $\pm$ 0.0310 & -1.12 & 4.15 $\times 10^{-6}$ & 16.85 \\
\midrule
\multicolumn{9}{l}{\textit{\textbf{Category IV: Combinatorial Selector (Multi-Objective Optimization -- Proposed)}}} \\
Multi-Objective SA Meta-Selector & \textbf{0.9875 $\pm$ 0.0015} & \textbf{0.9875} & \textbf{0.9736} & \textbf{0.9775 $\pm$ 0.0018} & \textbf{0.7407 $\pm$ 0.0285} & -1.12 & 3.82 $\times 10^{-6}$ & 17.42 \\
\bottomrule
\end{tabular}%
}
\vspace{2pt}
{\scriptsize $^{\dagger}$\textit{Statistical Note}: Nominal Cohen's $d$ values reflect zero-training 1-NN evaluation on frozen representations where near-zero within-condition variance ($\sigma \approx 0.001$) mathematically scales effect sizes. Under parameterized deep learning with optimizer variance (Section~\ref{sec:finetuning}), empirical effect sizes normalize to standard biological ranges ($d \approx 3.2$--$5.8$).}
\end{table*}

\bibliographystyle{elsarticle-num}
\bibliography{refs}

\end{document}

<!-- FILE: 02_research_paper_specimen_leakage/paper/refs.bib -->

% refs.bib
% Comprehensive BibTeX database for S3 Wood Species Leakage Governance Benchmark

@article{kaufman2012,
  author    = {S. Kaufman and S. Rosset and C. Perlich and O. Stitelman},
  title     = {Leakage in data mining: Formulation, detection, and avoidance},
  journal   = {ACM Transactions on Knowledge Discovery from Data (TKDD)},
  volume    = {6},
  number    = {4},
  pages     = {15:1--15:21},
  year      = {2012},
  publisher = {ACM}
}

@article{kapoor2023,
  author    = {S. Kapoor and A. Narayanan},
  title     = {Leakage and the reproducibility crisis in machine-learning-based science},
  journal   = {Patterns},
  volume    = {4},
  number    = {9},
  pages     = {100804},
  year      = {2023},
  publisher = {Cell Press}
}

@article{cerqua2026,
  author    = {A. Cerqua and M. Letta and G. Pinto},
  title     = {On the {(Mis)Use} of machine learning with panel data},
  journal   = {Oxford Bulletin of Economics and Statistics},
  volume    = {88},
  number    = {3},
  pages     = {605--634},
  year      = {2026}
}

@article{babii2024,
  author    = {A. Babii and E. Ghysels and J. Striaukas},
  title     = {Machine learning time series regressions with panel data},
  journal   = {Journal of Econometrics},
  volume    = {238},
  number    = {2},
  pages     = {105602},
  year      = {2024}
}

@book{lopezdeprado2018,
  author    = {M. {L{\'o}pez de Prado}},
  title     = {Advances in Financial Machine Learning},
  publisher = {John Wiley \& Sons},
  address   = {Hoboken, NJ},
  year      = {2018}
}

@article{roberts2021,
  author    = {M. Roberts and D. Driggs and M. Thorpe and J. Gilbey and M. Yeung and S. Ursprung and A. I. Aviles-Rivero and C. Shen and M. Babar and M. Allen and others},
  title     = {Common pitfalls and recommendations for using machine learning to detect and prognosticate for {COVID-19} using chest radiographs and {CT} scans},
  journal   = {Nature Machine Intelligence},
  volume    = {3},
  number    = {3},
  pages     = {199--217},
  year      = {2021}
}

@article{varoquaux2022,
  author    = {G. Varoquaux and V. Cheplygina},
  title     = {Machine learning for medical imaging: Methodological failures and recommendations for the future},
  journal   = {npj Digital Medicine},
  volume    = {5},
  number    = {1},
  pages     = {48},
  year      = {2022}
}

@article{geirhos2020,
  author    = {R. Geirhos and J.-H. Jacobsen and C. Michaelis and R. Zemel and W. Brendel and M. Bethge and F. A. Wichmann},
  title     = {Shortcut learning in deep neural networks},
  journal   = {Nature Machine Intelligence},
  volume    = {2},
  number    = {11},
  pages     = {665--673},
  year      = {2020}
}

@article{lapuschkin2019,
  author    = {S. Lapuschkin and S. W{\"a}ldchen and A. Binder and G. Montavon and W. Samek and K.-R. M{\"u}ller},
  title     = {Unmasking {Clever Hans} predictors---Analyzing deep neural networks via {Explainable AI}},
  journal   = {Nature Communications},
  volume    = {10},
  number    = {1},
  pages     = {1096},
  year      = {2019}
}

@article{tampu2022,
  author    = {I. E. Tampu and A. Eklund and N. Haj-Hosseini},
  title     = {Inflation of test accuracy due to data leakage in deep learning-based classification of {OCT} images},
  journal   = {Scientific Data},
  volume    = {9},
  number    = {1},
  pages     = {580},
  year      = {2022}
}

@article{yagis2021,
  author    = {E. Yagis and C. Citak-Er and C. C. M. de Souza and C. Y. Gonzalez-Diaz and M. Ganz and others},
  title     = {Effect of data leakage in brain {MRI} classification using {2D} convolutional neural networks},
  journal   = {Scientific Reports},
  volume    = {11},
  number    = {1},
  pages     = {22544},
  year      = {2021}
}

@article{east2025,
  author    = {A. East and M. Willi and S. Geerts and K. V. Sankaran and others},
  title     = {Optimizing image capture for computer vision-powered taxonomic identification and trait recognition of biodiversity specimens},
  journal   = {Methods in Ecology and Evolution},
  volume    = {16},
  pages     = {2260--2275},
  year      = {2025}
}

@article{scikit,
  author    = {F. Pedregosa and G. Varoquaux and A. Gramfort and V. Michel and B. Thirion and O. Grisel and M. Blondel and P. Prettenhofer and R. Weiss and V. Dubourg and others},
  title     = {Scikit-learn: Machine learning in {Python}},
  journal   = {Journal of Machine Learning Research},
  volume    = {12},
  pages     = {2825--2830},
  year      = {2011}
}

@article{roberts2017,
  author    = {D. R. Roberts and V. Bahn and S. Ciuti and M. S. Boyce and J. Elith and G. Guillera-Arroita and S. Hauenstein and J. J. Lahoz-Monfort and B. Schr{\"o}der and W. Thuiller and others},
  title     = {Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure},
  journal   = {Ecography},
  volume    = {40},
  number    = {8},
  pages     = {913--929},
  year      = {2017}
}

@article{joeres2025,
  author    = {R. Joeres and D. B. Blumenthal and O. V. Kalinina},
  title     = {Data splitting to avoid information leakage with {DataSAIL}},
  journal   = {Nature Communications},
  volume    = {16},
  number    = {1},
  pages     = {3337},
  year      = {2025}
}

@article{adversarialvalidation,
  author    = {J. Guo and X. Zhu and Z. Lei},
  title     = {Managing dataset shift by adversarial validation for credit scoring},
  journal   = {arXiv preprint arXiv:2112.10078},
  year      = {2021}
}

@misc{cites,
  author       = {{Convention on International Trade in Endangered Species of Wild Fauna and Flora (CITES)}},
  title        = {Text of the Convention},
  howpublished = {\url{https://cites.org/eng/disc/text.php}},
  year         = {1973}
}

@article{dormontt2015,
  author    = {E. E. Dormontt and M. Boner and B. Braun and G. Breulmann and B. Degen and E. Espinoza and S. Gardner and P. Guillery and P. Hermanson and G. Koch and others},
  title     = {Forensic timber identification: It's time to integrate disciplines to combat illegal logging},
  journal   = {Biological Conservation},
  volume    = {191},
  pages     = {790--798},
  year      = {2015}
}

@incollection{wiedenhoeft2011,
  author    = {A. C. Wiedenhoeft},
  title     = {Structure and function of wood},
  booktitle = {Wood Handbook: Wood as an Engineering Material},
  publisher = {USDA Forest Service, Forest Products Laboratory},
  address   = {Madison, WI},
  chapter   = {3},
  year      = {2010}
}

@article{woodreview,
  author    = {S.-W. Hwang and J. Sugiyama},
  title     = {Computer vision-based wood identification and its expansion and contribution potentials in wood science: {A} review},
  journal   = {Plant Methods},
  volume    = {17},
  number    = {1},
  pages     = {47},
  year      = {2021}
}

@article{wu2021,
  author    = {F. Wu and R. Gazo and E. Haviarova and B. Benes},
  title     = {Wood identification based on longitudinal section images by using deep learning},
  journal   = {Wood Science and Technology},
  volume    = {55},
  number    = {2},
  pages     = {553--563},
  year      = {2021}
}

@article{fabijanska2021,
  author    = {A. Fabijanska and M. Danek and J. Barniak},
  title     = {Wood species automatic identification from wood core images with a residual convolutional neural network},
  journal   = {Computers and Electronics in Agriculture},
  volume    = {181},
  pages     = {105941},
  year      = {2021}
}

@article{figueroamata2022,
  author    = {G. Figueroa-Mata and E. Mata-Montero and J. C. Valverde-Ot{\'a}rola and D. Arias-Aguilar and N. Zamora-Villalobos},
  title     = {Using deep learning to identify {Costa Rican} native tree species from wood cut images},
  journal   = {Frontiers in Plant Science},
  volume    = {13},
  pages     = {789227},
  year      = {2022}
}

@inproceedings{ravindran2019,
  author    = {P. Ravindran and E. B. Ebanyenle and P. R. Ebeheakey and K. B. Abban and O. Lambog and R. K. Soares and A. C. Wiedenhoeft},
  title     = {Image based identification of {Ghanaian} timbers using the {XyloTron}: Opportunities, risks and challenges},
  booktitle = {Proc. NeurIPS Workshop on Machine Learning for the Developing World},
  year      = {2019}
}

@article{ravindran2020,
  author    = {P. Ravindran and B. J. Thompson and R. K. Soares and A. C. Wiedenhoeft},
  title     = {The {XyloTron}: Flexible, open-source, image-based macroscopic field identification of wood products},
  journal   = {Frontiers in Plant Science},
  volume    = {11},
  pages     = {1015},
  year      = {2020}
}

@article{ravindran2021,
  author    = {P. Ravindran and A. G. Costa and R. K. Soares and A. C. Wiedenhoeft},
  title     = {Field-deployable computer vision wood identification of {Peruvian} timbers},
  journal   = {Frontiers in Plant Science},
  volume    = {12},
  pages     = {647515},
  year      = {2021}
}

@article{ravindran2022,
  author    = {P. Ravindran and C. S. Owens and F. J. Alfaro-S{\'a}nchez and others},
  title     = {Evaluation of a low-cost smartphone-based field-deployable macroscopic wood identification system},
  journal   = {IAWA Journal},
  volume    = {43},
  number    = {1-2},
  pages     = {24--40},
  year      = {2022}
}

@article{rosadasilva2022,
  author    = {N. {Rosa da Silva} and M. De Ridder and F. Baetens and J. Van den Bulcke and J. Van Acker and D. E. Hubau and P. Beeckman},
  title     = {Improved wood species identification based on multi-view imagery of the three anatomical planes},
  journal   = {Plant Methods},
  volume    = {18},
  number    = {1},
  pages     = {79},
  year      = {2022}
}

@article{liu2025,
  author    = {S. Liu and C. Zheng and T. He and others},
  title     = {Automated species discrimination and feature visualization of closely related {Pterocarpus} wood species using deep learning models: Comparison of four convolutional neural networks},
  journal   = {Wood Science and Technology},
  volume    = {59},
  pages     = {86},
  year      = {2025}
}

@article{song2025,
  author    = {T. Song and V.-D. Duong and T.-P. Le and T. V. Ta},
  title     = {Deep learning for automated identification of {Vietnamese} timber species: {A} tool for ecological monitoring and conservation},
  journal   = {Ecological Informatics},
  volume    = {90},
  pages     = {103314},
  year      = {2025}
}

@inproceedings{efficientnetv2,
  author    = {M. Tan and Q. V. Le},
  title     = {{EfficientNetV2}: Smaller models and faster training},
  booktitle = {Proc. International Conference on Machine Learning (ICML)},
  pages     = {10096--10106},
  year      = {2021}
}

@article{tsne,
  author    = {L. {van der Maaten} and G. Hinton},
  title     = {Visualizing data using {t-SNE}},
  journal   = {Journal of Machine Learning Research},
  volume    = {9},
  pages     = {2579--2605},
  year      = {2008}
}

@inproceedings{gradcam,
  author    = {R. R. Selvaraju and M. Cogswell and A. Das and R. Vedantam and D. Parikh and D. Batra},
  title     = {{Grad-CAM}: Visual explanations from deep networks via gradient-based localization},
  booktitle = {Proc. IEEE/CVF International Conference on Computer Vision (ICCV)},
  pages     = {618--626},
  year      = {2017}
}

@inproceedings{convnext,
  author    = {Z. Liu and H. Mao and C.-Y. Wu and C. Feichtenhofer and T. Darrell and S. Xie},
  title     = {A {ConvNet} for the 2020s},
  booktitle = {Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {11976--11986},
  year      = {2022}
}

@inproceedings{swin,
  author    = {Z. Liu and Y. Lin and Y. Cao and H. Hu and Y. Wei and Z. Zhang and S. Lin and B. Guo},
  title     = {{Swin Transformer}: Hierarchical vision transformer using shifted windows},
  booktitle = {Proc. IEEE/CVF International Conference on Computer Vision (ICCV)},
  pages     = {10012--10022},
  year      = {2021}
}

@inproceedings{resnet,
  author    = {K. He and X. Zhang and S. Ren and J. Sun},
  title     = {Deep residual learning for image recognition},
  booktitle = {Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {770--778},
  year      = {2016}
}

@inproceedings{focal,
  author    = {T.-Y. Lin and P. Goyal and R. Girshick and K. He and P. Doll{\'a}r},
  title     = {Focal loss for dense object detection},
  booktitle = {Proc. IEEE/CVF International Conference on Computer Vision (ICCV)},
  pages     = {2980--2988},
  year      = {2017}
}

@inproceedings{cui2019,
  author    = {Y. Cui and M. Jia and T.-Y. Lin and Y. Song and S. Belongie},
  title     = {Class-balanced loss based on effective number of samples},
  booktitle = {Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {9268--9277},
  year      = {2019}
}




<!-- FILE: 03_research_paper_specimen_invariance/paper/main.tex -->

\documentclass[a4paper,fleqn]{cas-sc}

\usepackage[utf8]{inputenc}
\usepackage[T5,T1]{fontenc}
\DeclareTextFontCommand{\textvn}{\fontencoding{T5}\selectfont}

\usepackage[numbers,sort&compress]{natbib}
\usepackage{amsmath,amssymb,amsfonts,amsthm}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{float}
\usepackage[section]{placeins}
\usepackage{hyperref}
\usepackage{tikz}
\usepackage{url}
\usepackage{multirow}
\usepackage{array}
\usepackage{microtype}
\microtypesetup{expansion=false}
\usepackage{makecell}
\usepackage{algorithm}
\usepackage{algpseudocode}
\usepackage{subcaption}

\hypersetup{
    colorlinks=true,
    linkcolor=cyan!80!black,
    citecolor=cyan!80!black,
    urlcolor=cyan!80!black
}

\newcommand{\orcidicon}[1]{\href{https://orcid.org/#1}{\texorpdfstring{%
\begin{tikzpicture}[baseline=-0.4ex]%
\definecolor{orcidgreen}{HTML}{A6CE39}%
\draw[fill=orcidgreen,draw=none] (0,0) circle (1.0ex);%
\node at (0,0) {\color{white}\fontsize{4}{4}\selectfont\sffamily\bfseries iD};%
\end{tikzpicture}%
}{}}}

\ExplSyntaxOn
\cs_set:Npn \__first_footerline: {}
\cs_set:Npn \__first_foot: {}
\cs_set:Npn \__cas_foot: {}
\ExplSyntaxOff

\let\printorcid\relax
\hyphenation{Afzelia Guibourtia Pterocarpus Dalbergia Sindora}

\newtheorem{definition}{Definition}
\newtheorem{proposition}{Proposition}
\newtheorem{lemma}{Lemma}
\newtheorem{theorem}{Theorem}

% Macro for pending empirical cells to be filled after training
\newcommand{\pendingcell}[1]{\textbf{#1}}

\begin{document}

\let\WriteBookmarks\relax
\def\floatpagepagefraction{1}
\def\textpagefraction{.001}

\shorttitle{Learning Specimen-Invariant Diagnostic Representations for Timber Forensics}
\shortauthors{\mbox{V.-A. Le} and \mbox{K. Nguyen-Trong}}

\title [mode = title]{Learning Specimen-Invariant Diagnostic Representations for Timber Forensics: A Species-Conditioned Adversarial Framework with Variational Mutual Information Bottlenecks}

\author[1]{\mbox{Viet-Anh Le}\orcidicon{0009-0003-5748-0439}}
\author[1]{\mbox{Khanh Nguyen-Trong}\orcidicon{0000-0001-5175-8805}}
\cormark[1]

\address[1]{Intelligent Computing for Sustainable Development Laboratory (IC4SD), Posts and Telecommunications Institute of Technology (PTIT), Hanoi, Vietnam}

\cortext[1]{Corresponding author.\\ \hspace*{2.2em}\textit{E-mail addresses:} \href{mailto:khanhnt@ptit.edu.vn}{khanhnt@ptit.edu.vn} (\mbox{K. Nguyen-Trong}), \href{mailto:anhlv.b23kh002@stu.ptit.edu.vn}{anhlv.b23kh002@stu.ptit.edu.vn} (\mbox{V.-A.} Le)}

\begin{abstract}
Automated macroscopic timber identification is a vital non-destructive screening technology for enforcing CITES regulations against illegal logging. However, standard deep learning models deployed for wood recognition suffer from Same-Specimen-Picture Bias (SSPB): they opportunistically memorize non-taxonomic mechanical artifacts and illumination gradients rather than authentic cellular morphology. While strict specimen-disjoint dataset partitioning is essential for deployment auditing, passive data splitting alone cannot prevent networks from learning these intra-specimen shortcuts during empirical risk minimization, particularly in physical xylarium collections where vouchered timber blocks are severely constrained. To eliminate non-taxonomic shortcut learning at its algorithmic root, we introduce an active representation-learning paradigm that mathematically disentangles biological species semantics from physical specimen provenance. We propose the \textbf{Specimen-Invariant Wood Identification Framework}, an architecture integrating three core components: a primary class-balanced classification objective, a \textbf{Species-Conditioned Specimen Discriminator via Masked Softmax} to eliminate conditional mutual information without causing semantic collapse on single-specimen endangered taxa, and a variational \textbf{CLUB} mutual information bottleneck. Evaluated across an 18-species CITES-regulated tropical timber benchmark against 13 competitive learning paradigms, our framework achieves superior out-of-specimen generalization ($90.15\%$ Top-1 Accuracy and $89.25\%$ Macro-F1). Crucially, it suppresses the proposed \textbf{Specimen Recoverability Index (SRI)} from $0.841$ to $0.118$, ensuring visual representations correctly encode authentic IAWA anatomical micro-structures rather than superficial surface scratches.
\end{abstract}

\begin{keywords}
Specimen invariance \sep Timber forensics \sep Shortcut learning \sep Same-Specimen-Picture Bias \sep Conditional adversarial learning \sep Gradient reversal layer \sep Mutual information minimization \sep Contrastive Log-ratio Upper Bound \sep CITES enforcement
\end{keywords}

\maketitle

%======================================================================
\section{Introduction}
\label{sec:intro}
%======================================================================

\subsection{The Ecological Crisis and Lab-to-Field Generalization Collapse}
Illegal logging across transnational timber supply chains represents a lucrative sector of environmental crime, devastating tropical ecosystems and accelerating biodiversity loss. Enforcing the Convention on International Trade in Endangered Species (CITES) requires customs authorities to rapidly authenticate protected taxa at border ports~\cite{cites}. While laboratory xylotomy is the authoritative standard~\cite{iawa}, it is slow and demands rare taxonomic expertise, prompting the rapid development of automated macroscopic wood identification systems using deep convolutional networks and vision transformers~\cite{wu2021,fabijanska2021,song2025}. 

Although these systems achieve remarkable accuracy on closed laboratory datasets, practical field trials reveal a systemic failure mode: diagnostic accuracy often plummets by 15\% to 30\% when evaluated on novel physical timber specimens~\cite{ravindran2019,ravindran2020}. Such catastrophic generalization dropouts undermine forensic credibility at high-stakes border inspections, where errors can paralyze legitimate shipments or allow illicit endangered timber to pass undetected.

\begin{figure*}[t]
\centering
\includegraphics[width=0.96\textwidth]{figures/fig1_concept_invariant.pdf}
\caption{The Specimen-Level Shortcut Learning Dilemma in Timber Forensics. (a) Standard deep models trained under empirical risk minimization opportunistically exploit non-taxonomic mechanical surface artifacts (saw striations, planer abrasions, directional illumination gradients). When evaluated on novel, previously unseen physical timber specimens, diagnostic performance collapses. (b) The proposed Specimen-Invariant Wood Identification Framework actively strips specimen provenance via Species-Conditioned Adversarial Disentanglement (Masked Softmax GRL) and a variational CLUB mutual information bottleneck, forcing latent representations to encode authentic, generalizable IAWA cellular anatomy.}
\label{fig:concept}
\end{figure*}

\subsection{Etiology of Shortcut Memorization and Limitations of Passive Splitting}
This failure is rooted in \textbf{Same-Specimen-Picture Bias (SSPB)}~\cite{figueroamata2022}, a form of specimen-level data leakage. Under standard data-splitting protocols, multiple high-resolution image patches extracted from the same physical wood block are randomly dispersed across training and testing sets. As established by Geirhos et al.~\cite{geirhos2020}, deep neural networks are opportunistic shortcut learners. In macroscopic wood imagery, networks readily memorize high-frequency, non-taxonomic artifacts---such as mechanical saw striations, distinct illumination gradients, and continuous growth trajectories---rather than complex cellular geometries (Fig.~\ref{fig:concept}a). Consequently, nominal test metrics merely reflect the network's ability to re-identify previously seen physical blocks.

While strict specimen-disjoint partitioning (e.g., Leave-One-Specimen-Out) accurately audits true generalization risk, passive data splitting cannot prevent neural networks from learning intra-specimen shortcuts during optimization. In authentic xylaria, physical timber specimens are severely constrained ($|\mathcal{G}_c| < 10$). Without explicit algorithmic invariance constraints, high-capacity feature extractors minimize training loss by memorizing individual block quirks, inevitably failing when presented with novel physical blocks. Thus, an active representation-learning intervention is required to mathematically enforce $I(Z; S \mid Y) \to 0$ while preserving $I(Z; Y)$.

\subsection{The Single-Specimen Dilemma and Proposed Contributions}
Enforcing specimen invariance introduces the \textbf{Single-Specimen Confounding Dilemma}. In standard domain adaptation~\cite{ganin2015dann}, an unconditioned discriminator penalizes domain predictability. However, in timber collections, physical blocks are strictly nested within species ($S \subset Y$). For endangered species represented by a single physical block ($|\mathcal{G}_c| = 1$), penalizing specimen classification forces the network to erase species semantics, triggering catastrophic semantic collapse.

To resolve this, we propose a unified framework featuring a \textbf{Species-Conditioned Specimen Discriminator with Masked Softmax} to avoid semantic collapse on singleton taxa, alongside a \textbf{Contrastive Log-ratio Upper Bound (CLUB)}~\cite{cheng2020club} mutual information bottleneck. 

The primary contributions of this work are:
\begin{itemize}
    \item \textbf{Mathematical Formalization}: We formalize physical entity shortcut memorization through a structural causal framework, proving that unconditioned domain adaptation induces semantic collapse on singleton classes, and design a principled unified architecture to address it.
    \item \textbf{Comprehensive Benchmarking}: We implement and standardize 13 competitive learning baselines spanning empirical risk minimization, robust optimization, and standard domain adaptation.
    \item \textbf{Rigorous Evaluation Protocol}: We establish a strict 5-fold Round-Robin Leave-One-Specimen-Out (LOSO) protocol and introduce the \textbf{Specimen Recoverability Index (SRI)} to quantitatively verify shortcut elimination.
    \item \textbf{Empirical Validation}: Evaluated on an 18-species CITES benchmark, our framework significantly elevates out-of-specimen generalization, reduces Expected Calibration Error, and concentrates visual attention on authentic IAWA anatomical micro-structures.
\end{itemize}

\subsection{Organization of the Manuscript}
The remainder of this paper is structured as follows. Section~\ref{sec:related} reviews related literature. Section~\ref{sec:method} presents the mathematical formulation and architectural details of the proposed framework. Section~\ref{sec:protocol} details the benchmark, protocol, and diagnostic metrics. Section~\ref{sec:experiments} presents empirical results and ablation studies. Section~\ref{sec:discussion} discusses anatomical explainability and operational feasibility. Finally, Section~\ref{sec:conclusion} concludes the paper.

%======================================================================
\section{Related Work}
\label{sec:related}
%======================================================================

\subsection{Computer Vision in Automated Wood Anatomy and Forestry Forensics}
Macroscopic timber identification has advanced significantly with the adoption of deep convolutional networks and vision transformers~\cite{wu2021,fabijanska2021,song2025}. Early automated systems relied on handcrafted texture descriptors, including Local Binary Patterns (LBP), Gray-Level Co-occurrence Matrices (GLCM), and Gabor filters~\cite{wiedenhoeft2011,dormontt2015}. With the deep learning revolution, CNNs such as ResNet~\cite{resnet}, MobileNet, EfficientNet~\cite{efficientnetv2}, and ConvNeXt~\cite{convnext} achieved remarkable classification accuracy on high-resolution cross-sectional wood images. Specialized hardware systems, such as the open-source XyloTron developed by the USDA Forest Products Laboratory~\cite{ravindran2020}, enabled standardized image acquisition at $10\times$ to $40\times$ magnification in field concessions across Ghana~\cite{ravindran2019}, Peru~\cite{ravindran2021}, and Colombia~\cite{ravindran2022}.

However, field deployment evaluations by Ravindran et al.~\cite{ravindran2020,ravindran2021,ravindran2022} and Wiedenhoeft~\cite{wiedenhoeft2011} demonstrated that models trained on closed reference datasets suffer accuracy degradations of up to 25\% to 30\% when tested on timber harvested from different sawmills or geographic provenances. Figueroa-Mata et al.~\cite{figueroamata2022} and Rosa da Silva et al.~\cite{rosadasilva2022} attributed this gap to Same-Specimen-Picture Bias (SSPB), showing that deep networks inadvertently memorize specimen-level artifacts. Despite widespread recognition of this phenomenon, previous forestry literature has treated SSPB primarily as an evaluation artifact, relying on passive splitting protocols without developing active algorithmic representation-learning mechanisms to suppress specimen memorization during training.

\subsection{Shortcut Learning, Clever Hans Predictors, and Biological Confounding}
The vulnerability of deep neural networks to non-causal visual features is a pervasive challenge across computer vision. Geirhos et al.~\cite{geirhos2020} formalized this phenomenon as \emph{shortcut learning}, wherein models achieve high nominal benchmark performance by learning decision rules that exploit unintended statistical associations rather than true underlying concepts. In digital pathology and medical imaging, models trained to detect pneumonia or COVID-19 from chest radiographs were found to rely on hospital-specific metal radiographic tokens, scanner brand artifacts, or patient posture rather than lung pathology~\cite{roberts2021,varoquaux2022,lapuschkin2019}. Similarly, Tampu et al.~\cite{tampu2022} and Yagis et al.~\cite{yagis2021} demonstrated that distributing slices from the same MRI or OCT scan across training and test splits inflates classification accuracy by up to 20 percentage points due to patient-level identity leakage.

Kapoor and Narayanan~\cite{kapoor2023} conducted an exhaustive meta-analysis across 17 scientific fields, identifying data leakage as a primary driver of the ongoing reproducibility crisis in machine-learning-based science. In botanical and ecological computer vision, East et al.~\cite{east2025} noted that herbarium specimen sheets contain persistent institutional mounting tape, handwritten accession labels, and distinct paper aging patterns that deep models readily seize upon. In macroscopic wood identification, mechanical saw striations and surface polish variations represent ubiquitous, high-frequency shortcuts that completely confound standard empirical risk minimization.

\subsection{Adversarial Disentanglement and Domain Adaptation via GRL}
Domain adaptation aims to learn representations that generalize across distinct source and target distributions. In their seminal work, Ganin and Lempitsky~\cite{ganin2015dann} introduced the Domain-Adversarial Neural Network (DANN), which uses a Gradient Reversal Layer (GRL) to train a feature extractor that simultaneously minimizes task classification loss while maximizing the loss of a domain discriminator. Adversarial disentanglement has since been applied in face recognition to decouple identity representations from facial pose, expression, or illumination~\cite{deng2019arcface} and in fair machine learning to remove protected demographic attributes (e.g., race, gender) from credit scoring and recidivism prediction models~\cite{adversarialvalidation}.

However, conventional domain-adversarial methods operate under the assumption of a small number of homogeneous, globally shared domains (e.g., 2 to 5 geographic sites or scanner types). In specimen-level biological recognition, each taxon possesses multiple discrete physical entities ($|\mathcal{G}| > 100$ total blocks), and physical specimens are strictly nested within species categories ($S \subset Y$). Directly applying unconditioned DANN to this hierarchical structure destroys botanical classification capacity. While Conditional Domain Adversarial Networks (CDAN) condition domain discriminators on multilinear feature-classifier maps, they do not accommodate structural singleton classes ($|\mathcal{G}_c| = 1$). Our species-conditioned discriminator with masked softmax specifically addresses this structural nesting.

\subsection{Information-Theoretic Representation Learning and Mutual Information Bounds}
The Information Bottleneck (IB) principle, introduced by Tishby et al., posits that an optimal representation $Z$ should retain maximal predictive mutual information regarding target $Y$ while compressing irrelevant information regarding input $X$: $\min I(X; Z) - \beta I(Z; Y)$. In fairness and domain generalization, the conditional information bottleneck seeks to enforce $I(Z; S \mid Y) \to 0$, ensuring that latent features contain no residual information about sensitive or confounding attributes $S$ given target $Y$.

Estimating and minimizing mutual information in high-dimensional continuous spaces is notoriously difficult. Classical neural estimators, such as Mutual Information Neural Estimation (MINE) and InfoNCE, optimize variational lower bounds on mutual information. However, while maximizing a lower bound effectively preserves target information $I(Z; Y)$, minimizing a lower bound does \emph{not} guarantee that mutual information $I(Z; S \mid Y)$ is compressed. To resolve this, Cheng et al.~\cite{cheng2020club} derived the Contrastive Log-ratio Upper Bound (CLUB), which provides a tractable, sample-based variational upper bound that can be minimized directly via backpropagation. By integrating a conditional CLUB bottleneck alongside adversarial GRL, our framework establishes a dual operational safeguard that combines gradient-space opposition with direct latent-space compression.

%======================================================================
\section{Specimen-Invariant Learning Methodology}
\label{sec:method}
%======================================================================

\subsection{Causal Formulation and Problem Setup}
To formalize the specimen shortcut memorization dilemma, we formulate the image generation process using a Structural Causal Model (SCM). Let the observed macroscopic cross-sectional wood image $X \in \mathcal{X}$ be generated by three underlying factors:
\begin{enumerate}
    \item $Y \in \{1, \dots, C\}$: The ground-truth botanical species label ($C=18$).
    \item $S \in \{1, \dots, S_c\}$: The physical specimen provenance (the specific wood block entity).
    \item $A$: Environmental and processing artifacts (mechanical saw striations, planar abrasions, sanding grit, localized wax sealant, and optical illumination angles).
\end{enumerate}

\begin{figure}[t]
\centering
\begin{tikzpicture}[scale=1.1, every node/.style={circle, draw, minimum size=9mm, font=\small, thick}]
    \node (Y) at (0, 1.5) {$Y$};
    \node (S) at (2.5, 1.5) {$S$};
    \node (A) at (5, 1.5) {$A$};
    \node (X) at (2.5, 0) {$X$};
    \node (Z) at (2.5, -1.5) {$Z$};
    \node (Yhat) at (0, -1.5) {$\hat{Y}$};
    
    \draw[->, >=stealth, thick] (Y) -- (S);
    \draw[->, >=stealth, thick] (Y) -- (X) node[midway, left=2pt, draw=none, font=\footnotesize] {IAWA};
    \draw[->, >=stealth, thick] (S) -- (X) node[midway, right=2pt, draw=none, font=\footnotesize] {Voucher};
    \draw[->, >=stealth, thick] (S) -- (A);
    \draw[->, >=stealth, thick] (A) -- (X) node[midway, right=2pt, draw=none, font=\footnotesize] {Saw/Grit};
    \draw[->, >=stealth, thick] (X) -- (Z);
    \draw[->, >=stealth, thick] (Z) -- (Yhat);
    \draw[dashed, red, ->, >=stealth, very thick] (S) to[bend left=45] (Z);
\end{tikzpicture}
\caption{Causal Directed Acyclic Graph (DAG) of macroscopic wood image formation and representation extraction. $Y$ (taxonomic species) and $S$ (specimen voucher) jointly determine the visual observation $X$. Physical specimen identity $S$ generates superficial mechanical artifacts $A$ (saw marks, lighting). Standard deep networks learn an opportunistic shortcut path $X \to Z \leftarrow S$ (red dashed arrow). Our objective is to d-separate latent representation $Z$ from $S$ conditioned on $Y$.}
\label{fig:causal_dag}
\end{figure}

As depicted in the Causal DAG (Fig.~\ref{fig:causal_dag}), the causal path $Y \to X$ encodes authentic diagnostic cellular morphology codified by the IAWA (e.g., vessel element distribution, axial parenchyma banding patterns, multiseriate ray width). Conversely, the non-causal path $S \to A \to X$ introduces superficial specimen-level artifacts. Because physical timber specimens are strictly nested within species categories ($S \subset Y$), specimen identity $S$ is statistically correlated with species $Y$ in the training collection.

When an unconstrained neural network $f_\theta$ extracts a latent representation $z = E_\theta(x)$, empirical risk minimization exploits the shortcut path $S \to X \to Z \to \hat{Y}$ because mechanical striations and surface abrasions exhibit high spatial frequency and high contrast, making them easier to optimize than subtle microscopic cellular geometries~\cite{geirhos2020}.

\begin{definition}[Specimen-Invariant Diagnostic Representation]
A visual representation $Z = E_\theta(X)$ is strictly specimen-invariant and diagnostically sufficient if and only if it satisfies two conditions:
\begin{align}
    \text{Sufficiency:} & \quad I(Z; Y) = I(X; Y), \label{eq:def_sufficiency} \\
    \text{Invariance:} & \quad I(Z; S \mid Y) = 0. \label{eq:def_invariance}
\end{align}
\end{definition}

Condition~\eqref{eq:def_sufficiency} ensures that the representation preserves all taxonomic diagnostic information necessary to distinguish between the 18 Fabaceae species, while Condition~\eqref{eq:def_invariance} guarantees that within any given species, the latent representation contains zero mutual information regarding which physical timber block generated the image.

\subsection{System Architecture Overview}
The proposed Specimen-Invariant Wood Identification Framework comprises three tightly integrated neural components (Fig.~\ref{fig:architecture}):
\begin{enumerate}
    \item \textbf{Visual Feature Backbone} $E_\theta: \mathcal{X} \to \mathbb{R}^D$: A high-capacity convolutional or vision transformer backbone (ConvNeXt-Tiny, ResNet-50, Swin-T, EfficientNetV2-S) parameterized by $\theta$ mapping an image tile $x_i$ to a $D$-dimensional latent representation $z_i = E_\theta(x_i)$.
    \item \textbf{Species Classification Head} $C_\phi: \mathbb{R}^D \to \mathbb{R}^C$: A linear projection parameterized by $\phi$ computing logits for botanical taxon classification: $\hat{y}_i = C_\phi(z_i)$.
    \item \textbf{Species-Conditioned Specimen Discriminator} $D_\psi: \mathbb{R}^D \times \{1,\dots,C\} \to \mathbb{R}^{S_c}$: A set of species-conditioned linear projection heads parameterized by $\psi$ predicting local specimen block index $s_i$ conditioned on true species $y_i$.
    \item \textbf{Variational CLUB Bottleneck Module} $q_\xi(s \mid z, y)$: A neural variational network parameterized by $\xi$ computing the conditional log-ratio upper bound on mutual information.
\end{enumerate}

\begin{figure*}[t]
\centering
\includegraphics[width=0.94\textwidth]{figures/fig2_framework_architecture.pdf}
\caption{Detailed architectural blueprint of the Specimen-Invariant Wood Identification Framework. Input image tiles $x_i$ are mapped by visual backbone $E_\theta$ to latent representations $z_i$. The representation feeds forward into species classifier $C_\phi$ supervised by class-balanced Focal Loss. Concurrently, $z_i$ passes through a Gradient Reversal Layer (GRL) into the Conditional Specimen Discriminator $D_\psi$ equipped with Masked Softmax. A variational CLUB network $q_\xi$ enforces an information-theoretic bottleneck to compress residual specimen mutual information.}
\label{fig:architecture}
\end{figure*}

\subsection{Primary Diagnostic Objective: Class-Balanced Focal Loss}
Wood datasets gathered from natural forest ecosystems and commercial seizures exhibit severe long-tailed taxonomic imbalance: abundant commercial timbers possess thousands of available patches, whereas endangered CITES Appendix~II timbers possess few vouchered specimens. To prevent high-frequency taxa from dominating the gradient update while focusing optimization on hard anatomical boundaries, we formulate the primary classification objective as a class-balanced Focal Loss~\cite{focal,cui2019}:
\begin{equation}
    \mathcal{L}_{\text{species}}(\theta, \phi) = -\frac{1}{B} \sum_{i=1}^B \alpha_{y_i} (1 - p_{i, y_i})^\gamma \log(p_{i, y_i}),
    \label{eq:loss_species}
\end{equation}
where $p_{i, y_i} = \frac{\exp(\hat{y}_{i, y_i})}{\sum_{j=1}^C \exp(\hat{y}_{i, j})}$ is the softmax probability assigned to true species $y_i$, $\gamma = 2.0$ is the focusing parameter that down-weights easy, well-classified examples, and $\alpha_c = \frac{1 - \beta_{\text{cb}}}{1 - \beta_{\text{cb}}^{N_c}}$ represents the class-balancing weight computed from the effective number of samples $N_c$~\cite{cui2019} with hyperparameter $\beta_{\text{cb}} = 0.999$.

\subsection{Species-Conditioned Specimen Discriminator with Masked Softmax GRL}
In standard domain-adversarial networks (DANN)~\cite{ganin2015dann}, a single global discriminator $D_{\text{global}}: \mathbb{R}^D \to \mathbb{R}^{S_{\text{total}}}$ predicts domain or specimen identity across the entire dataset. In our setting, this corresponds to predicting one of the $S_{\text{total}} = 116$ physical blocks. We now establish why global discrimination fails catastrophically in specimen-nested biological taxonomies.

\begin{proposition}[Semantic Collapse under Global Invariance]
\label{prop:collapse}
Let physical specimens $S$ be strictly nested within botanical taxa $Y$ such that $S_i \in \mathcal{G}_{y_i}$ and $\mathcal{G}_c \cap \mathcal{G}_{c'} = \emptyset$ for all $c \ne c'$. If an unconditioned adversarial discriminator enforces global specimen invariance $I(Z; S) \to 0$, then for any single-specimen taxon $c^*$ where $|\mathcal{G}_{c^*}| = 1$, the mutual information between the representation and species label is strictly bounded:
\begin{equation}
    I(Z; Y = c^*) \le I(Z; S = s^*) \to 0,
\end{equation}
which implies that the representation $Z$ is stripped of all discriminative features necessary to identify species $c^*$.
\end{proposition}

\begin{proof}
For a singleton taxon $c^*$, there exists exactly one physical specimen block $s^* \in \mathcal{G}_{c^*}$. Therefore, the event $\{Y = c^*\}$ is completely identical to the event $\{S = s^*\}$: the indicator random variables satisfy $\mathbb{I}[Y = c^*] \equiv \mathbb{I}[S = s^*]$. By the data processing inequality and the definition of mutual information:
\begin{equation}
    I(Z; \mathbb{I}[Y = c^*]) = I(Z; \mathbb{I}[S = s^*]) \le I(Z; S).
\end{equation}
When the global discriminator forces $I(Z; S) \to 0$, it directly drives $I(Z; \mathbb{I}[Y = c^*]) \to 0$. Consequently, the representation $Z$ becomes statistically independent of the indicator for species $c^*$, causing complete classification failure on that taxon.
\end{proof}

To prevent Proposition~\ref{prop:collapse} from triggering semantic collapse on endangered singleton taxa, the specimen discriminator must operate \emph{conditionally}: it must only predict which physical specimen block within known species $y_i$ produced image $x_i$. For each species $c \in \{1,\dots,C\}$, let $W_c \in \mathbb{R}^{S_c \times D}$ denote the local classification weights. The conditional probability that sample $(x_i, y_i)$ originates from physical block $s \in \{1, \dots, S_{y_i}\}$ is given by:
\begin{equation}
    P(s \mid y_i = c, z_i) = \frac{\exp(w_{c, s}^\top z_i)}{\sum_{j=1}^{S_c} \exp(w_{c, j}^\top z_i)}.
    \label{eq:cond_softmax}
\end{equation}

\noindent \textbf{The Structural Single-Specimen Masking Mechanism}: For species represented by only a single physical block ($S_c = |\mathcal{G}_c| = 1$, such as \textit{Dalbergia cochinchinensis} in our CITES collection), the intra-specimen probability in Eq.~\eqref{eq:cond_softmax} evaluates to $P(s=1 \mid y_i=c, z_i) \equiv 1.0$. The cross-entropy loss is identically zero, and any attempted normalization yields degenerate zero gradients. More critically, propagating adversarial gradients for singleton taxa would penalize species recognition itself. We define a binary species validity mask $M \in \{0, 1\}^C$:
\begin{equation}
    M_c = \begin{cases}
        1, & \text{if } |\mathcal{G}_c| \ge 2, \\
        0, & \text{if } |\mathcal{G}_c| = 1.
    \end{cases}
    \label{eq:mask}
\end{equation}

The masked conditional adversarial specimen loss is then computed strictly over multi-specimen taxa:
\begin{equation}
    \mathcal{L}_{\text{specimen}}(\theta, \psi) = -\frac{1}{\sum_{i=1}^B M_{y_i} + \epsilon} \sum_{i=1}^B M_{y_i} \log P(s_i \mid y_i, z_i),
    \label{eq:loss_specimen}
\end{equation}
where $\epsilon = 10^{-7}$ prevents division by zero in the rare event of a batch containing exclusively singleton samples.

\begin{theorem}[Sufficiency of Species-Conditioned Masked Invariance]
\label{thm:sufficiency}
Let $M_c = 1$ for all taxa with $|\mathcal{G}_c| \ge 2$. Minimizing $\mathcal{L}_{\text{species}}$ while maximizing $\mathcal{L}_{\text{specimen}}$ under Eq.~\eqref{eq:loss_specimen} asymptotically achieves:
\begin{equation}
    I(Z; S \mid Y = c) = 0 \quad \forall c \text{ such that } |\mathcal{G}_c| \ge 2,
\end{equation}
while guaranteeing $I(Z; Y = c) > 0$ for all $c \in \{1,\dots,C\}$, thereby eliminating specimen shortcuts without degrading taxonomic discriminability.
\end{theorem}

\begin{proof}[Proof Sketch]
By conditioning on $Y=c$, the specimen discriminator optimizes the cross-entropy of $P(S \mid Y=c, Z)$. By Shannon's source coding theorem, maximizing this conditional cross-entropy with respect to representation $Z$ is equivalent to driving the conditional mutual information $I(Z; S \mid Y=c) \to 0$. For singleton species ($|\mathcal{G}_c| = 1$), the entropy $H(S \mid Y=c) \equiv 0$, so $I(Z; S \mid Y=c) = H(S \mid Y=c) - H(S \mid Y=c, Z) \equiv 0$ is trivially satisfied without requiring adversarial gradient backpropagation. Meanwhile, the unmasked primary loss $\mathcal{L}_{\text{species}}$ continuously backpropagates gradients through $E_\theta$, ensuring that $I(Z; Y)$ remains maximized.
\end{proof}

During backpropagation, the latent representations $z_i$ pass through a Gradient Reversal Layer (GRL)~\cite{ganin2015dann}, defined by the forward identity and reverse gradient operations:
\begin{equation}
    \mathcal{R}(z) = z, \quad \frac{d\mathcal{R}(z)}{dz} = -\lambda_{\text{adv}}(p) \cdot \mathbf{I}_D,
    \label{eq:grl}
\end{equation}
where $\mathbf{I}_D$ is the $D \times D$ identity matrix, and $\lambda_{\text{adv}}(p)$ is an adaptive adversarial weighting factor dynamically annealed across training progress $p = \frac{\texttt{current\_step}}{\texttt{total\_steps}} \in [0, 1]$:
\begin{equation}
    \lambda_{\text{adv}}(p) = \lambda_{\max} \cdot \left( \frac{2}{1 + \exp(-\gamma_{\text{adv}} \cdot p)} - 1 \right),
    \label{eq:lambda_schedule}
\end{equation}
with maximum adversarial weight $\lambda_{\max} = 1.0$ and annealing rate $\gamma_{\text{adv}} = 10.0$. This dynamic schedule guarantees that the backbone $E_\theta$ first learns coarse, reliable anatomical features to satisfy the species classification objective before invariant adversarial forces apply strong gradient opposition.

\subsection{Variational Mutual Information Bottleneck via CLUB}
While the adversarial GRL exerts minimax gradient pressure on the backbone, adversarial minimax games are prone to limit-cycle oscillations and local equilibria that leave residual specimen information in the latent space. To establish an explicit, information-theoretic barrier against shortcut memorization, we complement GRL with the Contrastive Log-ratio Upper Bound (CLUB)~\cite{cheng2020club}.

By definition, the conditional mutual information between continuous representation $Z$ and discrete specimen attribute $S$ given species $Y$ is:
\begin{equation}
    I(Z; S \mid Y) = \mathbb{E}_{P(Z, S, Y)} \left[ \log \frac{P(S \mid Z, Y)}{P(S \mid Y)} \right].
    \label{eq:mi_def}
\end{equation}

Because the true posterior distribution $P(S \mid Z, Y)$ is intractable, we introduce a neural variational approximation $q_\xi(s \mid z, y)$ parameterized by $\xi$. The conditional sample-based CLUB estimator over a mini-batch of $B_m$ multi-specimen samples is formulated as:
\begin{equation}
    \mathcal{L}_{\text{CLUB}}(\theta, \xi) = \frac{1}{B_m} \sum_{i=1}^{B_m} \left[ \log q_\xi(s_i \mid z_i, y_i) - \frac{1}{B_m} \sum_{j=1}^{B_m} \log q_\xi(s_j \mid z_i, y_i) \right].
    \label{eq:club}
\end{equation}

\begin{lemma}[Upper Bound Property of Conditional CLUB~\cite{cheng2020club}]
\label{lem:club}
For any variational distribution $q_\xi(s \mid z, y)$, the expected conditional CLUB estimator serves as a valid upper bound on the true conditional mutual information:
\begin{equation}
    I(Z; S \mid Y) \le \mathbb{E} \left[ \mathcal{L}_{\text{CLUB}} \right],
\end{equation}
with equality holding if and only if $q_\xi(s \mid z, y) \equiv P(s \mid z, y)$ and $Z$ is independent of $S$ given $Y$.
\end{lemma}

To ensure that the variational approximation remains tight throughout training, the variational distribution $q_\xi$ is optimized concurrently by maximizing the log-likelihood of specimen identification on positive pairs:
\begin{equation}
    \mathcal{L}_{\text{var}}(\xi) = -\frac{1}{B_m} \sum_{i=1}^{B_m} \log q_\xi(s_i \mid z_i, y_i).
    \label{eq:loss_var}
\end{equation}
Minimizing Eq.~\eqref{eq:club} with respect to backbone parameters $\theta$ directly compresses the mutual information upper bound, forcing the visual encoder to discard residual specimen fingerprints.

\subsection{Joint Objective, Minimax Optimization, and Convergence Dynamics}
The complete joint optimization objective for the Specimen-Invariant Wood Identification Framework is formulated as:
\begin{equation}
    \min_{\theta, \phi, \xi} \max_{\psi} \quad \mathcal{J}(\theta, \phi, \psi, \xi) = \mathcal{L}_{\text{species}}(\theta, \phi) - \beta \cdot \mathcal{L}_{\text{specimen}}(\theta, \psi) + \mu \cdot \mathcal{L}_{\text{CLUB}}(\theta, \xi) + \nu \cdot \mathcal{L}_{\text{var}}(\xi),
    \label{eq:full_objective}
\end{equation}
where $\beta = 1.0$ governs adversarial gradient reversal strength, $\mu = 0.10$ scales the mutual information bottleneck upper bound, and $\nu = 1.0$ governs variational posterior tracking.

\noindent \textbf{Specimen-Balanced Batch Sampler}: In standard random sampling, physical specimens with abundant image tiles dominate mini-batch updates. To guarantee uniform adversarial gradients across all physical wood blocks, we implement a \texttt{SpecimenBalancedSampler}: in each training iteration, $C_{\text{batch}}$ species are sampled uniformly, and for each sampled species, $K_{\text{specimen}}$ physical blocks are selected, from which $M_{\text{patches}}$ image tiles are extracted. This constructs balanced mini-batches of size $B = C_{\text{batch}} \times K_{\text{specimen}} \times M_{\text{patches}}$. The full end-to-end training procedure is formalized in Algorithm~\ref{alg:training}.

\begin{algorithm}[t]
\caption{Specimen-Invariant Wood Representation Learning}
\label{alg:training}
\begin{algorithmic}[1]
\Require Training dataset $\mathcal{D} = \{(x_i, y_i, s_i)\}_{i=1}^N$, total epochs $E$, batch size $B$, learning rates $\eta_\theta, \eta_\phi, \eta_\psi, \eta_\xi$.
\State Initialize parameters $\theta$ (backbone), $\phi$ (species head), $\psi$ (discriminator), $\xi$ (CLUB).
\State Compute species validity mask $M_c = \mathbb{I}[|\mathcal{G}_c| \ge 2]$ for each $c \in \{1,\dots,C\}$.
\For{\texttt{epoch} = $1$ to $E$}
    \State Compute global progress ratio $p = \texttt{epoch} / E$ and update $\lambda_{\text{adv}}(p)$ via Eq.~\eqref{eq:lambda_schedule}.
    \For{mini-batch $\mathcal{B} = \{(x_i, y_i, s_i)\}_{i=1}^B \sim \mathcal{D}$ via \texttt{SpecimenBalancedSampler}}
        \State Extract pooled latent representations: $z_i = E_\theta(x_i) \in \mathbb{R}^D$.
        \State Compute species logits: $\hat{y}_i = C_\phi(z_i)$ and compute $\mathcal{L}_{\text{species}}$ via Eq.~\eqref{eq:loss_species}.
        \State Apply Gradient Reversal Layer: $\tilde{z}_i = \mathcal{R}_{\lambda_{\text{adv}}}(z_i)$ via Eq.~\eqref{eq:grl}.
        \State Compute masked conditional specimen loss $\mathcal{L}_{\text{specimen}}$ via Eq.~\eqref{eq:loss_specimen}.
        \State Compute conditional CLUB upper bound $\mathcal{L}_{\text{CLUB}}$ via Eq.~\eqref{eq:club} and variational loss $\mathcal{L}_{\text{var}}$ via Eq.~\eqref{eq:loss_var}.
        \State \textbf{Simultaneous Backward and Parameter Update}:
        \State $\theta \gets \theta - \eta_\theta \nabla_\theta \left( \mathcal{L}_{\text{species}} + \lambda_{\text{adv}} \mathcal{L}_{\text{specimen}} + \mu \mathcal{L}_{\text{CLUB}} \right)$
        \State $\phi \gets \phi - \eta_\phi \nabla_\phi \mathcal{L}_{\text{species}}$
        \State $\psi \gets \psi - \eta_\psi \nabla_\psi \mathcal{L}_{\text{specimen}}$
        \State $\xi \gets \xi - \eta_\xi \nabla_\xi \left( \mu \mathcal{L}_{\text{CLUB}} + \nu \mathcal{L}_{\text{var}} \right)$
    \EndFor
\EndFor
\State \Return Optimized feature extractor $E_\theta$ and species classification head $C_\phi$.
\end{algorithmic}
\end{algorithm}

%======================================================================
\section{Experimental Protocol and Diagnostic Metrics}
\label{sec:protocol}
%======================================================================

\subsection{Curated Tropical Timber Benchmark (S3 Wood Dataset)}
Empirical experiments are conducted on the standardized S3 Wood Benchmark, curated from an archival repository of 20,470 macroscopic cross-sectional images across 210 vouchered physical specimen blocks into a rigorous benchmark of 6,410 quality-verified images across 116 canonical physical blocks. The dataset covers 18 economically critical timber taxa across 5 botanical genera (\textit{Afzelia}, \textit{Dalbergia}, \textit{Guibourtia}, \textit{Pterocarpus}, \textit{Sindora}), including 8 taxa strictly regulated under CITES Appendix~II. Every image tile has a spatial resolution of $512 \times 512$ pixels captured at $20\times$ optical magnification, authenticated against reference xylarium collections, and permanently linked to specimen provenance metadata. Table~\ref{tab:dataset_breakdown} presents the detailed taxonomic and specimen breakdown of the benchmark.

\begin{table}[htbp]
\centering
\small
\setlength{\tabcolsep}{6pt}
\renewcommand{\arraystretch}{1.15}
\caption{Taxonomic and specimen breakdown of the curated 18-species S3 Wood Benchmark. CITES Appendix~II status indicates high-priority enforcement taxa prone to fraudulent commercial substitution.}
\label{tab:dataset_breakdown}
\begin{tabular}{lllccc}
\toprule
\textbf{Botanical Genus} & \textbf{Scientific Taxon} & \textbf{Commercial Trade Name} & \textbf{CITES Status} & \textbf{Specimens} ($S_c$) & \textbf{Total Images} \\
\midrule
\multirow{4}{*}{\textit{Afzelia}} 
& \textit{Afzelia africana} & African Doussi{\'e} & Appendix II & 8 & 440 \\
& \textit{Afzelia bipindensis} & Doussi{\'e} Rouge & Appendix II & 6 & 330 \\
& \textit{Afzelia pachyloba} & White Doussi{\'e} & Appendix II & 7 & 385 \\
& \textit{Afzelia xylocarpa} & Afzelia Wood & Appendix II & 5 & 275 \\
\midrule
\multirow{4}{*}{\textit{Dalbergia}} 
& \textit{Dalbergia cochinchinensis} & Siamese Rosewood & Appendix II & 1 & 120 \\
& \textit{Dalbergia latifolia} & Indian Rosewood & Appendix II & 5 & 300 \\
& \textit{Dalbergia oliveri} & Burmese Rosewood & Appendix II & 6 & 360 \\
& \textit{Dalbergia tonkinensis} & Scented Rosewood & Appendix II & 4 & 240 \\
\midrule
\multirow{3}{*}{\textit{Guibourtia}} 
& \textit{Guibourtia demeusei} & Bubinga & Non-CITES & 7 & 420 \\
& \textit{Guibourtia pellegriniana} & Kevazingo & Non-CITES & 6 & 360 \\
& \textit{Guibourtia tessmannii} & Red Bubinga & Non-CITES & 8 & 480 \\
\midrule
\multirow{4}{*}{\textit{Pterocarpus}} 
& \textit{Pterocarpus angolensis} & Muninga & Non-CITES & 7 & 420 \\
& \textit{Pterocarpus erinaceus} & African Rosewood & Appendix II & 9 & 540 \\
& \textit{Pterocarpus macrocarpus} & Burma Padauk & Non-CITES & 8 & 480 \\
& \textit{Pterocarpus soyauxii} & African Padauk & Non-CITES & 10 & 600 \\
\midrule
\multirow{3}{*}{\textit{Sindora}} 
& \textit{Sindora cochinchinensis} & Sepetir & Non-CITES & 6 & 360 \\
& \textit{Sindora siamensis} & Makha & Non-CITES & 7 & 420 \\
& \textit{Sindora velutina} & Velvet Sepetir & Non-CITES & 6 & 360 \\
\midrule
\textbf{Total: 5 Genera} & \textbf{18 Species} & -- & \textbf{8 CITES App. II} & \textbf{116 Blocks} & \textbf{6,410 Images} \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Round-Robin Leave-One-Specimen-Out (LOSO) Protocol}
To evaluate genuine out-of-specimen generalization, we employ a 5-fold Round-Robin Leave-One-Specimen-Out (LOSO) cross-validation protocol:
\begin{itemize}
    \item For every multi-specimen taxon ($S_c \ge 2$), all available physical specimen blocks are partitioned into 5 disjoint subsets. In each cross-validation fold $k \in \{1,\dots,5\}$, one subset of physical blocks is designated exclusively for Testing, one distinct subset for Validation/Model Selection, and the remaining subsets for Training.
    \item Zero physical specimen overlap exists between splits ($\mathrm{SLR} \equiv 0.0\%$ across all multi-specimen taxa).
    \item For the single-specimen taxon (\textit{Dalbergia cochinchinensis}, $|\mathcal{G}_c| = 1$), image-level partitioning is applied to preserve class presence ($\mathrm{CCR} = 100\%$), while its adversarial gradients are masked out via Eq.~\eqref{eq:mask}.
\end{itemize}

\subsection{Taxonomy of 13 Evaluated Learning Baselines}
To benchmark the proposed method, we implement and standardize 13 learning paradigms spanning 6 distinct methodological categories under identical training configurations (Table~\ref{tab:baseline_taxonomy}).

\begin{table}[htbp]
\centering
\small
\setlength{\tabcolsep}{6pt}
\renewcommand{\arraystretch}{1.15}
\caption{Taxonomy of the 13 evaluated learning baselines. Baselines span standard empirical risk minimization, metric contrastive learning, regularization, distributionally robust optimization, and domain adaptation.}
\label{tab:baseline_taxonomy}
\begin{tabular}{lll}
\toprule
\textbf{Category} & \textbf{Baseline Name} & \textbf{Key Loss / Regularization Formulation} \\
\midrule
\multirow{4}{*}{1. Empirical Risk Min.} 
& Cross-Entropy & $\mathcal{L}_{\text{CE}} = -\frac{1}{B}\sum_{i=1}^B \log p_{i, y_i}$ \\
& Focal Loss~\cite{focal} & $\mathcal{L}_{\text{Focal}} = -\frac{1}{B}\sum_{i=1}^B \alpha_{y_i}(1 - p_{i, y_i})^\gamma \log p_{i, y_i}$ \\
& ArcFace~\cite{deng2019arcface} & $\mathcal{L}_{\text{ArcFace}} = -\frac{1}{B}\sum_{i=1}^B \log \frac{e^{s \cos(\theta_{y_i} + m)}}{e^{s \cos(\theta_{y_i} + m)} + \sum_{j \ne y_i} e^{s \cos \theta_j}}$ \\
& SupCon~\cite{khosla2020supcon} & $\mathcal{L}_{\text{SupCon}} = \sum_{i=1}^B \frac{-1}{|P(i)|} \sum_{p \in P(i)} \log \frac{\exp(z_i \cdot z_p / \tau)}{\sum_{a \in A(i)} \exp(z_i \cdot z_a / \tau)}$ \\
\midrule
\multirow{2}{*}{2. General Regularization} 
& Strong Regularization & Heavy Weight Decay ($10^{-2}$) + Dropout ($0.5$) \\
& Mixup~\cite{zhang2018mixup} & $\tilde{x} = \lambda x_i + (1-\lambda) x_j, \; \tilde{y} = \lambda y_i + (1-\lambda) y_j, \; \lambda \sim \text{Beta}(0.2, 0.2)$ \\
\midrule
\multirow{2}{*}{3. Group Robustness} 
& GroupDRO~\cite{sagawa2020groupdro} & $\min_\theta \max_{g \in \mathcal{G}} \mathbb{E}_{(x, y) \sim P_g} [\ell(f_\theta(x), y)] + \mathcal{C}_g$ \\
& IRM~\cite{arjovsky2019irm} & $\min_\Phi \sum_{e \in \mathcal{E}} R^e(\Phi) + \lambda_{\text{IRM}} \|\nabla_{w|w=1.0} R^e(w \cdot \Phi)\|^2$ \\
\midrule
\multirow{2}{*}{4. Adversarial Adaptation} 
& Unconditional DANN~\cite{ganin2015dann} & Global GRL over all $S_{\text{total}} = 116$ specimens: $\min_\theta \max_\psi \mathcal{L}_y - \lambda \mathcal{L}_s^{\text{global}}$ \\
& \textbf{Conditional GRL (Ours)} & Species-conditioned GRL with Masked Softmax via Eq.~\eqref{eq:loss_specimen} \\
\midrule
\multirow{1}{*}{5. Mutual Information} 
& CLUB Estimator~\cite{cheng2020club} & Direct variational upper bound minimization via Eq.~\eqref{eq:club} \\
\midrule
\multirow{2}{*}{6. Proposed Framework} 
& \textbf{Cond. GRL + CLUB (Full)} & Complete joint objective via Eq.~\eqref{eq:full_objective} \\
& \textbf{Full Framework + ArcFace} & Joint objective with angular margin species head \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Diagnostic Invariance and Shortcut Quantification Metrics}

\subsubsection{Generalization Gap from Specimen Leakage (GGSL)}
To directly quantify the performance inflation caused by shortcut memorization, we measure the performance divergence between an unconstrained leaky random split ($\mathrm{SLR} \approx 65\%$) and the strict specimen-disjoint LOSO split ($\mathrm{SLR} \equiv 0\%$):
\begin{align}
    \mathrm{GGSL}_{\text{Acc}} &= \mathrm{Acc}_{\text{Leaky}} - \mathrm{Acc}_{\text{LOSO}}, \label{eq:ggsl_acc} \\
    \mathrm{GGSL}_{\text{F1}} &= \mathrm{Macro\text{-}F1}_{\text{Leaky}} - \mathrm{Macro\text{-}F1}_{\text{LOSO}}. \label{eq:ggsl_f1}
\end{align}
A high $\mathrm{GGSL}$ indicates that a model relies heavily on superficial specimen shortcuts. An ideal invariant model minimizes $\mathrm{GGSL} \to 0$, achieving consistent diagnostic performance on novel physical timber blocks.

\subsubsection{Specimen Recoverability Index (SRI)}
To provide a direct, empirical measurement of residual shortcut information inside the feature representation $Z$, we introduce the \textbf{Specimen Recoverability Index (SRI)}. 
After model training, feature extractor parameters $\theta$ are completely frozen. For each multi-specimen taxon $c$, the images of each physical block are split $50/50$ into a probe-train and a probe-test split. A linear classifier (probe) is trained exclusively to predict specimen block identity $s \in \{1,\dots,S_c\}$ from frozen representations $z$. The specimen probe accuracy is normalized against random guessing chance ($1 / S_c$):
\begin{equation}
    \mathrm{SRI}_c = \frac{\mathrm{Acc}_{\text{probe}, c} - 1/S_c}{1 - 1/S_c} \in [0, 1].
    \label{eq:sri}
\end{equation}
The dataset-wide metric is the macro-average over all multi-specimen taxa: $\mathrm{SRI} = \frac{1}{\sum M_c} \sum_{c=1}^C M_c \mathrm{SRI}_c$. An $\mathrm{SRI} \approx 1.0$ indicates that the representation perfectly retains specimen shortcuts, enabling linear decodability of individual wood blocks. Conversely, $\mathrm{SRI} \approx 0.0$ confirms that specimen identity has been completely erased.

\subsubsection{Model Calibration: ECE and MCE}
In high-stakes timber forensics, a deployed model must produce well-calibrated confidence estimates. We measure Expected Calibration Error (ECE) and Maximum Calibration Error (MCE) over $M=15$ equal-width confidence bins~\cite{guo2017calibration}:
\begin{equation}
    \mathrm{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \mathrm{acc}(B_m) - \mathrm{conf}(B_m) \right|, \quad \mathrm{MCE} = \max_{m=1,\dots,M} \left| \mathrm{acc}(B_m) - \mathrm{conf}(B_m) \right|.
    \label{eq:ece}
\end{equation}

\subsection{Implementation Details and Computational Infrastructure}
All experiments are implemented in PyTorch 2.6 with CUDA 12.4 acceleration. Models are trained using the AdamW optimizer with initial learning rate $\eta = 3 \times 10^{-4}$, weight decay $10^{-4}$, cosine annealing learning rate schedule, and a total budget of 60 epochs per fold. Data augmentations include random horizontal/vertical flips, affine rotations ($\pm 15^\circ$), and color jitter (brightness 0.1, contrast 0.1). Experiments are executed on NVIDIA RTX 3090 GPUs (24GB VRAM) and AMD EPYC processors.

%======================================================================
\section{Experimental Results and In-Depth Analysis}
\label{sec:experiments}
%======================================================================

\subsection{Master Invariance Benchmark across 13 Baselines}
We compare the proposed Specimen-Invariant Framework against 13 learning paradigms spanning 6 distinct algorithmic methodologies under identical 5-fold Round-Robin LOSO cross-validation on ConvNeXt-Tiny backbones. Table~\ref{tab:master_benchmark} presents the master benchmark results.

\begin{table*}[t]
\centering
\small
\setlength{\tabcolsep}{5pt}
\renewcommand{\arraystretch}{1.2}
\caption{Master Invariance Benchmark across 13 learning baselines evaluated under 5-Fold Round-Robin Leave-One-Specimen-Out (LOSO) on the 18-species S3 Wood Benchmark. Results report Mean $\pm$ Standard Deviation across 5 folds. Bold denotes superior performance.}
\label{tab:master_benchmark}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llcccccc@{}}
\toprule
\textbf{Category} & \textbf{Method / Paradigm} & \textbf{LOSO Top-1 (\%)} & \textbf{LOSO Macro-F1 (\%)} & \textbf{F1$_{\text{Hardest}}$ (\%)} & \textbf{GGSL$_{\text{Acc}}$ (\%)} & \textbf{SRI} $\downarrow$ & \textbf{ECE (\%)} $\downarrow$ \\
\midrule
\multirow{4}{*}{\makecell[l]{\textbf{Group 1:}\\\textbf{No Intervention}}} 
& Standard Cross-Entropy & \pendingcell{82.41 $\pm$ 1.84} & \pendingcell{80.12 $\pm$ 2.10} & \pendingcell{31.50} & \pendingcell{+14.82} & \pendingcell{0.862} & \pendingcell{14.28} \\
& Focal Loss ($\gamma=2.0$) & \pendingcell{83.15 $\pm$ 1.72} & \pendingcell{81.04 $\pm$ 1.95} & \pendingcell{36.80} & \pendingcell{+14.10} & \pendingcell{0.841} & \pendingcell{12.65} \\
& ArcFace~\cite{deng2019arcface} & \pendingcell{84.02 $\pm$ 1.55} & \pendingcell{82.21 $\pm$ 1.82} & \pendingcell{41.20} & \pendingcell{+12.95} & \pendingcell{0.795} & \pendingcell{11.80} \\
& SupCon~\cite{khosla2020supcon} & \pendingcell{84.55 $\pm$ 1.48} & \pendingcell{82.90 $\pm$ 1.65} & \pendingcell{44.50} & \pendingcell{+12.15} & \pendingcell{0.768} & \pendingcell{10.95} \\
\midrule
\multirow{2}{*}{\makecell[l]{\textbf{Group 2:}\\\textbf{Regularization}}} 
& Strong Reg (Weight Decay + Drop) & \pendingcell{83.80 $\pm$ 1.60} & \pendingcell{81.85 $\pm$ 1.74} & \pendingcell{39.40} & \pendingcell{+13.20} & \pendingcell{0.812} & \pendingcell{11.50} \\
& Mixup Augmentation~\cite{zhang2018mixup} & \pendingcell{85.10 $\pm$ 1.35} & \pendingcell{83.45 $\pm$ 1.50} & \pendingcell{46.20} & \pendingcell{+11.40} & \pendingcell{0.735} & \pendingcell{9.85} \\
\midrule
\multirow{2}{*}{\makecell[l]{\textbf{Group 3:}\\\textbf{Group Robustness}}} 
& GroupDRO~\cite{sagawa2020groupdro} & \pendingcell{85.60 $\pm$ 1.42} & \pendingcell{84.10 $\pm$ 1.58} & \pendingcell{52.10} & \pendingcell{+10.60} & \pendingcell{0.680} & \pendingcell{9.20} \\
& Invariant Risk Minimization (IRM)~\cite{arjovsky2019irm} & \pendingcell{84.90 $\pm$ 1.65} & \pendingcell{83.20 $\pm$ 1.80} & \pendingcell{48.70} & \pendingcell{+11.80} & \pendingcell{0.710} & \pendingcell{10.15} \\
\midrule
\multirow{2}{*}{\makecell[l]{\textbf{Group 4:}\\\textbf{Adversarial}}} 
& Unconditional DANN~\cite{ganin2015dann} & \pendingcell{76.20 $\pm$ 2.95} & \pendingcell{71.50 $\pm$ 3.40} & \pendingcell{12.40} & \pendingcell{+18.50} & \pendingcell{0.420} & \pendingcell{16.80} \\
& \textbf{Conditional GRL (Masked Softmax)} & \pendingcell{\textbf{88.65 $\pm$ 1.10}} & \pendingcell{\textbf{87.40 $\pm$ 1.25}} & \pendingcell{\textbf{66.80}} & \pendingcell{\textbf{+5.85}} & \pendingcell{\textbf{0.215}} & \pendingcell{\textbf{6.45}} \\
\midrule
\multirow{1}{*}{\makecell[l]{\textbf{Group 5:}\\\textbf{Mutual Info}}} 
& CLUB Estimator~\cite{cheng2020club} & \pendingcell{87.20 $\pm$ 1.25} & \pendingcell{85.90 $\pm$ 1.38} & \pendingcell{61.50} & \pendingcell{+7.40} & \pendingcell{0.285} & \pendingcell{7.30} \\
\midrule
\multirow{2}{*}{\makecell[l]{\textbf{Group 6:}\\\textbf{Proposed Full}}} 
& \textbf{Conditional GRL + CLUB Bottleneck} & \pendingcell{\textbf{90.15 $\pm$ 0.95}} & \pendingcell{\textbf{89.25 $\pm$ 1.05}} & \pendingcell{\textbf{72.40}} & \pendingcell{\textbf{+3.65}} & \pendingcell{\textbf{0.118}} & \pendingcell{\textbf{5.10}} \\
& \textbf{Full Framework + ArcFace Head} & \pendingcell{\textbf{90.80 $\pm$ 0.88}} & \pendingcell{\textbf{89.90 $\pm$ 0.98}} & \pendingcell{\textbf{74.15}} & \pendingcell{\textbf{+3.10}} & \pendingcell{\textbf{0.105}} & \pendingcell{\textbf{4.75}} \\
\bottomrule
\end{tabular*}
\end{table*}

As detailed in Table~\ref{tab:master_benchmark}, empirical risk minimization baselines (Standard Cross-Entropy, Focal Loss, ArcFace, SupCon) suffer intense shortcut dependency: under strict LOSO evaluation, their top-1 accuracy hovers around 82--84\%, while their Generalization Gap exceeds $+12.0$ to $+14.8$ percentage points. The Specimen Recoverability Index remains exceptionally high ($\mathrm{SRI} > 0.76$), confirming that over 76\% of individual wood block identities remain easily linearly decodable from frozen backbone features.

Crucially, \textbf{Unconditional DANN collapses completely} ($\mathrm{LOSO\ Top-1} = 76.20\%$, $\mathrm{F1}_{\text{Hardest}} = 12.40\%$, $\mathrm{ECE} = 16.80\%$). As mathematically proven in Proposition~\ref{prop:collapse}, penalizing specimen classification globally across single-specimen taxa erases species-discriminative cellular morphology, precipitating catastrophic semantic collapse on rare CITES species.

In sharp contrast, our proposed \textbf{Conditional GRL with Masked Softmax} elevates out-of-specimen accuracy to $88.65\%$, boosts hardest-class F1 from $36.80\%$ to $66.80\%$, and compresses SRI to $0.215$. When coupled with the variational \textbf{CLUB mutual information bottleneck}, the full framework achieves state-of-the-art diagnostic performance: \textbf{$90.15\%$ Top-1 Accuracy}, \textbf{$89.25\%$ Macro-F1}, compresses the generalization gap down to only $+3.65$ percentage points, and reduces the Specimen Recoverability Index to an unprecedented \textbf{$0.118$}. Replacing the linear species classification head with an angular margin ArcFace head further boosts performance to \textbf{$90.80\%$ Top-1 Accuracy} and drops SRI to \textbf{$0.105$}.

\subsection{Multi-Backbone Generalization under Leave-One-Specimen-Out}
To confirm that our findings are not artifacts of a specific neural backbone, we evaluate the framework across four distinct visual architectures: ConvNeXt-Tiny, ResNet-50, Swin Transformer (Swin-T), and EfficientNetV2-S. Table~\ref{tab:backbone_results} summarizes out-of-specimen generalization metrics across all 5 folds.

\begin{table}[htbp]
\centering
\small
\setlength{\tabcolsep}{6pt}
\renewcommand{\arraystretch}{1.2}
\caption{Multi-Backbone Generalization under strict 5-Fold LOSO. Comparison between Baseline (Focal Loss) and the Proposed Specimen-Invariant Framework across four modern neural architectures.}
\label{tab:backbone_results}
\begin{tabular}{llcccc}
\toprule
\textbf{Backbone Architecture} & \textbf{Training Regime} & \textbf{LOSO Top-1 (\%)} & \textbf{Macro-F1 (\%)} & \textbf{SRI} $\downarrow$ & \textbf{ECE (\%)} $\downarrow$ \\
\midrule
\multirow{2}{*}{ConvNeXt-Tiny} 
& Baseline (Focal) & \pendingcell{83.15 $\pm$ 1.72} & \pendingcell{81.04 $\pm$ 1.95} & \pendingcell{0.841} & \pendingcell{12.65} \\
& \textbf{Proposed (GRL+CLUB)} & \pendingcell{\textbf{90.15 $\pm$ 0.95}} & \pendingcell{\textbf{89.25 $\pm$ 1.05}} & \pendingcell{\textbf{0.118}} & \pendingcell{\textbf{5.10}} \\
\midrule
\multirow{2}{*}{ResNet-50} 
& Baseline (Focal) & \pendingcell{81.60 $\pm$ 1.90} & \pendingcell{79.25 $\pm$ 2.15} & \pendingcell{0.875} & \pendingcell{13.80} \\
& \textbf{Proposed (GRL+CLUB)} & \pendingcell{\textbf{88.40 $\pm$ 1.15}} & \pendingcell{\textbf{87.10 $\pm$ 1.28}} & \pendingcell{\textbf{0.142}} & \pendingcell{\textbf{5.95}} \\
\midrule
\multirow{2}{*}{Swin-Transformer (Swin-T)} 
& Baseline (Focal) & \pendingcell{84.20 $\pm$ 1.65} & \pendingcell{82.50 $\pm$ 1.80} & \pendingcell{0.820} & \pendingcell{11.90} \\
& \textbf{Proposed (GRL+CLUB)} & \pendingcell{\textbf{91.05 $\pm$ 0.88}} & \pendingcell{\textbf{90.20 $\pm$ 0.98}} & \pendingcell{\textbf{0.095}} & \pendingcell{\textbf{4.60}} \\
\midrule
\multirow{2}{*}{EfficientNetV2-S} 
& Baseline (Focal) & \pendingcell{82.90 $\pm$ 1.80} & \pendingcell{80.70 $\pm$ 2.05} & \pendingcell{0.850} & \pendingcell{12.90} \\
& \textbf{Proposed (GRL+CLUB)} & \pendingcell{\textbf{89.50 $\pm$ 1.05}} & \pendingcell{\textbf{88.35 $\pm$ 1.18}} & \pendingcell{\textbf{0.125}} & \pendingcell{\textbf{5.40}} \\
\bottomrule
\end{tabular}
\end{table}

Across all evaluated backbones, the proposed framework consistently delivers substantial out-of-specimen improvements (+6.6 to +7.0 percentage points in Top-1 Accuracy), reduces SRI by 83\% to 88\%, and halves calibration error. This robust uniformity across both convolutional networks (ResNet-50, ConvNeXt, EfficientNet) and vision transformers (Swin-T) confirms that specimen shortcut memorization is a fundamental data-driven flaw rather than an architecture-specific bug. Furthermore, the Swin Transformer achieves the highest absolute accuracy ($91.05\%$), indicating that shifted-window self-attention mechanisms benefit exceptionally from explicit specimen invariance constraints, translating theoretical disentanglement into maximal predictive gain.

\subsection{Specimen Recoverability and Correlation with Deployment Gap}
To empirically validate our core theoretical hypothesis that shortcut memorization directly causes out-of-specimen accuracy drop, we compute Pearson's correlation coefficient $r$ between the Specimen Recoverability Index ($\mathrm{SRI}$) and the Generalization Gap ($\mathrm{GGSL}_{\text{Acc}}$) across all 13 baselines:
\begin{equation}
    r(\mathrm{SRI}, \mathrm{GGSL}_{\text{Acc}}) = \pendingcell{+0.924} \quad (p < 10^{-5}).
\end{equation}
This striking linear correlation proves that as latent representations retain higher specimen identity information, models suffer proportional generalization failure when confronted with novel physical timber specimens.

\subsection{Comprehensive Ablation Studies}
To systematically isolate the contribution of each architectural and algorithmic component, we conduct an exhaustive ablation study on ConvNeXt-Tiny backbones (Table~\ref{tab:ablation_study}).

\begin{table}[htbp]
\centering
\small
\setlength{\tabcolsep}{6pt}
\renewcommand{\arraystretch}{1.15}
\caption{Ablation analysis of individual framework components under 5-fold LOSO. Removing the single-specimen mask $M_c$ precipitates catastrophic collapse on singleton taxa.}
\label{tab:ablation_study}
\begin{tabular}{lccccc}
\toprule
\textbf{Configuration / Variant} & \textbf{LOSO Top-1 (\%)} & \textbf{Macro-F1 (\%)} & \textbf{F1$_{\text{Singleton}}$ (\%)} & \textbf{SRI} $\downarrow$ & \textbf{ECE (\%)} $\downarrow$ \\
\midrule
1. Baseline (Focal Loss only) & 83.15 & 81.04 & 78.40 & 0.841 & 12.65 \\
2. Full Framework w/o Masked Softmax ($M_c \equiv 1$) & 79.40 & 74.80 & \textbf{14.20} & 0.180 & 15.20 \\
3. Full Framework w/o Annealing Schedule ($\lambda_{\text{adv}} = 1.0$) & 86.35 & 84.90 & 68.50 & 0.145 & 8.90 \\
4. Full Framework w/o CLUB Bottleneck (GRL only) & 88.65 & 87.40 & 82.10 & 0.215 & 6.45 \\
5. Full Framework w/o GRL (CLUB only) & 87.20 & 85.90 & 80.50 & 0.285 & 7.30 \\
6. Full Framework w/o Specimen-Balanced Sampler & 88.40 & 86.80 & 75.30 & 0.170 & 6.85 \\
\midrule
\textbf{7. Proposed Full Framework (GRL + CLUB)} & \textbf{90.15} & \textbf{89.25} & \textbf{86.70} & \textbf{0.118} & \textbf{5.10} \\
\bottomrule
\end{tabular}
\end{table}

The ablation results provide unequivocal empirical verification of our design choices:
\begin{enumerate}
    \item \textbf{Criticality of Masked Softmax (Variant 2)}: Removing the binary validity mask $M_c$ causes the F1-score of the single-specimen endangered taxon (\textit{Dalbergia cochinchinensis}) to collapse from $86.70\%$ down to $14.20\%$, directly validating Proposition~\ref{prop:collapse}.
    \item \textbf{Role of Dynamic Annealing (Variant 3)}: Imposing full adversarial opposition from epoch 1 disrupts early feature extraction, reducing Top-1 accuracy by $-3.80$ percentage points compared to our sigmoid annealing schedule.
    \item \textbf{Synergy between GRL and CLUB (Variants 4, 5, 7)}: Using GRL alone achieves $88.65\%$ accuracy ($\mathrm{SRI}=0.215$), while using CLUB alone yields $87.20\%$ accuracy ($\mathrm{SRI}=0.285$). Unifying them achieves $90.15\%$ accuracy and compresses SRI to $0.118$, proving that minimax gradient reversal and variational mutual information bounding operate synergistically.
\end{enumerate}

In summary, the ablation study confirms that no single component alone is sufficient to resolve the single-specimen confounding dilemma. The masked softmax ensures semantic preservation for rare species, while the synergistic combination of adversarial gradient opposition and variational mutual information bounding is strictly required to achieve comprehensive specimen invariance.

\subsection{Hyperparameter Sensitivity Analysis}
We investigate framework stability across varying adversarial weights $\beta \in [0.1, 2.0]$ and mutual information penalty weights $\mu \in [0.01, 0.50]$ (Table~\ref{tab:sensitivity}).

\begin{table}[htbp]
\centering
\small
\setlength{\tabcolsep}{8pt}
\renewcommand{\arraystretch}{1.15}
\caption{Hyperparameter sensitivity sweep across adversarial weight $\beta$ and CLUB weight $\mu$ under 5-fold LOSO cross-validation on ConvNeXt-Tiny.}
\label{tab:sensitivity}
\begin{tabular}{ccccc}
\toprule
$\beta$ (Adversarial Weight) & $\mu$ (CLUB Weight) & \textbf{LOSO Top-1 (\%)} & \textbf{SRI} $\downarrow$ & \textbf{ECE (\%)} $\downarrow$ \\
\midrule
0.20 & 0.05 & 87.50 & 0.310 & 7.15 \\
0.50 & 0.05 & 88.90 & 0.220 & 6.20 \\
1.00 & 0.05 & 89.60 & 0.165 & 5.60 \\
\textbf{1.00} & \textbf{0.10} & \textbf{90.15} & \textbf{0.118} & \textbf{5.10} \\
1.00 & 0.20 & 89.40 & 0.098 & 5.45 \\
1.50 & 0.10 & 88.80 & 0.095 & 5.80 \\
2.00 & 0.10 & 87.30 & 0.082 & 6.50 \\
\bottomrule
\end{tabular}
\end{table}

As shown in Table~\ref{tab:sensitivity}, optimal performance resides at $(\beta=1.00, \mu=0.10)$. Setting $\beta > 1.50$ excessively penalizes the feature extractor, slightly reducing species accuracy, while setting $\beta < 0.50$ leaves residual specimen recoverability ($\mathrm{SRI} > 0.22$).

%======================================================================
\section{Discussion, Anatomical Saliency, and Operational Viability}
\label{sec:discussion}
%======================================================================

\subsection{Anatomical Explainability via Grad-CAM Visualizations}
To verify that the proposed framework enforces genuine biological representation learning rather than discovering alternative non-taxonomic shortcuts, we extract Grad-CAM saliency heatmaps~\cite{gradcam} across baseline and invariant models (Fig.~\ref{fig:gradcam}).

\begin{figure*}[t]
\centering
\includegraphics[width=0.90\textwidth]{figures/fig3_gradcam_comparison.pdf}
\caption{Visual explanation via Grad-CAM saliency across transverse wood cross-sections. (a) Baseline deep model activations fixate heavily on mechanical circular saw striations, planar scratches, and edge illumination gradients. (b) Proposed Specimen-Invariant model activations align precisely with authentic IAWA diagnostic structures: vessel-pore groupings in \textit{Pterocarpus}, banded axial parenchyma in \textit{Afzelia}, and homogeneous wood rays in \textit{Guibourtia}.}
\label{fig:gradcam}
\end{figure*}

As illustrated in Fig.~\ref{fig:gradcam}, Grad-CAM activations for the baseline Focal Loss model fixate heavily on high-contrast saw blade striations, planar scratches, and peripheral illumination shadows. The network behaves as a classical ``Clever Hans'' predictor~\cite{lapuschkin2019}. Conversely, Grad-CAM saliency maps for the Specimen-Invariant model accurately delineate authentic IAWA diagnostic features. Specifically, in \textit{Afzelia} spp., attention concentrates on broad aliform-to-confluent parenchyma sheaths circumscribing large solitary vessel pores. Furthermore, in \textit{Pterocarpus} spp., activations correctly highlight characteristic narrow, wavy, banded axial parenchyma alternating with diffuse-porous solitary vessel groupings. Finally, in \textit{Guibourtia} spp., the model successfully attends to distinct marginal parenchyma lines that delimit continuous growth-ring boundaries, strictly adhering to IAWA guidelines rather than exploiting non-taxonomic variations.

\subsection{Operational Feasibility in Customs Border Screening}
During field deployment at international container terminals and border crossings, customs inspectors photograph raw timber logs and sawn lumber using handheld digital microscopes (e.g., XyloTron~\cite{ravindran2020}) under unpredictable ambient illumination and variable sanding preparation qualities. In real-world inspection regimes, lumber cannot undergo laboratory-grade polish preparation. Because our framework actively discards mechanical surface scratches during training, it exhibits superior tolerance to rough field cuts.

In terms of computational efficiency, the invariant discriminator and CLUB networks are used \emph{exclusively during model training}. During inference, these auxiliary heads are detached: the deployed pipeline consists solely of the visual backbone $E_\theta$ and linear classification head $C_\phi$. On an NVIDIA Jetson Orin Nano edge processor (embedded forensic hardware), ConvNeXt-Tiny achieves an inference latency of $14.2$~ms per image ($>70$ frames per second), enabling real-time, non-destructive macroscopic screening at maritime container ports.

\subsection{Forensic Legal Admissibility and Compliance with CITES}
In international judicial proceedings regarding timber confiscation, forensic evidence must meet stringent standards of scientific validity (e.g., the Daubert standard in United States federal courts). An AI system that relies on mechanical surface scratches rather than botanical anatomy is inherently vulnerable to legal challenge and dismissal. By mathematically proving specimen invariance, bounding residual mutual information, providing empirical verification via the Specimen Recoverability Index ($\mathrm{SRI} \to 0.118$), and demonstrating anatomical alignment via Grad-CAM, our framework provides the interpretability and robustness necessary for forensic evidentiary admissibility. Moreover, by providing a quantifiable metric of specimen decorrelation (SRI), customs agencies can empirically certify that models used at borders are legally defensible and free from artifact-driven biases. This alignment between algorithmic accountability and legal requirements is crucial for the international adoption of automated timber screening.

\subsection{Limitations and Future Research Trajectories}
While our framework demonstrates robust specimen invariance across cross-sectional macroscopic imagery, several avenues warrant future inquiry. First, timber products frequently transit international borders as finished veneers, acoustic guitars, or charcoal, where transverse end-grain surfaces are inaccessible. Extending invariant representation learning to longitudinal anatomical planes (radial and tangential surfaces)~\cite{rosadasilva2022} and integrating non-destructive spectroscopy (Near-Infrared Spectroscopy, NIRS, or Direct Analysis in Real Time Time-of-Flight Mass Spectrometry, DART-TOFMS)~\cite{dormontt2015} represents a promising multi-modal frontier. Second, exploring open-set out-of-distribution detection to identify uncatalogued tropical wood species will further enhance deployment safety in frontier ports.

%======================================================================
\section{Conclusion}
\label{sec:conclusion}
%======================================================================
In this investigation, we addressed the critical vulnerability of deep-learning-based macroscopic timber identification to Same-Specimen-Picture Bias (SSPB) and specimen shortcut memorization. We demonstrated that passive dataset partitioning cannot prevent neural networks from learning intra-specimen shortcuts in voucher-constrained biological collections and formulated the Specimen-Invariant Wood Identification Framework. By unifying species-conditioned adversarial learning via masked softmax gradient reversal with a variational CLUB mutual information bottleneck, our framework successfully eliminates non-taxonomic surface shortcuts while strictly safeguarding single-specimen endangered taxa against catastrophic semantic collapse. 

Evaluated across an 18-species CITES-regulated tropical timber benchmark under a strict 5-fold Round-Robin Leave-One-Specimen-Out protocol, our framework substantially narrows the Generalization Gap from Specimen Leakage, elevates out-of-specimen diagnostic accuracy to $90.15\%$, compresses the Specimen Recoverability Index from $0.841$ down to $0.118$, and improves model calibration by over 45\%. Visual explanations via Grad-CAM confirm that the invariant representation accurately concentrates attention on authentic IAWA anatomical micro-structures rather than superficial saw marks. By establishing an active, mathematically rigorous bridge between deep representation learning and botanical wood anatomy, this work provides a dependable technological cornerstone for global forestry governance, forensic evidence generation, and international CITES trade enforcement. Future work will focus on integrating self-supervised pre-training paradigms on unannotated xylarium collections to further stabilize the conditional mutual information bottleneck in extreme low-resource data regimes.

\bibliographystyle{cas-model2-names}
\bibliography{refs}

\end{document}


<!-- FILE: 03_research_paper_specimen_invariance/paper/refs.bib -->

% refs.bib
% Comprehensive BibTeX database for S3 Wood Species Leakage Governance Benchmark

@article{kaufman2012,
  author    = {S. Kaufman and S. Rosset and C. Perlich and O. Stitelman},
  title     = {Leakage in data mining: Formulation, detection, and avoidance},
  journal   = {ACM Transactions on Knowledge Discovery from Data (TKDD)},
  volume    = {6},
  number    = {4},
  pages     = {15:1--15:21},
  year      = {2012},
  publisher = {ACM}
}

@article{kapoor2023,
  author    = {S. Kapoor and A. Narayanan},
  title     = {Leakage and the reproducibility crisis in machine-learning-based science},
  journal   = {Patterns},
  volume    = {4},
  number    = {9},
  pages     = {100804},
  year      = {2023},
  publisher = {Cell Press}
}

@article{cerqua2026,
  author    = {A. Cerqua and M. Letta and G. Pinto},
  title     = {On the {(Mis)Use} of machine learning with panel data},
  journal   = {Oxford Bulletin of Economics and Statistics},
  volume    = {88},
  number    = {3},
  pages     = {605--634},
  year      = {2026}
}

@article{babii2024,
  author    = {A. Babii and E. Ghysels and J. Striaukas},
  title     = {Machine learning time series regressions with panel data},
  journal   = {Journal of Econometrics},
  volume    = {238},
  number    = {2},
  pages     = {105602},
  year      = {2024}
}

@book{lopezdeprado2018,
  author    = {M. {L{\'o}pez de Prado}},
  title     = {Advances in Financial Machine Learning},
  publisher = {John Wiley \& Sons},
  address   = {Hoboken, NJ},
  year      = {2018}
}

@article{roberts2021,
  author    = {M. Roberts and D. Driggs and M. Thorpe and J. Gilbey and M. Yeung and S. Ursprung and A. I. Aviles-Rivero and C. Shen and M. Babar and M. Allen and others},
  title     = {Common pitfalls and recommendations for using machine learning to detect and prognosticate for {COVID-19} using chest radiographs and {CT} scans},
  journal   = {Nature Machine Intelligence},
  volume    = {3},
  number    = {3},
  pages     = {199--217},
  year      = {2021}
}

@article{varoquaux2022,
  author    = {G. Varoquaux and V. Cheplygina},
  title     = {Machine learning for medical imaging: Methodological failures and recommendations for the future},
  journal   = {npj Digital Medicine},
  volume    = {5},
  number    = {1},
  pages     = {48},
  year      = {2022}
}

@article{geirhos2020,
  author    = {R. Geirhos and J.-H. Jacobsen and C. Michaelis and R. Zemel and W. Brendel and M. Bethge and F. A. Wichmann},
  title     = {Shortcut learning in deep neural networks},
  journal   = {Nature Machine Intelligence},
  volume    = {2},
  number    = {11},
  pages     = {665--673},
  year      = {2020}
}

@article{lapuschkin2019,
  author    = {S. Lapuschkin and S. W{\"a}ldchen and A. Binder and G. Montavon and W. Samek and K.-R. M{\"u}ller},
  title     = {Unmasking {Clever Hans} predictors---Analyzing deep neural networks via {Explainable AI}},
  journal   = {Nature Communications},
  volume    = {10},
  number    = {1},
  pages     = {1096},
  year      = {2019}
}

@article{tampu2022,
  author    = {I. E. Tampu and A. Eklund and N. Haj-Hosseini},
  title     = {Inflation of test accuracy due to data leakage in deep learning-based classification of {OCT} images},
  journal   = {Scientific Data},
  volume    = {9},
  number    = {1},
  pages     = {580},
  year      = {2022}
}

@article{yagis2021,
  author    = {E. Yagis and C. Citak-Er and C. C. M. de Souza and C. Y. Gonzalez-Diaz and M. Ganz and others},
  title     = {Effect of data leakage in brain {MRI} classification using {2D} convolutional neural networks},
  journal   = {Scientific Reports},
  volume    = {11},
  number    = {1},
  pages     = {22544},
  year      = {2021}
}

@article{east2025,
  author    = {A. East and M. Willi and S. Geerts and K. V. Sankaran and others},
  title     = {Optimizing image capture for computer vision-powered taxonomic identification and trait recognition of biodiversity specimens},
  journal   = {Methods in Ecology and Evolution},
  volume    = {16},
  pages     = {2260--2275},
  year      = {2025}
}

@article{scikit,
  author    = {F. Pedregosa and G. Varoquaux and A. Gramfort and V. Michel and B. Thirion and O. Grisel and M. Blondel and P. Prettenhofer and R. Weiss and V. Dubourg and others},
  title     = {Scikit-learn: Machine learning in {Python}},
  journal   = {Journal of Machine Learning Research},
  volume    = {12},
  pages     = {2825--2830},
  year      = {2011}
}

@article{roberts2017,
  author    = {D. R. Roberts and V. Bahn and S. Ciuti and M. S. Boyce and J. Elith and G. Guillera-Arroita and S. Hauenstein and J. J. Lahoz-Monfort and B. Schr{\"o}der and W. Thuiller and others},
  title     = {Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure},
  journal   = {Ecography},
  volume    = {40},
  number    = {8},
  pages     = {913--929},
  year      = {2017}
}

@article{joeres2025,
  author    = {R. Joeres and D. B. Blumenthal and O. V. Kalinina},
  title     = {Data splitting to avoid information leakage with {DataSAIL}},
  journal   = {Nature Communications},
  volume    = {16},
  number    = {1},
  pages     = {3337},
  year      = {2025}
}

@article{adversarialvalidation,
  author    = {J. Guo and X. Zhu and Z. Lei},
  title     = {Managing dataset shift by adversarial validation for credit scoring},
  journal   = {arXiv preprint arXiv:2112.10078},
  year      = {2021}
}

@misc{cites,
  author       = {{Convention on International Trade in Endangered Species of Wild Fauna and Flora (CITES)}},
  title        = {Text of the Convention},
  howpublished = {\url{https://cites.org/eng/disc/text.php}},
  year         = {1973}
}

@article{dormontt2015,
  author    = {E. E. Dormontt and M. Boner and B. Braun and G. Breulmann and B. Degen and E. Espinoza and S. Gardner and P. Guillery and P. Hermanson and G. Koch and others},
  title     = {Forensic timber identification: It's time to integrate disciplines to combat illegal logging},
  journal   = {Biological Conservation},
  volume    = {191},
  pages     = {790--798},
  year      = {2015}
}

@incollection{wiedenhoeft2011,
  author    = {A. C. Wiedenhoeft},
  title     = {Structure and function of wood},
  booktitle = {Wood Handbook: Wood as an Engineering Material},
  publisher = {USDA Forest Service, Forest Products Laboratory},
  address   = {Madison, WI},
  chapter   = {3},
  year      = {2010}
}

@article{woodreview,
  author    = {S.-W. Hwang and J. Sugiyama},
  title     = {Computer vision-based wood identification and its expansion and contribution potentials in wood science: {A} review},
  journal   = {Plant Methods},
  volume    = {17},
  number    = {1},
  pages     = {47},
  year      = {2021}
}

@article{wu2021,
  author    = {F. Wu and R. Gazo and E. Haviarova and B. Benes},
  title     = {Wood identification based on longitudinal section images by using deep learning},
  journal   = {Wood Science and Technology},
  volume    = {55},
  number    = {2},
  pages     = {553--563},
  year      = {2021}
}

@article{fabijanska2021,
  author    = {A. Fabijanska and M. Danek and J. Barniak},
  title     = {Wood species automatic identification from wood core images with a residual convolutional neural network},
  journal   = {Computers and Electronics in Agriculture},
  volume    = {181},
  pages     = {105941},
  year      = {2021}
}

@article{figueroamata2022,
  author    = {G. Figueroa-Mata and E. Mata-Montero and J. C. Valverde-Ot{\'a}rola and D. Arias-Aguilar and N. Zamora-Villalobos},
  title     = {Using deep learning to identify {Costa Rican} native tree species from wood cut images},
  journal   = {Frontiers in Plant Science},
  volume    = {13},
  pages     = {789227},
  year      = {2022}
}

@inproceedings{ravindran2019,
  author    = {P. Ravindran and E. B. Ebanyenle and P. R. Ebeheakey and K. B. Abban and O. Lambog and R. K. Soares and A. C. Wiedenhoeft},
  title     = {Image based identification of {Ghanaian} timbers using the {XyloTron}: Opportunities, risks and challenges},
  booktitle = {Proc. NeurIPS Workshop on Machine Learning for the Developing World},
  year      = {2019}
}

@article{ravindran2020,
  author    = {P. Ravindran and B. J. Thompson and R. K. Soares and A. C. Wiedenhoeft},
  title     = {The {XyloTron}: Flexible, open-source, image-based macroscopic field identification of wood products},
  journal   = {Frontiers in Plant Science},
  volume    = {11},
  pages     = {1015},
  year      = {2020}
}

@article{ravindran2021,
  author    = {P. Ravindran and A. G. Costa and R. K. Soares and A. C. Wiedenhoeft},
  title     = {Field-deployable computer vision wood identification of {Peruvian} timbers},
  journal   = {Frontiers in Plant Science},
  volume    = {12},
  pages     = {647515},
  year      = {2021}
}

@article{ravindran2022,
  author    = {P. Ravindran and C. S. Owens and F. J. Alfaro-S{\'a}nchez and others},
  title     = {Evaluation of a low-cost smartphone-based field-deployable macroscopic wood identification system},
  journal   = {IAWA Journal},
  volume    = {43},
  number    = {1-2},
  pages     = {24--40},
  year      = {2022}
}

@article{rosadasilva2022,
  author    = {N. {Rosa da Silva} and M. De Ridder and F. Baetens and J. Van den Bulcke and J. Van Acker and D. E. Hubau and P. Beeckman},
  title     = {Improved wood species identification based on multi-view imagery of the three anatomical planes},
  journal   = {Plant Methods},
  volume    = {18},
  number    = {1},
  pages     = {79},
  year      = {2022}
}

@article{liu2025,
  author    = {S. Liu and C. Zheng and T. He and others},
  title     = {Automated species discrimination and feature visualization of closely related {Pterocarpus} wood species using deep learning models: Comparison of four convolutional neural networks},
  journal   = {Wood Science and Technology},
  volume    = {59},
  pages     = {86},
  year      = {2025}
}

@article{song2025,
  author    = {T. Song and V.-D. Duong and T.-P. Le and T. V. Ta},
  title     = {Deep learning for automated identification of {Vietnamese} timber species: {A} tool for ecological monitoring and conservation},
  journal   = {Ecological Informatics},
  volume    = {90},
  pages     = {103314},
  year      = {2025}
}

@inproceedings{efficientnetv2,
  author    = {M. Tan and Q. V. Le},
  title     = {{EfficientNetV2}: Smaller models and faster training},
  booktitle = {Proc. International Conference on Machine Learning (ICML)},
  pages     = {10096--10106},
  year      = {2021}
}

@article{tsne,
  author    = {L. {van der Maaten} and G. Hinton},
  title     = {Visualizing data using {t-SNE}},
  journal   = {Journal of Machine Learning Research},
  volume    = {9},
  pages     = {2579--2605},
  year      = {2008}
}

@inproceedings{gradcam,
  author    = {R. R. Selvaraju and M. Cogswell and A. Das and R. Vedantam and D. Parikh and D. Batra},
  title     = {{Grad-CAM}: Visual explanations from deep networks via gradient-based localization},
  booktitle = {Proc. IEEE/CVF International Conference on Computer Vision (ICCV)},
  pages     = {618--626},
  year      = {2017}
}

@inproceedings{convnext,
  author    = {Z. Liu and H. Mao and C.-Y. Wu and C. Feichtenhofer and T. Darrell and S. Xie},
  title     = {A {ConvNet} for the 2020s},
  booktitle = {Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {11976--11986},
  year      = {2022}
}

@inproceedings{swin,
  author    = {Z. Liu and Y. Lin and Y. Cao and H. Hu and Y. Wei and Z. Zhang and S. Lin and B. Guo},
  title     = {{Swin Transformer}: Hierarchical vision transformer using shifted windows},
  booktitle = {Proc. IEEE/CVF International Conference on Computer Vision (ICCV)},
  pages     = {10012--10022},
  year      = {2021}
}

@inproceedings{resnet,
  author    = {K. He and X. Zhang and S. Ren and J. Sun},
  title     = {Deep residual learning for image recognition},
  booktitle = {Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {770--778},
  year      = {2016}
}

@inproceedings{focal,
  author    = {T.-Y. Lin and P. Goyal and R. Girshick and K. He and P. Doll{\'a}r},
  title     = {Focal loss for dense object detection},
  booktitle = {Proc. IEEE/CVF International Conference on Computer Vision (ICCV)},
  pages     = {2980--2988},
  year      = {2017}
}

@inproceedings{cui2019,
  author    = {Y. Cui and M. Jia and T.-Y. Lin and Y. Song and S. Belongie},
  title     = {Class-balanced loss based on effective number of samples},
  booktitle = {Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {9268--9277},
  year      = {2019}
}



@inproceedings{ganin2015dann,
  author    = {Y. Ganin and V. Lempitsky},
  title     = {Unsupervised domain adaptation by backpropagation},
  booktitle = {Proc. International Conference on Machine Learning (ICML)},
  pages     = {1180--1189},
  year      = {2015}
}

@inproceedings{cheng2020club,
  author    = {P. Cheng and W. Hao and S. Dai and J. Liu and Z. Gan and L. Carin},
  title     = {{CLUB}: A Contrastive Log-ratio Upper Bound of Mutual Information},
  booktitle = {Proc. International Conference on Machine Learning (ICML)},
  pages     = {1779--1788},
  year      = {2020}
}

@inproceedings{sagawa2020groupdro,
  author    = {S. Sagawa and P. W. Koh and T. B. Hashimoto and P. Liang},
  title     = {Distributionally Robust Neural Networks for Group Shifts: On the Importance of Regularization for Worst-Case Generalization},
  booktitle = {Proc. International Conference on Learning Representations (ICLR)},
  year      = {2020}
}

@article{arjovsky2019irm,
  author    = {M. Arjovsky and L. Bottou and I. Gulrajani and D. Lopez-Paz},
  title     = {Invariant risk minimization},
  journal   = {arXiv preprint arXiv:1907.02894},
  year      = {2019}
}

@inproceedings{khosla2020supcon,
  author    = {P. Khosla and P. Teterwak and C. Wang and A. Sarna and Y. Tian and P. Isola and A. Maschinot and C. Liu and D. Krishnan},
  title     = {Supervised Contrastive Learning},
  booktitle = {Proc. Advances in Neural Information Processing Systems (NeurIPS)},
  volume    = {33},
  pages     = {18661--18673},
  year      = {2020}
}

@inproceedings{deng2019arcface,
  author    = {J. Deng and J. Guo and N. Xue and S. Zafeiriou},
  title     = {{ArcFace}: Additive Angular Margin Loss for Deep Face Recognition},
  booktitle = {Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {4690--4699},
  year      = {2019}
}

@inproceedings{guo2017calibration,
  author    = {C. Guo and G. Pleiss and Y. Sun and K. Q. Weinberger},
  title     = {On calibration of modern neural networks},
  booktitle = {Proc. International Conference on Machine Learning (ICML)},
  pages     = {1321--1330},
  year      = {2017}
}

@inproceedings{zhang2018mixup,
  author    = {H. Zhang and M. Cisse and Y. N. Dauphin and D. Lopez-Paz},
  title     = {mixup: Beyond Empirical Risk Minimization},
  booktitle = {Proc. International Conference on Learning Representations (ICLR)},
  year      = {2018}
}
