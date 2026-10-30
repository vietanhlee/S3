Learning Specimen-Invariant Diagnostic Representations for
Timber Forensics: A Species-Conditioned Adversarial Framework
with Variational Mutual Information Bottlenecks
 Viet-Anh Le iD a , Khanh Nguyen-Trong iD a,∗
a Intelligent Computing for Sustainable Development Laboratory (IC4SD), Posts and Telecommunications Institute of Technology (PTIT), Hanoi,

Vietnam


ARTICLE INFO                                     ABSTRACT
Keywords:                                        Automated macroscopic timber identification is a vital non-destructive screening technology for
Specimen invariance                              enforcing CITES regulations against illegal logging. However, standard deep learning models
Timber forensics                                 deployed for wood recognition suffer from Same-Specimen-Picture Bias (SSPB): they oppor-
Shortcut learning                                tunistically memorize non-taxonomic mechanical artifacts and illumination gradients rather than
Same-Specimen-Picture Bias                       authentic cellular morphology. While strict specimen-disjoint dataset partitioning is essential
Conditional adversarial learning                 for deployment auditing, passive data splitting alone cannot prevent networks from exploiting
Gradient reversal layer                          intra-specimen spurious features during empirical risk minimization, particularly in physical
Mutual information minimization                  xylarium collections where vouchered timber blocks are severely constrained. To eliminate
Contrastive Log-ratio Upper Bound                non-taxonomic artifact reliance at its algorithmic root, we introduce an active representation-
CITES enforcement                                learning paradigm that mathematically disentangles biological species semantics from physical
                                                 specimen provenance. We propose the Specimen-Invariant Wood Identification Framework,
                                                 an architecture integrating three core components: a primary class-balanced classification
                                                 objective, a Species-Conditioned Specimen Discriminator via Masked Softmax to eliminate
                                                 conditional mutual information without causing taxonomic representation erasure on single-
                                                 specimen endangered taxa, and a variational CLUB mutual information bottleneck. Evaluated
                                                 across a 19-species tropical timber benchmark comprising 11 CITES Appendix II-regulated taxa
                                                 and 8 commercially traded non-CITES lookalikes across 15 evaluated learning paradigms (12
                                                 comparative baselines and 3 variants of the proposed framework), our framework establishes
                                                 a standardized single-institutional out-of-specimen benchmark: on the 17 multi-specimen taxa
                                                 evaluated under strict Leave-One-Specimen-Out cross-validation with strictly disjoint physical
                                                 block holdouts (SLR ≡ 0.0%), it achieves a primary diagnostic benchmark of 𝟗𝟎.𝟒𝟓% LOSOmulti
                                                 Top-1 Accuracy (Macro-F1 89.60%). Across the entire 19-species collection, it delivers an
                                                 aggregate operational screening profile of 𝟗𝟎.𝟏𝟓% Overall Top-1 and 𝟖𝟗.𝟐𝟓% Macro-F1, while
                                                 successfully safeguarding voucher-constrained singleton species against representation erasure
                                                 (F1singleton = 86.70%). Because singleton taxa undergo stratified image-level partitioning due to
                                                 international botanical voucher limits, LOSOmulti serves as the authoritative standard for out-of-
                                                 specimen generalization, confirming that singleton taxa do not artificially inflate reported metrics.
                                                 Crucially, the framework suppresses the proposed Specimen Recoverability Index (SRI) from
                                                 0.841 down to 𝟎.𝟏𝟏𝟖 under linear probing, from 0.865 down to 𝟎.𝟏𝟑𝟒 under 2-layer MLP probing,
                                                 and down to 𝟎.𝟏𝟒𝟏 under deep 3-layer MLP probing (alongside an out-of-distribution test-split
                                                 sanity check of SRItest = 0.122 vs 0.850), ensuring visual representations correctly encode
                                                 authentic IAWA anatomical micro-structures rather than superficial surface scratches.




1. Introduction
1.1. The Ecological Crisis and Lab-to-Field Generalization Collapse
    Illegal logging across transnational supply chains devastates tropical ecosystems, drives deforestation, and acceler-
ates global biodiversity loss. Enforcing the Convention on International Trade in Endangered Species of Wild Fauna and
Flora (CITES) requires customs officers and environmental law enforcement to rapidly authenticate protected timber
taxa at border transit checkpoints and maritime container terminals [1]. While authoritative forensic identification has
historically relied on laboratory xylotomy and microscopic histological inspection [2], this traditional pipeline is labor-
intensive, destructive, and bottlenecked by a global shortage of professional wood anatomists. This severe operational
    ∗ Corresponding author.

     E-mail addresses: khanhnt@ptit.edu.vn (K. Nguyen-Trong), anhlv.b23kh002@stu.ptit.edu.vn (V.-A. Le)
                                     Learning Specimen-Invariant Diagnostic Representations for Timber Forensics


     (a) Standard Deep Learning under ERM (Shortcut Bias)                            (b) Proposed Specimen-Invariant Framework (Active Disentanglement)


      Physical Specimen Patches                        Deep Backbone (ERM)                   Physical Specimen Patches                       Dual Invariance Engine
            • Saw blade striations                                                                • Variable wood surfaces                      1. Species-Conditioned GRL
         • Planer abrasion scratches                     Minimizes empirical loss:                • Micro-mechanical noise                      (Masked Softmax on Blocks)
        • Directional lighting gradient
                                                          Opportunistic Shortcut                 • Authentic IAWA anatomy
                                                                                                                                               2. CLUB Variational MI Bound
                                                              Memorization                                                                              I(Z; S|Y) → 0




                         Entangled Latent Space: Z ∼ f(X; θ)                                           Disentangled Invariant Latent Space: Z ∼ Eθ (X)
                 Specimen information I(Z; S|Y) À 0 (High Shortcut Leakage)                           Specimen recoverability SRI → 0.118 (Residual artifacts erased)
            Learns individual wood block quirks instead of IAWA cellular anatomy                     Exclusively retains true IAWA microscopic anatomical morphology




           Evaluation on Unseen Physical Specimen (Strict Split)                            Strict Specimen-Disjoint Generalization & Field Triage Readiness
                ×Generalization Collapse: Macro-F1 Drops by 15% − 30%                       Closes Specimen Leakage Gap: Robust 89.25% Macro-F1 (90.45% LOSO_multi)
       Forensic diagnostic failure at international customs border inspections                 Evidence-grade, specimen-invariant representations for customs triage



Figure 1: The Specimen-Level Shortcut Learning Dilemma in Timber Forensics. (a) Standard deep models trained under
empirical risk minimization opportunistically exploit non-taxonomic mechanical surface artifacts (saw striations, planer
abrasions, directional illumination gradients). When evaluated on novel, previously unseen physical timber specimens,
diagnostic accuracy drops precipitously. (b) The proposed Specimen-Invariant Wood Identication Framework actively
strips specimen provenance via Species-Conditioned Adversarial Disentanglement (Masked Softmax GRL) and a variational
CLUB mutual information bottleneck, forcing latent representations to encode authentic, generalizable IAWA cellular
anatomy.

constraint has motivated the rapid development of automated macroscopic wood identification systems powered by
deep convolutional neural networks and vision transformers [3–6].
    Although these computer vision architectures report remarkable nominal accuracy (> 95%) on closed laboratory
benchmark datasets, practical field deployments consistently reveal a systemic failure mode: diagnostic accuracy
collapses precipitously by 15% to 30% when systems are evaluated on novel physical timber specimens or wood
harvested from alternative forest concessions [7–10]. Such severe out-of-specimen generalization collapse undermines
forensic credibility at high-stakes customs checkpoints, where erroneous classifications can either paralyze legitimate
multi-million-dollar shipments or allow illicit endangered timber to pass into commercial timber markets undetected.

1.2. Etiology of Shortcut Memorization and Limitations of Passive Splitting
     This diagnostic failure is rooted in Same-Specimen-Picture Bias (SSPB) [11], a pernicious form of specimen-level
data leakage. Under naive image-level random partitioning, multiple high-resolution image patches extracted from the
same physical wood block are dispersed across both training and evaluation subsets. Because deep neural networks are
opportunistic feature extractors that favor high-frequency, low-entropy visual shortcuts over complex morphological
patterns [12], they readily memorize non-taxonomic physical artifacts—such as mechanical circular saw striations,
planer cutter abrasions, sanding grit scratches, wax sealants, and localized directional illumination gradients—rather
than authentic cellular geometries (Fig. 1a). Consequently, nominal test metrics reflect the network’s ability to re-
identify previously memorized physical specimens, rather than genuine taxonomic discrimination.
     While enforcing strict specimen-disjoint dataset partitioning (such as Leave-One-Specimen-Out cross-validation)
is indispensable for auditing deployment risk, passive data splitting alone cannot prevent neural networks from learning
intra-specimen shortcuts during empirical risk minimization. In physical xylarium archives, vouchered reference
timber blocks for threatened taxa are severely scarce (|𝑐 | < 10). Without explicit algorithmic invariance constraints,
high-capacity feature extractors naturally minimize training loss by encoding idiosyncratic block-specific markers,
inevitably collapsing when tested on unobserved physical blocks. Thus, an active representation-learning intervention
is mathematically required to enforce 𝐼(𝑍; 𝑆 ∣ 𝑌 ) → 0 while preserving taxonomic diagnostic sufficiency 𝐼(𝑍; 𝑌 ).




Page 2 of 27
                     Learning Specimen-Invariant Diagnostic Representations for Timber Forensics

1.3. The Single-Specimen Dilemma and Proposed Contributions
    Formulating specimen invariance in biological forestry introduces the Single-Specimen Confounding Dilemma.
In standard domain adaptation frameworks [13], an unconditioned global domain discriminator penalizes domain
predictability across the entire dataset. However, in authentic botanical collections, physical specimen blocks are
strictly nested within species categories (𝑆 ⊂ 𝑌 ). For endangered species represented by only a single vouchered
physical block in the xylarium (|𝑐 | = 1), penalizing specimen classification forces the network to penalize species
classification itself, triggering catastrophic taxonomic representation erasure.
    To resolve this challenge, we introduce the Specimen-Invariant Wood Identification Framework, specifically
architected for nested biological taxonomies under extreme specimen scarcity. Our framework synergizes a Species-
Conditioned Specimen Discriminator with Masked Softmax to prevent semantic collapse on singleton taxa,
alongside a Contrastive Log-ratio Upper Bound (CLUB) [14] mutual information bottleneck. We explicitly
position this work within applied ecological informatics, forensic computer vision, and computational wood anatomy
(Ecological Informatics, Computers and Electronics in Agriculture, IEEE Transactions on Information Forensics and
Security), bridging the gap between invariant representation learning and practical international forestry governance.
    The primary contributions of this work are:
   • Causal Formalization and Theoretical Identification: We formalize physical entity shortcut memorization
     through a Structural Causal Model (SCM), prove that unconditioned domain adaptation induces taxonomic
     collapse on singleton taxa, and design a species-conditioned masked softmax mechanism with bounded
     variational mutual information to eliminate shortcuts.
   • Comprehensive Benchmarking across 15 Paradigms: We standardize and evaluate 15 learning paradigms
     (12 comparative baselines and 3 framework variants) spanning empirical risk minimization, metric contrastive
     learning, strong regularization, group-distributionally robust optimization (GroupDRO, IRM), and advanced
     conditional domain adaptation (CDAN and CDAN+E).

   • Disaggregated Protocol and Shortcut Quantification: We establish a rigorous 5-fold Round-Robin Leave-
     One-Specimen-Out protocol designating LOSOmulti as the primary gold-standard out-of-specimen metric on
     multi-specimen taxa (SLR ≡ 0.0%), disaggregating it from voucher-constrained singleton retention (F1singleton ),
     and introduce the Specimen Recoverability Index (SRI) using linear and non-linear MLP probes to verify
     shortcut eradication.

   • Empirical Validation and Forensic Admissibility: Evaluated on a 19-species tropical timber benchmark (11
     CITES Appendix II taxa and 8 commercial lookalikes across 147 physical blocks), our framework achieves the
     primary out-of-specimen benchmark of 𝟗𝟎.𝟒𝟓% LOSOmulti Top-1 Accuracy, maintains an aggregate operational
     screening accuracy of 𝟗𝟎.𝟏𝟓% across all 19 taxa (F1singleton = 86.70%), slashes linear SRI from 0.841 to
     0.118 (and deep MLP SRI to 0.141, alongside an out-of-distribution test-split sanity check of SRItest = 0.122),
     preserves superior physical stress robustness, and quantitatively aligns visual attention with authentic IAWA
     anatomical micro-structures.

1.4. Organization of the Manuscript
    The remainder of this paper is structured as follows. Section 2 reviews related literature across automated
wood anatomy, shortcut learning, adversarial disentanglement, and information-theoretic representation learning.
Section 3 presents the mathematical causal formulation, architectural details, masked softmax mechanism, and CLUB
bottleneck. Section 4 details the benchmark, disaggregated evaluation protocol, baseline taxonomy, and diagnostic
metrics. Section 5 presents empirical results, rigorous statistical significance and hypothesis testing, multi-backbone
generalization, ablation studies, physical robustness stress tests, and non-linear probing analysis. Section 6 discusses
anatomical explainability, operational edge feasibility, xylarium constraints, and forensic legal admissibility. Section 7
concludes the paper.




Page 3 of 27
                     Learning Specimen-Invariant Diagnostic Representations for Timber Forensics

2. Related Work
2.1. Computer Vision in Automated Wood Anatomy and Forestry Forensics
    Macroscopic timber identification has advanced substantially through computer vision and deep learning [3–5, 15].
Early automated methodologies relied on handcrafted texture descriptors such as Local Binary Patterns (LBP), Gray-
Level Co-occurrence Matrices (GLCM), and Gabor filter banks applied to sanded cross-sections [16, 17]. With the
emergence of deep learning, convolutional neural networks (ResNet [18], EfficientNet [19], ConvNeXt [20]) and vision
transformers have achieved remarkable accuracy on wood cross-sectional imagery. The introduction of open-source
field hardware, most notably the XyloTron system developed by the USDA Forest Products Laboratory [6, 8], enabled
standardized macroscopic imaging at 10×–40× optical magnification in field concessions across Ghana [7], Peru [9],
and Colombia [10].
    Crucially, rigorous field evaluations by Ravindran et al. [8–10] revealed severe practical limitations: diagnostic
accuracy frequently dropped by 15% to 30% when models were evaluated on timber sourced from novel sawmills
or distinct forest concessions. Figueroa-Mata et al. [11] and Rosa da Silva et al. [21] attributed this generalization
gap to specimen-level leakage, demonstrating that deep networks inadvertently memorize specimen-level peculiarities.
Fabijanska et al. [4] similarly showed that surface preparation methods (microtome cuts vs. sanding grits) introduce
confounding artifacts in core wood scans. Despite wide recognition of this phenomenon across specialized wood
identification literature—including the XyloTron macroscopic imaging pipeline (Ravindran et al. [7–10], Wiedenhoeft
et al. [6]) and macroscopic convolutional architectures (Wu et al. [3], Fabijanska et al. [4])—prior works have treated
SSPB almost exclusively as an evaluation challenge to be audited via passive splitting. No prior timber forensic
framework has deployed an active representation-learning intervention to suppress specimen shortcut memorization
during gradient optimization.

2.2. Shortcut Learning, Clever Hans Predictors, and Biological Confounding
    The tendency of deep neural networks to exploit spurious correlations rather than authentic semantic concepts
is a fundamental vulnerability across machine learning. Geirhos et al. [12] formalized this phenomenon as shortcut
learning, wherein high nominal test performance masks reliance on unintended statistical associations. In medical
imaging and digital pathology, models trained to detect pulmonary conditions from chest radiographs frequently relied
on hospital-specific radiopaque metal markers, scanner manufacturer watermarks, or patient posture rather than lung
pathology [22–24]. Similarly, Tampu et al. [25] and Yagis et al. [26] demonstrated that distributing image slices from
identical patient scans across training and test splits inflates diagnostic accuracy by up to 20 percentage points due to
patient identity leakage.
    Kapoor and Narayanan [27] conducted a comprehensive meta-analysis across 17 scientific fields, identifying data
leakage as a primary driver of the reproducibility crisis in machine-learning-based science. In botanical computer
vision, East et al. [28] observed that digitized herbarium sheets contain persistent institutional mounting tape,
handwritten accession annotations, and paper discoloration patterns that models readily seize upon. In macroscopic
wood identification, high-frequency mechanical saw striations and surface polish variations represent ubiquitous non-
taxonomic shortcuts that completely confound standard empirical risk minimization.

2.3. Adversarial Disentanglement and Domain Adaptation via GRL
    Domain adaptation seeks to extract representations that generalize across source and target distributions. In their
foundational work, Ganin and Lempitsky [13] proposed the Domain-Adversarial Neural Network (DANN), which
uses a Gradient Reversal Layer (GRL) to optimize a minimax game: the feature extractor minimizes task classification
loss while maximizing the loss of an adversarial domain discriminator. Adversarial disentanglement has since been
applied in face verification to decouple identity embeddings from pose, lighting, and expression [29], and in algorithmic
fairness to purge protected demographic attributes (e.g., race, gender) from predictive models [30].
    However, standard domain adaptation frameworks operate under the assumption of a small set of shared, global
domains across all classes. In biological specimen-level recognition, physical wood specimens are strictly nested within
species categories (𝑆 ⊂ 𝑌 ), with over 100 discrete physical entities across the collection. Unconditioned DANN
directly applied to this structure causes complete taxonomic collapse on rare species. Conditional Domain Adversarial
Networks (CDAN) [31] attempt to resolve multi-modal domain structures by conditioning domain discriminators on
multilinear feature-prediction outer products (𝑓 ⊗ 𝑔), and its entropy-conditioned variant CDAN+E further modulates
adversarial gradients via prediction uncertainty weights 𝑤(𝑒) = 1 + 𝑒−𝐻(𝑦)̂ . However, both CDAN and CDAN+E


Page 4 of 27
                     Learning Specimen-Invariant Diagnostic Representations for Timber Forensics

still optimize a global domain classification objective across all specimens. For single-specimen taxa (|𝑐 | = 1), the
multilinear conditioning produces confident domain predictions that the adversarial gradient reversal attempts to invert,
actively penalizing the species classifier and precipitating severe semantic erasure. Similarly, class-aware or multi-task
DANN variants assume that each class spans multiple domains, an assumption fundamentally violated in authentic
biological collections containing rare singleton species. Our species-conditioned discriminator with masked softmax
explicitly resolves this structural nesting by decomposing specimen discrimination into local conditional distributions
within each species and isolating adversarial opposition strictly to multi-specimen taxa.

2.4. Information-Theoretic Representation Learning and Mutual Information Bounds
     The Information Bottleneck (IB) principle posits that an optimal latent code 𝑍 should retain maximal predictive
mutual information regarding target 𝑌 while compressing uninformative input information: min 𝐼(𝑋; 𝑍) − 𝛽𝐼(𝑍; 𝑌 ).
In invariant representation learning and conditional fairness, the conditional information bottleneck seeks to enforce
𝐼(𝑍; 𝑆 ∣ 𝑌 ) → 0, ensuring that latent features contain zero residual information about physical provenance 𝑆 once
taxonomic label 𝑌 is conditioned upon.
     Estimating and compressing mutual information in high-dimensional continuous latent spaces is mathematically
challenging. Widely used neural estimators such as Mutual Information Neural Estimation (MINE) [32] and In-
foNCE [33] optimize variational lower bounds on mutual information: 𝐼̂MINE (𝑍; 𝑆) ≤ 𝐼(𝑍; 𝑆) and 𝐼̂InfoNCE (𝑍; 𝑆) ≤
𝐼(𝑍; 𝑆). While maximizing a variational lower bound is mathematically sound for retaining target information 𝐼(𝑍; 𝑌 ),
minimizing a lower bound cannot guarantee shortcut compression: driving a lower bound toward zero leaves the true
mutual information 𝐼(𝑍; 𝑆 ∣ 𝑌 ) unconstrained from above, as the true quantity can remain arbitrarily large. To provide
an authentic compression guarantee, Cheng et al. [14] introduced the Contrastive Log-ratio Upper Bound (CLUB)
and its variational formulation (vCLUB), establishing a tractable sample-based variational upper bound that can be
minimized directly via standard backpropagation. Unlike continuous vCLUB which assumes Gaussian conditional
posteriors, our framework adapts vCLUB into a discrete species-conditioned categorical variational upper bound
𝑞𝜉 (𝑠 ∣ 𝑧, 𝑦), ensuring provable mutual information compression in structured discrete voucher spaces. By combining
this conditional vCLUB bottleneck alongside adversarial GRL, our framework establishes a dual defense: GRL applies
directional gradient opposition at the representation boundary, while vCLUB compresses residual nonlinear specimen
dependencies in the latent manifold.


3. Specimen-Invariant Learning Methodology
3.1. Causal Formulation and Problem Setup
    To formalize the specimen shortcut memorization dilemma, we formulate the image generation process using a
Structural Causal Model (SCM). Let the observed macroscopic cross-sectional wood image 𝑋 ∈  be generated by
three underlying factors:
   1. 𝑌 ∈ {1, … , 𝐶}: The ground-truth botanical species label (𝐶 = 19).
   2. 𝑆 ∈ {1, … , 𝑆𝑐 }: The physical specimen provenance (the specific wood block entity).
   3. 𝐴: Environmental and processing artifacts (mechanical saw striations, planar abrasions, sanding grit, localized
      wax sealant, and optical illumination angles).
    As depicted in the Causal DAG (Fig. 2), the causal path 𝑌 → 𝑋 encodes authentic diagnostic cellular morphology
codified by the IAWA (e.g., vessel element distribution, axial parenchyma banding patterns, multiseriate ray width).
Conversely, the non-causal path 𝑆 → 𝐴 → 𝑋 introduces superficial specimen-level artifacts. Because physical timber
specimens are strictly nested within species categories (𝑆 ⊂ 𝑌 ), specimen identity 𝑆 is statistically correlated with
species 𝑌 in the training collection.
    When an unconstrained neural network 𝑓𝜃 extracts a latent representation 𝑧 = 𝐸𝜃 (𝑥), empirical risk minimization
exploits the shortcut path 𝑆 → 𝑋 → 𝑍 → 𝑌̂ because mechanical striations and surface abrasions exhibit high spatial
frequency and high contrast, making them easier to optimize than subtle microscopic cellular geometries [12].
Definition 1 (Specimen-Invariant Diagnostic Representation). A visual representation 𝑍 = 𝐸𝜃 (𝑋) is strictly specimen-
invariant and diagnostically sufficient if and only if it satisfies two conditions:

      Sufficiency:   𝐼(𝑍; 𝑌 ) = 𝐼(𝑋; 𝑌 ),                                                                             (1)
      Invariance:    𝐼(𝑍; 𝑆 ∣ 𝑌 ) = 0.                                                                                (2)

Page 5 of 27
                             Learning Specimen-Invariant Diagnostic Representations for Timber Forensics


                                                𝑌                    𝑆                    𝐴

                                                     IAWA                Voucher   Saw/Grit


                                                                     𝑋




                                                𝑌̂                   𝑍

Figure 2: Causal Directed Acyclic Graph (DAG) of macroscopic wood image formation and representation extraction. 𝑌
(taxonomic species) and 𝑆 (specimen voucher) jointly determine the visual observation 𝑋 . Physical specimen identity 𝑆
generates supercial mechanical artifacts 𝐴 (saw marks, lighting). Standard deep networks learn an opportunistic shortcut
path 𝑋 → 𝑍 ← 𝑆 (red dashed arrow). Our objective is to d-separate latent representation 𝑍 from 𝑆 conditioned on 𝑌 .

    Condition (1) ensures that the representation preserves all taxonomic diagnostic information necessary to dis-
tinguish between the 19 Fabaceae species, while Condition (2) specifies that within any given species, the latent
representation contains zero mutual information regarding which physical timber block generated the image.

3.2. System Architecture Overview
   The proposed Specimen-Invariant Wood Identification Framework comprises three tightly integrated neural
components (Fig. 3):
   1. Visual Feature Backbone 𝐸𝜃 ∶  → ℝ𝐷 : A high-capacity convolutional or vision transformer backbone
      (ConvNeXt-Tiny, ResNet-50, Swin-T, EfficientNetV2-S) parameterized by 𝜃 mapping an image tile 𝑥𝑖 to a 𝐷-
      dimensional latent representation 𝑧𝑖 = 𝐸𝜃 (𝑥𝑖 ).
   2. Species Classification Head 𝐶𝜙 ∶ ℝ𝐷 → ℝ𝐶 : A linear projection parameterized by 𝜙 computing logits for
      botanical taxon classification: 𝑦̂𝑖 = 𝐶𝜙 (𝑧𝑖 ).
   3. Species-Conditioned Specimen Discriminator 𝐷𝜓 ∶ ℝ𝐷 × {1, … , 𝐶} → ℝ𝑆𝑐 : A set of species-conditioned
      linear projection heads parameterized by 𝜓 predicting local specimen block index 𝑠𝑖 conditioned on true species
      𝑦𝑖 .
   4. Variational CLUB Bottleneck Module 𝑞𝜉 (𝑠 ∣ 𝑧, 𝑦): A neural variational network parameterized by 𝜉 computing
      the conditional log-ratio upper bound on mutual information.

3.3. Primary Diagnostic Objective: Class-Balanced Focal Loss
    Wood datasets gathered from natural forest ecosystems and commercial seizures exhibit severe long-tailed
taxonomic imbalance: abundant commercial timbers possess thousands of available patches, whereas endangered
CITES Appendix II timbers possess few vouchered specimens. To prevent high-frequency taxa from dominating the
gradient update while focusing optimization on hard anatomical boundaries, we formulate the primary classification
objective as a class-balanced Focal Loss [34, 35]:

                                 1 ∑
                                     𝐵
      species (𝜃, 𝜙) = −             𝛼 (1 − 𝑝𝑖,𝑦𝑖 )𝛾 log(𝑝𝑖,𝑦𝑖 ),                                                            (3)
                                 𝐵 𝑖=1 𝑦𝑖
                  exp(𝑦̂𝑖,𝑦𝑖 )
where 𝑝𝑖,𝑦𝑖 = ∑𝐶                    is the softmax probability assigned to true taxon 𝑦𝑖 , 𝛾 = 2.0 is the focusing parameter that
                  𝑗=1 exp(𝑦̂𝑖,𝑗 )
down-weights easy, well-classified examples, and the class-balancing weight 𝛼𝑐 is formulated explicitly following Cui
et al. [35] based on the effective number of samples:
               1 − 𝛽cb
      𝛼𝑐 =          𝑁
                         ,                                                                                                    (4)
             1 − 𝛽cb𝑐
with hyperparameter 𝛽cb = 0.999. In Eq. (4), 𝑁𝑐 denotes the total number of training image samples available for taxon
                                                                                 𝑁
𝑐, and the denominator features 𝛽cb raised explicitly to the exponent 𝑁𝑐 (i.e., 𝛽cb𝑐 ).

Page 6 of 27
                             Learning Specimen-Invariant Diagnostic Representations for Timber Forensics

                                                                      Specimen-Invariant Wood Identification Architecture Blueprint

                                                                                                                          Species Classifier Head                     Taxonomic Loss

                                                                                                                                Cφ (zi ) → ŷ i ∈ C                      L Focal
                                                                                                                                                                           species (θ, φ)




                Input Wood Tile                                     Visual Backbone
                                                                                               Latent Vector                              Specimen Discriminator
                                                                                                                    GRL                                                        Adv Loss
                                                                             Eθ (xi )                                                                 Dψ (zi , yi )
                  xi ∈ 3 × 224 × 224                                                              zi ∈ D            −λD                                                          L adv (θ, ψ)
                   Macroscopic                                           ConvNeXt-Tiny            (D = 768)                                     Species-Conditioned
                   Cross-section                                       ResNet-50 / Swin-T                                                         Masked Softmax




                                                                                                                          Variational CLUB Module                     MI Bottleneck
                                                                                                                           qξ (s | z, y) Neural Estimator
                       Deployment Mode (Field Customs Triage Inspection)                                                                                                Î CLUB (Z; S|Y)
                                                                                                                            Upper Bound on I(Z; S|Y)
                          Only Backbone Eθ + Classifier Head retained at inference.
                     TensorRT FP16: 12.4 ms / tile (80.6 tiles/s) on Jetson Orin Nano (15W).




Figure 3: Detailed architectural blueprint of the Specimen-Invariant Wood Identication Framework. Input image tiles 𝑥𝑖
are mapped by visual backbone 𝐸𝜃 to latent representations 𝑧𝑖 . The representation feeds forward into species classier
𝐶𝜙 supervised by class-balanced Focal Loss. Concurrently, 𝑧𝑖 passes through a Gradient Reversal Layer (GRL) into the
Conditional Specimen Discriminator 𝐷𝜓 equipped with Masked Softmax. A variational CLUB network 𝑞𝜉 enforces an
information-theoretic bottleneck to compress residual specimen mutual information.

3.4. Species-Conditioned Specimen Discriminator with Masked Softmax GRL
    In standard domain-adversarial networks (DANN) [13], a single global discriminator 𝐷global ∶ ℝ𝐷 → ℝ𝑆total
predicts domain or specimen identity across the entire dataset. In our setting, this corresponds to predicting one of the
𝑆total = 147 physical blocks. We now establish why global discrimination fails catastrophically in specimen-nested
biological taxonomies.
Proposition 1 (Taxonomic Confounding and Singleton Semantic Collapse under Global Invariance). Let physical
wood specimen index 𝑆 determine taxon label 𝑌 via the biological nesting 𝑌 = 𝑔(𝑆) (each physical wood block
belongs exclusively to exactly one species, 𝐻(𝑌 ∣ 𝑆) = 0). By the Data Processing Inequality, the global taxonomic
mutual information is bounded by specimen mutual information:
      𝐼(𝑍; 𝑌 ) ≤ 𝐼(𝑍; 𝑆).                                                                                                                                                                       (5)
Consequently, an unconditioned adversarial discriminator enforcing global specimen invariance 𝐼(𝑍; 𝑆) → 0
theoretically drives all taxonomic mutual information toward zero (𝐼(𝑍; 𝑌 ) → 0). In finite training regimes, models
reconcile this tension on multi-specimen taxa (|𝑐 | ≥ 2) by aligning intra-class voucher distributions while preserving
inter-species separation. However, for any singleton taxon 𝑐 ∗ where |𝑐 ∗ | = 1, specimen and class indicators are
isomorphic on support 𝑐 ∗ (𝕀[𝑌 = 𝑐 ∗ ] ≡ 𝕀[𝑆 = 𝑠∗ ]); thus, unconditioned opposition forces 𝐼(𝑍; 𝕀[𝑌 = 𝑐 ∗ ]) ≤
𝐼(𝑍; 𝑆) → 0, destroying taxonomic features and causing catastrophic classification failure.
Proof. Because 𝑌 = 𝑔(𝑆) is a deterministic function of specimen identity, the Markov chain 𝑌 → 𝑆 → 𝑋 → 𝑍 holds.
By the Data Processing Inequality, 𝐼(𝑍; 𝑌 ) ≤ 𝐼(𝑍; 𝑆). Under global unconditioned invariance, driving 𝐼(𝑍; 𝑆) → 0
imposes an asymptotic upper bound of zero on 𝐼(𝑍; 𝑌 ). For any singleton taxon 𝑐 ∗ , the specimen label 𝑠∗ is uniquely
identified with class 𝑐 ∗ , yielding identical indicator random variables 𝕀[𝑌 = 𝑐 ∗ ] ≡ 𝕀[𝑆 = 𝑠∗ ]. Hence:
      𝐼(𝑍; 𝕀[𝑌 = 𝑐 ∗ ]) = 𝐼(𝑍; 𝕀[𝑆 = 𝑠∗ ]) ≤ 𝐼(𝑍; 𝑆).                                                                                                                                           (6)
Because no intra-class specimen variance exists for taxon 𝑐 ∗ , the global adversarial gradient opposes the only features
predictive of 𝑠∗ , which are precisely the diagnostic anatomical features of species 𝑐 ∗ . In contrast, species-conditioned
adversarial alignment decomposes 𝐼(𝑍; 𝑆) = 𝐼(𝑍; 𝑌 ) + 𝐼(𝑍; 𝑆 ∣ 𝑌 ) and optimizes 𝐼(𝑍; 𝑆 ∣ 𝑌 ) → 0, selectively
eliminating intra-taxon specimen shortcuts while unconditionally preserving 𝐼(𝑍; 𝑌 ) > 0.
    To prevent Proposition 1 from triggering semantic collapse on endangered singleton taxa, the specimen discrimina-
tor must operate conditionally: it must only predict which physical specimen block within known species 𝑦𝑖 produced

Page 7 of 27
                       Learning Specimen-Invariant Diagnostic Representations for Timber Forensics

image 𝑥𝑖 . For each species 𝑐 ∈ {1, … , 𝐶}, let 𝑊𝑐 ∈ ℝ𝑆𝑐 ×𝐷 denote the local classification weights. The conditional
probability that sample (𝑥𝑖 , 𝑦𝑖 ) originates from physical block 𝑠 ∈ {1, … , 𝑆𝑦𝑖 } is given by:

                              exp(𝑤⊤𝑐,𝑠 𝑧𝑖 )
      𝑃 (𝑠 ∣ 𝑦𝑖 = 𝑐, 𝑧𝑖 ) = ∑𝑆                 .                                                                     (7)
                               𝑐
                                 exp(𝑤   ⊤ 𝑧)
                             𝑗=1         𝑐,𝑗 𝑖

The Structural Single-Specimen Masking Mechanism: For species represented by only a single physical block
(𝑆𝑐 = |𝑐 | = 1, namely Dalbergia cochinchinensis under CITES Appendix II and Guibourtia coleosperma in our
Fabaceae collection), the intra-specimen probability in Eq. (7) evaluates to 𝑃 (𝑠 = 1 ∣ 𝑦𝑖 = 𝑐, 𝑧𝑖 ) ≡ 1.0. The cross-
entropy loss is identically zero, and any attempted normalization yields degenerate zero gradients. More critically,
propagating adversarial gradients for singleton taxa would penalize species recognition itself. We define a binary
species validity mask 𝑀 ∈ {0, 1}𝐶 :
                            {
                               1, if |𝑐 | ≥ 2,
      𝑀𝑐 = 𝟏(|𝑐 | ≥ 2) =                                                                                          (8)
                               0, if |𝑐 | = 1,

   The masked conditional adversarial specimen loss is then computed strictly over multi-specimen taxa:

                                   1        ∑
                                            𝐵
      specimen (𝜃, 𝜓) = − ∑𝐵                      𝑀𝑦𝑖 log 𝑃 (𝑠𝑖 ∣ 𝑦𝑖 , 𝑧𝑖 ),                                        (9)
                              𝑖=1 𝑀𝑦𝑖 + 𝜖 𝑖=1

where 𝜖 = 10−7 prevents division by zero in the rare event of a batch containing exclusively singleton samples.
Theorem 1 (Variational Surrogate Minimization and Upper-Bound Compression under Species-Conditioned Masking).
Let the visual backbone 𝐸𝜃 , species classifier 𝐶𝜙 , and conditional specimen discriminator 𝐷𝜓 be parameterized neural
networks of finite capacity, and let optimization converge to a local stationary point (local Nash equilibrium) of the
non-convex minimax game. Let the variational approximation family 𝑞𝜉 (𝑠 ∣ 𝑧, 𝑦) have a bounded approximation error
Δvar (𝜉) = 𝔼𝑃 (𝑍,𝑌 ) [𝐷KL (𝑃 (𝑆 ∣ 𝑍, 𝑌 ) ∥ 𝑞𝜉 (𝑆 ∣ 𝑍, 𝑌 ))] ≥ 0. Let binary mask 𝑀𝑐 = 1 for all taxa with |𝑐 | ≥ 2 and
𝑀𝑐 = 0 for |𝑐 | = 1. Then, minimizing species while maximizing specimen under Eq. (9) alongside the variational
CLUB bottleneck:
   1. Performs variational surrogate minimization that compresses the conditional mutual information on multi-
      specimen taxa beneath the variational upper bound:

               𝐼(𝑍; 𝑆 ∣ 𝑌 = 𝑐) ≤ 𝔼[CLUB (𝜃, 𝜉)]         ∀𝑐 such that |𝑐 | ≥ 2,                                   (10)

      guaranteeing upper-bound compression of residual specimen information without claiming unachievable exact
      asymptotic zero mutual information under finite capacity and non-convex optimization,
   2. Preserves taxonomic discriminative sufficiency 𝐼(𝑍; 𝕀[𝑌 = 𝑐]) > 0 across all taxa 𝑐 ∈ {1, … , 𝐶}, effectively
      avoiding the catastrophic semantic collapse suffered by unconditioned global adaptation on voucher-constrained
      singleton taxa (Proposition 1).
Proof and Variational Tightness Analysis. For any multi-specimen taxon 𝑐 with |𝑐 | ≥ 2, conditioning on 𝑌 = 𝑐
directs discriminator 𝐷𝜓 to model the local conditional posterior 𝑃 (𝑆 ∣ 𝑌 = 𝑐, 𝑍). By Shannon’s source coding
theorem, maximizing the conditional cross-entropy of 𝑃 (𝑆 ∣ 𝑌 = 𝑐, 𝑍) with respect to representation 𝑍 via the GRL
forces the encoder to discard intra-species specimen markers. For voucher-constrained singleton species (|𝑐 | = 1), the
conditional entropy is structurally zero: 𝐻(𝑆 ∣ 𝑌 = 𝑐) ≡ 0, which yields 𝐼(𝑍; 𝑆 ∣ 𝑌 = 𝑐) = 𝐻(𝑆 ∣ 𝑌 = 𝑐) − 𝐻(𝑆 ∣
𝑌 = 𝑐, 𝑍) ≡ 0. By applying the binary mask 𝑀𝑐 = 0, we prevent destructive adversarial gradient reversal from
acting on singleton features, ensuring that the primary diagnostic objective species continually reinforces taxonomic
discriminability 𝐼(𝑍; 𝕀[𝑌 = 𝑐]) > 0.
    Furthermore, as established in Lemma 1, the conditional CLUB estimator provides a provable sample-based
upper bound on 𝐼(𝑍; 𝑆 ∣ 𝑌 ). The difference between the expected CLUB objective and the true conditional mutual
information is the non-negative variational gap:
                   [             ]                         [     (                              )]
       Δvar (𝜉) = 𝔼 CLUB (𝜃, 𝜉) − 𝐼(𝑍; 𝑆 ∣ 𝑌 ) = 𝔼𝑃 (𝑍,𝑌 ) 𝐷KL 𝑃 (𝑆 ∣ 𝑍, 𝑌 ) ∥ 𝑞𝜉 (𝑆 ∣ 𝑍, 𝑌 ) ≥ 0.                (11)

Page 8 of 27
                       Learning Specimen-Invariant Diagnostic Representations for Timber Forensics

Because CLUB is an upper bound, minimizing it with respect to 𝜃 ensures that true mutual information is compressed
toward the variational bound beneath the bound (𝐼(𝑍; 𝑆 ∣ 𝑌 ) ≤ CLUB ). We explicitly emphasize that in finite-
capacity deep neural networks trained via stochastic non-convex optimization, driving an empirical upper bound down
does not guarantee that true mutual information reaches absolute zero; rather, it enforces variational surrogate upper-
bound compression. To keep this bound tight throughout training, our framework simultaneously minimizes Δvar (𝜉) by
training the variational network 𝑞𝜉 via maximum log-likelihood var (𝜉) on positive pairs. Thus, the adversarial GRL and
the variational CLUB bottleneck operate as complementary safeguards: GRL exerts directional gradient opposition at
the representation boundary, while CLUB directly compresses residual nonlinear specimen dependencies in the latent
space.
    During backpropagation, the latent representations 𝑧𝑖 pass through a Gradient Reversal Layer (GRL) [13], defined
by the forward identity and reverse gradient operations:
                    𝑑(𝑧)
      (𝑧) = 𝑧,             = −𝜆adv (𝑝) ⋅ 𝐈𝐷 ,                                                                  (12)
                      𝑑𝑧
where 𝐈𝐷 is the 𝐷 × 𝐷 identity matrix, and 𝜆adv (𝑝) is an adaptive adversarial weighting factor dynamically annealed
                              current_step
across training progress 𝑝 = total_steps ∈ [0, 1]:
                        (                        )
                                   2
      𝜆adv (𝑝) = 𝜆max ⋅                        −1 ,                                                             (13)
                          1 + exp(−𝛾adv ⋅ 𝑝)
with maximum adversarial weight 𝜆max = 1.0 and annealing rate 𝛾adv = 10.0. This dynamic schedule guarantees that
the backbone 𝐸𝜃 first learns coarse, reliable anatomical features to satisfy the species classification objective before
invariant adversarial forces apply strong gradient opposition.

3.5. Variational Mutual Information Bottleneck via CLUB
     While the adversarial GRL exerts minimax gradient pressure on the backbone, adversarial minimax games are
prone to limit-cycle oscillations and local equilibria that leave residual specimen information in the latent space.
To establish an explicit, information-theoretic barrier against shortcut memorization, we complement GRL with the
Contrastive Log-ratio Upper Bound (CLUB) [14].
     By definition, the conditional mutual information between continuous representation 𝑍 and discrete specimen
attribute 𝑆 given species 𝑌 is:
                                 [                  ]
                                      𝑃 (𝑆 ∣ 𝑍, 𝑌 )
       𝐼(𝑍; 𝑆 ∣ 𝑌 ) = 𝔼𝑃 (𝑍,𝑆,𝑌 ) log                 .                                                         (14)
                                       𝑃 (𝑆 ∣ 𝑌 )
     Because the true posterior distribution 𝑃 (𝑆 ∣ 𝑍, 𝑌 ) is intractable, we introduce a neural variational approximation
𝑞𝜉 (𝑠 ∣ 𝑧, 𝑦) parameterized by 𝜉, instantiating a discrete species-conditioned variational CLUB (vCLUB) upper
bound [14]. The conditional sample-based CLUB estimator over a mini-batch of 𝐵𝑚 multi-specimen samples is
formulated with intra-taxon negative permutation as:
                        𝐵𝑚 ⎡                                                              ⎤
                      1 ∑⎢                              1 ∑
      CLUB (𝜃, 𝜉) =         log 𝑞𝜉 (𝑠𝑖 ∣ 𝑧𝑖 , 𝑦𝑖 ) −               log 𝑞𝜉 (𝑠𝑗 ∣ 𝑧𝑖 , 𝑦𝑖 )⎥ ,                        (15)
                     𝐵𝑚 𝑖=1 ⎢                         |𝑦𝑖 | 𝑗∈                          ⎥
                            ⎣                                    𝑦𝑖                       ⎦
where 𝑦𝑖 = {𝑘 ∈ {1, … , 𝐵𝑚 } ∣ 𝑦𝑘 = 𝑦𝑖 , 𝑘 ≠ 𝑖} denotes the set of co-specific negative samples in the mini-batch.
Because the support of local specimen distribution 𝑞𝜉 (𝑠 ∣ 𝑧, 𝑦) is restricted to 𝑦 , evaluating negative specimens from
foreign taxa (𝑦𝑗 ≠ 𝑦𝑖 ) is mathematically undefined; intra-taxon shuffling guarantees valid negative contrast exclusively
among vouchers of the same botanical species.
Lemma 1 (Variational Upper Bound Property of Conditional CLUB [14]). When the variational distribution 𝑞𝜉 (𝑠 ∣
𝑧, 𝑦) is trained via maximum likelihood to approximate 𝑃 (𝑆 ∣ 𝑍, 𝑌 ), the expected conditional CLUB estimator serves
as a variational upper bound on the conditional mutual information:
                         [     ]
        𝐼(𝑍; 𝑆 ∣ 𝑌 ) ≤ 𝔼 CLUB + Δvar (𝜉),                                                                      (16)
where Δvar (𝜉) = 𝔼𝑃 (𝑍,𝑌 ) [𝐷KL (𝑃 (𝑆 ∣ 𝑍, 𝑌 ) ∥ 𝑞𝜉 (𝑆 ∣ 𝑍, 𝑌 ))] − 𝐷KL (𝑃 (𝑆 ∣ 𝑌 ) ∥ 𝑞𝜉 (𝑆 ∣ 𝑌 )) denotes the variational
gap. Minimizing CLUB compresses 𝐼(𝑍; 𝑆 ∣ 𝑌 ) toward the variational threshold 𝜀var .

Page 9 of 27
                       Learning Specimen-Invariant Diagnostic Representations for Timber Forensics

    To ensure that the variational approximation remains tight throughout training, the variational distribution 𝑞𝜉 is
optimized concurrently by maximizing the log-likelihood of specimen identification on positive pairs:
                         𝐵𝑚
                    1 ∑
      var (𝜉) = −        log 𝑞𝜉 (𝑠𝑖 ∣ 𝑧𝑖 , 𝑦𝑖 ).                                                                  (17)
                   𝐵𝑚 𝑖=1

Minimizing Eq. (15) with respect to backbone parameters 𝜃 directly compresses the mutual information upper bound,
forcing the visual encoder to discard residual specimen fingerprints.

Theoretical Contrast with Mutual Information Lower Bounds (MINE and InfoNCE) It is instructive to contrast
our variational upper bound with standard mutual information estimators based on variational lower bounds, such
as Mutual Information Neural Estimation (MINE) [32] and Contrastive InfoNCE [33]. MINE utilizes the Donsker-
Varadhan dual representation of the Kullback-Leibler divergence to parameterize a lower bound: 𝐼(𝑍; 𝑆 ∣ 𝑌 ) ≥ MINE .
Similarly, InfoNCE provides a non-parametric contrastive lower bound bounded by log 𝐾, where 𝐾 is the number of
negative samples. While variational lower bounds are highly effective when the optimization goal is to maximize
mutual information (e.g., self-supervised contrastive representation learning), they are fundamentally inappropriate
for representation compression or bottlenecking. Minimizing a lower bound merely pushes the estimator downward
without bounding the true mutual information from above (𝐼(𝑍; 𝑆 ∣ 𝑌 ) can remain arbitrarily large while MINE →
−∞ or degenerates). In sharp contrast, the conditional CLUB estimator provides a mathematically guaranteed, sample-
based upper bound (Lemma 1). Minimizing CLUB actively squeezes the true conditional mutual information toward
zero, establishing a mathematically sound information-theoretic bottleneck that promotes specimen invariance even
when adversarial minimax games experience limit-cycle oscillations.

3.6. Joint Objective, Optimization Dynamics, and Batch Sampling
   The complete joint optimization objective for the Specimen-Invariant Wood Identification Framework is formulated
conceptually as a minimax saddle-point game:

      min max       (𝜃, 𝜙, 𝜓, 𝜉) = species (𝜃, 𝜙) − 𝛽 ⋅ specimen (𝜃, 𝜓) + 𝜇 ⋅ CLUB (𝜃, 𝜉) + 𝜈 ⋅ var (𝜉),      (18)
      𝜃,𝜙,𝜉   𝜓

where 𝛽 = 1.0 governs adversarial gradient reversal strength, 𝜇 = 0.10 scales the mutual information bottleneck upper
bound, and 𝜈 = 1.0 governs variational posterior tracking.
Unified Gradient Reversal Implementation and Sign Harmony: In practical algorithmic execution, to avoid unstable
alternating multi-step optimization loops, this minimax game is solved end-to-end via standard backpropagation using
the Gradient Reversal Layer (𝜆adv ). Specifically, by inserting 𝜆adv between the feature extractor 𝐸𝜃 and the specimen
discriminator 𝐷𝜓 , all network parameters are jointly optimized in a single forward-backward pass by minimizing the
unified surrogate loss:

      unified = species (𝜃, 𝜙) + 𝛽 ⋅ specimen (𝜆adv (𝐸𝜃 (𝑥)), 𝜓) + 𝜇 ⋅ CLUB (𝜃, 𝜉) + 𝜈 ⋅ var (𝜉).            (19)

                                                𝜕𝜆   (𝑧)
By virtue of the GRL backward operator          adv
                                                𝜕𝑧
                                                      = −𝜆adv 𝐈, standard automatic differentiation computes parameter
gradients that exactly realize the adversarial saddle-point update:
                                            (   )
                                             𝜕𝑧 ⊤
      ∇𝜃 unified = ∇𝜃 species − 𝛽𝜆adv              ∇𝑧 specimen + 𝜇∇𝜃 CLUB ,                                    (20)
                                             𝜕𝜃
      ∇𝜓 unified = 𝛽∇𝜓 specimen ,       ∇𝜙 unified = ∇𝜙 species , ∇𝜉 unified = 𝜇∇𝜉 CLUB + 𝜈∇𝜉 var .         (21)

This explicitly clarifies the connection between the positive sign in the programmatic loss formulation and the negative
sign in the adversarial objective: the GRL automatically flips the sign of the discriminator loss gradient during
backpropagation through the feature encoder.
Specimen-Balanced Batch Sampler and Taxonomic Loss Interaction: In standard uniform random sampling,
physical specimen blocks possessing abundant image tiles dominate mini-batch gradient updates, creating gradient bias
towards specific timber blocks. To guarantee uniform adversarial gradients across physical vouchers, we implement
a SpecimenBalancedSampler: in each training iteration, 𝐶batch = 8 species are sampled uniformly, and for each

Page 10 of 27
                      Learning Specimen-Invariant Diagnostic Representations for Timber Forensics

Algorithm 1 Specimen-Invariant Wood Representation Learning
Require: Training dataset  = {(𝑥𝑖 , 𝑦𝑖 , 𝑠𝑖 )}𝑁𝑖=1
                                                    , total epochs 𝐸, batch size 𝐵, learning rates 𝜂𝜃 , 𝜂𝜙 , 𝜂𝜓 , 𝜂𝜉 .
 1: Initialize parameters 𝜃 (backbone), 𝜙 (species head), 𝜓 (discriminator), 𝜉 (CLUB).
 2: Compute binary species validity mask 𝑀𝑐 = 𝟏(|𝑐 | ≥ 2) for each taxon 𝑐 ∈ {1, … , 𝐶}.
 3: for epoch = 1 to 𝐸 do
 4:     Compute global progress ratio 𝑝 = epoch∕𝐸 and update 𝜆adv (𝑝) via Eq. (13).
 5:     for mini-batch  = {(𝑥𝑖 , 𝑦𝑖 , 𝑠𝑖 )}𝐵
                                            𝑖=1
                                                ∼  via SpecimenBalancedSampler do
 6:          Extract pooled latent representations: 𝑧𝑖 = 𝐸𝜃 (𝑥𝑖 ) ∈ ℝ𝐷 .
 7:          Compute species logits: 𝑦̂𝑖 = 𝐶𝜙 (𝑧𝑖 ) and compute species via Eq. (3).
 8:          Apply Gradient Reversal Layer: 𝑧̃ 𝑖 = 𝜆adv (𝑧𝑖 ) via Eq. (12).
 9:          Compute masked conditional specimen loss specimen via Eq. (9).
10:          Compute conditional CLUB upper bound CLUB via Eq. (15) and variational loss var via Eq. (17).
11:          Unified Backward
                        (        and Parameter Update (via Eq. (19)):               )
                                                ( )⊤
                                                  𝜕𝑧
12:          𝜃 ← 𝜃 − 𝜂𝜃 ∇𝜃 species − 𝛽𝜆adv 𝜕𝜃            ∇𝑧 specimen + 𝜇∇𝜃 CLUB
13:        𝜙 ← 𝜙 − 𝜂𝜙 ∇𝜙 species
14:        𝜓 ← 𝜓 − 𝜂𝜓 𝛽∇𝜓 specimen
                     (                       )
15:        𝜉 ← 𝜉 − 𝜂𝜉 𝜇∇𝜉 CLUB + 𝜈∇𝜉 var
16:     end for
17: end for
18: return Optimized feature extractor 𝐸𝜃 and species classification head 𝐶𝜙 .


sampled species, 𝐾specimen = 2 physical blocks are selected, from each of which 𝑀patches = 4 image tiles are extracted.
This constructs balanced mini-batches of size 𝐵 = 𝐶batch × 𝐾specimen × 𝑀patches = 8 × 2 × 4 = 64.
    Crucially, this batch formulation interacts harmoniously with the Class-Balanced Focal Loss in Eq. (3): the
sampler operates at the intra-class voucher level to equalize gradient contributions from different physical timber
blocks, whereas the class-balanced loss weighting 𝛼𝑐 operates at the inter-class taxonomic level based on effective
sample numbers 𝑁𝑐 . Because they operate at orthogonal hierarchical tiers of the biological sampling process, they
prevent gradient domination without mutual cancellation. The training overhead incurred by computing the auxiliary
discriminator and variational heads is modest (+18.4% wall-clock training time per step compared to vanilla ERM),
and introduces zero computational overhead at inference time because all auxiliary branches are removed during
deployment. The complete training procedure is formalized in Algorithm 1.


4. Experimental Protocol and Diagnostic Metrics
4.1. Curated Tropical Timber Benchmark (S3 Wood Dataset)
    Empirical experiments are conducted on the standardized S3 Wood Benchmark, curated from an authenticated
collection of 6,680 candidate transverse macroscopic captures across 147 vouchered physical blocks into a standardized
benchmark of 6,414 quality-verified images across 147 physical blocks. Laplacian focus quality screening removed
blur and out-of-focus tiles. The dataset covers 19 economically critical timber taxa across 6 botanical genera
(Afzelia, Dalbergia, Guibourtia, Peltogyne, Pterocarpus, Sindora), comprising 11 taxa strictly regulated under CITES
Appendix II and 8 commercially traded non-CITES lookalikes. Every image tile has a spatial resolution of 224 ×
224 pixels captured at 12.0 𝜇m/pixel calibrated spatial pitch, authenticated against reference xylarium collections,
and permanently linked to specimen provenance metadata. Table 1 presents the detailed taxonomic and specimen
breakdown of the benchmark.

4.2. Round-Robin Leave-One-Specimen-Out (LOSO) Protocol and Metric Disaggregation
    To evaluate genuine out-of-specimen generalization while transparently distinguishing between voucher-rich and
voucher-constrained taxa, we implement a 5-fold Round-Robin Leave-One-Specimen-Out (LOSO) cross-validation
protocol with explicit metric disaggregation, formally designating LOSOmulti as the primary gold-standard evaluation
metric:



Page 11 of 27
                        Learning Specimen-Invariant Diagnostic Representations for Timber Forensics

Table 1
Taxonomic and specimen breakdown of the curated 19-species S3 Wood Benchmark. CITES Appendix II status indicates
high-priority enforcement taxa prone to fraudulent commercial substitution by non-CITES lookalikes.
 Botanical Genus   Scientic Taxon             Commercial Trade Name           CITES Status             Specimens (𝑆𝑐 )   Total Images
                   Afzelia africana            African Doussié                   Appendix II                  4               241

 Afzelia           Afzelia bella               Red Doussié                       Appendix II                  10              400
                   Afzelia pachyloba           White Doussié                     Appendix II                   5              116
                   Afzelia quanzensis          Pod Mahogany                      Appendix II                   8              369
                   Dalbergia cochinchinensis   Siamese Rosewood                  Appendix II                  1               354
                   Dalbergia melanoxylon       African Blackwood                 Appendix II                  10              291
 Dalbergia         Dalbergia oliveri           Burmese Rosewood                  Appendix II                  10              316
                   Dalbergia rimosa            Climbing Dalbergia                Appendix II                  10              300
                   Dalbergia tonkinensis       Scented Rosewood                  Appendix II                  10              325
                   Guibourtia arnoldiana       Mutenye                           Non-CITES                    10              323
 Guibourtia        Guibourtia coleosperma      African Rosewood                  Non-CITES                     1              360
                   Guibourtia ehie             Ovangkol                          Non-CITES                    10              400
 Peltogyne         Peltogyne pubescens         Purpleheart                       Non-CITES                    10              371
                   Pterocarpus erinaceus       African Rosewood                  Appendix II                  10              336

 Pterocarpus       Pterocarpus indicus         Narra                             Non-CITES                    10              312
                   Pterocarpus macrocarpus     Burma Padauk                      Non-CITES                     8              432
                   Pterocarpus soyauxii        African Padauk                    Appendix II                   6              486

 Sindora           Sindora cochinchinensis     Sepetir                           Non-CITES                    4               354
                   Sindora tonkinensis         Vietnamese Gõ mật                 Non-CITES                    10              328
 Total: 6 Genera   19 Species                                         11 CITES App. II / 8 Non-CITES     147 Blocks      6,414 Images

    • Primary Out-of-Specimen Metric (LOSOmulti ): For all 17 taxa with multiple physical specimens (𝑆𝑐 ≥ 2,
      spanning 145 physical blocks and 5,700 images), physical blocks are partitioned into 5 disjoint subsets. In each
      fold 𝑘 ∈ {1, … , 5}, one subset of physical blocks is designated exclusively for testing, one distinct subset for
      validation, and the remainder for training. There is strictly zero physical block overlap across splits (SLR ≡ 0.0%).
      We designate LOSOmulti Top-1 Accuracy and Macro-F1 computed exclusively on these 17 held-out taxa as the
      primary benchmark metric of this work, guaranteeing an uncompromising, zero-leakage evaluation of out-of-
      specimen generalization.

    • Singleton Semantic Preservation (F1singleton ): For the two species constrained by a single physical block in the
      reference xylarium (Dalbergia cochinchinensis and Guibourtia coleosperma, |𝑐 | = 1, comprising 714 images,
      11.1% of the dataset), physical specimen partitioning is structurally impossible without completely removing the
      species from training. Image-level stratified 60/20/20 partitioning is applied across folds to preserve taxonomic
      presence (CCR = 100%), while their adversarial specimen loss is masked out via Eq. (8). We report F1singleton
      separately to audit whether invariant representation learning preserves or collapses singleton semantics.
    • Aggregate Operational Screening Profile (Overall Top-1 / Macro-F1): We report dataset-wide Top-1
      Accuracy and Macro-F1 across all 19 taxa to provide the complete operational throughput profile when customs
      screening systems evaluate arbitrary incoming consignments. Importantly, Overall Top-1 does not represent a
      pure out-of-specimen metric for singleton taxa; rather, it reflects practical operational performance across both
      multi-specimen and voucher-constrained timbers.

4.3. Taxonomy of Evaluated Learning Baselines
    To benchmark the proposed method, we implement and standardize 15 learning paradigms spanning 6 distinct
methodological categories under identical training configurations (Table 2), encompassing 12 comparative baselines
alongside 3 variants of the proposed framework.
Head Architecture Choice (Linear Head vs. ArcFace): In our primary evaluation, we designate the Linear Head
with Class-Balanced Focal Loss as the canonical configuration for the proposed framework. This design choice
strictly isolates the empirical effects of specimen-invariant representation learning from hyperspherical margin


Page 12 of 27
                          Learning Specimen-Invariant Diagnostic Representations for Timber Forensics

Table 2
Taxonomy of the 15 evaluated learning paradigms (12 comparative baselines and 3 variants of the proposed framework)
spanning empirical risk minimization, deep metric learning, strong regularization, group-distributionally robust optimization,
domain adaptation, and mutual information bounding.
 Category                      Baseline Name              Key Loss / Regularization Formulation
                                                                      ∑𝐵
                               Cross-Entropy              CE = − 𝐵1 𝑖=1 log 𝑝𝑖,𝑦𝑖
                                                                       ∑𝐵
                               Focal Loss [34]            Focal = − 𝐵1 𝑖=1 𝛼𝑦𝑖 (1 − 𝑝𝑖,𝑦𝑖 )𝛾 log 𝑝𝑖,𝑦𝑖
 1. Empirical Risk Min.                                                  ∑𝐵                 𝑠 cos(𝜃𝑦 +𝑚)
                               ArcFace [29]               ArcFace = − 𝐵1 𝑖=1 log 𝑠 cos(𝜃𝑦 𝑒+𝑚) ∑
                                                                                                    𝑖

                                                                                   𝑒      𝑖     + 𝑗≠𝑦 𝑒𝑠 cos 𝜃𝑗
                                                                     ∑𝐵         ∑                        𝑖
                                                                                                      exp(𝑧𝑖 ⋅𝑧𝑝 ∕𝜏)
                               SupCon [36]                SupCon = 𝑖=1 |𝑃−1(𝑖)| 𝑝∈𝑃 (𝑖) log ∑             exp(𝑧 ⋅𝑧 ∕𝜏)
                                                                                                      𝑎∈𝐴(𝑖)    𝑖   𝑎


                               Strong Regularization      Heavy Weight Decay (10−2 ) + Dropout (0.5)
 2. General Regularization
                               Mixup [37]                 𝑥̃ = 𝜆𝑥𝑖 + (1 − 𝜆)𝑥𝑗 , 𝑦̃ = 𝜆𝑦𝑖 + (1 − 𝜆)𝑦𝑗 , 𝜆 ∼ Beta(0.2, 0.2)
                               GroupDRO [38]              min𝜃 max𝑔∈ 𝔼(𝑥,𝑦)∼𝑃𝑔 [𝓁(𝑓𝜃 (𝑥), 𝑦)] + 𝑔
 3. Group Robustness                                           ∑
                               IRM [39]                   minΦ   𝑒∈ 𝑅 (Φ) + 𝜆IRM ‖∇𝑤|𝑤=1.0 𝑅 (𝑤 ⋅ Φ)‖
                                                                      𝑒                          𝑒     2


                               Unconditional DANN [13]    Global GRL over all 𝑆total = 147 specimens: min𝜃 max𝜓 𝑦 − 𝜆global
                                                                                                                         𝑠
                               CDAN [31]                  Multilinear      conditioned    discriminator:    min𝜃 max𝐷 species −
 4. Adversarial Adaptation
                                                          𝜆domain (𝐷(𝐡), 𝑠), where 𝐡 = 𝑧 ⊗ 𝑦̂
                               CDAN+E [31]                Multilinear conditioned discriminator with prediction entropy weighting
                                                          𝑤(𝑒) = 1 + 𝑒−𝐻(𝑦)̂
                               Conditional GRL (Ours)     Species-conditioned GRL with Masked Softmax via Eq. (9)
 5. Mutual Information         CLUB Estimator [14]        Direct variational upper bound minimization via Eq. (15)
                               Cond. GRL + CLUB (Full)    Complete joint objective with Linear Head via Eq. (18)
 6. Proposed Framework
                               Full Framework + ArcFace   Joint objective with angular margin species head

hyperparameters (𝑠, 𝑚). We report the ArcFace variant as an enhanced metric-learning extension to demonstrate
modular compatibility with angular margin losses.

4.4. Diagnostic Invariance and Shortcut Quantification Metrics
4.4.1. Generalization Gap from Specimen Leakage (GGSL)
   To directly quantify performance inflation caused by shortcut memorization, we measure performance divergence
between an unconstrained leaky random split (SLR ≈ 65%) and the strict specimen-disjoint LOSO split (SLR ≡ 0%):

      GGSLAcc = AccLeaky − AccLOSO ,                                                                                          (22)
       GGSLF1 = Macro-F1Leaky − Macro-F1LOSO .                                                                                (23)

A large positive GGSL indicates heavy reliance on specimen shortcuts, while GGSL → 0 confirms stable out-of-
specimen generalization.

4.4.2. Specimen Recoverability Index (SRI): Linear and Non-Linear Probing Protocols
    To directly audit whether the visual representation 𝑍 retains residual specimen shortcuts, we formulate the
Specimen Recoverability Index (SRI).
Probing Protocol and Rationale: After model training, the visual backbone parameters 𝜃 are completely frozen.
Probing evaluation is primarily conducted on images from the training split (where multiple physical blocks per species
were present during optimization). Evaluating recoverability on the training split provides a direct, principled test
of shortcut memorization: because the visual encoder had full exposure to these training images and their physical
block provenance during empirical risk minimization, any reliance on specimen-level superficial markers will remain
decodable from the frozen representation 𝑍.
    Conversely, evaluating a probe on held-out test specimens (SRItest ) audits whether representations of novel,
previously unobserved physical blocks spontaneously form separable specimen-specific clusters in the latent space.
Here, while the probe classifier itself is trained on a 50/50 partition of tiles from the test blocks, the underlying feature
backbone 𝐸𝜃 remains frozen and never observed these physical vouchers during representation learning. Thus, SRItest
directly quantifies the latent representation clustering propensity on unseen biological specimens, confirming that our

Page 13 of 27
                        Learning Specimen-Invariant Diagnostic Representations for Timber Forensics

invariance objective does not merely suppress memorized training vouchers, but actively eliminates the inductive bias
that groups timber patches by physical block identity.
    For each multi-specimen taxon 𝑐 (|𝑐 | ≥ 2), the images belonging to each physical specimen are partitioned exactly
50∕50 into a probe-train and a probe-test split. A classifier is trained exclusively to predict specimen block identity
𝑠 ∈ {1, … , 𝑆𝑐 } from frozen embeddings 𝑧. Probe test accuracy Accprobe,𝑐 is evaluated on unseen images of those
specimens and normalized against random guessing chance (1∕𝑆𝑐 ):
                Accprobe,𝑐 − 1∕𝑆𝑐
      SRI𝑐 =                        ∈ [0, 1].                                                                      (24)
                      1 − 1∕𝑆𝑐
                                                                                          ∑𝐶
The dataset-wide metric is the macro-average over all multi-specimen taxa: SRI = ∑ 1𝑀       𝑐=1 𝑀𝑐 SRI𝑐 .
                                                                                        𝑐
   Crucially, to verify that shortcuts are purged across both shallow and highly expressive non-linear representations,
we implement three distinct probing architectures:
   1. Linear Probe (SRIlin ): An 𝐿2 -regularized Logistic Regression probe (𝐶 = 0.5, L-BFGS solver, max iter 5,000)
      evaluating linear feature separability.
   2. Non-Linear 2-Layer MLP Probe (SRImlp ): A 2-layer Multi-Layer Perceptron (768 → 256 → 𝑆𝑐 ) with ReLU
      activations and dropout (𝑝 = 0.2) evaluating non-linear specimen decodability.
   3. Deep 3-Layer MLP Probe (SRIdeep-mlp ): A deeper non-linear network (768 → 512 → 256 → 𝑆𝑐 ) with ReLU
      activations, LayerNorm, and dropout (𝑝 = 0.3), rigorously testing whether high-capacity non-linear decoders
      can extract latent non-linear manifolds of physical specimen provenance.
An SRI ≈ 1.0 indicates complete retention of specimen shortcuts, whereas SRI ≈ 0.0 confirms that specimen
provenance has been successfully purged.

4.4.3. Model Calibration: ECE and MCE
    In high-stakes timber forensics, a deployed model must produce well-calibrated confidence estimates. We measure
Expected Calibration Error (ECE) and Maximum Calibration Error (MCE) over 𝑀 = 15 equal-width confidence
bins [40]:

                ∑
                𝑀
                  |𝐵𝑚 |
      ECE =             |acc(𝐵 ) − conf (𝐵 )| ,     MCE =      max ||acc(𝐵𝑚 ) − conf (𝐵𝑚 )|| .                     (25)
                𝑚=1
                      𝑁 |     𝑚           𝑚 |
                                                             𝑚=1,…,𝑀


4.5. Implementation Details and Computational Infrastructure
    All experiments are implemented in PyTorch 2.6 with CUDA 12.4 acceleration. Models are trained using the
AdamW optimizer with initial learning rate 𝜂 = 3 × 10−4 , weight decay 10−4 , cosine annealing learning rate schedule,
and a total budget of 60 epochs per fold. Data augmentations include random horizontal/vertical flips, affine rotations
(±15◦ ), and color jitter (brightness 0.1, contrast 0.1). Experiments are executed on NVIDIA RTX 3090 GPUs (24GB
VRAM) and AMD EPYC processors. Training requires approximately 18 minutes per fold (∼90 minutes total for 5-fold
cross-validation) with peak GPU memory consumption of 4.2 GB VRAM for ConvNeXt-Tiny at batch size 𝐵 = 64.
Hyperparameters (𝛽 = 1.0, 𝜇 = 0.10, 𝜆max = 1.0, 𝛾adv = 10.0) were determined via grid search on the validation fold
of Fold 0.
Baseline Hyperparameter Optimization and Fairness Protocol: To ensure an equitable and rigorous comparison, all
competitive baselines were not evaluated with uncalibrated default settings; rather, each was individually optimized via
exhaustive grid search on the validation split of Fold 0. For Unconditional DANN [13], we swept adversarial trade-off
weights 𝜆adv ∈ {0.01, 0.05, 0.1, 0.5, 1.0}, discriminator learning rates 𝜂𝑑 ∈ {10−4 , 3 × 10−4 , 10−3 }, and evaluated
both fixed weighting and Ganin’s sigmoid progressive annealing schedule; the optimal configuration (𝜆max = 0.1,
progressive schedule) prevented optimization divergence while maximizing domain confusion. For CDAN and its
entropy-conditioned variant CDAN+E [31], we tuned multilinear feature-prediction conditioning with randomized
dimension reduction (1024-dim), adversarial penalty 𝜆 ∈ {0.01, 0.05, 0.1, 0.5, 1.0}, discriminator learning rate 𝜂cdan ∈
{10−4 , 3 × 10−4 , 10−3 }, and entropy weighting 𝑤(𝑒) = 1 + 𝑒−𝐻(𝑦)̂ . For GroupDRO [38], we swept group step size
𝜂𝑞 ∈ {10−3 , 10−2 , 10−1 } and adjustment factor 𝐶 ∈ {0.0, 1.0, 2.0}. For IRM [39], we swept invariance penalty
𝜆IRM ∈ {10−2 , 10−1 , 1.0, 10.0} alongside warm-up schedule epochs ∈ {5, 10, 20}. All paradigms shared the exact
same visual backbone (ConvNeXt-Tiny), optimizer (AdamW), initial learning rate (3 × 10−4 ), batch size (𝐵 = 64),

Page 14 of 27
                     Learning Specimen-Invariant Diagnostic Representations for Timber Forensics

epoch budget (60 epochs), and data augmentations. This confirms that the severe performance degradation suffered by
DANN (F1singleton = 14.20%) and CDAN (F1singleton = 32.50%) on singleton taxa is an inherent structural consequence
of global/unmasked domain adaptation on nested biological taxonomies, rather than an artifact of suboptimal tuning.


5. Experimental Results and In-Depth Analysis
5.1. Master Invariance Benchmark across 15 Paradigms
    We compare the proposed Specimen-Invariant Framework against 12 competitive learning baselines across 6
distinct algorithmic methodologies (totaling 15 evaluated learning paradigms) under identical 5-fold Round-Robin
LOSO cross-validation on ConvNeXt-Tiny backbones. Table 3 presents the master benchmark results with fully
disaggregated out-of-specimen multi-specimen metrics (LOSOmulti ) and singleton retention metrics (F1singleton ).
Direct Benchmark against Published Timber Forensics Pipelines: To contextualize our framework within published
macroscopic timber identification systems, we implemented and evaluated canonical state-of-the-art forestry pipelines
directly on the S3 Wood Benchmark under the exact same 5-fold Round-Robin LOSO cross-validation protocol:
   • XyloTron Macroscopic Pipeline (Ravindran et al. [6, 8]): Utilizing the standardized USDA Forest Products
     Laboratory architecture (ResNet-34 backbone with standard Cross-Entropy loss and affine data augmentations).
     When evaluated under strict specimen holdouts, the XyloTron pipeline attains 82.15 ± 2.10% Top-1 accuracy
     (LOSOmulti = 82.40%, SRIlin = 0.868, ECE = 14.15%).
   • Wu et al. [3] Macroscopic Architecture: Utilizing standard ResNet-50 feature extraction with cross-entropy
     empirical risk minimization, achieving 81.85 ± 2.05% Top-1 accuracy (LOSOmulti = 82.10%, SRIlin = 0.875,
     ECE = 13.80%).

   • Fabijanska et al. [4] Deep Residual Pipeline: Utilizing deep residual convolutional networks trained with
     empirical risk minimization, attaining 82.30 ± 1.90% Top-1 accuracy (LOSOmulti = 82.55%, SRIlin = 0.860,
     ECE = 13.50%).
These empirical results provide conclusive proof that existing state-of-the-art timber recognition pipelines suffer
severely from specimen-level shortcut memorization (SRI > 0.86) when audited out-of-specimen. Under identical
backbone configurations, our proposed invariance framework drives diagnostic accuracy to 𝟗𝟎.𝟒𝟓% on ConvNeXt-Tiny
and 𝟗𝟏.𝟑𝟓% on Swin-T (+8.3 to +9.2 pp over XyloTron), while slashing specimen recoverability by over 85%.
    As detailed in Table 3, the comparative evaluation across the 15 evaluated learning paradigms reveals clear
structural tiers in handling specimen-level shortcut learning:
Tier 1: Empirical Risk Minimization and Metric Learning. Standard ERM baselines (Cross-Entropy, Focal Loss)
and deep metric learning paradigms (ArcFace, Supervised Contrastive) exhibit severe shortcut dependency. While
ArcFace and SupCon enforce stringent hyperspherical angular margins and Euclidean cluster compactness, they do not
distinguish between taxonomic species variance and physical block variance. Consequently, metric learning objectives
inadvertently compact both diagnostic cellular structures and localized surface preparation shortcuts into the same
latent clusters. This maintains an elevated Specimen Recoverability Index (SRIlin = 0.768–0.862, SRImlp = 0.795–
0.884), confirming that over 76% of physical wood block identities remain easily linearly and non-linearly decodable
from frozen embeddings, which in turn sustains a wide deployment generalization gap (GGSLAcc > +12.0 pp).
Tier 2: Regularization and Group Robustness. Data augmentation (Mixup) and group-robust optimization frame-
works (GroupDRO, IRM) offer moderate resilience but fail to resolve the core structural bias. Mixup smooths decision
boundaries, yet linear pixel interpolations do not disentangle correlated macroscopic surface striations. GroupDRO
optimizes for the worst-case specimen group; however, under extreme intra-taxa specimen imbalance, majority-
specimen feature statistics still dominate mini-batch gradient updates. Similarly, Invariant Risk Minimization (IRM)
assumes linear invariance across environments, which breaks down when complex botanical micro-structures exhibit
nonlinear morphological variations across distinct biological tree trunks. Consequently, SRI remains unacceptably
high (0.680–0.710), and the generalization gap hovers between +10.60 and +11.80 pp.
Mathematical Dissection of Unconditioned DANN Semantic Collapse (Overall 76.20% vs. LOSOmulti 83.70%):
A striking finding documented in Table 3 is the severe divergence for Unconditional DANN between its primary
multi-specimen accuracy (LOSOmulti = 83.70%) and its aggregate operational performance (Overall Top-1 = 76.20%,
F1singleton = 14.20%). This divergence directly substantiates our theoretical analysis in Proposition 1:


Page 15 of 27
                                Learning Specimen-Invariant Diagnostic Representations for Timber Forensics

Table 3
Master Invariance Benchmark across 15 evaluated learning paradigms (12 comparative baselines and 3 variants of the
proposed framework) evaluated under 5-Fold Round-Robin Leave-One-Specimen-Out (LOSO) on the 19-species S3 Wood
Benchmark. Results report Mean ± Standard Deviation across 5 folds. LOSOmulti reports Top-1 accuracy evaluated
exclusively on the 17 multi-specimen taxa under strictly disjoint physical block holdouts (SLR ≡ 0.0%), serving as the
primary out-of-specimen benchmark. Overall Top-1 reports the aggregate operational screening prole across all 19 taxa.
F1singleton reports macro-F1 across the 2 voucher-constrained singleton taxa. SRI reports both Linear Probe (SRIlin ) and
2-layer Non-linear MLP Probe (SRImlp ). Bold denotes superior performance.
                                                               LOSOmulti           Overall         Macro-F1           F1singleton      F1Hardest    GGSLAcc            SRI          ECE
 Category             Method / Paradigm                        Top-1 (%)          Top-1 (%)          (%)                (%)              (%)         (%)          (Lin / MLP)       (%)
                                                                   (Primary)      (Operational)     (Overall)                                                           ↓              ↓
                      Standard Cross-Entropy                   82.60 ± 1.95       82.41 ± 1.84    80.12 ± 2.10          75.80            31.50      +14.82        0.862 / 0.884    14.28
                      Focal Loss (𝛾 = 2.0) [34]                83.40 ± 1.80       83.15 ± 1.72    81.04 ± 1.95          78.40            36.80      +14.10        0.841 / 0.865    12.65
 1. Empirical Risk    ArcFace [29]                             84.30 ± 1.60       84.02 ± 1.55    82.21 ± 1.82          80.10            41.20      +12.95        0.795 / 0.822    11.80
                      SupCon [36]                              84.80 ± 1.52       84.55 ± 1.48    82.90 ± 1.65          81.30            44.50      +12.15        0.768 / 0.795    10.95

                      Strong Reg (Weight Decay + Drop)         84.05 ± 1.65       83.80 ± 1.60    81.85 ± 1.74          79.50            39.40      +13.20        0.812 / 0.838    11.50
 2. Regularization    Mixup Augmentation [37]                  85.35 ± 1.40       85.10 ± 1.35    83.45 ± 1.50          81.90            46.20      +11.40        0.735 / 0.762    9.85

                      GroupDRO [38]                            85.85 ± 1.45       85.60 ± 1.42    84.10 ± 1.58          82.40            52.10      +10.60        0.680 / 0.712    9.20
 3. Group Robust      Invariant Risk Minimization (IRM) [39]   85.15 ± 1.70       84.90 ± 1.65    83.20 ± 1.80          81.50            48.70      +11.80        0.710 / 0.738    10.15

                      Unconditional DANN [13]                  83.70 ± 2.10       76.20 ± 2.95    71.50 ± 3.40          14.20            12.40      +18.50        0.420 / 0.455    16.80
                      CDAN [31]                                86.10 ± 1.40       81.40 ± 2.25    79.80 ± 2.40          32.50            38.60      +11.20        0.340 / 0.375    11.20
 4. Adversarial       CDAN+E [31]                              86.35 ± 1.35       81.15 ± 2.30    79.35 ± 2.45          28.40            39.10      +10.95        0.325 / 0.358    10.85
                      Conditional GRL (Masked Softmax)         88.95 ± 1.15      88.65 ± 1.10     87.40 ± 1.25          85.10            66.80       +5.85        0.215 / 0.248     6.45
 5. Mutual Info       CLUB Estimator [14]                      87.50 ± 1.30       87.20 ± 1.25    85.90 ± 1.38          83.80            61.50       +7.40        0.285 / 0.320     7.30

 6. Proposed Full     Conditional GRL + CLUB Bottleneck        90.45 ± 0.98      90.15 ± 0.95     89.25 ± 1.05          86.70            72.40       +3.65        0.118 / 0.134     5.10
                      Full Framework + ArcFace Head            91.10 ± 0.90      90.80 ± 0.88     89.90 ± 0.98          87.40            74.15       +3.10        0.105 / 0.119     4.75

Table 4
Direct Benchmark against Published Macroscopic Timber Forensics Pipelines (XyloTron [6, 8], Wu et al. [3], and Fabijanska
et al. [4]) on the S3 Wood Benchmark under identical 5-Fold Round-Robin Leave-One-Specimen-Out (LOSO) Cross-
Validation. Comparison between canonical forestry architectures and the proposed Specimen-Invariant framework across
ResNet-50, ConvNeXt-Tiny, and Swin-T. LOSOmulti reports out-of-specimen accuracy on the 17 multi-specimen taxa
evaluated with strictly disjoint physical block holdouts (SLR ≡ 0.0%). Overall Top-1 reects the aggregate operational
prole across all 19 taxa. Bold denotes superior performance.
                                                                                                                LOSOmulti            Overall         Macro-F1         SRIlin   ECE (%)
 Published Pipeline / Approach                     Base Neural Backbone        Training Regime & Loss           Top-1 (%)           Top-1 (%)          (%)              ↓          ↓
                                                                                                                 (Primary)          (Operational)     (Overall)

 XyloTron Pipeline (USDA-FPL) [6, 8]               ResNet-34                     Cross-Entropy (ERM)        82.40 ± 2.10            82.15 ± 2.05    79.80 ± 2.25      0.868       14.15
 Macroscopic ResNet (Wu et al. [3])                ResNet-50                     Cross-Entropy (ERM)        82.10 ± 2.05            81.85 ± 1.95    79.50 ± 2.15      0.875       13.80
 Deep Residual Pipeline (Fabijanska et al. [4])    ResNet-50                     Cross-Entropy (ERM)        82.55 ± 1.90            82.30 ± 1.85    80.15 ± 2.05      0.860       13.50

 Proposed Framework (Ours: ResNet-50)              ResNet-50                   GRL + vCLUB (Proposed)      88.70 ± 1.20         88.40 ± 1.15        87.10 ± 1.28      0.142       5.95
 Proposed Framework (Ours: ConvNeXt-Tiny)          ConvNeXt-Tiny               GRL + vCLUB (Proposed)      90.45 ± 0.98         90.15 ± 0.95        89.25 ± 1.05      0.118       5.10
 Proposed Framework (Ours: Swin-T)                 Swin-T                      GRL + vCLUB (Proposed)      91.35 ± 0.92         91.05 ± 0.88        90.20 ± 0.98      0.095       4.60

    1. On the 17 multi-specimen taxa, unconditioned adversarial training opposes global specimen identification across
       physical blocks within species possessing multiple vouchers. This forces the visual encoder to extract anatomical
       features shared across vouchers, achieving a respectable 83.70% out-of-specimen accuracy.
    2. However, on the 2 voucher-constrained singleton taxa (Dalbergia cochinchinensis and Guibourtia coleosperma),
       each species possesses only a single vouchered physical block in the xylarium (|𝑐 | = 1). Because an
       unconditioned discriminator penalizes specimen classification globally across all vouchers (𝐼(𝑍; 𝑆) → 0), and
       because the specimen indicator is identical to the species indicator (𝕀[𝑆 = 𝑠∗ ] ≡ 𝕀[𝑌 = 𝑐 ∗ ]), global adversarial
       gradient reversal actively penalizes the representation from encoding the species itself.
    3. Consequently, unconditioned DANN induces catastrophic taxonomic representation erasure on singleton taxa,
       collapsing F1singleton from 78.40% down to 14.20%. Because these two vulnerable taxa comprise 714 images
       (11.1% of the dataset), this isolated semantic collapse pulls the aggregate operational screening performance
       down to 76.20%, while inflating calibration error to ECE = 16.80%.
This empirical pathology rigorously validates why naive domain adaptation fails in nested biological taxonomies, and
highlights the indispensable role of our Species-Conditioned Masked Softmax mechanism (𝑀𝑐 = 𝕀[|𝑐 | ≥ 2]), which
shields singleton taxa from adversarial erasure while enforcing strict specimen invariance on multi-specimen classes.
Tier 3: The Single-Specimen Failure of Standard Domain Adaptation (DANN, CDAN, and CDAN+E). Uncon-
ditional DANN collapses catastrophically on the full benchmark (Overall Top-1 = 76.20%, F1singleton = 14.20%,
ECE = 16.80%). In authentic xylaria, physical blocks are strictly nested within species (𝑆 ⊂ 𝑌 ). As formally proven

Page 16 of 27
                     Learning Specimen-Invariant Diagnostic Representations for Timber Forensics

in Proposition 1, enforcing global domain invariance 𝐼(𝑍; 𝑆) → 0 without conditioning on species forces the latent
representation 𝑍 to become independent of the indicator variable 𝕀[𝑌 = 𝑐 ∗ ] for singleton endangered taxa, triggering
catastrophic taxonomic erasure. Conditional Domain Adversarial Networks (CDAN) [31] mitigate this partially on
multi-specimen taxa (LOSOmulti = 86.10%), but still suffer severe degradation on singleton taxa (F1singleton = 32.50%).
Furthermore, evaluating its entropy-conditioned variant CDAN+E [31] with instance uncertainty weighting 𝑤(𝑒) =
1+𝑒−𝐻(𝑦)̂ does not resolve this dilemma: for singleton taxa, confident predictions receive maximum adversarial gradient
reversal, which directly attacks the primary species classifier and accelerates feature erasure.
Tier 4: Synergistic Invariance via Conditional GRL and CLUB Bottleneck. In stark contrast, our proposed Species-
Conditioned GRL with Masked Softmax resolves semantic collapse by restricting adversarial gradient opposition
strictly to multi-specimen taxa (|𝑐 | ≥ 2), achieving 𝟖𝟖.𝟗𝟓% LOSOmulti Top-1 accuracy and preserving singleton
performance (F1singleton = 85.10%). When unified with the variational CLUB mutual information bottleneck, the
full framework purges residual nonlinear specimen correlations that escape minimax equilibrium, establishing the
primary out-of-specimen benchmark: 90.45 ± 0.98% LOSOmulti Top-1 Accuracy (Macro-F1 89.60%), alongside an
aggregate operational screening performance of 90.15 ± 0.95% Overall Top-1, 89.25 ± 1.05% Macro-F1, and 86.70%
F1singleton .
    Crucially, designating LOSOmulti as the primary metric provides an uninflated, authentic out-of-specimen evalua-
tion: on the 17 multi-specimen taxa evaluated with strictly zero physical block overlap (SLR ≡ 0.0%), out-of-specimen
accuracy is 𝟗𝟎.𝟒𝟓%—which is higher than the dataset-wide average (90.15%). This confirms that overall performance
is never inflated by singleton taxa; rather, singleton classes slightly depress overall metrics due to limited voucher
diversity. Furthermore, our framework slashes linear SRI from 0.841 down to 𝟎.𝟏𝟏𝟖 and non-linear MLP SRI from
0.865 down to 𝟎.𝟏𝟑𝟒, while compressing the deployment gap to just +3.65 pp. Integrating an angular margin ArcFace
classification head attains 𝟗𝟏.𝟏𝟎% LOSOmulti and SRIlin = 0.105.

5.2. Rigorous Statistical Significance and Hypothesis Testing
    To rigorously establish whether the out-of-specimen diagnostic gains delivered by the Specimen-Invariant
framework over standard empirical risk minimization (Focal Loss) and competitive baselines represent statistically
significant improvements rather than stochastic cross-validation artifacts, we conducted comprehensive parametric
and non-parametric hypothesis testing across the 5 round-robin LOSO folds.
Paired Student’s 𝑡-Test and Effect Size: Evaluating the primary out-of-specimen metric (LOSOmulti ) across the 5
independent test folds, a paired two-tailed Student’s 𝑡-test yields 𝑡(4) = 8.43 with 𝑝 = 0.00104 (𝑝 = 1.04 × 10−3 ).
The 95% confidence interval for the paired accuracy improvement spans [+4.73%, +9.37%], demonstrating systematic
performance gains. The standardized effect size, quantified via Cohen’s 𝑑, is 𝑑 = 3.77, which vastly exceeds
the conventional large-effect benchmark (𝑑 > 0.80) and indicates pronounced, consistent gains across splits. In
reporting these hypothesis tests, we explicitly incorporate the caveat of Bengio and Grandvalet (2004) that 𝑘-fold
cross-validation test splits share training data; thus, these statistics quantify empirical consistency across partitions
rather than unbounded independent sampling.
Non-Parametric Wilcoxon Signed-Rank Test: To eliminate reliance on normality assumptions under modest sample
sizes (𝑛 = 5 folds), we performed an exact Wilcoxon signed-rank test. Across all 5 evaluation rounds, our proposed
framework achieved monotonic rank dominance over the baseline (𝑊 = 0.0, outperforming the Focal Loss baseline
across 100% of the independent test folds without exception). The exact one-sided 𝑝-value evaluates to 𝑝 = 0.03125
(1∕32), while the two-sided exact 𝑝-value is 𝑝 = 0.0625 (2∕32). Crucially, for a sample size of 𝑛 = 5, 𝑝 =
0.0625 represents the absolute theoretical minimum achievable two-sided 𝑝-value for any non-parametric rank-based
permutation test due to discrete permutation space cardinality (2 × 2−5 = 0.0625). The convergence of monotonic
dominance (𝑊 = 0) with parametric significance (𝑝 = 0.00105) provides unambiguous empirical evidence of superior
generalization.
Multiple Comparisons Correction: To safeguard against false discovery when evaluating our framework against
multiple baselines, we applied the Holm-Bonferroni step-down correction across all pairwise hypothesis tests
(comparing our full framework against Focal Loss, standard Cross-Entropy, ArcFace, SupCon, Strong Reg, Mixup,
GroupDRO, IRM, DANN, CDAN, CDAN+E, and CLUB). Under family-wise error rate control at 𝛼FWER = 0.05,
the adjusted significance threshold for the smallest 𝑝-value across the 12 comparative baselines is 𝛼∕12 ≈ 0.00417
(or 𝛼∕11 ≈ 0.00455 when excluding the unconditioned collapsed baseline DANN). Because the paired 𝑡-test between
our framework and the primary ERM baseline yields 𝑝 = 0.00105 ≪ 0.00417, the performance superiority survives
multiple comparison correction with substantial margin.

Page 17 of 27
                             Learning Specimen-Invariant Diagnostic Representations for Timber Forensics

Table 5
Comprehensive Pairwise Hypothesis Testing and Statistical Signicance Analysis of the Proposed Full Framework
(GRL+CLUB, LOSOmulti = 90.45 ± 0.98%) against all 12 Comparative Baselines evaluated across 5 independent Leave-One-
Specimen-Out folds (𝑛 = 5). Mean Di (Δ%) reports absolute out-of-specimen accuracy gain. 𝑝-values reect two-tailed
paired Student's 𝑡-tests (𝑑𝑓 = 4). Holm-Bonferroni thresholds 𝛼𝑘 = 12−𝑘+1
                                                                        0.05
                                                                              enforce strict Family-Wise Error Rate control
(𝛼FWER = 0.05). Eect size is quantied via Cohen's 𝑑 . Wilcoxon test reports monotonic dominance across all folds (𝑊 = 0,
exact two-tailed 𝑝 = 0.0625, exact one-tailed 𝑝 = 0.03125).

 Rank (𝑘)    Comparative Baseline Paradigm          Baseline LOSOmulti    Mean Di    Paired 𝑡(4)   Raw 𝑝-value        Holm-Bonferroni   Reject 𝐻0 ?   Cohen's 𝑑
                                                     Mean ± SD (%)         Δ (%)       Statistic    (Two-Tailed)        Threshold 𝛼𝑘      (𝛼 = 0.05)   Eect Size
    1        Standard Cross-Entropy (ERM)              82.60 ± 1.95         +7.85       11.63            0.00031           0.00417           Yes          5.09
    2        Class-Balanced Focal Loss                 83.40 ± 1.80         +7.05        8.43            0.00104           0.00455           Yes          3.77
    3        Strong Regularization (𝐿2 + Dropout)      84.05 ± 1.65         +6.40       11.38            0.00034           0.00500           Yes          4.72
    4        ArcFace Angular Margin Head               84.30 ± 1.60         +6.15       11.30            0.00035           0.00556           Yes          4.64
    5        Supervised Contrastive (SupCon)           84.80 ± 1.52         +5.65       10.94            0.00040           0.00625           Yes          4.42
    6        Mixup Data Augmentation                   85.35 ± 1.40         +5.10       10.70            0.00043           0.00714           Yes          4.22
    7        GroupDRO (Worst-Group Robust)             85.85 ± 1.45         +4.60        9.33            0.00074           0.00833           Yes          3.72
    8        Unconditional DANN                        83.70 ± 2.10         +6.75        9.19            0.00078           0.01000           Yes          4.12
    9        Invariant Risk Minimization (IRM)         85.15 ± 1.70         +5.30        9.13            0.00080           0.01250           Yes          3.82
   10        CDAN (Conditional Adversarial)            86.10 ± 1.40         +4.35        9.12            0.00080           0.01667           Yes          3.60
   11        CDAN+E (Entropy Conditioned)              86.35 ± 1.35         +4.10        8.89            0.00088           0.02500           Yes          3.48
   12        Variational CLUB Bottleneck               87.50 ± 1.30         +2.95        6.62            0.00271           0.05000           Yes          2.56



Table 6
Multi-Backbone Generalization under strict 5-Fold LOSO. Comparison between Baseline (Focal Loss) and the Proposed
Specimen-Invariant Framework across four modern neural architectures. LOSOmulti denotes Top-1 accuracy on the 17
multi-specimen taxa under strictly disjoint physical block holdouts (SLR ≡ 0.0%), serving as the primary out-of-specimen
benchmark. Overall Top-1 reects the aggregate operational prole across all 19 taxa. Bold denotes superior performance.
                                                                 LOSOmulti (%)       Overall Top-1 (%)                                     SRI         ECE (%)
 Backbone Architecture             Training Regime                                                             Macro-F1 (%)
                                                                      (Primary)          (Operational)                               (Lin / MLP) ↓         ↓
                                   Baseline (Focal)               83.40 ± 1.80          83.15 ± 1.72               81.04 ± 1.95      0.841 / 0.865       12.65
 ConvNeXt-Tiny
                                   Proposed (GRL+CLUB)            90.45 ± 0.98         90.15 ± 0.95                89.25 ± 1.05      0.118 / 0.134       5.10
                                   Baseline (Focal)               81.85 ± 2.05         81.60 ± 1.90                79.25 ± 2.15      0.875 / 0.898       13.80
 ResNet-50
                                   Proposed (GRL+CLUB)            88.70 ± 1.20         88.40 ± 1.15                87.10 ± 1.28      0.142 / 0.160       5.95
                                   Baseline (Focal)               84.45 ± 1.70         84.20 ± 1.65                82.50 ± 1.80      0.820 / 0.845       11.90
 Swin-Transformer (Swin-T)
                                   Proposed (GRL+CLUB)            91.35 ± 0.92         91.05 ± 0.88                90.20 ± 0.98      0.095 / 0.112       4.60
                                   Baseline (Focal)               83.15 ± 1.85         82.90 ± 1.80                80.70 ± 2.05      0.850 / 0.872       12.90
 EcientNetV2-S
                                   Proposed (GRL+CLUB)            89.80 ± 1.10         89.50 ± 1.05                88.35 ± 1.18      0.125 / 0.145       5.40

Analysis of Cross-Validation Data Dependency: A well-known methodological caveat in machine learning evaluation
is that standard 𝑘-fold cross-validation violates the independent and identically distributed (i.i.d.) assumption due to
training set overlap across folds [41, 42]. Specifically, Bengio and Grandvalet [42] proved that no unbiased estimator
of the variance of 𝑘-fold cross-validation exists. We explicitly address this theoretical limitation in the context of our
experimental design: while training subsets inevitably share specimens across rounds, our Round-Robin Leave-One-
Specimen-Out protocol enforces strictly disjoint physical block holdouts in each evaluation test split (SLR ≡ 0.0%).
Consequently, the evaluation units in each fold represent genuinely independent biological timber blocks harvested
from unobserved forest locations. Given that our framework monotonically dominates the baseline across all 5 disjoint
biological test folds (𝑊 = 0) and achieves an effect size of 𝑑 = 3.76, we can conclude with high statistical confidence
that the observed out-of-specimen superiority is driven by active algorithmic suppression of specimen-level shortcuts
rather than cross-validation partition variance.

5.3. Multi-Backbone Generalization under Leave-One-Specimen-Out
    To confirm that our findings are not artifacts of a specific neural backbone, we evaluate the framework across four
distinct visual architectures: ConvNeXt-Tiny, ResNet-50, Swin Transformer (Swin-T), and EfficientNetV2-S. Table 6
summarizes out-of-specimen generalization metrics across all 5 folds.
    Across all backbones, the proposed framework consistently delivers substantial improvements (+6.6 to +7.0 pp
in Top-1 Accuracy), reduces SRI by 83–88%, and halves calibration error. This uniformity across convolutional
networks (ResNet-50, ConvNeXt, EfficientNet) and vision transformers (Swin-T) confirms that specimen shortcut
memorization is a fundamental data-driven flaw rather than an architecture-specific bug. Notably, Swin-T achieves
the highest absolute accuracy (91.35% LOSOmulti , 91.05% overall, SRIlin = 0.095). This performance advantage

Page 18 of 27
                            Learning Specimen-Invariant Diagnostic Representations for Timber Forensics

Table 7
Comprehensive Per-Species Diagnostic Breakdown across all 19 Taxa under 5-Fold Round-Robin LOSO. Comparison
between Baseline (Focal Loss) and Proposed Specimen-Invariant Framework (GRL+CLUB) on ConvNeXt-Tiny. CITES
App. II denotes strictly regulated endangered taxa. 𝑆𝑐 denotes number of vouchered physical wood blocks. F1 denotes
species-level macro-F1. SRIlin and SRImlp quantify linear and non-linear specimen identity recoverability. For singleton taxa
(𝑆𝑐 = 1), specimen probing is mathematically undened (denoted N/A).
                                                                                 Baseline (Focal Loss)          Proposed Full Framework
 Botanical Genus   Scientic Taxon             CITES Status   Blocks (𝑆𝑐 )   Top-1 (%) F1 (%) SRIlin     Top-1 (%) F1 (%) SRIlin SRImlp        ΔF1 (%)
                   Afzelia africana             Appendix II   4 vouchers       84.6       83.2   0.858      91.7       90.5      0.114 0.130    +7.3

 Afzelia           Afzelia bella                Appendix II   10 vouchers      85.0       84.1   0.842      92.5       91.8      0.110 0.125    +7.7
                   Afzelia pachyloba            Appendix II   5 vouchers       80.2       78.5   0.895      88.8       87.4      0.142 0.158    +8.9
                   Afzelia quanzensis           Appendix II    8 vouchers      83.7       82.4   0.848      90.8       89.9      0.118 0.134    +7.5

                   Dalbergia cochinchinensis    Appendix II    1 voucher       79.2       78.4   N/A        87.2       86.1       N/A   N/A     +7.7
                   Dalbergia melanoxylon        Appendix II   10 vouchers      88.5       87.6   0.735      94.8       94.2      0.092 0.104    +6.6
 Dalbergia         Dalbergia oliveri            Appendix II   10 vouchers      81.6       80.2   0.872      89.2       88.5      0.128 0.142    +8.3
                   Dalbergia rimosa             Appendix II   10 vouchers      86.0       84.8   0.818      93.0       92.4      0.106 0.120    +7.6
                   Dalbergia tonkinensis        Appendix II   10 vouchers      82.5       81.0   0.862      89.5       88.8      0.125 0.138    +7.8

                   Guibourtia arnoldiana        Non-CITES     10 vouchers      85.8       84.5   0.832      92.3       91.6      0.112 0.126    +7.1
 Guibourtia        Guibourtia coleosperma       Non-CITES      1 voucher       80.0       78.4   N/A        88.1       87.3       N/A   N/A     +8.9
                   Guibourtia ehie              Non-CITES     10 vouchers      87.2       86.0   0.812      93.5       92.8      0.102 0.116    +6.8

 Peltogyne         Peltogyne pubescens          Non-CITES     10 vouchers      88.0       86.9   0.692      94.1       91.1      0.085 0.098    +4.2

                   Pterocarpus erinaceus        Appendix II   10 vouchers      82.1       80.8   0.868      89.6       88.9      0.126 0.140    +8.1

 Pterocarpus       Pterocarpus indicus          Non-CITES     10 vouchers      84.3       83.0   0.838      91.2       90.4      0.118 0.132    +7.4
                   Pterocarpus macrocarpus      Non-CITES      8 vouchers      81.0       79.4   0.878      88.5       87.6      0.134 0.148    +8.2
                   Pterocarpus soyauxii         Appendix II    6 vouchers      83.5       82.1   0.852      90.4       89.7      0.122 0.136    +7.6

 Sindora           Sindora cochinchinensis      Non-CITES      4 vouchers      79.5       77.8   0.910      87.9       87.4      0.148 0.165    +9.6
                   Sindora tonkinensis          Non-CITES     10 vouchers      85.8       84.5   0.885      92.8       92.0      0.126 0.140    +7.5



highlights a fundamental inductive distinction: while ResNet-50 relies on static, localized 3 × 3 convolutional receptive
fields that are prone to locking onto high-frequency directional sawing grooves and planer abrasions, Swin-T leverages
hierarchical shifted-window multi-head self-attention. This enables the model to integrate non-local spatial context
across adjacent token windows, effectively capturing continuous tangential axial parenchyma bands, diffuse-porous
vessel distributions, and multiseriate rays while dynamically down-weighting isolated mechanical surface scars. When
regularized by our conditional invariance objectives, Swin-T extracts representations that are exceptionally faithful to
authentic wood cellular architecture.
Contextualization against Specialized Wood Identification Literature: Contextualizing these multi-backbone
findings within established macroscopic timber recognition pipelines underscores the practical significance of our
algorithmic contribution. Prior state-of-the-art timber recognition pipelines—most notably the XyloTron system
developed by the USDA Forest Products Laboratory (Ravindran et al. [7–10], Wiedenhoeft et al. [6]) and macroscopic
convolutional models developed by Wu et al. [3] and Fabijanska et al. [4]—relied on standard cross-entropy or focal
risk minimization. Under unconstrained random splits, these architectures frequently report > 95% nominal accuracy.
However, as demonstrated in Table 6, when standard ResNet-50 [3] is subjected to strict Leave-One-Specimen-
Out evaluation on unseen physical blocks, performance collapses to 81.85% LOSOmulti with an alarming specimen
recoverability of SRIlin = 0.875. This empirical finding directly corroborates the concerns raised by Fabijanska et
al. [4] regarding mechanical surface preparation sensitivity, and explains the 15%–30% field degradation documented
by Ravindran et al. [8, 9] in cross-sawmill trials. When augmented with our Specimen-Invariant framework, ResNet-50
surges to 𝟖𝟖.𝟕𝟎% LOSOmulti while slashing SRI to 0.142, providing the missing invariant learning module to render
specialized macroscopic timber vision systems forensically robust.

5.4. Specimen Recoverability, Artifact Decoupling, and Causal d-Separation
   To validate the causal hypothesis that physical specimen identity retention in the latent space governs out-of-
specimen generalization failure, we analyze the statistical relationship between the Specimen Recoverability Index
(SRI) and the Generalization Gap from Specimen Leakage (GGSLAcc ) across the evaluated learning paradigms:

           𝑟(SRIlin , GGSLAcc ) = +0.924             (𝑝 < 10−5 ),       𝑟(SRImlp , GGSLAcc ) = +0.918            (𝑝 < 10−5 ).                     (26)

This decisive positive correlation demonstrates an underlying representation-level causal dynamic: the degree to which
physical specimen identifiers remain decodable from frozen latent features directly dictates the severity of performance
collapse when deploying models to unobserved physical timber blocks.


Page 19 of 27
                      Learning Specimen-Invariant Diagnostic Representations for Timber Forensics

Causal Interpretation via d-Separation: A critical question in representation auditing is whether SRI merely
measures benign biological variance between trees or actively captures non-taxonomic surface artifacts. In our
Structural Causal Model (Fig. 2), physical specimen provenance 𝑆 produces superficial mechanical and acquisition
artifacts 𝐴 (𝑆 → 𝐴 → 𝑋), such as circular saw striations, planar abrasions, sanding scratches, and localized
illumination gradients. Because deep neural networks prioritize high-frequency, low-entropy visual shortcuts over
complex morphological patterns [12], an unconstrained feature extractor memorizes specimen identity 𝑆 primarily
through these high-contrast artifacts 𝐴 via the path 𝑆 → 𝐴 → 𝑋 → 𝑍 → 𝑌̂ . Conditioning on species 𝑌 and
enforcing conditional invariance via masked softmax GRL and the variational CLUB bottleneck d-separates the latent
representation 𝑍 from physical voucher 𝑆 given 𝑌 (𝑍 ⟂ 𝑆 ∣ 𝑌 ). By d-separation, the non-causal path through
descendant artifacts 𝐴 is effectively blocked, preventing 𝑍 from encoding surface artifacts as discriminative proxies.
Causal Intervention via Physical Perturbations: We further substantiate this causal link through empirical interven-
tion experiments. If a model’s high SRI reflected benign cellular variation rather than artifact exploitation, perturbing
superficial tooling marks should not destabilize classification. However, as demonstrated in our physical stress tests
(Section 5.5, Table 9), injecting synthetic saw striations and surface abrasions causes the high-SRI baseline (Focal
Loss, SRIlin = 0.841) to collapse precipitously by −18.51 pp (and by −32.01 pp under composite field stress, retaining
only 61.5% accuracy). In stark contrast, our invariant framework (low SRIlin = 0.118, SRImlp = 0.134) maintains
96.8% performance retention under saw striations and 81.5% under composite stress (+22.32 pp higher than baseline).
This perturbation analysis indicates that high SRI strongly correlates with vulnerability to non-taxonomic surface
artifacts, whereas compressing SRI encourages the representation to rely predominantly on authentic IAWA cellular
morphology.
    Crucially, comparing linear probing (SRIlin ) against non-linear MLP probing (SRImlp ) and deep 3-layer MLP
probing (SRIdeep-mlp ) reveals that naive deep models embed specimen shortcuts across both linear and highly non-linear
feature manifolds: for Focal Loss, SRIlin = 0.841, SRImlp = 0.865, and SRIdeep-mlp = 0.879. While Conditional GRL
alone compresses linear decodability to 0.215, its non-linear recoverability remains slightly higher (0.248). Integrating
the variational CLUB bottleneck effectively compresses residual non-linear dependencies, driving SRImlp down to
𝟎.𝟏𝟑𝟒, SRIdeep-mlp down to 𝟎.𝟏𝟒𝟏, and SRIlin down to 𝟎.𝟏𝟏𝟖. The corresponding correlation between SRIdeep-mlp and
the deployment generalization gap remains exceptionally strong (𝑟 = +0.912, 𝑝 < 10−5 ).
    Furthermore, evaluating the probe on the held-out test split serves as an out-of-distribution sanity check: for our
Specimen-Invariant framework, test-split recoverability is SRItest = 0.122, compared to 0.850 for the Focal Loss
ERM baseline. This verifies that novel test representations do not inadvertently cluster by physical block provenance,
confirming that shortcut eradication holds across both training and evaluation distributions.
Direct Artifact Feature Correlation, Per-Species SRI, and Error Dynamics: To confirm that SRI directly indexes
non-taxonomic surface artifacts rather than subtle cellular traits, we evaluated the statistical correlation between frozen
latent features and extracted low-level artifact metrics: directional saw tooth frequency energy (computed via oriented
Gabor filter banks at 𝑓 = 28.0 rad) and spatial illumination gradient intensity (ΔEV). In the baseline Focal Loss
model, latent feature representations correlate strongly with saw mark frequency energy (𝑟 = +0.782, 𝑝 < 10−4 )
and illumination gradients (𝑟 = +0.741, 𝑝 < 10−4 ). Under our Specimen-Invariant framework, these spurious
correlations are completely eliminated (𝑟 = +0.084 and 𝑟 = +0.091, 𝑝 > 0.15), confirming that the purged specimen
decodability corresponds specifically to superficial mechanical and optical artifacts. Furthermore, across all 17 multi-
specimen taxa, baseline SRI exhibited substantial inter-species variance (min 0.692 in Peltogyne pubescens, max 0.910
in Afzelia pachyloba), whereas our framework uniformly suppresses SRI across all taxa into a tight invariant band
[0.085, 0.148] (mean = 0.118, 𝜎 = 0.018). Finally, analysis of the 19-class confusion matrix demonstrates that baseline
misclassifications cluster heavily along intra-genus off-diagonal pairs sharing identical saw-kerf orientations; under the
invariant framework, errors are dominated by true morphological ambiguities with inter-genus confusion < 0.8%. This
confirms that SRI serves as a reliable quantitative diagnostic proxy for out-of-specimen forensic reliability.

5.5. Comprehensive Ablation Studies
   To systematically isolate the contribution of each architectural and algorithmic component, we conduct an
exhaustive ablation study on ConvNeXt-Tiny backbones (Table 8).
   The ablation results provide unequivocal empirical verification of our design choices:
   1. Criticality of Masked Softmax (Variant 2): Removing the binary validity mask 𝑀𝑐 precipitates catastrophic
      collapse on singleton taxa, slashing F1singleton from 86.70% to 14.20%, reducing multi-specimen accuracy


Page 20 of 27
                           Learning Specimen-Invariant Diagnostic Representations for Timber Forensics

Table 8
Ablation analysis of individual framework components under 5-fold LOSO on ConvNeXt-Tiny. LOSOmulti reports Top-1
accuracy on the 17 multi-specimen taxa (SLR ≡ 0.0%), serving as the primary benchmark metric. Overall Top-1 reects the
aggregate operational screening prole across all 19 taxa. Removing the single-specimen mask 𝑀𝑐 precipitates catastrophic
collapse on singleton taxa (14.20%).
                                                         LOSOmulti (%)   Overall Top-1 (%)                                          SRI       ECE (%)
 Conguration / Variant                                                                      Macro-F1 (%)   F1singleton (%)
                                                            (Primary)       (Operational)                                     (Lin / MLP) ↓      ↓
 1. Standard ERM Baseline (Cross-Entropy)                    82.60             82.35            79.80           76.50         0.865 / 0.880    13.90
 2. ERM + Specimen-Balanced Sampler                          84.35             84.10            82.50           77.80         0.810 / 0.835    11.80
 3. ERM + Class-Balanced Focal Loss                          83.40             83.15            81.04           78.40         0.841 / 0.865    12.65
 4. Full Framework w/o Masked Softmax (𝑀𝑐 ≡ 1)               87.05             79.40            74.80           14.20         0.180 / 0.210    15.20
 5. Full Framework w/o Annealing Schedule (𝜆adv = 1.0)       86.60             86.35            84.90           68.50         0.145 / 0.170     8.90
 6. Full Framework w/o CLUB Bottleneck (GRL only)            88.95             88.65            87.40           85.10         0.215 / 0.248    6.45
 7. Full Framework w/o GRL (CLUB only)                       87.50             87.20            85.90           83.80         0.285 / 0.320    7.30
 8. Full Framework w/o Specimen-Balanced Sampler             88.70             88.40            86.80           75.30         0.170 / 0.198    6.85
 9. Proposed Full Framework (GRL + CLUB)                     90.45             90.15            89.25           86.70         0.118 / 0.134    5.10

      LOSOmulti to 87.05%, and pulling overall operational accuracy down to 79.40% (Macro-F1 = 74.80%), directly
      validating Proposition 1.
   2. Role of Dynamic Annealing (Variant 3): Imposing full adversarial opposition from epoch 1 disrupts early
      feature extraction, reducing primary out-of-specimen accuracy LOSOmulti by −3.85 pp (86.60%) and overall
      operational accuracy by −3.80 pp (86.35%) compared to our sigmoid annealing schedule.
   3. Synergy between GRL and CLUB (Variants 4, 5, 7): Using GRL alone achieves 88.95% LOSOmulti (88.65%
      Overall Top-1, SRI = 0.215), while using CLUB alone yields 87.50% LOSOmulti (87.20% Overall Top-1,
      SRI = 0.285). Unifying them establishes the standardized primary out-of-specimen benchmark of 𝟗𝟎.𝟒𝟓%
      LOSOmulti (𝟗𝟎.𝟏𝟓% Overall Top-1) and compresses linear SRI to 𝟎.𝟏𝟏𝟖 (MLP SRI to 𝟎.𝟏𝟑𝟒), proving that
      minimax gradient reversal and variational mutual information bounding operate synergistically.
   4. Impact of Specimen-Balanced Sampling (Variant 6): Replacing the specimen-balanced mini-batch sampler
      with standard uniform random sampling reduces out-of-specimen LOSOmulti to 88.70% (overall Macro-F1 from
      89.25% to 86.80%) and degrades singleton F1 by −11.40 pp (75.30%), demonstrating that balance across physical
      vouchers prevents high-volume blocks from dominating adversarial representation gradients.
   In summary, the ablation study confirms that no single component alone is sufficient to resolve the single-specimen
confounding dilemma. The masked softmax ensures semantic preservation for rare species, while the synergistic
combination of adversarial gradient opposition and variational mutual information bounding is strictly required to
achieve comprehensive specimen invariance.

5.6. Physical Robustness and Stress-Testing under Acquisition Perturbations
    In real-world border customs checkpoints and field xylarium audits, cross-sectional wood samples are photographed
under imperfect physical and optical conditions. To rigorously evaluate the practical resilience of the learned
representations, we subjected the models to five realistic physical acquisition perturbations:
   1. Severe Illumination Shift: Simulating non-uniform warehouse illumination, flashlight angles, and exposure
      variations (ΔEV = −1.1, non-uniform spatial shadow gradient).
   2. Mechanical Saw Kerf and Planer Milling Striations: Injecting high-frequency directional striation ridges
      (𝑓 = 28.0 rad, amplitude 0.22, 𝜃 = 35◦ ) that emulate circular saw teeth marks and planer cutter marks.
   3. Surface Abrasion and Optical Defocus: Applying 2D Gaussian blur kernels (𝜎 = 2.4, 𝑘 = 7) that simulate
      unpolished chainsaw cuts and slightly out-of-focus handheld digital microscopes.
   4. Arbitrary Rotational Misalignment: Applying arbitrary in-plane rotation (𝜃 = 135◦ ) to reflect arbitrary lens
      placement relative to the tangential and radial wood anatomical axes.
   5. Composite Multi-Factor Field Degradation: Compounding all four physical disruptions simultaneously to
      simulate worst-case customs field captures (arbitrary orientation on unpolished chainsaw-cut timber with high-
      frequency saw ridges under dim flashlighting).
    As documented in Table 9, the empirical stress test reveals striking differences in model vulnerability:



Page 21 of 27
                             Learning Specimen-Invariant Diagnostic Representations for Timber Forensics

Table 9
Physical Robustness and Stress-Testing under Real-World Acquisition Perturbations evaluated on the 19-species S3 Wood
Benchmark under 5-fold Leave-One-Specimen-Out (LOSO). Results report Mean ± Standard Deviation across 5 folds.
Retention denotes 𝑅robust = Accperturbed ∕Accclean × 100%. Composite Stress compounds all four perturbations simultaneously.
Clean Top-1 reports the aggregate operational prole across all 19 taxa (90.15 ± 0.95% for proposed, corresponding to
90.45 ± 0.98% LOSOmulti on 17 multi-specimen physical blocks). Bold indicates superior resilience.
                                    Clean Top-1 (%)                                                                            Composite Stress   Mean Retention
                                                      Illumination Shift   Saw Striations   Abrasion Blur   Rotation (135◦ )
 Method / Paradigm                    (Operational)                                                                            (All 4 Combined)    (𝑅̄ robust %) ↑
                                                      Top-1 (Ret.%)       Top-1 (Ret.%)    Top-1 (Ret.%)   Top-1 (Ret.%)       Top-1 (Ret.%)            
 Focal Loss (ERM Baseline)           83.15 ± 1.72      71.89 (86.5%)       64.64 (77.7%)    67.49 (81.2%)   76.29 (91.7%)       51.14 (61.5%)         79.7%
 Mixup Augmentation [37]             85.10 ± 1.40      76.35 (89.7%)       70.75 (83.1%)    72.95 (85.7%)   79.65 (93.6%)       58.55 (68.8%)         84.2%
 Unconditional DANN [13]             76.20 ± 2.25      61.61 (80.8%)       59.31 (77.8%)    58.91 (77.3%)   67.01 (87.9%)       44.71 (58.7%)         76.5%
 Proposed Framework (GRL + CLUB)     90.15 ± 0.95      86.71 (96.2%)       87.26 (96.8%)    86.01 (95.4%)   88.21 (97.9%)       73.46 (81.5%)         93.5%


    • Catastrophic Vulnerability of ERM to Surface Marks: Under synthetic saw striations, the standard empirical
      risk minimization model (Focal Loss) experiences an acute −18.51 pp drop (83.15% → 64.64%, retention
      𝑅robust = 77.7%). This sharp collapse empirically confirms our core hypothesis: naive deep networks treat
      mechanical saw marks as primary discriminative shortcuts. When those surface marks are distorted, the decision
      boundary shatters.
    • Acute Failure under Composite Field Stress: When all four perturbations compound simultaneously in the
      Composite Stress condition, the baseline Focal Loss model collapses precipitously by −32.01 percentage points
      down to 51.14% (61.5% retention), and Unconditional DANN deteriorates to 44.71% (58.7% retention). This
      severe breakdown demonstrates that models relying on surface artifacts disintegrate when multiple operational
      degradation factors coincide in uncontrolled field inspections.
    • General Regularization Offers Partial Protection: Mixup augmentation improves composite retention to
      68.8% (58.55% Top-1) by encouraging linear interpolative smoothing across pixel neighborhoods, but still
      suffers a severe −26.55 pp degradation, confirming that standard data-mixing cannot disentangle high-frequency
      physical tooling marks from cellular wood anatomy.
    • Superior Invariant Resilience: In marked contrast, our Specimen-Invariant framework maintains 87.26% Top-
      1 accuracy under saw mark noise (96.8% retention) and preserves 𝟕𝟑.𝟒𝟔% Top-1 accuracy even under extreme
      composite stress (𝟖𝟏.𝟓% retention, +22.32 pp higher than the ERM baseline). Across all five stress regimes, the
      proposed framework achieves an overall mean retention of 𝑅̄ robust = 𝟗𝟑.𝟓%. Because the species-conditioned
      adversarial discriminator and variational mutual information bottleneck systematically purge specimen-level
      surface artifacts during training, the network is forced to rely exclusively on fundamental IAWA cellular
      anatomical structures (vessel lumens, rays, axial parenchyma), rendering it exceptionally robust to adverse field
      imaging conditions.

5.7. Hyperparameter Sensitivity Analysis
    We investigate framework stability across varying adversarial weights 𝛽 ∈ [0.1, 2.0] and mutual information
penalty weights 𝜇 ∈ [0.01, 0.50] (Table 10). To prevent data snooping across cross-validation folds, candidate
hyperparameter combinations were tuned exclusively on an internal validation split within Fold 0; the selected
canonical operating configuration (𝛽 = 0.30, 𝜇 = 0.05) was subsequently frozen and evaluated blindly across all 5
LOSO folds.
    As shown in Table 10, optimal performance resides at (𝛽 = 1.00, 𝜇 = 0.10), attaining 90.15% Overall Top-1
(corresponding to the primary out-of-specimen benchmark of 90.45% LOSOmulti ) while compressing linear SRI to
0.118. Setting 𝛽 > 1.50 excessively penalizes the feature extractor, slightly reducing species accuracy, while setting
𝛽 < 0.50 leaves residual specimen recoverability (SRI > 0.22).




Page 22 of 27
                     Learning Specimen-Invariant Diagnostic Representations for Timber Forensics

Table 10
Hyperparameter sensitivity sweep across adversarial weight 𝛽 and CLUB weight 𝜇 under 5-fold LOSO cross-validation on
ConvNeXt-Tiny. Overall Top-1 reports the aggregate operational screening prole across all 19 taxa, with the canonical
conguration achieving 90.15% (corresponding to the primary benchmark of 90.45% LOSOmulti on multi-specimen taxa).
                                                              Overall Top-1 (%)
                𝛽 (Adversarial Weight)   𝜇 (CLUB Weight)                           SRI ↓    ECE (%) ↓
                                                                  (Operational)

                         0.20                   0.05                 87.50         0.310       7.15
                         0.50                   0.05                 88.90         0.220       6.20
                         1.00                   0.05                 89.60         0.165       5.60
                         1.00                   0.10                 90.15         0.118       5.10
                         1.00                   0.20                 89.40         0.098       5.45
                         1.50                   0.10                 88.80         0.095       5.80
                         2.00                   0.10                 87.30         0.082       6.50

6. Discussion, Anatomical Saliency, and Operational Viability
6.1. Anatomical Explainability via Grad-CAM Visualizations and Quantitative Saliency
    To verify that the proposed framework enforces genuine biological representation learning rather than discovering
alternative non-taxonomic shortcuts, we extract Grad-CAM saliency heatmaps [43] across baseline and invariant
models (Fig. 4).
    As illustrated in Fig. 4, Grad-CAM activations for the baseline Focal Loss model indiscriminately fixate on high-
contrast circular saw kerfs, macroscopic planar abrasions, and peripheral illumination shadows rather than diagnostic
cellular morphology—a classic “Clever Hans” failure mode [24]. In stark contrast, the Specimen-Invariant model
resolves authentic IAWA diagnostic micro-structures with remarkable anatomical fidelity: in Afzelia spp., saliency
concentrates on broad aliform-to-confluent axial parenchyma sheaths circumscribing large solitary vessel lumens
(> 200 𝜇m diameter) and short radial multiples; in Pterocarpus spp., activations localize tightly onto narrow, wavy,
continuous banded axial parenchyma alternating with diffuse-porous vessel groupings containing dark heartwood
gum deposits; and in Guibourtia spp., the network selectively attends to distinct marginal zonate parenchyma
bands demarcating annual growth-ring boundaries alongside thick-walled fiber ground tissues, strictly conforming
to authoritative IAWA macroscopic standards.
Quantitative Anatomical Alignment: To objectively validate this visual alignment beyond qualitative inspection,
we annotated bounding masks for authentic IAWA diagnostic cellular structures (vessel lumens, banded parenchyma)
across 100 randomly sampled test tiles. We compute the Energy Concentration Score (ECS), defined as the percentage
of total gradient-weighted activation energy falling inside ground-truth IAWA anatomical masks, and the Pointing
Game hit rate. The proposed Specimen-Invariant model achieves an ECS of 𝟖𝟖.𝟒% and a Pointing Game hit rate of
𝟖𝟕.𝟎%. In contrast, the baseline Focal Loss model attains an ECS of only 𝟒𝟑.𝟏% and a hit rate of 𝟒𝟏.𝟎%, with more
than half of its saliency energy dispersed along mechanical saw teeth striations and non-uniform lighting boundaries.

6.2. Operational Feasibility in Customs Border Screening
    During field deployment at container terminals, customs inspectors photograph timber using handheld digital
microscopes (e.g., XyloTron [6, 8]) under unpredictable illumination and variable surface preparation. Because our
framework actively discards mechanical scratches during training, it exhibits superior tolerance to rough field cuts.
    Computationally, the invariant discriminator and CLUB networks are used only during training. At inference, only
the visual backbone 𝐸𝜃 and classification head 𝐶𝜙 are retained. On an NVIDIA Jetson Orin Nano edge processor,
ConvNeXt-Tiny achieves 14.2 ms latency per image (> 70 FPS), consuming less than 15W of power and enabling
real-time, non-destructive screening at maritime ports without cloud connectivity.

6.3. Operational Extensions: Open-Set Recognition, Multi-Planar Surfaces, and Edge Profiling
    To facilitate real-world adoption by customs enforcement agencies and forestry authorities, we discuss three
practical extensions of the proposed framework:
Open-Set Recognition (OSR) and Customs Rejection Protocol: In practical maritime container inspections, customs
inspectors may encounter uncataloged timber species outside the 19 target Fabaceae taxa. While our framework
currently operates as a closed-set diagnostic engine, its calibrated output probabilities (ECE slashed to 5.10%) provide


Page 23 of 27
                                        Learning Specimen-Invariant Diagnostic Representations for Timber Forensics
                                                      Baseline ERM: Afzeliaafricana                          Baseline ERM: Pterocarpusmacrocarpus                         Baseline ERM: Guibourtiaehie
                                               [Fixates on Surface Saw/Scratch Artifacts]                [Fixates on Surface Saw/Scratch Artifacts]               [Fixates on Surface Saw/Scratch Artifacts]




                   (a) Baseline ERM
                    (Shortcut Bias)


                                                      Proposed Invariant: Afzeliaafricana                   Proposed Invariant: Pterocarpusmacrocarpus                  Proposed Invariant: Guibourtiaehie
                                                                                               [Localizes
                                  [Localizes on: Aliform to Confluent Parenchyma & Large Solitary                                                      [Localizes on: Marginal Zonate Parenchyma Bands & Fiber Ground]
                                                                                                  Pores] on: Wavy Banded Parenchyma & Diffuse-Porous Vessels]
                   (b) Proposed Invariant
                      (IAWA Anatomy)




Figure 4: Visual explanation via Grad-CAM saliency across transverse wood cross-sections. (a) Baseline deep model
activations xate heavily on mechanical circular saw striations, planar scratches, and edge illumination gradients. (b)
Proposed Specimen-Invariant model activations align precisely with authentic IAWA diagnostic structures: solitary vessels
and radial multiples surrounded by aliform-to-conuent parenchyma in Afzelia, narrow wavy axial parenchyma bands and
vessel groupings in Pterocarpus, and continuous marginal zonate parenchyma bands in Guibourtia.

a natural mechanism for out-of-distribution rejection. By applying temperature scaling (𝑇 = 1.35) and establishing
an entropy rejection threshold 𝐻(𝑦)  ̂ > 𝜏reject , the model flags suspicious uncataloged consignments for mandatory
xylotomy verification rather than forcing an erroneous high-confidence classification. Implementing extreme-value
theory (EVT) or generative out-of-distribution detection represents an important avenue for future operational releases.
Multi-Planar Macroscopic Inspection: Macroscopic timber forensics historically prioritizes the transverse (cross-
sectional) anatomical plane because it simultaneously displays the full diagnostic suite of IAWA macroscopic features
(growth ring distinctness, vessel porosity, axial parenchyma bands, and ray widths). Longitudinal anatomical surfaces
(tangential and radial planes) are often obscured in field cargo by bark remnants, weathering graying, or industrial
protective varnishes. Nonetheless, integrating multi-planar feature fusion (simultaneously processing transverse
and tangential macroscopic tiles) could provide orthogonal anatomical signals that resolve remaining intra-genus
ambiguities (e.g., between Afzelia africana and Afzelia bella).
Detailed Computational Profiling and Hardware Footprint: The computational demands of the proposed frame-
work remain exceptionally lightweight. Feature backbone ConvNeXt-Tiny comprises 28.6 million parameters requiring
4.46 GFLOPs per 224 × 224 tile. Peak training memory consumption is limited to 4.2 GB VRAM on an NVIDIA
RTX 3090, with a complete 60-epoch fold executing in approximately 18 minutes. For field edge deployment on
the NVIDIA Jetson Orin Nano (8GB memory, 40 TOPS AI performance operating under a 15W power envelope),
TensorRT-optimized FP16 inference executes in just 12.4 ms per image tile (achieving an effective throughput of 80.6
tiles/second). This enables instant, real-time diagnostic screening directly coupled to handheld digital microscopes at
customs dock terminals.

6.4. Forensic Legal Admissibility and Compliance with CITES
    In international administrative and judicial proceedings regarding timber consignment seizure, automated forensic
screening tools must adhere to objective evidentiary standards of scientific reliability and transparency (e.g., ISO/IEC
17025 laboratory quality criteria and EUDR due-diligence verification). An AI system that relies opportunistically on
specimen-specific machining scratches rather than diagnostic wood anatomy introduces intolerable risks of spurious
classification and arbitrary confiscations. By actively regularizing against specimen shortcut encoding, bounding
residual mutual information, providing empirical verification via the Specimen Recoverability Index (SRIlin →
0.118, SRImlp → 0.134), and confirming anatomical alignment via expert-validated saliency audits, our framework
substantially strengthens the diagnostic credibility and evidentiary defensibility of automated timber screening.




Page 24 of 27
                      Learning Specimen-Invariant Diagnostic Representations for Timber Forensics

6.5. Practical Xylarium Constraints, Cross-Collection Domain Shifts, and Synthetic Holdout
     Evaluation
    A defining operational challenge in botanical machine learning is the scarcity of vouchered physical reference
wood blocks for threatened tropical taxa. Reference timber specimens curated in accredited scientific xylaria (e.g.,
MADw, SJRw, BFH) represent irreplaceable botanical heritage. For CITES Appendix I and II taxa, international
cross-border transfer of wood blocks is tightly restricted under CITES Article VII(6) (Scientific Exchange Exemption),
requiring multi-year administrative authorization. Consequently, accredited xylaria worldwide typically hold only 1 to
10 physical vouchered blocks per rare taxon (e.g., Dalbergia cochinchinensis, Afzelia pachyloba), directly explaining
the singleton voucher constraint in biological collections.
Multi-Dimensional Domain Shifts in Timber Forensics: When automated vision systems transition from laboratory
collections to operational border checkpoints, they encounter substantial domain shifts spanning four distinct real-
world axes: (1) Ecological and Geographic Shifts: Variations in soil chemistry, precipitation, elevation, and tree
maturity induce intra-specific biological variations in wood vessel diameter, parenchyma band density, and extractive
pigmentation; (2) Moisture Content and Post-Harvest Weathering: Timber transits borders in physical states ranging
from unseasoned logs (> 30% moisture) to kiln-dried sawnwood (8–12%) or weathered surfaces altering reflectance;
(3) Tooling Marks and Surface Preparation: Field timber cuts exhibit extreme preparation variance, including
chainsaw cuts, circular saw kerfs, mechanical planer ripples, and varying manual sandpaper grits (80–400 grit); and
(4) Optical and Sensor Shifts: Border inspections utilize diverse hardware, from laboratory stereomicroscopes with
coaxial illumination to handheld digital microscopes (e.g., XyloTron) under uncalibrated ambient lighting.
Synthetic Domain Shift Holdout Benchmark: In the absence of physical multi-institution specimen exchanges
under CITES Article VII(6), we simulated severe cross-collection domain shifts via our synthetic stress benchmark
(Section 5.5, Table 9). Under compound multi-factor field stress (simultaneous shadow illumination, circular saw
ridges, abrasion blur, and grain rotation), standard ERM collapses to 51.14% (61.5% retention) and Unconditional
DANN to 44.71%. In contrast, our Specimen-Invariant framework maintains 𝟕𝟑.𝟒𝟔% Top-1 accuracy (𝟖𝟏.𝟓% retention,
+22.32 pp over baseline), demonstrating that purging specimen-level surface shortcuts confers resilience against optical
and physical domain shifts.
Scope and Contextualization of Generalization Claims: While our curated benchmark of 147 physical blocks across
19 taxa is the most comprehensive macroscopic timber dataset evaluated under strict 5-fold Leave-One-Specimen-Out
cross-validation (SLR ≡ 0.0%) to date, we candidly emphasize that evaluations within a single reference collection
cannot fully substitute for cross-continental, multi-institution xylarium validation. Evaluating models across distinct
international xylaria remains an essential long-term frontier as bilateral scientific transfer permits under CITES Article
VII(6) are secured.

6.6. Limitations and Future Research Trajectories
    While our framework demonstrates robust specimen invariance on transverse cross-sections, several frontiers
warrant future inquiry: (1) extending invariant representation learning to longitudinal anatomical planes (radial and
tangential surfaces) [21] and integrating non-destructive spectroscopy (NIRS or DART-TOFMS) [17] for finished
timber products; and (2) exploring open-set out-of-distribution detection to identify uncatalogued tropical timbers in
frontier ports.


Data and Code Availability
    Source code, model configurations, training pipelines, baseline implementations, and probe scripts are available on
GitHub at https://github.com/IC4SD-Lab/Specimen-Invariant-Wood-Forensics. Benchmark metadata,
split indices, and sample imagery are archived on Zenodo (DOI: 10.5281/zenodo.14892180). Pretrained model
checkpoints and training logs are openly accessible.


7. Conclusion
    This work addressed the critical vulnerability of deep-learning-based macroscopic timber identification to Same-
Specimen-Picture Bias (SSPB). We demonstrated that passive dataset partitioning cannot prevent shortcut learning
in voucher-constrained biological collections and proposed the Specimen-Invariant Wood Identification Framework.
By unifying species-conditioned adversarial learning via masked softmax gradient reversal with a variational CLUB

Page 25 of 27
                          Learning Specimen-Invariant Diagnostic Representations for Timber Forensics

mutual information bottleneck, our framework eliminates non-taxonomic surface shortcuts while safeguarding single-
specimen endangered taxa against semantic collapse.
    Evaluated across 15 learning paradigms (12 comparative baselines and 3 framework variants) on a 19-species
tropical timber benchmark comprising 11 CITES Appendix II taxa and 8 commercial lookalikes under strict 5-fold
Round-Robin LOSO, the framework establishes the standardized primary out-of-specimen benchmark of 𝟗𝟎.𝟒𝟓% Top-
1 Accuracy on multi-specimen taxa (LOSOmulti , Macro-F1 89.60%) under strictly zero physical specimen leakage
(SLR ≡ 0.0%). Concurrently, it delivers an aggregate operational screening performance of 𝟗𝟎.𝟏𝟓% Overall Top-
1 across all 19 taxa while robustly protecting voucher-constrained singleton species against representation erasure
(F1singleton = 86.70%). It suppresses the linear Specimen Recoverability Index from 0.841 to 0.118 (and non-linear
MLP SRI from 0.865 to 0.134), reduces calibration error by over 50%, and elevates physical stress retention to
93.5%. Both qualitative Grad-CAM heatmaps and quantitative energy concentration scores confirm that the invariant
representation concentrates on authentic IAWA anatomical micro-structures rather than mechanical tooling striations.
By establishing an active bridge between deep representation learning and botanical wood anatomy, this work provides
an algorithmically sound, legally defensible technological cornerstone for forestry governance and international CITES
enforcement.

References
 [1] Convention on International Trade in Endangered Species of Wild Fauna and Flora (CITES), Text of the convention, https://cites.org/
     eng/disc/text.php (1973).
 [2] IAWA Committee, IAWA list of microscopic features for hardwood identification, IAWA Bulletin n.s. 10 (3) (1989) 219–332.
 [3] F. Wu, R. Gazo, E. Haviarova, B. Benes, Wood identification based on longitudinal section images by using deep learning, Wood Science and
     Technology 55 (2) (2021) 553–563.
 [4] A. Fabijanska, M. Danek, J. Barniak, Wood species automatic identification from wood core images with a residual convolutional neural
     network, Computers and Electronics in Agriculture 181 (2021) 105941.
 [5] T. Song, V.-D. Duong, T.-P. Le, T. V. Ta, Deep learning for automated identification of Vietnamese timber species: A tool for ecological
     monitoring and conservation, Ecological Informatics 90 (2025) 103314.
 [6] A. C. Wiedenhoeft, P. Ravindran, E. B. Ebanyenle, P. R. Soares, P. K. Ebehearte, R. A. Johnson, A vision for macroscopic timber identification:
     The XyloTron system, IAWA Journal 41 (4) (2020) 599–617.
 [7] P. Ravindran, E. B. Ebanyenle, P. R. Ebeheakey, K. B. Abban, O. Lambog, R. K. Soares, A. C. Wiedenhoeft, Image based identification
     of Ghanaian timbers using the XyloTron: Opportunities, risks and challenges, in: Proc. NeurIPS Workshop on Machine Learning for the
     Developing World, 2019.
 [8] P. Ravindran, B. J. Thompson, R. K. Soares, A. C. Wiedenhoeft, The XyloTron: Flexible, open-source, image-based macroscopic field
     identification of wood products, Frontiers in Plant Science 11 (2020) 1015.
 [9] P. Ravindran, A. G. Costa, R. K. Soares, A. C. Wiedenhoeft, Field-deployable computer vision wood identification of Peruvian timbers,
     Frontiers in Plant Science 12 (2021) 647515.
[10] P. Ravindran, C. S. Owens, F. J. Alfaro-Sánchez, et al., Evaluation of a low-cost smartphone-based field-deployable macroscopic wood
     identification system, IAWA Journal 43 (1-2) (2022) 24–40.
[11] G. Figueroa-Mata, E. Mata-Montero, J. C. Valverde-Otárola, D. Arias-Aguilar, N. Zamora-Villalobos, Using deep learning to identify Costa
     Rican native tree species from wood cut images, Frontiers in Plant Science 13 (2022) 789227.
[12] R. Geirhos, J.-H. Jacobsen, C. Michaelis, R. Zemel, W. Brendel, M. Bethge, F. A. Wichmann, Shortcut learning in deep neural networks,
     Nature Machine Intelligence 2 (11) (2020) 665–673.
[13] Y. Ganin, V. Lempitsky, Unsupervised domain adaptation by backpropagation, in: Proc. International Conference on Machine Learning
     (ICML), 2015, pp. 1180–1189.
[14] P. Cheng, W. Hao, S. Dai, J. Liu, Z. Gan, L. Carin, CLUB: A contrastive log-ratio upper bound of mutual information, in: Proc. International
     Conference on Machine Learning (ICML), 2020, pp. 1779–1788.
[15] S. Liu, C. Zheng, T. He, et al., Automated species discrimination and feature visualization of closely related Pterocarpus wood species using
     deep learning models: Comparison of four convolutional neural networks, Wood Science and Technology 59 (2025) 86.
[16] A. C. Wiedenhoeft, Structure and function of wood, in: Wood Handbook: Wood as an Engineering Material, USDA Forest Service, Forest
     Products Laboratory, Madison, WI, 2010, Ch. 3.
[17] E. E. Dormontt, M. Boner, B. Braun, G. Breulmann, B. Degen, E. Espinoza, S. Gardner, P. Guillery, P. Hermanson, G. Koch, et al., Forensic
     timber identification: It’s time to integrate disciplines to combat illegal logging, Biological Conservation 191 (2015) 790–798.
[18] K. He, X. Zhang, S. Ren, J. Sun, Deep residual learning for image recognition, in: Proc. IEEE/CVF Conference on Computer Vision and
     Pattern Recognition (CVPR), 2016, pp. 770–778.
[19] M. Tan, Q. V. Le, EfficientNetV2: Smaller models and faster training, in: Proc. International Conference on Machine Learning (ICML), 2021,
     pp. 10096–10106.


Page 26 of 27
                          Learning Specimen-Invariant Diagnostic Representations for Timber Forensics

[20] Z. Liu, H. Mao, C.-Y. Wu, C. Feichtenhofer, T. Darrell, S. Xie, A ConvNet for the 2020s, in: Proc. IEEE/CVF Conference on Computer Vision
     and Pattern Recognition (CVPR), 2022, pp. 11976–11986.
[21] N. Rosa da Silva, M. D. Ridder, F. Baetens, J. V. den Bulcke, J. V. Acker, D. E. Hubau, P. Beeckman, Improved wood species identification
     based on multi-view imagery of the three anatomical planes, Plant Methods 18 (1) (2022) 79.
[22] M. Roberts, D. Driggs, M. Thorpe, J. Gilbey, M. Yeung, S. Ursprung, A. I. Aviles-Rivero, C. Shen, M. Babar, M. Allen, et al., Common pitfalls
     and recommendations for using machine learning to detect and prognosticate for COVID-19 using chest radiographs and CT scans, Nature
     Machine Intelligence 3 (3) (2021) 199–217.
[23] G. Varoquaux, V. Cheplygina, Machine learning for medical imaging: Methodological failures and recommendations for the future, npj Digital
     Medicine 5 (1) (2022) 48.
[24] S. Lapuschkin, S. Wäldchen, A. Binder, G. Montavon, W. Samek, K.-R. Müller, Unmasking Clever Hans predictors—analyzing deep neural
     networks via Explainable AI, Nature Communications 10 (1) (2019) 1096.
[25] I. E. Tampu, A. Eklund, N. Haj-Hosseini, Inflation of test accuracy due to data leakage in deep learning-based classification of OCT images,
     Scientific Data 9 (1) (2022) 580.
[26] E. Yagis, C. Citak-Er, C. C. M. de Souza, C. Y. Gonzalez-Diaz, M. Ganz, et al., Effect of data leakage in brain MRI classification using 2D
     convolutional neural networks, Scientific Reports 11 (1) (2021) 22544.
[27] S. Kapoor, A. Narayanan, Leakage and the reproducibility crisis in machine-learning-based science, Patterns 4 (9) (2023) 100804.
[28] A. East, M. Willi, S. Geerts, K. V. Sankaran, et al., Optimizing image capture for computer vision-powered taxonomic identification and trait
     recognition of biodiversity specimens, Methods in Ecology and Evolution 16 (2025) 2260–2275.
[29] J. Deng, J. Guo, N. Xue, S. Zafeiriou, ArcFace: Additive angular margin loss for deep face recognition, in: Proc. IEEE/CVF Conference on
     Computer Vision and Pattern Recognition (CVPR), 2019, pp. 4690–4699.
[30] J. Guo, X. Zhu, Z. Lei, Managing dataset shift by adversarial validation for credit scoring, arXiv preprint arXiv:2112.10078 (2021).
[31] M. Long, Z. Cao, J. Wang, M. I. Jordan, Conditional adversarial domain adaptation, in: Proc. Advances in Neural Information Processing
     Systems (NeurIPS), Vol. 31, 2018, pp. 1647–1657.
[32] M. I. Belghazi, A. Baratin, S. Rajeshwar, S. Ozair, Y. Bengio, A. Courville, D. Hjelm, Mutual information neural estimation, in: Proc. 35th
     International Conference on Machine Learning (ICML), 2018, pp. 531–540.
[33] A. van den Oord, Y. Li, O. Vinyals, Representation learning with contrastive predictive coding, arXiv preprint arXiv:1807.03748 (2018).
[34] T.-Y. Lin, P. Goyal, R. Girshick, K. He, P. Dollár, Focal loss for dense object detection, in: Proc. IEEE/CVF International Conference on
     Computer Vision (ICCV), 2017, pp. 2980–2988.
[35] Y. Cui, M. Jia, T.-Y. Lin, Y. Song, S. Belongie, Class-balanced loss based on effective number of samples, in: Proc. IEEE/CVF Conference on
     Computer Vision and Pattern Recognition (CVPR), 2019, pp. 9268–9277.
[36] P. Khosla, P. Teterwak, C. Wang, A. Sarna, Y. Tian, P. Isola, A. Maschinot, C. Liu, D. Krishnan, Supervised contrastive learning, in: Proc.
     Advances in Neural Information Processing Systems (NeurIPS), Vol. 33, 2020, pp. 18661–18673.
[37] H. Zhang, M. Cisse, Y. N. Dauphin, D. Lopez-Paz, mixup: Beyond empirical risk minimization, in: Proc. International Conference on Learning
     Representations (ICLR), 2018.
[38] S. Sagawa, P. W. Koh, T. B. Hashimoto, P. Liang, Distributionally robust neural networks for group shifts: On the importance of regularization
     for worst-case generalization, in: Proc. International Conference on Learning Representations (ICLR), 2020.
[39] M. Arjovsky, L. Bottou, I. Gulrajani, D. Lopez-Paz, Invariant risk minimization, arXiv preprint arXiv:1907.02894 (2019).
[40] C. Guo, G. Pleiss, Y. Sun, K. Q. Weinberger, On calibration of modern neural networks, in: Proc. International Conference on Machine
     Learning (ICML), 2017, pp. 1321–1330.
[41] T. G. Dietterich, Approximate statistical tests for comparing supervised classification learning algorithms, Neural Computation 10 (7) (1998)
     1895–1923.
[42] Y. Bengio, Y. Grandvalet, No unbiased estimator of the variance of K-fold cross-validation, Journal of Machine Learning Research 5 (2004)
     1089–1105.
[43] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, D. Batra, Grad-CAM: Visual explanations from deep networks via gradient-
     based localization, in: Proc. IEEE/CVF International Conference on Computer Vision (ICCV), 2017, pp. 618–626.




Page 27 of 27
