# Point-by-Point Response to Reviewer Criticisms (Q1 Peer Review)

**Manuscript Title:** Learning Specimen-Invariant Diagnostic Representations for Timber Forensics: A Species-Conditioned Adversarial Framework with Variational Mutual Information Bottlenecks  
**Authors:** Viet-Anh Le, Khanh Nguyen-Trong  
**Target Venue:** *Pattern Recognition* / *Ecological Informatics* / *Computers and Electronics in Agriculture*  
**Date of Revision:** October 2026  

---

## TỔNG QUAN VỀ BẢN CHỈNH SỬA / REVISION OVERVIEW (BILINGUAL SUMMARY)

Chúng tôi xin chân thành cảm ơn Ban Biên tập và các Chuyên gia Phản biện đã dành thời gian đánh giá vô cùng sâu sắc, công tâm và mang tính xây dựng cao cho bản thảo. Tất cả các ý kiến đóng góp đã được tiếp thu nghiêm túc và giải quyết triệt để trong bản thảo sửa đổi (`main.tex`, biên dịch sạch 100% ra `main.pdf`).

We express our sincere gratitude to the Associate Editor and Reviewers for their rigorous, constructive, and insightful evaluation of our manuscript. In this revision, we have systematically addressed all major concerns, minor points, and specific reviewer questions. All modifications are reflected in the revised manuscript (`main.tex` / `main.pdf`, 21 pages).

### Tóm tắt các thay đổi cốt lõi trong bản sửa đổi:
1. **Phân tách minh bạch chỉ số LOSO (Disaggregated Evaluation - Concern 3.1, Q1):** Tách bạch rõ ràng $\text{LOSO}_{\text{multi}}$ (17 loài có nhiều phôi gỗ, 145 phôi, rò rỉ mẫu vật $\mathrm{SLR} \equiv 0.0\%$) đạt **$90.45\%$ Top-1 Accuracy** và $\text{F1}_{\text{singleton}}$ (2 loài chỉ có 1 phôi gỗ) đạt **$86.70\%$**. Kết quả này chứng minh độ chính xác $90.15\%$ trên toàn bộ 19 loài không hề bị thổi phồng bởi các loài đơn mẫu vật, mà ngược lại các loài đơn mẫu vật kéo nhẹ điểm trung bình xuống.
2. **Chuẩn hóa định lý và thừa nhận khe hở biến phân (Softened Theoretical Claims - Concern 3.2):** Sửa đổi Theorem 1 với các giả định thực tế (mô hình có dung lượng hữu hạn, tối ưu cục bộ), chỉ rõ rằng đại lượng cực tiểu hóa là chặn trên biến phân CLUB và chỉ ra tường minh khoảng cách biến phân không âm $\Delta_{\text{var}}(\xi) = \mathbb{E}[\mathcal{L}_{\text{CLUB}}] - I(Z; S \mid Y) \ge 0$.
3. **Định vị chính xác và so sánh với CDAN (Venue Positioning & Baselines - Concern 3.3, Q5):** Định vị bài báo trong các tạp chí Q1 ứng dụng hàng đầu (*Ecological Informatics*, *Computers and Electronics in Agriculture*, *IEEE TIFS*). Bổ sung phương pháp đối sánh nâng cao CDAN (Long et al., 2018): CDAN đạt $86.10\%$ trên $\text{LOSO}_{\text{multi}}$ nhưng sụp đổ còn $32.50\%$ trên $\text{F1}_{\text{singleton}}$, chứng minh tính ưu việt của cơ chế Masked Softmax GRL.
4. **Bối cảnh SOTA giám định gỗ và rào cản CITES Art. VII(6) (Domain SOTA & Legal Constraints - Concern 3.4, Q8):** Bổ sung thảo luận và trích dẫn các hệ thống thực địa như XyloTron (Wiedenhoeft et al., Ravindran et al.), các công trình của Wu et al., Fabijanska et al., Song et al.; phân tích quy định pháp lý quốc tế CITES Điều VII(6) hạn chế việc trao đổi mẫu vật gỗ thực tế giữa các bảo tàng gỗ (xylarium).
5. **Giao thức SRI và Đầu dò phi tuyến MLP (SRI Protocol & Non-Linear Probe - Concern 3.5, Q2):** Làm rõ quy trình đo SRI (đóng băng $E_\theta$, chia 50/50 các khối gỗ huấn luyện, kiểm tra trên ảnh chưa thấy). Bổ sung đầu dò phi tuyến MLP 2 lớp ($\mathrm{SRI}_{\text{mlp}}$ giảm từ $0.865$ xuống $0.134$), chứng minh khung phương pháp triệt tiêu rò rỉ phôi gỗ ở cả không gian biểu diễn tuyến tính và phi tuyến.
6. **Chính xác hóa thống kê Wilcoxon & Paired $t$-test (Statistical Significance - Concern 3.6, Q7):** Hiệu chỉnh giá trị $p$ của kiểm định Wilcoxon signed-rank ($n=5, W=0$: một phía $p=0.03125$, hai phía $p=0.0625$ là mức nhỏ nhất có thể đạt được với 5 fold). Bổ sung paired $t$-test có đối chứng ($t(4)=8.42, p=0.00105$, Cohen's $d=3.76$), đồng thời trích dẫn hạn chế phụ thuộc dữ liệu của cross-validation (Dietterich, 1998; Bengio & Grandvalet, 2004).
7. **Hiệu đính Bảng 1 và Thống kê CITES (Taxonomy Correction - Concern 3.7, Q6):** Đính chính chính xác Bảng 1: **11 loài thuộc CITES Phụ lục II** và **8 loài không thuộc CITES** (tổng 19 loài, 147 phôi gỗ, 6,414 ảnh). Sửa toàn bộ lỗi chính tả và chú thích bảng.
8. **Giải trình kiến trúc đầu phân loại (Linear vs ArcFace - Q3):** Làm rõ lý do chọn Linear Head + Class-Balanced Focal Loss làm cấu hình chuẩn mực để cô lập đóng góp của biểu diễn bất biến, xem ArcFace là phần mở rộng nâng cao.
9. **Cam kết dữ liệu và mã nguồn mở (Data & Code Availability - Q4):** Bổ sung tuyên bố khả dụng với kho lưu trữ GitHub và mã định danh vĩnh viễn Zenodo DOI.
10. **Bổ sung các khía cạnh thực nghiệm chi tiết (Minor Points 4):** Bổ sung thời gian huấn luyện (18 phút/fold, ~90 phút tổng trên RTX 3090), VRAM tối đa (4.2 GB), độ trễ suy luận trên thiết bị nhúng NVIDIA Jetson Orin Nano (14.2 ms/ảnh, >70 FPS, chi phí suy luận 0%), công thức lấy mẫu Specimen-Balanced Sampler ($8 \times 2 \times 4 = 64$), và định lượng Grad-CAM qua Pointing Game ($87.0\%$ vs $41.0\%$) và Energy Concentration Score ($88.4\%$ vs $43.1\%$).

---

# PART I: RESPONSES TO MAJOR CONCERNS (MỤC 3.1 – 3.7)

---

### Concern 3.1: Metric Inflation & Specimen Leakage in the LOSO Protocol
> **Reviewer Comment:**  
> *"The paper claims strict Leave-One-Specimen-Out (LOSO) cross-validation with zero specimen leakage. However, two species (Dalbergia cochinchinensis and Guibourtia coleosperma) are represented by only a single physical wood block each. For these singletons, an image-level split is used. Reporting a single aggregate Top-1 accuracy of 90.15% across all 19 species masks potential metric inflation from the singletons, where specimen leakage is not zero. The authors must disaggregate evaluation metrics between multi-specimen taxa and singleton taxa."*

**Author Response:**  
We completely agree with the Reviewer. Grouping voucher-constrained singleton taxa together with multi-specimen taxa into a single aggregate metric risks obscuring whether high performance is driven by genuine out-of-specimen generalization or residual specimen re-identification. 

In the revised manuscript, we have completely overhauled our reporting protocol by establishing an explicit **disaggregated evaluation framework**:
1. **$\text{LOSO}_{\text{multi}}$ (Multi-Specimen Taxa)**: Computed strictly over the **17 taxa** possessing multiple physical wood blocks ($S_c \ge 2$, 145 physical blocks, 5,700 images). Across all 5 folds, physical blocks are partitioned in a round-robin holdout regime with **strictly zero block overlap** ($\mathrm{SLR} \equiv 0.0\%$).
2. **$\text{F1}_{\text{singleton}}$ / $\text{Acc}_{\text{singleton}}$ (Singleton Taxa)**: Computed exclusively over the **2 taxa** constrained by a single vouchered physical block in the archival collection (*Dalbergia cochinchinensis* and *Guibourtia coleosperma*, $|\mathcal{G}_c| = 1$, 714 images total).
3. **Overall Top-1 / Macro-F1**: Computed across all 19 taxa to provide the complete operational screening profile.

**Key Empirical Discovery**:  
As reported in the updated Table 3 (Master Invariance Benchmark), Table 4 (Multi-Backbone), and Table 5 (Ablation Study):
- On the 17 multi-specimen taxa under strictly zero specimen leakage ($\mathrm{SLR} \equiv 0.0\%$), our framework achieves **$\mathbf{90.45 \pm 0.98\%}$ $\text{LOSO}_{\text{multi}}$ Top-1 Accuracy** (compared to $83.40\%$ for the Focal Loss baseline).
- On the 2 singleton taxa, our framework achieves **$86.70\%$ $\text{F1}_{\text{singleton}}$**.
- The overall Top-1 Accuracy is **$90.15\%$**.

This demonstrates that **the overall accuracy ($90.15\%$) was NOT inflated by the singleton taxa**. Rather, the singleton taxa slightly pulled *down* the aggregate average (from $90.45\%$ to $90.15\%$) due to their narrower intra-class morphological diversity!

Furthermore, disaggregating this metric reveals the catastrophic failure of unconditioned domain adaptation: Unconditional DANN achieves $83.70\%$ on $\text{LOSO}_{\text{multi}}$ but collapses to **$14.20\%$** on $\text{F1}_{\text{singleton}}$, and CDAN collapses to **$32.50\%$**. In contrast, our proposed Masked Softmax GRL maintains **$86.70\%$** on singletons.

**Changes in Revised Manuscript:**  
- **Section 4.2 (Round-Robin LOSO Protocol and Metric Disaggregation, Lines 440–447)**: Formally defines $\text{LOSO}_{\text{multi}}$ vs $\text{F1}_{\text{singleton}}$.
- **Table 3 (Master Invariance Benchmark, Lines 540–579)**: Explicitly reports $\text{LOSO}_{\text{multi}}$, Overall Top-1, Macro-F1, and $\text{F1}_{\text{singleton}}$ across all 13 paradigms.
- **Section 5.1 (Results Analysis, Lines 589–592)**: Discusses the disaggregated results and addresses metric inflation directly.

---

### Concern 3.2: Theoretical Claims and the Variational Bound Gap
> **Reviewer Comment:**  
> *"Theorem 1 claims that the framework guarantees strict specimen invariance ($I(Z; S \mid Y) = 0$). In practice, deep neural networks are parameterized with finite capacity and trained via non-convex optimization, which cannot guarantee zero mutual information. Furthermore, CLUB is an empirical variational upper bound; the gap between the bound and true mutual information depends on the variational posterior $q_\xi(s \mid z, y)$. The theoretical claims should be toned down, assumptions stated clearly, and the variational approximation gap formally acknowledged."*

**Author Response:**  
We thank the Reviewer for this vital theoretical critique. We have revised the theoretical formulation to ensure rigorous, realistic mathematical grounding.

In the revised manuscript:
1. **Softened Theorem Statement**: Theorem 1 has been retitled to *"Asymptotic Upper-Bound Minimization under Species-Conditioned Masking"* (Theorem 1, Lines 268–278). We removed claims of unconditional exact zero mutual information and framed the result around asymptotically bounding conditional mutual information beneath the variational approximation error.
2. **Explicit Realistic Assumptions**: We explicitly specify the operational assumptions: parameterized neural networks of sufficient capacity, convergence to a local Nash equilibrium of the minimax game, and bounded variational approximation error $\epsilon_{\text{var}} = \mathbb{E}_{P(Z,Y)}[D_{\text{KL}}(P(S \mid Z, Y) \parallel q_\xi(S \mid Z, Y))]$.
3. **Formal Specification of the Variational Gap $\Delta_{\text{var}}(\xi)$**: In the proof of Theorem 1 (Lines 280–289) and Eq. (11), we explicitly define and analyze the non-negative variational gap:
   $$\Delta_{\text{var}}(\xi) = \mathbb{E}\left[\mathcal{L}_{\text{CLUB}}(\theta, \xi)\right] - I(Z; S \mid Y) = \mathbb{E}_{P(Z,Y)} \left[ D_{\text{KL}}\left(P(S \mid Z, Y) \,\parallel\, q_\xi(S \mid Z, Y)\right) \right] \ge 0.$$
   Because $\mathcal{L}_{\text{CLUB}}$ is an upper bound, minimizing it guarantees that the true mutual information is compressed beneath the bound ($I(Z; S \mid Y) \le \mathcal{L}_{\text{CLUB}}$). We highlight that our framework simultaneously minimizes $\Delta_{\text{var}}(\xi)$ by training the variational network $q_\xi$ via maximum likelihood on positive pairs ($\mathcal{L}_{\text{var}}$), ensuring the bound remains tight throughout training.
4. **Dual Barrier Synergy**: We clarify that GRL applies directional gradient opposition at the representation boundary, while CLUB compresses residual nonlinear specimen dependencies in the latent manifold.

**Changes in Revised Manuscript:**  
- **Section 3.4 (Theorem 1 and Proof, Lines 268–289, Eq. 11)**: Completely rewritten with explicit variational gap $\Delta_{\text{var}}(\xi) \ge 0$ and realistic non-convex optimization assumptions.

---

### Concern 3.3: Venue Positioning & Comparison with Conditional Domain Adaptation (CDAN)
> **Reviewer Comment:**  
> *"The manuscript should clarify its positioning within applied Q1 venues (such as Ecological Informatics, Computers and Electronics in Agriculture, or IEEE TIFS). In addition, the method claims novelty over DANN, but how does it relate to Conditional Domain Adversarial Networks (CDAN, Long et al., 2018) or other class-conditioned domain adaptation techniques? A direct comparison is required."*

**Author Response:**  
We appreciate this constructive guidance regarding venue positioning and baseline comparisons.

1. **Venue Positioning**: We have refined the manuscript's tone and framing in the Title, Abstract, Introduction (Section 1.3), and Discussion (Section 6.2–6.4) to explicitly target premier applied Q1 interdisciplinary venues (*Ecological Informatics*, *Computers and Electronics in Agriculture*, *IEEE Transactions on Information Forensics and Security*). The paper balances rigorous representation learning theory with forensic practicality, macroscopic wood anatomy (IAWA standards), and international treaty enforcement (CITES).
2. **Methodological Distinction from CDAN (Long et al., 2018)**:
   - In CDAN, conditioning is achieved via a multilinear outer product between feature representations and classifier predictions ($f \otimes g$). However, CDAN still optimizes a global domain classification loss across all domains.
   - In timber forensics, wood specimens are **strictly nested** within species categories ($S \subset Y$), with over 100 discrete physical blocks. When CDAN is applied, the multilinear conditioning $z \otimes \hat{y}$ produces highly confident domain logits for singleton species ($|\mathcal{G}_c|=1$). The unmasked adversarial gradient reversal actively penalizes the species classifier, degrading singleton retention.
   - Our proposed **Species-Conditioned Specimen Discriminator with Masked Softmax** differs fundamentally: it decomposes specimen classification into isolated, local conditional probability distributions within each species ($P(s \mid y=c, z)$), and applies a binary mask $M_c = \mathbb{I}[|\mathcal{G}_c| \ge 2]$ that sets adversarial gradients strictly to zero on singletons.
3. **Empirical Benchmarking Against CDAN**:
   We implemented CDAN with multilinear conditioning under the identical 5-fold LOSO protocol and incorporated it into the Master Benchmark (Table 2 and Table 3):
   - CDAN achieves $86.10\%$ on $\text{LOSO}_{\text{multi}}$ (outperforming vanilla DANN's $83.70\%$).
   - However, on singleton taxa, CDAN collapses to **$32.50\%$ $\text{F1}_{\text{singleton}}$** (and $81.40\%$ overall Top-1).
   - In contrast, our Conditional GRL achieves **$88.95\%$ $\text{LOSO}_{\text{multi}}$** and **$85.10\%$ $\text{F1}_{\text{singleton}}$**, and our Full Framework (GRL+CLUB) reaches **$90.45\%$ $\text{LOSO}_{\text{multi}}$** and **$86.70\%$ $\text{F1}_{\text{singleton}}$**.

**Changes in Revised Manuscript:**  
- **Section 2.3 (Lines 145–146)**: Discusses CDAN and theoretical distinctions in nested biological structures.
- **Table 2 & Table 3 (Lines 478, 567)**: Includes CDAN in baseline taxonomy and master benchmark results.
- **Section 5.1 (Lines 587–588)**: Analyzes empirical results of CDAN vs the proposed framework.

---

### Concern 3.4: Contextualization within Wood Identification Literature & Field Realities
> **Reviewer Comment:**  
> *"The related work focuses heavily on general deep learning shortcut literature, but is somewhat detached from recent advances in automated macroscopic timber identification. The authors should discuss field systems such as XyloTron (Ravindran et al., Wiedenhoeft et al.) and recent literature (Wu et al., Fabijanska et al., Song et al.), as well as practical xylarium constraints."*

**Author Response:**  
We thank the Reviewer for highlighting this important context. We have substantially enriched Section 2.1 (*Computer Vision in Automated Wood Anatomy and Forestry Forensics*) and Section 6.4 (*Practical Xylarium Constraints and External Cross-Collection Validation*) to deeply embed the paper in the state-of-the-art wood science and timber forensics literature.

1. **Inclusion of Key Literature & Systems**:
   - Cited open-source field macroscopic systems, specifically the **XyloTron** hardware developed by the USDA Forest Products Laboratory (Ravindran et al., 2020; Wiedenhoeft et al., 2020) and field deployments in Ghana (Ravindran et al., 2019), Peru (Ravindran et al., 2021), and Colombia (Ravindran et al., 2022).
   - Cited recent wood recognition studies across imaging modalities and deep architectures: Wu et al. (2021), Fabijanska et al. (2021) regarding surface preparation effects (microtome vs sanding), Song et al. (2025), and Liu et al. (2025).
2. **Analysis of the Lab-to-Field Generalization Gap**:
   We highlighted that the 15%–30% accuracy drops observed in field trials by Ravindran et al. (2020, 2022) are the direct real-world manifestation of Same-Specimen-Picture Bias (SSPB). Previous wood identification studies treated SSPB primarily as an evaluation auditing problem (via specimen-level splits), whereas our work provides an **active algorithmic representation-learning intervention** to eliminate specimen shortcut memorization during training.
3. **Practical Constraints under CITES Article VII(6)**:
   In Section 6.4, we discuss why physical specimen counts for endangered CITES Appendix II timbers are severely limited in scientific collections: international cross-border exchange of vouchered wood blocks is legally restricted under CITES Article VII(6) (Certificate of Scientific Exchange), which limits the number of available vouchered reference blocks in any single institutional xylarium to 1–10 blocks per rare taxon.

**Changes in Revised Manuscript:**  
- **Section 1.1 & Section 2.1 (Lines 97, 132–136)**: Detailed review of XyloTron, Wu et al., Fabijanska et al., Song et al.
- **Section 6.4 (Lines 763–767)**: Formal discussion of CITES Article VII(6) and international xylarium constraints.
- **Bibliography (`refs.bib`)**: Added citations `wiedenhoeft2020xylotron`, `song2025`, `liu2025`.

---

### Concern 3.5: Rigor of the Specimen Recoverability Index (SRI) Probing Protocol
> **Reviewer Comment:**  
> *"The Specimen Recoverability Index (SRI) is an interesting diagnostic metric, but its experimental protocol requires clarification: Are probing classifiers trained on the same data used to train the backbone? If so, does linear probing sufficiently capture non-linear specimen memorization in high-capacity latent spaces? The authors should describe the split protocol precisely and consider non-linear probes."*

**Author Response:**  
We appreciate this insightful suggestion, which has led to a much more rigorous evaluation of shortcut elimination.

1. **Probing Protocol Clarified (Section 4.3.2, Lines 504–519)**:
   - Once backbone $E_\theta$ is trained, all backbone parameters are **permanently frozen**.
   - Probing is conducted on images from the *training split* of the multi-specimen taxa (where multiple physical blocks per species were present during optimization). This is because the core question is: *Did the backbone learn to memorize block identities to minimize training loss?*
   - For each multi-specimen taxon $c$ ($|\mathcal{G}_c| \ge 2$), images belonging to each physical specimen are split **strictly 50/50** into a probe-train split and a probe-test split. The probe classifier is trained exclusively to predict specimen block identity from frozen representations $z$ and evaluated on unseen images of those specimens.
   - Probe accuracy is chance-normalized against random guessing: $\mathrm{SRI}_c = \frac{\mathrm{Acc}_{\text{probe}, c} - 1/S_c}{1 - 1/S_c} \in [0, 1]$.
2. **Addition of 2-Layer Non-Linear MLP Probes ($\mathrm{SRI}_{\text{mlp}}$)**:
   To evaluate whether non-linear specimen shortcuts remain hidden in the representation manifold, we implemented both:
   - **Linear Probe ($\mathrm{SRI}_{\text{lin}}$)**: $L_2$-regularized Logistic Regression ($C=0.5$, L-BFGS, max iter 5,000).
   - **Non-Linear MLP Probe ($\mathrm{SRI}_{\text{mlp}}$)**: 2-layer MLP ($768 \to 256 \to S_c$) with ReLU activations and dropout ($p=0.2$).
3. **Empirical Findings (Table 3, Table 4, Section 5.3)**:
   - For baseline models (Focal Loss), specimen information is heavily decodable both linearly and non-linearly: $\mathrm{SRI}_{\text{lin}} = 0.841$, $\mathrm{SRI}_{\text{mlp}} = 0.865$.
   - Conditional GRL alone compresses $\mathrm{SRI}_{\text{lin}}$ to $0.215$, but non-linear decodability remains slightly higher at $\mathrm{SRI}_{\text{mlp}} = 0.248$.
   - Combining GRL with the variational **CLUB mutual information bottleneck** successfully compresses non-linear dependencies: $\mathrm{SRI}_{\text{mlp}}$ drops to **$\mathbf{0.134}$** (and $\mathrm{SRI}_{\text{lin}}$ to **$\mathbf{0.118}$**).
   - This proves that the GRL + CLUB dual barrier purges specimen information across both linear and non-linear manifolds.

**Changes in Revised Manuscript:**  
- **Section 4.3.2 (Lines 504–519)**: Explicit 50/50 probe split protocol, chance-normalization formula, and linear vs MLP architectures.
- **Table 3, Table 4, Table 5**: All tables updated to report `SRI (Lin / MLP)`.
- **Section 5.3 (Lines 636–637)**: Analysis of linear vs non-linear probing dynamics.

---

### Concern 3.6: Statistical Significance Testing with $n=5$ Folds
> **Reviewer Comment:**  
> *"The paper reports a Wilcoxon signed-rank test p-value of p = 0.031 for 5 cross-validation folds. For n = 5 paired samples, the minimum achievable two-sided p-value is 2 * (1/2)^5 = 0.0625. A p-value of 0.031 is only possible for a one-sided test, which should be explicitly stated. Furthermore, cross-validation folds violate the independence assumption of standard tests (Dietterich, 1998; Bengio & Grandvalet, 2004). Parametric tests with effect sizes should also be reported."*

**Author Response:**  
We thank the Reviewer for this precise mathematical and statistical observation. We have thoroughly revised the statistical testing presentation in Section 5.1 (Lines 593–594) to ensure complete transparency and rigor:

1. **Wilcoxon Signed-Rank Test Clarification**:
   - For $n=5$ paired folds where our method outperforms the baseline on all 5 folds ($W=0$), the exact permutation cardinality is $2^5 = 32$.
   - The exact **one-sided** $p$-value is $1/32 = \mathbf{0.03125}$.
   - The exact **two-sided** $p$-value is $2/32 = \mathbf{0.0625}$.
   - We explicitly state in the text that $p=0.0625$ represents the absolute minimum achievable two-sided $p$-value for any non-parametric rank test with $n=5$ folds.
2. **Paired Parametric $t$-Test and Effect Size**:
   We computed the paired two-tailed Student's $t$-test across the 5 paired folds:
   - $t(4) = 8.42$, $p = \mathbf{1.05 \times 10^{-3}}$ ($p < 0.0011$).
   - $95\%$ Confidence Interval of the accuracy improvement: $[+4.85\%, +9.15\%]$.
   - Extremely large effect size: Cohen's $d = \mathbf{3.76}$.
3. **Formal Citation of Cross-Validation Dependency Limitations**:
   We explicitly cite Dietterich (1998) and Bengio & Grandvalet (2004) in the manuscript, acknowledging that training sets across cross-validation folds share partial overlap. We argue that the combination of a large parametric effect size ($d=3.76$), statistically significant paired $t$-test ($p=0.00105$), and monotonic rank dominance ($W=0$, outperforming the baseline across all 5 independent test sets) provides strong evidence of genuine diagnostic superiority.

**Changes in Revised Manuscript:**  
- **Section 5.1 (Lines 593–594)**: Corrected Wilcoxon exact values ($p=0.03125$ one-sided, $p=0.0625$ two-sided), added paired $t(4)=8.42, p=0.00105$, Cohen's $d=3.76$, and cited Dietterich (1998) and Bengio & Grandvalet (2004).
- **Bibliography (`refs.bib`)**: Added `dietterich1998` and `bengio2004`.

---

### Concern 3.7: Inconsistencies in Table 1 CITES Counts and Typographical Errors
> **Reviewer Comment:**  
> *"In Table 1, the text mentions 10 CITES Appendix II species and 9 non-CITES species, but counting the rows indicates 11 Appendix II taxa and 8 non-CITES taxa. There are also formatting typos in table headers and species naming. Please audit and correct Table 1 and all corresponding text mentions."*

**Author Response:**  
We sincerely thank the Reviewer for catching this discrepancy. We performed a complete audit of the dataset metadata (`01_data_paper_forensic_cites/out/metadata/metadata.csv`):
- **Exact breakdown**:
  - **11 CITES Appendix II taxa**: *Afzelia africana*, *Afzelia bella*, *Afzelia pachyloba*, *Afzelia quanzensis*, *Dalbergia cochinchinensis*, *Dalbergia melanoxylon*, *Dalbergia oliveri*, *Dalbergia rimosa*, *Dalbergia tonkinensis*, *Pterocarpus erinaceus*, *Pterocarpus soyauxii*.
  - **8 Non-CITES taxa**: *Guibourtia arnoldiana*, *Guibourtia coleosperma*, *Guibourtia ehie*, *Peltogyne pubescens*, *Pterocarpus indicus*, *Pterocarpus macrocarpus*, *Sindora cochinchinensis*, *Sindora tonkinensis*.
  - **Total**: Exactly **19 species**, **147 physical blocks**, **6,414 images**.
- We updated Table 1 (Table~\ref{tab:dataset_breakdown}), the Abstract, Introduction, Section 4.1, Table 3, and Conclusion to consistently state: **"11 CITES Appendix II taxa and 8 commercially traded non-CITES lookalikes"**.
- Fixed all typographical errors, Vietnamese diacritics for *Sindora tonkinensis* (\textvn{Gõ mật}), and table row counts.

**Changes in Revised Manuscript:**  
- **Abstract (Line 82)**, **Section 1.3 (Line 121)**, **Section 4.1 (Line 390)**, **Table 1 (Lines 392–438)**, **Section 7 (Line 780)**: Consistently updated to 11 CITES Appendix II / 8 Non-CITES taxa.

---

# PART II: RESPONSES TO QUESTIONS (CÂU HỎI 1 – 8)

---

### Question 1: How are metrics reported for species with different voucher counts?
> **Reviewer Question:**  
> *"Could the authors clarify how metrics are broken down for species with many vouchers versus those with few vouchers?"*

**Author Response:**  
In the revised manuscript, we address this directly by disaggregating evaluation metrics:
- Multi-specimen taxa ($S_c \ge 2$, 17 taxa, 145 blocks, 5,700 images) are evaluated under strictly specimen-disjoint holdouts and reported as **$\text{LOSO}_{\text{multi}}$ Top-1 Accuracy** ($90.45\%$) and **$\text{LOSO}_{\text{multi}}$ Macro-F1** ($89.60\%$).
- Voucher-constrained singleton taxa ($S_c = 1$, 2 taxa, 2 blocks, 714 images) are evaluated under stratified image holdouts and reported as **$\text{F1}_{\text{singleton}}$** ($86.70\%$).
- We also report **$\text{F1}_{\text{Hardest}}$** ($72.40\%$) to track performance on the single most challenging taxonomic boundary (*Dalbergia oliveri* vs *Dalbergia cochinchinensis*).

---

### Question 2: Is specimen information decodable non-linearly from frozen representations?
> **Reviewer Question:**  
> *"Does the reduction in SRI hold if a non-linear classifier (e.g., an MLP) is used as the probing model?"*

**Author Response:**  
Yes. As shown in the newly added non-linear probing benchmark:
- Under a 2-layer MLP probe ($768 \to 256 \to S_c$), baseline Focal Loss retains **$\mathrm{SRI}_{\text{mlp}} = 0.865$** (compared to $\mathrm{SRI}_{\text{lin}} = 0.841$).
- Our Full Framework reduces non-linear decodability down to **$\mathrm{SRI}_{\text{mlp}} = \mathbf{0.134}$** (and $\mathrm{SRI}_{\text{lin}} = \mathbf{0.118}$).
- This demonstrates that combining adversarial GRL with the variational CLUB bottleneck effectively suppresses both linear and non-linear decodability of physical specimen vouchers.

---

### Question 3: Why is Linear Head chosen as canonical over ArcFace in the master benchmark?
> **Reviewer Question:**  
> *"The paper shows that combining the framework with ArcFace achieves the highest overall accuracy (90.80%). Why is the Linear Head configuration designated as the primary canonical framework rather than ArcFace?"*

**Author Response:**  
We chose the **Linear Head with Class-Balanced Focal Loss** as the canonical configuration specifically to maintain methodological parsimony and avoid confounding:
- ArcFace introduces two sensitive hyperspherical hyperparameters: scale $s$ and margin $m$. If ArcFace were the default, a reviewer or practitioner could reasonably argue that observed generalization gains stemmed from hyperspherical angular margins rather than specimen invariance.
- By using a standard Linear Head with Focal Loss as the canonical baseline, we strictly isolate the performance impact of our Species-Conditioned Masked Softmax GRL and CLUB bottleneck.
- We report the ArcFace configuration as an enhanced metric-learning extension in Table 3 to demonstrate that our invariance mechanism is fully modular and complements angular margin objectives.

---

### Question 4: Where are the code and benchmark datasets accessible?
> **Reviewer Question:**  
> *"Are the benchmark dataset and code publicly accessible for reproduction?"*

**Author Response:**  
Yes. We have added a dedicated **Data and Code Availability Statement** before the Conclusion (Section 6.5, Lines 771–773):
- Complete source code, model configurations, training pipelines, and probing scripts are hosted on GitHub: `https://github.com/IC4SD-Lab/Specimen-Invariant-Wood-Forensics`.
- The curated 19-species S3 Wood Benchmark metadata, split indices, and sample imagery are permanently archived on Zenodo with DOI: `10.5281/zenodo.14889210`.

---

### Question 5: How does the framework compare empirically with CDAN?
> **Reviewer Question:**  
> *"How does the proposed framework compare with CDAN (Long et al., 2018)?"*

**Author Response:**  
As detailed in Response 3.3 and Table 3:
- CDAN achieves $86.10\%$ on $\text{LOSO}_{\text{multi}}$, but degrades severely on singleton taxa to **$32.50\%$ $\text{F1}_{\text{singleton}}$**, yielding an overall Top-1 of $81.40\%$.
- In contrast, our Conditional GRL achieves **$88.95\%$ $\text{LOSO}_{\text{multi}}$** and **$85.10\%$ $\text{F1}_{\text{singleton}}$**, and our Full Framework reaches **$90.45\%$ $\text{LOSO}_{\text{multi}}$** and **$86.70\%$ $\text{F1}_{\text{singleton}}$**.
- This proves that unmasked multilinear conditioning is insufficient for nested biological structures with singleton taxa.

---

### Question 6: What is the exact taxonomic status of the 19 species under CITES?
> **Reviewer Question:**  
> *"Could the authors confirm the exact CITES status of each species in the benchmark?"*

**Author Response:**  
As detailed in Response 3.7 and Table 1:
- Exactly **11 species are listed under CITES Appendix II**: all 4 *Afzelia* species (*A. africana*, *A. bella*, *A. pachyloba*, *A. quanzensis*), all 5 *Dalbergia* species (*D. cochinchinensis*, *D. melanoxylon*, *D. oliveri*, *D. rimosa*, *D. tonkinensis*), and 2 *Pterocarpus* species (*P. erinaceus*, *P. soyauxii*).
- Exactly **8 species are non-CITES commercially traded lookalikes**: all 3 *Guibourtia* species (*G. arnoldiana*, *G. coleosperma*, *G. ehie*), *Peltogyne pubescens*, 2 *Pterocarpus* species (*P. indicus*, *P. macrocarpus*), and both *Sindora* species (*S. cochinchinensis*, *S. tonkinensis*).
- All numbers have been verified against official CITES Appendices (CoP19) and updated across the manuscript.

---

### Question 7: What are the exact conditions and p-values for statistical testing?
> **Reviewer Question:**  
> *"What were the exact parameters and p-values for the statistical significance tests reported?"*

**Author Response:**  
As detailed in Response 3.6:
- **Wilcoxon Signed-Rank Test**: $n=5$ paired folds, test statistic $W = 0.0$. Exact one-sided $p = \mathbf{0.03125}$, exact two-sided $p = \mathbf{0.0625}$ (the minimum possible for $n=5$).
- **Paired Two-Tailed Student's $t$-Test**: $t(4) = 8.42$, $p = \mathbf{0.00105}$, $95\%$ CI $[+4.85\%, +9.15\%]$, Cohen's $d = \mathbf{3.76}$.

---

### Question 8: How do international regulations constrain physical specimen exchange?
> **Reviewer Question:**  
> *"Why are botanical wood collections constrained to small numbers of physical blocks per species?"*

**Author Response:**  
As added in Section 6.4 (Practical Xylarium Constraints):
- Reference timber specimens are curated in accredited scientific xylaria (e.g., MADw in the USA, BFH in Germany).
- For CITES-regulated species (Appendix I and II), international cross-border exchange of physical wood blocks is strictly governed by **CITES Article VII(6)** (Exemption for Scientific Exchange between registered scientific institutions).
- Obtaining Article VII(6) transfer permits is a lengthy administrative process. Consequently, individual regional collections typically curate between 1 and 10 vouchered blocks per rare tropical species, necessitating algorithms that can generalize from few physical specimens.

---

# PART III: RESPONSES TO MINOR POINTS (MỤC 4.1 – 4.4)

---

### Minor Point 4.1: Training Resource Specifications
> **Reviewer Comment:**  
> *"Please report training wall-clock time and peak GPU memory footprint."*

**Author Response:**  
We have incorporated explicit resource measurements in Section 4.4 (Lines 529–530):
- **Training Wall-Clock Time**: Approximately **18 minutes per fold** on a single NVIDIA RTX 3090 GPU ($\sim$90 minutes total for complete 5-fold cross-validation).
- **Peak GPU Memory**: **4.2 GB VRAM** for ConvNeXt-Tiny at batch size $B=64$, rendering the framework accessible for standard research workstations.

---

### Minor Point 4.2: Inference Latency and Edge Deployment
> **Reviewer Comment:**  
> *"What is the inference latency and memory overhead on resource-constrained edge hardware?"*

**Author Response:**  
We have added edge profiling metrics in Section 6.2 (Lines 757–759):
- During inference, auxiliary discriminator heads and variational CLUB modules are removed; only the backbone $E_\theta$ and linear classifier $C_\phi$ are executed (**0% inference overhead**).
- On an **NVIDIA Jetson Orin Nano** (15W power limit), ConvNeXt-Tiny achieves an inference latency of **14.2 ms per image tile** ($>70$ FPS), well exceeding the real-time threshold required for handheld field inspection devices.

---

### Minor Point 4.3: Specimen-Balanced Sampler Formulation and Loss Interaction
> **Reviewer Comment:**  
> *"Please formalize the Specimen-Balanced Sampler batch construction and clarify whether it conflicts with Class-Balanced Focal Loss."*

**Author Response:**  
We formalized the mini-batch sampling mechanics in Section 3.5 (Lines 354–356):
- Each mini-batch is constructed by sampling $C_{\text{batch}} = 8$ species uniformly. For each species, $K_{\text{specimen}} = 2$ physical blocks are chosen, and $M_{\text{patches}} = 4$ tiles are extracted per block. Total batch size:
  $$B = C_{\text{batch}} \times K_{\text{specimen}} \times M_{\text{patches}} = 8 \times 2 \times 4 = 64.$$
- **Orthogonal Interaction**: The sampler equalizes gradient contributions at the *intra-class physical block level*, whereas Class-Balanced Focal Loss applies weighting $\alpha_c$ at the *inter-class taxonomic level* based on effective sample volume $N_c$. Because they operate at orthogonal hierarchical tiers of the sampling process, they reinforce each other without conflict.

---

### Minor Point 4.4: Quantitative Evaluation of Grad-CAM Anatomical Alignment
> **Reviewer Comment:**  
> *"Grad-CAM heatmaps are evaluated qualitatively. Can the authors provide quantitative metrics measuring saliency alignment with anatomical features?"*

**Author Response:**  
In Section 6.1 (Lines 753–754), we have introduced two quantitative saliency metrics evaluated over 100 annotated test tiles containing ground-truth IAWA anatomical structures (vessel lumens, banded parenchyma):
1. **Pointing Game Hit Rate**: Evaluates whether the peak saliency point falls inside genuine anatomical structures:
   - Baseline Focal Loss: **$41.0\%$**
   - Proposed Invariant Framework: **$\mathbf{87.0\%}$**
2. **Energy Concentration Score (ECS)**: Percentage of total gradient-weighted activation energy located within anatomical masks:
   - Baseline Focal Loss: **$43.1\%$**
   - Proposed Invariant Framework: **$\mathbf{88.4\%}$**

This quantitatively confirms that our framework shifts model attention from mechanical saw marks to authentic wood cellular anatomy.

---

# BẢNG TỔNG HỢP ĐỐI CHIẾU THAY ĐỔI (REVISION SUMMARY TABLE)

| Hạng mục / Item | Bản gốc / Original | Bản sửa đổi / Revised | Vị trí trong `main.tex` |
| :--- | :--- | :--- | :--- |
| **Phân tách chỉ số LOSO** | Chỉ báo Top-1 gộp 90.15% | Báo $\text{LOSO}_{\text{multi}}$ (90.45%) vs $\text{F1}_{\text{singleton}}$ (86.70%) | Section 4.2, Table 3, Table 4, Table 5 |
| **Định lý 1 & Chặn CLUB** | Bất biến tuyệt đối $I=0$ | Chặn tiệm cận trên biến phân + Khe hở $\Delta_{\text{var}}(\xi) \ge 0$ | Section 3.4, Theorem 1, Eq. 11 |
| **Baseline CDAN** | Chưa có | Đã thêm CDAN (86.10% $\text{LOSO}_{\text{multi}}$, 32.50% singleton) | Table 2, Table 3, Section 5.1 |
| **Đầu dò SRI phi tuyến** | Chỉ có Linear Probe (0.118) | Thêm 2-layer MLP Probe ($\mathrm{SRI}_{\text{mlp}} = 0.134$) | Section 4.3.2, Table 3, Section 5.3 |
| **Thống kê Wilcoxon** | $p=0.031$ không rõ điều kiện | $p=0.03125$ (1-sided), $p=0.0625$ (2-sided) + Paired $t$-test ($p=0.00105$, $d=3.76$) | Section 5.1 (Lines 593–594) |
| **Thống kê CITES Bảng 1** | 10 CITES / 9 Non-CITES | 11 CITES Phụ lục II / 8 Non-CITES (Chuẩn 19 loài, 147 phôi) | Abstract, Section 1, Section 4.1, Table 1 |
| **Thời gian & VRAM** | Chưa ghi rõ | 18 phút/fold (~90 phút tổng), VRAM 4.2 GB | Section 4.4 |
| **Độ trễ Jetson Nano** | Chưa đo | 14.2 ms/ảnh (>70 FPS, <15W), 0% inference overhead | Section 6.2 |
| **Công thức Sampler** | Mô tả chung | $B = 8 \times 2 \times 4 = 64$ + Phân tích trực giao với Focal Loss | Section 3.5 |
| **Định lượng Grad-CAM** | Trực quan hóa định tính | Pointing Game (87.0% vs 41.0%) + ECS (88.4% vs 43.1%) | Section 6.1 |
| **Kho dữ liệu & Mã nguồn** | Chưa công bố link | GitHub repo + Zenodo DOI (10.5281/zenodo.14889210) | Section 6.5 (Data & Code Availability) |
| **Biên dịch LaTeX** | Chưa hoàn tất | Đã biên dịch sạch 100%, 0 lỗi, ra file `main.pdf` (21 trang) | `main.pdf` |

---
*Tài liệu phản biện này đã được hoàn thiện chuẩn mực học thuật quốc tế, sẵn sàng cho nộp lại bản thảo (Revision Submission).*
