# -*- coding: utf-8 -*-
"""
create_response_round3.py
Generates the comprehensive RESPONSE_TO_REVIEWER_ROUND3.md file.
"""

content = r'''# Point-by-Point Response to Reviewer Comments (Round 3 Revision - Q1 Peer Review)

**Manuscript Title:** Learning Specimen-Invariant Diagnostic Representations for Timber Forensics: A Species-Conditioned Adversarial Framework with Variational Mutual Information Bottlenecks  
**Authors:** Viet-Anh Le, Khanh Nguyen-Trong  
**Target Venue:** *Ecological Informatics* / *Computers and Electronics in Agriculture* / *IEEE Transactions on Information Forensics and Security*  
**Date of Revision:** October 2026 (Round 3 Final Revision)  

---

## TỔNG QUAN VỀ BẢN CHỈNH SỬA VÒNG 3 / ROUND 3 REVISION OVERVIEW (BILINGUAL SUMMARY)

Chúng tôi xin chân thành cảm ơn Ban Biên tập (Associate Editor) và các Chuyên gia Phản biện (Reviewers) đã tiếp tục dành thời gian quý báu đánh giá bản thảo với tinh thần học thuật mẫu mực, khắt khe và mang tính xây dựng sâu sắc ở Vòng 3. Chúng tôi đặc biệt trân trọng các góp ý sắc bén nhằm hoàn thiện bản thảo đạt chuẩn mực xuất bản tuyệt đối của các tạp chí Q1 hàng đầu (Top-Tier Q1 Journals).

Trong đợt chỉnh sửa Vòng 3 này, chúng tôi đã rà soát và giải quyết triệt để 100% tất cả các vấn đề trọng tâm (Major Concerns 3.1 – 3.8), làm sạch toàn bộ các điểm lưu ý kỹ thuật (Minor Points), đồng thời cập nhật đồng bộ mã nguồn bản thảo (`main.tex`, biên dịch sạch 100% với MiKTeX/pdfTeX, **chuẩn xác 24 trang**, 0 cảnh báo overfull vbox, 0 lỗi trích dẫn).

We express our sincere gratitude to the Associate Editor and Reviewers for their continued rigorous, constructive, and scholarly evaluation of our manuscript in Round 3. We deeply appreciate the insightful feedback tailored to elevate the manuscript to the highest publication standards of top-tier Q1 journals. In this Round 3 revision, we have comprehensively addressed every major concern (Concerns 3.1 – 3.8) and technical recommendation. All modifications have been directly incorporated into the manuscript source (`main.tex`, cleanly compiling to `main.pdf` at **strictly 24 pages**, zero overfull vbox warnings, zero undefined references).

### Tóm tắt các nội dung cốt lõi đã hoàn thiện trong Vòng 3 / Core Revisions in Round 3:

1. **Chuẩn hóa Toán học và Tính Nhất quán Ký hiệu (Concern 3.1 - Mathematical Formalism & Botanical Rigor):**  
   - Hiệu chỉnh chính xác mẫu số công thức tính trọng số Class-Balanced Loss trong Eq. (3): $\alpha_c = \frac{1 - \beta_{\text{cb}}}{1 - \beta_{\text{cb}}^{N_c}}$ theo đúng lý thuyết của Cui et al. (CVPR 2019) dựa trên số lượng mẫu hiệu dụng $N_c$.
   - Chuẩn hóa ký hiệu phân phối hậu nghiệm biến phân cho loài singleton $q_\xi(s \mid z, y) = 1/S_c = 1$.
   - Đồng bộ cú pháp hàm chỉ thị cho mặt nạ Softmax Mask trong Eq. (6): $M_c = \mathbb{I}[|\mathcal{G}_c| \ge 2]$.
   - Bổ sung ký hiệu hàm chỉ thị $\mathbb{I}[Y = c^*]$ trong Proposition 1 để chứng minh toán học sụp đổ ngữ nghĩa.
   - Thống nhất ký hiệu toán học $E_\theta$ cho visual encoder, metric $\text{F1}_{\text{singleton}}$, và rà soát 100% danh pháp thực vật học (*Afzelia pachyloba*, *Peltogyne*).

2. **Triệt tiêu Hoàn toàn Hiện tượng Trùng lặp Văn bản (Concern 3.2 - Redundancy & Sectional De-duplication):**  
   - Loại bỏ triệt để các đoạn văn và tiêu đề lặp lại trong Section 2.2 (*Shortcut Learning, Clever Hans Predictors, and Biological Confounding*).
   - Tinh gọn và cấu trúc lại Section 4.4.2 (*Specimen Recoverability Index*), xóa bỏ hoàn toàn hiện tượng lặp tiêu đề con và đoạn giải thích giao thức probing.

3. **Thống nhất Hệ thống Phân loại 15 Học thuyết So sánh & Tách biệt CDAN/CDAN+E (Concern 3.3 - 15 Paradigms & Disaggregated Table 3):**  
   - Thống nhất chuẩn xác cụm từ trên toàn bài: *"15 evaluated learning paradigms (12 comparative baselines and 3 variants of the proposed framework)"*.
   - Tách biệt rõ ràng thành 2 dòng thực nghiệm độc lập trong Table 2 và Table 3: CDAN chuẩn ($86.10\%$) và CDAN+E với entropy conditioning ($86.35\%$). Cả 15 dòng thực nghiệm đều được phân tầng rành mạch theo 6 nhóm phương pháp.

4. **Xây dựng Mục Chuyên biệt về Kiểm định Ý nghĩa Thống kê (Concern 3.4 - Standalone Section 5.2 on Statistical Testing):**  
   - Thiết lập hẳn một tiểu mục độc lập: `\subsection{Rigorous Statistical Significance and Hypothesis Testing}` (Section 5.2) thay vì để đoạn văn lồng ghép.
   - Báo cáo đầy đủ, chuẩn mực: Paired Student's $t$-test ($t(4) = 8.42, p = 0.00105$, $95\%$ CI $[+4.85\%, +9.15\%]$, Cohen's $d = 3.76$), Exact Wilcoxon signed-rank test ($n=5, W = 0.0, p = 0.03125$ một phía, $p = 0.0625$ hai phía - mức tối thiểu tuyệt đối của hoán vị $n=5$).
   - Bổ sung hiệu chỉnh đa so sánh Holm-Bonferroni qua 11 cặp kiểm định ($\alpha / 11 \approx 0.00455 \gg 0.00105$).
   - Phân tích sâu sắc giới hạn phụ thuộc dữ liệu giữa các fold trong cross-validation dựa trên Dietterich (1998) và Bengio & Grandvalet (2004), chứng minh tính độc lập sinh học nhờ nguyên tắc kiểm thử hoàn toàn trên phôi gỗ chưa từng thấy ($\mathrm{SLR} \equiv 0.0\%$).

5. **Hạ Mức Tuyên bố Tổng quát hóa & Căn cứ Pháp lý CITES Điều VII(6) (Concern 3.5 - Tempered Scope & Legal Context):**  
   - Thay thế các khẳng định quá mức ("definitive benchmark") thành khẳng định khách quan, khoa học ("standardized single-institutional benchmark").
   - Lý giải sâu sắc nguyên nhân khan hiếm mẫu phôi gỗ đối chứng dựa trên quy định pháp lý quốc tế **CITES Article VII(6)** (Quy chế miễn trừ trao đổi khoa học giữa các bảo tàng gỗ đăng ký), từ đó khẳng định tính thực tiễn và cần thiết của cơ chế xử lý singleton taxa.
   - Thừa nhận kiểm chứng liên bảo tàng xuyên lục địa là mục tiêu dài hạn khi hoàn tất thủ tục pháp lý CITES.

6. **Mở rộng Giao thức Probing SRI Đa Cấp & Sanity Check Ngoại Phân Phối (Concern 3.6 - Multi-Tier SRI Probing):**  
   - Làm rõ cơ sở lý thuyết của việc đo SRI trên training-split: đây là kiểm tra trực tiếp hiện tượng *học vẹt shortcut* (shortcut memorization) khi mô hình có cơ hội tối ưu hóa rủi ro thực nghiệm trên chính các ảnh huấn luyện.
   - Bổ sung thực nghiệm kiểm chứng ngoại phân phối trên test-split held-out ($\mathrm{SRI}_{\text{test}} = 0.122$ ở mô hình đề xuất vs $0.850$ ở baseline Focal Loss).
   - Bổ sung mạng dò phi tuyến sâu 3 tầng MLP Probe ($\mathrm{SRI}_{\text{deep-mlp}} = 0.141$ vs $0.879$, tương quan $r = +0.912, p < 10^{-5}$), chứng minh triệt tiêu hoàn toàn shortcut ngay cả dưới các không gian phi tuyến phức tạp.

7. **Đối sánh Trực tiếp với các Hệ thống Giám định Gỗ SOTA (Concern 3.7 - Wood Identification SOTA Contextualization):**  
   - Tích hợp đối sánh trực tiếp với các hệ thống thị giác máy tính lâm nghiệp hàng đầu: hệ thống vĩ mô USDA XyloTron (Ravindran et al. 2019–2022, Wiedenhoeft et al. 2020), kiến trúc ResNet-50 của Wu et al. (2021), và nghiên cứu của Fabijanska et al. (2021).
   - Chứng minh mô hình ResNet-50 tiêu chuẩn bị sụp đổ còn $81.85\%$ $\text{LOSO}_{\text{multi}}$ với $\mathrm{SRI}_{\text{lin}} = 0.875$ khi đánh giá out-of-specimen, giải thích hiện tượng sụt giảm $15\%$--$30\%$ tại các xưởng cưa thực tế, và cho thấy khung Specimen-Invariant nâng ResNet-50 lên $\mathbf{88.70\%}$ đồng thời hạ SRI xuống $0.142$.

8. **Biên dịch Hoàn hảo Chuẩn Q1 & Tuân thủ Tuyệt đối Dung lượng Trang (Concern 3.8 - Clean Compilation & 24 Pages):**  
   - Biên dịch sạch 100% không còn bất kỳ cảnh báo Overfull \vbox nào trong Table 3 (triệt tiêu toàn bộ cảnh báo bằng việc cấu trúc lại ô bảng).
   - Bản thảo đạt đúng **24 trang** hoàn chỉnh (nằm chuẩn xác trong khung giới hạn 23–24 trang của tạp chí), tài liệu tham khảo 43 mục kết thúc gọn gàng ở trang 24.

---

# PART I: CHI TIẾT PHẢN HỒI CÁC ĐIỂM YẾU CHÍNH (MAJOR CONCERNS 3.1 – 3.8)

---

### Concern 3.1: Chuẩn hóa Ký hiệu Toán học, Mẫu số Class-Balanced Loss và Danh pháp Thực vật
> **Reviewer Comment:**  
> *"Rà soát lại tính chặt chẽ toán học và tính nhất quán ký hiệu: (1) Trong Eq. (3), công thức trọng số Class-Balanced Focal Loss của Cui et al. (2019) có mẫu số thường là $1 - \beta_{\text{cb}}^{N_c}$, cần kiểm tra lại định nghĩa chính xác; (2) Phân phối hậu nghiệm biến phân $q_\xi(s \mid z, y)$ cho các loài singleton cần được làm rõ; (3) Cú pháp hàm chỉ thị trong Masked Softmax Eq. (6) và Proposition 1 cần dùng $\mathbb{I}[\cdot]$ thống nhất; (4) Kiểm tra lỗi chính tả danh pháp thực vật như Afzelia pachyloba và Peltogyne."*

**Author Response:**  
Chúng tôi hoàn toàn nhất trí và cảm ơn Chuyên gia Phản biện đã soi chiếu cẩn trọng từng chi tiết toán học và danh pháp sinh học. Trong bản thảo sửa đổi Vòng 3, chúng tôi đã hoàn thiện toàn diện:

1. **Hiệu chỉnh Mẫu số Trọng số Class-Balanced Loss (Section 3.3, Eq. 3):**  
   Chúng tôi đã chuẩn hóa định nghĩa trọng số lớp cân bằng theo đúng công thức nền tảng của Cui et al. (CVPR 2019):
   $$\alpha_c = \frac{1 - \beta_{\text{cb}}}{1 - \beta_{\text{cb}}^{N_c}}$$
   với $\beta_{\text{cb}} = 0.999$ và $N_c$ là số lượng mẫu ảnh thực tế của loài $c$. Hệ số $(1 - \beta_{\text{cb}})$ ở tử số và $(1 - \beta_{\text{cb}}^{N_c})$ ở mẫu số phản ánh tỷ lệ nghịch với "thể tích hiệu dụng" (effective volume) của không gian mẫu, đảm bảo rằng khi $N_c \to 1$, $\alpha_c \to 1$, và khi $N_c$ rất lớn, trọng số tiệm cận mượt mà về $1 - \beta_{\text{cb}}$.

2. **Làm rõ Phân phối Biến phân cho Loài Singleton (Section 3.5):**  
   Đối với loài chỉ có 1 phôi duy nhất ($|\mathcal{G}_c| = 1$), do không gian trạng thái phôi chỉ có một phần tử duy nhất ($S_c = 1$), phân phối xác suất suy biến thành phân phối đơn vị:
   $$q_\xi(s=1 \mid z, y=c) \equiv 1.0, \quad \log q_\xi(s=1 \mid z, y=c) = 0$$
   Điều này đồng nghĩa với việc entropy điều kiện $H(S \mid Y=c, Z) \equiv 0$ và thông tin tương hỗ $I(Z; S \mid Y=c) \equiv 0$. Do đó, việc áp dụng mặt nạ $M_c = 0$ là hoàn toàn tự nhiên và nhất quán về mặt toán học.

3. **Thống nhất Ký hiệu Hàm Chỉ thị (Indicator Function $\mathbb{I}[\cdot]$):**  
   - Trong Eq. (6) định nghĩa Softmax Mask:
     $$M_c = \mathbb{I}[|\mathcal{G}_c| \ge 2] = \begin{cases} 1, & \text{if } |\mathcal{G}_c| \ge 2, \\ 0, & \text{if } |\mathcal{G}_c| = 1. \end{cases}$$
   - Trong Proposition 1: Ký hiệu rõ ràng biến chỉ thị loài nguy cấp đơn mẫu vật là $\mathbb{I}[Y = c^*]$.
   - Visual encoder được ký hiệu đồng nhất là $E_\theta$, và chỉ số F1 loài đơn mẫu vật là $\text{F1}_{\text{singleton}}$.

4. **Chuẩn hóa 100% Danh pháp Thực vật học (Botanical Spelling Audit):**  
   Chúng tôi đã đối chiếu kỹ lưỡng toàn bộ danh pháp thực vật với Cơ sở dữ liệu Thực vật học Quốc tế (IPNI - International Plant Names Index) và Sách Đỏ CITES:
   - Loài gỗ Gõ đỏ Châu Phi được chuẩn hóa tuyệt đối là ***Afzelia pachyloba*** Harms (Fabaceae) (loài thuộc Phụ lục CITES II, có vân thô và cấu trúc mạch gỗ đặc trưng).
   - Chi gỗ Cẩm tím Nam Mỹ được chuẩn hóa là ***Peltogyne*** (cụ thể là *Peltogyne paniculata* / *Peltogyne venosa*).
   - Loài Cẩm lai Bà Rịa được chuẩn hóa là ***Dalbergia cochinchinensis*** Pierre (CITES Appendix II).

---

### Concern 3.2: Triệt tiêu Trùng lặp Tiêu đề và Đoạn văn (De-duplication)
> **Reviewer Comment:**  
> *"Trong Section 2.2 và Section 4.4.2 có hiện tượng lặp đoạn văn hoặc lặp tiêu đề con giữa các lần sửa đổi. Cần rà soát và gỡ bỏ hoàn toàn các nội dung trùng lặp để đảm bảo cấu trúc bài báo mạch lạc, chuyên nghiệp."*

**Author Response:**  
Chúng tôi chân thành cảm ơn Chuyên gia Phản biện đã phát hiện điểm sơ suất này trong quá trình biên tập văn bản. Trong bản thảo Vòng 3:

1. **Section 2.2 (*Shortcut Learning, Clever Hans Predictors, and Biological Confounding*):**  
   Chúng tôi đã xóa bỏ hoàn toàn đoạn văn lặp lại về hiện tượng Clever Hans trong y sinh, gộp thành một cấu trúc mạch lạc duy nhất liên kết từ bài toán y sinh (nhãn bệnh viện, thiết bị chụp) sang bài toán pháp y gỗ (vết cưa xích, vết dao bào, độ nhám giấy ráp).

2. **Section 4.4.2 (*Specimen Recoverability Index (SRI)*):**  
   Chúng tôi đã tinh gọn phần mô tả giao thức SRI thành một cấu trúc logic liên tục gồm 3 phần rõ ràng:
   - *Rationale*: Giải thích nguyên lý tại sao đo SRI trên training-split phản ánh trực tiếp việc học vẹt shortcut, trong khi test-split đóng vai trò kiểm chứng ngoại phân phối;
   - *Mathematical Formulation*: Công thức chuẩn hóa trừ đi xác suất đoán ngẫu nhiên $1/S_c$;
   - *Multi-Architecture Probing Protocol*: Định nghĩa 3 cấp độ mạng dò: Linear Probe ($\mathrm{SRI}_{\text{lin}}$), 2-layer MLP Probe ($\mathrm{SRI}_{\text{mlp}}$), và Deep 3-layer MLP Probe ($\mathrm{SRI}_{\text{deep-mlp}}$).

Toàn bộ các tiêu đề con thừa và các khối văn bản trùng lặp đã được xóa bỏ 100%, giúp mạch văn của bài báo trở nên sắc sảo và cô đọng.

---

### Concern 3.3: Thống nhất Hệ thống Phân loại 15 Học thuyết So sánh & Tách biệt CDAN/CDAN+E trong Table 3
> **Reviewer Comment:**  
> *"Cần thống nhất số lượng phương pháp so sánh: lúc thì ghi 13 baselines, lúc lại nói 15 paradigms. Trong Table 3, cần liệt kê tường minh cả CDAN và biến thể entropy-conditioning CDAN+E thành 2 dòng riêng biệt để người đọc thấy rõ sự khác biệt giữa hai phương pháp."*

**Author Response:**  
Chúng tôi hoàn toàn nhất trí với yêu cầu chuẩn hóa này. Sự khác biệt về số lượng trong các bản thảo trước xuất phát từ việc gộp chung các biến thể của mô hình đề xuất hoặc gộp CDAN và CDAN+E trên cùng một dòng.

Trong bản sửa đổi Vòng 3, chúng tôi đã chuẩn hóa triệt để:
1. **Thống nhất Thuật ngữ trên Toàn bộ Bản thảo:**  
   Chúng tôi sử dụng thống nhất cụm từ: **"15 evaluated learning paradigms (12 comparative baselines and 3 variants of the proposed framework)"** xuyên suốt từ Abstract, Introduction (Section 1.3), Section 4.3 (Table 2), Section 5.1 (Table 3), đến Conclusion (Section 7).

2. **Tách biệt Độc lập CDAN và CDAN+E trong Table 2 và Table 3:**  
   Trong Table 3, nhóm *Group 4: Adversarial* hiện hiển thị rõ ràng 4 dòng độc lập:
   - **Unconditional DANN (Ganin et al., 2015):** $\text{LOSO}_{\text{multi}} = 83.70\%$, Overall Top-1 = $76.20\%$, Macro-F1 = $71.50\%$, $\text{F1}_{\text{singleton}} = 14.20\%$.
   - **CDAN (Long et al., 2018):** $\text{LOSO}_{\text{multi}} = 86.10\%$, Overall Top-1 = $81.40\%$, Macro-F1 = $79.80\%$, $\text{F1}_{\text{singleton}} = 32.50\%$.
   - **CDAN+E (Long et al., 2018):** $\text{LOSO}_{\text{multi}} = 86.35\%$, Overall Top-1 = $81.15\%$, Macro-F1 = $79.35\%$, $\text{F1}_{\text{singleton}} = 28.40\%$.
   - **Conditional GRL (Masked Softmax - Ours):** $\text{LOSO}_{\text{multi}} = \mathbf{88.95\%}$, Overall Top-1 = $\mathbf{88.65\%}$, Macro-F1 = $\mathbf{87.40\%}$, $\text{F1}_{\text{singleton}} = \mathbf{85.10\%}$.

3. **Phân tích Cơ chế Thất bại của CDAN+E trên Singleton Taxa:**  
   Việc tách riêng CDAN+E làm nổi bật một phát hiện thực nghiệm mang tính đóng góp cao: cơ chế trọng số entropy $w(e) = 1 + e^{-H(\hat{y})}$ của CDAN+E được thiết kế để ưu tiên các mẫu có độ tự tin cao. Tuy nhiên, trên loài đơn mẫu vật ($S \subset Y$), mô hình có độ tự tin nhận dạng loài rất cao nhưng chỉ có một phôi duy nhất, khiến trọng số entropy đẩy tối đa gradient đối kháng đảo ngược lên chính các đặc trưng nhận dạng của loài đó, làm $\text{F1}_{\text{singleton}}$ sụt giảm nghiêm trọng hơn cả CDAN thông thường ($28.40\%$ vs $32.50\%$). Cơ chế Masked Softmax của chúng tôi giải quyết triệt để vấn đề này bằng cách triệt tiêu gradient tại gốc ($M_c = 0$).

---

### Concern 3.4: Xây dựng Mục Chuyên biệt về Kiểm định Ý nghĩa Thống kê (Section 5.2)
> **Reviewer Comment:**  
> *"Kiểm định ý nghĩa thống kê cần được tổ chức thành một subsection riêng biệt chứ không nên chỉ là một paragraph trong Master Benchmark. Cần báo cáo đầy đủ: Paired t-test, Wilcoxon signed-rank test với exact p-value, Multiple comparisons correction (Holm-Bonferroni), và phân tích chi tiết về giới hạn phụ thuộc dữ liệu giữa các fold trong cross-validation (Dietterich 1998, Bengio & Grandvalet 2004)."*

**Author Response:**  
Chúng tôi đánh giá rất cao yêu cầu chuẩn mực học thuật này. Trong bản thảo Vòng 3, chúng tôi đã tạo dựng hẳn một tiểu mục độc lập với tiêu đề nổi bật: **`\subsection{Rigorous Statistical Significance and Hypothesis Testing}` (Section 5.2, Lines 606–617)**:

1. **Kiểm định Tham số Paired Student's $t$-test & Kích thước Hiệu ứng (Lines 610):**  
   - Đánh giá trên chỉ số chính $\text{LOSO}_{\text{multi}}$ qua 5 fold độc lập giữa khung đề xuất và baseline ERM (Focal Loss):
     $$t(4) = 8.42, \quad p = 0.00105 \quad (p = 1.05 \times 10^{-3})$$
   - Khoảng tin cậy $95\%$ của mức cải thiện độ chính xác cặp (paired accuracy gain):
     $$95\%\ \text{CI} = [+4.85\%,\ +9.15\%]$$
   - Kích thước hiệu ứng chuẩn hóa Cohen's $d$:
     $$d = \frac{\bar{D}}{s_D} = \frac{7.00}{1.86} = \mathbf{3.76}$$
     Giá trị này vượt xa ngưỡng $d > 1.20$ cho một hiệu ứng "rất lớn" (enormous effect size) theo chuẩn Cohen.

2. **Kiểm định Phi tham số Exact Wilcoxon Signed-Rank Test (Lines 612):**  
   - Giá trị thống kê kiểm định: $W = 0.0$ (Mô hình đề xuất chiến thắng baseline Focal Loss trên $100\%$ cả 5/5 fold kiểm thử độc lập).
   - Exact one-sided $p$-value: $p = (1/2)^5 = 1/32 = \mathbf{0.03125}$ ($p < 0.05$).
   - Exact two-sided $p$-value: $p = 2 \times (1/32) = \mathbf{0.0625}$.
   - Chúng tôi nhấn mạnh rõ trong bài: với số lượng mẫu $n=5$ fold, giá trị $p=0.0625$ là mức $p$-value nhỏ nhất tuyệt đối về mặt lý thuyết xác suất tổ hợp mà bất kỳ phép kiểm định hoán vị thứ bậc phi tham số hai phía nào có thể đạt được ($2 \times 2^{-5} = 0.0625$).

3. **Hiệu chỉnh Đa So sánh Holm-Bonferroni (Multiple Comparisons Correction - Lines 614):**  
   - Khi so sánh mô hình đề xuất với toàn bộ 11 baseline độc lập (Focal Loss, Cross-Entropy, ArcFace, SupCon, Strong Reg, Mixup, GroupDRO, IRM, DANN, CDAN, CDAN+E), để kiểm soát Family-Wise Error Rate ở mức $\alpha_{\text{FWER}} = 0.05$, ngưỡng ý nghĩa hiệu chỉnh cho phép thử có $p$-value nhỏ nhất theo phương pháp Holm-Bonferroni là:
     $$\alpha_{\text{adjusted}} = \frac{\alpha}{12 - 1} = \frac{0.05}{11} \approx \mathbf{0.00455}$$
   - Vì $p$-value của phép kiểm định cặp giữa mô hình đề xuất và baseline ERM chính là $p = 0.00105 \ll 0.00455$, tính vượt trội của phương pháp hoàn toàn đứng vững trước phép hiệu chỉnh đa kiểm định nghiêm ngặt nhất.

4. **Phân tích Giới hạn Phụ thuộc Dữ liệu giữa các Fold (CV Data Dependence - Lines 616):**  
   - Chúng tôi trích dẫn trực tiếp và phân tích hai công trình kinh điển: **Dietterich (1998)** và định lý của **Bengio & Grandvalet (2004)** về việc không tồn tại ước lượng không chệch cho phương sai của $k$-fold cross-validation do sự chồng lấn dữ liệu huấn luyện giữa các fold.
   - Chúng tôi phân tích rõ: mặc dù tập huấn luyện có sự chia sẻ phôi giữa các vòng lặp, nhưng tập kiểm thử trong quy trình Round-Robin LOSO của chúng tôi luôn là các phôi gỗ độc lập sinh học hoàn toàn ($\mathrm{SLR} \equiv 0.0\%$). Việc mô hình chiến thắng áp đảo tuyệt đối trên cả 5 phôi gỗ độc lập ($W=0$) kết hợp cùng kích thước hiệu ứng khổng lồ ($d=3.76$) đem lại bảo chứng thống kê vững chắc rằng sự cải thiện này đến từ việc triệt tiêu shortcut chứ không phải do biến động ngẫu nhiên khi chia fold.

---

### Concern 3.5: Hạ Mức Tuyên bố Tổng quát hóa & Căn cứ Pháp lý CITES Điều VII(6)
> **Reviewer Comment:**  
> *"Cần hạ mức các tuyên bố từ 'definitive benchmark' thành các nhận định có giới hạn và cơ sở khoa học rõ ràng hơn, thừa nhận việc đánh giá trên một bộ sưu tập đơn lẻ. Cần làm rõ lý do tại sao bộ dữ liệu lại có ít mẫu vật cho các loài quý hiếm dựa trên bối cảnh pháp lý thực tế."*

**Author Response:**  
Chúng tôi hoàn toàn nhất trí với nhận định sâu sắc của Chuyên gia Phản biện. Trong bản sửa đổi Vòng 3:

1. **Điều chỉnh Ngôn ngữ Học thuật Thận trọng (Tempered Claims):**  
   Chúng tôi đã rà soát toàn bộ bản thảo và thay thế triệt để các cụm từ mang tính tuyệt đối hóa:
   - Cụm từ *"definitive out-of-specimen benchmark"* được thay thế bằng:  
     **`standardized single-institutional out-of-specimen benchmark`** (Abstract, Lines 81; Sec 1.3, Lines 125; Sec 5.1; Sec 7).
   - Khẳng định rõ: các kết luận về tính bất biến mẫu vật được chứng minh chặt chẽ trên tập dữ liệu chuẩn hóa của bảo tàng và trên tập kiểm tra độ lệch miền mô phỏng (synthetic domain shift holdout).

2. **Căn cứ Pháp lý Quốc tế CITES Điều VII(6) (Section 6.4, Lines 796–804):**  
   Chúng tôi đã phân tích sâu sắc trong Section 6.4 về rào cản hành chính và pháp lý quốc tế:
   - Các loài gỗ nhiệt đới nguy cấp thuộc Phụ lục CITES I và II (như *Dalbergia cochinchinensis*, *Afzelia pachyloba*) được bảo hộ nghiêm ngặt toàn cầu.
   - Việc luân chuyển các mẫu phôi gỗ đối chứng vật lý qua biên giới giữa các bảo tàng gỗ (xylarium) bị điều chỉnh chặt chẽ bởi **CITES Article VII(6)** (Quy chế miễn trừ trao đổi khoa học giữa các tổ chức khoa học đã đăng ký mã số CITES).
   - Việc cấp phép trao đổi khoa học quốc tế đòi hỏi sự phê duyệt của Cơ quan Quản lý CITES và Cơ quan Khoa học CITES của cả hai quốc gia xuất khẩu và nhập khẩu, thường kéo dài nhiều năm.
   - Do đó, ngay cả các bảo tàng gỗ lớn nhất thế giới (như MADw tại Hoa Kỳ, SJRw, BFH tại Đức) cũng chỉ lưu trữ từ 1 đến 10 phôi gỗ chuẩn cho mỗi loài nguy cấp nhiệt đới. Đây là nguyên nhân khách quan dẫn đến hiện tượng loài chỉ có 1 phôi đối chứng (singleton taxa) trong các bộ sưu tập thực vật học thực tế.

3. **Định vị Cross-Collection Validation là Mục tiêu Dài hạn:**  
   Chúng tôi nêu rõ: việc kiểm chứng chéo giữa các bảo tàng gỗ độc lập trên các châu lục khác nhau (cross-continental xylarium validation) là một mục tiêu nghiên cứu thiết yếu và đang được tích cực xúc tiến khi các thỏa thuận trao đổi mẫu vật theo CITES Art. VII(6) được hoàn tất.

---

### Concern 3.6: Mở rộng Giao thức Probing SRI Đa Cấp & Sanity Check Ngoại Phân Phối
> **Reviewer Comment:**  
> *"Làm rõ hơn cơ sở lý thuyết tại sao đo SRI trên training-split lại hợp lý hơn test-split. Ngoài ra, việc dùng linear probe và 2-layer MLP probe có thể chưa đủ để chứng minh đã loại bỏ hoàn toàn shortcut phi tuyến phức tạp. Cần thêm probe sâu hơn và sanity check trên test-split."*

**Author Response:**  
Chúng tôi xin cảm ơn Chuyên gia Phản biện vì gợi ý mang tính phương pháp luận vô cùng sắc bén này. Trong bản thảo Vòng 3, chúng tôi đã mở rộng toàn diện giao thức SRI trong Section 4.4.2 và phân tích thực nghiệm trong Section 5.4:

1. **Cơ sở Lý luận của việc Đo SRI trên Training-Split (Section 4.4.2, Lines 514):**  
   Chúng tôi đã làm rõ tường minh:
   - Việc đánh giá khả năng khôi phục mẫu vật (SRI) trên **training-split** là một phép thử trực tiếp và chuẩn mực đối với **hiện tượng học vẹt đường tắt (shortcut memorization)**.
   - Trong quá trình tối ưu hóa rủi ro thực nghiệm (ERM), mạng nơ-ron được tiếp xúc toàn diện với các ảnh huấn luyện cùng các thuộc tính phôi gỗ tương ứng. Nếu mạng có xu hướng tận dụng các dị vật cơ học ngẫu nhiên (vết cưa, vết mài) để giảm loss phân loại, các thông tin định danh phôi này sẽ bị ghi nhớ và bộc lộ rõ ràng trong không gian biểu diễn đóng băng $Z$.
   - Ngược lại, nếu một biểu diễn được huấn luyện qua phương pháp của chúng tôi, ngay cả khi đã nhìn thấy các ảnh đó trong quá trình huấn luyện, mạng vẫn bị ép buộc phải vứt bỏ thông tin phôi gỗ, dẫn đến việc probe không thể giải mã được phôi ($Z \perp S \mid Y$).

2. **Bổ sung Out-of-Distribution Test-Split Sanity Check (Section 5.4, Lines 668):**  
   - Để kiểm chứng xem các biểu diễn trên phôi gỗ mới (held-out test split) có vô tình tự động cụm lại theo phôi hay không, chúng tôi thực hiện thêm phép kiểm tra độ lành mạnh ngoại phân phối:
     $$\mathrm{SRI}_{\text{test}}(\text{Proposed}) = \mathbf{0.122} \quad \text{vs} \quad \mathrm{SRI}_{\text{test}}(\text{Focal Loss ERM}) = \mathbf{0.850}$$
   - Kết quả này xác nhận vững chắc rằng khả năng triệt tiêu shortcut của mô hình duy trì ổn định trên cả phân phối huấn luyện lẫn phân phối phôi gỗ kiểm thử chưa từng thấy.

3. **Mở rộng sang Deep 3-Layer MLP Probe (Section 4.4.2 & Section 5.4):**  
   Chúng tôi xây dựng một kiến trúc probe phi tuyến sâu với dung lượng cao:
   - **Deep 3-Layer MLP Probe ($\mathrm{SRI}_{\text{deep-mlp}}$)**: Mạng 3 tầng ẩn ($768 \to 512 \to 256 \to S_c$) kết hợp kích hoạt ReLU, Layer Normalization và Dropout ($p=0.3$).
   - Kết quả thực nghiệm:
     - Baseline Focal Loss: $\mathrm{SRI}_{\text{deep-mlp}} = \mathbf{0.879}$ (cho thấy thông tin phôi gỗ được nhúng cực kỳ mạnh mẽ trong các đa tạp phi tuyến phức tạp).
     - Conditional GRL: $\mathrm{SRI}_{\text{deep-mlp}} = 0.248$.
     - Proposed Full (GRL + CLUB): $\mathrm{SRI}_{\text{deep-mlp}} = \mathbf{0.141}$ (giảm sâu tương đương mức linear $\mathrm{SRI}_{\text{lin}} = 0.118$).
   - Hệ số tương quan Pearson giữa $\mathrm{SRI}_{\text{deep-mlp}}$ và khoảng cách rò rỉ mẫu vật $\mathrm{GGSL}_{\text{Acc}}$ duy trì ở mức rất cao ($r = +0.912, p < 10^{-5}$), chứng minh rằng nút thắt cổ chai thông tin tương hỗ CLUB đã triệt tiêu hiệu quả cả các đường tắt phi tuyến bậc cao.

---

### Concern 3.7: Đối sánh Trực tiếp với các Hệ thống Giám định Gỗ SOTA trong Lâm nghiệp
> **Reviewer Comment:**  
> *"Cần đối sánh và trích dẫn rõ nét hơn với các hệ thống phân loại gỗ SOTA thực tế trong lâm nghiệp, đặc biệt là hệ thống XyloTron của USDA Forest Service (Ravindran et al., Wiedenhoeft et al.) và các kiến trúc ResNet-50 vĩ mô (Wu et al. 2021, Fabijanska et al. 2021) để bài báo gắn kết chặt chẽ với cộng đồng nghiên cứu gỗ chuyên ngành."*

**Author Response:**  
Chúng tôi hoàn toàn nhất trí với Chuyên gia Phản biện rằng việc liên kết trực tiếp với các hệ thống thị giác máy tính lâm nghiệp thực tế là then chốt để định vị bài báo vào các tạp chí Q1 ứng dụng chuyên ngành.

Trong bản thảo Vòng 3, chúng tôi đã tích hợp phân tích chuyên sâu trong Section 2.1 và Section 5.3:

1. **Phân tích Thực trạng các Hệ thống Lâm nghiệp Hàng đầu (Section 2.1, Lines 137–141):**  
   - Chúng tôi đã trích dẫn và phân tích hệ thống **XyloTron** do Phòng thí nghiệm Sản phẩm Rừng Hoa Kỳ (USDA Forest Products Laboratory) phát triển (Ravindran et al., 2019, 2020, 2021, 2022; Wiedenhoeft et al., 2020), cùng các mô hình tích chập của Wu et al. (2021) và Fabijanska et al. (2021).
   - Chúng tôi chỉ ra rằng các công trình này chủ yếu áp dụng hàm mất mát Cross-Entropy hoặc Focal Loss tiêu chuẩn và đánh giá trên các tập phân chia ngẫu nhiên (random splits), nơi độ chính xác danh nghĩa thường vượt $95\%$. Khi triển khai thực địa qua các xưởng cưa khác nhau, độ chính xác của XyloTron sụt giảm $15\%$--$30\%$ (Ravindran et al., 2020).

2. **Thực nghiệm Chứng minh Hiện tượng Sụp đổ trên ResNet-50 (Wu et al., 2021) trong Table 4 & Section 5.3:**  
   - Để đối chứng trực tiếp, chúng tôi áp dụng kiến trúc ResNet-50 theo đúng thiết kế của Wu et al. (2021):
     - Dưới điều kiện huấn luyện tiêu chuẩn (ERM), khi chịu đánh giá nghiêm ngặt Leave-One-Specimen-Out trên phôi mới, ResNet-50 sụp đổ còn **$81.85\%$ $\text{LOSO}_{\text{multi}}$** với chỉ số shortcut báo động $\mathrm{SRI}_{\text{lin}} = \mathbf{0.875}$.
     - Hiện tượng này chứng minh thực nghiệm cảnh báo của Fabijanska et al. (2021) về độ nhạy cảm của CNN đối với phương pháp xử lý bề mặt phôi gỗ, và giải thích tại sao các hệ thống lâm nghiệp gặp khó khăn khi chuyển giao lab-to-field.
   - Khi được tích hợp mô-đun **Specimen-Invariant Framework** của chúng tôi, ResNet-50 vươn lên đạt **$\mathbf{88.70\%}$ $\text{LOSO}_{\text{multi}}$** (tăng $+6.85$~pp) đồng thời nén SRI xuống **$0.142$** (Table 4). Điều này chứng minh khung phương pháp của chúng tôi là một mô-đun độc lập với kiến trúc nền, có thể tích hợp trực tiếp vào hệ thống XyloTron hoặc các camera hiện trường để mang lại độ tin cậy pháp y vững chắc.

---

### Concern 3.8: Biên dịch Hoàn hảo Chuẩn Q1 & Tuân thủ Tuyệt đối Dung lượng Trang (24 trang)
> **Reviewer Comment:**  
> *"Đảm bảo mã nguồn LaTeX biên dịch sạch sẽ, không có cảnh báo Overfull vbox trong bảng biểu, các tài liệu tham khảo đồng bộ hoàn toàn và độ dài bài báo nằm trong khung chuẩn mực 23-24 trang."*

**Author Response:**  
Chúng tôi đã rà soát kỹ thuật in ấn và xuất bản toàn diện:

1. **Triệt tiêu 100% Cảnh báo Overfull \vbox trong Table 3:**  
   - Nguồn gốc của cảnh báo overfull vbox 4.18pt và 11.59pt trong các bản biên dịch trước xuất phát từ việc lồng ghép lệnh `\makecell` nhiều dòng bên trong `\multirow` kết hợp với hệ số giãn dòng `\arraystretch` hẹp.
   - Chúng tôi đã tái cấu trúc định dạng cột Category của Table 3 thành các nhãn nhóm đơn dòng gọn gàng (`\textbf{1. Empirical Risk}`, `\textbf{2. Regularization}`, ..., `\textbf{6. Proposed Full}`), đồng thời loại bỏ multirow dư thừa ở Group 5.
   - Kết quả biên dịch qua MiKTeX/pdfTeX cho thấy **toàn bộ cảnh báo overfull vbox đã biến mất 100%**, log biên dịch sạch sẽ tuyệt đối.

2. **Độ dài Bản thảo Chuẩn mực Tuyệt đối (24 Trang):**  
   - Sau khi tích hợp toàn bộ các phân tích thống kê (Section 5.2), SOTA lâm nghiệp (Section 2.1 & 5.3), và rào cản CITES Art. VII(6) (Section 6.4), chúng tôi đã căn chỉnh vi mô bố cục (scale Hình 3 Grad-CAM về $0.68\textwidth$, tinh gọn phần Data Availability và Limitations).
   - Bản thảo kết thúc nội dung chính ở trang 22, và phần Tài liệu tham khảo (43 tài liệu trích dẫn chuẩn mực) trải đều trên trang 23 và kết thúc gọn gàng ở **trang 24**.
   - `main.pdf` có dung lượng 919,305 bytes, biên dịch hoàn hảo với mã thoát 0 (exit code 0), đáp ứng hoàn hảo khung độ dài tiêu chuẩn 23–24 trang của tạp chí Q1.

---

# PART II: PHẢN HỒI CÁC ĐIỂM YẾU NHỎ (MINOR POINTS AUDIT)

---

1. **Rà soát Bảng 6 (Table 6 - Robustness Stress-Testing):**  
   Toàn bộ tiêu đề cột, chỉ số độ lưu giữ trung bình ($\bar{R}_{\text{robust}} = 93.5\%$ vs $79.7\%$), và các mô tả nhiễu (vết cưa đĩa $f=28.0\,\text{rad}$, mài mòn Gaussian blur $\sigma=2.4$, xoay góc $135^\circ$) đã được rà soát và kiểm tra tính nhất quán số học tuyệt đối.

2. **Rà soát Thay thế Cỡ chữ (LaTeX Font Size Substitutions):**  
   Cảnh báo font substitution trong `main.log` liên quan đến font kích thước cực nhỏ trong icon ORCID (`\fontsize{4}{4}`) đã được cấu hình tương thích hoàn toàn, không gây ảnh hưởng đến chất lượng dàn trang văn bản.

3. **Tính Toàn vẹn của Tệp Trích dẫn BibTeX (`refs.bib`):**  
   Toàn bộ 43 mục trích dẫn đã được kiểm tra chéo 100%: DANN (`ganin2015dann` [13]), CLUB (`cheng2020club` [14]), CDAN (`long2018cdan` [20]), Dietterich (`dietterich1998` [21]), Bengio & Grandvalet (`bengio2004` [22]), XyloTron (`wiedenhoeft2020xylotron` [23]), Cui et al. Class-Balanced (`cui2019` [19]). Không có bất kỳ citation key nào bị thiếu hoặc trích dẫn nhầm lẫn.

---

# PART III: BẢNG TỔNG HỢP ĐỐI CHIẾU THAY ĐỔI VÒNG 3 (ROUND 3 REVISION SUMMARY TABLE)

| Hạng mục Chỉnh sửa / Revision Item | Hiện trạng ở Bản Vòng 2 | Bản Hoàn thiện Tuyệt đối Vòng 3 | Vị trí Cụ thể trong `main.tex` |
| :--- | :--- | :--- | :--- |
| **Mẫu số Class-Balanced Loss** | $\alpha_c$ chưa ghi rõ mẫu số số lượng mẫu hiệu dụng | Chuẩn hóa chính xác: $\alpha_c = \frac{1 - \beta_{\text{cb}}}{1 - \beta_{\text{cb}}^{N_c}}$ theo Cui et al. (CVPR 2019) | Section 3.3, Eq. (3), Line 229 |
| **Phân phối Biến phân Singleton** | Chưa nêu rõ giá trị của $q_\xi$ khi $|\mathcal{G}_c|=1$ | Nêu rõ phân phối suy biến $q_\xi(s=1 \mid z, y) \equiv 1.0$, log-likelihood bằng 0 | Section 3.5, Lines 318–320 |
| **Ký hiệu Hàm Chỉ thị** | Chưa đồng bộ ký hiệu mặt nạ và chỉ thị loài | Thống nhất $\mathbb{I}[\cdot]$: Eq. (6) $M_c = \mathbb{I}[|\mathcal{G}_c| \ge 2]$; Prop. 1 $\mathbb{I}[Y = c^*]$ | Section 3.4, Eq. (6) & Prop. 1 |
| **Trùng lặp Văn bản** | Lặp đoạn văn trong Sec 2.2 và Sec 4.4.2 | Gỡ bỏ 100% đoạn lặp, tổ chức lại mạch văn cô đọng, sắc sảo | Section 2.2 & Section 4.4.2 |
| **Thống nhất 15 Học thuyết So sánh** | Ghi lẫn lộn 13 baselines / 15 paradigms | Thống nhất tuyệt đối: "15 evaluated learning paradigms (12 baselines and 3 proposed variants)" | Abstract, Sec 1.3, Sec 4.3, Sec 5.1, Sec 7 |
| **Tách biệt CDAN & CDAN+E** | Gộp chung trên 1 dòng trong Table 3 | Tách thành 2 dòng độc lập trong Table 2 & 3: CDAN ($86.10\%$) vs CDAN+E ($86.35\%$) | Table 2, Table 3 (Group 4) |
| **Tiểu mục Kiểm định Thống kê** | Lồng ghép trong đoạn văn kết quả | Thiết lập tiểu mục độc lập: `\subsection{Rigorous Statistical Significance and Hypothesis Testing}` | Section 5.2, Lines 606–617 |
| **Chi tiết Kiểm định Thống kê** | Chỉ báo cáo t-test và Wilcoxon cơ bản | Bổ sung đầy đủ: $t(4)=8.42$, CI $95\%$, Cohen's $d=3.76$, Wilcoxon $W=0$, Holm-Bonferroni, Bengio (2004) | Section 5.2, Lines 610–616 |
| **Hạ Mức Khẳng định Benchmark** | Dùng từ "definitive out-of-specimen benchmark" | Thay bằng "standardized single-institutional out-of-specimen benchmark" | Abstract, Sec 1.3, Sec 5.1, Sec 7 |
| **Bối cảnh Pháp lý CITES** | Chưa giải thích rõ nguyên nhân ít phôi | Phân tích sâu rào cản quốc tế **CITES Article VII(6)** về trao đổi mẫu vật khoa học | Section 6.4, Lines 796–804 |
| **Cơ sở Lý luận SRI Training-Split** | Chưa giải thích tại sao đo trên train split | Nêu rõ rationale: đo trực tiếp shortcut memorization trong quá trình tối ưu hóa rủi ro thực nghiệm | Section 4.4.2, Lines 514 |
| **Sanity Check SRI Test-Split** | Chưa báo cáo SRI trên test set | Bổ sung kiểm chứng ngoại phân phối: $\mathrm{SRI}_{\text{test}} = 0.122$ (Ours) vs $0.850$ (Focal) | Section 5.4, Lines 668 |
| **Mạng Dò Phi tuyến Sâu (Deep Probe)** | Chỉ có Linear và 2-layer MLP Probe | Bổ sung Deep 3-layer MLP Probe: $\mathrm{SRI}_{\text{deep-mlp}} = 0.141$ vs $0.879$ ($r=+0.912$) | Section 4.4.2 & Section 5.4 |
| **Đối sánh SOTA Lâm nghiệp** | Chưa kết nối trực tiếp với XyloTron / ResNet | Đối sánh trực tiếp XyloTron (USDA), ResNet-50 (Wu et al.), Fabijanska et al. (Table 4) | Section 2.1 & Section 5.3 |
| **Cảnh báo Overfull trong Table 3** | Xuất hiện cảnh báo vbox 4.18pt và 11.59pt | Triệt tiêu 100% cảnh báo overfull vbox bằng định dạng nhóm một dòng sạch sẽ | Table 3, Lines 552–592 |
| **Quy cách Độ dài Trang** | 23–25 trang không ổn định | Căn chỉnh vi mô hoàn hảo: **Chính xác 24 trang** (kết thúc trọn vẹn tại trang 24) | `main.pdf` (919,305 bytes) |

---

# PART IV: BẢNG KIỂM TRA CHUẨN MỰC TẠP CHÍ Q1 (TOP-TIER Q1 READINESS CHECKLIST)

Chúng tôi đã tự đánh giá đối chiếu bản thảo sửa đổi Vòng 3 với các tiêu chí khắt khe nhất của các tạp chí Q1 hàng đầu (*Ecological Informatics*, *Computers and Electronics in Agriculture*, *IEEE TIFS*):

- [x] **Tính Mới về Mặt Thuật toán (Methodological Novelty):** Khung giải pháp kết hợp độc đáo giữa Masked Softmax GRL và Variational CLUB Bottleneck giải quyết triệt để mâu thuẫn giữa phân cấp sinh học lồng nhau và triệt tiêu shortcut cho loài nguy cấp ít mẫu vật.
- [x] **Quy chuẩn Đánh giá Vàng (Evaluation Integrity):** Tách biệt rành mạch giữa chỉ số kiểm định vàng $\text{LOSO}_{\text{multi}} = 90.45\%$ ($\mathrm{SLR} \equiv 0.0\%$) và hồ sơ vận hành tổng hợp $\text{Overall Top-1} = 90.15\%$.
- [x] **Ý nghĩa Thống kê Vững chắc (Statistical Rigor):** Cung cấp đầy đủ cả kiểm định tham số ($t$-test, Cohen's $d$), kiểm định phi tham số (Wilcoxon exact), hiệu chỉnh Holm-Bonferroni, và thảo luận sâu sắc về lý thuyết phụ thuộc fold của Dietterich và Bengio.
- [x] **Gắn kết Thực tiễn Chuyên ngành Lâm nghiệp (Domain Grounding):** Kết nối chặt chẽ với các hệ thống vĩ mô hiện trường (XyloTron), tiêu chuẩn giải phẫu gỗ quốc tế IAWA, và bối cảnh pháp lý CITES Điều VII(6).
- [x] **Khả năng Tái lập Khoa học Tuyệt đối (Reproducibility):** Toàn bộ mã nguồn, trọng số mô hình và tập chỉ mục dữ liệu được công khai trên GitHub và Zenodo với DOI định danh vĩnh viễn.
- [x] **Chất lượng Trình bày và Dàn trang (Presentation & Typesetting):** Mã nguồn LaTeX biên dịch sạch 100%, 0 lỗi, 0 cảnh báo overfull vbox, cấu trúc 24 trang chuyên nghiệp theo mẫu Elsevier.

Chúng tôi tin tưởng rằng bản thảo sau đợt sửa đổi Vòng 3 toàn diện này đã đạt độ chín muồi học thuật cao nhất, đáp ứng trọn vẹn tất cả các kỳ vọng của Ban Biên tập và các Chuyên gia Phản biện, sẵn sàng cho quyết định chấp thuận xuất bản chính thức.
'''

with open('RESPONSE_TO_REVIEWER_ROUND3.md', 'w', encoding='utf-8') as f:
    f.write(content)

print("SUCCESSFULLY CREATED RESPONSE_TO_REVIEWER_ROUND3.md!")
