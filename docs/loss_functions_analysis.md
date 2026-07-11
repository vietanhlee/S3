# Hướng Dẫn Chuyên Sâu: Toán Học Và Khoa Học Của Các Hàm Loss Trong Metric Learning

Tài liệu này trình bày chi tiết về bản chất toán học, cơ sở khoa học và cơ chế hoạt động của **14 hàm loss** được lựa chọn cho bài toán nhận diện bề mặt gỗ (Wood Surface Identification) trên bộ dữ liệu S3 — bao gồm 19 loài thuộc 6 chi (*Dalbergia*, *Pterocarpus*, *Afzelia*, *Guibourtia*, *Sindora*, *Peltogyne*).

---

## Danh Sách Ký Hiệu Chung
*   $x_i \in \mathbb{R}^D$: Vector đặc trưng (feature/embedding) của mẫu thứ $i$ đã chuẩn hóa L2 ($\|x_i\|_2 = 1$).
*   $y_i$: Nhãn lớp (class label) của mẫu $x_i$.
*   $d(x_i, x_j) = \|x_i - x_j\|_2$: Khoảng cách Euclid giữa hai vector đặc trưng.
*   $S_{i,j} = x_i^T x_j$: Độ tương đồng Cosine giữa hai vector đặc trưng.
*   $[x]_+ = \max(0, x)$: Hàm ReLU / Bản lề (hinge function).
*   $\alpha, \beta, \lambda, m, \gamma, \tau, s$: Các siêu tham số điều chỉnh.

---

## I. Nhóm Học Dựa Trên Bộ Ba / Bộ Nhiều Mẫu (Triplet & Multi-Tuple Based)

### 1. Vanilla Triplet Loss (Baseline)

#### A. Công thức Toán học
Vanilla Triplet Loss tính toán trung bình trên **tất cả** các bộ ba $(a, p, n)$ hợp lệ trong batch (All Triplet Mining):

$$\mathcal{L}_{Vanilla-Triplet} = \frac{1}{|T|} \sum_{(a,p,n) \in T} \left[ d(a, p)^2 - d(a, n)^2 + \alpha \right]_+$$

Trong đó $T$ là tập hợp tất cả các bộ ba có loss > 0 trong batch, $\alpha$ là margin cố định.

#### B. Cơ sở Khoa học
*   **Lấy mẫu ngẫu nhiên:** Không sử dụng bất kỳ thuật toán mining nào. Tất cả các bộ ba hợp lệ (anchor cùng lớp với positive, khác lớp với negative) đều được tính loss.
*   **Hạn chế:** Phần lớn các bộ ba ngẫu nhiên là "trivial" (tức $d(a,n) \gg d(a,p) + \alpha$), dẫn đến loss = 0 và gradient bị triệt tiêu. Mô hình hội tụ chậm và lãng phí tài nguyên tính toán.
*   **Vai trò trong nghiên cứu:** Là baseline cơ bản nhất để đánh giá mức cải thiện thực tế của các thuật toán mining và loss nâng cao.

---

### 2. Semi-hard Triplet Loss (CVPR 2015)

#### A. Công thức Toán học
Triplet Loss chuẩn hóa mối quan hệ giữa ba mẫu: **Anchor** ($a$), **Positive** ($p$) và **Negative** ($n$).

$$\mathcal{L}_{Triplet} = \sum_{i=1}^{N} \left[ d(a_i, p_i)^2 - d(a_i, n_i)^2 + \alpha \right]_+$$

#### B. Cơ chế Khai thác Mẫu Semi-hard
**Semi-hard Mining** lựa chọn mẫu Negative $n_i$ cho Anchor $a_i$ và Positive $p_i$ thỏa mãn điều kiện:

$$d(a_i, p_i)^2 < d(a_i, n_i)^2 < d(a_i, p_i)^2 + \alpha$$

#### C. Cơ sở Khoa học
*   **Vùng biên Margin:** Negative nằm xa Anchor hơn Positive một chút nhưng vẫn nằm trong phạm vi margin.
*   **Tác động:** Tạo ra gradient có độ lớn vừa phải, giúp mô hình hội tụ ổn định và học được các đặc trưng tinh vi ở vùng biên phân tách lớp mà không bị nhiễu bởi các mẫu cực đoan (outliers).

---

### 3. Soft-Margin Triplet Loss

#### A. Công thức Toán học
Thay thế hàm hinge $[\cdot]_+$ cứng nhắc bằng một hàm mũ mịn:

$$\mathcal{L}_{Soft-Triplet} = \frac{1}{N} \sum_{i=1}^{N} \log \left( 1 + \exp\left( d(a_i, p_i)^2 - d(a_i, n_i)^2 \right) \right)$$

#### B. Cơ sở Khoa học
*   **Gradient liên tục:** Triplet Loss gốc triệt tiêu hoàn toàn gradient khi khoảng cách đạt ngưỡng an toàn ($loss = 0$). Hàm Soft-Margin loại bỏ tham số margin $\alpha$, liên tục duy trì một lượng gradient nhỏ để kéo các cặp Positive sát vào nhau và đẩy Negative ra xa vô cực.
*   **Tránh bão hòa:** Giảm sự nhạy cảm của mô hình đối với giá trị margin cố định, giúp thích ứng tốt hơn với cấu trúc dữ liệu thực tế.

---

### 4. Angular Loss (ICCV 2017)

#### A. Công thức Toán học
Thay vì tối ưu hóa khoảng cách Euclid tuyệt đối, Angular Loss tối ưu hóa cấu trúc hình học dựa trên góc tam giác tạo bởi triplet $(a, p, n)$:

$$\mathcal{L}_{Angular} = \frac{1}{|T|} \sum_{(a,p,n) \in T} \log \left( 1 + \exp\left( 4 \tan^2\alpha \cdot (x_a + x_p)^T x_n - 2 (1 + \tan^2\alpha) \cdot x_a^T x_p \right) \right)$$

Trong đó $\alpha$ là giới hạn góc tại mẫu anchor.

#### B. Cơ sở Khoa học
*   **Bất biến tỷ lệ (Scale Invariance):** Khoảng cách Euclid thay đổi khi ta co giãn (scale) các vector đặc trưng. Góc $\theta$ có tính chất bất biến tỷ lệ.
*   **Tác động hình học:** Ép góc $\angle pan$ tại đỉnh Negative lớn hơn một ngưỡng xác định, giúp định hình biên giới phân lớp bền vững trước các thay đổi về độ sáng hoặc cường độ đặc trưng.
*   **Phù hợp cho ảnh macro gỗ:** Ảnh macro chụp ở các mức zoom khác nhau sẽ thay đổi cường độ đặc trưng nhưng không thay đổi cấu trúc góc giữa các embedding. Angular Loss bảo toàn mối quan hệ hình học này.

---

### 5. Multi-Similarity (MS) Loss (CVPR 2019)

#### A. Công thức Toán học
$$\mathcal{L}_{MS} = \frac{1}{N} \sum_{i=1}^{N} \left\{ \frac{1}{\alpha} \log \left[ 1 + \sum_{k \in P_i} \exp(-\alpha (S_{i,k} - \lambda)) \right] + \frac{1}{\beta} \log \left[ 1 + \sum_{l \in N_i} \exp(\beta (S_{i,l} - \lambda)) \right] \right\}$$

#### B. Cơ sở Khoa học
*   **Trọng số động:** Đạo hàm của $\mathcal{L}_{MS}$ đối với độ tương đồng Cosine $S_{i,k}$ tỷ lệ thuận với $\frac{\exp(-\alpha S_{i,k})}{\sum \exp(-\alpha S_{i,p})}$. Cặp nào càng khó thì nhận lượng cập nhật gradient càng lớn.
*   **Lọc mẫu thông minh:** Chỉ tối ưu hóa các mẫu Positive nằm dưới mức tối đa của Negative và ngược lại.
*   **Phù hợp cho ảnh macro gỗ:** Các mẫu gỗ hai loài khác chi nhưng có vân giống nhau sẽ tự động nhận gradient lớn hơn, thúc đẩy mô hình tập trung giải quyết các ranh giới khó nhất.

---

## II. Nhóm Học Dựa Trên Cặp Mẫu (Pair-Based / Contrastive Loss)

### 1. Vanilla Contrastive Loss (Baseline — CVPR 2005)

#### A. Công thức Toán học

$$\mathcal{L}_{Contrastive} = (1 - y_{i,j}) \frac{1}{2} d(x_i, x_j)^2 + y_{i,j} \frac{1}{2} \left[ m - d(x_i, x_j) \right]_+^2$$

Trong đó $y_{i,j} = 0$ nếu $x_i$ và $x_j$ cùng lớp (cặp Positive), $y_{i,j} = 1$ nếu khác lớp (cặp Negative), và $m$ là margin khoảng cách cố định.

#### B. Cơ sở Khoa học
*   **Kéo-đẩy trực tiếp:** Ép khoảng cách giữa các cặp Positive về 0 ($d \to 0$) và đẩy các cặp Negative vượt qua ngưỡng margin ($d \geq m$).
*   **Hạn chế:** Margin $m$ cố định cho toàn bộ các lớp gỗ là không linh hoạt — có loài dễ phân biệt (khác chi) và loài rất khó phân biệt (cùng chi). Ép positive về hẳn 0 dễ gây overfitting.
*   **Vai trò trong nghiên cứu:** Là baseline pair-based cơ bản nhất để so sánh với các hàm loss pair-based nâng cao (Circle Loss, Multi-Similarity Loss).

---

### 2. Online Hard Pair Contrastive Loss

#### A. Công thức Toán học
Sử dụng cùng công thức Contrastive Loss, nhưng **chỉ tính loss cho các cặp khó nhất** trong batch:

*   **Hardest Positive:** Cặp Positive có khoảng cách lớn nhất: $p^* = \arg\max_{p \in P_i} d(x_i, x_p)$
*   **Hardest Negative:** Cặp Negative có khoảng cách nhỏ nhất: $n^* = \arg\min_{n \in N_i} d(x_i, x_n)$

$$\mathcal{L}_{HardContrastive} = \frac{1}{2} d(x_i, x_{p^*})^2 + \frac{1}{2} \left[ m - d(x_i, x_{n^*}) \right]_+^2$$

#### B. Cơ sở Khoa học
*   **Tập trung tài nguyên:** Thay vì tính loss cho hàng nghìn cặp dễ (gradient ≈ 0), chỉ tối ưu các cặp nằm tại ranh giới phân lớp.
*   **Hạn chế:** Rất nhạy cảm với nhãn bị gán sai hoặc ảnh bị nhiễu, vì mô hình liên tục tập trung vào các mẫu lỗi cực đoan.

---

### 3. Lifted Structured Loss (CVPR 2016)

#### A. Công thức Toán học
$$\mathcal{L}_{Lifted} = \frac{1}{2|P|} \sum_{(i,j) \in P} \left[ d(x_i, x_j) + \log \left( \sum_{k: y_k \neq y_i} \exp( \alpha - d(x_i, x_k) ) + \sum_{k: y_k \neq y_j} \exp( \alpha - d(x_j, x_k) ) \right) \right]_+^2$$

#### B. Cơ sở Khoa học
*   **Xấp xỉ trơn của cực đại:** Hàm $\log\left( \sum \exp(\alpha - d) \right)$ hoạt động như một hàm xấp xỉ trơn của hàm $\max_k (\alpha - d(x_i, x_k))$.
*   **Tập trung lực đẩy:** Tự động dồn lực đẩy vào các mẫu Negative có khoảng cách gần nhất (khó nhất), đẩy nhanh sự hội tụ.
*   **Tối ưu hóa toàn batch:** Xem xét cấu trúc khoảng cách của toàn bộ batch thay vì từng cặp riêng lẻ.

---

### 4. Circle Loss — Chế độ Cặp (Pair-based CVPR 2020)

#### A. Công thức Toán học
$$\mathcal{L}_{Circle-Pair} = \log \left[ 1 + \sum_{j \in N_i} \exp(\gamma \alpha_j^n (s_j^n - \Delta_n)) \sum_{k \in P_i} \exp(-\gamma \alpha_k^p (s_k^p - \Delta_p)) \right]$$

Với các hệ số điều chỉnh gradient tự thích ứng:

$$\alpha_k^p = [O_p - s_k^p]_+, \quad \alpha_j^n = [s_j^n - O_n]_+$$

#### B. Cơ sở Khoa học
*   **Tối ưu hóa không đối xứng:** Positive đã đủ gần sẽ bị giảm lực kéo ($\alpha_k^p \to 0$), tránh overfitting. Negative ở quá gần sẽ bị tăng lực đẩy ($\alpha_j^n$ cực đại).
*   **Biên hình tròn:** Tạo ra không gian phân lớp có cấu trúc trơn tru và gom cụm tối ưu.
*   **Phù hợp cho ảnh macro gỗ:** Cơ chế tự thích ứng giúp tránh overfitting trên các mẫu gỗ dễ nhận diện (khác chi) trong khi tập trung mạnh vào các mẫu cùng chi khó phân biệt.

---

## III. Nhóm Học Phân Loại Margin Góc, Proxy-based Và Học Tương Phản Lớp

### 1. Proxy Anchor Loss (CVPR 2020)

#### A. Công thức Toán học
$$\mathcal{L}_{PA} = \frac{1}{|P^+|} \sum_{p \in P^+} \log \left[ 1 + \sum_{x \in X_p^+} \exp(-\alpha(s(x, p) - \delta)) \right] + \frac{1}{|P|} \sum_{p \in P} \log \left[ 1 + \sum_{x \in X_p^-} \exp(\alpha(s(x, p) + \delta)) \right]$$

#### B. Cơ sở Khoa học
*   **Proxy làm Anchor:** Thay vì so sánh ảnh với ảnh ($O(N^3)$), thuật toán so sánh ảnh với các Proxy lớp cố định ($O(N \times C)$). Proxy Anchor thu hút mẫu cùng lớp và đẩy lùi mẫu khác lớp cực kỳ nhanh chóng.
*   **Phù hợp cho ảnh macro gỗ:** Với chỉ 18 class, mỗi class chỉ cần 1 proxy → chi phí tính toán cực thấp, tốc độ hội tụ nhanh, gradient ổn định vì proxy đóng vai trò neo cố định cho mỗi loài.

---

### 2. ArcFace (Additive Angular Margin Loss — CVPR 2019)

#### A. Công thức Toán học
$$\mathcal{L}_{ArcFace} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp\left( s \cdot \cos(\theta_{y_i} + m) \right)}{\exp\left( s \cdot \cos(\theta_{y_i} + m) \right) + \sum_{j \neq y_i} \exp\left( s \cdot \cos\theta_j \right)}$$

#### B. Cơ sở Khoa học
*   **Biên độ góc Geodesic:** Việc cộng trực tiếp góc biên độ $m$ trong hàm $\cos(\theta + m)$ tương đương với việc ép buộc các đặc trưng cùng lớp phải nằm trong một hình nón góc cực hẹp bao quanh trọng số lớp $W_j$.
*   **Tính phân tách cực cao:** ArcFace tối đa hóa khoảng cách góc giữa các lớp khác nhau trên mặt cầu đơn vị, tạo ra không gian biểu diễn có tính phân tách đặc trưng (discriminative power) mạnh mẽ nhất trong các dòng Softmax cải tiến.
*   **Phù hợp cho ảnh macro gỗ:** SOTA cho fine-grained recognition khi inter-class similarity cao (các loài *Dalbergia* vs *Pterocarpus* có vân gỗ rất giống nhau). Biên góc trên mặt cầu tạo ra các cụm hình nón cực kỳ phân tách.

---

### 3. SubCenter ArcFace (ECCV 2020)

#### A. Công thức Toán học
Mở rộng ArcFace bằng cách gán $K$ sub-center cho mỗi lớp. Với mỗi mẫu $x_i$ thuộc lớp $y_i$, chọn sub-center gần nhất:

$$\cos\theta_{y_i} = \max_{k=1}^{K} \frac{x_i^T W_{y_i}^{(k)}}{\|x_i\|_2 \|W_{y_i}^{(k)}\|_2}$$

Sau đó áp dụng angular margin giống ArcFace:

$$\mathcal{L}_{SubCenter} = -\log \frac{\exp\left( s \cdot \cos(\theta_{y_i} + m) \right)}{\exp\left( s \cdot \cos(\theta_{y_i} + m) \right) + \sum_{j \neq y_i} \exp\left( s \cdot \cos\theta_j \right)}$$

#### B. Cơ sở Khoa học
*   **Phân bố đa đỉnh (Multi-modal):** Thay vì ép tất cả mẫu cùng lớp vào một center duy nhất, SubCenter cho phép $K$ center đại diện cho $K$ "mode" phân bố khác nhau. Ví dụ: cùng loài *Dalbergia cochinchinensis* nhưng ảnh macro từ gỗ lõi vs gỗ giác sẽ có texture khác biệt rõ rệt.
*   **Kháng nhiễu tự nhiên:** Các mẫu "outlier" sẽ tự động bị gán vào sub-center riêng biệt, không làm méo center chính của lớp.
*   **Phù hợp cho ảnh macro gỗ:** Giải quyết trực tiếp vấn đề intra-class variation lớn — cùng một loài gỗ, vân gỗ thay đổi rõ rệt theo vị trí cắt, độ tuổi cây, và điều kiện sinh trưởng. Mỗi sub-center tự động học một "mode" phân bố vân gỗ riêng.

---

### 4. SoftTriple Loss (ICCV 2019)

#### A. Công thức Toán học
SoftTriple sử dụng $K$ center có thể học cho mỗi lớp và tính xác suất thuộc lớp $c$ qua phép tổng hợp mềm (soft assignment):

$$S_{i,c} = \sum_{k=1}^{K} \frac{\exp\left( \frac{1}{\gamma} x_i^T w_c^{(k)} \right)}{\sum_{t=1}^{K} \exp\left( \frac{1}{\gamma} x_i^T w_c^{(t)} \right)} \cdot x_i^T w_c^{(k)}$$

Loss cuối cùng áp dụng normalized softmax với margin:

$$\mathcal{L}_{SoftTriple} = -\log \frac{\exp\left( \lambda (S_{i,y_i} - \delta) \right)}{\exp\left( \lambda (S_{i,y_i} - \delta) \right) + \sum_{j \neq y_i} \exp\left( \lambda S_{i,j} \right)}$$

#### B. Cơ sở Khoa học
*   **Thống nhất Proxy và Triplet:** SoftTriple chứng minh toán học rằng việc tối ưu hóa normalized softmax trên nhiều proxy (center) cho mỗi lớp tương đương với việc tối ưu hóa smooth triplet loss trên các center đó.
*   **Soft Assignment:** Thay vì hard assignment (chỉ chọn center gần nhất như SubCenter ArcFace), SoftTriple sử dụng trọng số mềm để kết hợp tất cả $K$ center theo mức độ tương đồng. Điều này giúp gradient lan truyền đều đến tất cả các center, tránh hiện tượng "center chết".
*   **Phù hợp cho ảnh macro gỗ:** Tự động phát hiện và mô hình hóa các sub-cluster (ví dụ: vùng gỗ lõi, vùng gỗ giác, vùng chuyển tiếp) trong cùng một loài gỗ mà không cần biết trước số lượng mode phân bố.

---

### 5. Supervised Contrastive Loss (SupCon — NeurIPS 2020)

#### A. Công thức Toán học
$$\mathcal{L}_{SupCon} = \sum_{i \in I} \frac{-1}{|P(i)|} \sum_{p \in P(i)} \log \frac{\exp\left( \frac{z_i^T z_p}{\tau} \right)}{\sum_{a \in A(i)} \exp\left( \frac{z_i^T z_a}{\tau} \right)}$$

#### B. Cơ sở Khoa học
*   **Đa cực tích cực (Multi-positive):** Mở rộng SimCLR sang môi trường giám sát, kéo tất cả các mẫu cùng lớp trong batch lại gần nhau và đẩy tất cả các mẫu khác lớp ra xa.
*   **Căn chỉnh toàn batch (Batch-wide Alignment):** Xem xét mối tương quan toàn diện giữa tất cả $N$ mẫu trong batch thay vì chỉ xét từng cặp/bộ ba riêng lẻ.
*   **Phù hợp cho ảnh macro gỗ:** Tạo biểu diễn tổng quát (generalized representation) mạnh nhất, phù hợp khi cần trích xuất đặc trưng chuyển giao hoặc khi batch chứa nhiều loài cùng chi.

---

## IV. Bảng So Sánh Toàn Diện 14 Hàm Loss Được Chọn

| # | Tên Hàm Loss | Paper | Độ Phức Tạp Batch | Ưu Điểm Nổi Bật | Phù Hợp Cho Gỗ S3 |
| :---: | :--- | :--- | :--- | :--- | :--- |
| 1 | **Vanilla Triplet** | — | $O(N^3)$ | Baseline đơn giản nhất, dễ triển khai. | ⭐ Baseline so sánh |
| 2 | **Semi-hard Triplet** | CVPR 2015 | $O(N^3)$ | Gradient ổn định, biên rõ ràng. | ⭐⭐ Baseline kinh điển |
| 3 | **Soft-Margin Triplet** | 2017 | $O(N^3)$ | Không cần chỉnh margin, gradient liên tục. | ⭐⭐ Cầu nối Triplet → SOTA |
| 4 | **Angular** | ICCV 2017 | $O(N^3)$ | Bất biến tỷ lệ đặc trưng (Scale-invariant). | ⭐⭐⭐ Zoom đa mức ảnh macro |
| 5 | **Multi-Similarity** | CVPR 2019 | $O(N^2)$ | Trọng số động tinh vi, SOTA pair-based. | ⭐⭐⭐⭐ Phân biệt loài cùng chi |
| 6 | **Vanilla Contrastive** | CVPR 2005 | $O(N^2)$ | Baseline pair-based đơn giản nhất. | ⭐ Baseline so sánh |
| 7 | **Hard Contrastive** | 2017 | $O(N^2)$ | Tập trung vào cặp khó nhất. | ⭐⭐ Baseline hard mining |
| 8 | **Lifted Structured** | CVPR 2016 | $O(N^2)$ | Tận dụng cấu trúc toàn batch, hội tụ nhanh. | ⭐⭐⭐ Nền tảng lý thuyết |
| 9 | **Circle Loss** | CVPR 2020 | $O(N^2)$ | Tối ưu hóa không đối xứng, adaptive margin. | ⭐⭐⭐⭐ Fine-grained, Re-ID |
| 10 | **Proxy Anchor** | CVPR 2020 | $O(N \times C)$ | Hội tụ siêu nhanh, chi phí tuyến tính. | ⭐⭐⭐⭐ 18 class nhỏ → cực hiệu quả |
| 11 | **ArcFace** | CVPR 2019 | $O(N \times C)$ | Gom cụm cực chặt, tối ưu phân tách góc. | ⭐⭐⭐⭐⭐ SOTA fine-grained |
| 12 | **SubCenter ArcFace** | ECCV 2020 | $O(N \times C \times K)$ | ArcFace + multi-modal intra-class. | ⭐⭐⭐⭐⭐ Vân gỗ đa dạng trong loài |
| 13 | **SoftTriple** | ICCV 2019 | $O(N \times C \times K)$ | Nhiều center/lớp, soft assignment. | ⭐⭐⭐⭐⭐ Sub-cluster vân gỗ tự động |
| 14 | **Supervised Contrastive** | NeurIPS 2020 | $O(N^2)$ | Batch-wide multi-positive, tổng quát hóa cao. | ⭐⭐⭐⭐ Representation learning |
## V. Phân Loại Tier Và Các Tệp Huấn Luyện Tương Ứng

| Nhóm (Tier) | # | Tên Tệp Huấn Luyện | Hàm Loss Được Áp Dụng |
| :---: | :---: | :--- | :--- |
| **Tier 1 (SOTA)** | 1 | [train_arcface.py](file:///g:/S3_paper/train_arcface.py) | ArcFace |
| | 2 | [train_subcenter_arcface.py](file:///g:/S3_paper/train_subcenter_arcface.py) | SubCenter ArcFace |
| | 3 | [train_multi_similarity.py](file:///g:/S3_paper/train_multi_similarity.py) | Multi-Similarity |
| | 4 | [train_circle.py](file:///g:/S3_paper/train_circle.py) | Circle Loss |
| | 5 | [train_supcon.py](file:///g:/S3_paper/train_supcon.py) | SupCon |
| **Tier 2 (Strong)** | 6 | [train_proxy_anchor.py](file:///g:/S3_paper/train_proxy_anchor.py) | Proxy Anchor |
| | 7 | [train_soft_triple.py](file:///g:/S3_paper/train_soft_triple.py) | SoftTriple |
| | 8 | [train_angular.py](file:///g:/S3_paper/train_angular.py) | Angular Loss |
| | 9 | [train_lifted_structured.py](file:///g:/S3_paper/train_lifted_structured.py) | Lifted Structured |
| **Tier 3 (Baseline)** | 10 | [train_triplet.py](file:///g:/S3_paper/train_triplet.py) | Vanilla Triplet Loss ✅ |
| | 11 | [train_semi_hard_triplet.py](file:///g:/S3_paper/train_semi_hard_triplet.py) | Semi-hard Triplet |
| | 12 | [train_soft_margin_triplet.py](file:///g:/S3_paper/train_soft_margin_triplet.py) | Soft-Margin Triplet |
| | 13 | [train_contrastive.py](file:///g:/S3_paper/train_contrastive.py) | Vanilla Contrastive Loss ✅ |
| | 14 | [train_hard_contrastive.py](file:///g:/S3_paper/train_hard_contrastive.py) | Online Hard Pair Contrastive |
| **Self-Supervised** | 15 | [train_barlow_twins.py](file:///g:/S3_paper/train_barlow_twins.py) | Barlow Twins |
| | 16 | [train_simclr.py](file:///g:/S3_paper/train_simclr.py) | SimCLR |
| | 17 | [train_simsiam.py](file:///g:/S3_paper/train_simsiam.py) | SimSiam |
| | 18 | [train_byol.py](file:///g:/S3_paper/train_byol.py) | BYOL |