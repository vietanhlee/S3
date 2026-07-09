# Hướng Dẫn Chuyên Sâu: Toán Học Và Khoa Học Của Các Hàm Loss Trong Metric Learning

Tài liệu này trình bày chi tiết về bản chất toán học, cơ sở khoa học và cơ chế hoạt động của các hàm loss nâng cao dùng trong học đặc trưng sâu (Deep Metric Learning / Representation Learning).

---

## Danh Sách Ký Hiệu Chung
*   $x_i \in \mathbb{R}^D$: Vector đặc trưng (feature/embedding) của mẫu thứ $i$ đã chuẩn hóa L2 ($\|x_i\|_2 = 1$).
*   $y_i$: Nhãn lớp (class label) của mẫu $x_i$.
*   $d(x_i, x_j) = \|x_i - x_j\|_2$: Khoảng cách Euclid giữa hai vector đặc trưng.
*   $S_{i,j} = x_i^T x_j$: Độ tương đồng Cosine giữa hai vector đặc trưng.
*   $[x]_+ = \max(0, x)$: Hàm ReLU / Bản lề (hinge function).
*   $\alpha, \beta, \lambda, m, \gamma, \tau$: Các siêu tham số điều chỉnh.

---

## I. Nhóm Học Dựa Trên Bộ Ba / Bộ Nhiều Mẫu (Triplet & Multi-Tuple Based)

### 1. Semi-hard Triplet Loss (CVPR 2015)

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

### 2. Soft-Margin Triplet Loss

#### A. Công thức Toán học
Thay thế hàm hinge $[\cdot]_+$ cứng nhắc bằng một hàm mũ mịn:

$$\mathcal{L}_{Soft-Triplet} = \frac{1}{N} \sum_{i=1}^{N} \log \left( 1 + \exp\left( d(a_i, p_i)^2 - d(a_i, n_i)^2 \right) \right)$$

#### B. Cơ sở Khoa học
*   **Gradient liên tục:** Triplet Loss gốc triệt tiêu hoàn toàn gradient khi khoảng cách đạt ngưỡng an toàn ($loss = 0$). Hàm Soft-Margin loại bỏ tham số margin $\alpha$, liên tục duy trì một lượng gradient nhỏ để kéo các cặp Positive sát vào nhau và đẩy Negative ra xa vô cực.
*   **Tránh bão hòa:** Giảm sự nhạy cảm của mô hình đối với giá trị margin cố định, giúp thích ứng tốt hơn với cấu trúc dữ liệu thực tế.

---

### 3. Quadruplet Loss (CVPR 2017)

#### A. Công thức Toán học
Huấn luyện mô hình dựa trên bộ bốn mẫu (Quadruplet) $(a, p, n_1, n_2)$ với $y_{n_2} \neq y_{a}$ và $y_{n_2} \neq y_{n_1}$:

$$\mathcal{L}_{Quad} = \sum_{i=1}^{N} \left( \left[ d(a_i, p_i)^2 - d(a_i, n_{i,1})^2 + \alpha_1 \right]_+ + \left[ d(a_i, p_i)^2 - d(n_{i,1}, n_{i,2})^2 + \alpha_2 \right]_+ \right)$$

Trong đó $\alpha_1, \alpha_2$ là hai margin khoảng cách riêng biệt.

#### B. Cơ sở Khoa học
*   **Ràng buộc toàn cục:** Triplet Loss chỉ ép khoảng cách nội lớp nhỏ hơn khoảng cách ngoại lớp *khi so với cùng một mẫu Anchor*. Ràng buộc thứ hai của Quadruplet Loss ($d(a, p) < d(n_1, n_2)$) ép khoảng cách giữa cặp Positive bất kỳ phải nhỏ hơn khoảng cách giữa hai mẫu Negative khác lớp bất kỳ.
*   **Tác động:** Giúp cấu trúc không gian đặc trưng đồng đều và phân tách toàn cục rõ ràng hơn.

---

### 4. Histogram Loss (NeurIPS 2016)

#### A. Công thức Toán học
Histogram Loss tối ưu hóa trực tiếp xác suất phân phối khoảng cách của các cặp Positive và Negative mà không cần thực hiện mining mẫu:

$$\mathcal{L}_{Hist} = \int_{0}^{2} p^-(x) \left( \int_{0}^{x} p^+(y) dy \right) dx$$

Trong đó $p^+(x)$ và $p^-(x)$ là hàm mật độ xác suất phân phối khoảng cách của các cặp Positive và Negative. Trong thực tế, các phân phối này được xấp xỉ bằng Histogram rời rạc thông qua phép nội suy tuyến tính một chiều.

#### B. Cơ sở Khoa học
*   **Tối ưu hóa AUC:** Hàm loss tối ưu trực tiếp diện tích dưới đường cong ROC (AUC), tức là cực đại hóa xác suất để khoảng cách của một cặp Positive ngẫu nhiên nhỏ hơn khoảng cách của một cặp Negative ngẫu nhiên.
*   **Không cần Mining:** Giải quyết triệt để sự phụ thuộc vào các thuật toán lấy mẫu phức tạp.

---

### 5. Angular Loss (ICCV 2017)

#### A. Công thức Toán học
Thay vì tối ưu hóa khoảng cách Euclid tuyệt đối, Angular Loss tối ưu hóa cấu trúc hình học dựa trên góc tam giác tạo bởi triplet $(a, p, n)$:

$$\mathcal{L}_{Angular} = \frac{1}{|T|} \sum_{(a,p,n) \in T} \log \left( 1 + \exp\left( 4 \tan^2\alpha \cdot (x_a + x_p)^T x_n - 2 (1 + \tan^2\alpha) \cdot x_a^T x_p \right) \right)$$

Trong đó $\alpha$ là giới hạn góc tại mẫu anchor.

#### B. Cơ sở Khoa học
*   **Bất biến tỷ lệ (Scale Invariance):** Khoảng cách Euclid thay đổi khi ta co giãn (scale) các vector đặc trưng. Góc $\theta$ có tính chất bất biến tỷ lệ.
*   **Tác động hình học:** Ép góc $\angle pan$ tại đỉnh Negative lớn hơn một ngưỡng xác định, giúp định hình biên giới phân lớp bền vững trước các thay đổi về độ sáng hoặc cường độ đặc trưng.

---

## II. Nhóm Học Dựa Trên Cặp Mẫu (Pair-Based / Contrastive Loss Cải Tiến)

Lấy **Contrastive Loss gốc (CVPR 2005)** làm quy chiếu:

$$\mathcal{L}_{Contrastive} = (1 - y_{i,j}) \frac{1}{2} d(x_i, x_j)^2 + y_{i,j} \frac{1}{2} \left[ m - d(x_i, x_j) \right]_+^2$$

---

### 1. Lifted Structured Loss (CVPR 2016)

#### A. Công thức Toán học
$$\mathcal{L}_{Lifted} = \frac{1}{2|P|} \sum_{(i,j) \in P} \left[ d(x_i, x_j) + \log \left( \sum_{k: y_k \neq y_i} \exp( \alpha - d(x_i, x_k) ) + \sum_{k: y_k \neq y_j} \exp( \alpha - d(x_j, x_k) ) \right) \right]_+^2$$

#### B. Cơ sở Khoa học
*   **Xấp xỉ trơn của cực đại:** Hàm $\log\left( \sum \exp(\alpha - d) \right)$ hoạt động như một hàm xấp xỉ trơn của hàm $\max_k (\alpha - d(x_i, x_k))$.
*   **Tập trung lực đẩy:** Tự động dồn lực đẩy vào các mẫu Negative có khoảng cách gần nhất (khó nhất), đẩy nhanh sự hội tụ.

---

### 2. Multi-Similarity (MS) Loss (CVPR 2019)

#### A. Công thức Toán học
$$\mathcal{L}_{MS} = \frac{1}{N} \sum_{i=1}^{N} \left\{ \frac{1}{\alpha} \log \left[ 1 + \sum_{k \in P_i} \exp(-\alpha (S_{i,k} - \lambda)) \right] + \frac{1}{\beta} \log \left[ 1 + \sum_{l \in N_i} \exp(\beta (S_{i,l} - \lambda)) \right] \right\}$$

#### B. Cơ sở Khoa học
*   **Trọng số động:** Đạo hàm của $\mathcal{L}_{MS}$ đối với độ tương đồng Cosine $S_{i,k}$ tỷ lệ thuận với $\frac{\exp(-\alpha S_{i,k})}{\sum \exp(-\alpha S_{i,p})}$. Cặp nào càng khó thì nhận lượng cập nhật gradient càng lớn.
*   **Lọc mẫu thông minh:** Chỉ tối ưu hóa các mẫu Positive nằm dưới mức tối đa của Negative và ngược lại.

---

### 3. Circle Loss - Chế độ Cặp (Pair-based CVPR 2020)

#### A. Công thức Toán học
$$\mathcal{L}_{Circle-Pair} = \log \left[ 1 + \sum_{j \in N_i} \exp(\gamma \alpha_j^n (s_j^n - \Delta_n)) \sum_{k \in P_i} \exp(-\gamma \alpha_k^p (s_k^p - \Delta_p)) \right]$$

Với các hệ số điều chỉnh gradient tự thích ứng:

$$\alpha_k^p = [O_p - s_k^p]_+, \quad \alpha_j^n = [s_j^n - O_n]_+$$

#### B. Cơ sở Khoa học
*   **Tối ưu hóa không đối xứng:** Positive đã đủ gần sẽ bị giảm lực kéo ($\alpha_k^p \to 0$), tránh overfitting. Negative ở quá gần sẽ bị tăng lực đẩy ($\alpha_j^n$ cực đại).
*   **Biên hình tròn:** Tạo ra không gian phân lớp có cấu trúc trơn tru và gom cụm tối ưu.

---

### 4. Ranked List Loss (RLL - CVPR 2019)

#### A. Công thức Toán học
$$\mathcal{L}_{RLL} = \frac{1}{N} \sum_{i=1}^{N} \left( \frac{1}{|P_i^*|} \sum_{p \in P_i^*} w_{i,p} [d(x_i, x_p) - \alpha]_+^2 + \frac{1}{|N_i^*|} \sum_{n \in N_i^*} w_{i,n} [\alpha - m - d(x_i, x_n)]_+^2 \right)$$

Trong đó $P_i^*$ và $N_i^*$ là tập hợp các Positive và Negative khó vượt qua ngưỡng khoảng cách $\alpha$ và $\alpha - m$.

#### B. Cơ sở Khoa học
*   **Dải bảo vệ (Margin Band):** Thiết lập một vùng đệm an toàn có độ rộng $m$ ở giữa các lớp. Bỏ qua hoàn toàn các mẫu đã được tối ưu hóa tốt để bảo toàn cấu trúc phân bố đặc trưng đã học.

---

### 5. Binomial Deviance Loss

#### A. Công thức Toán học
$$\mathcal{L}_{Binomial} = \sum_{(i,j) \in P} \log\left( 1 + \exp\left( -\alpha (S_{i,j} - \beta) \right) \right) + \sum_{(i,k) \in N} \log\left( 1 + \exp\left( \alpha (S_{i,k} - \beta) \right) \right)$$

#### B. Cơ sở Khoa học
*   **Gradient trơn:** Thay thế hàm cắt bản lề (hinge) bằng hàm log-entropy nhị thức giúp gradient luôn liên tục, mượt mà trên toàn dải khoảng cách.
*   **Kháng nhiễu:** Hàm logarit làm giảm độ dốc đối với các mẫu lỗi cực đoan (outliers), hạn chế hiện tượng bão hòa hoặc lệch không gian biểu diễn do nhãn dữ liệu bị gán sai.

---

### 6. Margin Loss với Distance-Weighted Sampling (ICCV 2017)

#### A. Công thức Toán học
$$\mathcal{L}_{Margin} = \sum_{(a,p)} [d(a,p) - \beta]_+^2 + \sum_{(a,n)} [\beta + \alpha - d(a,n)]_+^2$$

Trong đó $\beta$ là ranh giới phân tách lớp tự động học cùng mô hình.
Phương pháp kết hợp thuật toán **Distance-Weighted Sampling** để lấy mẫu Negative theo xác suất tỷ lệ nghịch với mật độ phân phối thể tích hình học trong không gian đa chiều $D$:

$$q(d) \propto d^{D-2} \left(1 - \frac{1}{4} d^2\right)^{\frac{D-3}{2}}$$

#### B. Cơ sở Khoa học
*   **Ranh giới động:** Thay vì cố định margin, mô hình tự học ranh giới tối ưu cho từng lớp ($\beta$).
*   **Khắc phục hiện tượng tập trung vùng biên:** Trong không gian nhiều chiều, lấy mẫu ngẫu nhiên sẽ chỉ chọn được các mẫu dễ (do phân phối thể tích tập trung ở biên ngoài). Lấy mẫu theo tỷ lệ thể tích $q(d)$ giúp tìm kiếm các mẫu Semi-hard chuẩn xác hơn.

---

## III. Nhóm Học Phân Loại Margin Góc Và Học Tương Phản

### 1. Proxy Anchor Loss (CVPR 2020)

#### A. Công thức Toán học
$$\mathcal{L}_{PA} = \frac{1}{|P^+|} \sum_{p \in P^+} \log \left[ 1 + \sum_{x \in X_p^+} \exp(-\alpha(s(x, p) - \delta)) \right] + \frac{1}{|P|} \sum_{p \in P} \log \left[ 1 + \sum_{x \in X_p^-} \exp(\alpha(s(x, p) + \delta)) \right]$$

#### B. Cơ sở Khoa học
*   **Proxy làm Anchor:** Thay vì so sánh ảnh với ảnh ($O(N^3)$), thuật toán so sánh ảnh với các Proxy lớp cố định ($O(N \times C)$). Proxy Anchor thu hút mẫu cùng lớp và đẩy lùi mẫu khác lớp cực kỳ nhanh chóng.

---

### 2. ArcFace (Additive Angular Margin Loss - CVPR 2019)

#### A. Công thức Toán học
$$\mathcal{L}_{ArcFace} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp\left( s \cdot \cos(\theta_{y_i} + m) \right)}{\exp\left( s \cdot \cos(\theta_{y_i} + m) \right) + \sum_{j \neq y_i} \exp\left( s \cdot \cos\theta_j \right)}$$

#### B. Cơ sở Khoa học
*   **Biên độ góc Geodesic:** Việc cộng trực tiếp góc biên độ $m$ trong hàm $\cos(\theta + m)$ tương đương với việc ép buộc các đặc trưng cùng lớp phải nằm trong một hình nón góc cực hẹp bao quanh trọng số lớp $W_j$.
*   **Tính phân tách cực cao:** ArcFace tối đa hóa khoảng cách góc giữa các lớp khác nhau trên mặt cầu đơn vị, tạo ra không gian biểu diễn có tính phân tách đặc trưng (discriminative power) mạnh mẽ nhất trong các dòng Softmax cải tiến.

---

### 3. Supervised Contrastive Loss (SupCon - NeurIPS 2020)

#### A. Công thức Toán học
$$\mathcal{L}_{SupCon} = \sum_{i \in I} \frac{-1}{|P(i)|} \sum_{p \in P(i)} \log \frac{\exp\left( \frac{z_i^T z_p}{\tau} \right)}{\sum_{a \in A(i)} \exp\left( \frac{z_i^T z_a}{\tau} \right)}$$

#### B. Cơ sở Khoa học
*   **Đa cực tích cực (Multi-positive):** Mở rộng SimCLR sang môi trường giám sát, kéo tất cả các mẫu cùng lớp trong batch lại gần nhau và đẩy tất cả các mẫu khác lớp ra xa.

---

### 4. InfoNCE / NT-Xent Loss

#### A. Công thức Toán học
Hàm loss nền tảng của học tương phản tự giám sát (Self-Supervised Contrastive Learning) và học tương phản đa phương thức (như CLIP):

$$\mathcal{L}_{InfoNCE} = -\log \frac{\exp\left( \frac{\text{sim}(z_i, z_j)}{\tau} \right)}{\sum_{k=1}^{2N} \mathbb{I}_{[k \neq i]} \exp\left( \frac{\text{sim}(z_i, z_k)}{\tau} \right)}$$

Trong đó:
*   $z_i, z_j$ là hai biểu diễn (views) được sinh ra từ cùng một ảnh gốc qua hai phép augment khác nhau (Positive pair).
*   $2N$ là số lượng mẫu trong batch sau khi nhân đôi qua augmentation.
*   $\text{sim}(u, v) = \frac{u^T v}{\|u\|_2 \|v\|_2}$ là độ tương đồng Cosine.
*   $\mathbb{I}$ là hàm chỉ thị.

#### B. Cơ sở Khoa học
*   **Ước lượng thông tin tương hỗ (Mutual Information):** InfoNCE thiết lập giới hạn dưới cho thông tin tương hỗ giữa hai góc nhìn (views) của cùng một thực thể dữ liệu.
*   **Alignment & Uniformity:** InfoNCE tối ưu hóa đồng thời tính đồng nhất (alignment) của các view cùng ảnh và sự phân bố đều (uniformity) của tất cả các ảnh khác nhau trên mặt cầu đơn vị để giữ tính tổng quát tối đa của đặc trưng trích xuất.
---

## IV. Bảng So Sánh Toàn Diện Các Hàm Loss

| Tên Hàm Loss | Độ Phức Tạp Batch | Cơ Chế Lấy Mẫu (Mining) | Ưu Điểm Nổi Bật | Nhược Điểm Chính | Trường Hợp Sử Dụng Tối Ưu |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Semi-hard Triplet** | $O(N^3)$ | Semi-hard Mining (FaceNet) | Gradient ổn định, biên rõ ràng. | Dễ bão hòa ở giai đoạn cuối. | Nhận diện khuôn mặt, dataset vừa. |
| **Soft-Margin Triplet**| $O(N^3)$ | Không bắt buộc | Không cần chỉnh margin $\alpha$, gradient liên tục. | Hội tụ chậm hơn Triplet gốc. | Khi khó thiết lập tham số margin. |
| **Quadruplet** | $O(N^4)$ | Quadruplet Mining | Tối ưu hóa cấu trúc khoảng cách toàn cục tốt hơn. | Độ phức tạp tính toán rất lớn. | Nhận diện sinh trắc học độ chính xác cao. |
| **Histogram** | $O(N^2)$ | Không cần lấy mẫu | Tối ưu hóa trực tiếp AUC, không phụ thuộc vào mining. | Khó tối ưu hóa song song trên GPU. | Tập dữ liệu có nhiều nhiễu nhãn. |
| **Angular** | $O(N^3)$ | Triplet Mining | Bất biến tỷ lệ đặc trưng (Scale-invariant). | Phức tạp trong việc thiết lập góc $\alpha$. | Dữ liệu biến đổi lớn về độ sáng/zoom. |
| **Lifted Structured** | $O(N^2)$ | Log-Sum-Exp (Smooth Max) | Tận dụng cấu trúc toàn batch, hội tụ nhanh. | Nhạy cảm với nhiễu cực đoan (outliers). | Image Retrieval chung. |
| **Multi-Similarity (MS)**| $O(N^2)$ | MS Pair Miner (độ tương đồng chéo) | Gán trọng số động tinh vi, là một trong những loss mạnh nhất. | Yêu cầu tinh chỉnh nhiều siêu tham số. | Image Retrieval phức tạp, SOTA. |
| **Circle Loss (Pair)** | $O(N^2)$ | Trọng số động tự thích ứng | Tối ưu hóa không đối xứng, tự động tránh overfitting. | Chỉ tối ưu tốt trên L2 normalized embeddings. | Fine-grained classification, Re-ID. |
| **Ranked List (RLL)** | $O(N^2)$ | Phân dải khoảng cách (Margin band) | Bảo toàn cấu trúc đã hội tụ, tránh xáo trộn đặc trưng. | Thiết lập nhiều ngưỡng khoảng cách phức tạp. | Image Retrieval quy mô lớn. |
| **Binomial Deviance** | $O(N^2)$ | Không bắt buộc | Kháng nhiễu cực tốt, gradient trơn tru. | Lực hội tụ ở vùng cận biên yếu hơn hinge. | Dữ liệu thực tế có nhãn bị sai lệch nhiều. |
| **Margin (DWS)** | $O(N^2)$ | Distance-Weighted Sampling | Tự học margin cho từng lớp, phân bổ mẫu hình học chuẩn. | Lấy mẫu DWS tốn tài nguyên CPU/GPU. | Bộ dữ liệu lớn có số chiều đặc trưng cao. |
| **Proxy Anchor** | $O(N \times C)$ | So sánh trực tiếp mẫu với Proxy lớp | Tốc độ hội tụ siêu nhanh, độ phức tạp tuyến tính. | Phụ thuộc vào chất lượng của khởi tạo Proxy. | Tập dữ liệu khổng lồ (hàng triệu ảnh, vạn lớp). |
| **ArcFace** | $O(N \times C)$ | Không cần (Tích hợp lớp phân loại) | Gom cụm cực kỳ chặt chẽ, tối ưu phân tách góc. | Khó hội tụ nếu khởi chạy với tốc độ học lớn. | Nhận diện khuôn mặt, Re-identification. |
| **Supervised Contrastive**| $O(N^2)$ | Tương phản đa cực tích cực | Học biểu diễn đặc trưng tổng quát và bền vững. | Cần kích thước batch cực lớn để hoạt động tốt. | Huấn luyện hai giai đoạn (Representation). |
