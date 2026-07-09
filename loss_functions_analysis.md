# Phân Tích Chuyên Sâu: Các Hàm Loss Cải Tiến Trong Metric Learning

Tài liệu này trình bày chi tiết về bản chất toán học, cơ sở khoa học và cơ chế hoạt động của các hàm loss nâng cao dùng trong học đặc trưng sâu (Deep Metric Learning / Representation Learning).

---

## Danh Sách Ký Hiệu Chung
*   $x_i \in \mathbb{R}^D$: Vector đặc trưng (feature/embedding) của mẫu thứ $i$.
*   $y_i$: Nhãn lớp (class label) của mẫu $x_i$.
*   $d(x_i, x_j) = \|f(x_i) - f(x_j)\|_2$: Khoảng cách Euclid giữa hai vector đặc trưng đã được chuẩn hóa L2.
*   $S_{i,j} = \frac{x_i^T x_j}{\|x_i\|_2 \|x_j\|_2}$: Độ tương đồng Cosine giữa hai vector.
*   $[x]_+ = \max(0, x)$: Hàm ReLU / Bản lề (hinge function).
*   $\alpha, \beta, \lambda, m$: Các siêu tham số (hyperparameters) như margin, scale, temperature.

---

## 1. Semi-hard Triplet Loss (CVPR 2015)

### A. Công thức Toán học
Triplet Loss chuẩn hóa mối quan hệ giữa ba mẫu: **Anchor** ($a$), **Positive** ($p$ - cùng lớp với Anchor) và **Negative** ($n$ - khác lớp với Anchor).

$$\mathcal{L}_{Triplet} = \sum_{i=1}^{N} \left[ d(a_i, p_i)^2 - d(a_i, n_i)^2 + \alpha \right]_+$$

Trong đó:
*   $\alpha > 0$ là biên độ khoảng cách tối thiểu (margin) bắt buộc giữa cặp positive và negative.

### B. Cơ chế Khai thác Mẫu Semi-hard (Semi-hard Mining)
Trong tập dữ liệu lớn, hầu hết các bộ ba ngẫu nhiên đều có loss bằng $0$ (mẫu dễ - easy triplets, thỏa mãn $d(a, n) > d(a, p) + \alpha$). Nếu huấn luyện trên các mẫu này, mô hình sẽ không học được gì và gradient bị biến mất. Ngược lại, các mẫu quá khó (hardest negatives, thỏa mãn $d(a, n) < d(a, p)$) lại dễ gây mất ổn định gradient ở giai đoạn đầu.

**Semi-hard Mining** (được giới thiệu trong FaceNet) lựa chọn mẫu Negative $n_i$ cho Anchor $a_i$ và Positive $p_i$ thỏa mãn điều kiện góc/khoảng cách:

$$d(a_i, p_i)^2 < d(a_i, n_i)^2 < d(a_i, p_i)^2 + \alpha$$

### C. Cơ sở Khoa học
*   **Vùng biên Margin:** Negative nằm xa Anchor hơn Positive một chút nhưng vẫn nằm trong phạm vi margin.
*   **Tác động:** Tạo ra gradient có độ lớn vừa phải, giúp mô hình hội tụ ổn định và học được các đặc trưng tinh vi ở vùng biên phân tách lớp mà không bị nhiễu bởi các mẫu cực đoan (outliers).

---

## 2. Lifted Structured Loss (CVPR 2016)

### A. Công thức Toán học
Thay vì tính toán trên các triplet hay cặp riêng lẻ, Lifted Structured Loss tối ưu cấu trúc batch bằng cách tính toán khoảng cách từ một cặp Positive tới tất cả các mẫu Negative trong batch:

$$\mathcal{L}_{Lifted} = \frac{1}{2|P|} \sum_{(i,j) \in P} \left[ \max \left( 0, \mathcal{J}_{i,j} \right) \right]^2$$

Với hàm cấu trúc $\mathcal{J}_{i,j}$ được định nghĩa là:

$$\mathcal{J}_{i,j} = d(x_i, x_j) + \log \left( \sum_{k: y_k \neq y_i} \exp( \alpha - d(x_i, x_k) ) + \sum_{k: y_k \neq y_j} \exp( \alpha - d(x_j, x_k) ) \right)$$

Trong đó:
*   $P$ là tập hợp tất cả các cặp Positive $(i, j)$ trong mini-batch.
*   $\alpha$ là tham số margin.

### B. Cơ sở Khoa học
*   **Hàm Log-Sum-Exp:** Hàm này đóng vai trò như một bộ xấp xỉ trơn (smooth approximation) của phép toán tìm Negative khó nhất ($\max_{k} (\alpha - d(x_i, x_k))$).
*   **Tận dụng Batch:** Thay vì cập nhật gradient dựa trên 1 cặp đơn lẻ, nó cập nhật gradient dựa trên mối tương quan giữa cặp positive và **toàn bộ** cấu trúc negative xung quanh, giúp không gian biểu diễn học được cấu trúc đa diện (manifold structure) trơn tru hơn.

---

## 3. N-Pair Loss (NeurIPS 2016)

### A. Công thức Toán học
Mở rộng cấu hình Triplet sang cấu hình đa lớp trong cùng một batch bằng cách tối ưu hóa đồng thời 1 Anchor với 1 Positive và $N-1$ mẫu Negative thuộc $N-1$ lớp khác nhau:

$$\mathcal{L}_{N-pair} = \frac{1}{N} \sum_{i=1}^{N} \log \left( 1 + \sum_{j \neq i} \exp( x_i^T x_j^+ - x_i^T x_i^+ ) \right)$$

Trong đó:
*   $x_i$ là Anchor của lớp thứ $i$.
*   $x_i^+$ là Positive tương ứng của lớp $i$.
*   $x_j^+$ là Positive của lớp $j$ (đóng vai trò là Negative đối với Anchor $x_i$).

### B. Cơ sở Khoa học
*   **Mối liên hệ với Cross-Entropy:** Công thức trên thực chất là hàm Softmax Cross-Entropy với logit là tích vô hướng tương đồng giữa các đặc trưng.
*   **Tối ưu hóa đa hướng:** Thay vì chỉ kéo 1 Positive và đẩy 1 Negative như Triplet Loss thông thường, N-Pair Loss ép Anchor phải phân biệt Positive của nó với tất cả các Negative khác lớp cùng một lúc, loại bỏ hiện tượng đẩy lớp này nhưng vô tình làm lệch lớp khác.

---

## 4. Multi-Similarity Loss (MS Loss - CVPR 2019)

### A. Công thức Toán học
MS Loss xây dựng một cơ chế gán trọng số động cho các cặp mẫu dựa trên ba độ tương đồng cốt lõi: *Self-similarity*, *Relative-similarity*, và *Mutual-similarity*.

$$\mathcal{L}_{MS} = \frac{1}{N} \sum_{i=1}^{N} \left\{ \frac{1}{\alpha} \log \left[ 1 + \sum_{k \in P_i} \exp(-\alpha (S_{i,k} - \lambda)) \right] + \frac{1}{\beta} \log \left[ 1 + \sum_{l \in N_i} \exp(\beta (S_{i,l} - \lambda)) \right] \right\}$$

Trong đó:
*   $S_{i,j}$ là độ tương đồng Cosine giữa mẫu $i$ và mẫu $j$.
*   $P_i$ là tập các Positive được chọn lọc của Anchor $i$.
*   $N_i$ là tập các Negative được chọn lọc của Anchor $i$.
*   $\alpha, \beta$ là các hệ số co giãn thang đo gradient của Positive và Negative.
*   $\lambda$ là biên độ tương đồng trung tâm.

### B. Cơ sở Khoa học
*   **Chọn mẫu thông minh (Informativeness):** Nó chỉ chọn các cặp Positive có độ tương đồng thấp hơn mức tối đa của Negative ($S_{i,k} < \max_{l \in N_i} S_{i,l} + \epsilon$) và các cặp Negative có độ tương đồng cao hơn mức tối thiểu của Positive ($S_{i,l} > \min_{k \in P_i} S_{i,k} - \epsilon$).
*   **Trọng số động:** Trọng số gradient của mỗi cặp được tự động định hình dựa trên khoảng cách của nó với các cặp khác. Các cặp cực khó sẽ nhận lượng cập nhật gradient rất lớn, giúp tối ưu hóa biên phân lớp chặt chẽ.

---

## 5. Circle Loss (CVPR 2020)

### A. Công thức Toán học
Circle Loss thiết lập sự tối ưu hóa không đối xứng đối với các cặp Positive và Negative bằng cách đưa vào các hệ số học động $\alpha_i^p$ và $\alpha_j^n$:

$$\mathcal{L}_{Circle} = \log \left[ 1 + \sum_{j \in N} \exp(\gamma \alpha_j^n (s_j^n - \Delta_n)) \sum_{i \in P} \exp(-\gamma \alpha_i^p (s_i^p - \Delta_p)) \right]$$

Với các hệ số điều chỉnh gradient động:

$$\alpha_i^p = [O_p - s_i^p]_+, \quad \alpha_j^n = [s_j^n - O_n]_+$$

Trong đó:
*   $s_i^p$ và $s_j^n$ là độ tương đồng Cosine của Positive và Negative.
*   $\gamma$ là tham số scale.
*   $\Delta_p, \Delta_n$ là các margin tương ứng.
*   $O_p = 1 + \Delta_p$ và $O_n = -\Delta_n$ là các điểm mốc tối ưu mục tiêu.

### B. Cơ sở Khoa học
*   **Tối ưu hóa linh hoạt:** Khi độ tương đồng Positive $s_i^p$ đã tiệm cận $1$ (hoặc $O_p$), hệ số $\alpha_i^p \to 0$, tức là mô hình sẽ ngừng kéo cặp này gần hơn nữa để tránh quá khớp (overfitting). Ngược lại, lực đẩy Negative $\alpha_j^n$ tăng lên nếu nó nằm quá gần vùng an toàn.
*   **Biên hình tròn:** Trong không gian biểu diễn toán học, biên quyết định giữa Positive và Negative của Circle Loss có dạng cung tròn. Điều này mang lại sự cân bằng hoàn hảo giữa việc thu hẹp khoảng cách nội bộ lớp và mở rộng khoảng cách liên lớp.

---

## 6. Proxy Anchor Loss (CVPR 2020)

### A. Công thức Toán học
Proxy Anchor Loss sử dụng các vector đại diện lớp (Proxies) để làm Anchor liên kết cấu trúc đặc trưng toàn batch:

$$\mathcal{L}_{PA} = \frac{1}{|P^+|} \sum_{p \in P^+} \log \left[ 1 + \sum_{x \in X_p^+} \exp(-\alpha(s(x, p) - \delta)) \right] + \frac{1}{|P|} \sum_{p \in P} \log \left[ 1 + \sum_{x \in X_p^-} \exp(\alpha(s(x, p) + \delta)) \right]$$

Trong đó:
*   $P$ là tập hợp tất cả các Proxy lớp.
*   $P^+$ là tập các Proxy có ít nhất một mẫu cùng lớp xuất hiện trong batch.
*   $s(x, p)$ là độ tương đồng Cosine giữa mẫu dữ liệu $x$ và Proxy $p$.
*   $X_p^+$ là tập các mẫu cùng lớp với Proxy $p$ trong batch.
*   $X_p^-$ là tập các mẫu khác lớp với Proxy $p$ trong batch.
*   $\alpha$ là tham số scale, $\delta$ là margin.

### B. Cơ sở Khoa học
*   **Giảm độ phức tạp tính toán:** Triplet Loss truyền thống so sánh các mẫu với nhau dẫn đến độ phức tạp $O(N^3)$. Proxy Anchor Loss so sánh mẫu với các Proxy cố định nên độ phức tạp giảm xuống tuyến tính $O(N \times C)$ ($C$ là số lượng lớp).
*   **Tối ưu liên kết:** Mỗi Proxy hoạt động như một thỏi nam châm thu hút các mẫu cùng lớp và đẩy lùi các mẫu khác lớp, giúp giải quyết triệt để vấn đề mất cân bằng mẫu và tiếng ồn trong mini-batch.

---

## 7. ArcFace (Additive Angular Margin Loss - CVPR 2019)

### A. Công thức Toán học
ArcFace thực hiện tối ưu hóa biên độ góc trực tiếp trong lớp Softmax phân loại bằng cách chuẩn hóa các vector đặc trưng và trọng số phân lớp:

$$\mathcal{L}_{ArcFace} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp\left( s \cdot \cos(\theta_{y_i} + m) \right)}{\exp\left( s \cdot \cos(\theta_{y_i} + m) \right) + \sum_{j \neq y_i} \exp\left( s \cdot \cos\theta_j \right)}$$

Trong đó:
*   $\theta_j$ là góc giữa vector đặc trưng trích xuất $x_i$ và trọng số lớp thứ $j$ ($W_j$).
*   $s$ là hệ số phóng đại thang đo (feature scale).
*   $m$ là biên độ góc cộng thêm (additive angular margin).

### B. Cơ sở Khoa học
*   **Biên độ góc geodesic:** Việc cộng trực tiếp góc biên độ $m$ trong hàm $\cos(\theta + m)$ tương đương với việc ép buộc các đặc trưng cùng lớp phải nằm trong một hình nón góc cực hẹp bao quanh trọng số lớp $W_j$.
*   **Tính phân tách cực cao:** ArcFace tối đa hóa khoảng cách góc giữa các lớp khác nhau trên mặt cầu đơn vị, tạo ra không gian biểu diễn có tính phân tách đặc trưng (discriminative power) mạnh mẽ nhất trong các dòng Softmax cải tiến.

---

## 8. Supervised Contrastive Loss (SupCon - NeurIPS 2020)

### A. Công thức Toán học
SupCon mở rộng khái niệm học tương phản tự giám sát sang môi trường học có giám sát bằng cách kéo tất cả các mẫu cùng lớp lại gần nhau:

$$\mathcal{L}_{SupCon} = \sum_{i \in I} \frac{-1}{|P(i)|} \sum_{p \in P(i)} \log \frac{\exp\left( \frac{z_i^T z_p}{\tau} \right)}{\sum_{a \in A(i)} \exp\left( \frac{z_i^T z_a}{\tau} \right)}$$

Trong đó:
*   $I$ là tập hợp các chỉ số mẫu trong batch.
*   $z_i$ là vector đặc trưng đã được chuẩn hóa L2 của mẫu $i$.
*   $P(i) = \{ p \in I : y_p = y_i \land p \neq i \}$ là tập hợp tất cả các mẫu cùng lớp với mẫu $i$ trong batch (tập Positive).
*   $A(i) = I \setminus \{i\}$ là tập hợp tất cả các mẫu khác mẫu $i$ trong batch.
*   $\tau > 0$ là tham số nhiệt độ (temperature).

### B. Cơ sở Khoa học
*   **Đa cực tích cực (Multi-positive):** Khác với tự giám sát SimCLR chỉ kéo 1 phiên bản augmented của chính nó, SupCon tận dụng nhãn giám sát để kéo **toàn bộ** các mẫu cùng lớp trong batch.
*   **Cấu trúc hình học:** Tối ưu hóa SupCon tương đương với việc ép cấu trúc không gian đặc trưng phân bố thành các cụm dày đặc phân tách rõ rệt trên mặt cầu đơn vị.

---

## 9. InfoNCE / NT-Xent Loss

### A. Công thức Toán học
Hàm loss nền tảng của học tương phản tự giám sát (Self-Supervised Contrastive Learning) và học tương phản đa phương thức (như CLIP):

$$\mathcal{L}_{InfoNCE} = -\log \frac{\exp\left( \frac{\text{sim}(z_i, z_j)}{\tau} \right)}{\sum_{k=1}^{2N} \mathbb{I}_{[k \neq i]} \exp\left( \frac{\text{sim}(z_i, z_k)}{\tau} \right)}$$

Trong đó:
*   $z_i, z_j$ là hai biểu diễn (views) được sinh ra từ cùng một ảnh gốc qua hai phép augment khác nhau (Positive pair).
*   $2N$ là số lượng mẫu trong batch sau khi nhân đôi qua augmentation.
*   $\text{sim}(u, v) = \frac{u^T v}{\|u\|_2 \|v\|_2}$ là độ tương đồng Cosine.
*   $\mathbb{I}$ là hàm chỉ thị.

### B. Cơ sở Khoa học
*   **Ước lượng thông tin tương hỗ (Mutual Information):** InfoNCE thiết lập giới hạn dưới cho thông tin tương hỗ giữa hai góc nhìn (views) của cùng một thực thể dữ liệu.
*   **Alignment & Uniformity:** InfoNCE tối ưu hóa đồng thời tính đồng nhất (alignment) của các view cùng ảnh và sự phân bố đều (uniformity) của tất cả các ảnh khác nhau trên mặt cầu đơn vị để giữ tính tổng quát tối đa của đặc trưng trích xuất.
