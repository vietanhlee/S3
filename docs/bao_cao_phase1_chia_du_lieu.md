# BÁO CÁO KHOA HỌC PHASE 1: PHƯƠNG PHÁP PHÂN CHIA DỮ LIỆU CHỐNG RÒ RỈ (DATA LEAKAGE) TRÊN BỘ DỮ LIỆU TỰ NHIÊN S3

---

## 1. Đặt Vấn Đề và Bản Chất Của Hiện Tượng Rò Rỉ Dữ Liệu (Data Leakage)

Trong xây dựng các mô hình Học sâu (Deep Learning) cho bài toán phân loại ảnh cấu trúc vi phẫu gỗ, hiện tượng **Rò rỉ dữ liệu (Data Leakage)** là một trong những nguyên nhân hàng đầu dẫn đến việc đánh giá sai lệch hiệu năng của mô hình. Cụ thể, mô hình có thể đạt độ chính xác cực kỳ cao trên tập kiểm thử (Test Set) trong phòng thí nghiệm nhưng lại suy giảm nghiêm trọng khi triển khai trên thực tế (Generalization Gap).

### 1.1. Rò rỉ cấp độ mẫu vật (Specimen-level / Sample-level Leakage)
Bộ dữ liệu ảnh gỗ thường có cấu trúc phân cấp (Hierarchical Structure):
* Một loài gỗ (Class) bao gồm nhiều mẫu vật lý khác nhau (mẫu thu thập từ các cây khác nhau, địa điểm khác nhau).
* Từ một mẫu vật lý (được đại diện bởi một thư mục con `subfolder`), người ta thực hiện cắt lát, chuẩn bị tiêu bản và chụp nhiều ảnh ở các góc độ, lát cắt hoặc điều kiện ánh sáng khác nhau.

Nếu áp dụng phương pháp phân chia ngẫu nhiên truyền thống ở cấp độ ảnh (như Stratified Random Split):
$$\{x_{i,1}, x_{i,2}, \dots, x_{i,m}\} \in S_j$$
Các bức ảnh thuộc cùng một mẫu vật $S_j$ sẽ bị phân mảnh và phân bổ đồng thời vào cả tập Huấn luyện (Train), Kiểm định (Validation) và Kiểm thử (Test).

### 1.2. Hậu quả học thuật và thực tiễn
Mô hình CNN (như `tf_efficientnet_b4` hoặc `convnext_tiny`) có năng lực biểu diễn rất mạnh. Khi xảy ra rò rỉ cấp độ mẫu vật, mô hình sẽ không học các đặc trưng phân loại mang tính phân loại học (Taxonomic Markers) như cấu trúc mạch gỗ, sự phân bố sợi gỗ hay cấu trúc tia. Thay vào đó, mô hình sẽ **ghi nhớ (memorize)** các đặc trưng nhiễu không liên quan của mẫu vật cụ thể đó (ví dụ: độ dày lát cắt, màu sắc chất nhuộm tiêu bản, vết xước dao cắt, cường độ sáng của camera thu nhận). 

Kết quả là mô hình đạt độ chính xác kiểm thử tiệm cận $100\%$ (do tập Test chứa các ảnh rò rỉ từ cùng mẫu vật trong tập Train), nhưng hoàn toàn thất bại khi nhận diện một mẫu gỗ mới chưa từng xuất hiện trong quá trình huấn luyện.

---

## 2. Nguyên Tắc Cô Lập Mẫu Vật (Specimen Group Isolation)

Để giải quyết triệt để vấn đề rò rỉ dữ liệu, chúng tôi đề xuất nguyên tắc **Cô lập cấp độ nhóm mẫu vật (Group Isolation at Subfolder Level)** làm nền tảng cốt lõi cho Phase 1.

### 2.1. Phát biểu toán học
Gọi $\mathcal{X} = \{x_1, x_2, \dots, x_M\}$ là tập hợp tất cả các ảnh trong bộ dữ liệu, và $\mathcal{S} = \{S_1, S_2, \dots, S_N\}$ là tập hợp các thư mục con đại diện cho các mẫu vật lý độc lập. Mỗi bức ảnh $x_i$ được gán nhãn lớp $y_i \in \mathcal{Y}$ và thuộc về một mẫu vật duy nhất $S(x_i) \in \mathcal{S}$.

Quá trình phân chia dữ liệu là việc phân hoạch tập hợp mẫu vật $\mathcal{S}$ thành 3 tập hợp rời rạc:
$$\mathcal{S}_{\text{train}}, \mathcal{S}_{\text{val}}, \mathcal{S}_{\text{test}} \subset \mathcal{S}$$
thỏa mãn điều kiện nghiêm ngặt:
$$\mathcal{S}_{\text{train}} \cap \mathcal{S}_{\text{val}} = \emptyset, \quad \mathcal{S}_{\text{train}} \cap \mathcal{S}_{\text{test}} = \emptyset, \quad \mathcal{S}_{\text{val}} \cap \mathcal{S}_{\text{test}} = \emptyset$$

Từ đó, các tập dữ liệu ảnh tương ứng được định nghĩa là:
$$\mathcal{D}_{\text{train}} = \{x_i \in \mathcal{X} \mid S(x_i) \in \mathcal{S}_{\text{train}}\}$$
$$\mathcal{D}_{\text{val}} = \{x_i \in \mathcal{X} \mid S(x_i) \in \mathcal{S}_{\text{val}}\}$$
$$\mathcal{D}_{\text{test}} = \{x_i \in \mathcal{X} \mid S(x_i) \in \mathcal{S}_{\text{test}}\}$$

Điều này đảm bảo rằng **không có bất kỳ ảnh nào thuộc cùng một mẫu vật lý xuất hiện đồng thời ở hai tập khác nhau**.

### 2.2. Xử lý trường hợp đặc biệt (Minority Classes)
Đối với các lớp gỗ có số lượng mẫu vật lý quá ít (dưới 3 subfolders):
* Nếu lớp chỉ có $1$ subfolder: Toàn bộ ảnh được đưa vào $\mathcal{D}_{\text{train}}$. Chấp nhận tập $\mathcal{D}_{\text{val}}$ và $\mathcal{D}_{\text{test}}$ của lớp đó trống để tránh rò rỉ.
* Nếu lớp có $2$ subfolders: Một subfolder đưa vào $\mathcal{D}_{\text{train}}$, một subfolder đưa vào $\mathcal{D}_{\text{test}}$ (hoặc $\mathcal{D}_{\text{val}}$).
* Nếu lớp có $\ge 3$ subfolders: Áp dụng các thuật toán phân hoạch xấp xỉ theo tỷ lệ đích (ví dụ: $60\%$ Train, $20\%$ Val, $20\%$ Test).

---

## 3. Cơ Chế Hoạt Động Chi Tiết Của Các Phương Pháp Phân Chia Dữ Liệu

Trong mã nguồn của dự án (đặc biệt trong tệp `split_methods.py`), chúng tôi đã triển khai 6 phương pháp phân chia dữ liệu chính thức được sử dụng trong quy trình so sánh hiệu năng (`train_split_comparison.py`). 

Trước khi thực hiện phân chia, các ảnh thô $x_i$ được đưa qua mô hình trích xuất đặc trưng mạng neural sâu để nhận được vector biểu diễn đặc trưng (feature embedding) $e_i \in \mathbb{R}^D$ (với $D = 768$ hoặc $1024$ tùy thuộc vào backbone mạng CNN/ViT). Vector biểu diễn đại diện cho một thư mục con mẫu vật lý $S_j$ ($j = 1, \dots, G$) được xác định bằng centroid của các ảnh thuộc thư mục con đó:
$$E_j = \frac{1}{|S_j|} \sum_{x_i \in S_j} e_i$$

### PP2: Phân chia theo Khoảng cách Mahalanobis Tiệm tiến (Mahalanobis Iterative Centroid Split)

* **Nguyên lý khoa học**: Phương pháp này vận dụng khoảng cách Mahalanobis để xác định các mẫu dị biệt (outliers) trong không gian đặc trưng đa chiều, từ đó phân bổ chúng vào tập kiểm thử nhằm tạo ra một kịch bản đánh giá dịch chuyển phân phối đặc trưng (feature distribution shift / covariate shift). Khoảng cách Mahalanobis tối ưu hơn khoảng cách Euclidean do nó tích hợp cấu trúc hiệp phương sai của dữ liệu, loại bỏ ảnh hưởng của tương quan chéo giữa các chiều đặc trưng.
* **Quy trình thuật toán chi tiết**:
  1. Với mỗi lớp $c$, trích xuất tập hợp các subfolder embeddings $\{E_1, E_2, \dots, E_G\}$.
  2. Nếu số lượng mẫu vật $G \ge 5$, áp dụng thuật toán Phân tích Thành phần Chính (PCA) để giảm số chiều từ $D$ xuống $d' = \min(G-2, 128)$ nhằm tránh hiện tượng "lời nguyền đa chiều" (curse of dimensionality) và đảm bảo tính ổn định của ma trận hiệp phương sai. Ký hiệu vector sau giảm chiều là $z_j \in \mathbb{R}^{d'}$.
  3. Khởi tạo tập đích cần chọn cho tập Test là $T$ (số lượng đích $N_{\text{test}}$) và tập Val là $V$ (số lượng đích $N_{\text{val}}$). Ký hiệu tập các mẫu vật còn lại chưa phân bổ là $U$ (ban đầu $U = \{1, \dots, G\}$).
  4. **Lặp chọn tập Test**: Lặp lại $N_{\text{test}}$ lần, tại mỗi bước:
     * Tính toán centroid (vector trung bình) của các phần tử trong $U$:
       $$\mu_U = \frac{1}{|U|} \sum_{j \in U} z_j$$
     * Tính toán ma trận hiệp phương sai của tập $U$, bổ sung tham số chính quy hóa $\epsilon = 10^{-6}$ để đảm bảo ma trận luôn khả nghịch:
       $$\Sigma_U = \frac{1}{|U| - 1} \sum_{j \in U} (z_j - \mu_U)(z_j - \mu_U)^T + \epsilon I$$
     * Tính khoảng cách Mahalanobis của từng mẫu $j \in U$ đến $\mu_U$:
       $$d_{\text{Mahal}}(z_j, \mu_U) = \sqrt{(z_j - \mu_U)^T \Sigma_U^{-1} (z_j - \mu_U)}$$
     * Rút mẫu có khoảng cách lớn nhất:
       $$j^* = \arg\max_{j \in U} d_{\text{Mahal}}(z_j, \mu_U)$$
     * Cập nhật: $T \leftarrow T \cup \{j^*\}$ và $U \leftarrow U \setminus \{j^*\}$.
  5. **Lặp chọn tập Val**: Lặp lại tương tự $N_{\text{val}}$ lần để chọn ra các mẫu từ tập $U$ còn lại đưa vào $V$ dựa trên khoảng cách Mahalanobis lớn nhất được tính toán động sau mỗi bước.
  6. Các mẫu còn lại trong $U$ sau hai quá trình trên được gán vào tập huấn luyện Train ($R = U$).
* **Hàm lượng học thuật**: Đóng vai trò là công cụ "Stress-test" ở cấp độ đặc trưng. Việc đẩy các mẫu vật lý có khoảng cách đặc trưng xa nhất vào tập Test/Val buộc mô hình phải học được các đặc trưng mang tính bất biến (domain-invariant features) của loài gỗ, thay vì học các đặc trưng dễ nhớ (shortcut learning) phân bổ tập trung gần centroid.

### PP4: Phân chia dựa trên Phân cụm Phân cấp (Hierarchical Clustering Split)

* **Nguyên lý khoa học**: Mẫu vật gỗ thu thập trong tự nhiên thường chứa các nhóm nhỏ phân tách tự nhiên (ví dụ do nguồn gốc thổ nhưỡng, thời tiết, hoặc các biến thể loài con). Phương pháp này tự động phân tách cấu trúc cụm đặc trưng tự nhiên ở cấp độ mẫu vật bằng thuật toán Phân cụm Phân cấp tích lũy (Agglomerative Clustering) và thực hiện phân chia cô lập theo cụm (cluster-level partition).
* **Quy trình thuật toán chi tiết**:
  1. Đại diện các mẫu vật bằng $z_j \in \mathbb{R}^{d'}$ (sau giảm chiều PCA).
  2. **Xác định số cụm tối ưu $K^*$**: 
     * Thực hiện phân cụm với số lượng cụm $K \in [3, 30]$ (giới hạn tối đa bởi $G-1$).
     * Tính toán Tổng bình phương khoảng cách trong cụm (WCSS):
       $$\text{WCSS}(K) = \sum_{k=1}^K \sum_{z_j \in \mathcal{C}_k} \|z_j - \mu_{\mathcal{C}_k}\|^2$$
     * Thực hiện chuẩn hóa cả hai trục $K$ và $\text{WCSS}(K)$ về đoạn $[0, 1]$ để loại bỏ sai lệch về tỷ lệ đơn vị (scaling bias):
       $$x_K = \frac{K - 3}{K_{\text{max}} - 3}, \quad y_K = \frac{\text{WCSS}(K) - \text{WCSS}_{\text{min}}}{\text{WCSS}_{\text{max}} - \text{WCSS}_{\text{min}}}$$
     * Tự động xác định điểm "Khuỷu tay" (Elbow point) $K^*$ bằng cách chọn điểm có khoảng cách vuông góc xa nhất đến đường chéo nối điểm đầu $(x_3, y_3)$ và điểm cuối $(x_{K_{\text{max}}}, y_{K_{\text{max}}})$:
       $$K^* = \arg\max_{K} \frac{|x_K + y_K - 1|}{\sqrt{2}}$$
  3. **Phân cụm phân cấp**: Áp dụng tiêu chuẩn liên kết Ward (Ward's linkage) để giảm thiểu tổng phương sai nội cụm khi gộp các cụm lại với nhau. Kết quả thu được các cụm $\{\mathcal{C}_1, \mathcal{C}_2, \dots, \mathcal{C}_{K^*}\}$.
  4. **Tính khoảng cách cụm**: Tính khoảng cách Mahalanobis từ centroid của từng cụm $\mu_{\mathcal{C}_k}$ tới centroid chung toàn lớp $\mu_{\text{global}}$:
     $$d(\mathcal{C}_k) = \sqrt{(\mu_{\mathcal{C}_k} - \mu_{\text{global}})^T \Sigma_{\text{global}}^{-1} (\mu_{\mathcal{C}_k} - \mu_{\text{global}})}$$
  5. Sắp xếp các cụm theo khoảng cách giảm dần. Phân bổ nguyên cụm theo thứ tự ưu tiên: cụm xa nhất vào Test, tiếp theo vào Val, và phần còn lại vào Train.
* **Hàm lượng học thuật**: Ngăn ngừa rò rỉ thông tin ở cấp độ tập hợp mẫu (Cluster-level data leakage). Phương pháp này giả định rằng nếu mô hình chỉ học các đặc trưng bề mặt của một nhóm mẫu cụ thể, nó sẽ thất bại hoàn toàn khi kiểm thử trên một cụm mẫu vật mới đại diện cho một biến thể sinh học hoàn toàn khác của cùng một loài gỗ.

### PP5: Phân chia theo Đồ thị Tương đồng Cosine (Cosine Similarity Graph Split)

* **Nguyên lý khoa học**: Trong thực tế chuẩn bị mẫu, có nhiều mẫu gỗ gần như là bản sao của nhau (near-duplicates) do chụp trên cùng một tiêu bản cắt liên tiếp (serial sections). Phương pháp này mô hình hóa mối quan hệ tương đồng cao giữa các mẫu vật dưới dạng một đồ thị và sử dụng lý thuyết đồ thị để phát hiện và cô lập các nhóm mẫu vật bị trùng lặp này.
* **Quy trình thuật toán chi tiết**:
  1. Tính toán ma trận tương đồng cosine $W \in \mathbb{R}^{G \times G}$ giữa các vector embedding subfolder $\{E_j\}$.
  2. Xây dựng đồ thị vô hướng $G_c = (V_c, E_c)$ thông qua ma trận kề thưa $A$ (Adjacency Matrix) bằng cách áp ngưỡng tương đồng nghiêm ngặt $\tau = 0.92$:
     $$A_{ab} = \begin{cases} 1 & \text{if } W_{ab} \ge \tau \text{ and } a \ne b \\ 0 & \text{otherwise} \end{cases}$$
  3. Thực hiện thuật toán duyệt đồ thị (BFS/DFS) để tìm các Thành phần liên thông (Connected Components) $\{\mathcal{CC}_1, \mathcal{CC}_2, \dots, \mathcal{CC}_M\}$. Mỗi thành phần liên thông đại diện cho một chuỗi các mẫu gỗ có độ tương đồng đặc trưng vượt ngưỡng chấp nhận được.
  4. Để phân bổ dữ liệu, tính centroid của từng thành phần liên thông $\mu_{\mathcal{CC}_m}$ và tính khoảng cách Mahalanobis của centroid đó đến centroid lớp.
  5. Sắp xếp các thành phần liên thông theo khoảng cách giảm dần và phân bổ nguyên khối các thành phần này vào Test, Val, Train.
* **Hàm lượng học thuật**: Giải quyết bài toán rò rỉ dữ liệu ẩn (implicit leakage) do sự tương đồng quá mức giữa các mẫu độc lập. Việc gom các mẫu có độ tương đồng $\ge 0.92$ vào cùng một tập dữ liệu đảm bảo mô hình không được hưởng lợi từ việc ghi nhớ các biến thể hình ảnh gần như trùng khớp.

### PP7: Phân chia bằng Kiểm định Đối kháng (Adversarial Validation Split)

* **Nguyên lý khoa học**: Kỹ thuật này mượn ý tưởng từ mạng đối kháng GAN để chủ động kiến tạo một sự lệch phân phối (domain/distribution shift) rõ rệt giữa tập huấn luyện và tập kiểm thử. Bằng cách huấn luyện một mạng phân biệt (Discriminator) cố gắng tách biệt hai phần dữ liệu mẫu vật, chúng ta có thể định lượng mức độ dị biệt của mỗi mẫu vật và thiết lập tập kiểm thử khó khăn nhất có thể.
* **Quy trình thuật toán chi tiết**:
  1. Với mỗi lớp $c$, chia ngẫu nhiên các mẫu vật thành hai nhóm tạm thời có nhãn mục tiêu: Nhóm A ($y_j = 0$) và Nhóm B ($y_j = 1$).
  2. Thiết lập một mạng neural Perceptron đa tầng (MLP Discriminator) $f_{\theta}: \mathbb{R}^D \to (0, 1)$ với cấu trúc:
     $$h = \text{ReLU}(W_1 E_j + b_1)$$
     $$\tilde{h} = \text{Dropout}(h, p=0.3)$$
     $$p_j = \sigma(W_2 \tilde{h} + b_2)$$
     Trong đó $\sigma(z) = \frac{1}{1 + e^{-z}}$ là hàm kích hoạt Sigmoid.
  3. Huấn luyện mạng discriminator tối ưu hóa hàm mất mát Entropy chéo nhị phân (Binary Cross Entropy Loss):
     $$\mathcal{L}(\theta) = - \frac{1}{G} \sum_{j=1}^G \left[ y_j \log(p_j) + (1 - y_j) \log(1 - p_j) \right]$$
     bằng thuật toán tối ưu Adam với tốc độ học $\eta = 10^{-3}$ trong 30 chu kỳ (epochs).
  4. Đánh giá độ dị biệt của từng mẫu vật thông qua độ lệch tuyệt đối so với ngưỡng không phân biệt ($0.5$):
     $$d_j = |p_j - 0.5| \in [0, 0.5]$$
     * Mẫu vật có $d_j \approx 0$ đại diện cho các mẫu "chung" nằm ở vùng chồng lấn phân phối của đặc trưng.
     * Mẫu vật có $d_j \approx 0.5$ đại diện cho các mẫu cực kỳ đặc trưng, dễ dàng bị discriminator phát hiện sự khác biệt.
  5. Sắp xếp danh sách mẫu vật theo $d_j$ giảm dần. Đưa các mẫu có điểm $d_j$ cao nhất vào tập Test (ưu tiên) và tập Val, phần còn lại có điểm thấp (phân phối chung) được giữ lại làm tập Train.
* **Hàm lượng học thuật**: Tạo ra một mô phỏng chuẩn mực cho bài toán thực tế (Real-world Domain Adaptation). Mô hình buộc phải tổng quát hóa các quy luật sinh học vượt qua rào cản phân phối đặc trưng nhân tạo mà mạng discriminator đã chỉ ra.

### PP8: Phân chia Phân tầng theo Nhóm (StratifiedGroupKFold Split)

* **Nguyên lý khoa học**: Phân hoạch tập dữ liệu dạng nhóm (group-structured) và không cân bằng lớp (imbalanced labels) đòi hỏi việc phân hoạch phải đồng thời thỏa mãn hai ràng buộc: (1) Cách ly nhóm để chống rò rỉ dữ liệu và (2) Giữ nguyên tỷ lệ phân bố của các lớp trong mỗi fold tương đồng với phân phối toàn bộ tập dữ liệu ban đầu.
* **Quy trình thuật toán chi tiết**:
  1. Định nghĩa mỗi thực thể là một mẫu vật (group) gắn liền với nhãn lớp tương ứng.
  2. Xác định số fold cần chia $K$ dựa trên tỷ lệ split đích (Ví dụ: tỷ lệ test $20\% \implies K=5$).
  3. Tính toán ma trận phân phối đích cho từng lớp $c$ trên mỗi fold $f$:
     $$T_{f, c} = \frac{1}{K} \sum_{g=1}^G w_{g, c}$$
     Trong đó $w_{g, c}$ là số lượng ảnh của lớp $c$ nằm trong group $g$.
  4. Tiến hành thuật toán phân bổ tham lam (Greedy Allocation): Sắp xếp các nhóm dựa trên số lượng mẫu và mức độ mất cân bằng lớp. Lần lượt gán mỗi nhóm $g$ vào fold $f^*$ sao cho cực tiểu hóa độ lệch phân phối tích lũy:
     $$f^* = \arg\min_{f} \sum_{c=1}^C \left( \frac{C_{f, c} + w_{g, c}}{T_{f, c}} - 1 \right)^2$$
     Với $C_{f, c}$ là số lượng mẫu của lớp $c$ hiện có trong fold $f$.
  5. Chọn fold thứ nhất làm tập Test. Gộp các fold còn lại và tiếp tục thực hiện thuật toán phân hoạch tương tự để tách tập Val và tập Train theo tỷ lệ mong muốn.
* **Hàm lượng học thuật**: Đây là phương pháp phân hoạch dữ liệu chuẩn mực nhất trong Học máy để tối ưu hóa sự ổn định của gradient trong quá trình lan truyền ngược (backpropagation). Nó đảm bảo tất cả các lớp đều có đại diện đầy đủ ở cả ba tập dữ liệu mà không vi phạm nguyên tắc cô lập mẫu vật.

### PP9: Phân tầng Phân cụm và Phân bổ Đa dải Khoảng cách (Agglomerative Stratified Split)

* **Nguyên lý khoa học**: Khi tập kiểm thử bị dồn quá nhiều mẫu dị biệt (outliers) như PP2 hay PP4, nó có thể dẫn đến việc đánh giá hiệu năng quá bi quan. Phương pháp này giải quyết vấn đề bằng cách phân tầng không gian đặc trưng thành các dải đồng tâm (Distance Bands) và lấy mẫu đại diện từ mọi dải cho tất cả các tập dữ liệu, vừa bảo vệ tính cô lập mẫu vừa bảo đảm tính đại diện đồng đều của phân phối.
* **Quy trình thuật toán chi tiết**:
  1. Áp dụng Phân cụm Phân cấp Agglomerative để gom các mẫu vật thành $K^*$ cụm tự nhiên (sử dụng Ward's linkage và Elbow method tương tự PP4). Ký hiệu các cụm là $\{\mathcal{C}_1, \dots, \mathcal{C}_{K^*}\}$.
  2. Tính khoảng cách Mahalanobis từ centroid của từng cụm $\mu_{\mathcal{C}_k}$ tới centroid chung lớp $\mu_{\text{global}}$.
  3. Sắp xếp danh sách cụm theo khoảng cách tăng dần và phân chia thành 3 dải khoảng cách vật lý rời rạc:
     * Dải Gần (Near Band): Gồm các cụm đặc trưng cho cấu trúc điển hình nhất của lớp gỗ.
     * Dải Vừa (Mid Band): Gồm các cụm trung gian.
     * Dải Xa (Far Band): Gồm các cụm chứa các mẫu dị biệt lớn nhất.
  4. **Phân bổ động (Dynamic Allocation)**: Với mỗi dải, xáo trộn ngẫu nhiên thứ tự các cụm, sau đó duyệt qua từng cụm và tính toán độ hụt số lượng ảnh hiện tại so với mục tiêu phân bổ của các tập Train, Val, Test:
     $$\Delta_{\text{train}} = \max(0, N^{\text{target}}_{\text{train}} - N^{\text{current}}_{\text{train}})$$
     $$\Delta_{\text{val}} = \max(0, N^{\text{target}}_{\text{val}} - N^{\text{current}}_{\text{val}})$$
     $$\Delta_{\text{test}} = \max(0, N^{\text{target}}_{\text{test}} - N^{\text{current}}_{\text{test}})$$
     Quyết định đưa toàn bộ cụm mẫu vật này vào tập dữ liệu đang có độ hụt $\Delta$ lớn nhất.
* **Hàm lượng học thuật**: Thiết lập một cơ chế phân tầng đặc trưng (feature stratification) tiên tiến ở cấp độ subfolder. Thuật toán đảm bảo cả tập Train, Val và Test đều nhận được một tỷ lệ tương thích các mẫu gỗ điển hình, mẫu gỗ trung bình và mẫu gỗ dị biệt, làm cho kết quả đánh giá mô hình khách quan và tiệm cận nhất với phân phối tự nhiên thực tế.

---

## 4. Phương Pháp Chuẩn Cuối (End Version Split) và Cơ Chế Hoán Đổi Val/Test

### 4.1. Bản chất của End Version Split
Trong thực tế, do sự khác biệt về đặc tính sinh học, phương pháp thu thập tiêu bản và kích thước mẫu của từng loài gỗ là hoàn toàn khác nhau. Một phương pháp phân chia duy nhất không thể tối ưu cho toàn bộ các lớp.

Chúng tôi đã thiết kế **End Version Split** (được triển khai trong tệp `train_final.py`), là một phương pháp lai (Hybrid Method) cho phép cấu hình phương pháp phân chia tối ưu riêng biệt cho từng lớp gỗ dựa trên kết quả đánh giá thực nghiệm ở Phase 1:

| Loài gỗ (Class) | Phương pháp phân chia áp dụng | Tập hoán đổi đích (Val/Test Swap) | Ý nghĩa học thuật |
| :--- | :---: | :---: | :--- |
| **Afzelia africana** | PP8 (StratifiedGroupKFold) | Val | Ưu tiên cân bằng lớp tối ưu |
| **Afzelia bella** | PP4 (Hierarchical Clustering) | Val | Cô lập cụm tương đồng, lấy Val làm Test |
| **Afzelia pachyloba** | PP2 (Mahalanobis Iterative) | Val | Đẩy mẫu dị biệt vào Val, hoán đổi làm Test |
| **Afzelia quanzensis** | PP7 (Adversarial Validation) | Val | Thách thức đối kháng trên tập Test hoán đổi |
| **Dalbergia cochinchinensis** | PP4 (Hierarchical Clustering) | Test | Cô lập cụm tương đồng, giữ nguyên Test |
| **Dalbergia melanoxylon** | PP2 (Mahalanobis Iterative) | Val | Đẩy mẫu dị biệt vào Val, hoán đổi làm Test |
| **Dalbergia oliveri** | PP9 (Agglom Stratified) | Test | Phân bổ đa dải khoảng cách đồng đều |
| **Dalbergia rimosa** | PP4 (Hierarchical Clustering) | Test | Giữ nguyên cô lập cụm trên tập Test |
| **Dalbergia tonkinensis** | PP4 (Hierarchical Clustering) | Val | Hoán đổi Val/Test sau khi phân cụm |
| **Guibourtia arnoldiana** | PP4 (Hierarchical Clustering) | Val | Hoán đổi Val/Test sau khi phân cụm |
| **Guibourtia coleosperma** | PP4 (Hierarchical Clustering) | Test | Giữ nguyên cô lập cụm trên tập Test |
| **Guibourtia ehie** | PP2 (Mahalanobis Iterative) | Val | Thử thách mẫu dị biệt |
| **Peltogyne pubescens** | PP4 (Hierarchical Clustering) | Val | Hoán đổi Val/Test |
| **Pterocarpus erinaceus** | PP2 (Mahalanobis Iterative) | Test | Thử thách mẫu dị biệt trực tiếp trên Test |
| **Pterocarpus indicus** | PP4 (Hierarchical Clustering) | Test | Giữ nguyên cô lập cụm |
| **Pterocarpus macrocarpus** | PP9 (Agglom Stratified) | Val | Phân bổ đa dải khoảng cách hoán đổi |
| **Pterocarpus soyauxii** | PP4 (Hierarchical Clustering) | Test | Giữ nguyên cô lập cụm |
| **Sindora cochinchinensis** | PP9 (Agglom Stratified) | Test | Giữ nguyên phân bổ đa dải |
| **Sindora tonkinensis** | PP7 (Adversarial Validation) | Test | Đánh giá đối kháng trực tiếp trên Test |

*(Lưu ý: Loài `Pterocarpus sp` đã bị loại bỏ hoàn toàn khỏi hệ thống dữ liệu huấn luyện do không đủ số lượng mẫu vật lý tối thiểu phục vụ kiểm thử không rò rỉ).*

### 4.2. Cơ chế hoán đổi Val/Test (Val/Test Swap Mechanism)
Cơ chế hoán đổi hoạt động dựa trên thuật toán sau:
1. Đối với một lớp gỗ được chỉ định chế độ `"val"` (ví dụ: `(PP4, "val")`):
   - Hàm phân chia được gọi bình thường để tạo ra ba tập con tạm thời: $\mathcal{D}_{\text{train\_temp}}$, $\mathcal{D}_{\text{val\_temp}}$, $\mathcal{D}_{\text{test\_temp}}$.
   - Thực hiện hoán đổi vị trí của tập Validation và Test:
     $$\mathcal{D}_{\text{train}} = \mathcal{D}_{\text{train\_temp}}$$
     $$\mathcal{D}_{\text{val}} = \mathcal{D}_{\text{test\_temp}}$$
     $$\mathcal{D}_{\text{test}} = \mathcal{D}_{\text{val\_temp}}$$
2. Đối với các lớp được cấu hình `"test"` (ví dụ: `(PP4, "test")`):
   - Giữ nguyên kết quả phân chia ban đầu:
     $$\mathcal{D}_{\text{train}} = \mathcal{D}_{\text{train\_temp}}, \quad \mathcal{D}_{\text{val}} = \mathcal{D}_{\text{val\_temp}}, \quad \mathcal{D}_{\text{test}} = \mathcal{D}_{\text{test\_temp}}$$

**Giải thích ý nghĩa khoa học**: Việc hoán đổi này nhằm cân bằng độ khó của tập Test và tập Val đối với từng lớp gỗ cụ thể. Với một số phương pháp phân chia (như PP2 hoặc PP4), việc đưa các mẫu dị biệt vào tập Validation (tạm thời) sẽ giúp quá trình chọn lọc siêu tham số (Hyperparameter Tuning) diễn ra thực chất hơn, sau đó hoán đổi tập Test (chứa các mẫu tiêu biểu hơn của Val) để kiểm tra độ ổn định bền vững của mô hình.

---

## 5. Kết Luận

Giải pháp phân chia dữ liệu chống rò rỉ trong Phase 1 đã chuyển đổi toàn bộ quy trình từ **Phân chia mức ảnh ngẫu nhiên** sang **Phân chia mức nhóm mẫu vật lý (`subfolder`)**. Sự kết hợp linh hoạt giữa các kỹ thuật học máy (PCA, Phân cụm Hierarchical, Đồ thị liên thông Cosine, Học đối kháng Discriminator) cùng cơ chế cấu hình lớp lai (End Version Split) đảm bảo:
1. **Loại bỏ hoàn toàn rò rỉ dữ liệu**: Các kết quả đo lường độ chính xác (Accuracy, F1-Score) phản ánh thực chất năng lực nhận diện loài của mô hình trên các mẫu gỗ chưa từng thấy.
2. **Khả năng tổng quát hóa cao**: Mô hình được huấn luyện dưới các kịch bản phân chia này sẽ có độ bền vững cao khi mang ra ứng dụng thực tế trên các mẫu lâm sản ngoài tự nhiên.
