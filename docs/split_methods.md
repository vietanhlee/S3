# Các Phương Pháp Chia Dữ Liệu Chống Rò Rỉ (Data Leakage) Trong Dự Án S3

Tài liệu này mô tả chi tiết cơ chế hoạt động, ưu/nhược điểm và khả năng chống rò rỉ dữ liệu (data leakage) của các phương pháp phân chia dữ liệu được triển khai trong dự án phân loại ảnh mặt cắt gỗ.

---

## 1. Tổng Quan Về Bài Toán Rò Rỉ Dữ Liệu
Trong bài toán phân loại ảnh vi phẫu gỗ, dữ liệu thường có cấu trúc phân cấp (ví dụ: nhiều ảnh chụp từ cùng một mẫu gỗ vật lý, được lưu trong cùng một thư mục con `subfolder`).
- **Rò rỉ cấp độ mẫu vật (Sample-level leakage)**: Nếu chia ngẫu nhiên đơn thuần, các ảnh của cùng một mẫu gỗ sẽ phân mảnh sang cả tập Train, Val và Test. Mô hình có thể "ghi nhớ" đặc điểm của mẫu gỗ cụ thể đó thay vì học được đặc trưng của loài, dẫn đến độ chính xác trên tập Test ảo (rất cao) nhưng thực tế mô hình không tổng quát hóa tốt.
- **Rò rỉ ảnh trùng lặp (Near-duplicate leakage)**: Các ảnh chụp liên tiếp của cùng một mẫu gỗ có độ tương đồng đặc trưng cực kỳ cao.

Dự án này triển khai các phương pháp chia dữ liệu từ cơ bản (Baseline) đến nâng cao nhằm kiểm soát rò rỉ dữ liệu ở các cấp độ khác nhau.

---

## 2. Chi Tiết Các Phương Pháp Chia Dữ Liệu (Split Methods)

### PP2: Mahalanobis Iterative Centroid Split
* **Cơ chế**: 
  1. Trích xuất đặc trưng (embedding) của ảnh bằng mô hình EfficientNet-B4.
  2. Với mỗi lớp (class), tính centroid (trung bình) của các embedding.
  3. Lặp lại quá trình: tính khoảng cách Mahalanobis từ các điểm còn lại đến centroid hiện tại, chọn mẫu có khoảng cách lớn nhất (outlier) đưa vào tập Test (hoặc Val), sau đó cập nhật lại centroid. Quá trình tiếp diễn cho đến khi đủ số lượng mẫu quy định cho Test/Val. Phần còn lại đưa vào tập Train.
  4. **Tối ưu hóa hiệu năng bằng PCA**: Sebelum tính khoảng cách Mahalanobis, embeddings của từng lớp được giảm chiều bằng PCA xuống còn tối đa 32 chiều ($d' = \min(n_{samples} - 2, 32)$). Việc này giúp giảm kích thước ma trận covariance, tăng tốc độ tính nghịch đảo ma trận lên hàng nghìn lần, đồng thời loại bỏ nhiễu và đa cộng tuyến.
* **Mục tiêu chống leakage**: Feature-level stress-test.
* **Đặc điểm**: Đẩy các mẫu ngoại lai (outliers) có đặc trưng lệch nhất làm Test set, giúp stress-test mô hình trong điều kiện khó khăn nhất.

### PP3: Group-based Split (Subfolder)
* **Cơ chế**: 
  1. Gom các mẫu ảnh theo thư mục con (`subfolder`), đại diện cho nguồn thu thập vật lý.
  2. Chia toàn bộ các subfolder vào các tập Train, Val, Test mà không bao giờ cắt đôi một subfolder.
  3. **Cơ chế Fallback**: Đối với các loài hiếm có số lượng subfolders `< 3` (hoặc tổng ảnh `< 3`), ta tự động chuyển sang phân chia ngẫu nhiên mức ảnh để tránh tình trạng tập Test hoặc Val bị trống (`support = 0`). Với các loài có `>= 3` subfolders, đảm bảo gán tối thiểu 1 subfolder cho Test và 1 cho Val trước khi phân bổ phần còn lại cho Train.
* **Mục tiêu chống leakage**: Source-level (chống rò rỉ nguồn mẫu vật lý).
* **Đặc điểm**: Đây là phương pháp trực quan và rất hiệu quả để cô lập nguồn dữ liệu vật lý.

### PP4: Hierarchical Clustering Split
* **Cơ chế**:
  1. Thực hiện phân cụm phân cấp (Agglomerative Clustering sử dụng Ward linkage) trên các embedding của từng lớp để tự động phát hiện các cụm ảnh tự nhiên có độ tương đồng cao.
  2. Gán nguyên cụm ảnh vào Test, Val, Train để đảm bảo không xé lẻ cụm. Ưu tiên các cụm nằm xa centroid chung của lớp nhất để đưa vào tập Test.
* **Mục tiêu chống leakage**: Cluster-level (cô lập theo cụm đặc trưng tự nhiên).
* **Đặc điểm**: Phù hợp khi cấu trúc thư mục (`subfolder`) bị thiếu hoặc không phản ánh đúng mức độ tương đồng vật lý của ảnh.

### PP5: Cosine Similarity Graph Split
* **Cơ chế**:
  1. Xây dựng đồ thị tương đồng Cosine (Cosine Similarity Graph) cho các ảnh trong từng lớp. Hai ảnh có độ tương đồng cosine $\ge 0.92$ (mặc định) sẽ được nối với nhau bằng một cạnh.
  2. Tìm các thành phần liên thông (connected components) trên đồ thị. Mỗi thành phần liên thông đại diện cho một nhóm ảnh gần như trùng lặp (near-duplicates).
  3. Chia dữ liệu theo component. Nếu một component quá lớn (chiếm $> 50\%$ tổng số ảnh của lớp), ta cho phép chia cắt component đó để đảm bảo phân tầng; ngược lại, giữ nguyên component đó trong cùng một split.
* **Mục tiêu chống leakage**: Graph-level (chống rò rỉ ảnh gần như trùng lặp).
* **Đặc điểm**: Ngăn chặn triệt để hiện tượng mô hình học thuộc lòng các ảnh gần như giống hệt nhau.

### PP6: Stratified Random Split (Baseline)
* **Cơ chế**: 
  1. Phân chia dữ liệu ngẫu nhiên nhưng cố gắng duy trì tỷ lệ nhãn lớp đồng nhất giữa các tập Train, Val và Test.
  2. Được tự triển khai thủ công duyệt qua từng lớp để tính toán và chia ngẫu nhiên thay vì sử dụng `train_test_split` của sklearn, giúp loại bỏ hoàn toàn lỗi crash khi gặp các lớp thiểu số chỉ có 1 hoặc 2 mẫu vật.
* **Mục tiêu chống leakage**: Không chống leakage (Baseline dùng để so sánh).
* **Đặc điểm**: Thể hiện hiệu năng của mô hình khi không áp dụng bất kỳ cơ chế kiểm soát rò rỉ dữ liệu nào.

### PP7: Adversarial Validation Split
* **Cơ chế**:
  1. Huấn luyện một bộ phân loại nhỏ (MLP Discriminator) để phân biệt giữa các phân đoạn dữ liệu ngẫu nhiên.
  2. Mô hình MLP sẽ chấm điểm mức độ "khác biệt" của từng ảnh.
  3. Những ảnh mà mô hình phân biệt tốt nhất (tự tin nhất là thuộc phân phối khác biệt) sẽ được đưa vào tập Test để tạo ra một tập kiểm tra cực kỳ thử thách.
* **Mục tiêu chống leakage**: Distribution-level (stress-test với sự lệch phân phối).
* **Đặc điểm**: Đánh giá độ bền bỉ của mô hình trước hiện tượng dịch chuyển phân phối dữ liệu (distribution shift).

### PP8: StratifiedGroupKFold Split
* **Cơ chế**: 
  1. Coi mỗi `subfolder` là một nhóm (group).
  2. Sử dụng thuật toán phân bổ tham lam (greedy) của `StratifiedGroupKFold` từ thư viện sklearn để phân chia các nhóm sao cho:
     - Các ảnh trong cùng một `subfolder` chỉ thuộc về duy nhất một split (Group Isolation).
     - Tỷ lệ các lớp giữa các split vẫn được phân bổ đồng đều nhất có thể (Stratification).
* **Mục tiêu chống leakage**: Vừa chống rò rỉ nhóm, vừa phân tầng nhãn lớp.
* **Đặc điểm**: Đây là **tiêu chuẩn vàng** đối với dữ liệu vi phẫu gỗ, cân bằng hoàn hảo giữa việc cô lập mẫu vật lý và giải quyết bài toán mất cân bằng lớp.

### PP9: Agglom Stratified Split
* **Cơ chế**:
  1. Gom cụm ảnh của từng lớp bằng thuật toán `AgglomerativeClustering`.
  2. Tính khoảng cách Euclidean từ centroid của mỗi cụm đến centroid chung của lớp.
  3. Sắp xếp các cụm theo khoảng cách từ gần đến xa, sau đó chia làm 3 dải khoảng cách:
     - **Gần** (Near): Các mẫu điển hình nằm sát tâm của lớp.
     - **Vừa** (Mid): Các mẫu trung bình.
     - **Xa** (Far): Các mẫu tiệm cận ngoại lai (outliers).
  4. Phân bổ các cụm từ mỗi dải khoảng cách vào các tập Train, Val, Test theo tỷ lệ mong muốn thông qua thuật toán phân bổ thiếu hụt động (dynamic deficiency allocation).
* **Mục tiêu chống leakage**: Phân tầng đặc trưng và khoảng cách (distance-based stratification).
* **Đặc điểm**: Đảm bảo cả 3 tập split đều nhận được các mẫu đại diện phân phối đồng đều ở cả 3 dải (gần, vừa, xa), giúp tập Test phản ánh chính xác hiệu năng của mô hình trên toàn dải phân phối của loài.
