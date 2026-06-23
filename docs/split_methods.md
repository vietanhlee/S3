# Các Phương Pháp Chia Dữ Liệu Chống Rò Rỉ (Data Leakage) Trong Dự Án S3

Tài liệu này mô tả chi tiết cơ chế hoạt động, ưu/nhược điểm và khả năng chống rò rỉ dữ liệu (data leakage) của các phương pháp phân chia dữ liệu được triển khai trong dự án phân loại ảnh mặt cắt gỗ, sau khi đã nâng cấp toàn bộ sang phân chia ở cấp độ subfolder.

---

## 1. Tổng Quan Về Bài Toán Rò Rỉ Dữ Liệu & Giải Pháp Cấp Độ Subfolder
Trong bài toán phân loại ảnh vi phẫu gỗ, dữ liệu thường có cấu trúc phân cấp (nhiều ảnh chụp từ cùng một mẫu gỗ vật lý được lưu trong cùng một thư mục con `subfolder`).
- **Rò rỉ cấp độ mẫu vật (Sample-level leakage)**: Nếu chia ngẫu nhiên đơn thuần, các ảnh của cùng một mẫu gỗ sẽ phân mảnh sang cả tập Train, Val và Test. Mô hình có thể "ghi nhớ" đặc điểm của mẫu gỗ cụ thể đó thay vì học được đặc trưng chung của loài, dẫn đến độ chính xác trên tập Test ảo (rất cao) nhưng thực tế mô hình không tổng quát hóa tốt.
- **Giải pháp triệt để**: Sử dụng **`subfolder`** làm đơn vị phân chia cơ sở (group isolation) cho mọi phương pháp chống leakage. Các ảnh thuộc cùng một `subfolder` chỉ được phép nằm trong duy nhất một tập (Train, Val hoặc Test).
- **Phân chia xấp xỉ**: Tuyệt đối không chia lẻ ảnh của bất kỳ `subfolder` nào để đạt tỷ lệ $60/20/20$. Chấp nhận tỷ lệ phân chia **xấp xỉ** và chấp nhận tập Val/Test trống đối với các loài có ít subfolders (1 hoặc 2 subfolders), giúp loại bỏ hoàn toàn data leakage.

---

## 2. Chi Tiết Các Phương Pháp Chia Dữ Liệu (Split Methods)

### PP2: Mahalanobis Iterative Centroid Split (Subfolder-level)
* **Cơ chế**: 
  1. Với mỗi lớp (class), gom các ảnh theo `subfolder` và tính mean embedding của mỗi subfolder.
  2. Giảm chiều các subfolder embeddings bằng PCA xuống còn tối đa 32 chiều (nếu số lượng subfolders $\ge 5$).
  3. Tính centroid chung của lớp dựa trên các subfolder embeddings.
  4. Chạy thuật toán Mahalanobis Iterative ở cấp độ subfolder: Lặp lại việc chọn subfolder có khoảng cách Mahalanobis xa nhất so với centroid của các subfolders còn lại đưa vào tập Test (tỷ lệ `test_ratio`), tiếp theo đưa vào Val (tỷ lệ `val_ratio`), phần còn lại đưa vào Train.
  5. Gán toàn bộ ảnh thuộc các subfolders đó vào các tập tương ứng.
* **Mục tiêu chống leakage**: Feature-level stress-test ở cấp độ mẫu vật lý.

### PP3: Group-based Split (Subfolder-level)
* **Cơ chế**: 
  1. Gom các mẫu ảnh theo thư mục con (`subfolder`).
  2. Xáo trộn ngẫu nhiên các subfolders và phân bổ chúng vào các tập Train, Val, Test mà không bao giờ cắt đôi một subfolder.
  3. Nếu lớp chỉ có 1 subfolder $\rightarrow$ Train. Nếu có 2 subfolders $\rightarrow$ 1 Train, 1 Test. Nếu $\ge 3$ subfolders $\rightarrow$ Phân bổ greedy nguyên vẹn.
* **Mục tiêu chống leakage**: Source-level (chống rò rỉ nguồn mẫu vật lý ngẫu nhiên).

### PP4: Hierarchical Clustering Split (Subfolder-level)
* **Cơ chế**:
  1. Gom các ảnh theo `subfolder` và tính mean embedding cho mỗi subfolder.
  2. Thực hiện phân cụm phân cấp (Agglomerative Clustering sử dụng Ward linkage) trên các subfolder embeddings để tự động phát hiện các cụm subfolders có độ tương đồng cao.
  3. Gán nguyên cụm subfolders vào Test, Val, Train để đảm bảo không xé lẻ cụm. Ưu tiên các cụm nằm xa centroid chung nhất đưa vào tập Test.
* **Mục tiêu chống leakage**: Cluster-level (cô lập theo cụm subfolders có đặc trưng tương đồng tự nhiên).

### PP5: Cosine Similarity Graph Split (Subfolder-level)
* **Cơ chế**:
  1. Gom các ảnh theo `subfolder` và tính mean embedding cho mỗi subfolder.
  2. Xây dựng đồ thị tương đồng cosine giữa các subfolders (similarity $\ge 0.92$ sẽ được nối cạnh).
  3. Tìm các thành phần liên thông (connected components) của các subfolders. Mỗi thành phần liên thông đại diện cho một nhóm các subfolders gần như trùng lặp (near-duplicates).
  4. Chia dữ liệu theo components subfolders nguyên vẹn vào Train, Val, Test.
* **Mục tiêu chống leakage**: Graph-level (chống rò rỉ cụm mẫu vật gần như trùng lặp).

### PP6: Stratified Random Split (Baseline)
* **Cơ chế**: 
  1. Phân chia dữ liệu ngẫu nhiên ở cấp độ **ảnh đơn lẻ** nhưng cố gắng duy trì tỷ lệ nhãn lớp đồng nhất giữa các tập Train, Val và Test.
  2. Được tự triển khai thủ công duyệt qua từng lớp để tính toán và chia ngẫu nhiên thay vì sử dụng `train_test_split` của sklearn, giúp loại bỏ hoàn toàn lỗi crash khi gặp các lớp thiểu số chỉ có 1 hoặc 2 mẫu vật.
* **Mục tiêu chống leakage**: Không chống leakage (Baseline dùng để chứng minh tác hại của leakage và so sánh hiệu năng).

### PP7: Adversarial Validation Split (Subfolder-level)
* **Cơ chế**:
  1. Gom các ảnh theo `subfolder` và tính mean embedding cho mỗi subfolder.
  2. Chia các subfolders thành 2 pool ngẫu nhiên tạm thời, huấn luyện một bộ phân loại nhỏ (MLP Discriminator) trên subfolder embeddings để phân biệt 2 pool.
  3. Tính điểm độ khó phân biệt của mỗi subfolder. Những subfolders mà discriminator dễ phân biệt nhất (tự tin nhất là thuộc phân phối khác biệt) sẽ được đưa vào tập Test để tạo ra tập kiểm tra thử thách nhất.
  4. Gán toàn bộ ảnh của các subfolders đó vào các tập tương ứng.
* **Mục tiêu chống leakage**: Distribution-level (stress-test với sự dịch chuyển phân phối ở cấp độ mẫu vật).

### PP8: StratifiedGroupKFold Split
* **Cơ chế**: 
  1. Coi mỗi `subfolder` là một nhóm (group).
  2. Sử dụng thuật toán phân bổ tham lam (greedy) của `StratifiedGroupKFold` từ thư viện sklearn để phân chia các nhóm sao cho:
     - Các ảnh trong cùng một `subfolder` chỉ thuộc về duy nhất một split (Group Isolation).
     - Tỷ lệ các lớp giữa các split vẫn được phân bổ đồng đều nhất có thể (Stratification).
* **Mục tiêu chống leakage**: Cân bằng tối ưu giữa cô lập mẫu vật lý và phân tầng nhãn lớp.

### PP9: Agglom Stratified Split (Subfolder-level)
* **Cơ chế**:
  1. Gom các ảnh theo `subfolder` và tính mean embedding cho mỗi subfolder.
  2. Gom cụm các subfolders bằng thuật toán `AgglomerativeClustering`.
  3. Tính khoảng cách Euclidean từ centroid của mỗi cụm subfolders đến centroid chung.
  4. Sắp xếp các cụm subfolders theo khoảng cách từ gần đến xa, sau đó chia làm 3 dải khoảng cách: Gần (Near), Vừa (Mid), và Xa (Far).
  5. Phân bổ các cụm subfolders từ mỗi dải khoảng cách vào các tập Train, Val, Test theo thuật toán dynamic allocation.
* **Mục tiêu chống leakage**: Phân tầng đặc trưng và khoảng cách (distance-based stratification) cấp độ subfolder.
