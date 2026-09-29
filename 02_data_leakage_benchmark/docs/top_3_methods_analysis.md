# Phân Tích Chuyên Sâu: 3 Phương Pháp Tốt Nhất Cho Nhận Diện Mặt Cắt Gỗ (S3 Dataset)

Tài liệu này phân tích và đánh giá 3 phương pháp Metric Learning được nhận định là **tốt nhất và phù hợp nhất** đối với đặc thù cấu trúc ảnh macro vân gỗ của bộ dữ liệu S3 (19 loài, 6 chi). 

---

## Tóm Tắt Đặc Thù Thách Thức Của Bộ Dữ Liệu S3
1.  **Tính chất Fine-Grained (Texture Vi Mô):** Đặc trưng phân biệt chủ yếu là cấu trúc lỗ mạch, tia gỗ, và thớ sợi gỗ. Các đặc trưng này rất nhỏ và tinh vi.
2.  **Độ tương đồng liên lớp cao (High Inter-class Similarity):** Các loài trong cùng một chi (ví dụ chi *Dalbergia* - Gỗ Trắc/Sưa hoặc chi *Pterocarpus* - Gỗ Giáng Hương) rất khó phân biệt bằng mắt thường hoặc mô hình phân loại chuẩn.
3.  **Biến động nội lớp lớn (High Intra-class Variation):** Vân gỗ thay đổi mạnh mẽ theo tuổi cây, vị trí cắt gỗ (gỗ giác - sapwood bên ngoài thường nhạt màu và ít vân vs gỗ lõi - heartwood bên trong sẫm màu và vân dày đặc).

---

## 3 Phương Pháp Tối Ưu Nhất Được Lựa Chọn

Dựa trên phân tích toán học và thực nghiệm DML (Deep Metric Learning), 3 phương pháp sau đây là tối ưu nhất để giải quyết triệt để các thách thức trên:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        3 PHƯƠNG PHÁP TỐT NHẤT                          │
├──────────────────────────┬──────────────────────────┬──────────────────┤
│   SubCenter ArcFace      │     SoftTriple Loss      │  Multi-Similarity│
│ (Multi-modal + Margin)   │ (Soft Multi-center proxy)│  (Dynamic Pairs) │
└──────────────────────────┴──────────────────────────┴──────────────────┘
```

---

### 1. SubCenter ArcFace (Additive Angular Margin với K Sub-Centers)
*Tệp triển khai:* [train_subcenter_arcface.py](file:///g:/S3_paper/train_subcenter_arcface.py)

#### A. Bản chất thuật toán
SubCenter ArcFace mở rộng thuật toán ArcFace bằng cách gán $K$ vector trọng số (sub-centers) cho mỗi lớp thay vì chỉ $1$. Khi một mẫu ảnh gỗ $x_i$ được đưa vào, mô hình tính cosine similarity với toàn bộ các sub-centers và chỉ chọn sub-center gần nhất để áp dụng biên độ góc geodesic $m$:

$$\cos\theta_{y_i} = \max_{k=1}^{K} \frac{x_i^T W_{y_i}^{(k)}}{\|x_i\|_2 \|W_{y_i}^{(k)}\|_2}$$

#### B. Tại sao cực kỳ phù hợp với ảnh gỗ S3?
*   **Giải quyết triệt để biến động sapwood/heartwood:** Phân bố vân gỗ của một loài thực chất là **đa đỉnh (multi-modal)**. SubCenter cho phép mô hình gán riêng 1 sub-center học cấu trúc vân gỗ giác (sapwood), 1 sub-center học cấu trúc vân gỗ lõi (heartwood), và 1 sub-center học vùng chuyển tiếp.
*   **Biên phân tách geodesic mạnh mẽ:** Thừa hưởng biên độ góc $m$ của ArcFace giúp nén các cụm loài gỗ lại cực kỳ chặt chẽ trên mặt cầu đơn vị, giúp phân biệt rõ nét các loài cùng chi có cấu trúc tương đồng.

#### C. Cấu hình khuyến nghị
*   `ARCFACE_SCALE (s) = 30.0`
*   `ARCFACE_MARGIN (m) = 0.50`
*   `NUM_SUBCENTERS (K) = 3` (đại diện cho: Gỗ Giác, Gỗ Lõi và Vùng trung gian/nhiễu).

---

### 2. SoftTriple Loss (Tích Hợp Nhiều Center với Gán Mềm)
*Tệp triển khai:* [train_soft_triple.py](file:///g:/S3_paper/train_soft_triple.py)

#### A. Bản chất thuật toán
Tương tự như SubCenter ArcFace, SoftTriple sử dụng $K$ centers cho mỗi lớp để học phân bố đa đỉnh. Tuy nhiên, thay vì gán cứng (chỉ lấy max), SoftTriple sử dụng cơ chế **gán mềm (soft assignment)** thông qua hàm softmax có kiểm soát nhiệt độ $\gamma$:

$$S_{i,c} = \sum_{k=1}^{K} \frac{\exp\left( \frac{1}{\gamma} x_i^T w_c^{(k)} \right)}{\sum_{t=1}^{K} \exp\left( \frac{1}{\gamma} x_i^T w_c^{(t)} \right)} \cdot x_i^T w_c^{(k)}$$

#### B. Tại sao cực kỳ phù hợp với ảnh gỗ S3?
*   **Ngăn ngừa hiện tượng "center chết" (Dead Centers):** Trong các bộ dữ liệu nhỏ như S3, việc chọn cứng `max` của SubCenter ArcFace đôi khi khiến một số sub-center không bao giờ được chọn và cập nhật gradient. SoftTriple phân bổ trọng số mềm lên tất cả các centers, giúp toàn bộ các vector đại diện đều được học một cách trơn tru.
*   **Học cấu trúc con tự động:** SoftTriple tự động phân bổ các ảnh gỗ có cấu trúc thớ tương tự nhau về chung một center phụ mà không cần chúng ta phải dán nhãn thủ công ảnh đó là sapwood hay heartwood.

#### C. Cấu hình khuyến nghị
*   `NUM_CENTERS (K) = 10` (Cho phép học nhiều biến thể thớ gỗ vi mô phức tạp).
*   `SOFTTRIPLE_LAMBDA (la) = 20.0`
*   `SOFTTRIPLE_GAMMA = 0.1`
*   `SOFTTRIPLE_TAU (margin) = 0.2`

---

### 3. Multi-Similarity (MS) Loss (Trọng Số Cặp Động)
*Tệp triển khai:* [train_multi_similarity.py](file:///g:/S3_paper/train_multi_similarity.py)

#### A. Bản chất thuật toán
Khác với 2 phương pháp trên (dựa trên phân loại/proxy), MS Loss hoạt động trên quan hệ **ảnh-ảnh (pair-based)**. Nó đồng thời phân tích 3 loại tương đồng: tương đồng bản thân (self-similarity), tương đồng tương đối (relative-similarity), và tương đồng tương hỗ (mutual-similarity) để gán trọng số gradient động cho các cặp mẫu.

#### B. Tại sao cực kỳ phù hợp với ảnh gỗ S3?
*   **Tập trung vào các cặp khó nhất (Hard Negative Mining):** Trong 19 loài gỗ, có những loài rất dễ phân biệt (khác chi như *Afzelia* vs *Sindora*) và những loài cực kỳ khó (cùng chi như *Dalbergia oliveri* vs *Dalbergia tonkinensis*). 
    *   MS Loss sẽ tự động triệt tiêu gradient của các cặp dễ (giảm thiểu overfitting).
    *   Nó sẽ tăng lũy thừa gradient của các cặp thớ gỗ cực kỳ giống nhau ở vùng biên ranh giới chi gỗ, ép mô hình phải tìm ra các đặc trưng vi mô sâu nhất để tách chúng ra.

#### C. Cấu hình khuyến nghị
*   `MS_ALPHA = 2.0` (Điều tiết lực kéo positive).
*   `MS_BETA = 50.0` (Đẩy negative mạnh mẽ).
*   `MS_MARGIN = 0.5`

---

## Bảng So Sánh Chiến Thuật Giữa 3 Phương Pháp

| Tiêu chí | SubCenter ArcFace | SoftTriple Loss | Multi-Similarity Loss |
| :--- | :--- | :--- | :--- |
| **Dạng tiếp cận** | Phân loại có Margin (Classification) | Proxy-based có gán mềm | So sánh cặp ảnh (Pair-based) |
| **Xử lý sapwood/heartwood** | Rất tốt (K sub-centers gán cứng) | Xuất sắc (K centers gán mềm trơn) | Trung bình (phụ thuộc vào Batch size) |
| **Độ nhạy ranh giới chi gỗ** | Rất mạnh (Angular Margin) | Mạnh (Logit Margin) | Cực mạnh (Dynamic Hard Negative) |
| **Yêu cầu Batch size** | Thấp/Vừa (Không nhạy cảm) | Thấp/Vừa (Không nhạy cảm) | Cao (Cần Batch lớn hoặc PK Sampler) |
| **Độ ổn định huấn luyện** | Rất cao | Rất cao | Khá (Cần kiểm soát learning rate kỹ) |
| **Mục tiêu ưu tiên** | Gom cụm chặt chẽ trên mặt cầu | Biểu diễn đa dạng thớ gỗ con | Tối ưu hóa ranh giới các loài khó |
