# Phân Tích Chuyên Sâu: 3 Phương Pháp Học Tự Giám Sát (SSL) Tốt Nhất Cho Ảnh Mặt Cắt Gỗ S3

Tài liệu này phân tích chi tiết về **3 phương pháp Học tự giám sát (Self-Supervised Learning - SSL)** tối ưu nhất cho bài toán học biểu diễn đặc trưng không nhãn từ ảnh macro mặt cắt gỗ của bộ dữ liệu S3.

---

## 3 Phương Pháp SSL Phù Hợp Nhất Cho Đặc Thù Vân Gỗ S3

Để học được biểu diễn đặc trưng của vân gỗ vi mô mà không phụ thuộc vào nhãn, mô hình cần tránh hiện tượng sụp đổ đặc trưng (representation collapse). Trong số các thuật toán SSL SOTA, 3 phương pháp sau đây là phù hợp nhất:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        3 PHƯƠNG PHÁP SSL TỐT NHẤT                      │
├──────────────────────────┬──────────────────────────┬──────────────────┤
│      Barlow Twins        │          BYOL            │     SimSiam      │
│  (Redundancy Reduction)  │    (Bootstrap Latent)    │ (Stop-gradient)  │
└──────────────────────────┴──────────────────────────┴──────────────────┘
```

---

### 1. Barlow Twins (Giảm Thiểu Tính Dư Thừa Thông Tin - ICML 2021)
*Tệp triển khai:* [train_barlow_twins.py](file:///g:/S3_paper/train_barlow_twins.py)

#### A. Cơ chế hoạt động
Barlow Twins áp dụng nguyên lý giảm thiểu thông tin dư thừa của nhà thần kinh học Horace Barlow. Thay vì so sánh khoảng cách giữa các mẫu, nó tính **ma trận tương quan chéo (cross-correlation matrix)** giữa các đặc trưng đầu ra của 2 view ảnh đã được chuẩn hóa theo batch, sau đó ép ma trận này tiến về ma trận đơn vị (Identity Matrix):
*   Đường chéo chính bằng $1$: Kéo biểu diễn của 2 view từ cùng một ảnh lại gần nhau (Invariance).
*   Đường chéo phụ bằng $0$: Triệt tiêu sự tương quan chéo giữa các chiều đặc trưng khác nhau (Redundancy Reduction).

#### B. Tại sao phù hợp với vân gỗ S3?
*   **Học đặc trưng đa dạng và độc lập:** Các vân gỗ macro có cấu trúc hình học phức tạp. Bằng cách ép các chiều của đặc trưng nhúng độc lập với nhau (đường chéo phụ = 0), Barlow Twins bắt buộc mô hình phải phân bổ tài nguyên để học nhiều đặc điểm thớ gỗ khác nhau (độ xốp, kích thước mạch, hướng thớ), thay vì chỉ tập trung vào một vài đặc trưng dominant dễ thấy (như màu sắc, độ sáng).
*   **Không nhạy cảm với Batch size:** Không cần mẫu âm (negative samples) nên không bị ảnh hưởng tiêu cực khi Batch size nhỏ.

---

### 2. BYOL (Bootstrap Your Own Latent - NeurIPS 2020)
*Tệp triển khai:* [train_byol.py](file:///g:/S3_paper/train_byol.py)

#### A. Cơ chế hoạt động
BYOL loại bỏ hoàn toàn việc sử dụng mẫu âm (negative samples). Nó sử dụng hai mạng: **Online** (Backbone + Projector + Predictor) và **Target** (Backbone + Projector). Mạng Online được huấn luyện để dự đoán chính xác biểu diễn đặc trưng của mạng Target. Mạng Target không tính gradient mà được cập nhật chậm bằng trung bình trượt lũy thừa (EMA) từ trọng số của mạng Online:

$$\theta_{target} \leftarrow \tau \theta_{target} + (1 - \tau) \theta_{online}$$

#### B. Tại sao phù hợp với vân gỗ S3?
*   **Loại bỏ hiện tượng "Đẩy nhầm mẫu âm" (False Negative Repulsion):** Trong ảnh thớ gỗ S3, hai loài khác nhau (ví dụ *Dalbergia oliveri* và *Dalbergia tonkinensis*) có kết cấu vân gỗ vi mô cực kỳ giống nhau. 
    *   Trong các loss contrastive dùng mẫu âm như SimCLR, mô hình sẽ coi chúng là negative và cố tình đẩy chúng ra xa nhau, gây rối loạn không gian nhúng của thớ gỗ.
    *   BYOL chỉ tập trung kéo gần các bản biến đổi của cùng một ảnh và tự nâng cấp biểu diễn đặc trưng thông qua mạng Target, giúp tránh hoàn toàn hiện tượng đẩy nhầm này.

---

### 3. SimSiam (Simple Siamese - CVPR 2021)
*Tệp triển khai:* [train_simsiam.py](file:///g:/S3_paper/train_simsiam.py)

#### A. Cơ chế hoạt động
SimSiam được coi là phiên bản tối giản hóa của BYOL. Nó sử dụng cấu trúc mạng Siamese đối xứng chia sẻ trọng số hoàn toàn, không sử dụng mẫu âm và **không sử dụng Target Network cập nhật EMA**. Để ngăn ngừa sụp đổ biểu diễn, SimSiam chỉ sử dụng toán tử **ngắt gradient (Stop-Gradient)** ở một nhánh khi tính loss đối xứng chéo:

$$\mathcal{L}_{SimSiam} = \frac{1}{2} \mathcal{D}(p_1, \text{stop\_gradient}(z_2)) + \frac{1}{2} \mathcal{D}(p_2, \text{stop\_gradient}(z_1))$$

#### B. Tại sao phù hợp với vân gỗ S3?
*   **Kiến trúc siêu tối giản, hội tụ nhanh:** SimSiam không cần duy trì một target model trong bộ nhớ giúp tiết kiệm đáng kể VRAM GPU (phù hợp khi chạy trên các dòng card phổ thông hoặc Kaggle T4).
*   **Học đặc trưng vân gỗ tự nhiên:** Bằng cách tối đa hóa cosine similarity giữa prediction của view này và projection của view kia thông qua toán tử stop-gradient, mô hình học được các đặc trưng bất biến cấu trúc hình học của vân gỗ trước các phép xoay, cắt hoặc thay đổi độ sáng.

---

## Bảng So Sánh Chiến Thuật Giữa 3 Phương Pháp SSL

| Tiêu chí | Barlow Twins | BYOL | SimSiam |
| :--- | :--- | :--- | :--- |
| **Cơ chế chống sụp đổ** | Khử tương quan chéo đặc trưng | Mạng Target EMA + Predictor | Toán tử Stop-Gradient |
| **Sử dụng mẫu âm (Negatives)**| Không | Không | Không |
| **Yêu cầu VRAM GPU** | Vừa phải | Cao nhất (Duy trì 2 model) | Thấp nhất (1 model chia sẻ trọng số) |
| **Tính chất đặc trưng học được** | Độc lập, chống trùng lặp thông tin | Đặc trưng bất biến mức cao | Đặc trưng hình học mịn |
| **Độ nhạy siêu tham số** | Thấp | Khá cao (Hệ số EMA $\tau$) | Rất thấp (Dễ huấn luyện) |
| **Mức độ phù hợp với S3** | ⭐⭐⭐⭐⭐ (Khai thác cấu trúc vân đa chiều) | ⭐⭐⭐⭐⭐ (Tránh đẩy nhầm loài tương đồng) | ⭐⭐⭐⭐ (Nhẹ nhàng, tiết kiệm tài nguyên) |

---

## So Sánh Với SimCLR (Tại sao SimCLR xếp sau?)
Mặc dù **SimCLR** (`train_simclr.py`) là một phương pháp contrastive SSL rất mạnh, nhưng đối với bài toán ảnh macro gỗ S3, nó có một số hạn chế:
1.  **Nhạy cảm lớn với Batch size:** SimCLR bắt buộc phải dùng Batch size cực lớn (thường $\geq 256$ hoặc $512$) để có đủ lượng mẫu âm làm phong phú mẫu số của hàm InfoNCE. Với tập dữ liệu nhỏ và tài nguyên giới hạn, batch size nhỏ sẽ làm giảm nghiêm trọng độ chính xác.
2.  **Vấn đề False Negatives:** Như đã phân tích, SimCLR sẽ vô tình đẩy các mẫu gỗ của loài khác nhau nhưng có vân giống nhau ra xa, làm nứt gãy tính liên tục của cấu trúc vân gỗ trong không gian nhúng.
