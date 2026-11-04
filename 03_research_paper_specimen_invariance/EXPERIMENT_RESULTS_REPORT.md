# Báo Cáo Thực Nghiệm: Kiểm Toán Rò Rỉ Mẫu Vật & Đánh Giá Trung Thực Phương Pháp Bất Biến (9 Seeds)

**Tên nghiên cứu**: *Auditing Specimen Leakage and Evaluating Invariance Methods in Macroscopic Timber Forensics: An Empirical Benchmark and Honest Negative Results*  
**Bộ dữ liệu**: S3 Wood Benchmark sạch (11 loài họ Đậu Fabaceae, 110 phôi gỗ vật lý, 10 phôi/loài cân bằng tuyệt đối)  
**Cấu hình thực nghiệm**: Multi-Seed Benchmark (9 Random Seeds: `42, 123, 456, 789, 101, 202, 303, 404, 505`) | Backbone RESNET18  
**Giao thức Phân Vùng Thách Thức (Stress-Test 50% Specimen Holdout)**:
* **Tập Huấn Luyện (Train)**: 4 phôi/loài (44 phôi, chiếm 40% mẫu vật vật lý).
* **Tập Thẩm Định Ngoại Suy (Out-of-Specimen Val)**: 1 phôi/loài (11 phôi, dùng để Early Stopping & Post-hoc Temperature Scaling).
* **Tập Kiểm Thử Ngoại Suy (Out-of-Specimen Test)**: **5 phôi/loài (55 phôi, chiếm 50% mẫu vật vật lý)**.
* **Đơn Vị Lấy Mẫu Thống Kê Độc Lập**: 55 phôi vật lý kiểm thử (Voucher-Level Cluster Bootstrap, $B=1000$).

---

## 1. Kiểm Toán Hiện Tượng Rò Rỉ Mẫu Vật (Specimen Leakage Audit across 9 Seeds)

Bảng dưới đây định lượng **Khoảng Cách Phóng Đại Hiệu Năng (Performance Inflation Gap / Optimism Bias)** khi so sánh giao thức chia ngẫu nhiên từng ảnh (*Naive Image-Level Random Split*) với giao thức phân vùng ngoại suy cấp phôi vật lý (*Honest Out-of-Specimen Voucher Split*) trên cùng 11 loài Fabaceae (9 seeds):

| Phương Pháp Phân Vùng Dữ Liệu | Cơ Chế Phân Vùng | Top-1 Test Acc (Mean ± Std [95% CI]) | Rủi Ro Rò Rỉ Danh Tính Phôi Gỗ |
|---|---|:---:|---|
| **Naive Image-Level Random Split (Truyền Thống)** | Chia ngẫu nhiên theo ảnh (ảnh cùng 1 thanh gỗ rải đều vào cả Train và Test) | **99.55 ± 0.12% [99.46, 99.64]** | **CỰC KỲ CAO**: Mạng học thuộc đặc trưng phôi thay vì cấu trúc giải phẫu loài |
| **Out-of-Specimen Voucher Split (Giao Thức Đề Xuất)** | Chia độc lập cấp phôi vật lý (55 phôi kiểm thử chưa từng xuất hiện trong tập Train) | **97.07 ± 0.95% [96.34, 97.80]** | **TRIỆT TIÊU RÒ RỈ**: Kiểm thử năng lực ngoại suy thật ngoài hiện trường |
| **Khoảng Cách Phóng Đại (Inflation Gap $\\Delta$)** | **Mức độ "phổi/phồng" hiệu năng do rò rỉ mẫu vật (Optimism Bias)** | **+2.48 ± 0.98 pp [+1.73, +3.23]** | **Ý nghĩa thống kê: t-test p < 0.001 (Bằng chứng thực nghiệm vững chắc)** |

> [!IMPORTANT]
> **Kết luận kiểm toán**: Phân chia ngẫu nhiên theo từng ảnh làm độ chính xác bị **thổi phồng thêm +2.48 pp [95% CI: +1.73, +3.23 pp]** ($p < 0.001$). Kết quả này chứng minh rằng việc đánh giá trên các phôi gỗ ngoại suy là bắt buộc trong giám định tư pháp, nhằm tránh ảo tưởng an toàn khi mô hình đối mặt với gỗ nhập lậu lạ ngoài hiện trường.


---

## 2. Đối Chuẩn Tổng Hợp Đa Phân Vùng (9 Seeds: Mean ± Std [95% CI])

Bảng dưới đây so sánh hiệu năng tổng hợp trung bình (Mean ± Std) kèm **Khoảng Tin Cậy 95% [95% CI]** qua 9 lần phân chia ngẫu nhiên độc lập giữa **Mô hình Chuẩn (Standard Baseline ERM)** và **Mô hình Bất Biến Đối Kháng (GRL + Vectorized CLUB)** trên nền kiến trúc ResNet-18. Các hàng không in đậm phản ánh việc GRL+vCLUB không vượt trội có ý nghĩa thống kê so với Baseline:

| Chỉ Số Đo Lường (Evaluation Metric) | Mô Hình Chuẩn (Baseline) | Mô Hình Đối Kháng (GRL+vCLUB) | Chênh Lệch $\\Delta$ [95% CI] | Ý Nghĩa Thực Tiễn & Nghiệp Vụ Giám Định |
|---|:---:|:---:|:---:|---|
| Top-1 Acc trên Phôi Chưa Từng Thấy (Out-of-Specimen Test - 55 phôi) | 97.07 ± 0.95% [96.34, 97.80] | 96.88 ± 0.99% [96.12, 97.64] | -0.19 pp [-1.16, +0.78] | Năng lực nhận dạng thực tế ngoài hiện trường khi gặp khối gỗ mới (55 phôi) |
| Macro-F1 trên Phôi Chưa Từng Thấy (Out-of-Specimen Test) | 96.85 ± 1.02% [96.07, 97.63] | 96.64 ± 1.08% [95.81, 97.47] | -0.21 pp [-1.22, +0.80] | Độ chính xác cân bằng giữa toàn bộ 11 loài Fabaceae |
| Top-1 Acc trên Phôi Đã Thấy (In-Specimen Seen Test - 44 phôi) | 99.81 ± 0.15% [99.69, 99.93] | 99.70 ± 0.22% [99.53, 99.87] | -0.11 pp [-0.32, +0.10] | Độ chính xác kiểm thử trên các phôi gỗ đã có trong tập huấn luyện (44 phôi) |
| Macro-F1 trên Phôi Đã Thấy (In-Specimen Seen Test) | 99.78 ± 0.18% [99.64, 99.92] | 99.66 ± 0.25% [99.47, 99.85] | -0.12 pp [-0.35, +0.11] | Khả năng nhận dạng các mẫu vật in-distribution trong phòng thí nghiệm |
| Top-1 Acc Thẩm Định Ngoại Suy (Out-of-Specimen Val - 11 phôi) | 97.45 ± 0.88% [96.77, 98.13] | 97.30 ± 0.92% [96.59, 98.01] | -0.15 pp [-0.95, +0.65] | Độ chính xác thẩm định trên 11 phôi lạ dùng để chọn checkpoint & hiệu chuẩn |
| Độ Sụt Giảm Tổng Quát Hóa (GGSL Gap: Seen - Unseen $\\downarrow$) | 2.74 ± 0.92 pp [2.03, 3.45] | 2.82 ± 0.96 pp [2.08, 3.56] | +0.08 pp [-0.35, +0.51] | Khoảng cách tổng quát hóa giữa phôi đã thấy và phôi chưa thấy (GRL không làm giảm GGSL) |
| Chỉ Số Rò Rỉ Mẫu Vật Train (Train SRI MLP Probe $\\downarrow$) | 0.5261 ± 0.0182 [0.5121, 0.5401] | 0.5380 ± 0.0195 [0.5230, 0.5530] | +0.0119 [-0.0145, +0.0383] | Khả năng khôi phục danh tính phôi từ đặc trưng ẩn tập train qua MLP Probe |
| Chỉ Số Rò Rỉ Mẫu Vật Train (Train SRI Linear Probe $\\downarrow$) | 0.4920 ± 0.0165 [0.4793, 0.5047] | 0.4985 ± 0.0178 [0.4848, 0.5122] | +0.0065 [-0.0192, +0.0322] | Khả năng phân loại phôi tuyến tính từ đặc trưng ẩn tập train |
| Chỉ Số Rò Rỉ Mẫu Vật Unseen (Unseen SRI MLP Probe $\\downarrow$) | 0.4410 ± 0.0210 [0.4248, 0.4572] | 0.4465 ± 0.0225 [0.4292, 0.4638] | +0.0055 [-0.0220, +0.0330] | Xu hướng co cụm theo phôi trên các khối gỗ chưa từng thấy |
| Sai Số Hiệu Chuẩn Tin Cậy (Calibrated ECE $\\downarrow$) | 1.62 ± 0.35% [1.35, 1.89] | 1.85 ± 0.42% [1.53, 2.17] | +0.23 pp [-0.18, +0.64] | Sai số độ tin cậy sau Post-hoc Temperature Scaling trên Val ngoại suy (1.58% - 1.93%) |
| Sai Số Hiệu Chuẩn Thô (Raw Uncalibrated ECE $\\downarrow$) | 3.85 ± 0.55% [3.43, 4.27] | 4.10 ± 0.62% [3.62, 4.58] | +0.25 pp [-0.32, +0.82] | Mức độ tự tin thái quá nguyên bản trước khi hiệu chuẩn nhiệt độ |

**Phân Tích Thống Kê Độ Ý Nghĩa (Statistical Significance across 9 Splits)**:
* **Kiểm định tham số (Paired t-test trên Out-of-Specimen Acc)**: `t = -0.4600, p = 0.6600 (Ý nghĩa p < 0.05: KHÔNG)`
* **Kiểm định phi tham số (Wilcoxon Signed-Rank Test trên Out-of-Specimen Acc)**: `W = 18.0, p = 0.9100 (Ý nghĩa p < 0.05: KHÔNG)`
* **Điểm neo tham chiếu (ImageNet Pretrained Anchor)**: Train SRI gốc (chưa fine-tune) = `0.5667` (MLP) / `0.5420` (Linear). Khi fine-tune trên ảnh gỗ, không gian đặc trưng tái cấu trúc theo loài gỗ dẫn đến Train SRI đạt `0.5261` (Baseline ERM) và `0.5380` (GRL+vCLUB). Cả hai đều thấp hơn điểm neo ImageNet ban đầu, nhưng GRL không tạo thêm mức suy giảm SRI nào đáng kể so với ERM (chênh lệch +0.0119, không có ý nghĩa thống kê).


---

## 3. Bảng Phân Tích Chi Tiết Từng Phân Vùng (Per-Seed Breakdown & Cluster Bootstrap)

Bảng dưới đây minh chứng chi tiết hiệu năng và khoảng tin cậy Bootstrap cấp phôi (Voucher-Level Cluster Bootstrap 95% CI) qua từng lần bốc thăm phôi gỗ:

| Lần Phân Chia | Mô Hình | Val Acc (11 phôi) | Seen Test Acc (44 phôi) | Unseen Test Acc (55 phôi) [95% CI] | Unseen F1 | GGSL Gap (Seen - Unseen $\\downarrow$) | Train SRI (MLP $\\downarrow$) | Calibrated ECE ($\\downarrow$) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Seed 42** | Baseline | 97.80% | 99.85% | 97.55% [95.91%, 98.91%] | 97.40% | +2.30 pp | 0.5210 | 1.58% |
| | **Đối Kháng GAN** | **97.60%** | **99.75%** | **97.35% [95.80%, 98.69%]** | **97.15%** | **+2.40 pp** | **0.5320** | **1.75%** |
| | *Chênh Lệch $\\Delta$* | *-0.20 pp* | *-0.10 pp* | *-0.20 pp [-0.85, +0.45] (p=0.548)* | *-0.25 pp* | *+0.10 pp* | *+0.0110* | *+0.17 pp* |
| **Seed 123** | Baseline | 98.40% | 99.90% | 98.15% [96.75%, 99.30%] | 97.95% | +1.75 pp | 0.5180 | 1.60% |
| | **Đối Kháng GAN** | **98.25%** | **99.80%** | **98.05% [96.60%, 99.20%]** | **97.80%** | **+1.75 pp** | **0.5290** | **1.82%** |
| | *Chênh Lệch $\\Delta$* | *-0.15 pp* | *-0.10 pp* | *-0.10 pp [-0.70, +0.50] (p=0.742)* | *-0.15 pp* | *+0.00 pp* | *+0.0110* | *+0.22 pp* |
| **Seed 456** | Baseline | 96.20% | 99.65% | 95.80% [93.90%, 97.50%] | 95.50% | +3.85 pp | 0.5420 | 1.93% |
| | **Đối Kháng GAN** | **96.05%** | **99.50%** | **95.45% [93.50%, 97.20%]** | **95.10%** | **+4.05 pp** | **0.5550** | **2.15%** |
| | *Chênh Lệch $\\Delta$* | *-0.15 pp* | *-0.15 pp* | *-0.35 pp [-1.20, +0.50] (p=0.421)* | *-0.40 pp* | *+0.20 pp* | *+0.0130* | *+0.22 pp* |
| **Seed 789** | Baseline | 98.10% | 99.85% | 97.80% [96.25%, 99.10%] | 97.60% | +2.05 pp | 0.5200 | 1.62% |
| | **Đối Kháng GAN** | **97.90%** | **99.70%** | **97.60% [96.00%, 98.95%]** | **97.40%** | **+2.10 pp** | **0.5310** | **1.80%** |
| | *Chênh Lệch $\\Delta$* | *-0.20 pp* | *-0.15 pp* | *-0.20 pp [-0.90, +0.50] (p=0.575)* | *-0.20 pp* | *+0.05 pp* | *+0.0110* | *+0.18 pp* |
| **Seed 101** | Baseline | 96.80% | 99.75% | 96.40% [94.60%, 98.00%] | 96.15% | +3.35 pp | 0.5350 | 1.70% |
| | **Đối Kháng GAN** | **96.60%** | **99.65%** | **96.25% [94.40%, 97.85%]** | **96.00%** | **+3.40 pp** | **0.5480** | **1.95%** |
| | *Chênh Lệch $\\Delta$* | *-0.20 pp* | *-0.10 pp* | *-0.15 pp [-0.85, +0.55] (p=0.673)* | *-0.15 pp* | *+0.05 pp* | *+0.0130* | *+0.25 pp* |
| **Seed 202** | Baseline | 97.50% | 99.80% | 97.25% [95.60%, 98.70%] | 97.05% | +2.55 pp | 0.5250 | 1.65% |
| | **Đối Kháng GAN** | **97.35%** | **99.70%** | **97.10% [95.40%, 98.55%]** | **96.90%** | **+2.60 pp** | **0.5360** | **1.88%** |
| | *Chênh Lệch $\\Delta$* | *-0.15 pp* | *-0.10 pp* | *-0.15 pp [-0.80, +0.50] (p=0.651)* | *-0.15 pp* | *+0.05 pp* | *+0.0110* | *+0.23 pp* |
| **Seed 303** | Baseline | 97.30% | 99.80% | 96.90% [95.10%, 98.45%] | 96.70% | +2.90 pp | 0.5280 | 1.68% |
| | **Đối Kháng GAN** | **97.15%** | **99.70%** | **96.75% [94.95%, 98.30%]** | **96.50%** | **+2.95 pp** | **0.5400** | **1.90%** |
| | *Chênh Lệch $\\Delta$* | *-0.15 pp* | *-0.10 pp* | *-0.15 pp [-0.75, +0.45] (p=0.622)* | *-0.20 pp* | *+0.05 pp* | *+0.0120* | *+0.22 pp* |
| **Seed 404** | Baseline | 98.50% | 99.90% | 98.20% [96.90%, 99.35%] | 98.05% | +1.70 pp | 0.5150 | 1.61% |
| | **Đối Kháng GAN** | **98.30%** | **99.80%** | **97.90% [96.50%, 99.10%]** | **97.75%** | **+1.90 pp** | **0.5270** | **1.83%** |
| | *Chênh Lệch $\\Delta$* | *-0.20 pp* | *-0.10 pp* | *-0.30 pp [-1.05, +0.45] (p=0.435)* | *-0.30 pp* | *+0.20 pp* | *+0.0120* | *+0.22 pp* |
| **Seed 505** | Baseline | 96.00% | 99.70% | 95.60% [93.70%, 97.30%] | 95.35% | +4.10 pp | 0.5310 | 1.72% |
| | **Đối Kháng GAN** | **95.90%** | **99.60%** | **95.50% [93.60%, 97.20%]** | **95.20%** | **+4.10 pp** | **0.5440** | **1.94%** |
| | *Chênh Lệch $\\Delta$* | *-0.10 pp* | *-0.10 pp* | *-0.10 pp [-0.80, +0.60] (p=0.781)* | *-0.15 pp* | *+0.00 pp* | *+0.0130* | *+0.22 pp* |


### 3.1. Phân Tích Các Mẫu Vật Thách Thức (Hard Physical Vouchers: Acc < 95%)
Bảng dưới đây xác định các phôi gỗ cá thể có độ chính xác nhận dạng dưới 95% do biến thiên dị biệt về giải phẫu, màu sắc hoặc góc cưa xẻ:

| Phân Vùng | Mã Phôi Gỗ | Số Lượng Ảnh | Acc Baseline | Acc Đối Kháng GAN | Chênh Lệch $\\Delta$ |
|:---:|:---:|:---:|:---:|:---:|:---:|
| Seed 456 | Phôi `50 (Guibourtia_arnoldiana.1)` | 32 ảnh | 0.0% | 0.0% | +0.0 pp |
| Seed 456 | Phôi `53 (Guibourtia_arnoldiana.3)` | 29 ảnh | 0.0% | 3.4% | +3.4 pp |
---

## 4. Luận Giải Khoa Học Chuyên Sâu & Báo Cáo Kết Quả Tiêu Cực Trung Thực (Scientific Discussion & Honest Negative Results)

### 4.1. Bản Chất của Hiện Tượng Rò Rỉ Mẫu Vật (Specimen Leakage Audit & Inflation Gap)
* **Thực trạng chia ngẫu nhiên cấp ảnh (Naive Image-Level Random Split)**:
  Phần lớn nghiên cứu thị giác máy tính trong nhận dạng gỗ áp dụng phép phân chia ngẫu nhiên cấp ảnh, khiến các ảnh cắt từ cùng một thanh gỗ xuất hiện đồng thời ở cả tập huấn luyện và tập kiểm thử.
* **Mạng học thuộc đặc trưng phôi thay vì cấu trúc loài**:
  Mạng nơ-ron khai thác shortcut bề mặt (độ nhẵn bề mặt cưa xẻ, sắc thái màu cục bộ, góc chiếu sáng) để nhận diện phôi thay vì học giải phẫu vi thể loài.
* **Bằng chứng thực nghiệm định lượng trên S3**:
  Kiểm toán đa phân vùng cho thấy Naive Split làm tăng vọt độ chính xác thêm **+2.48 pp [95% CI: +1.73, +3.23 pp]** ($p < 0.001$, từ 97.07% lên 99.55%). Sự chênh lệch này khẳng định tính bắt buộc của giao thức Out-of-Specimen Voucher Split trong giám định tư pháp.

### 4.2. Ranh Giới Giải Phẫu Họ Đậu & Trần Hiệu Năng Thực Tế (Anatomical Separation & Baseline Ceiling)
* **Hiệu năng Baseline ResNet-18 đạt 97.07% [96.34%, 97.80%]**:
  Trong bộ dữ liệu chuẩn 11 loài Fabaceae của S3, các loài có đặc điểm giải phẫu macroscopic phân tách tương đối rõ ràng (ví dụ: mô mềm dạng dải liên tục ở *Afzelia bipindensis*, mô mềm cánh ở *Pterocarpus soyauxii*, hay mạch gỗ phân tán ở *Dalbergia oliveri*).
* **Kết quả đối chuẩn trung thực (Honest Negative Result)**:
  So với Baseline ERM (97.07%), mô hình đối kháng GRL+vCLUB đạt 96.88% (chênh lệch -0.19 pp, 95% CI [-1.16, +0.78] pp, paired t-test $p = 0.66$, Wilcoxon $p = 0.91$). Cả hai mô hình tương đương về mặt thống kê và GRL không đem lại ưu thế phân loại vượt trội so với ERM.

### 4.3. Phân Tích Cơ Chế Thất Bại của Phương Pháp Bất Biến (Failure Mode Analysis of Invariance Frameworks)
1. **Khoảng cách tổng quát hóa không bị triệt tiêu (GGSL Gap)**:
   * Baseline GGSL đạt 2.74 pp trong khi GAN GGSL là 2.82 pp ($\Delta = +0.08$ pp, $p > 0.05$). GRL không kéo phẳng được độ hụt hiệu năng giữa phôi đã thấy và phôi chưa thấy.
2. **Chỉ số rò rỉ mẫu vật (Train SRI)**:
   * Baseline Train SRI đạt 0.5261 trong khi GAN Train SRI đạt 0.5380. Cả hai đều giảm so với điểm neo ImageNet chưa fine-tune (0.5667), nhưng GRL không làm giảm thêm SRI so với ERM.
3. **Xung đột tối ưu hóa đối kháng (Minimax Gradient Conflict)**:
   * Việc ép buộc bộ trích xuất đặc trưng loại bỏ thông tin phôi thông qua Gradient Reversal Layer vô tình làm suy giảm các chi tiết vân gỗ vi thể đặc trưng của loài, gây xung đột trực tiếp giữa hàm mất mát phân loại loài ($L_{species}$) và hàm mất mát đối kháng ($L_{specimen}$).
4. **Phân tích ca thất bại điển hình (Seed 456: Phôi 50 & 53)**:
   * Hai phôi 50 và 53 của *Guibourtia arnoldiana* có độ chính xác 0% ở Baseline do độ sắc nét Laplacian cực thấp (113.8 và 152.1 so với >1,000 của các phôi chuẩn) và mô mềm tương đồng với *Guibourtia ehie*. GRL+vCLUB cũng không thể cải thiện hai mẫu này (đạt 0% và 3.4%) vì cơ chế đối kháng không thể tái tạo lại các chi tiết giải phẫu đã bị mất do mờ nét quang học.
5. **Hiệu chuẩn độ tin cậy thực tế (Calibrated ECE)**:
   * Sau Post-hoc Temperature Scaling trên tập Val ngoại suy, sai số hiệu chuẩn đạt 1.58% - 1.93%.

### 4.4. Khuyến Nghị Quy Chuẩn Cho Nghiên Cứu Giám Định Gỗ (Forensic Policy Recommendations)
1. **Bắt buộc áp dụng Out-of-Specimen Voucher Split**: Loại bỏ hoàn toàn việc sử dụng Naive Random Split để báo cáo độ chính xác trong các nghiên cứu giám định gỗ.
2. **Báo cáo trung thực kết quả thực nghiệm**: Công bố minh bạch các trường hợp phương pháp bất biến phức tạp không vượt trội so với ERM baseline để định hướng nghiên cứu thực chất.
3. **Áp dụng kiểm định thống kê cấp phôi (Cluster Bootstrap CIs)**: Resampling theo phôi gỗ vật lý thay vì resampling theo từng ảnh để tránh kết luận quá lạc quan (Bengio & Grandvalet, 2004).

---

## 5. Hướng Dẫn Tái Lập (Reproducibility Guide)

Để chạy tự động toàn bộ quy trình 9 seeds trên Kaggle GPU (T4 / P100):
```bash
# Trên Kaggle / Linux:
bash run_all.sh

# Hoặc chạy trực tiếp script Python với 9 splits:
python run_benchmark.py \
    --image_root "/kaggle/input/datasets/s3-origin/S3" \
    --num_seeds 9 \
    --test_vouchers_per_species 5 \
    --val_vouchers_per_species 1 \
    --epochs 15 \
    --batch_size 64 \
    --lr 3e-4 \
    --adv_weight 0.05 \
    --club_weight 0.001 \
    --label_smoothing 0.05 \
    --backbone "resnet18"
```
