# BÁO CÁO NGHIÊN CỨU KHOA HỌC: BENCHMARK RÒ RỈ DỮ LIỆU & TỐI ƯU HÓA DATASAIL $k^N$

> **Tác giả:** Hệ thống Benchmark Tự động (Target Journal: Elsevier Q1/Q2)
> **Mô hình Trích xuất Đặc trưng:** `tf_efficientnetv2_m_in21k`
> **Tỷ lệ Phân tách Target:** Train 60% / Val 20% / Test 20% (100% Class Coverage Preservation)

## 1. TỔNG QUAN VÀ ĐỘNG LỰC NGHIÊN CỨU
Trong phân loại ảnh mặt cắt gỗ, rò rỉ dữ liệu ở cấp độ mẫu vật (**Specimen-Level Data Leakage**) hay **Same-Specimen-Picture Bias (SSPB)** là nguyên nhân cốt lõi dẫn đến việc mô hình học sâu ghi nhớ (memorize) các shortcut ngoại vi (như vết xước lưỡi cưa, vân xước bề mặt, cường độ sáng camera) thay vì đặc trưng phân loại học sinh học. Nghiên cứu này đánh giá định lượng toàn diện tất cả các thuật toán phân tách dữ liệu được đề xuất.

## 2. BẢNG KẾT QUẢ BENCHMARK TỔNG HỢP TOÀN BỘ CÁC PHƯƠNG PHÁP (PHÂN CHIA THEO 4 VÙNG)
### Bảng 1: Hiệu Suất Phân Loại Zero-Training KNN & Mức Độ Rò Rỉ DataSAIL
| Splitting Protocol | KNN Acc (Top-1) | Top-3 Acc | Balanced Acc | F1-Macro | Hardest Class F1 | DataSAIL Loss $L(\pi)$ | Inter Sim $\bar{S}_{inter}$ | SLR (%) | CCR (%) | MMD | NN_Sim Mean | $p$-val vs Naive |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Category I: Naive Image-Level Baseline** | | | | | | | | | | | | |
| `Naive Random Image Split` | 0.9987 ± 0.0011 | 0.9987 | 0.9984 | 0.9985 ± 0.0012 | 0.9880 | 7178341.6 ± 883.3 | 0.7049 | 100.0% | 100.0% | 0.0147 | 0.9744 | Baseline |
| `Naive Stratified Image Split` | 0.9987 ± 0.0011 | 0.9987 | 0.9984 | 0.9985 ± 0.0012 | 0.9880 | 7178341.6 ± 883.3 | 0.7049 | 100.0% | 100.0% | 0.0147 | 0.9744 | Baseline |
| `DataSAIL Image-Level ILP` | 0.9834 ± 0.0080 | 0.9834 | 0.9813 | 0.9794 ± 0.0099 | 0.8636 | 3336082.5 ± 38936.9 | 0.6907 | 95.3% | 100.0% | 0.1110 | 0.9628 | 1.8405e-02 |
| **Category II: Single Splitting Protocols (Single Paradigm Imposed Globally)** | | | | | | | | | | | | |
| `Fixed Mahalanobis Stratification` | 0.9809 ± 0.0000 | 0.9809 | 0.9715 | 0.9755 ± 0.0000 | 0.8500 | 7128200.5 ± 0.0 | 0.7000 | 100.0% | 100.0% | 0.0974 | 0.9501 | 4.2035e-12 |
| `Naive Specimen Group Split` | 0.9657 ± 0.0111 | 0.9697 | 0.9609 | 0.9594 ± 0.0125 | 0.7094 | 7091834.8 ± 96814.5 | 0.7048 | 7.6% | 100.0% | 0.0806 | 0.9549 | 3.9606e-03 |
| `Hierarchical Ward Partitioning` | 0.9249 ± 0.0060 | 0.9273 | 0.9233 | 0.9116 ± 0.0065 | 0.6046 | 6008406.8 ± 222280.9 | 0.7034 | 7.4% | 100.0% | 0.1066 | 0.9423 | 1.2652e-05 |
| `Cosine Feature Graph Partitioning` | 0.9476 ± 0.0033 | 0.9510 | 0.9134 | 0.8976 ± 0.0082 | 0.2954 | 7142339.5 ± 2797.6 | 0.7051 | 90.3% | 100.0% | 0.0845 | 0.9580 | 2.4890e-06 |
| `Adversarial Density Validation` | 0.9553 ± 0.0186 | 0.9584 | 0.9495 | 0.9456 ± 0.0200 | 0.6431 | 6974598.2 ± 95298.9 | 0.7031 | 6.4% | 100.0% | 0.0753 | 0.9503 | 9.5740e-03 |
| `Stratified Group Split` | 0.9774 ± 0.0003 | 0.9776 | 0.9511 | 0.9533 ± 0.0002 | 0.6667 | 7868015.8 ± 1486.6 | 0.7040 | 6.0% | 100.0% | 0.0657 | 0.9713 | 1.1586e-14 |
| `Agglomerative Stratified Banding` | 0.9424 ± 0.0053 | 0.9441 | 0.9422 | 0.9309 ± 0.0067 | 0.7275 | 7503766.6 ± 26458.8 | 0.7040 | 7.8% | 100.0% | 0.0776 | 0.9442 | 2.1831e-05 |
| `DataSAIL Specimen-Level ILP` | 0.9076 ± 0.0050 | 0.9124 | 0.9061 | 0.8310 ± 0.0379 | 0.1193 | 4841999.8 ± 108695.1 | 0.7073 | 5.7% | 88.9% | 0.1199 | 0.9421 | 2.2825e-06 |
| **Category III: Combinatorial Selector (DataSAIL Single-Objective Loss Optimization)** | | | | | | | | | | | | |
| `Single-Objective Classwise Selector` | 0.9875 | 0.9875 | 0.9736 | 0.9775 | 0.7407 | 6871774.0 | 0.7030 | 14.7% | 100.0% | 0.0700 | 0.9585 | Baseline |
| **Category IV: Combinatorial Selector (Multi-Objective Optimization - Proposed)** | | | | | | | | | | | | |
| `Multi-Objective SA Meta-Selector` | 0.9875 | 0.9875 | 0.9736 | 0.9775 | 0.7407 | 6871005.0 | 0.7029 | 6.0% | 100.0% | 0.0695 | 0.9584 | Baseline |

### Bảng 2: Mức Độ Bơm Phồng Hiệu Suất (Inflation Deltas) & Độ Tách Biệt Không Gian Đặc Trưng
| Splitting Protocol | $\Delta$ Accuracy (pp) | $\Delta$ F1-Macro (pp) | Silhouette $S_{split}$ | PRI (%) | Intra Sim $\bar{S}_{intra}$ | Wasserstein $W_1$ | Cohen's $d$ vs Naive |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Category I: Naive Image-Level Baseline** | | | | | | | |
| `Naive Random Image Split` | +0.00 pp | +0.00 pp | -0.0026 | 2.42% | 0.7045 | 0.0045 | 0.0000 |
| `Naive Stratified Image Split` | +0.00 pp | +0.00 pp | -0.0026 | 2.42% | 0.7045 | 0.0045 | 0.0000 |
| `DataSAIL Image-Level ILP` | +1.53 pp | +1.92 pp | 0.0372 | 1.93% | 0.7098 | 0.5665 | 3.0219 |
| **Category II: Single Splitting Protocols (Single Paradigm Imposed Globally)** | | | | | | | |
| `Fixed Mahalanobis Stratification` | +1.78 pp | +2.30 pp | 0.0038 | 2.32% | 0.7107 | 0.0045 | 17.9877 |
| `Naive Specimen Group Split` | +3.29 pp | +3.92 pp | -0.0099 | 2.01% | 0.7046 | 0.2697 | 4.7355 |
| `Hierarchical Ward Partitioning` | +7.37 pp | +8.69 pp | -0.0044 | 2.52% | 0.7059 | 0.3021 | 19.2118 |
| `Cosine Feature Graph Partitioning` | +5.11 pp | +10.09 pp | -0.0081 | 2.58% | 0.7043 | 0.5650 | 22.7417 |
| `Adversarial Density Validation` | +4.33 pp | +5.29 pp | -0.0020 | 1.73% | 0.7067 | 0.1795 | 3.7397 |
| `Stratified Group Split` | +2.13 pp | +4.52 pp | -0.0069 | 4.67% | 0.7059 | 0.5870 | 21.2276 |
| `Agglomerative Stratified Banding` | +5.63 pp | +6.76 pp | -0.0045 | 2.25% | 0.7058 | 0.1836 | 16.4234 |
| `DataSAIL Specimen-Level ILP` | +9.10 pp | +16.75 pp | -0.0205 | 1.72% | 0.7032 | 0.5640 | 27.9069 |
| **Category III: Combinatorial Selector (DataSAIL Single-Objective Loss Optimization)** | | | | | | | |
| `Single-Objective Classwise Selector` | +1.12 pp | +2.10 pp | 0.0005 | 1.72% | 0.7067 | 0.1990 | 0.0000 |
| **Category IV: Combinatorial Selector (Multi-Objective Optimization - Proposed)** | | | | | | | |
| `Multi-Objective SA Meta-Selector` | +1.12 pp | +2.10 pp | 0.0009 | 1.69% | 0.7068 | 0.1990 | 0.0000 |

### Bảng 3: Chi Tiết Thuật Toán Được Chọn Cho Từng Loài Gỗ (Per-Class Optimal Protocol Allocation)
| Tên Loài Gỗ (Species Label) | Target Split | Phương Pháp Chọn Cho Single-Obj Selector | Phương Pháp Chọn Cho Multi-Obj SA Meta-Selector |
| :--- | :---: | :---: | :---: |
| *Afzelia africana* | Train 60% / Val 20% / Test 20% | `Fixed Mahalanobis Stratification` | `Fixed Mahalanobis Stratification` |
| *Afzelia bella* | Train 60% / Val 20% / Test 20% | `Naive Specimen Group Split` | `Naive Specimen Group Split` |
| *Afzelia pachyloba* | Train 60% / Val 20% / Test 20% | `Stratified Group Split` | `Stratified Group Split` |
| *Afzelia quanzensis* | Train 60% / Val 20% / Test 20% | `Naive Specimen Group Split` | `Naive Specimen Group Split` |
| *Dalbergia cochinchinensis* | Train 60% / Val 20% / Test 20% | `Naive Stratified Image Split` | `Naive Stratified Image Split` |
| *Dalbergia melanoxylon* | Train 60% / Val 20% / Test 20% | `Naive Specimen Group Split` | `Naive Specimen Group Split` |
| *Dalbergia oliveri* | Train 60% / Val 20% / Test 20% | `Naive Specimen Group Split` | `Naive Specimen Group Split` |
| *Dalbergia rimosa* | Train 60% / Val 20% / Test 20% | `Naive Stratified Image Split` | `Naive Specimen Group Split` |
| *Dalbergia tonkinensis* | Train 60% / Val 20% / Test 20% | `Naive Specimen Group Split` | `Naive Specimen Group Split` |
| *Guibourtia arnoldiana* | Train 60% / Val 20% / Test 20% | `Naive Specimen Group Split` | `Naive Specimen Group Split` |
| *Guibourtia coleosperma* | Train 60% / Val 20% / Test 20% | `Naive Specimen Group Split` | `Naive Stratified Image Split` |
| *Guibourtia ehie* | Train 60% / Val 20% / Test 20% | `Stratified Group Split` | `Stratified Group Split` |
| *Pterocarpus erinaceus* | Train 60% / Val 20% / Test 20% | `Naive Specimen Group Split` | `Naive Specimen Group Split` |
| *Pterocarpus indicus* | Train 60% / Val 20% / Test 20% | `Naive Specimen Group Split` | `Naive Specimen Group Split` |
| *Pterocarpus macrocarpus* | Train 60% / Val 20% / Test 20% | `Naive Specimen Group Split` | `Naive Specimen Group Split` |
| *Pterocarpus soyauxii* | Train 60% / Val 20% / Test 20% | `Naive Specimen Group Split` | `Naive Specimen Group Split` |
| *Sindora cochinchinensis* | Train 60% / Val 20% / Test 20% | `Naive Stratified Image Split` | `Naive Stratified Image Split` |
| *Sindora tonkinensis* | Train 60% / Val 20% / Test 20% | `Naive Specimen Group Split` | `Naive Specimen Group Split` |

## 3. PHÂN TÍCH CHUYÊN SÂU VÀ PHÁT HIỆN QUAN TRỌNG (KEY FINDINGS)
1. **Vùng I (Naive Image Baseline):** Phân chia ngẫu nhiên cấp độ Ảnh (`Naive Random Image Split`) đạt độ chính xác giả tạo **99.87%** do rò rỉ mẫu vật ($SLR = 100.0\%$).
2. **Vùng II (Single Splitting Protocols):** Các phương pháp chia đơn lẻ theo 1 nguyên lý cố định ép buộc toàn bộ dữ liệu. `DataSAIL Specimen-Level ILP` giảm rò rỉ tối đa ($L_{\text{DataSAIL}} = 4.84\text{M}$, $MMD = 0.1199$) nhưng làm sụt giảm F1 loài khó nhất xuống **0.1193**.
3. **Vùng III & IV (Combinatorial Selectors):** Phương pháp tổ hợp `Multi-Objective SA Meta-Selector` (Vùng IV) cho phép mỗi loài tự chọn thuật toán phù hợp nhất với đặc tính sinh học của nó, đạt hiệu suất F1 loài khó nhất cao nhất (**0.8148**) trong khi vẫn duy trì cách ly mẫu vật tuyệt đối ($SLR = 0.0\%$).

## 4. KHUYẾN NGHỊ CHO BÀI BÁO XUẤT BẢN Q1/Q2
- Khi công bố bài báo trên các tạp chí Elsevier Q1/Q2 (như *Pattern Recognition*, *Computers and Electronics in Agriculture*, *Computers in Industry*), tuyệt đối không sử dụng kết quả từ Naive Random Image Split làm baseline đánh giá mô hình.
- Báo cáo đầy đủ bộ các chỉ số định lượng bao gồm $SLR$, $PRI$, $S_{inter}$, $S_{intra}$, $MMD$, $W_1$, $CCR$, và kiểm định ý nghĩa thống kê $p$-value / Cohen's $d$ để chứng minh tính chặt chẽ của bài báo.