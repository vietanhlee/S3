# Cẩm Nang Thực Thi Thực Nghiệm & Trích Xuất Số Liệu Chuẩn Cho Paper 3

**Công trình**: *Learning Specimen-Invariant Diagnostic Representations for Timber Forensics: Countering Same-Specimen-Picture Bias under Extreme Biological Scarcity*  
**Thư mục làm việc**: `03_research_paper_specimen_invariance/`  
**Ngôn ngữ thực thi**: Python 3.9+ / PyTorch 2.0+ (CUDA hỗ trợ)

---

## 1. Danh Mục Các Tệp Mã Nguồn Cốt Lõi

| Tệp Mã Nguồn | Chức Năng Chính | Đầu Vào | Đầu Ra |
|---|---|---|---|
| [`train.py`](file:///g:/S3_paper/03_research_paper_specimen_invariance/train.py) | Huấn luyện từng mô hình đơn lẻ (hỗ trợ 15 phương pháp, 4 backbones, 5 fold LOSO). | `metadata.csv`, thư mục ảnh | `best_model.pth`, `train_summary.json` |
| [`run_parallel_dispatcher.py`](file:///g:/S3_paper/03_research_paper_specimen_invariance/run_parallel_dispatcher.py) | Bộ điều phối tự động chạy đa GPU (DataParallel 2x T4 Kaggle hoặc Task-Parallel). | Danh sách phương pháp / backbones | Thư mục checkpoint của từng mô hình |
| [`evaluate.py`](file:///g:/S3_paper/03_research_paper_specimen_invariance/evaluate.py) | Đánh giá chuyên sâu: $\text{LOSO}_{\text{multi}}$, Overall Top-1, $\text{F1}_{\text{singleton}}$, SRI (Linear & MLP), ECE/MCE. | Checkpoint `.pth`, fold index | `evaluation_report.json`, biểu đồ t-SNE |
| [`run_ablation_study.py`](file:///g:/S3_paper/03_research_paper_specimen_invariance/run_ablation_study.py) | Chạy ma trận Ablation Study (Masked Softmax, Sampler, Annealing, vCLUB). | Cấu hình các biến thể ablation | `ablation_results.json` (Table 8) |
| [`run_robustness_study.py`](file:///g:/S3_paper/03_research_paper_specimen_invariance/run_robustness_study.py) | Kiểm thử độ bền vật lý (nhiễu ánh sáng, vệt cưa máy, mờ xước, xoay góc, composite). | Checkpoints đã huấn luyện | `robustness_results.json` (Table 9) |
| [`run_pareto_sweep.py`](file:///g:/S3_paper/03_research_paper_specimen_invariance/run_pareto_sweep.py) | Quét siêu tham số độ nhạy đối kháng $\beta \in [0.1, 2.0]$ và vCLUB $\mu \in [0.01, 0.50]$. | Grid tham số | `pareto_sweep_results.json` (Table 10) |
| [`collect_all_benchmark_results.py`](file:///g:/S3_paper/03_research_paper_specimen_invariance/collect_all_benchmark_results.py) | Quét toàn bộ kết quả runs và sinh trực tiếp 8 bảng mã nguồn LaTeX chuẩn Elsevier. | Toàn bộ log trong output dir | 8 file `.tex` tại `summary_tables/` |
| [`run_full_experiment_suite.py`](file:///g:/S3_paper/03_research_paper_specimen_invariance/run_full_experiment_suite.py) | Master CLI Runner: điều phối toàn bộ chu trình thực nghiệm bằng 1 lệnh duy nhất. | Tham số `--mode` | Toàn bộ quy trình hoàn tất |

---

## 2. Các Kịch Bản Thực Thi Chi Tiết

### Kịch Bản 0: Kiểm tra nhanh thông suốt toàn bộ hệ thống (Sanity Check / 1 Epoch)
Chạy thử nghiệm 1 epoch để xác nhận toàn bộ pipeline nạp dữ liệu, mạng nơ-ron, GRL, vCLUB và bộ xuất bảng hoạt động trơn tru:
```bash
python run_full_experiment_suite.py --mode quick_test
```

---

### Kịch Bản 1: Huấn luyện Framework Đề Xuất (GRL + vCLUB) Trên Máy Đơn GPU / Local
Huấn luyện mô hình đề xuất trên Fold 0 với backbone ConvNeXt-Tiny:
```bash
python train.py --method conditional_grl_club --backbone convnext_tiny --fold 0 --epochs 40 --batch_size 64
```
*Để huấn luyện trọn vẹn 5-fold Round-Robin LOSO:*
```bash
for ($f = 0; $f -le 4; $f++) {
    python train.py --method conditional_grl_club --backbone convnext_tiny --fold $f --epochs 40 --batch_size 64
}
```

---

### Kịch Bản 2: Huấn luyện Song Song Trên Kaggle (Dual GPU T4 DataParallel)
Kaggle cung cấp 2 GPU T4 (mỗi GPU 16GB). Script `run_parallel_dispatcher.py` tự động kích hoạt `torch.nn.DataParallel` để gộp cả 2 GPU huấn luyện đồng thời:
```bash
# Huấn luyện 6 phương pháp cốt lõi (Focal, Mixup, DANN, CLUB, Conditional GRL, Proposed Full):
python run_parallel_dispatcher.py --mode baselines --fold 0 --epochs 40

# Huấn luyện trên cả 4 kiến trúc thị giác (ConvNeXt-Tiny, ResNet-50, EfficientNetV2-S, Swin-T):
python run_parallel_dispatcher.py --mode backbones --fold 0 --epochs 40
```

---

### Kịch Bản 3: Chạy Toàn Bộ Ablation Study (Phân Tích Thành Phần - Bảng 8)
Thực thi ma trận kiểm định từng thành phần cấu trúc:
```bash
python run_ablation_study.py --backbone convnext_tiny --fold 0 --epochs 40
```
Script sẽ tự động chạy các biến thể:
1. Standard ERM Baseline
2. ERM + Specimen-Balanced Sampler (Đối chứng công bằng)
3. ERM + Class-Balanced Focal Loss (Đối chứng công bằng)
4. W/o Masked Softmax ($M_c \equiv 1$ - kiểm chứng sụp đổ singleton)
5. W/o Annealing Schedule ($\lambda_{\text{adv}} = 1.0$)
6. W/o CLUB Bottleneck (Chỉ dùng GRL)
7. W/o GRL (Chỉ dùng CLUB)
8. W/o Specimen-Balanced Sampler
9. Proposed Full Framework (GRL + vCLUB)

---

### Kịch Bản 4: Chạy Kiểm Thử Độ Bền Vật Lý (Stress-Testing - Bảng 9)
Đánh giá độ bền biểu diễn trước các biến dạng thực địa (vệt cưa đĩa, mờ xước, ánh sáng kho bãi, xoay góc $135^\circ$, composite stress):
```bash
python run_robustness_study.py --output_base_dir specimen_invariance_outputs
```

---

### Kịch Bản 5: Quét Độ Nhạy Siêu Tham Số (Sensitivity Sweep - Bảng 10)
Quét lưới $(\beta, \mu)$ trên tập validation của Fold 0:
```bash
python run_pareto_sweep.py --backbone convnext_tiny --fold 0 --epochs 15
```

---

### Kịch Bản 6: Trích Xuất Toàn Bộ Số Liệu & Tự Động Sinh 8 Bảng LaTeX
Sau khi huấn luyện hoàn tất, chạy lệnh này để trích xuất sạch sẽ toàn bộ metrics:
```bash
python collect_all_benchmark_results.py --output_base_dir specimen_invariance_outputs
```
Toàn bộ 8 bảng LaTeX sẽ được lưu tại thư mục:  
`specimen_invariance_outputs/summary_tables/`
* `table3_master_benchmark.tex` $\to$ Khớp với Table 3 trong `main.tex`
* `table4_forestry_benchmark.tex` $\to$ Khớp với Table 4 trong `main.tex`
* `table5_pairwise_significance.tex` $\to$ Khớp với Table 5 trong `main.tex`
* `table6_backbones.tex` $\to$ Khớp với Table 6 trong `main.tex`
* `table7_perspecies_breakdown.tex` $\to$ Khớp với Table 7 trong `main.tex`
* `table8_ablation_study.tex` $\to$ Khớp với Table 8 trong `main.tex`
* `table9_robustness_study.tex` $\to$ Khớp với Table 9 trong `main.tex`
* `table10_sensitivity_sweep.tex` $\to$ Khớp với Table 10 trong `main.tex`
* `master_benchmark_metrics.csv` $\to$ File tổng hợp toàn bộ số liệu dạng bảng tính Excel.

---

## 3. Bản Đồ Tương Thích & Đảm Bảo Tính Nhất Quán 100%

Mã nguồn đã được rà soát và cấu hình để đồng bộ tuyệt đối với bài báo xuất bản:
1. **Kích thước ảnh**: Tự động nhận diện và xử lý chuẩn $224 \times 224$ pixels ($12.0\,\mu\text{m/pixel}$).
2. **Loài thứ 19**: Tự động nhận diện đúng danh pháp *Sindora tonkinensis* (10 vouchers) từ `metadata.csv`.
3. **Mặt nạ đối kháng $M_c$**: Được tính tự động theo công thức $\mathbf{1}(|\mathcal{G}_c| \ge 2)$ che chắn hoàn hảo cho 2 loài singleton (*D. cochinchinensis*, *G. coleosperma*).
4. **vCLUB Intra-Taxon Permutation**: Bộ lấy mẫu âm bản của vCLUB chỉ xáo trộn trong cùng loài co-specific ($y_j = y_i$), đảm bảo chặt chẽ toán học 100%.
5. **Giao thức SRI Probe**: Đã tích hợp cả Linear Ridge Probe và Deep 3-Layer MLP Probe để đo độ nén tương hỗ thông tin mẫu vật cả tuyến tính và phi tuyến.
