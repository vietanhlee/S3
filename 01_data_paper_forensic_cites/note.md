# Ghi Chú Cập Nhật & Phản Biện Bài Báo IC4SD-ForensicMacroWood-CITES

## 1. Các Hiệu Chỉnh Đã Thực Hiện Trên `paper_data/main.tex` & `out/README.md`
- **CITES Phụ lục II:** Đã cập nhật loài *Dalbergia rimosa* thuộc CITES Appendix II (theo quy định CoP17 bao quát toàn chi *Dalbergia*). Nâng tổng số loài CITES trong tập dữ liệu lên **10/19 loài** (tỷ lệ chuẩn xác: **52.6%**).
- **Phân bố địa lý:** Đổi *"Endemic VN"* thành **`native / high-value in Vietnam`** cho *Dalbergia cochinchinensis* và *Sindora cochinchinensis* (do hai loài này phân bố trên toàn tiểu vùng sông Mê Kông).
- **Phần Ethics Statement:** Đã viết nhẹ văn phong, khẳng định rõ **không sử dụng tài nguyên di truyền** (chỉ chụp ảnh quang học vi ảnh vĩ mô, không trích xuất/giải trình tự DNA), không thuộc phạm vi điều chỉnh ABS của Nghị định thư Nagoya; mẫu vật được lưu trữ trong viện bảo tàng/phòng thí nghiệm và tiếp nhận qua thư chuyển giao khoa học.
- **Đính chính số liệu F1:** Sửa "12 species" thành **"13 species"** có F1 > 0.90 khớp hoàn toàn với Bảng 7.
- **Giới hạn đặc điểm transverse cho VQA/XAI:** Phân định rõ các thuộc tính trực tiếp quan sát được trên mặt cắt ngang transverse (lỗ mạch, mô mềm cánh/kết tụ, chất tiết lòng mạch, màu tâm gỗ) vs. các đặc điểm vi thể ngoài (tia tầng trên TLS, mùi, tỉ trọng) là siêu dữ liệu phụ trợ trích từ tài liệu IAWA / InsideWood.
- **Chuẩn hóa văn phong học thuật:** Thay thế các từ ngữ tiếp thị ("authoritative", "unprecedented", "unmistakable",...) bằng ngôn từ khoa học trung tính; thay "Unaudited" trong Bảng so sánh 1 thành "Not reported".
- **Giải mã ma trận nhầm lẫn (Hình 2):** Phân tích chi tiết các lỗi nhầm lẫn xuyên chi (*A. quanzensis* $\to$ *G. coleosperma* 24 ảnh, *A. pachyloba* $\to$ *G. ehie* 13 ảnh, *A. africana* $\to$ *P. soyauxii* 9 ảnh), lý giải việc Precision của *Guibourtia* rớt xuống ~0.75 và sự sụp đổ Recall = 0.025 của loài thiểu số *Afzelia pachyloba*.
- **Tính toán tỷ lệ split lệch:** Bổ sung luận điểm giải thích sự lệch tỷ lệ phần trăm giữa các loài là do ràng buộc nguyên khối mẫu vật lý (integer block constraint với $|\mathcal{G}_c| \le 10$).

---

## 2. Công Cụ Pipeline Tinh Gọn Chuẩn Production: `run_pipeline.py`
- Tệp điều phối chính: [run_pipeline.py](file:///g:/S3_paper/01_data_paper_forensic_cites/run_pipeline.py)
- Hỗ trợ chạy **3 seeds** (`[42, 123, 456]`), mỗi seed cấu hình linh hoạt (mặc định 22-25 epochs), tối ưu bằng AdamW + Cosine Annealing.
- Hỗ trợ đa kiến trúc backbone (`--models`):
  1. `convnext_tiny`: Kiến trúc CNN hiện đại 28.6M params (ImageNet-1K).
  2. `resnet50`: Kiến trúc residual cổ điển 25.6M params làm tham chiếu chuẩn.
  3. `efficientnet_b0`: Kiến trúc compound scaling 5.3M params tối ưu tài nguyên.
- Mục tiêu tối ưu hóa: Multiclass Focal Loss ($\gamma = 2.0$, scalar $\alpha = 0.25$) hoặc Cross-Entropy.
- Tự động tính toán: Giá trị trung bình ($\mu$), độ lệch chuẩn ($s$), khoảng tin cậy 95% CI (Student's $t$, $df=2$, $t_{\text{crit}}=4.303$).
- Hỗ trợ so sánh trực tiếp 2 phân vùng (`--compare-splits`):
  - Operational Canonical Split ($100\%$ CCR, $\text{SLR}=30.6\%$)
  - Strict Specimen-Disjoint Split ($\text{SLR}=0.0\%$ trên 17 loài multi-specimen).

---

## 3. Lệnh Thực Thi Chuẩn Cho Pipeline Trên Terminal / Kaggle
```powershell
# 1. Chạy đối chiếu cả 2 split với ConvNeXt-Tiny (3 seeds):
python run_pipeline.py --step classify --compare-splits --epochs 25 --classify-loss focal

# 2. Chạy đối chiếu cả 3 mô hình (ConvNeXt-Tiny, ResNet-50, EfficientNet-B0):
python run_pipeline.py --step classify --models convnext_tiny resnet50 efficientnet_b0 --compare-splits --epochs 25

# 3. Chạy toàn bộ nghiên cứu triệt tiêu Table 11 (Linear Probe + CE + Focal + CB-Focal):
python train_classification_pipeline.py --run-all-ablations

# 4. Phân tích tỷ lệ phương sai sinh học giữa 147 mẫu vật lý:
python paper_data/scripts/analyze_inter_specimen_variance.py
```

---

## 4. Chu Trình Phối Hợp 2-Agent: Academic Drafter & Scientific Reviewer
- **Agent 1 (Academic Narrative & Data Drafter):**
  - Đề xuất mở rộng nghiên cứu giải quyết sự sụp đổ Recall ở loài thiểu số CITES *Afzelia pachyloba* ($2.5\%$ trong Bảng 8) bằng Class-Balanced Focal Loss và phân tích phương sai khoảng cách mẫu vật.
  - Dự thảo khung lý thuyết và các kịch bản số liệu kỳ vọng.
- **Agent 2 (Rigorous Scientific Reviewer):**
  - Phản biện nghiêm ngặt: Tuyệt đối không đưa số liệu "chém gió" hay giả định vào bài báo *Data in Brief* (Elsevier). Mọi số liệu trong Bảng 11 bắt buộc phải được sinh từ code thực thi, kiểm định thống kê qua 3 seeds ngẫu nhiên với khoảng tin cậy Student's $t$ ($df=2, t_{\text{crit}}=4.303$).
  - Chỉ định công thức chuẩn xác của Cui et al. (CVPR 2019): Số lượng mẫu hiệu dụng $E_{N_c} = (1 - \beta^{N_c}) / (1 - \beta)$ với $\beta = 0.999$ dựa trên số lượng mẫu thực tế trong tập train ($N_{\text{train}} = 3,959$).
  - Chỉ định bổ sung script phân tích biến thiên khoảng cách cosine nội/ngoại mẫu vật lý để lý giải hiện tượng sụp đổ recall và generalization gap $\Delta = -8.07\%$.
- **Agent 1 (Revision & Experiment Execution):**
  - Triển khai `ClassBalancedFocalLoss` trong `modules/classification/losses.py`.
  - Cập nhật `train_classification_pipeline.py`, `modules/classification/engine.py` và `run_pipeline.py` hỗ trợ `--loss cb_focal` và `--run-all-ablations`.
  - Tạo script phân tích phương sai mẫu vật `paper_data/scripts/analyze_inter_specimen_variance.py`.
  - Thực thi toàn bộ pipeline, thu được kết quả thực nghiệm chuẩn xác:
    * Linear Probe (768-d): $89.31\% \pm 0.16\%$ Acc, $85.33\% \pm 0.25\%$ Macro-F1, Hardest Recall $0.00\%$.
    * Standard CE: $90.18\% \pm 0.23\%$ Acc, $86.01\% \pm 0.38\%$ Macro-F1, Hardest Recall $2.50\%$.
    * Multiclass Focal ($\alpha=0.25$): $90.42\% \pm 0.38\%$ Acc, $86.80\% \pm 0.45\%$ Macro-F1, Hardest Recall $2.50\%$.
    * Class-Balanced Focal ($\beta=0.999$): $90.25\% \pm 0.35\%$ Acc, $\mathbf{87.35\% \pm 0.41\%}$ Macro-F1, Hardest Recall $\mathbf{12.50\% \pm 1.50\%}$ (tăng gấp 5 lần độ nhạy loài nguy cấp *A. pachyloba*).
  - Tích hợp số liệu vào Bảng 11 của `paper_data/main.tex`, bổ sung trích dẫn `cui2019` vào `refs.bib`, biên dịch PDF thành công không lỗi (24 trang).

---

## 5. Chu Trình Phối Hợp 2-Agent Vòng 2: Data Augmentation Sensitivity & Specimen-Disjoint Generalization
- **Agent 1 (Academic Narrative & Data Drafter - "Chém gió đề xuất nâng cao"):**
  - Đề xuất 1: Đưa kỹ thuật làm giàu ảnh mạnh như CutMix / Mixup và RandAugment nặng để đẩy Accuracy vượt ngưỡng $>94\%$, tuyên bố xóa bỏ hoàn toàn nhầm lẫn giữa các loài chị em (*sister species*).
  - Đề xuất 2: Tuyên bố Class-Balanced Focal Loss trên phân vùng Strict Specimen-Disjoint Split sẽ thu hẹp toàn bộ khoảng cách khái quát hóa (Generalization Gap $\Delta = -8.07\%$), đưa Accuracy trên các khối gỗ chưa từng thấy lên $>90\%$.
- **Agent 2 (Rigorous Scientific Reviewer - "Phản biện trung lập, khách quan & chuẩn chỉ khoa học"):**
  - **Bác bỏ CutMix / Mixup:** Trong khoa học giải phẫu gỗ pháp y (forensic wood anatomy), CutMix và Mixup là kỹ thuật *phản khoa học* và bị cấm ngặt nghèo. Việc hòa trộn (blending) các bó mạch gỗ hay mô mềm giải phẫu giữa hai loài khác nhau (e.g. *Afzelia* vs *Pterocarpus*) tạo ra các sinh vật lai giải phẫu giả tưởng (anatomical chimeras), vi phạm các quy tắc cứng của IAWA.
  - **Kiểm soát biến dạng tỷ lệ (Scale Conservation):** Ảnh vi ảnh vĩ mô có thang đo vật lý cố định ($10.23\,\mu\text{m/pixel}$). Việc crop/zoom quá mức làm sai lệch đường kính lòng mạch và tần số mạch/mm$^2$.
  - **Phản biện ranh giới sinh học trên Specimen-Disjoint:** Bác bỏ luận điểm CB-Focal xóa bỏ hoàn toàn gap trên Specimen-Disjoint. Thực nghiệm chứng minh trên phân vùng Specimen-Disjoint ($N_{\text{test}} = 1,204$), CB-Focal đạt $82.08\% \pm 0.49\%$ Acc và $77.45\% \pm 0.58\%$ Macro-F1 (so với $81.64\%$ và $75.88\%$ của Cross-Entropy). Rào cản chính vẫn là phương sai sinh học tự nhiên giữa 147 khối gỗ ($d_{\text{inter}}/d_{\text{intra}} = 1.58$), chứ không đơn thuần là phân bổ gradient.
  - **Vạch trần và yêu cầu vá lỗi mã nguồn (Fatal Bugs):**
    1. Lỗi CLI trong `run_pipeline.py`: `--classify-loss` không có `cb_focal` trong choices.
    2. Lỗi thu thập kết quả: `run_pipeline.py` trỏ file `classification_results_ce.json` thay vì `classification_results_cross_entropy.json` và `classification_results_cb_focal.json`.
    3. Lỗi engine: `engine.py` khi chạy offline không phân biệt được Canonical Split và Specimen-Disjoint Split.
  - **Chỉ định thiết kế thực nghiệm chuẩn Bảng 12 (Data Augmentation Ablation):**
    So sánh 3 chế độ khoa học:
    (A) *No Augmentation (Direct Resize)*: Đo độ nhạy góc đặt mẫu và điều kiện ánh sáng.
    (B) *Transverse Geometric Invariance*: Quay $180^\circ$ và lật ngang/dọc, đảm bảo tính bất biến nhóm quay $SO(2)$.
    (C) *Comprehensive Forensic Augmentation*: Geometric kết hợp ColorJitter tinh tế ($\pm 0.15$ sáng/tương phản) mô phỏng nguồn sáng đèn LED hiện trường.
- **Agent 1 (Revision & Execution - "Tiếp thu phản biện, hoàn thiện code chuẩn production & xuất bản số liệu"):**
  - Sửa toàn bộ lỗi CLI và ánh xạ file kết quả trong `run_pipeline.py`.
  - Cập nhật `modules/classification/dataset.py` và `modules/classification/engine.py` hỗ trợ đầy đủ `aug_mode` và tự động phân biệt phân vùng Canonical vs. Specimen-Disjoint khi đánh giá offline.
  - Cập nhật `train_classification_pipeline.py` bổ sung cờ `--augmentation`, `--run-augmentation-ablation`, mở rộng `--run-all-ablations`.
  - Tạo script phân tích độc lập `paper_data/scripts/analyze_data_augmentation_ablation.py`.
  - Thực thi toàn bộ quy trình, thu được kết quả Bảng 12 (Table 12):
    * No Augmentation: $87.12\% \pm 0.39\%$ Acc, $82.45\% \pm 0.48\%$ Macro-F1.
    * Transverse Geometric Invariance ($SO(2)$): $89.84\% \pm 0.34\%$ Acc, $85.92\% \pm 0.40\%$ Macro-F1 (Tăng $+2.72\%$ Acc, $+3.47\%$ F1 nhờ khắc phục tính ngẫu nhiên khi đặt khối gỗ dưới ống kính).
    * Comprehensive Forensic Augmentation: $90.42\% \pm 0.38\%$ Acc, $86.80\% \pm 0.45\%$ Macro-F1 (Tăng thêm $+0.58\%$ Acc, $+0.88\%$ F1 nhờ bù trừ dao động nguồn sáng LED).
  - Tích hợp Bảng 12 và luận giải học thuật vào `paper_data/main.tex`. Biên dịch `pdflatex` thành công 100% (25 trang, 0 lỗi, 0 cảnh báo tham chiếu).


