# Auditing Specimen Leakage and Evaluating Invariance Methods in Macroscopic Timber Forensics: An Empirical Benchmark and Honest Negative Results
## Research Paper 2: Voucher-Aware Evaluation Protocols, Leakage Audits, and Disentangled Invariant Representations

[![Target Journal: Elsevier Pattern Recognition](https://img.shields.io/badge/Target_Journal-Elsevier_PR-blue.svg)]()
[![Impact Factor: 7.5](https://img.shields.io/badge/Impact_Factor-7.5_(Q1)-success.svg)]()
[![Target Journal: Elsevier COMPAG](https://img.shields.io/badge/Target_Journal-Elsevier_COMPAG-green.svg)]()
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Thư mục này chứa toàn bộ mã nguồn cấp sản xuất (production-grade), khung kiểm toán rò rỉ mẫu vật (**Specimen Leakage Audit**), giao thức đánh giá ngoại suy cấp phôi vật lý (**Voucher-Aware Multi-Split Benchmark: 9 Random Seeds**) và kiểm chứng thực nghiệm trung thực đối với phương pháp biểu diễn bất biến (**Adversarial Invariant Framework: Masked Softmax GRL + Variational CLUB Bottleneck** trên nền **ResNet-18**).

---

## 📌 1. Trọng Tâm Khoa Học & Định Vị Bài Báo (Scientific Narrative Pivot)

Thay vì phóng đại hiệu năng bằng các tuyên bố thiếu căn cứ, nghiên cứu này tập trung vào kiểm toán rò rỉ mẫu vật thực chất và công bố kết quả âm tính trung thực (honest negative results):

```mermaid
flowchart TD
    subgraph Problem["Vấn Đề Cốt Lõi: Shortcut Learning Cấp Mẫu Vật"]
        A["Bộ dữ liệu ảnh gỗ vi thể vĩ mô<br/>(Nhiều ảnh cắt từ cùng 1 phôi vật lý)"]
        B["Naive Image-Level Random Split<br/>(Ảnh cùng phôi rải vào cả Train và Test)"]
        C["Ảo Tưởng An Toàn (Optimism Bias)<br/>Acc 'ảo' 99.55%<br/>Mạng học thuộc vết xước/màu sắc của phôi"]
        A --> B --> C
    end

    subgraph Audit["Giải Pháp 1: Giao Thức Đánh Giá Chống Rò Rỉ"]
        D["Kiểm Toán Rò Rỉ Mẫu Vật (Leakage Audit)<br/>Đo khoảng cách phóng đại hiệu năng (+2.48 pp, p < 0.001)"]
        E["Voucher-Aware Stress-Test (50% Holdout)<br/>55 phôi kiểm thử hoàn toàn ngoại suy"]
        F["Thống Kê Vững Chắc Cấp Phôi (Statistical Power)<br/>9 Seeds, Wilcoxon Test & Cluster Bootstrap (B=1000)"]
        C --> D
        D --> E --> F
    end

    subgraph Mitigation["Giải Pháp 2: Đánh Giá Trung Thực Phương Pháp Bất Biến"]
        G["Extractor E_θ (ResNet-18)"]
        H["Conditioned Discriminator D_ψ (Masked Softmax)"]
        I["Variational CLUB Bottleneck min I(Z; S | Y)"]
        J["Hiệu Chuẩn Tin Cậy Post-hoc Temperature Scaling (ECE 1.58% - 1.93%)"]
        E --> G
        G --> H
        G --> I
        G --> J
    end
```

### Các phát hiện và đóng góp chính:
1. **Kiểm toán định lượng hiện tượng rò rỉ mẫu vật (Specimen Leakage Audit across 9 Seeds)**:
   - Chứng minh thực nghiệm rằng phương pháp chia ngẫu nhiên cấp ảnh (*Naive Random Split*) làm **phóng đại độ chính xác thêm +2.48 pp [95% CI: +1.73, +3.23 pp]** ($p < 0.001$, từ 97.07% lên 99.55%) do mạng nơ-ron học thuộc đặc trưng cá thể phôi (*Specimen Shortcut Phenotype Bias - SSPB*).
2. **Trần hiệu năng giải phẫu họ Đậu (Anatomical Margin & Baseline Ceiling)**:
   - Khi stress-test trên 55 phôi lạ (50% holdout), mô hình Baseline ResNet-18 đạt chính xác **97.07% ± 0.95% [96.34%, 97.80%]** do khoảng cách phân tách cấu trúc giải phẫu vĩ mô liên loài giữa 11 loài Fabaceae sạch tương đối rõ ràng.
   - Mô hình đối kháng GRL+vCLUB đạt **96.88% ± 0.99% [96.12%, 97.64%]** ($\Delta = -0.19\text{ pp}$, 95% CI [-1.16, +0.78] pp, paired t-test $p = 0.66$, Wilcoxon $p = 0.91$). Cả hai mô hình tương đương về mặt thống kê, không có sự vượt trội của GRL.
3. **Báo cáo kết quả tiêu cực trung thực & Phân tích nguyên nhân gốc rễ (Honest Negative Results)**:
   - **Generalization Gap (GGSL)**: Baseline đạt 2.74 pp so với GAN 2.82 pp ($\Delta = +0.08\text{ pp}$, $p > 0.05$). GRL không kéo giảm được GGSL.
   - **Chỉ số rò rỉ mẫu vật (Train SRI)**: Baseline đạt 0.5261 so với GAN 0.5380 (điểm neo ImageNet chưa fine-tune là 0.5667). GRL không làm giảm thêm SRI so với ERM.
   - **Xung đột tối ưu hóa đối kháng (Minimax Gradient Conflict)**: Gradient đối kháng triệt tiêu đặc trưng phôi đồng thời làm nhiễu các dấu hiệu vi thể quan trọng của loài, dẫn đến xung đột tối ưu.
   - **Phân tích ca thất bại (Seed 456)**: Phôi 50 & 53 của *Guibourtia arnoldiana* có độ mờ quang học cao (Laplacian variance 113.8 & 152.1) bị phân loại sai sang *Guibourtia ehie*; GRL không thể khôi phục các chi tiết giải phẫu đã bị mất do mờ nét.
   - **Hiệu chuẩn độ tin cậy**: Post-hoc Temperature Scaling trên Val ngoại suy đưa ECE về mức 1.58% - 1.93%.

---

## 📂 2. Cấu trúc Thư mục Khoa học Chuẩn Hóa (Clean Production Layout)

```
03_research_paper_specimen_invariance/
├── README.md                          # Tài liệu hướng dẫn & giải trình khoa học toàn diện
├── requirements.txt                   # Danh mục thư viện phụ thuộc
├── run_benchmark.py                   # Master Benchmark Pipeline (Dynamic 4-partition, 9 Seeds, Leakage Audit, ResNet-18)
├── run_all.sh                         # 1-Click Master Bash Script (Kaggle / Linux GPU - Mặc định 9 Seeds, ResNet-18)
│
├── dataset.py                         # Module Dataset: Quét ảnh đĩa, Naive Random Split & Dynamic Voucher Split
├── models_gan.py                      # Kiến trúc: ResNet-18 Baseline, Masked Softmax GRL, Vectorized CLUB, Temperature Scaling
├── evaluate_dual.py                   # Script đánh giá đa giao thức: Seen/Unseen, Train/Test SRI Probing, Calibrated ECE
├── generate_report.py                 # Bộ biên dịch báo cáo Markdown khoa học, Kiểm toán Rò rỉ & Cluster Bootstrap
│
└── paper/                             # Bản thảo bài báo khoa học Elsevier Pattern Recognition
    ├── main.tex                       # File LaTeX manuscript hoàn chỉnh
    ├── main.pdf                       # Bản PDF biên dịch
    ├── refs.bib                       # Cơ sở dữ liệu trích dẫn BibTeX
    ├── cas-sc.cls & enumitem.sty      # Template Elsevier CAS-SC
    ├── generate_figures.py            # Script vẽ vector đồ thị khoa học PDF
    ├── RESPONSE_TO_REVIEWERS.md       # Hồ sơ giải trình & phản biện chi tiết
    └── figures/                       # Sơ đồ kiến trúc & đồ thị vector PDF
```

---

## 🔬 3. Giao thức Thực nghiệm Thách thức (50% Voucher Holdout & Leakage Audit)

Thực nghiệm được triển khai trên 11 loài họ Đậu Fabaceae sạch (110 phôi gỗ vật lý, 10 phôi/loài cân bằng tuyệt đối):

| Phân Vùng Dữ Liệu | Số Phôi / Loài | Tổng Số Phôi | Tỷ Lệ Mẫu Vật | Vai Trò & Nghiệp Vụ Khoa Học |
| :--- | :---: | :---: | :---: | :--- |
| **Naive Random Split (Audit)** | — | — | Toàn bộ ảnh | Phân chia ngẫu nhiên từng ảnh để đo độ phồng hiệu năng (Inflation Gap) |
| **Out-of-Specimen Test** | **5 phôi** | **55 phôi** | **50%** | **Stress-test kiểm thử ngoại suy thực tế trên các khúc gỗ hoàn toàn mới** |
| **Out-of-Specimen Val** | **1 phôi** | **11 phôi** | **10%** | **Thẩm định độc lập để Early Stopping & Calibrate Temperature** |
| **In-Specimen Train Set** | 4 phôi | 44 phôi | 32% (80% pool) | Huấn luyện mạng nơ-ron nhận diện loài và triệt tiêu phôi gỗ |
| **In-Specimen Seen Test** | 4 phôi | 44 phôi | 8% (20% pool) | Kiểm thử nội bộ in-distribution (đo lường độ sụt giảm GGSL) |

---

## 💻 4. Hướng dẫn Lệnh Chạy (Terminal Commands)

### Lệnh 1: Chạy 1-Click Toàn bộ Thực nghiệm 9 Seeds trên Kaggle GPU (T4 / P100)
Chạy script bash tự động tích hợp sẵn cả Leakage Audit và 9 Seeds:
```bash
# Trên Kaggle hoặc Linux GPU:
bash run_all.sh
```

### Lệnh 2: Chạy Trực tiếp Python CLI với Tham số Tùy chỉnh (`run_benchmark.py`)
```bash
python run_benchmark.py \
    --image_root "/kaggle/input/datasets/b23dckh002lvitanh/s3-origin/S3" \
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

### Lệnh 3: Đánh giá Độc lập Checkpoint Mô hình (`evaluate_dual.py`)
```bash
python evaluate_dual.py \
    --baseline_ckpt outputs/baseline/best_baseline.pth \
    --gan_ckpt outputs/gan/best_gan.pth \
    --output_file outputs/comparison_results.json \
    --backbone resnet18
```

---

## 📊 5. Báo Cáo Thực Nghiệm Thực Tế (Empirical Results)

Toàn bộ kết quả thực nghiệm được tổng hợp tự động và minh bạch từ quá trình chạy thực tế trên GPU qua script `run_benchmark.py`:
* **Báo cáo khoa học chính thức**: Xem chi tiết tại [EXPERIMENT_RESULTS_REPORT.md](EXPERIMENT_RESULTS_REPORT.md)
* **Kết quả JSON chi tiết**: Lưu trữ tại `outputs/multi_seed_results.json`
