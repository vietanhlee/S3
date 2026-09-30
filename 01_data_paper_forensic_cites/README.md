# ForensicMacroWood-CITES: Dataset Paper & Technical Validation Pipeline
## Elsevier *Data in Brief* Dataset Publication

[![Data Paper: Elsevier Data in Brief](https://img.shields.io/badge/Journal-Elsevier_Data_in_Brief-orange.svg)]()
[![Dataset Status: Verified](https://img.shields.io/badge/Dataset-Verified-brightgreen.svg)]()
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Thư mục này chứa toàn bộ quy trình tiền xử lý, kiểm soát chất lượng hình ảnh, sinh mã băm mật mã và băm tri giác khử trùng lặp 2 cấp độ, cùng pipeline huấn luyện mô hình kỹ thuật (Technical Validation) cho bộ dữ liệu **ForensicMacroWood-CITES** chuẩn bị xuất bản trên tạp chí **Elsevier *Data in Brief***.

---

## 📌 1. Giới thiệu Bộ Dữ liệu

Bộ dữ liệu cung cấp **6,410 ảnh chụp mặt cắt ngang vĩ mô (transverse cross-section)** độ phân giải cao ($512 \times 512$ pixels, phóng đại quang học $20\times$), thu thập từ **116 phôi gỗ tiêu bản vật lý** thuộc **18 loài thực vật nguy cấp và thương mại** (Họ Đậu - Fabaceae) thuộc 5 chi:
* ***Afzelia*** (4 loài): *A. africana*, *A. bipindensis*, *A. pachyloba*, *A. xylocarpa* (Toàn bộ CITES Phụ lục II).
* ***Dalbergia*** (4 loài): *D. cochinchinensis*, *D. latifolia*, *D. oliveri*, *D. tonkinensis* (Toàn bộ CITES Phụ lục II).
* ***Guibourtia*** (3 loài): *G. demeusei*, *G. pellegriniana*, *G. tessmannii* (Gỗ Bubinga thương mại đối chiếu).
* ***Pterocarpus*** (4 loài): *P. angolensis*, *P. erinaceus* (CITES App. II), *P. macrocarpus*, *P. soyauxii*.
* ***Sindora*** (3 loài): *S. cochinchinensis*, *S. siamensis*, *S. velutina* (Gỗ Gõ Mật đối chiếu).

---

## 📂 2. Cấu trúc Thư mục

```
01_data_paper_forensic_cites/
├── modules/                               # Các modules chức năng lõi
│   ├── metadata_extraction.py            # Trích xuất EXIF, độ phân giải, kiểm tra phôi gỗ
│   ├── image_quality.py                  # Tính độ nét Laplacian (sigma^2 >= 100)
│   ├── deduplication.py                  # Mã băm SHA-256 (Level 1) và dHash/pHash (Level 2)
│   └── classification.py                 # DataLoader, Focal Loss, và vòng lặp huấn luyện
├── utils/                                # Tiện ích thống kê và hiển thị
│   ├── statistical_testing.py            # Tính toán Mean +- Std, 95% CI, p-value qua 5 seeds
│   └── visualization.py                  # Xuất ma trận nhầm lẫn chuẩn Elsevier
├── out/                                  # Thư mục đầu ra của pipeline
│   ├── metadata/metadata.csv             # Bảng siêu dữ liệu phôi gỗ và thông số ảnh
│   └── splits/split_canonical.csv        # Phân chia dữ liệu chuẩn xác thực
├── paper_data/                           # Bản thảo LaTeX Elsevier Data in Brief
│   ├── main.tex                          # File bài báo chính
│   ├── refs.bib                          # Danh mục tài liệu tham khảo
│   └── fig/                              # Biểu đồ và hình ảnh minh họa cho bài báo
├── requirements.txt                      # Danh sách thư viện phụ thuộc
├── run_pipeline.py                       # Master CLI Runner (Tự động hóa 2 bước)
└── train_classification_pipeline.py      # Script huấn luyện phân loại ConvNeXt-Tiny
```

---

## ⚙️ 3. Quy trình Tự động hóa 2 Bước (Automated Pipeline)

Quy trình xử lý dữ liệu được chia làm 2 giai đoạn độc lập:
1. **Bước 1 (`assets`)**: Kiểm soát chất lượng hình ảnh, lọc mờ bằng biến sai toán tử Laplacian ($\sigma^2 \ge 100$), tính mã băm mật mã SHA-256 (khử trùng lặp tuyệt đối) và mã băm tri giác dHash/pHash với khoảng cách Hamming $\le 3$ (khử trùng lặp tương đối).
2. **Bước 2 (`classify`)**: Huấn luyện mô hình ConvNeXt-Tiny qua **3 random seeds** (42, 123, 456), tự động tổng hợp kết quả thống kê (Mean $\pm$ Std, khoảng tin cậy 95\% CI) và vẽ ma trận nhầm lẫn chuẩn publication.

---

## 💻 4. Hướng dẫn Lệnh Chạy (Terminal Commands)

Mở terminal (PowerShell trên Windows hoặc Bash trên Linux) và chuyển vào thư mục này:
```bash
cd g:/S3_paper/01_data_paper_forensic_cites
```

### Lệnh 1: Chạy Toàn Bộ Pipeline Tự Động (Chuẩn bài báo)
Chạy lần lượt từ xử lý dữ liệu, trích xuất metadata đến huấn luyện kiểm định 3 seeds:
```powershell
python run_pipeline.py --all --data-dir "g:/S3_paper/S3"
```

### Lệnh 2: Chỉ chạy Bước 1 (Sinh Metadata & Khử trùng lặp Assets)
```powershell
python run_pipeline.py --step assets --data-dir "g:/S3_paper/S3"
```

### Lệnh 3: Chỉ chạy Bước 2 (Huấn luyện Baseline Đa Hạt Giống 3 Seeds)
Tự động chạy qua 3 seeds (42, 123, 456), lưu checkpoint và vẽ ma trận nhầm lẫn trung bình:
```powershell
python run_pipeline.py --step classify
```

### Lệnh 4: Chạy Thử Nhanh (Single-Seed Test)
Chỉ chạy 1 seed (seed 42) để kiểm tra luồng huấn luyện nhanh:
```powershell
python run_pipeline.py --step classify --single-seed
```

### Lệnh 5: Huấn luyện Trực tiếp Bằng CLI Nâng Cao (`train_classification_pipeline.py`)
Tùy chỉnh linh hoạt backbone, hàm mất mát, batch size và epoch:
```powershell
# Huấn luyện với Focal Loss (khuyến nghị cho dữ liệu CITES mất cân bằng)
python train_classification_pipeline.py --model-name convnext_tiny --loss focal --epochs 22 --batch-size 64 --lr 5e-4

# Huấn luyện đối chiếu với Cross-Entropy chuẩn
python train_classification_pipeline.py --model-name convnext_tiny --loss cross_entropy --epochs 22 --batch-size 64

# Tự động huấn luyện lần lượt cả 2 loss để so sánh hiệu năng
python train_classification_pipeline.py --run-both --epochs 22
```

---

## 📊 5. Kết quả Kiểm định Kỹ thuật (Technical Validation Results)

Kết quả kiểm định trên tập Test độc lập qua 3 random seeds (Mean $\pm$ Std):
* **Top-1 Accuracy**: $90.42\% \pm 0.38\%$ (Khoảng tin cậy 95\% CI: $[89.48\%, 91.36\%]$)
* **Macro-Averaged Precision**: $91.67\% \pm 0.42\%$
* **Macro-Averaged Recall**: $88.18\% \pm 0.51\%$
* **Macro-Averaged F1-Score**: $86.80\% \pm 0.45\%$ (Khoảng tin cậy 95\% CI: $[85.68\%, 87.92\%]$)

Mọi biểu đồ ma trận nhầm lẫn và bảng biểu LaTeX được tự động kết xuất vào thư mục `paper_data/fig/` để nhúng trực tiếp vào file bài báo `paper_data/main.tex`.
