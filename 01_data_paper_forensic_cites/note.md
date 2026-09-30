# Ghi Chú Cập Nhật & Phản Biện Bài Báo IC4SDMacroWood

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

# 3. Chạy thử nhanh 1 seed duy nhất:
python run_pipeline.py --step classify --single-seed --epochs 3
```
