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

## 2. Công Cụ Benchmark Đa Seed Chuẩn Production: `run_multiseed_baselines.py`
- Tệp thực thi: [run_multiseed_baselines.py](file:///g:/S3_paper/run_multiseed_baselines.py)
- Hỗ trợ chạy **5 seeds** (`[42, 123, 456, 789, 1024]`), mỗi seed **17 epochs**, phân tách learning rate (`1e-4` backbone, `5e-4` head).
- Đánh giá song song 3 phương pháp:
  1. `linear_probe`: Đánh giá vector đặc trưng 768-d của ConvNeXt-Tiny (ImageNet-1K).
  2. `cross_entropy`: Standard Cross-Entropy Baseline.
  3. `focal`: Multiclass Focal Loss ($\gamma = 2.0$, scalar $\alpha = 0.25$).
- Tự động tính toán: Giá trị trung bình ($\mu$), độ lệch chuẩn ($\sigma$), khoảng tin cậy 95% CI (Student's $t$, $df=4$), kiểm định Paired $t$-test, $p$-value và Cohen's $d$.
- Báo cáo chi tiết đã lưu tại: [docs/multiseed_baseline_benchmark_report.md](file:///g:/S3_paper/docs/multiseed_baseline_benchmark_report.md).

---

## 3. Lệnh Thực Thi Cho Bạn Trên Terminal / CMD
```powershell
# Chạy benchmark 5 seeds x 17 epochs:
python run_multiseed_baselines.py --epochs 17 --seeds 42 123 456 789 1024 --gpu 0

# Hoặc chạy thử nhanh 1 seed:
python run_multiseed_baselines.py --epochs 3 --seeds 42 --gpu 0
```
