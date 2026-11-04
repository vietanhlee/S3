# Báo Cáo Bàn Giao Hậu Kiểm (Sentinel Handoff Report)

**Dự án**: Kiểm toán rò rỉ mẫu vật (Specimen Leakage Audit) & Báo cáo kết quả âm trung thực (Honest Negative Results) trong Giám định Gỗ Vi thể Vĩ mô  
**Mã hồ sơ yêu cầu**: `ORIGINAL_REQUEST.md` (Approved Task Prompt)  
**Thời gian hoàn thành**: 2026-11-04T05:44:00Z  
**Phán quyết kiểm toán tối hậu**: **`VICTORY CONFIRMED`** (Thẩm định viên độc lập `teamwork_victory_auditor`)  

---

## 1. Observation (Các quan sát thực tế và bằng chứng nghiệm thu)

1. **Biên dịch LaTeX và Hồ sơ Xuất bản**:
   - Tệp nguồn bản thảo: `03_research_paper_specimen_invariance/paper/main.tex` (62,118 bytes).
   - Tệp xuất bản nghiệm thu: `03_research_paper_specimen_invariance/paper/main.pdf` (14 trang hoàn chỉnh, 447,884 bytes).
   - Kiểm tra log biên dịch: 0 fatal errors, 0 undefined citations (`[?]`), 0 undefined references (`??`).
   - Đồ thị nhúng PDF: 3/3 tệp vector PDF (`fig1_concept_invariant.pdf`, `fig2_framework_architecture.pdf`, `fig3_gradcam_comparison.pdf`) được tái tạo sạch, 0 overclaim, 0 loài ngoại lai.

2. **Dữ liệu & Thực nghiệm Bổ sung 11 Loài Fabaceae (S3)**:
   - Dữ liệu chuẩn xác: Đúng 11 loài họ Đậu (*Fabaceae*), 110 phôi vật lý (10 phôi/loài cân bằng tuyệt đối), 3,702 ảnh tiêu chuẩn.
   - Kiểm toán rò rỉ ngẫu nhiên trên 9 hạt giống (`[42, 123, 456, 789, 101, 202, 303, 404, 505]`):
     - Image-Level Random Split: Top-1 Test Acc = **$99.55\% \pm 0.12\%$** [95% CI: 99.46%, 99.64%].
     - Out-of-Specimen Voucher Split: Top-1 Test Acc = **$97.07\% \pm 0.95\%$** [95% CI: 96.34%, 97.80%].
     - Độ phóng đại do rò rỉ (Performance Inflation Gap): **$\Delta_{\text{inflation}} = +2.48 \pm 0.98\text{ pp}$** [95% CI: $+1.73, +3.23\text{ pp}$], paired $t$-test $t = 7.59, p = 0.0001 < 0.001$.
   - Báo cáo trung thực kết quả âm của phương pháp bất biến (GRL+vCLUB vs ERM Baseline):
     - Out-of-Specimen Test Acc: Baseline $97.07\% \pm 0.95\%$ vs GRL+vCLUB $96.88\% \pm 0.99\%$ ($\Delta = -0.19\text{ pp}$, 95% CI $[-1.16, +0.78]\text{ pp}$, $p = 0.66$ $t$-test, $p = 0.91$ Wilcoxon signed-rank).
     - Generalization Gap (GGSL): Baseline $2.74 \pm 0.92\text{ pp}$ vs GRL+vCLUB $2.82 \pm 0.96\text{ pp}$ ($\Delta = +0.08\text{ pp}$, 95% CI $[-0.35, +0.51]\text{ pp}$, $p = 0.67$).
     - Train SRI (MLP Probe): Baseline $0.5261$ vs GRL $0.5380$ ($\Delta = +0.0119$, 95% CI $[-0.0145, +0.0383]$, $p = 0.34$), đối chiếu chuẩn với điểm neo ImageNet $0.5667$.
     - Calibrated ECE: Dao động trong khoảng thực nghiệm $1.58\% - 1.93\%$.
     - Toàn bộ hàng kết quả của GAN trong các bảng đối sánh đã được unbold do không có ý nghĩa thống kê vượt trội.

3. **Chẩn đoán Ca Thất bại Phôi 50 & 53 (Seed 456)**:
   - Danh tính: Cả hai phôi đều thuộc *Guibourtia arnoldiana* (`.1` và `.3`).
   - Độ sắc nét quang học Laplacian: Phôi 50 đạt 113.81, Phôi 53 đạt 152.08 (thấp hơn nhiều so với mức $>1,000$ của các phôi chuẩn).
   - Cơ chế thất bại: Nhầm lẫn 100% sang loài chị em cùng chi (*Guibourtia ehie*) do cấu trúc giải phẫu tương đồng và ảnh bị mờ nét, khiến mô hình học thuộc đặc trưng quang học bề mặt. GRL không thể khôi phục được thông tin cấu trúc đã mất do chất lượng ảnh.

4. **Định lý Bengio & Grandvalet (2004) & Độ Phụ thuộc giữa các Split**:
   - Độ trùng lặp phôi giữa 9 seeds kiểm thử là $50.45\%$ (trung bình $27.75 / 55$ phôi), chứng minh các phân vùng phụ thuộc một phần và khoảng tin cậy tính trung bình có xu hướng lạc quan, khẳng định tính cần thiết của kỹ thuật Voucher-Level Cluster Bootstrap ($B=1000$).

---

## 2. Logic Chain (Chuỗi lập luận pháp y)

- **Từ Dữ liệu Thực nghiệm đến Định vị Khoa học**:
  Số liệu đo đạc trực tiếp trên 9 hạt giống cho thấy phân vùng ngẫu nhiên gây phóng đại kết quả $+2.48\text{ pp}$ ($p < 0.001$), trong khi phương pháp bất biến GRL+vCLUB không đem lại bất kỳ cải thiện nào có ý nghĩa thống kê so với ERM baseline (Acc $\Delta = -0.19\text{ pp}$, GGSL $\Delta = +0.08\text{ pp}$, SRI không giảm, tất cả CI đều chứa 0). Do đó, việc tái định vị bài báo sang "Kiểm toán rò rỉ mẫu vật và Báo cáo kết quả âm trung thực" là hoàn toàn phù hợp với thực tế dữ liệu và nâng cao giá trị phương pháp luận khoa học.
- **Từ Kiểm toán Nghiêm ngặt đến Phê duyệt Nghiệm thu**:
  Quy trình kiểm toán hai tầng (Auditor M3 trong vòng lặp điều phối và Victory Auditor độc lập tại chặng cuối) đã phát hiện và xử lý triệt để tàn dư overclaim trong đồ thị vector Hình 1 và mã nguồn sinh hình. Sau khi tái biên dịch và kiểm tra luồng nhị phân, toàn bộ hệ thống đã đạt trạng thái sạch tuyệt đối 100%.

---

## 3. Caveats (Các lưu ý khoa học)

1. **Hiệu ứng trần (Ceiling Effect)**: Độ chính xác cơ sở của ResNet-18 đã đạt $97.07\%$ trên 11 loài Fabaceae, khiến không gian cải thiện còn lại rất hẹp trước các ranh giới loài rõ ràng.
2. **Sự phụ thuộc phân vùng phôi**: Vì chỉ có 110 phôi vật lý được chia cho 9 phân vùng hạt giống, khoảng tin cậy cross-seed có tính chất hơi lạc quan; các phân tích trong bài đã ghi chú rõ ràng hạn chế này theo khuyến nghị thống kê của Bengio & Grandvalet (2004).

---

## 4. Conclusion (Kết luận)

Dự án đã hoàn thành trọn vẹn 100% tất cả các yêu cầu R1, R2, R3 và các tiêu chí nghiệm thu đề ra trong `ORIGINAL_REQUEST.md`. Văn bản bài báo, bảng biểu số liệu, đồ thị vector và mã nguồn thực nghiệm đã đồng nhất hoàn hảo, sẵn sàng cho việc nộp bài tới các tạp chí uy tín (Q1/Q2).

---

## 5. Verification Method (Phương thức tái kiểm chứng)

- Biên dịch LaTeX:
  ```powershell
  cd g:\S3_paper\03_research_paper_specimen_invariance\paper
  pdflatex -interaction=nonstopmode -halt-on-error main.tex
  bibtex main
  pdflatex -interaction=nonstopmode -halt-on-error main.tex
  pdflatex -interaction=nonstopmode -halt-on-error main.tex
  ```
- Kiểm tra hình ảnh & dữ liệu:
  ```powershell
  python generate_figures.py
  python -c "from dataset import scan_and_build_dataset; df = scan_and_build_dataset(); assert len(df)==3702 and df['taxon_name'].nunique()==11 and df['specimen_id'].nunique()==110"
  ```
