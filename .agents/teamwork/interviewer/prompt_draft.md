Sử dụng nhóm tối đa 3 agents. Viết lại và hoàn thiện bài báo khoa học trong thư mục `03_research_paper_specimen_invariance/paper/main.tex` và các tệp liên quan theo định vị "Kiểm toán rò rỉ mẫu vật (Specimen Leakage Audit) và Báo cáo kết quả âm trung thực đối với các phương pháp bất biến", đồng thời thực hiện các thực nghiệm bổ sung trong mã nguồn để giải quyết triệt để tất cả các phản biện khoa học.

Working directory: 03_research_paper_specimen_invariance
Integrity mode: development

## Requirements

### R1. Tái định vị luận điểm khoa học (Scientific Re-framing)
Chuyển đổi toàn diện trọng tâm và luận điểm của bài báo (Abstract, Introduction, Contributions, Method, Discussion, Conclusion) từ hướng "đề xuất phương pháp mới vượt trội" sang hướng "kiểm toán rò rỉ mẫu vật và đánh giá khách quan các phương pháp học biểu diễn bất biến":
- Xác lập 3 đóng góp khoa học chính: (1) định lượng hiện tượng rò rỉ mẫu vật (specimen leakage) trong nhận dạng gỗ vi mô, (2) đề xuất giao thức voucher-level kèm kỹ thuật bootstrap cấp phôi gỗ, (3) phân tích thực nghiệm cho thấy GRL+vCLUB không cải thiện có ý nghĩa thống kê so với ERM baseline trong thiết lập này kèm mổ xẻ nguyên nhân (ranh giới loài họ Đậu quá rõ ràng, trần hiệu năng, và động lực tối ưu hóa đối kháng).
- Sửa tiêu đề bài báo để loại bỏ khẳng định chưa được chứng minh (thay cụm "Mitigating" thành "Evaluating Invariance Methods" hoặc tương đương phù hợp).
- Báo cáo trung thực kết quả âm: GRL+vCLUB không cải thiện so với ERM baseline (Acc -0.19 pp, 95% CI [-1.16, +0.78], p = 0.66 t-test, p = 0.91 Wilcoxon; các chỉ số SRI, GGSL, ECE đều có CI chứa 0).
- Không in đậm toàn bộ hàng kết quả của phương pháp GAN/đối kháng trong các bảng số liệu; chỉ làm nổi bật ở những chỉ số thực sự có ý nghĩa vượt trội.

### R2. Xóa bỏ triệt để 8 mâu thuẫn giữa diễn giải văn bản và số liệu thực nghiệm
Rà soát và sửa đổi toàn bộ các câu trong bài báo đang mâu thuẫn với số liệu thực tế:
1. Sửa diễn giải về GGSL tại mục 4.3: Không khẳng định GAN "triệt tiêu GGSL → 0 pp" hay "kéo xuống +2.82 pp từ 1.04 pp", phản ánh chính xác số liệu bảng (baseline 2.74 pp, GAN 2.82 pp).
2. Sửa phát biểu về Train SRI: Không tuyên bố "Train SRI giảm mạnh", phản ánh đúng thực tế GAN (0.538) không giảm so với baseline (0.526).
3. Sửa đoạn so sánh với điểm neo ImageNet: Loại bỏ logic tự mâu thuẫn giữa các giá trị 0.5261, 0.5380 và điểm neo ImageNet 0.5667.
4. Sửa số liệu Baseline tại mục 4.2: Điều chỉnh mức "~98.5-99.0%" thành con số thực nghiệm chính xác trong bảng (97.07%).
5. Sửa con số ECE: Điều chỉnh phát biểu "ECE < 1%" thành khoảng thực tế trong bảng (1.58-1.93%).
6. Chuẩn hóa tên loài thực vật: Loại bỏ *Dialium cochinchinense* và *Pterocarpus macrocarpus* khỏi các phân tích giải phẫu của 11 loài nghiên cứu; chỉ dùng đúng danh sách 11 loài trong bộ dữ liệu S3.
7. Điều chỉnh tiêu đề và thuật ngữ không để xảy ra overclaim.
8. Tiết chế tuyên bố về nghiên cứu của bên thứ ba: Không khẳng định các công bố khác đạt 99.8-100% là "bị phồng số liệu", chỉ nêu rõ trên tập dữ liệu thực nghiệm của bài báo, việc chia ngẫu nhiên cấp ảnh gây phóng đại kết quả bao nhiêu điểm phần trăm.

### R3. Thực nghiệm bổ sung và phân tích mã nguồn
Triển khai và cập nhật các phân tích thực nghiệm trong mã nguồn `03_research_paper_specimen_invariance/`:
- Chạy đánh giá random-split trên cả 9 seeds để tính toán khoảng tin cậy (95% CI) cho độ phóng đại do rò rỉ phôi gỗ (leakage inflation gap), đưa kết quả có CI vào bài báo.
- Kiểm tra dữ liệu ảnh của phôi 50 và 53 (seed 456 - nơi baseline đạt 0%): Phân tích nguyên nhân (nhãn sai, chất lượng ảnh, hay đặc điểm giải phẫu dị biệt) và ghi nhận minh bạch trong phần thảo luận.
- Chẩn đoán cơ chế GRL: Ghi log loss của discriminator và kiểm tra hành vi đối kháng khi điều chỉnh trọng số đối kháng (`adv_weight`).
- Ghi chú rõ tính chất phụ thuộc một phần giữa 9 splits (do cùng lấy từ tập 110 phôi) khiến khoảng tin cậy trung bình giữa các seed có tính chất lạc quan.

## Acceptance Criteria

### LaTeX Compilation & Structure
- [ ] Tệp `paper/main.tex` biên dịch sạch ra `paper/main.pdf` (qua pdflatex hoặc latexmk), không có lỗi fatal, không có broken citation `[?]` hay broken reference `??`.
- [ ] Tiêu đề, Abstract, Contributions và Conclusion phản ánh đúng định vị "kiểm toán rò rỉ + kết quả âm trung thực", không còn từ ngữ tuyên bố phương pháp đề xuất vượt trội.

### Empirical Consistency & Contradiction Resolution
- [ ] Toàn bộ 8 điểm mâu thuẫn nêu trong R2 được xử lý triệt để; không còn bất kỳ câu khẳng định nào trong văn bản trái ngược với số liệu trong các bảng.
- [ ] Các bảng số liệu không in đậm toàn bộ dòng GAN; chỉ làm nổi bật nơi phương pháp thực sự tốt hơn có ý nghĩa thống kê.
- [ ] Tên các loài thực vật trong toàn bộ văn bản và chú thích hình ảnh khớp 100% với danh mục 11 loài thực tế của bộ dữ liệu S3.

### Supplementary Experiments & Diagnostics
- [ ] Mức độ phóng đại rò rỉ ngẫu nhiên (random-split leakage inflation gap) được báo cáo kèm khoảng tin cậy 95% CI trên 9 seeds.
- [ ] Báo cáo phân tích về hai trường hợp dị biệt phôi 50 và 53 được đưa vào bài báo minh bạch.
- [ ] Phần thảo luận phân tích rõ ràng nguyên nhân kết quả âm của GRL+vCLUB (cơ chế discriminator, ceiling effect của 11 loài họ Đậu, và sự phụ thuộc giữa các split).
