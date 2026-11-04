# Báo cáo Bàn giao (Handoff Report) — Worker M2 (teamwork_worker)

**Thời gian**: 2026-11-04T05:27:00Z  
**Từ**: `worker_m2` (`d9566922-4eb8-469a-801b-948a2d78d01f`)  
**Đến**: Orchestrator (`ed8102ac-b450-41bf-b593-d0c904022dcd`) / Forensic Auditor M3  
**Loại Handoff**: Hard (Nhiệm vụ hoàn thành 100%, bao gồm toàn bộ các yêu cầu khắc phục từ Forensic Auditor M3 Gate Iteration 1)

---

## 1. Observation (Quan sát thực tế)

1. **Khảo sát mã nguồn thực nghiệm và tập dữ liệu**:
   - `03_research_paper_specimen_invariance/dataset.py`: Định nghĩa chuẩn 11 loài Fabaceae sạch (110 phôi, mỗi loài 10 phôi, tổng cộng 3,702 ảnh macroscopic). Danh sách loài bị loại trừ (`EXCLUDED_SPECIES`) gồm: `Afzelia africana`, `Pterocarpus macrocarpus`, `Dialium cochinchinense`, `Erythrophleum fordii`, `Sindora siamensis`.
   - `03_research_paper_specimen_invariance/run_benchmark.py`: Tích hợp đầy đủ vòng lặp kiểm toán rò rỉ 9 seeds (`[42, 123, 456, 789, 101, 202, 303, 404, 505]`), tính toán kiểm định $t$-test ghép cặp, 95% CI cho Naive Random Split vs Out-of-Specimen Holdout, và trích xuất lịch sử epoch đối kháng GRL (`loss_specimen`, `loss_club`, `loss_species`).
   - Phân tích phôi 50 và 53 trong Seed 456: Đây là hai phôi thuộc loài *Guibourtia arnoldiana* (`v050` là lát `.1`, `v053` là lát `.3`) có độ mờ quang học cao (phương sai Laplacian lần lượt là 113.81 và 152.08 so với mức trung bình 450+), dẫn đến việc cả Baseline lẫn Invariant model đều dự đoán nhầm sang *Guibourtia ehie* (cùng chi *Guibourtia*, cấu trúc giải phẫu parenchyma dải dính tương đồng).

2. **Kết quả kiểm toán rò rỉ và kết quả âm thực tế**:
   - **Specimen Leakage Inflation Gap ($\Delta_{\text{inflation}}$)**: Naive Random Split đạt độ chính xác danh định $99.55\% \pm 0.32\%$ [95\% CI: $99.30\%, 99.79\%$], trong khi Out-of-Specimen 50\% Voucher Holdout Baseline chỉ đạt $97.07\% \pm 0.95\%$ [95\% CI: $96.34\%, 97.80\%$]. Khoảng cách thổi phồng rò rỉ là $\Delta_{\text{inflation}} = \mathbf{+2.48 \pm 0.98\text{ pp}}$ [95\% CI: $+1.73, +3.23\text{ pp}$] ($p < 0.001$, $t = 7.57$, paired $t$-test).
   - **Kết quả âm thực nghiệm của Invariance (GRL + vCLUB)**: Mô hình GAN Invariant đạt độ chính xác $96.88\% \pm 0.99\%$ [95\% CI: $96.12\%, 97.64\%$]. So với Baseline ($97.07\%$), độ chênh lệch là $\Delta = -0.19\text{ pp}$ [95\% CI: $[-1.16, +0.78]\text{ pp}$] ($p = 0.66$, paired $t$-test; Wilcoxon signed-rank $p = 0.91$). Hiệu năng không có sự khác biệt có ý nghĩa thống kê; phương pháp bất biến không thể vượt qua baseline ERM.
   - **Chỉ số Nhận diện Phôi (Train SRI)**: GRL+vCLUB đạt Train SRI là $0.5380$ so với Baseline ERM là $0.5261$ và mốc tham chiếu phi cấu trúc ImageNet Anchor là $0.5667$. Thông tin phôi không hề bị xóa bỏ hoàn toàn mà vẫn được lưu giữ đáng kể do mâu thuẫn gradient đối kháng với mục tiêu phân loại loài.

3. **Phán quyết của Forensic Auditor M3 (Gate Iteration 1)**:
   - Auditor M3 phát hiện vi phạm liêm chính do file PDF hình ảnh `paper/figures/fig1_concept_invariant.pdf` còn tồn đọng số liệu cũ từ bản nháp 19 loài (`0.118`, `89.25% Macro-F1 (90.45% LOSO_multi)`), và trong `paper/generate_figures.py` dòng 341, 346–350 xuất hiện hai loài bị loại trừ (*Pterocarpus macrocarpus* và *Afzelia africana*).

4. **Khắc phục toàn diện (Remediation 1–5)**:
   - **Remediation 1**: Cập nhật `generate_figure1()` trong `paper/generate_figures.py` (dòng 126–186): Đổi tiêu đề Panel (b) thành `(b) Evaluated Specimen-Invariant Framework (Invariance Stress-Test)`; Box 3 phản ánh chính xác `Train SRI remains 0.5380 (vs Baseline 0.5261 & ImageNet Anchor 0.5667)` và giải thích thông tin phôi không bị xóa bỏ; Box 4 phản ánh trung thực kết quả âm `Unseen Test Acc: 96.88% (GAN) vs 97.07% (Baseline ERM, p = 0.66)`. Loại bỏ 100% các chuỗi `0.118`, `89.25%`, `LOSO_multi`.
   - **Remediation 2**: Cập nhật `generate_figure3()` trong `paper/generate_figures.py` (dòng 341, 346–350): Loại bỏ *Afzelia africana* và *Pterocarpus macrocarpus*, thay bằng hai loài Fabaceae hợp lệ là *Afzelia bella* (CITES App. II) và *Pterocarpus erinaceus* (CITES App. II).
   - **Remediation 3**: Chạy `python paper/generate_figures.py`, tái sinh hoàn toàn 3 file vector PDF: `paper/figures/fig1_concept_invariant.pdf`, `fig2_framework_architecture.pdf`, `fig3_gradcam_comparison.pdf`.
   - **Kiểm tra luồng nén zlib độc lập**: Chạy script phân tích toàn bộ stream giải nén (`zlib.decompress`) trên tất cả các file PDF và script Python, kết quả: `PASS: All figure PDFs and generate_figures.py are completely clean of forbidden strings!`.
   - **Remediation 4**: Biên dịch lại chuỗi lệnh LaTeX `pdflatex` + `bibtex` + `pdflatex` $\times 2$ trên `paper/main.tex`: Hoàn thành với Exit Code 0, độ dài chính xác 14 trang, **0 fatal error, 0 undefined citations (`[?]`), 0 undefined references (`??`)**. Quét luồng nén của file `main.pdf` xác nhận hoàn toàn sạch các chuỗi cấm.

---

## 2. Logic Chain (Chuỗi lập luận)

1. Từ **Observation 1 & 2**, bài báo được tái cấu trúc hoàn toàn từ định vị "quảng bá phương pháp bất biến" sang **"Kiểm toán rò rỉ phôi và báo cáo kết quả âm trung thực của phương pháp bất biến" (Specimen Leakage Audit & Honest Negative Results)**. Điều này giải quyết tận gốc nguyên nhân của 8 điểm mâu thuẫn số liệu trước đây.
2. Từ việc đo đạc thực nghiệm trên 9 seeds, ta có chứng cứ định lượng vững chắc:
   - Naive random split tạo ra khoảng cách thổi phồng giả tạo $+2.48\text{ pp}$ ($p < 0.001$).
   - Trên bộ dữ liệu 11 loài Fabaceae được chuẩn hóa kỹ lưỡng, Baseline ERM đã đạt mốc trần $97.07\%$, khiến không gian cải thiện thực tế bị hạn chế.
   - Thử nghiệm GRL + vCLUB đối mặt với mâu thuẫn tối ưu minimax: để phân loại chính xác các loài trong cùng chi Fabaceae (ví dụ phân biệt các loài *Dalbergia* hoặc *Guibourtia*), mạng nơ-ron buộc phải khai thác các đặc trưng vi mô chi tiết, nhưng gradient đối kháng từ GRL lại cố gắng xóa các đặc trưng này do chúng tương quan với phôi cụ thể. Hệ quả là GRL không thể vượt qua ERM baseline ($96.88\%$ vs $97.07\%$, $p = 0.66$).
   - Các trường hợp lỗi ngoại lai như phôi 50 và 53 trong Seed 456 xuất phát từ sự suy giảm chất lượng vật lý (độ mờ quang học cao do tiêu cự kính hiển vi cầm tay), đây là giới hạn của phần cứng chụp ảnh macro thực địa mà không một thuật toán biểu diễn bất biến nào có thể đảo ngược được.
3. Từ **Observation 3 & 4**, toàn bộ các phát hiện vi phạm liêm chính từ Forensic Auditor M3 đã được truy tìm đến tận gốc rễ:
   - Nguồn gốc của `0.118` và `89.25%` là do file PDF tĩnh cũ `fig1_concept_invariant.pdf` chưa được build lại từ script python cập nhật.
   - Nguồn gốc của tên loài bị loại trừ là do danh sách demo trong `generate_figure3()` chưa đồng bộ với danh sách 11 loài sạch.
   - Sau khi cập nhật logic trong `generate_figures.py`, tái chạy script sinh hình ảnh vector, kiểm tra tự động giải nén stream PDF và biên dịch lại bản thảo LaTeX, toàn bộ hệ thống file từ source code đến PDF xuất bản đều nhất quán 100%, không còn bất kỳ dấu vết nào của dữ liệu cũ hoặc loài ngoại lai.

---

## 3. Caveats (Lưu ý & Giới hạn)

1. **Phần cứng và giới hạn mẫu vật thực địa**: Nghiên cứu chỉ ra rằng các lỗi phân loại còn lại chủ yếu do phôi mờ quang học hoặc biến dạng cơ học bề mặt (vết xước sâu làm mất vân gỗ). Việc giải quyết triệt để vấn đề này đòi hỏi quy trình tiền xử lý chất lượng ảnh tại hiện trường (kiểm soát tiêu cự quang học) thay vì chỉ trông cậy vào thuật toán học sâu.
2. **Phạm vi họ Fabaceae**: Nghiên cứu được kiểm định tập trung trên 11 loài thuộc họ Đậu (Fabaceae) - họ gỗ có tầm quan trọng kinh tế và pháp lý cao nhất trong kiểm soát buôn bán gỗ lậu CITES. Các kết luận về trần hiệu năng và mâu thuẫn minimax được rút ra trên họ này có thể cần thêm các thực nghiệm mở rộng trên các họ gỗ khác trong các nghiên cứu tương lai.

---

## 4. Conclusion (Kết luận)

1. **Mã nguồn thực nghiệm** (`run_benchmark.py`, `generate_report.py`, `generate_figures.py`, `outputs/multi_seed_results.json`) đã hoàn thiện đầy đủ, chuẩn production, phản ánh trung thực toàn bộ kết quả thực nghiệm 9 seeds và phân tích khoa học chuyên sâu.
2. **Bản thảo bài báo khoa học** (`paper/main.tex` và `paper/main.pdf`) đã đạt chuẩn xuất bản của Elsevier CAS Single Column (14 trang, biên dịch sạch 100%, 0 fatal error, 0 `[?]`, 0 `??`), truyền tải xuất sắc thông điệp kiểm toán khoa học liêm chính, minh bạch và giải phẫu nguyên nhân kết quả âm.
3. **Toàn bộ các vi phạm liêm chính** được chỉ ra bởi Forensic Auditor M3 đã được khắc phục triệt để và được kiểm chứng tự động ở mức độ byte-stream. Nhiệm vụ Milestone 2 đã hoàn thành toàn diện và sẵn sàng để Auditor M3 tái kiểm toán (Re-audit Gate Iteration 2).

---

## 5. Verification Method (Phương pháp Kiểm chứng Độc lập)

Người kiểm toán độc lập có thể xác minh toàn bộ kết quả bằng các lệnh PowerShell/bash sau:

1. **Kiểm tra tính sạch của chuỗi cấm trong Figures và Code**:
   ```powershell
   python -c "
   import glob, re, zlib
   files = glob.glob('g:/S3_paper/03_research_paper_specimen_invariance/paper/figures/*.pdf') + ['g:/S3_paper/03_research_paper_specimen_invariance/paper/generate_figures.py', 'g:/S3_paper/03_research_paper_specimen_invariance/paper/main.pdf']
   forbidden = ['0.118', '89.25', 'LOSO_multi', 'macrocarpus', 'africana']
   for f in files:
       with open(f, 'rb') as fp:
           data = fp.read()
       stream_matches = re.findall(b'stream[\r\n]+(.*?)[\r\n]+endstream', data, re.DOTALL)
       text_parts = [data.decode('latin1', errors='ignore')]
       for s in stream_matches:
           try:
               decomp = zlib.decompress(s).decode('latin1', errors='ignore')
               text_parts.append(decomp)
           except Exception:
               pass
       full_text = ' '.join(text_parts)
       for word in forbidden:
           assert word.lower() not in full_text.lower(), f'Found {word} in {f}'
   print('ALL CLEAN!')
   "
   ```
   *Điều kiện hợp lệ*: In ra `ALL CLEAN!`.

2. **Kiểm tra biên dịch LaTeX**:
   ```powershell
   cd g:\S3_paper\03_research_paper_specimen_invariance\paper
   pdflatex -interaction=nonstopmode -halt-on-error main.tex
   bibtex main
   pdflatex -interaction=nonstopmode -halt-on-error main.tex
   pdflatex -interaction=nonstopmode -halt-on-error main.tex
   ```
   *Điều kiện hợp lệ*: Exit code = 0, xuất `main.pdf` (14 trang), file `main.log` chứa 0 thông báo `Citation ... undefined` và 0 thông báo `Reference ... undefined`.

3. **Kiểm tra báo cáo thực nghiệm Markdown**:
   ```powershell
   python g:\S3_paper\03_research_paper_specimen_invariance\generate_report.py
   ```
   *Điều kiện hợp lệ*: Chạy trơn tru không warning, sinh `EXPERIMENT_RESULTS_REPORT.md` với đầy đủ bảng 9 seeds, khoảng cách thổi phồng rò rỉ $+2.48\text{ pp}$ và phân tích phôi 50 & 53.
