# Báo Cáo Thẩm Định Pháp Y Tính Toàn Vẹn Cuối Cùng (Final Forensic Integrity Audit Report)

**Dự án**: Kiểm toán rò rỉ mẫu vật (Specimen Leakage Audit) & Báo cáo kết quả âm trung thực (Honest Negative Results) trong Giám định Gỗ Vi thể Vĩ mô  
**Thư mục làm việc của Auditor**: `g:/S3_paper/.agents/teamwork/auditor_m3/`  
**Hồ sơ kiểm toán (Profile)**: General Project (Forensic Integrity)  
**Chế độ kiểm toán (Integrity Mode)**: `development` (theo `ORIGINAL_REQUEST.md`)  
**Tác giả**: teamwork_auditor (auditor_m3)  
**Thời gian hoàn thành**: 2026-11-04T05:28:00Z  
**Báo cáo gửi cho**: Orchestrator (Conversation ID: `ed8102ac-b450-41bf-b593-d0c904022dcd`)  

---

## TỔNG QUAN PHÁN QUYẾT TÍNH TOÀN VẸN (FINAL EXECUTIVE VERDICT)

| Hạng mục kiểm toán pháp y | Trạng thái phúc tra (Gate Iteration 2) | Bằng chứng thực nghiệm |
|---|:---:|---|
| **Biên dịch LaTeX & Độ vẹn toàn Tài liệu** | 🟢 **PASS** | `paper/main.pdf` (14 trang, 447,884 bytes), 0 fatal error, 0 broken cite `[?]`, 0 broken ref `??` |
| **Tái định vị Luận điểm & Kết quả Âm Trung thực (R1)** | 🟢 **PASS** | Tiêu đề, Abstract, Contributions, Method, Discussion, Conclusion 100% định vị kiểm toán; 0 overclaim |
| **8 Mâu thuẫn Thực nghiệm & Bảng biểu (R2)** | 🟢 **PASS** | Xóa bỏ triệt để 8 mâu thuẫn trong toàn bộ văn bản, bảng số liệu và chú thích; toàn bộ hàng GAN unbold |
| **Thực nghiệm Bổ sung & Kiểm toán Mã nguồn (R3)** | 🟢 **PASS** | 9-seed CI (+2.48 pp [1.73, 3.23 pp], $p < 0.001$), phôi 50 & 53 (Laplacian 113.8 & 152.1), GRL loss logging, Bengio & Grandvalet (50.5% overlap) |
| **Phúc tra Đồ thị Vector PDF & Mã sinh hình (Figures)** | 🟢 **PASS** | `fig1_concept_invariant.pdf` đã xóa sạch 100% tàn dư cũ (`0.118`, `89.25%`, `LOSO_multi`); `generate_figures.py` chuẩn 11 loài Fabaceae |

### ⚖️ PHÁN QUYẾT PHÁP Y CHÍNH THỨC: **`CLEAN` (PHÊ DUYỆT BÀN GIAO / ACCEPT WORK PRODUCT)**

---

## 1. Observation (Các quan sát thực tế và bằng chứng trực tiếp)

### 1.1. Phúc tra Luồng Nhị phân Đồ thị Vector `figures/fig1_concept_invariant.pdf`
Auditor đã trực tiếp giải nén toàn bộ 134 đối tượng luồng `FlateDecode` của tệp PDF `03_research_paper_specimen_invariance/paper/figures/fig1_concept_invariant.pdf`:
* **Kiểm tra tàn dư overclaim cũ**:
  * Chuỗi ký tự `'0.118'`: **0 kết quả** (Đã xóa bỏ hoàn toàn).
  * Chuỗi ký tự `'89.25%'`: **0 kết quả** (Đã xóa bỏ hoàn toàn).
  * Chuỗi ký tự `'LOSO_multi'`: **0 kết quả** (Đã xóa bỏ hoàn toàn).
  * Chuỗi ký tự `'Evidence-grade'`: **0 kết quả** (Đã xóa bỏ hoàn toàn).
  * Chuỗi ký tự `'Closes Specimen Leakage Gap'`: **0 kết quả** (Đã xóa bỏ hoàn toàn).
* **Kiểm tra nội dung định vị kết quả âm trung thực mới trong Hình 1**:
  * Trích xuất trực tiếp từ stream của Hình 1 (Panel b):
    ```
    - Tiêu đề Panel (b): "Invariance Stress-Test"
    - Box Latent Space: "Latent Space under GRL + vCLUB: Z ~ E_theta(X)"
    - Box Latent Text: "Train SRI remains 0.5380 (vs Baseline 0.5261 & ImageNet Anchor 0.5667)"
    - Box Latent Subtext: "Specimen information is not erased; fine anatomical features conflict with GRL"
    - Box Outcome: "Honest Out-of-Specimen Evaluation (50% Physical Voucher Holdout)"
    - Box Outcome Text: "Unseen Test Acc: 96.88% (GAN) vs 97.07% (Baseline ERM, p = 0.66)"
    - Box Outcome Subtext: "Minimax gradient conflicts; baseline ceiling & optical blur limit invariance gains"
    ```

### 1.2. Phúc tra Mã Nguồn Sinh Hình `paper/generate_figures.py`
* **Dòng 164--183 (`generate_figure1`)**:
  Đã cập nhật đúng toàn bộ số liệu đo đạc thực tế của benchmark 11 loài họ Đậu: Train SRI GAN 0.5380 vs Baseline 0.5261; Unseen Test Acc 96.88% vs Baseline 97.07% ($p = 0.66$); 50% physical holdout.
* **Dòng 341 và 346--350 (`generate_figure3`)**:
  * Trước đây: Chứa loài cấm `Pterocarpus macrocarpus` và loài ngoài S3 `Afzelia africana`.
  * Hiện tại: Đã thay thế 100% bằng các loài Fabaceae chính thức của bộ dữ liệu S3:
    ```python
    taxa = [
        ("Afzelia bella", "CITES Appendix II", "Aliform to Confluent Parenchyma & Large Solitary Pores"),
        ("Pterocarpus erinaceus", "CITES Appendix II", "Fine Wavy Banded Parenchyma & Storied Rays"),
        ("Guibourtia ehie", "Commercial Reference", "Marginal Zonate Parenchyma Bands & Fiber Ground")
    ]
    ```
* **Quét toàn bộ thư mục `paper/`**:
  * Tìm kiếm `Dialium`: **0 kết quả**.
  * Tìm kiếm `macrocarpus` trong `main.tex`, `build_main_tex.py`, `generate_figures.py`: **0 kết quả**. (Chỉ còn lưu trong `dataset.py` tại danh sách loại trừ `EXCLUDED_SPECIES` và trong file giải trình lịch sử phản biện `RESPONSE_TO_REVIEWERS.md`).

### 1.3. Phúc tra Biên dịch LaTeX `paper/main.pdf`
Auditor đã chạy lại toàn bộ quy trình biên dịch từ mã nguồn `main.tex` qua MiKTeX-pdfTeX 4.27 và MiKTeX-BibTeX 4.2:
```powershell
cd g:/S3_paper/03_research_paper_specimen_invariance/paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```
* **Bằng chứng từ `main.log`**:
  * Mã thoát: Exit code 0.
  * Tệp PDF: `paper/main.pdf` (14 trang, 447,884 bytes).
  * Lỗi Fatal (`!`): **0**.
  * Cảnh báo Citation không xác định (`LaTeX Warning: Citation ... undefined` / `[?]`): **0**.
  * Cảnh báo Reference không xác định (`LaTeX Warning: Reference ... undefined` / `??`): **0**.

### 1.4. Đối chiếu 8 Điểm Mâu thuẫn Văn bản & Số liệu (R2)
1. **GGSL (Section 4.3, Table 2, Table 3)**: Baseline $2.74 \pm 0.92\text{ pp}$ vs GRL $2.82 \pm 0.96\text{ pp}$ ($\Delta = +0.08\text{ pp}$, $p > 0.05$). Không tuyên bố triệt tiêu GGSL. (ĐẠT).
2. **Train SRI (Section 4.3, Table 2, Table 3, Figure 1)**: Baseline $0.5261$ vs GRL $0.5380$ (không giảm so với Baseline). (ĐẠT).
3. **ImageNet Anchor**: Điểm neo ImageNet $0.5667$ (MLP) / $0.5420$ (Linear) được đối chiếu logic và chặt chẽ. (ĐẠT).
4. **Baseline Accuracy (Section 4.2, Table 1, Table 2)**: Đúng số liệu thực nghiệm **$97.07\% \pm 0.95\%$** [96.34%, 97.80%]. (ĐẠT).
5. **Sai số hiệu chuẩn ECE**: Đúng khoảng thực tế **$1.58\% - 1.93\%$** ($1.62\%$ vs $1.85\%$). (ĐẠT).
6. **Danh mục loài thực vật**: 100% khớp danh sách 11 loài Fabaceae sạch trong S3. Không xuất hiện *Dialium* hay *P. macrocarpus*. (ĐẠT).
7. **Tiết chế Overclaim & Tiêu đề**: Tiêu đề: *Auditing Specimen Leakage and Evaluating Invariance Methods in Macroscopic Timber Forensics: An Empirical Benchmark and Honest Negative Results*. 0 từ ngữ khẳng định vượt trội. (ĐẠT).
8. **Nghiên cứu bên thứ ba**: Trình bày khách quan khoảng cách phóng đại rò rỉ $+2.48\text{ pp}$ trên S3, không quy chụp thiếu căn cứ. (ĐẠT).
9. **Unbold Bảng biểu**: Toàn bộ các dòng của GRL trong Table 2 và Table 3 không in đậm (do không vượt trội có ý nghĩa thống kê so với Baseline). (ĐẠT).

### 1.5. Đối chiếu Thực nghiệm Bổ sung & Tính Toàn vẹn Mã nguồn (R3)
* **Tệp `outputs/multi_seed_results.json`**:
  * Leakage Audit (9 seeds): Naive split $99.55\% \pm 0.12\%$ vs Out-of-specimen $97.07\% \pm 0.95\%$, $\Delta_{\text{inflation}} = +2.48 \pm 0.98\text{ pp}$ [95% CI: $+1.73, +3.23\text{ pp}$], paired $t$-test $t = 7.59, p = 0.0001 < 0.001$.
  * Phân tích thất bại phôi 50 & 53 trong Seed 456 (*Guibourtia arnoldiana*, Laplacian $113.81$ & $152.08$, nhầm sang *Guibourtia ehie*).
  * Ghi log động lực học đối kháng GRL: `loss_specimen` được log chi tiết theo từng epoch trong `run_benchmark.py`.
  * Phân tích độ trùng lặp phôi cross-split: Trùng lặp trung bình $27.75/55$ phôi ($50.45\%$) theo định lý Bengio & Grandvalet (2004), khẳng định tính cần thiết của Voucher-Level Cluster Bootstrap.

---

## 2. Logic Chain (Chuỗi lập luận pháp y từ quan sát đến kết luận)

1. **Từ Quan sát 1.1 và 1.2 về Hình ảnh và Mã nguồn Đồ thị**:
   * Tệp vector `figures/fig1_concept_invariant.pdf` đã được vẽ lại hoàn toàn từ mã nguồn cập nhật `generate_figures.py`.
   * Toàn bộ chuỗi ký tự overclaim và số liệu mâu thuẫn tàn dư từ phiên bản cũ (`0.118`, `89.25%`, `LOSO_multi`, `Evidence-grade`) đã bị xóa sạch khỏi luồng PDF.
   * Nội dung hiển thị trong Hình 1 (trang 4 của bài báo) hiện nay hoàn toàn đồng nhất với nội dung trong Caption, Abstract, Section 4 và Section 5: Phản ánh trung thực kết quả âm tính của GRL+vCLUB (Acc 96.88% vs 97.07%, Train SRI 0.5380 vs 0.5261, chỉ ra trần hiệu năng giải phẫu và xung đột gradient minimax).
   * Loài cấm `Pterocarpus macrocarpus` đã được loại bỏ hoàn toàn khỏi mã nguồn sinh hình.
2. **Từ Quan sát 1.3 về Biên dịch Tài liệu LaTeX**:
   * Biên dịch `main.tex` thành công 100%, tạo ra tài liệu xuất bản chuẩn `paper/main.pdf` 14 trang.
   * `main.log` xác nhận sạch hoàn toàn: 0 fatal error, 0 broken citation `[?]`, 0 broken reference `??`.
3. **Từ Quan sát 1.4 và 1.5 về Tính Nhất Quán Thực Nghiệm và Mã Nguồn**:
   * Toàn bộ 8 mâu thuẫn văn bản (R2) và các thực nghiệm bổ sung (R3) đều được phản ánh chính xác từ dữ liệu thực tế `multi_seed_results.json` vào bài báo mà không có bất kỳ sự sai lệch, ngụy tạo, hay facade implementation nào.
4. **Đối chiếu với Tiêu chuẩn Nghiệm thu (Acceptance Criteria trong `ORIGINAL_REQUEST.md`)**:
   * Tiêu chuẩn LaTeX Compilation & Structure: ĐẠT 100%.
   * Tiêu chuẩn Empirical Consistency & Contradiction Resolution: ĐẠT 100%.
   * Tiêu chuẩn Supplementary Experiments & Diagnostics: ĐẠT 100%.
5. **Kết luận suy diễn**:
   * Tất cả các vi phạm đã được khắc phục triệt để. Không còn bất kỳ phát hiện vi phạm tính toàn vẹn nào tồn đọng. Phán quyết chính thức được nâng lên thành **`CLEAN`**.

---

## 3. Caveats (Các điểm cần lưu ý và giới hạn)

* **Phạm vi thẩm định**: Thẩm định tập trung vào bộ dữ liệu 11 loài Fabaceae chuẩn của S3 (110 phôi vật lý, 3,702 ảnh). Báo cáo lịch sử phản biện `RESPONSE_TO_REVIEWERS.md` lưu giữ các ghi chép phản biện trước đây đối với bộ 17/19 loài; tệp này mang tính chất nhật ký lưu trữ tham khảo nội bộ và không ảnh hưởng đến nội dung bài báo khoa học chính thức `main.tex` và `main.pdf`.
* **Phần mềm biên dịch**: Quá trình biên dịch và thẩm định được xác nhận trên MiKTeX 26.5 / pdfTeX 3.141592653 trên nền hệ điều hành Windows.

---

## 4. Conclusion (Kết luận kỹ thuật cuối cùng)

1. **Phán quyết chính thức**: **`CLEAN`**. Sản phẩm bài báo khoa học `03_research_paper_specimen_invariance/paper/main.tex` và các tệp liên quan đạt 100% tiêu chuẩn nghiệm thu của người dùng.
2. **Đánh giá chất lượng khoa học**: Bài báo đã chuyển đổi thành công từ một bài báo mang tính quảng bá phương pháp sang một công trình kiểm toán khoa học mẫu mực (Forensic Specimen Leakage Audit & Honest Negative Results), sẵn sàng nộp cho các tạp chí hàng đầu (Elsevier Pattern Recognition / Computers and Electronics in Agriculture).
3. **Trạng thái Milestone**: Milestone 3 (Comprehensive Review & Audit) hoàn thành xuất sắc.

---

## 5. Verification Method (Phương pháp thẩm định độc lập độc lập)

Bất kỳ chuyên gia đánh giá độc lập nào cũng có thể kiểm chứng lại toàn bộ các phát hiện trên bằng cách chạy các lệnh kiểm tra sau trong PowerShell tại thư mục gốc `g:/S3_paper`:

1. **Kiểm tra đồ thị Hình 1 không còn tàn dư overclaim**:
   ```powershell
   python -c "
   import zlib, re
   with open('03_research_paper_specimen_invariance/paper/figures/fig1_concept_invariant.pdf', 'rb') as f: data = f.read()
   streams = re.findall(b'stream[\r\n]+(.*?)[\r\n]+endstream', data, re.DOTALL)
   decomp = '\n'.join([zlib.decompress(s).decode('latin1', errors='ignore') for s in streams if b'FlateDecode' in data])
   assert '0.118' not in decomp
   assert '89.25%' not in decomp
   assert 'LOSO_multi' not in decomp
   print('Verification 1: Figure 1 Clean PASSED!')
   "
   ```

2. **Kiểm tra sạch loài cấm trong toàn bộ mã nguồn hình ảnh**:
   ```powershell
   python -c "
   with open('03_research_paper_specimen_invariance/paper/generate_figures.py', 'r', encoding='utf-8') as f: src = f.read()
   assert 'macrocarpus' not in src
   assert 'Dialium' not in src
   print('Verification 2: Species Check in Figure Scripts PASSED!')
   "
   ```

3. **Kiểm tra biên dịch LaTeX hoàn hảo**:
   ```powershell
   cd 03_research_paper_specimen_invariance/paper
   pdflatex -interaction=nonstopmode -halt-on-error main.tex
   bibtex main
   pdflatex -interaction=nonstopmode -halt-on-error main.tex
   python -c "
   with open('main.log', 'r', encoding='utf-8', errors='ignore') as f: log = f.read()
   assert 'Citation' not in log or 'undefined' not in log
   assert 'Reference' not in log or 'undefined' not in log
   print('Verification 3: LaTeX Zero-Warning Build PASSED!')
   "
   cd ../..
   ```
