# Báo Cáo Khảo Sát Kỹ Thuật Toàn Diện (Handoff Report) — Explorer M1
**Dự án**: Kiểm toán rò rỉ mẫu vật (Specimen Leakage Audit) & Báo cáo kết quả âm trung thực (Honest Negative Result)  
**Thư mục mục tiêu**: `03_research_paper_specimen_invariance/`  
**Tác giả**: teamwork_explorer (explorer_m1)  
**Thời gian hoàn thành**: 2026-11-04T04:18:00Z  

---

## 1. Observation (Các quan sát thực tế và bằng chứng trực tiếp)

### 1.1. Cấu trúc Mã nguồn và Dữ liệu 11 Loài Fabaceae S3
* **Tệp cấu hình và quét dữ liệu**: `03_research_paper_specimen_invariance/dataset.py` (dòng 38--64).
  * Danh sách 11 loài họ Đậu (*Fabaceae*) mục tiêu sạch:
    1. *Afzelia bella* (CITES App. II, 10 phôi, 400 ảnh)
    2. *Dalbergia melanoxylon* (CITES App. II, 10 phôi, 291 ảnh)
    3. *Dalbergia oliveri* (CITES App. II, 10 phôi, 316 ảnh)
    4. *Dalbergia rimosa* (CITES App. II, 10 phôi, 300 ảnh)
    5. *Dalbergia tonkinensis* (CITES App. II, 10 phôi, 325 ảnh)
    6. *Guibourtia arnoldiana* (Non-CITES, 10 phôi, 323 ảnh)
    7. *Guibourtia ehie* (Non-CITES, 10 phôi, 400 ảnh)
    8. *Peltogyne pubescens* (Non-CITES, 10 phôi, 371 ảnh)
    9. *Pterocarpus erinaceus* (CITES App. II, 10 phôi, 336 ảnh)
    10. *Pterocarpus indicus* (Non-CITES, 10 phôi, 312 ảnh)
    11. *Sindora tonkinensis* (Non-CITES, 10 phôi, 328 ảnh)
  * Tổng cộng: Đúng 6 chi (*Afzelia*, *Dalbergia*, *Guibourtia*, *Peltogyne*, *Pterocarpus*, *Sindora*), 11 loài, 110 phôi vật lý (10 phôi/loài cân bằng tuyệt đối), 3,702 ảnh tiêu chuẩn (đối chiếu trực tiếp từ `01_data_paper_forensic_cites/out/metadata/metadata.csv`).
  * 8 loài đã bị loại bỏ hoàn toàn (`EXCLUDED_SPECIES`, dòng 54--63 của `dataset.py`): *Sindora cochinchinensis*, *Pterocarpus soyauxii*, *Afzelia pachyloba*, *Afzelia quanzensis*, *Afzelia africana*, *Guibourtia coleosperma*, *Pterocarpus macrocarpus*, *Dalbergia cochinchinensis*.

### 1.2. Kịch bản Phân vùng & Cấu hình Đa Hạt Giống (9 Seeds)
* **Tệp thực thi**: `03_research_paper_specimen_invariance/run_benchmark.py`.
  * Danh sách hạt giống (`CANDIDATE_SEEDS`, dòng 560--563): `[42, 123, 456, 789, 101, 202, 303, 404, 505]` (9 seeds).
  * Giao thức phân vùng ngoại suy cấp phôi (`split_dataset_by_seed`, `dataset.py`, dòng 215--250):
    * 5 phôi/loài $\to$ Out-of-Specimen Unseen Test (55 phôi, chiếm đúng 50% số phôi vật lý).
    * 1 phôi/loài $\to$ Out-of-Specimen Val (11 phôi, chiếm 10% để Early Stopping & Calibrate Temperature).
    * 4 phôi/loài $\to$ Train Pool (44 phôi, 40%): phân chia 80% Train, 20% In-Specimen Seen Test.
  * Phân tích phụ thuộc giữa 9 phân vùng: Do 55 phôi kiểm thử đều rút từ tập 110 phôi, độ trùng lặp phôi giữa hai seed bất kỳ có giá trị trung bình là **27.8 phôi / 55 phôi** (tương đương 50.5% overlap, min 20, max 33). Điều này chứng minh định lý Bengio & Grandvalet (2004) rằng các phân vùng phụ thuộc lẫn nhau một phần, khiến khoảng tin cậy tính giữa các seed có tính chất lạc quan (optimistic).
  * Hiện trạng thực nghiệm rò rỉ: Dòng 704--714 của `run_benchmark.py` hiện chỉ chạy `train_naive_random_leakage_baseline` trên 1 seed duy nhất (`active_seeds[0] = 42`). Chưa có vòng lặp chạy Naive Random Split trên toàn bộ 9 seeds để tính 95% CI cho khoảng cách phóng đại rò rỉ (leakage inflation gap).

### 1.3. Điều tra Trường hợp Dị biệt Phôi 50 và 53 (Seed 456 - Baseline đạt 0% Acc)
* **Ánh xạ danh tính phôi**:
  * Khi sắp xếp 110 phôi theo thứ tự bảng chữ cái của `specimen_id` (`dataset.py`, dòng 207--209):
    * Index 50: `Guibourtia_arnoldiana_3411._Guibourtia_arnoldiana.1` (32 ảnh)
    * Index 53: `Guibourtia_arnoldiana_3411._Guibourtia_arnoldiana.3` (29 ảnh)
    * Cả hai phôi đều thuộc loài *Guibourtia arnoldiana* (Gỗ Muntenye / Mutenye).
* **Phân vùng tại Seed 456**:
  * Tập Train (4 phôi): Index 54 (`.4`), 55 (`.5`), 58 (`.8`), 59 (`.9`).
  * Tập Val (1 phôi): Index 57 (`.7`).
  * Tập Unseen Test (5 phôi): Index 50 (`.1`), 51 (`.10`), 52 (`.2`), 53 (`.3`), 56 (`.6`).
* **Đặc tính quang học và giải phẫu thực tế**:
  * Độ nét Laplacian: Phôi 50 có Laplacian variance trung bình chỉ **113.81** (min 34.29, max 182.20), phôi 53 là **152.08** (min 112.55, max 204.90). Mức này thấp hơn rõ rệt so với các loài gỗ cứng khác trong tập (ví dụ *Dalbergia oliveri* 2,993, *Peltogyne pubescens* 1,102).
  * Nhãn taxon: 100% metadata chính xác, không bị gán sai loài trong thư mục đĩa.
  * Tương đồng giải phẫu học (*anatomical_features.csv*): *Guibourtia arnoldiana* và *Guibourtia ehie* (cùng chi *Guibourtia*) có cấu trúc mô mềm hình thang dạng thang (*scalariform & trapezoidal*), màu tâm gỗ nâu vàng ánh đồng có vệt sọc tối. Khi ERM baseline chỉ học trên 4 phôi train (54, 55, 58, 59), nó ghi nhớ đặc trưng bề mặt cục bộ (vết xước cưa xẻ, hướng chiếu sáng) của 4 phôi này. Khi gặp phôi 50 và 53 ở góc cưa xẻ khác hoặc độ tương phản thấp, mạng ERM nhầm lẫn toàn bộ 100% sang loài chị em cùng chi (*Guibourtia ehie*), dẫn đến độ chính xác 0.0%.

### 1.4. Cơ chế Mô hình GRL + vCLUB và Động lực Đối kháng
* **Kiến trúc**: `models_gan.py`
  * `BackboneWrapper` bọc ResNet-18 (512-d feature).
  * `SpeciesClassifier`: Linear head 512 $\to$ 11.
  * `SpeciesConditionedSpecimenDiscriminator`: GRL $\to$ MLP (512 $\to$ 256 $\to$ 128 $\to$ 44) kèm mặt nạ loài `mask` ($11 \times 44$).
  * `VariationalCLUB`: Vectorized Gaussian variational upper bound $I(Z; S \mid Y)$.
* **Hàm mất mát và tham số**:
  * `loss_total = loss_species + (adv_weight * alpha * loss_specimen) + (club_weight * loss_club)` (dòng 507, `run_benchmark.py`).
  * `adv_weight = 0.05`, `club_weight = 0.001`.
  * Nhược điểm hiện tại: Script huấn luyện `train_gan_one_seed` chỉ in `train_acc` và `val_acc`, hoàn toàn không lưu lịch sử epoch của `loss_specimen` (discriminator loss) hay `loss_club`. Không thể quan sát được hiện tượng cân bằng Nash hay gradient đối kháng triệt tiêu lẫn nhau.

### 1.5. Môi trường Biên dịch LaTeX
* Kiểm tra lệnh hệ thống tại `03_research_paper_specimen_invariance/paper/`:
  * `pdflatex` (MiKTeX-pdfTeX 4.27) và `bibtex` (MiKTeX-BibTeX 4.2) hoạt động hoàn hảo.
  * `latexmk` bị lỗi thiếu script engine `perl` trên Windows.
  * Chuỗi lệnh biên dịch sạch:
    ```powershell
    pdflatex -interaction=nonstopmode -halt-on-error main.tex
    bibtex main
    pdflatex -interaction=nonstopmode -halt-on-error main.tex
    pdflatex -interaction=nonstopmode -halt-on-error main.tex
    ```
    Biên dịch thành công ra file `main.pdf` (27 trang, exit code 0, 0 fatal error).

---

## 2. Logic Chain: Định Vị và Ánh Xạ 8 Mâu Thuẫn (R2) & Tái Định Vị (R1)

### 2.1. Phân Tích Chi Tiết 8 Mâu Thuẫn Trong Văn Bản (R2)

| # | Điểm mâu thuẫn | Vị trí xuất hiện trong bài/mã nguồn | Diễn giải sai lệch cũ | Thực tế số liệu bảng thực nghiệm | Phương án giải quyết dứt điểm |
|---|---|---|---|---|---|
| **1** | **GGSL Gap tại Mục 4.3** | `README.md` (dòng 53), `generate_report.py` (dòng 301--303), `paper/main.tex` | Tuyên bố GAN "triệt tiêu GGSL $\to$ 0 pp" hoặc kéo giảm từ 1.04 pp về 0 pp | Baseline GGSL là **2.74 pp**, trong khi GAN GGSL là **2.82 pp** (GAN thực tế còn cao hơn +0.08 pp) | Sửa lại hoàn toàn văn bản: Nêu rõ GRL+vCLUB không triệt tiêu GGSL, mức giảm là không có ý nghĩa thống kê (2.74 pp vs 2.82 pp, CI chứa 0). |
| **2** | **Phát biểu về Train SRI** | `README.md` (dòng 54), `generate_report.py` (dòng 304--305), `paper/main.tex` (dòng 85, 126) | Tuyên bố "Train SRI giảm mạnh, không gian đặc trưng ẩn đã quên phôi" | Baseline Train SRI là **0.5261**, GAN Train SRI là **0.5380** (GAN không hề giảm, thậm chí cao hơn +0.0119) | Sửa phát biểu: Ghi nhận trung thực Train SRI của GAN (0.538) không giảm so với Baseline (0.526), khoảng tin cậy chênh lệch chứa giá trị 0. |
| **3** | **Điểm neo tham chiếu ImageNet** | `generate_report.py` (dòng 168), `run_benchmark.py` (dòng 596) | Phát biểu: "ImageNet gốc = 0.5667. Huấn luyện ERM làm tăng mức rò rỉ lên 0.5261, trong khi GAN kéo giảm về 0.5380" | Mâu thuẫn toán học logic: $0.5261 < 0.5667$ (ERM làm giảm chứ không tăng!), và $0.5380 > 0.5261$ (GAN không giảm so với ERM!) | Viết lại logic: ImageNet pretrained anchor là 0.5667; cả hai mô hình fine-tune đều nén nhẹ thông tin phôi về ~0.526--0.538, nhưng GRL không đem lại lợi thế nén bổ sung so với ERM. |
| **4** | **Baseline Acc tại Mục 4.2** | `README.md` (dòng 50), `generate_report.py` (dòng 293, 297) | Tuyên bố Baseline ResNet-18 đạt "~98.5% -- 99.0%" | Bảng số liệu thực nghiệm chính xác ghi nhận Baseline Unseen Top-1 là **97.07%** | Sửa con số "~98.5-99.0%" thành đúng **97.07%** trong toàn bộ văn bản thảo luận. |
| **5** | **Sai số hiệu chuẩn ECE** | `README.md` (dòng 55), `generate_report.py` (dòng 307) | Tuyên bố dập tắt ECE về mức an toàn "< 1%" | Bảng số liệu thực tế đo được ECE sau hiệu chuẩn nhiệt độ dao động trong khoảng **1.58% -- 1.93%** | Điều chỉnh phát biểu thành "đạt mức 1.58% -- 1.93%" thay vì khẳng định dưới 1%. |
| **6** | **Loài thực vật ngoại lai** | `generate_report.py` (dòng 295), `paper/main.tex` (dòng 443, 770) | Nêu *Dialium cochinchinense* và *Pterocarpus macrocarpus* trong thảo luận giải phẫu 11 loài họ Đậu S3 | *P. macrocarpus* đã bị loại bỏ vào `EXCLUDED_SPECIES`, còn *Dialium cochinchinense* không có trong S3 | Loại bỏ hoàn toàn 2 tên loài này khỏi mọi phân tích giải phẫu. Chỉ thảo luận đối chiếu trên 11 loài sạch: *Afzelia bella*, *Pterocarpus erinaceus*, *Dalbergia tonkinensis*, v.v. |
| **7** | **Từ ngữ Overclaim và Tiêu đề** | `paper/main.tex` (Title, Abstract), `README.md` (Title) | Dùng các từ "Mitigating", "Superior", "Provably", "Disentangled representations" | Kết quả thực nghiệm cho thấy GRL+vCLUB không vượt trội so với ERM ($\Delta\text{Acc} = -0.19$ pp, $p=0.66$) | Sửa tiêu đề thành: "Auditing Specimen Leakage and Evaluating Invariance Methods in Macroscopic Timber Forensics: An Empirical Benchmark and Honest Negative Results". Loại bỏ mọi từ khẳng định vượt trội. |
| **8** | **Tuyên bố về nghiên cứu bên thứ ba** | `generate_report.py` (dòng 154), `README.md` (dòng 22) | Tuyên bố quy chụp: "Các công bố đạt 99.8-100% trên random split bị phồng số liệu" | Thiếu căn cứ bao quát mọi bộ dữ liệu khác ngoài thực địa | Tiết chế văn phong học thuật: Chỉ nêu rõ trên bộ dữ liệu thực nghiệm S3, việc phân chia ngẫu nhiên từng ảnh tạo ra độ phóng đại hiệu năng (leakage inflation gap) bao nhiêu điểm phần trăm. |

### 2.2. Kế hoạch Tái Định Vị Luận Điểm Khoa Học (R1)
Chuyển đổi toàn bộ cấu trúc bài báo từ "quảng bá mô hình mới" sang "bài báo kiểm toán rò rỉ và báo cáo kết quả âm trung thực":
1. **Ba Đóng Góp Cốt Lõi (Core Contributions)**:
   - *Contribution 1*: Định lượng hiện tượng rò rỉ mẫu vật (Specimen Leakage Audit) trong thị giác máy tính nhận dạng gỗ, chứng minh mức độ phóng đại hiệu năng (Optimism Bias) do chia ngẫu nhiên từng ảnh.
   - *Contribution 2*: Chuẩn hóa giao thức đánh giá ngoại suy cấp phôi vật lý (50% physical voucher holdout) kèm kỹ thuật kiểm định Bootstrap cấp phôi (Voucher-Level Cluster Bootstrap, $B=1000$).
   - *Contribution 3*: Báo cáo trung thực kết quả âm của phương pháp biểu diễn bất biến đối kháng (GRL + vCLUB không cải thiện có ý nghĩa thống kê so với ERM baseline, Acc $-0.19$ pp, 95% CI $[-1.16, +0.78]$, $p=0.66$ t-test, $p=0.91$ Wilcoxon) và giải phẫu ba nguyên nhân gốc rễ (ranh giới giải phẫu loài họ Đậu quá rõ ràng, trần hiệu năng ~97%, và sự xung đột gradient trong tối ưu hóa đối kháng).
2. **Quy tắc Kiểm toán Bảng biểu (Table Unbolding)**:
   - Tuyệt đối không in đậm toàn bộ hàng kết quả của GRL+vCLUB / GAN.
   - Chỉ in đậm ở các chỉ số có sự khác biệt vượt trội và có ý nghĩa thống kê ($p < 0.05$). Đối với các chỉ số mà khoảng tin cậy chứa 0 hoặc $p > 0.05$, giữ nguyên định dạng thường.

---

## 3. Caveats (Các điểm cần lưu ý và giới hạn khảo sát)

1. **Vắng mặt tệp ảnh vật lý cục bộ**: Trong thư mục workspace hiện tại không có ảnh `.png` thô (do ảnh gốc lưu trên Kaggle `/kaggle/input/datasets/b23dckh002lvitanh/s3-origin/S3`), nhưng toàn bộ metadata chi tiết của từng bức ảnh (độ nét Laplacian, mã hash sha256, đường dẫn, nhãn phôi) đều được bảo toàn 100% trong `01_data_paper_forensic_cites/out/metadata/metadata.csv`.
2. **File `multi_seed_results.json` chưa được lưu trữ vĩnh viễn trong repo**: Kết quả đo đạc 9 seeds cần được tổng hợp chính thức trong Milestone 2 bằng script cập nhật.
3. **Phụ thuộc môi trường LaTeX**: Môi trường Windows hiện tại không có `perl`, do đó lệnh `latexmk` không chạy được, nhưng quy trình biên dịch tự động qua `pdflatex` và `bibtex` đã được kiểm chứng hoạt động hoàn hảo và tạo ra file PDF chuẩn xác.

---

## 4. Conclusion (Kết luận kỹ thuật & Hướng dẫn cụ thể cho Worker M2)

Mã nguồn và bản thảo hiện tại đã sẵn sàng để tái cấu trúc toàn diện trong Milestone 2. Nhiệm vụ cụ thể cho Agent Worker (M2) được đúc kết như sau:

### Nhiệm vụ Kỹ thuật Mã Nguồn (Code Tasks for Worker):
1. **Mở rộng Leakage Audit ra 9 Seeds**:
   - Chỉnh sửa `run_benchmark.py`: Trong hàm `train_naive_random_leakage_baseline`, cho chạy lặp qua cả 9 seeds (hoặc tích hợp vào vòng lặp seed chính) để tính toán $\text{Mean} \pm \text{Std}$ và 95% CI của độ phóng đại rò rỉ $\Delta_{\text{inflation}} = \mathrm{Acc}_{\text{naive}} - \mathrm{Acc}_{\text{unseen}}$.
2. **Ghi log động lực học GRL & Discriminator**:
   - Cập nhật `train_gan_one_seed` trong `run_benchmark.py`: Lưu mảng lịch sử theo từng epoch của `loss_species`, `loss_specimen` (discriminator loss), `loss_club`, và `alpha`. Lưu thông tin này vào file JSON kết quả để phục vụ vẽ đồ thị hoặc phân tích hành vi đối kháng khi điều chỉnh `adv_weight`.
3. **Phân tích minh bạch 2 trường hợp phôi 50 & 53**:
   - Trích xuất bảng số liệu chi tiết về phôi 50 và 53 của *Guibourtia arnoldiana* trong seed 456 (Laplacian mean 113.8 và 152.1, sự tương đồng giải phẫu với *Guibourtia ehie*) để đưa vào phần Thảo luận của bài báo.

### Nhiệm vụ Văn bản Bài Báo (LaTeX Tasks for Worker):
1. **Đổi tiêu đề và Tái định vị (R1)**: Cập nhật Title, Abstract, Contributions, Method, Discussion, Conclusion trong `paper/main.tex` theo đúng định vị kiểm toán rò rỉ và kết quả âm trung thực.
2. **Xóa bỏ triệt để 8 mâu thuẫn (R2)**:
   - GGSL: Sửa thành Baseline 2.74 pp, GAN 2.82 pp.
   - Train SRI: Sửa thành Baseline 0.526, GAN 0.538 (không giảm).
   - ImageNet anchor: Sửa logic so sánh 0.5667 vs 0.526 vs 0.538.
   - Baseline Acc: Sửa thành 97.07%.
   - ECE: Sửa thành 1.58% -- 1.93%.
   - Xóa bỏ *Dialium cochinchinense* và *Pterocarpus macrocarpus*, chỉ dùng đúng 11 loài S3.
   - Bỏ từ ngữ overclaim ("Mitigating" $\to$ "Evaluating", loại bỏ "provably", "guarantees").
   - Tiết chế tuyên bố về nghiên cứu bên thứ ba.
3. **Kiểm toán Bảng biểu**: Bỏ in đậm toàn bộ dòng GAN trong các bảng số liệu.
4. **Biên dịch**: Chạy chuỗi lệnh `pdflatex` + `bibtex` + `pdflatex` $\times 2$ đảm bảo `paper/main.pdf` không có lỗi fatal, không có `[?]` hay `??`.

---

## 5. Verification Method (Phương pháp thẩm định độc lập)

Để độc lập thẩm định tính xác thực của các phát hiện trong báo cáo này, người thẩm định (Reviewer) có thể chạy các lệnh sau trong PowerShell tại thư mục gốc `g:/S3_paper`:

1. **Thẩm định danh sách 11 loài và 110 phôi vật lý**:
   ```powershell
   python -c "import pandas as pd; df = pd.read_csv('01_data_paper_forensic_cites/out/metadata/metadata.csv'); target = ['Afzelia bella', 'Dalbergia melanoxylon', 'Dalbergia oliveri', 'Dalbergia rimosa', 'Dalbergia tonkinensis', 'Guibourtia arnoldiana', 'Guibourtia ehie', 'Peltogyne pubescens', 'Pterocarpus erinaceus', 'Pterocarpus indicus', 'Sindora tonkinensis']; df_11 = df[df['class_name'].isin(target)]; print('Images:', len(df_11), 'Vouchers:', df_11['specimen_id'].nunique())"
   ```
   *Kết quả mong đợi*: Đúng 3,702 ảnh và 110 phôi (10 phôi/loài).

2. **Thẩm định phôi 50 & 53 trong seed 456**:
   ```powershell
   python -c "import pandas as pd; df = pd.read_csv('01_data_paper_forensic_cites/out/metadata/metadata.csv'); target = ['Afzelia bella', 'Dalbergia melanoxylon', 'Dalbergia oliveri', 'Dalbergia rimosa', 'Dalbergia tonkinensis', 'Guibourtia arnoldiana', 'Guibourtia ehie', 'Peltogyne pubescens', 'Pterocarpus erinaceus', 'Pterocarpus indicus', 'Sindora tonkinensis']; df_11 = df[df['class_name'].isin(target)]; u_v = sorted(df_11['specimen_id'].unique()); print('Index 50:', u_v[50]); print('Index 53:', u_v[53])"
   ```
   *Kết quả mong đợi*: `Guibourtia_arnoldiana_3411._Guibourtia_arnoldiana.1` và `Guibourtia_arnoldiana_3411._Guibourtia_arnoldiana.3`.

3. **Thẩm định biên dịch LaTeX**:
   ```powershell
   cd g:/S3_paper/03_research_paper_specimen_invariance/paper
   pdflatex -interaction=nonstopmode -halt-on-error main.tex
   bibtex main
   pdflatex -interaction=nonstopmode -halt-on-error main.tex
   ```
   *Kết quả mong đợi*: Exit code 0, tạo ra file `main.pdf`.
