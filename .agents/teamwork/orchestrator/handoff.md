# Báo Cáo Nghiệm Thu & Bàn Giao Dự Án (Project Orchestrator Final Handoff)

**Dự án**: Viết lại và hoàn thiện bài báo khoa học `03_research_paper_specimen_invariance/paper/main.tex` theo định vị "Kiểm toán rò rỉ mẫu vật (Specimen Leakage Audit) và Báo cáo kết quả âm trung thực đối với các phương pháp bất biến"  
**Orchestrator Working Directory**: `g:/S3_paper/.agents/teamwork/orchestrator/`  
**Parent Agent**: sentinel (`cadbab1c-f46f-47c9-9b4a-01e020b746c0`)  
**Thời gian hoàn thành**: 2026-11-04T05:30:00Z  
**Phán quyết kiểm toán độc lập**: **`CLEAN` (Forensic Integrity Pass — Gate Iteration 2)**  

---

## 1. Observation (Tổng hợp Thực nghiệm và Bằng chứng Trực tiếp)

1. **Quy mô và Ranh giới Dữ liệu**:
   - Đúng 11 loài thuộc họ Đậu (*Fabaceae*) mục tiêu sạch thuộc 6 chi (*Afzelia*, *Dalbergia*, *Guibourtia*, *Peltogyne*, *Pterocarpus*, *Sindora*), cân bằng tuyệt đối 10 phôi/loài, tổng cộng 110 phôi vật lý và 3,702 ảnh hiển vi vĩ mô (trích xuất chuẩn xác từ `01_data_paper_forensic_cites/out/metadata/metadata.csv`).
   - Đã loại bỏ hoàn toàn các loài ngoài phạm vi, đặc biệt là *Dialium cochinchinense* và *Pterocarpus macrocarpus*.

2. **Kiểm toán Rò rỉ Mẫu vật (Specimen Leakage Audit) trên 9 Seeds**:
   - Naive Random Split (chia ngẫu nhiên theo ảnh): Top-1 Test Acc = **$99.55\% \pm 0.12\%$** [95% CI: 99.46%, 99.64%].
   - Out-of-Specimen Voucher Split (chia theo phôi vật lý ngoại suy, ERM Baseline): Top-1 Test Acc = **$97.07\% \pm 0.95\%$** [95% CI: 96.34%, 97.80%].
   - Mức độ phóng đại rò rỉ (Leakage Inflation Gap): $\Delta_{\text{inflation}} = \mathbf{+2.48 \pm 0.98\text{ pp}}$ [95% CI: $+1.73, +3.23\text{ pp}$], paired $t$-test $t = 7.59, p = 0.0001 < 0.001$.

3. **Báo cáo Trung thực Kết quả Âm (Honest Negative Results) của GRL+vCLUB**:
   - Out-of-Specimen Test Acc: Baseline $97.07\% \pm 0.95\%$ vs GRL+vCLUB $96.88\% \pm 0.99\%$ ($\Delta = -0.19\text{ pp}$, 95% CI $[-1.16, +0.78]\text{ pp}$, paired $t$-test $p = 0.66$, Wilcoxon $p = 0.91$).
   - Generalization Gap (GGSL): Baseline $2.74 \pm 0.92\text{ pp}$ vs GRL+vCLUB $2.82 \pm 0.96\text{ pp}$ ($\Delta = +0.08\text{ pp}$, 95% CI $[-0.35, +0.51]\text{ pp}$, $p > 0.05$).
   - Train SRI: Baseline $0.5261 \pm 0.0182$ vs GRL+vCLUB $0.5380 \pm 0.0195$ ($\Delta = +0.0119$, 95% CI $[-0.0145, +0.0383]$, $p = 0.34$, ImageNet anchor $0.5667$).
   - Calibrated ECE: $1.62\%$ vs $1.85\%$ (khoảng thực tế $1.58\% - 1.93\%$).
   - Toàn bộ các dòng kết quả của GAN trong Table 2 và Table 3 đã unbold hoàn toàn do không đạt ý nghĩa thống kê.

4. **Giải phẫu Thực nghiệm và Chẩn đoán Cơ chế**:
   - Phôi 50 (`Guibourtia_arnoldiana.1`) và Phôi 53 (`Guibourtia_arnoldiana.3`) trong seed 456 đạt $0.0\%$ baseline accuracy do độ sắc nét Laplacian rất thấp ($113.8$ và $152.1$) và hình thái mô mềm tương đồng với loài chị em *Guibourtia ehie*.
   - GRL discriminator loss (`loss_specimen`) được ghi log đầy đủ theo epoch, chỉ ra xung đột gradient minimax giữa phân loại loài và loại bỏ phôi.
   - Độ trùng lặp phôi giữa các seed kiểm thử là $50.5\%$ (27.75/55 phôi), xác thực định lý Bengio & Grandvalet (2004) rằng các split phụ thuộc một phần, chứng minh tính cần thiết của Voucher-Level Cluster Bootstrap ($B=1000$).

5. **Biên dịch Xuất bản LaTeX**:
   - `paper/main.pdf` biên dịch thành công 14 trang (447,884 bytes), 0 fatal errors, 0 broken citation `[?]`, 0 broken reference `??`.
   - Toàn bộ các hình ảnh vector PDF (`fig1_concept_invariant.pdf`, `fig2_framework_architecture.pdf`, `fig3_gradcam_comparison.pdf`) đã được tái tạo và kiểm tra luồng nhị phân sạch 100%, không còn chuỗi overclaim hay số liệu mâu thuẫn cũ.

---

## 2. Logic Chain (Chuỗi Lập luận và Ra Quyết định Điều phối)

1. **Khảo sát Nền tảng (Milestone 1 — Explorer M1)**:
   - Rà soát toàn bộ mã nguồn, cấu hình 9 seeds, metadata 11 loài và định vị chính xác vị trí của 8 mâu thuẫn văn bản cùng phôi 50 & 53 trong seed 456. Thiết lập lộ trình kỹ thuật cho Worker.
2. **Thực thi và Tái cấu trúc (Milestone 2 — Worker M2)**:
   - Triển khai vòng lặp kiểm toán rò rỉ 9 seeds, trích xuất 95% CI, bổ sung logging discriminator loss, viết lại toàn bộ `main.tex` theo định vị mới, và biên dịch `main.pdf`.
3. **Thẩm định Pháp y Lần 1 & Thực thi Veto (Gate Iteration 1 — Auditor M3)**:
   - Auditor M3 phát hiện tàn dư overclaim trong `figures/fig1_concept_invariant.pdf` (SRI -> 0.118, 89.25% Macro-F1) và loài cấm trong `generate_figures.py`.
   - Áp dụng nguyên tắc Binary Veto nghiêm ngặt: Orchestrator từ chối nghiệm thu, chuyển toàn bộ bằng chứng và chỉ đạo Worker M2 khắc phục khẩn cấp.
4. **Khắc phục & Phúc tra Phê duyệt (Gate Iteration 2 — Worker M2 & Auditor M3)**:
   - Worker M2 sửa `generate_figures.py`, tái tạo đồ thị vector sạch, xóa bỏ loài cấm, tái biên dịch `main.pdf`.
   - Auditor M3 phúc tra độc lập bằng cách giải nén 134 luồng nhị phân PDF và kiểm tra log biên dịch: Xác nhận sạch 100% và ban hành phán quyết **`CLEAN`**.

---

## 3. Caveats (Các điểm Lưu ý Khoa học)

- **Bộ sưu tập đơn cơ sở (Single-Xylarium Scope)**: Toàn bộ 110 phôi được chuẩn bị đồng nhất tại phòng thí nghiệm chuyên khảo; kiểm chứng ngoại suy đa viện bảo tàng quốc tế (cross-collection) sẽ là hướng mở rộng tiếp theo khi hoàn tất hiệp định CITES Article VII(6).
- **Mặt cắt ngang hiển vi quang học vĩ mô**: Nghiên cứu tập trung vào bề mặt cắt ngang (transverse section); mặt cắt xuyên tâm (radial) và tiếp tuyến (tangential) chưa bao gồm trong đợt khảo sát này.

---

## 4. Conclusion & Milestone State

- **Milestone 1 (Survey & Analysis)**: DONE
- **Milestone 2 (Code Experiments & Paper Refactoring)**: DONE
- **Milestone 3 (Forensic Integrity Audit)**: DONE (Verdict: CLEAN)
- **Active Subagents**: 0 running (All 3 agents completed).
- **Pending Decisions**: None.
- **Remaining Work**: None. Dự án hoàn thành 100% tiêu chuẩn nghiệm thu.

---

## 5. Key Artifacts

1. `g:/S3_paper/03_research_paper_specimen_invariance/paper/main.tex` — Tệp nguồn LaTeX bài báo khoa học.
2. `g:/S3_paper/03_research_paper_specimen_invariance/paper/main.pdf` — Bản in PDF 14 trang hoàn chỉnh, biên dịch sạch không warning.
3. `g:/S3_paper/03_research_paper_specimen_invariance/outputs/multi_seed_results.json` — Kết quả định lượng chi tiết trên 9 seeds.
4. `g:/S3_paper/03_research_paper_specimen_invariance/EXPERIMENT_RESULTS_REPORT.md` — Báo cáo thực nghiệm mở rộng bằng Markdown.
5. `g:/S3_paper/03_research_paper_specimen_invariance/paper/figures/fig1_concept_invariant.pdf` — Sơ đồ khái niệm Hình 1 đã cập nhật kết quả âm trung thực.
6. `g:/S3_paper/.agents/teamwork/auditor_m3/handoff.md` — Báo cáo thẩm định pháp y độc lập của Auditor M3 (Verdict: CLEAN).
7. `g:/S3_paper/.agents/teamwork/orchestrator/GATE_STATUS.md` — Biên bản kiểm tra cổng Iteration 1 (FAIL) và Iteration 2 (PASS).
