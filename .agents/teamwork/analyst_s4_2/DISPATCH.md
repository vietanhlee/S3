## 2026-09-29T12:47:52Z
Bạn là Chuyên gia Đánh giá Toán học & Biểu diễn Nhân quả Đối kháng (Adversarial Mathematical & Causal Representation Analyst - Candidate 2) cho Segment 4: paper02_causal_theory_proofs.

### THÔNG TIN ĐẦU VÀO VÀ ĐỊA BÀN LÀM VIỆC:
- Working Directory của bạn: G:/S3_paper/.agents/teamwork/analyst_s4_2
- Output Path bắt buộc: G:/S3_paper/.agents/teamwork/group_paper02_causal_theory_proofs/handoff_2.md (và lưu bản sao tại G:/S3_paper/.agents/teamwork/analyst_s4_2/handoff.md)
- Parent Conversation ID (gửi message báo cáo khi xong): 11980243-badc-4c28-90fa-62b9971bcf97
- Input Format: latex
- Analysis Partition Path: G:/S3_paper/.agents/teamwork/document_orchestrator_1/ANALYSIS_PARTITION.md
- Text Map Path: G:/S3_paper/.agents/teamwork/document_orchestrator_1/DOCUMENT_TEXT_MAP.md
- Target Manuscript: G:/S3_paper/03_research_paper_specimen_invariance/paper/main.tex (Đặc biệt tập trung vào Sections 1, 2, 3.1, 3.2, 3.3, 3.4)
- Mã nguồn liên quan:
  + G:/S3_paper/03_research_paper_specimen_invariance/models/grl.py
  + G:/S3_paper/03_research_paper_specimen_invariance/models/full_model.py
  + G:/S3_paper/03_research_paper_specimen_invariance/losses/

### NHIỆM VỤ AUDIT ĐỐI KHÁNG TRỌNG TÂM (R3 Part 1):
1. Mệnh đề 1 & Chứng minh (Proposition 1 & Proof - Semantic Collapse trên Singleton Taxa):
   - Phản biện toán học gắt gao đạo hàm thông tin lý thuyết (information-theoretic derivation) chặn trên I(Z; Y = c*) <= I(Z; S = s*) -> 0 khi specimen và loài bị cộng tuyến (collinear / singleton).
   - Kiểm tra chặt chẽ Đồ thị Nhân quả (Causal DAG), cơ chế điều kiện hoá (conditioning mechanics), quy tắc chuỗi tương hỗ thông tin (mutual information chain rule), và Bất đẳng thức Xử lý Dữ liệu (Data Processing Inequality).
   - Phân tích xem việc áp đặt bất biến mẫu vật (specimen invariance) dưới huấn luyện đối kháng thuần túy (unmasked adversarial training) có thực sự xóa bỏ danh tính loài đối với các taxa chỉ có 1 mẫu vật (singleton taxa) hay không. Phát hiện bất kỳ bước nhảy logic (leaps), giả định ngầm không có căn cứ (unjustified steps), hoặc ký hiệu mơ hồ (notation ambiguities).

2. Định lý 1 & Cơ chế Masked Softmax GRL có điều kiện theo loài (Species-Conditioned Masked Softmax GRL - Theorem 1):
   - Kiểm định tính chính xác toán học của Theorem 1 và cơ chế mặt nạ (M_c = 0 đối với các lớp đơn mẫu vật singleton) trong bộ phân biệt mẫu vật đối kháng (adversarial specimen discriminator).
   - Xác minh động lực học đảo ngược gradient (gradient reversal dynamics - GRL), công thức trò chơi đối kháng minimax, tính ổn định tại điểm cân bằng Nash, và chứng minh rằng việc áp mặt nạ ngăn ngừa suy biến ngữ nghĩa (semantic collapse) mà vẫn bảo toàn tính bất biến mẫu vật trên các lớp đa mẫu vật (multi-specimen classes).

3. Đối chiếu Mã nguồn và Bản thảo (Code-to-Manuscript Verification):
   - Đối chiếu công thức toán học của mặt nạ trong Theorem 1 với mã PyTorch trong `models/grl.py` và `models/full_model.py`.
   - Xác nhận cụ thể: Mặt nạ được áp dụng trước softmax (trong logits) hay sau softmax (trên probabilities)? Trọng số hàm mất mát (loss weighting) có nhất quán giữa phương trình trong bài báo và code thực thi không?

4. Cung cấp Đạo hàm Toán học Chặt chẽ, Sửa đổi Công thức và Các đoạn Patch LaTeX Diff:
   - Viết các đạo hàm hiệu chỉnh chi tiết, chuẩn xác từng ký hiệu toán học.
   - Cung cấp các đoạn diff LaTeX cụ thể (chỉ rõ line number, equation number trong `main.tex`), sẵn sàng áp dụng trực tiếp để khắc phục triệt để mọi lỗi toán học và lập luận.
