import os

latex_path = "g:/S3_paper/latex_paper.tex"
draft_workspace_path = "g:/S3_paper/latex_report_draft.md"
draft_artifact_path = "C:/Users/levie/.gemini/antigravity-ide/brain/017a870b-91fb-42c5-9a83-a01052282b5b/latex_report_draft.md"

if not os.path.exists(latex_path):
    print(f"Error: {latex_path} not found.")
    exit(1)

with open(latex_path, "r", encoding="utf-8") as f:
    latex_content = f.read()

markdown_template = f"""# BÁO CÁO HỌC THUẬT (Q1 JOURNAL REPORT SKELETON)

Tài liệu này bao gồm hai phần chính:
1. **Source Code LaTeX hoàn chỉnh (`main.tex`)** ở dạng khung sườn chuẩn học thuật hai cột (định dạng Overleaf/IEEEtran twocolumn) với các công thức toán học và cấu trúc bảng số liệu cực kỳ chi tiết, được tự động scale để vừa vặn với chiều rộng của cột.
2. **Nội dung bản thảo chi tiết bằng tiếng Việt** tương ứng với từng phần để bạn dễ dàng làm việc và thảo luận với giảng viên hướng dẫn.

---

## 1. Source Code LaTeX (`main.tex`)

Bạn có thể copy toàn bộ nội dung khối code dưới đây và dán thẳng vào trình soạn thảo Overleaf hoặc LaTeX compiler:

```latex
{latex_content}
```
"""

# Write to workspace
with open(draft_workspace_path, "w", encoding="utf-8") as f:
    f.write(markdown_template)
print(f"Synchronized to {draft_workspace_path}")

# Write to artifact directory (ensure parent dirs exist)
os.makedirs(os.path.dirname(draft_artifact_path), exist_ok=True)
with open(draft_artifact_path, "w", encoding="utf-8") as f:
    f.write(markdown_template)
print(f"Synchronized to {draft_artifact_path}")
