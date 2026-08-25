"""
train_circle.py — Circle Loss (Pair-based, CVPR 2020)
=====================================================
Tối ưu hóa không đối xứng trên mặt cầu đơn vị.
Positive đã gần → giảm lực kéo, Negative ở gần → tăng lực đẩy.

Các cải tiến & Tối ưu hóa siêu tham số:
1. CIRCLE_GAMMA = 64.0 (Giảm từ 80.0 xuống 64.0): Khắc phục hiện tượng nổ gradient
   và bị phụ thuộc quá mức vào các cặp nhiễu/mẫu vật kỳ dị ở các loài gỗ cùng chi.
2. Tách biệt Đạo hàm (alpha.detach()): Giúp alpha_p và alpha_n đóng vai trò làm
   hệ số điều chỉnh gradient động chuẩn xác theo đúng lý thuyết công bố tại CVPR 2020.
3. An toàn Số học (masked_fill với -1e4): Loại bỏ triệt để hiện tượng NaN/Overflow
   do -1e10 gây ra trong hàm logsumexp của PyTorch autograd.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from train_base import BaseMetricTrainer

# ===== CẤU HÌNH =====
CONFIG = {
	"OUTPUT_DIR": "outputs_circle",
	"CIRCLE_GAMMA": 64.0,       # Giảm từ 80.0 xuống 64.0 để ổn định gradient và hội tụ hài hòa giữa các loài
	"CIRCLE_MARGIN": 0.25,      # Margin tối ưu 0.25 (Target Δp=0.75, Δn=0.25)
	"EPOCHS": 50,
	"PATIENCE": 25,
	"LR": 1e-4,
	"P_CLASSES": 19,
	"K_SAMPLES": 20,
}
# =====================


class CircleLoss(nn.Module):
	"""Circle Loss (Pair-based) — adaptive soft-margin trên mặt cầu đơn vị với Đạo hàm Tách biệt."""

	def __init__(self, gamma: float = 64.0, margin: float = 0.25) -> None:
		super().__init__()
		self.gamma = gamma
		self.margin = margin
		self.O_p = 1.0 + margin
		self.O_n = -margin
		self.Delta_p = 1.0 - margin
		self.Delta_n = margin

	def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
		# Ma trận Cosine Similarity trên mặt cầu đơn vị
		similarity_matrix = torch.matmul(embeddings, embeddings.t())
		batch_size = embeddings.size(0)

		labels_col = labels.unsqueeze(1)
		is_positive = (labels_col == labels_col.t()).float()
		is_negative = 1.0 - is_positive

		diag_mask = torch.eye(batch_size, device=embeddings.device)
		is_positive = torch.clamp(is_positive - diag_mask, min=0.0)

		# Tính trọng số gradient động alpha_p và alpha_n (sử dụng .detach() để giữ đúng dạng đạo hàm lý thuyết)
		sim_detach = similarity_matrix.detach()
		alpha_p = torch.clamp(self.O_p - sim_detach, min=0.0)
		alpha_n = torch.clamp(sim_detach - self.O_n, min=0.0)

		exp_p = -self.gamma * alpha_p * (similarity_matrix - self.Delta_p)
		exp_n = self.gamma * alpha_n * (similarity_matrix - self.Delta_n)

		# Masking an toàn số học bằng masked_fill (-1e4 tránh tràn số trong logsumexp)
		exp_p = exp_p.masked_fill(is_positive == 0, -1e4)
		exp_n = exp_n.masked_fill(is_negative == 0, -1e4)

		logsumexp_p = torch.logsumexp(exp_p, dim=1)
		logsumexp_n = torch.logsumexp(exp_n, dim=1)

		has_pos = is_positive.sum(dim=1) > 0
		has_neg = is_negative.sum(dim=1) > 0
		valid_anchors = has_pos & has_neg

		if not valid_anchors.any():
			return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

		loss_anchors = F.softplus(logsumexp_p[valid_anchors] + logsumexp_n[valid_anchors])
		return loss_anchors.mean()


class CircleTrainer(BaseMetricTrainer):

	def get_method_name(self) -> str:
		return "Circle Loss (Pair-based)"

	def get_loss_config(self) -> dict:
		return {
			"circle_gamma": self.config["CIRCLE_GAMMA"],
			"circle_margin": self.config["CIRCLE_MARGIN"],
		}

	def build_loss(self, num_classes: int) -> nn.Module:
		return CircleLoss(
			gamma=self.config["CIRCLE_GAMMA"],
			margin=self.config["CIRCLE_MARGIN"],
		)


if __name__ == "__main__":
	trainer = CircleTrainer(CONFIG)
	trainer.run()
