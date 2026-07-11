"""
train_angular.py — Angular Loss (ICCV 2017)
============================================
Tối ưu hóa góc tam giác thay vì khoảng cách Euclid tuyệt đối.
Bất biến tỷ lệ (scale-invariant) — phù hợp ảnh macro gỗ chụp ở các mức zoom khác nhau.
"""

import math
import torch
import torch.nn as nn
from train_base import BaseMetricTrainer

# ===== CẤU HÌNH =====
CONFIG = {
	"OUTPUT_DIR": "outputs_angular",
	"ANGULAR_ALPHA_DEG": 45.0,   # Giới hạn góc (đơn vị độ)
	"EPOCHS": 30,
	"PATIENCE": 10,
	"LR": 1e-4,
	"P_CLASSES": 19,
	"K_SAMPLES": 20,
}
# =====================


class AngularLoss(nn.Module):
	"""Angular Loss — tối ưu hóa cấu trúc góc của bộ ba.

	L = log(1 + exp(4*tan²α * (xa+xp)ᵀxn - 2*(1+tan²α) * xaᵀxp))
	"""

	def __init__(self, alpha_deg: float = 45.0) -> None:
		super().__init__()
		alpha_rad = math.radians(alpha_deg)
		self.tan2_alpha = math.tan(alpha_rad) ** 2

	def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
		batch_size = embeddings.size(0)
		labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)
		diag = torch.eye(batch_size, dtype=torch.bool, device=embeddings.device)
		pos_mask = labels_eq & ~diag
		neg_mask = ~labels_eq

		losses = []
		for i in range(batch_size):
			pos_indices = torch.where(pos_mask[i])[0]
			neg_indices = torch.where(neg_mask[i])[0]

			if len(pos_indices) == 0 or len(neg_indices) == 0:
				continue

			xa = embeddings[i]
			for p_idx in pos_indices:
				xp = embeddings[p_idx]
				center = xa + xp  # (D,)

				# Cosine similarity: (xa+xp)ᵀxn cho tất cả negative
				sim_center_neg = torch.matmul(embeddings[neg_indices], center)  # (N_neg,)
				sim_ap = torch.dot(xa, xp)

				# Angular constraint
				logits = 4.0 * self.tan2_alpha * sim_center_neg - 2.0 * (1.0 + self.tan2_alpha) * sim_ap

				# Soft-plus (smooth approximation)
				loss = torch.log1p(torch.exp(logits))
				losses.append(loss.mean())

		if len(losses) == 0:
			return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
		return torch.stack(losses).mean()


class AngularTrainer(BaseMetricTrainer):

	def get_method_name(self) -> str:
		return "Angular Loss"

	def get_loss_config(self) -> dict:
		return {"angular_alpha_deg": self.config["ANGULAR_ALPHA_DEG"]}

	def build_loss(self, num_classes: int) -> nn.Module:
		return AngularLoss(alpha_deg=self.config["ANGULAR_ALPHA_DEG"])


if __name__ == "__main__":
	trainer = AngularTrainer(CONFIG)
	trainer.run()
