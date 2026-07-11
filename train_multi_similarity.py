"""
train_multi_similarity.py — Multi-Similarity (MS) Loss (CVPR 2019)
==================================================================
Gán trọng số gradient động dựa trên 3 loại similarity.
Lọc mẫu thông minh — mẫu gỗ càng khó phân biệt nhận gradient càng lớn.
"""

import torch
import torch.nn as nn
from train_base import BaseMetricTrainer

# ===== CẤU HÌNH =====
CONFIG = {
	"OUTPUT_DIR": "outputs_multi_similarity",
	"MS_ALPHA": 2.0,       # Siêu tham số alpha
	"MS_BETA": 50.0,       # Siêu tham số beta
	"MS_MARGIN": 0.5,      # Siêu tham số lambda/margin
	"EPOCHS": 30,
	"PATIENCE": 10,
	"LR": 1e-4,
	"P_CLASSES": 19,
	"K_SAMPLES": 20,
}
# =====================


class MultiSimilarityLoss(nn.Module):
	"""Multi-Similarity Loss — trọng số gradient động tinh vi."""

	def __init__(self, alpha: float = 2.0, beta: float = 50.0,
	             base_margin: float = 0.5) -> None:
		super().__init__()
		self.alpha = alpha
		self.beta = beta
		self.base_margin = base_margin

	def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
		similarity_matrix = torch.matmul(embeddings, embeddings.t())
		batch_size = embeddings.size(0)

		labels_col = labels.unsqueeze(1)
		is_positive = (labels_col == labels_col.t()).float()
		is_negative = 1.0 - is_positive

		diag_mask = torch.eye(batch_size, device=embeddings.device)
		is_positive = torch.clamp(is_positive - diag_mask, min=0.0)

		loss_all = []
		for i in range(batch_size):
			pos_idx = torch.where(is_positive[i] > 0)[0]
			neg_idx = torch.where(is_negative[i] > 0)[0]

			if len(pos_idx) == 0 or len(neg_idx) == 0:
				continue

			sim_pos = similarity_matrix[i, pos_idx]
			sim_neg = similarity_matrix[i, neg_idx]

			# Hard pair mining
			max_neg_sim = sim_neg.max()
			min_pos_sim = sim_pos.min()

			pos_mask = sim_pos < max_neg_sim + 0.1
			neg_mask = sim_neg > min_pos_sim - 0.1

			sim_pos_mined = sim_pos[pos_mask] if pos_mask.any() else min_pos_sim.unsqueeze(0)
			sim_neg_mined = sim_neg[neg_mask] if neg_mask.any() else max_neg_sim.unsqueeze(0)

			loss_pos = torch.log(1.0 + torch.sum(
				torch.exp(-self.alpha * (sim_pos_mined - self.base_margin))
			)) / self.alpha
			loss_neg = torch.log(1.0 + torch.sum(
				torch.exp(self.beta * (sim_neg_mined - self.base_margin))
			)) / self.beta

			loss_all.append(loss_pos + loss_neg)

		if len(loss_all) == 0:
			return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
		return torch.stack(loss_all).mean()


class MultiSimilarityTrainer(BaseMetricTrainer):

	def get_method_name(self) -> str:
		return "Multi-Similarity Loss"

	def get_loss_config(self) -> dict:
		return {
			"ms_alpha": self.config["MS_ALPHA"],
			"ms_beta": self.config["MS_BETA"],
			"ms_margin": self.config["MS_MARGIN"],
		}

	def build_loss(self, num_classes: int) -> nn.Module:
		return MultiSimilarityLoss(
			alpha=self.config["MS_ALPHA"],
			beta=self.config["MS_BETA"],
			base_margin=self.config["MS_MARGIN"],
		)


if __name__ == "__main__":
	trainer = MultiSimilarityTrainer(CONFIG)
	trainer.run()
