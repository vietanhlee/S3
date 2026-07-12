"""
train_semi_hard_triplet.py — Semi-hard Triplet Loss (FaceNet Style, CVPR 2015)
===============================================================================
Chọn các negative nằm trong vùng biên margin: d(a,p) < d(a,n) < d(a,p) + margin.
Baseline kinh điển được sử dụng rộng rãi trong tất cả các benchmark DML.
"""

import torch
import torch.nn as nn
from train_base import BaseMetricTrainer

# ===== CẤU HÌNH =====
CONFIG = {
	"OUTPUT_DIR": "outputs_semi_hard_triplet",
	"TRIPLET_MARGIN": 0.3,
	"EPOCHS": 80,
	"PATIENCE": 25,
	"LR": 1e-4,
	"P_CLASSES": 19,
	"K_SAMPLES": 20,
}
# =====================


class SemiHardTripletLoss(nn.Module):
	"""Semi-hard Triplet Loss — FaceNet style online mining.

	Chọn negative n sao cho: d(a,p)^2 < d(a,n)^2 < d(a,p)^2 + margin
	"""

	def __init__(self, margin: float = 0.3) -> None:
		super().__init__()
		self.margin = margin

	def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
		dist_mat = torch.cdist(embeddings, embeddings, p=2).pow(2)
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

			for p_idx in pos_indices:
				d_ap = dist_mat[i, p_idx]

				# Tìm semi-hard negatives: d_ap < d_an < d_ap + margin
				d_an_all = dist_mat[i, neg_indices]
				semi_hard_mask = (d_an_all > d_ap) & (d_an_all < d_ap + self.margin)

				if semi_hard_mask.any():
					# Chọn negative gần nhất trong vùng semi-hard
					d_an = d_an_all[semi_hard_mask].min()
				else:
					# Fallback: chọn hardest negative (gần nhất)
					d_an = d_an_all.min()

				loss = torch.clamp(d_ap - d_an + self.margin, min=0.0)
				if loss > 0:
					losses.append(loss)

		if len(losses) == 0:
			return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
		return torch.stack(losses).mean()


class SemiHardTripletTrainer(BaseMetricTrainer):

	def get_method_name(self) -> str:
		return "Semi-hard Triplet Loss (FaceNet)"

	def get_loss_config(self) -> dict:
		return {"triplet_margin": self.config["TRIPLET_MARGIN"]}

	def build_loss(self, num_classes: int) -> nn.Module:
		return SemiHardTripletLoss(margin=self.config["TRIPLET_MARGIN"])


if __name__ == "__main__":
	trainer = SemiHardTripletTrainer(CONFIG)
	trainer.run()
