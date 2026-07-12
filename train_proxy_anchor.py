"""
train_proxy_anchor.py — Proxy Anchor Loss (CVPR 2020)
=====================================================
So sánh ảnh với Proxy lớp thay vì ảnh-ảnh. O(N×C) thay vì O(N²).
Hội tụ cực nhanh, gradient ổn định — 18 class → cực hiệu quả.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from train_base import BaseMetricTrainer

# ===== CẤU HÌNH =====
CONFIG = {
	"OUTPUT_DIR": "outputs_proxy_anchor",
	"PROXY_ALPHA": 32.0,
	"PROXY_MARGIN": 0.1,
	"EPOCHS": 60,
	"PATIENCE": 25,
	"LR": 1e-4,
	"WEIGHT_DECAY": 1e-4,
	"P_CLASSES": 19,
	"K_SAMPLES": 20,
}
# =====================


class ProxyAnchorLoss(nn.Module):
	"""Proxy Anchor Loss — so sánh mẫu với proxy đại diện từng lớp."""

	def __init__(self, num_classes: int, embedding_dim: int = 256,
	             alpha: float = 32.0, margin: float = 0.1) -> None:
		super().__init__()
		self.alpha = alpha
		self.margin = margin
		self.num_classes = num_classes
		# Proxy vectors có thể học (learnable)
		self.proxies = nn.Parameter(torch.randn(num_classes, embedding_dim))
		nn.init.kaiming_normal_(self.proxies, mode="fan_out")

	def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
		# Chuẩn hóa proxy
		proxies = F.normalize(self.proxies, p=2, dim=1)
		# Ma trận cosine similarity: (B, C)
		similarity = torch.matmul(embeddings, proxies.t())

		# Positive proxies: proxy của lớp mà batch có chứa ít nhất 1 mẫu
		unique_labels = torch.unique(labels)

		loss_pos = torch.tensor(0.0, device=embeddings.device)
		loss_neg = torch.tensor(0.0, device=embeddings.device)

		for p_idx in range(self.num_classes):
			# Tìm các mẫu cùng lớp với proxy p_idx
			pos_mask = (labels == p_idx)
			neg_mask = (labels != p_idx)

			if pos_mask.any():
				# Positive term: kéo các mẫu cùng lớp về gần proxy
				pos_sim = similarity[pos_mask, p_idx]
				loss_pos += torch.log(1.0 + torch.sum(
					torch.exp(-self.alpha * (pos_sim - self.margin))
				))

			if neg_mask.any():
				# Negative term: đẩy các mẫu khác lớp ra xa proxy
				neg_sim = similarity[neg_mask, p_idx]
				loss_neg += torch.log(1.0 + torch.sum(
					torch.exp(self.alpha * (neg_sim + self.margin))
				))

		n_pos_proxies = len(unique_labels)
		loss = loss_pos / max(n_pos_proxies, 1) + loss_neg / self.num_classes
		return loss


class ProxyAnchorTrainer(BaseMetricTrainer):

	def get_method_name(self) -> str:
		return "Proxy Anchor Loss"

	def get_loss_config(self) -> dict:
		return {
			"proxy_alpha": self.config["PROXY_ALPHA"],
			"proxy_margin": self.config["PROXY_MARGIN"],
		}

	def build_loss(self, num_classes: int) -> nn.Module:
		return ProxyAnchorLoss(
			num_classes=num_classes,
			embedding_dim=self.config["EMBEDDING_DIM"],
			alpha=self.config["PROXY_ALPHA"],
			margin=self.config["PROXY_MARGIN"],
		)

	def build_train_sampler(self, labels: list):
		# Proxy Anchor là proxy-based loss, không dùng PK Sampler mà dùng RandomSampler thông thường
		return None


if __name__ == "__main__":
	trainer = ProxyAnchorTrainer(CONFIG)
	trainer.run()
