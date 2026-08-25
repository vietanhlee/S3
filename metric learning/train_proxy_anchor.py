"""
train_proxy_anchor.py — Proxy Anchor Loss (CVPR 2020)
=====================================================
So sánh ảnh với Proxy đại diện từng lớp thay vì ảnh-ảnh.
Được tối ưu hóa Vectorized 100% trên GPU (không dùng vòng lặp Python)
và điều chỉnh margin chuẩn hóa PROXY_MARGIN = 0.10 cho sự cân bằng tuyệt đối
giữa lực kéo Positive và lực đẩy Negative cho các loài gỗ trong cùng Chi.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from train_base import BaseMetricTrainer, PKSampler

# ===== CẤU HÌNH =====
CONFIG = {
	"OUTPUT_DIR": "outputs_proxy_anchor",
	"PROXY_ALPHA": 32.0,     # Scale factor alpha chuẩn 32.0
	"PROXY_MARGIN": 0.10,    # Margin delta chuẩn 0.10 cho Proxy Anchor (Target Δp=0.10, Δn=-0.10)
	"EPOCHS": 50,
	"PATIENCE": 25,
	"LR": 1e-4,
	"WEIGHT_DECAY": 1e-4,
	"P_CLASSES": 19,
	"K_SAMPLES": 20,
}
# =====================


class ProxyAnchorLoss(nn.Module):
	"""Proxy Anchor Loss (CVPR 2020) — Vectorized & An toàn số học 100% trên GPU."""

	def __init__(self, num_classes: int, embedding_dim: int = 256,
	             alpha: float = 32.0, margin: float = 0.10) -> None:
		super().__init__()
		self.alpha = alpha
		self.margin = margin
		self.num_classes = num_classes
		# Proxy vectors có thể học (learnable proxies)
		self.proxies = nn.Parameter(torch.randn(num_classes, embedding_dim))
		nn.init.kaiming_normal_(self.proxies, mode="fan_out")

	def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
		# Chuẩn hóa Proxy và Embedding về mặt cầu đơn vị
		proxies = F.normalize(self.proxies, p=2, dim=1)        # (C, D)
		sim_matrix = torch.matmul(embeddings, proxies.t())     # Cosine similarity: (B, C)

		# One-hot mask cho nhãn mục tiêu trong batch
		one_hot = F.one_hot(labels, num_classes=self.num_classes).float()  # (B, C)
		pos_mask = one_hot.bool()
		neg_mask = ~pos_mask

		# Tính exponent cho positive term và negative term
		pos_exp = -self.alpha * (sim_matrix - self.margin)
		neg_exp = self.alpha * (sim_matrix + self.margin)

		# Masking an toàn số học bằng masked_fill (-1e4 tránh tràn số trong logsumexp)
		pos_exp = pos_exp.masked_fill(~pos_mask, -1e4)
		neg_exp = neg_exp.masked_fill(~neg_mask, -1e4)

		# Logsumexp theo chiều batch cho từng proxy: (C,)
		pos_term = torch.logsumexp(pos_exp, dim=0)
		neg_term = torch.logsumexp(neg_exp, dim=0)

		# Xác định các proxy có ít nhất 1 mẫu positive trong batch hiện tại
		has_pos = pos_mask.any(dim=0)
		num_pos_proxies = max(int(has_pos.sum().item()), 1)

		# Tính loss tổng hợp theo đúng công thức CVPR 2020 gốc
		loss_pos = torch.log1p(torch.exp(pos_term[has_pos])).sum() / num_pos_proxies
		loss_neg = torch.log1p(torch.exp(neg_term)).sum() / self.num_classes

		return loss_pos + loss_neg


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
		# Sử dụng PK Sampler để đảm bảo cân bằng các mẫu/lớp trong từng batch
		return PKSampler(labels, p=self.config["P_CLASSES"], k=self.config["K_SAMPLES"])


if __name__ == "__main__":
	trainer = ProxyAnchorTrainer(CONFIG)
	trainer.run()
