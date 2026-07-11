"""
train_arcface.py — ArcFace / Additive Angular Margin Loss (CVPR 2019)
=====================================================================
Ép biên độ góc m vào cos(θ + m) trên mặt cầu đơn vị.
SOTA fine-grained recognition — gom cụm hình nón cực hẹp cho từng loài.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from train_base import BaseMetricTrainer

# ===== CẤU HÌNH =====
CONFIG = {
	"OUTPUT_DIR": "outputs_arcface",
	"ARCFACE_SCALE": 30.0,
	"ARCFACE_MARGIN": 0.50,
	"EPOCHS": 30,
	"PATIENCE": 10,
	"LR": 1e-4,
	"WEIGHT_DECAY": 1e-4,
	"P_CLASSES": 19,
	"K_SAMPLES": 20,
}
# =====================


class ArcMarginProduct(nn.Module):
	"""ArcFace Margin Product: cos(θ + m) với scale factor s.

	Input: L2-normalized embeddings (B, D)
	Output: scaled cosine logits (B, num_classes)
	"""

	def __init__(self, embedding_dim: int, num_classes: int,
	             scale: float = 30.0, margin: float = 0.50) -> None:
		super().__init__()
		self.scale = scale
		self.margin = margin
		self.num_classes = num_classes

		self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
		nn.init.xavier_uniform_(self.weight)

		self.cos_m = math.cos(margin)
		self.sin_m = math.sin(margin)
		# cos(π - m) = -cos(m)
		self.th = math.cos(math.pi - margin)
		self.mm = math.sin(math.pi - margin) * margin

	def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
		# Chuẩn hóa weight
		W = F.normalize(self.weight, p=2, dim=1)
		# Cosine similarity
		cosine = F.linear(embeddings, W)
		cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

		sine = torch.sqrt(1.0 - cosine.pow(2))

		# cos(θ + m) = cos(θ)cos(m) - sin(θ)sin(m)
		phi = cosine * self.cos_m - sine * self.sin_m

		# Xử lý trường hợp θ + m > π → dùng xấp xỉ tuyến tính
		phi = torch.where(cosine > self.th, phi, cosine - self.mm)

		# One-hot labels → áp dụng margin chỉ cho lớp đúng
		one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
		output = one_hot * phi + (1.0 - one_hot) * cosine

		return output * self.scale


class ArcFaceLoss(nn.Module):
	"""ArcFace = ArcMarginProduct + CrossEntropyLoss.

	Interface thống nhất: forward(embeddings, labels) → loss scalar.
	"""

	def __init__(self, embedding_dim: int, num_classes: int,
	             scale: float = 30.0, margin: float = 0.50) -> None:
		super().__init__()
		self.arc_margin = ArcMarginProduct(embedding_dim, num_classes, scale, margin)
		self.ce = nn.CrossEntropyLoss()

	def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
		logits = self.arc_margin(embeddings, labels)
		return self.ce(logits, labels)


class ArcFaceTrainer(BaseMetricTrainer):

	def get_method_name(self) -> str:
		return "ArcFace (Additive Angular Margin)"

	def get_loss_config(self) -> dict:
		return {
			"arcface_scale": self.config["ARCFACE_SCALE"],
			"arcface_margin": self.config["ARCFACE_MARGIN"],
		}

	def build_loss(self, num_classes: int) -> nn.Module:
		return ArcFaceLoss(
			embedding_dim=self.config["EMBEDDING_DIM"],
			num_classes=num_classes,
			scale=self.config["ARCFACE_SCALE"],
			margin=self.config["ARCFACE_MARGIN"],
		)

	def build_train_sampler(self, labels: list):
		# ArcFace là classification-based loss, không dùng PK Sampler mà dùng RandomSampler thông thường
		return None


if __name__ == "__main__":
	trainer = ArcFaceTrainer(CONFIG)
	trainer.run()
