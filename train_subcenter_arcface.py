"""
train_subcenter_arcface.py — SubCenter ArcFace (ECCV 2020)
==========================================================
Mở rộng ArcFace: K sub-center mỗi lớp để học phân bố đa đỉnh.
Giải quyết trực tiếp intra-class variation lớn của vân gỗ.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from train_base import BaseMetricTrainer

# ===== CẤU HÌNH =====
CONFIG = {
	"OUTPUT_DIR": "outputs_subcenter_arcface",
	"ARCFACE_SCALE": 30.0,
	"ARCFACE_MARGIN": 0.50,
	"NUM_SUBCENTERS": 3,    # Số sub-center mỗi lớp
	"FOCAL_GAMMA": 2.0,      # Siêu tham số gamma cho Focal Loss
	"EPOCHS": 80,
	"PATIENCE": 25,
	"LR": 1e-4,
	"WEIGHT_DECAY": 1e-4,
	"P_CLASSES": 19,
	"K_SAMPLES": 20,
}
# =====================


class SubCenterArcMarginProduct(nn.Module):
	"""SubCenter ArcFace: K sub-centers per class.

	Mỗi lớp có K weight vectors; lấy cosine max (gần nhất) trước khi áp margin.
	"""

	def __init__(self, embedding_dim: int, num_classes: int, num_subcenters: int = 3,
	             scale: float = 30.0, margin: float = 0.50) -> None:
		super().__init__()
		self.scale = scale
		self.margin = margin
		self.num_classes = num_classes
		self.num_subcenters = num_subcenters

		# Weight shape: (num_classes * K, embedding_dim)
		self.weight = nn.Parameter(torch.FloatTensor(num_classes * num_subcenters, embedding_dim))
		nn.init.xavier_uniform_(self.weight)

		self.cos_m = math.cos(margin)
		self.sin_m = math.sin(margin)
		self.th = math.cos(math.pi - margin)
		self.mm = math.sin(math.pi - margin) * margin

	def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
		W = F.normalize(self.weight, p=2, dim=1)
		# Cosine similarity: (B, C*K)
		cosine_all = F.linear(embeddings, W)

		# Reshape → (B, C, K) → lấy max trên K sub-centers → (B, C)
		cosine_all = cosine_all.view(-1, self.num_classes, self.num_subcenters)
		cosine, _ = cosine_all.max(dim=2)  # (B, C)
		cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

		# Áp dụng angular margin cho lớp đúng
		sine = torch.sqrt(1.0 - cosine.pow(2))
		phi = cosine * self.cos_m - sine * self.sin_m
		phi = torch.where(cosine > self.th, phi, cosine - self.mm)

		one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
		output = one_hot * phi + (1.0 - one_hot) * cosine

		return output * self.scale


class FocalLoss(nn.Module):
	"""Focal Loss có gán trọng số cân bằng lớp alpha (class_weights)."""

	def __init__(self, gamma: float = 2.0, alpha: torch.Tensor = None) -> None:
		super().__init__()
		self.gamma = gamma
		self.alpha = alpha

	def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
		ce = F.cross_entropy(logits, targets, reduction="none")
		pt = torch.exp(-ce)
		if self.alpha is not None:
			alpha_t = self.alpha[targets]
		else:
			alpha_t = 1.0
		loss = alpha_t * (1.0 - pt) ** self.gamma * ce
		return loss.mean()


class SubCenterArcFaceLoss(nn.Module):
	"""SubCenter ArcFace = SubCenterArcMarginProduct + FocalLoss."""

	def __init__(self, embedding_dim: int, num_classes: int, num_subcenters: int = 3,
	             scale: float = 30.0, margin: float = 0.50,
	             gamma: float = 2.0, class_weights: torch.Tensor = None) -> None:
		super().__init__()
		self.arc_margin = SubCenterArcMarginProduct(
			embedding_dim, num_classes, num_subcenters, scale, margin,
		)
		self.focal = FocalLoss(gamma=gamma, alpha=class_weights)

	def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
		logits = self.arc_margin(embeddings, labels)
		return self.focal(logits, labels)


class SubCenterArcFaceTrainer(BaseMetricTrainer):

	def get_method_name(self) -> str:
		return "SubCenter ArcFace"

	def get_loss_config(self) -> dict:
		return {
			"arcface_scale": self.config["ARCFACE_SCALE"],
			"arcface_margin": self.config["ARCFACE_MARGIN"],
			"num_subcenters": self.config["NUM_SUBCENTERS"],
			"focal_gamma": self.config["FOCAL_GAMMA"],
		}

	def build_loss(self, num_classes: int) -> nn.Module:
		class_weights = getattr(self, "class_weights", None)
		return SubCenterArcFaceLoss(
			embedding_dim=self.config["EMBEDDING_DIM"],
			num_classes=num_classes,
			num_subcenters=self.config["NUM_SUBCENTERS"],
			scale=self.config["ARCFACE_SCALE"],
			margin=self.config["ARCFACE_MARGIN"],
			gamma=self.config["FOCAL_GAMMA"],
			class_weights=class_weights,
		)

	def build_train_sampler(self, labels: list):
		# SubCenter ArcFace là classification-based loss, không dùng PK Sampler mà dùng RandomSampler thông thường
		return None


if __name__ == "__main__":
	trainer = SubCenterArcFaceTrainer(CONFIG)
	trainer.run()
