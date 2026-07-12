"""
train_simsiam.py — SimSiam (Simple Siamese, CVPR 2021)
======================================================
Học tự giám sát đối xứng đơn giản không cần negative pairs hay target network.
Sử dụng toán tử stop-gradient để tránh hiện tượng sụp đổ đặc trưng (representation collapse).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd

from train_base import BaseMetricTrainer
from train_supcon import DoubleViewTransform, SupConDataset

# ===== CẤU HÌNH =====
CONFIG = {
	"OUTPUT_DIR": "outputs_simsiam",
	"PREDICTOR_DIM": 64,   # Chiều ẩn của predictor MLP
	"BATCH_SIZE": 128,
	"EPOCHS": 60,
	"PATIENCE": 25,
	"LR": 1e-4,
}
# =====================


class SimSiamModel(nn.Module):
	"""Wrapper cho SimSiam Model.

	- Train mode: Trả về (p, z) là prediction và projection.
	- Eval mode: Trả về embedding L2-normalized z để đồng bộ với BaseMetricTrainer.
	"""

	def __init__(self, base_model: nn.Module, predictor_dim: int = 64) -> None:
		super().__init__()
		self.base_model = base_model
		# Lấy embedding_dim từ projection head của base_model (mặc định 256)
		embedding_dim = base_model.projector[-1].out_features

		self.predictor = nn.Sequential(
			nn.Linear(embedding_dim, predictor_dim),
			nn.BatchNorm1d(predictor_dim),
			nn.ReLU(inplace=True),
			nn.Linear(predictor_dim, embedding_dim),
		)

	def forward(self, x: torch.Tensor):
		if not self.training:
			return self.base_model(x)  # Trả về embedding L2-normalized thông thường cho eval
		z = self.base_model(x)  # L2-normalized projection head output
		p = self.predictor(z)
		return p, z


class SimSiamLoss(nn.Module):
	"""SimSiam loss function: D(p, z) = - (p^T z) / (||p|| * ||z||)."""

	def __init__(self) -> None:
		super().__init__()

	def _neg_cosine_similarity(self, p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
		# p và z đã được normalize sẵn trong code
		p = F.normalize(p, p=2, dim=1)
		z = F.normalize(z, p=2, dim=1)
		return -(p * z).sum(dim=1).mean()

	def forward(self, outputs, labels=None) -> torch.Tensor:
		# Trong validation/evaluation loader, outputs là single tensor (B, D) do eval mode của model
		if not isinstance(outputs, tuple):
			return torch.tensor(0.0, device=outputs.device, requires_grad=True)

		p1, z1, p2, z2 = outputs
		# SimSiam loss: ngắt gradient (detach) của z
		loss1 = self._neg_cosine_similarity(p1, z2.detach())
		loss2 = self._neg_cosine_similarity(p2, z1.detach())
		return 0.5 * (loss1 + loss2)


class SimSiamTrainer(BaseMetricTrainer):

	def get_method_name(self) -> str:
		return "SimSiam (Simple Siamese)"

	def get_loss_config(self) -> dict:
		return {"predictor_dim": self.config["PREDICTOR_DIM"]}

	def build_loss(self, num_classes: int) -> nn.Module:
		return SimSiamLoss()

	def build_model(self) -> nn.Module:
		base_model = super().build_model()
		return SimSiamModel(base_model, predictor_dim=self.config["PREDICTOR_DIM"])

	def build_train_sampler(self, labels: list):
		return None

	def get_batch_size(self) -> int:
		return self.config.get("BATCH_SIZE", 128)

	def build_train_dataset(self, df: pd.DataFrame, class_to_idx: dict, transform) -> Dataset:
		double_tf = DoubleViewTransform(transform, transform)
		return SupConDataset(df, class_to_idx, transform=double_tf)

	def train_one_epoch(self, model: nn.Module, loader: DataLoader,
	                    optimizer: torch.optim.Optimizer, criterion: nn.Module,
	                    device: torch.device, epoch: int, total_epochs: int) -> float:
		model.train()
		total_loss = 0.0
		n_batches = 0

		from tqdm import tqdm
		pbar = tqdm(loader, desc=f"Train {epoch}/{total_epochs}")
		for (img1, img2), _ in pbar:
			img1 = img1.to(device, non_blocking=True)
			img2 = img2.to(device, non_blocking=True)

			optimizer.zero_grad()
			p1, z1 = model(img1)
			p2, z2 = model(img2)

			# Gom kết quả làm đầu vào cho loss
			outputs = (p1, z1, p2, z2)
			loss = criterion(outputs)
			loss.backward()
			optimizer.step()

			total_loss += loss.item()
			n_batches += 1
			pbar.set_postfix(loss=f"{total_loss / n_batches:.4f}")

		return total_loss / max(n_batches, 1)


if __name__ == "__main__":
	trainer = SimSiamTrainer(CONFIG)
	trainer.run()
