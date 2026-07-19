"""
train_byol.py — BYOL (Bootstrap Your Own Latent, NeurIPS 2020)
=============================================================
Học tự giám sát sử dụng hai mạng Online và Target. Mạng Target được cập nhật
bằng trung bình trượt lũy thừa (EMA) của mạng Online. Không cần âm bản (negative samples).
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd

from train_base import BaseMetricTrainer
from train_supcon import DoubleViewTransform, SupConDataset

# ===== CẤU HÌNH =====
CONFIG = {
	"OUTPUT_DIR": "outputs_byol",
	"PREDICTOR_DIM": 64,   # Chiều ẩn cho online predictor MLP
	"BYOL_DECAY": 0.99,    # Hệ số suy giảm EMA để cập nhật target network (tau)
	"BATCH_SIZE": 128,
	"EPOCHS": 50,
	"PATIENCE": 25,
	"LR": 1e-4,
}
# =====================


class BYOLModel(nn.Module):
	"""BYOL Model: chứa mạng Online (online_model + online_predictor) và mạng Target (target_model)."""

	def __init__(self, base_model: nn.Module, predictor_dim: int = 64) -> None:
		super().__init__()
		self.online_model = base_model
		embedding_dim = base_model.projector[-1].out_features

		# Mạng Online Predictor
		self.online_predictor = nn.Sequential(
			nn.Linear(embedding_dim, predictor_dim),
			nn.BatchNorm1d(predictor_dim),
			nn.ReLU(inplace=True),
			nn.Linear(predictor_dim, embedding_dim),
		)

		# Mạng Target (được khởi tạo bằng cách sao chép mạng Online và đóng băng)
		self.target_model = copy.deepcopy(self.online_model)
		for p in self.target_model.parameters():
			p.requires_grad = False

	@torch.no_grad()
	def update_target_network(self, decay: float = 0.99) -> None:
		"""Cập nhật Target Network theo công thức EMA: target = decay * target + (1 - decay) * online."""
		for p_online, p_target in zip(self.online_model.parameters(), self.target_model.parameters()):
			p_target.data = decay * p_target.data + (1.0 - decay) * p_online.data

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# Luôn trả về đặc trưng của mạng Online khi gọi forward thông thường (eval/visualization)
		return self.online_model(x)


class BYOLLoss(nn.Module):
	"""BYOL Loss: tối thiểu hóa khoảng cách cosin bình phương giữa online prediction và target projection."""

	def forward(self, p: torch.Tensor, z: torch.Tensor = None) -> torch.Tensor:
		# Nếu z là None hoặc là nhãn lớp (LongTensor/IntTensor) truyền vào từ evaluate_loss, ta ở chế độ validation
		if z is None or (isinstance(z, torch.Tensor) and z.dtype in (torch.int64, torch.int32, torch.int16, torch.int8)):
			return torch.tensor(0.0, device=p.device, requires_grad=True)

		p = F.normalize(p, p=2, dim=1)
		z = F.normalize(z, p=2, dim=1)
		return 2.0 - 2.0 * (p * z).sum(dim=1).mean()


class BYOLTrainer(BaseMetricTrainer):

	def get_method_name(self) -> str:
		return "BYOL (Bootstrap Your Own Latent)"

	def get_loss_config(self) -> dict:
		return {
			"predictor_dim": self.config["PREDICTOR_DIM"],
			"byol_decay": self.config["BYOL_DECAY"],
		}

	def build_loss(self, num_classes: int) -> nn.Module:
		return BYOLLoss()

	def build_model(self) -> nn.Module:
		base_model = super().build_model()
		return BYOLModel(base_model, predictor_dim=self.config["PREDICTOR_DIM"])

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

			# Online predictions
			z1_online = model.online_model(img1)
			p1_online = model.online_predictor(z1_online)

			z2_online = model.online_model(img2)
			p2_online = model.online_predictor(z2_online)

			# Target projections (không tính gradient)
			with torch.no_grad():
				z1_target = model.target_model(img1)
				z2_target = model.target_model(img2)

			# BYOL Loss đối xứng chéo
			loss1 = criterion(p1_online, z2_target)
			loss2 = criterion(p2_online, z1_target)
			loss = 0.5 * (loss1 + loss2)

			loss.backward()
			optimizer.step()

			# Cập nhật Target Network bằng EMA
			model.update_target_network(decay=self.config["BYOL_DECAY"])

			total_loss += loss.item()
			n_batches += 1
			pbar.set_postfix(loss=f"{total_loss / n_batches:.4f}")

		return total_loss / max(n_batches, 1)


if __name__ == "__main__":
	trainer = BYOLTrainer(CONFIG)
	trainer.run()
