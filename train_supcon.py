"""
train_supcon.py — Supervised Contrastive (SupCon) Loss (NeurIPS 2020)
=====================================================================
Batch-wide multi-positive alignment — kéo tất cả các ảnh cùng loài trong batch lại gần.
Tạo biểu diễn tổng quát mạnh nhất, phù hợp với ảnh thớ gỗ macro.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd

from train_base import BaseMetricTrainer

# ===== CẤU HÌNH =====
CONFIG = {
	"OUTPUT_DIR": "outputs_supcon",
	"TEMPERATURE": 0.07,
	"BATCH_SIZE": 128,   # Batch size lớn hơn cho SupCon
	"EPOCHS": 50,
	"PATIENCE": 10,
	"LR": 1e-4,
}
# =====================


class DoubleViewTransform:
	"""Sinh hai view ngẫu nhiên khác nhau từ cùng một ảnh để huấn luyện SupCon."""

	def __init__(self, transform1, transform2) -> None:
		self.transform1 = transform1
		self.transform2 = transform2

	def __call__(self, img):
		return self.transform1(img), self.transform2(img)


class SupConDataset(Dataset):
	"""Dataset sinh view kép cho SupCon."""

	def __init__(self, df: pd.DataFrame, class_to_idx: dict, transform=None) -> None:
		self.df = df.reset_index(drop=True)
		self.class_to_idx = class_to_idx
		self.transform = transform
		self.labels = [class_to_idx[lbl] for lbl in self.df["label"]]

	def __len__(self) -> int:
		return len(self.df)

	def __getitem__(self, idx: int):
		row = self.df.iloc[idx]
		with Image.open(row["path"]) as img:
			img = img.convert("RGB")
		if self.transform:
			img1, img2 = self.transform(img)
		else:
			img1 = transforms.ToTensor()(img)
			img2 = transforms.ToTensor()(img)
		return (img1, img2), self.labels[idx]


class SupConLoss(nn.Module):
	"""Supervised Contrastive Learning Loss (NeurIPS 2020)."""

	def __init__(self, temperature: float = 0.07) -> None:
		super().__init__()
		self.temperature = temperature

	def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
		# features: shape (B, N_views, D) hoặc (B, D)
		# labels: shape (B)
		if len(features.shape) == 2:
			features = features.unsqueeze(1)  # Chuyển thành (B, 1, D) để tương thích với single view eval

		device = features.device
		batch_size = features.shape[0]
		n_views = features.shape[1]

		# Flatten views: shape (B * V, D)
		features = features.view(batch_size * n_views, -1)
		features = F.normalize(features, p=2, dim=1)

		# Tính ma trận cosine similarity chéo (B * V, B * V)
		similarity_matrix = torch.matmul(features, features.t()) / self.temperature

		# Trừ max để ổn định số mũ (numerical stability)
		logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
		logits = similarity_matrix - logits_max.detach()

		# Nhân bản labels tương ứng với số views
		labels = labels.view(-1, 1)
		mask = torch.eq(labels, labels.t()).float().to(device)  # (B, B)
		mask = mask.repeat(n_views, n_views)  # (B * V, B * V)

		# Tạo mask loại bỏ chính nó (self-contrast mask)
		logits_mask = torch.scatter(
			torch.ones_like(mask),
			1,
			torch.arange(batch_size * n_views, device=device).view(-1, 1),
			0
		)
		mask = mask * logits_mask

		# Tính log_prob
		exp_logits = torch.exp(logits) * logits_mask
		log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-9)

		# Tính mean log-likelihood over positive
		mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-9)

		# Loss
		loss = -mean_log_prob_pos
		loss = loss.view(n_views, batch_size).mean()
		return loss


class SupConTrainer(BaseMetricTrainer):

	def get_method_name(self) -> str:
		return "Supervised Contrastive (SupCon) Loss"

	def get_loss_config(self) -> dict:
		return {"temperature": self.config["TEMPERATURE"]}

	def build_loss(self, num_classes: int) -> nn.Module:
		return SupConLoss(temperature=self.config["TEMPERATURE"])

	def build_train_sampler(self, labels: list):
		# SupCon không dùng PK Sampler, dùng RandomSampler (shuffle=True) chuẩn của DataLoader
		return None

	def get_batch_size(self) -> int:
		# Lấy BATCH_SIZE chỉ định riêng cho SupCon (mặc định 128)
		return self.config.get("BATCH_SIZE", 128)

	def build_train_dataset(self, df: pd.DataFrame, class_to_idx: dict, transform) -> Dataset:
		# Vì train transform nhận vào là transforms.Compose của ảnh đơn,
		# ta tạo DoubleViewTransform để sinh ra 2 views ngẫu nhiên cho SupCon.
		# LƯU Ý: train_tf được sinh ra từ Base class qua build_transforms.
		# Ta clone transform đó làm 2 views độc lập để augment ngẫu nhiên khác nhau.
		# Ở đây ta có thể dùng trực tiếp transform để áp dụng 2 lần khác nhau.
		# (Do các augmentation có tính ngẫu nhiên nên gọi 2 lần sẽ ra 2 view khác nhau).
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
		for (img1, img2), labels in pbar:
			img1 = img1.to(device, non_blocking=True)
			img2 = img2.to(device, non_blocking=True)
			labels = labels.to(device, non_blocking=True)

			optimizer.zero_grad()
			emb1 = model(img1)
			emb2 = model(img2)
			# stack features thành (B, 2, D)
			features = torch.stack([emb1, emb2], dim=1)
			loss = criterion(features, labels)
			loss.backward()
			optimizer.step()

			total_loss += loss.item()
			n_batches += 1
			pbar.set_postfix(loss=f"{total_loss / n_batches:.4f}")

		return total_loss / max(n_batches, 1)


if __name__ == "__main__":
	trainer = SupConTrainer(CONFIG)
	trainer.run()
