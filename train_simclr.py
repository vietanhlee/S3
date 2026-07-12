"""
train_simclr.py — SimCLR (Unsupervised Contrastive, NeurIPS 2020)
================================================================
Học tự giám sát tương phản InfoNCE sử dụng 2 views ngẫu nhiên cho mỗi ảnh.
Kéo gần bản biểu diễn của cùng một ảnh và đẩy xa tất cả các ảnh khác trong batch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd

from train_base import BaseMetricTrainer
from train_supcon import DoubleViewTransform, SupConDataset

# ===== CẤU HÌNH =====
CONFIG = {
	"OUTPUT_DIR": "outputs_simclr",
	"TEMPERATURE": 0.5,     # Nhiệt độ mặc định cho SimCLR
	"BATCH_SIZE": 128,
	"EPOCHS": 60,
	"PATIENCE": 25,
	"LR": 1e-4,
}
# =====================


class SimCLRLoss(nn.Module):
	"""SimCLR Unsupervised Contrastive Loss (InfoNCE)."""

	def __init__(self, temperature: float = 0.5) -> None:
		super().__init__()
		self.temperature = temperature

	def forward(self, features: torch.Tensor, labels=None) -> torch.Tensor:
		# features: (B, N_views, D) hoặc (B, D)
		# Nếu là validation (single view), trả về 0 loss để tránh crash
		if len(features.shape) == 2 or features.shape[1] < 2:
			return torch.tensor(0.0, device=features.device, requires_grad=True)

		device = features.device
		batch_size = features.shape[0]
		n_views = features.shape[1]  # Thường là 2

		# Flatten views thành (B * V, D)
		features = features.view(batch_size * n_views, -1)
		features = F.normalize(features, p=2, dim=1)

		# Tính ma trận cosine similarity chéo (B * V, B * V)
		similarity_matrix = torch.matmul(features, features.t()) / self.temperature

		# Trừ max để ổn định số mũ
		logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
		logits = similarity_matrix - logits_max.detach()

		# Tạo nhãn positive cho InfoNCE:
		# Cặp positive là (2i, 2i+1) và (2i+1, 2i)
		# Tạo mask positive chéo:
		targets = torch.arange(batch_size, device=device)
		targets = targets.view(-1, 1).repeat(1, n_views).view(-1)  # (B * V)
		mask = torch.eq(targets.unsqueeze(0), targets.unsqueeze(1)).float()  # (B*V, B*V)

		# Loại bỏ đường chéo chính (self-contrast)
		diag_mask = torch.eye(batch_size * n_views, device=device)
		mask = mask - diag_mask

		# Tạo logits mask (chỉ giữ lại các phần tử không phải tự tương phản)
		logits_mask = 1.0 - diag_mask
		exp_logits = torch.exp(logits) * logits_mask
		log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-9)

		# Tính mean log-likelihood over positive
		mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-9)

		loss = -mean_log_prob_pos
		return loss.mean()


class SimCLRTrainer(BaseMetricTrainer):

	def get_method_name(self) -> str:
		return "SimCLR (Unsupervised Contrastive)"

	def get_loss_config(self) -> dict:
		return {"temperature": self.config["TEMPERATURE"]}

	def build_loss(self, num_classes: int) -> nn.Module:
		return SimCLRLoss(temperature=self.config["TEMPERATURE"])

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
			emb1 = model(img1)
			emb2 = model(img2)

			features = torch.stack([emb1, emb2], dim=1)
			loss = criterion(features)
			loss.backward()
			optimizer.step()

			total_loss += loss.item()
			n_batches += 1
			pbar.set_postfix(loss=f"{total_loss / n_batches:.4f}")

		return total_loss / max(n_batches, 1)


if __name__ == "__main__":
	trainer = SimCLRTrainer(CONFIG)
	trainer.run()
