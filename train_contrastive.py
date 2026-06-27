"""
train_contrastive.py
====================
Metric Learning với Contrastive Loss (Online Hard Pair Mining).
- Backbone: ConvNeXt-Tiny (freeze 90%)
- Projection Head: 768 → 256 chiều + L2 Normalize
- PK Sampler: P=12 classes × K=8 samples/class (batch=96)
- 200 epochs, EarlyStopping theo Recall@1 trên tập Val
- Đánh giá: Recall@K, Precision@K, mAP, AUC
- Chia dữ liệu: End Version Split (reuse từ train_final.py)
"""

import os
import gc
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms

import timm
from timm.data import resolve_data_config
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.manifold import TSNE

# ===== CẤU HÌNH =====
ROOT_DIR = r"/kaggle/input/datasets/b23dckh002lvitanh/s3-origin/S3"
OUTPUT_DIR = "outputs_contrastive"
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
SEED = 42
P_CLASSES = 12          # Số class mỗi batch
K_SAMPLES = 8           # Số ảnh mỗi class trong batch
EPOCHS = 100
PATIENCE = 10           # EarlyStopping patience
LR = 1e-4
WEIGHT_DECAY = 1e-4
EMBEDDING_DIM = 256     # Projection head output
MARGIN = 1.0            # Contrastive loss margin
FREEZE_RATIO = 0.90
MODEL_NAME = "convnext_tiny"
COSINE_THRESHOLD = 0.92  # Cho PP5 (End Version Split)
EMB_BATCH_SIZE = 64      # Batch size cho compute embeddings (chia dữ liệu)
# =====================

# Import utilities từ codebase hiện tại
from train import (
	set_seed,
	get_device,
	collect_image_samples,
	build_dataframe,
	compute_embeddings,
	freeze_model_layers,
	summarize_model,
	log_split_summary,
	eda_split_class_distribution,
)
from train_final import end_version_split
from split_methods import validate_split


# ============================================================
# Dataset & PK Sampler
# ============================================================

class MetricImageDataset(Dataset):
	"""Dataset cho metric learning, trả về (image, label_idx)."""

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
			img = self.transform(img)
		return img, self.labels[idx]


class PKSampler(Sampler):
	"""
	PK Batch Sampler: mỗi batch chứa P classes × K samples/class.
	Đảm bảo mỗi batch có đủ positive pairs cho online mining.
	"""

	def __init__(self, labels: list, p: int, k: int) -> None:
		self.labels = labels
		self.p = p
		self.k = k

		# Group indices theo label
		self.label_to_indices: dict[int, list[int]] = {}
		for idx, lbl in enumerate(labels):
			self.label_to_indices.setdefault(lbl, []).append(idx)

		self.unique_labels = list(self.label_to_indices.keys())

		# Số batch mỗi epoch (duyệt hết tập train ít nhất 1 lần)
		self.n_batches = max(1, len(labels) // (p * k))

	def __iter__(self):
		for _ in range(self.n_batches):
			# Chọn ngẫu nhiên P classes
			p_actual = min(self.p, len(self.unique_labels))
			selected_labels = random.sample(self.unique_labels, p_actual)

			batch = []
			for lbl in selected_labels:
				indices = self.label_to_indices[lbl]
				if len(indices) >= self.k:
					sampled = random.sample(indices, self.k)
				else:
					# Nếu class có ít hơn K mẫu → sample có lặp lại
					sampled = random.choices(indices, k=self.k)
				batch.extend(sampled)

			yield batch

	def __len__(self) -> int:
		return self.n_batches


# ============================================================
# Model
# ============================================================

class MetricModel(nn.Module):
	"""
	ConvNeXt-Tiny backbone + Projection Head → L2 Normalized embeddings.
	Output: vector 256 chiều đã chuẩn hóa L2 (nằm trên unit hypersphere).
	"""

	def __init__(self, embedding_dim: int = 256, freeze_ratio: float = 0.90) -> None:
		super().__init__()
		self.backbone = timm.create_model(
			MODEL_NAME, pretrained=True, num_classes=0, global_pool="avg",
		)
		freeze_model_layers(self.backbone, freeze_ratio)

		# Lấy backbone output dimension
		backbone_dim = self.backbone.num_features  # 768 cho ConvNeXt-Tiny

		self.projector = nn.Sequential(
			nn.Linear(backbone_dim, backbone_dim),
			nn.BatchNorm1d(backbone_dim),
			nn.ReLU(inplace=True),
			nn.Linear(backbone_dim, embedding_dim),
		)
		self.model_name = f"{MODEL_NAME}_contrastive"

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		features = self.backbone(x)
		embeddings = self.projector(features)
		embeddings = F.normalize(embeddings, p=2, dim=1)
		return embeddings


# ============================================================
# Contrastive Loss — Online Hard Pair Mining
# ============================================================

class OnlineContrastiveLoss(nn.Module):
	"""
	Contrastive Loss với Online Hard Pair Mining.
	Trong mỗi batch, cho mỗi anchor:
	  - Hard positive: mẫu cùng class có khoảng cách LỚN nhất
	  - Hard negative: mẫu khác class có khoảng cách NHỎ nhất
	Loss = mean( D(a,p)^2 + max(0, margin - D(a,n))^2 )
	"""

	def __init__(self, margin: float = 1.0) -> None:
		super().__init__()
		self.margin = margin

	def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
		# Ma trận khoảng cách Euclid (n x n)
		dist_matrix = torch.cdist(embeddings, embeddings, p=2)
		n = embeddings.size(0)

		# Ma trận mask
		labels_col = labels.unsqueeze(1)  # (n, 1)
		is_positive = (labels_col == labels_col.t()).float()  # (n, n)
		is_negative = 1.0 - is_positive
		mask_diag = 1.0 - torch.eye(n, device=embeddings.device)
		is_positive = is_positive * mask_diag
		is_negative = is_negative * mask_diag

		# Hard positive: khoảng cách lớn nhất trong cùng class
		pos_dists = dist_matrix * is_positive
		hardest_pos, _ = pos_dists.max(dim=1)

		# Hard negative: khoảng cách nhỏ nhất trong khác class
		neg_dists = dist_matrix + (1.0 - is_negative) * 1e6
		hardest_neg, _ = neg_dists.min(dim=1)

		# Chỉ tính loss cho anchor có cả positive lẫn negative
		has_pos = is_positive.sum(dim=1) > 0
		has_neg = is_negative.sum(dim=1) > 0
		valid = has_pos & has_neg

		if valid.sum() == 0:
			return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

		pos_loss = hardest_pos[valid] ** 2
		neg_loss = F.relu(self.margin - hardest_neg[valid]) ** 2
		loss = (pos_loss + neg_loss).mean()
		return loss


# ============================================================
# Training Loop
# ============================================================

def train_one_epoch(
	model: nn.Module,
	loader: DataLoader,
	optimizer: torch.optim.Optimizer,
	criterion: nn.Module,
	device: torch.device,
	epoch: int,
	total_epochs: int,
) -> float:
	"""Huấn luyện 1 epoch, trả về loss trung bình."""
	model.train()
	total_loss = 0.0
	n_batches = 0

	pbar = tqdm(loader, desc=f"Train {epoch}/{total_epochs}")
	for images, labels in pbar:
		images = images.to(device, non_blocking=True)
		labels = labels.to(device, non_blocking=True)

		optimizer.zero_grad()
		embeddings = model(images)
		loss = criterion(embeddings, labels)
		loss.backward()
		optimizer.step()

		total_loss += loss.item()
		n_batches += 1
		pbar.set_postfix(loss=f"{total_loss / n_batches:.4f}")

	return total_loss / max(n_batches, 1)


# ============================================================
# Evaluation — Retrieval Metrics
# ============================================================

@torch.no_grad()
def extract_all_embeddings(
	model: nn.Module, loader: DataLoader, device: torch.device,
) -> tuple[torch.Tensor, np.ndarray]:
	"""Trích xuất embeddings và labels cho toàn bộ loader."""
	model.eval()
	all_embs = []
	all_labels = []
	for images, labels in loader:
		images = images.to(device, non_blocking=True)
		embs = model(images)
		all_embs.append(embs.cpu())
		all_labels.extend(labels.tolist())
	return torch.cat(all_embs, dim=0), np.array(all_labels)


def evaluate_retrieval(
	model: nn.Module,
	loader: DataLoader,
	device: torch.device,
	class_names: list[str],
	k_values: list[int] | None = None,
) -> dict:
	"""
	Tính toán Recall@K, Precision@K, mAP, AUC trên một tập dữ liệu.
	Mỗi ảnh làm query, toàn bộ ảnh còn lại là gallery.
	"""
	if k_values is None:
		k_values = [1, 5, 10]

	embeddings, labels = extract_all_embeddings(model, loader, device)
	n = len(labels)

	# Ma trận khoảng cách Euclid (n x n) trên CPU
	dist_matrix = torch.cdist(embeddings, embeddings, p=2).numpy()

	recall_at_k = {k: 0.0 for k in k_values}
	precision_at_k = {k: 0.0 for k in k_values}
	aps = []

	for i in range(n):
		# Loại bỏ chính mình
		dists = dist_matrix[i].copy()
		dists[i] = np.inf
		sorted_indices = np.argsort(dists)[:-1]

		# Ground truth: cùng class
		retrieved_labels = labels[sorted_indices]
		is_relevant = (retrieved_labels == labels[i])
		n_positives = int((labels == labels[i]).sum()) - 1  # trừ chính mình

		if n_positives == 0:
			continue

		# Recall@K và Precision@K
		for k in k_values:
			top_k_relevant = is_relevant[:k]
			recall_at_k[k] += float(top_k_relevant.any())
			precision_at_k[k] += float(top_k_relevant.sum()) / k

		# Average Precision (AP)
		cumsum = np.cumsum(is_relevant).astype(np.float64)
		precision_curve = cumsum / np.arange(1, n, dtype=np.float64)
		ap = (precision_curve * is_relevant).sum() / n_positives
		aps.append(ap)

	# Trung bình
	n_valid = max(len(aps), 1)
	for k in k_values:
		recall_at_k[k] /= n_valid
		precision_at_k[k] /= n_valid
	mAP = float(np.mean(aps)) if aps else 0.0

	# AUC — dùng negative distance làm score (càng gần → score càng cao)
	auc = 0.0
	try:
		pair_labels = []
		pair_scores = []
		for i in range(n):
			for j in range(i + 1, n):
				pair_labels.append(int(labels[i] == labels[j]))
				pair_scores.append(-dist_matrix[i][j])
		auc = float(roc_auc_score(pair_labels, pair_scores))
	except Exception:
		auc = 0.0

	results: dict = {"mAP": mAP, "AUC": auc}
	for k in k_values:
		results[f"Recall@{k}"] = recall_at_k[k]
		results[f"Precision@{k}"] = precision_at_k[k]

	return results


def format_retrieval_report(results: dict, prefix: str = "") -> str:
	"""Định dạng bảng kết quả retrieval dạng text."""
	header = f"{'Metric':<20} | {'Value':>10}"
	sep = "-" * 35
	lines = [
		f"\n{'=' * 35}",
		f"  Retrieval Report ({prefix})",
		f"{'=' * 35}",
		header,
		sep,
	]
	for k, v in results.items():
		lines.append(f"{k:<20} | {v * 100:>9.2f}%")
	lines.append("=" * 35)
	return "\n".join(lines)


class MetricGradCAM:
	def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
		self.model = model
		self.target_layer = target_layer
		self.activations = None
		self.forward_handle = target_layer.register_forward_hook(self._forward_hook)

	def _forward_hook(self, module, inputs, output):
		self.activations = output

	def remove(self) -> None:
		self.forward_handle.remove()

	def __call__(self, input_tensor: torch.Tensor, prototype: torch.Tensor) -> np.ndarray:
		self.model.zero_grad()
		emb = self.model(input_tensor)
		score = (emb * prototype.unsqueeze(0)).sum()
		if self.activations is None:
			raise RuntimeError("GradCAM hook did not capture activations")

		grads = torch.autograd.grad(score, self.activations, retain_graph=True)[0]
		weights = grads.mean(dim=(2, 3), keepdim=True)
		cam = (weights * self.activations).sum(dim=1, keepdim=True)
		cam = F.relu(cam)
		cam = cam.squeeze().detach().cpu().numpy()
		if cam.ndim == 0:
			cam = np.array([[float(cam)]])
		elif cam.ndim == 1:
			cam = cam[None, :]
		cam -= cam.min()
		if cam.max() > 0:
			cam /= cam.max()
		return cam


def find_last_conv_layer(model: nn.Module) -> nn.Module | None:
	last_conv = None
	for module in model.modules():
		if isinstance(module, nn.Conv2d):
			last_conv = module
	return last_conv


def overlay_cam_on_image(image: Image.Image, cam: np.ndarray, alpha: float = 0.4) -> Image.Image:
	cam_resized = Image.fromarray((cam * 255).astype(np.uint8)).resize(image.size)
	cam_resized = np.array(cam_resized) / 255.0
	color_map = plt.get_cmap("jet")
	heatmap = color_map(cam_resized)[:, :, :3]
	img = np.array(image).astype(np.float32) / 255.0
	overlay = img * (1 - alpha) + heatmap * alpha
	overlay = np.clip(overlay, 0, 1)
	return Image.fromarray((overlay * 255).astype(np.uint8))


def select_gradcam_representatives(df: pd.DataFrame, seed: int = 42) -> dict[str, list[dict]]:
	representatives = {'Dalbergia': [], 'Pterocarpus': []}
	unique_classes = df['label'].unique()
	for genus in ['Dalbergia', 'Pterocarpus']:
		genus_classes = [c for c in unique_classes if c.startswith(genus)]
		genus_classes = sorted(genus_classes)
		for cls in genus_classes:
			cls_df = df[df['label'] == cls]
			if len(cls_df) >= 2:
				sampled = cls_df.sample(n=2, random_state=seed).reset_index(drop=True)
			else:
				sampled = cls_df.sample(n=2, replace=True, random_state=seed).reset_index(drop=True)
			for _, row in sampled.iterrows():
				representatives[genus].append({
					'path': row['path'],
					'label': row['label'],
					'species': cls.replace(genus + " ", "")
				})
	return representatives


@torch.no_grad()
def compute_class_prototypes(model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int) -> torch.Tensor:
	model.eval()
	embeddings = []
	labels = []
	for images, batch_labels in loader:
		images = images.to(device, non_blocking=True)
		embs = model(images)
		embeddings.append(embs.cpu())
		labels.extend(batch_labels.tolist())
	embeddings = torch.cat(embeddings, dim=0)
	labels = np.array(labels)
	embedding_dim = embeddings.shape[1]
	prototypes = torch.zeros(num_classes, embedding_dim)
	for c in range(num_classes):
		idx = np.where(labels == c)[0]
		if len(idx) > 0:
			class_embs = embeddings[idx]
			proto = class_embs.mean(dim=0)
			proto = F.normalize(proto, p=2, dim=0)
			prototypes[c] = proto
		else:
			prototypes[c] = torch.zeros(embedding_dim)
	return prototypes


def generate_gradcam_maps(
	model: nn.Module,
	representatives: list[dict],
	prototypes: torch.Tensor,
	class_to_idx: dict[str, int],
	transform,
	device: torch.device
) -> list[np.ndarray]:
	target_layer = find_last_conv_layer(model)
	if target_layer is None:
		print("Warning: Không tìm thấy Conv2d layer trong model để vẽ Grad-CAM")
		return [np.zeros((224, 224)) for _ in representatives]
	gradcam = MetricGradCAM(model, target_layer)
	model.eval()
	cam_maps = []
	for rep in representatives:
		with Image.open(rep['path']) as img:
			img = img.convert("RGB")
		input_tensor = transform(img).unsqueeze(0).to(device)
		class_idx = class_to_idx[rep['label']]
		proto = prototypes[class_idx].to(device)
		cam = gradcam(input_tensor, proto)
		cam_maps.append(cam)
	gradcam.remove()
	return cam_maps


def plot_gradcam_comparison(
	representatives: list[dict],
	before_cams: list[np.ndarray],
	after_cams: list[np.ndarray],
	genus_name: str,
	output_path: Path
) -> None:
	n_samples = len(representatives)
	if n_samples == 0:
		return
	fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))
	if n_samples == 1:
		axes = np.expand_dims(axes, axis=0)
	for i in range(n_samples):
		rep = representatives[i]
		with Image.open(rep['path']) as img:
			img = img.convert("RGB")
		axes[i, 0].imshow(img)
		axes[i, 0].axis('off')
		axes[i, 0].set_title(f"{rep['label']}\nSample {i % 2 + 1}", fontsize=10, fontweight='bold')

		cam_before = before_cams[i]
		overlay_before = overlay_cam_on_image(img, cam_before)
		axes[i, 1].imshow(overlay_before)
		axes[i, 1].axis('off')
		axes[i, 1].set_title("Before Training", fontsize=9)

		cam_after = after_cams[i]
		overlay_after = overlay_cam_on_image(img, cam_after)
		axes[i, 2].imshow(overlay_after)
		axes[i, 2].axis('off')
		axes[i, 2].set_title("After Training", fontsize=9)
	plt.suptitle(f"Grad-CAM Comparison for Genus {genus_name}", fontsize=14, fontweight='bold', y=0.99)
	plt.tight_layout()
	plt.savefig(output_path, dpi=200, bbox_inches='tight')
	plt.close()
	print(f"Grad-CAM comparison saved → {output_path}")


def plot_tsne_comparison(
	before_embs: np.ndarray,
	after_embs: np.ndarray,
	labels: np.ndarray,
	class_names: list[str],
	output_path: Path
) -> None:
	print("Running t-SNE dimensionality reduction...")
	n_samples = len(labels)
	perp = min(30, max(5, n_samples // 4))
	tsne = TSNE(n_components=2, perplexity=perp, random_state=42, n_iter=1000)
	embs_2d_before = tsne.fit_transform(before_embs)
	embs_2d_after = tsne.fit_transform(after_embs)

	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
	unique_labels = np.unique(labels)
	cmap = plt.get_cmap("tab20", len(unique_labels))

	for idx, label_idx in enumerate(unique_labels):
		class_name = class_names[label_idx]
		color = cmap(idx)
		mask = (labels == label_idx)
		ax1.scatter(
			embs_2d_before[mask, 0], embs_2d_before[mask, 1],
			color=color, label=class_name, alpha=0.8, edgecolors='k', s=50
		)
		ax2.scatter(
			embs_2d_after[mask, 0], embs_2d_after[mask, 1],
			color=color, label=class_name, alpha=0.8, edgecolors='k', s=50
		)

	ax1.set_title("Feature Space Before Metric Learning (t-SNE)", fontsize=12, fontweight='bold')
	ax1.grid(alpha=0.3)
	ax1.set_xlabel("t-SNE Dimension 1")
	ax1.set_ylabel("t-SNE Dimension 2")

	ax2.set_title("Feature Space After Metric Learning (t-SNE)", fontsize=12, fontweight='bold')
	ax2.grid(alpha=0.3)
	ax2.set_xlabel("t-SNE Dimension 1")
	ax2.set_ylabel("t-SNE Dimension 2")

	handles, labels_legend = ax2.get_legend_handles_labels()
	fig.legend(handles, labels_legend, loc='center right', bbox_to_anchor=(0.99, 0.5), fontsize=9)

	plt.tight_layout()
	plt.subplots_adjust(right=0.83)
	plt.savefig(output_path, dpi=200, bbox_inches='tight')
	plt.close()
	print(f"t-SNE comparison saved → {output_path}")


def calculate_pairwise_distances(embeddings: np.ndarray, labels: np.ndarray) -> tuple[list[float], list[float]]:
	n = len(labels)
	dist_matrix = np.sqrt(np.maximum(2.0 - 2.0 * np.dot(embeddings, embeddings.T), 0.0))
	intra_dists = []
	inter_dists = []
	for i in range(n):
		for j in range(i + 1, n):
			d = dist_matrix[i, j]
			if labels[i] == labels[j]:
				intra_dists.append(d)
			else:
				inter_dists.append(d)
	return intra_dists, inter_dists


def plot_distance_analysis(
	before_embs: np.ndarray,
	after_embs: np.ndarray,
	labels: np.ndarray,
	output_path: Path
) -> None:
	intra_before, inter_before = calculate_pairwise_distances(before_embs, labels)
	intra_after, inter_after = calculate_pairwise_distances(after_embs, labels)

	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

	ax1.hist(intra_before, bins=30, alpha=0.6, density=True, label="Intra-class (Same Class)", color="#3498db")
	ax1.hist(inter_before, bins=30, alpha=0.6, density=True, label="Inter-class (Diff Class)", color="#e74c3c")
	ax1.set_xlabel("Euclidean Distance")
	ax1.set_ylabel("Density")
	ax1.set_title("Distance Distribution Before Training", fontsize=11, fontweight='bold')
	ax1.legend()
	ax1.grid(alpha=0.3)

	ax2.hist(intra_after, bins=30, alpha=0.6, density=True, label="Intra-class (Same Class)", color="#2ecc71")
	ax2.hist(inter_after, bins=30, alpha=0.6, density=True, label="Inter-class (Diff Class)", color="#e74c3c")
	ax2.set_xlabel("Euclidean Distance")
	ax2.set_ylabel("Density")
	ax2.set_title("Distance Distribution After Training", fontsize=11, fontweight='bold')
	ax2.legend()
	ax2.grid(alpha=0.3)

	plt.suptitle("Pairwise Distance Analysis: Intra-class vs Inter-class", fontsize=13, fontweight='bold')
	plt.tight_layout()
	plt.savefig(output_path, dpi=200, bbox_inches='tight')
	plt.close()
	print(f"Distance analysis distribution saved → {output_path}")


def evaluate_loss(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
	model.eval()
	total_loss = 0.0
	n_batches = 0
	with torch.no_grad():
		for images, labels in loader:
			images = images.to(device, non_blocking=True)
			labels = labels.to(device, non_blocking=True)
			embeddings = model(images)
			loss = criterion(embeddings, labels)
			total_loss += loss.item()
			n_batches += 1
	return total_loss / max(n_batches, 1)


def plot_metrics_summary(history: dict, model: nn.Module, val_loader: DataLoader, test_loader: DataLoader, device: torch.device, output_dir: Path) -> None:
	fig, axes = plt.subplots(2, 4, figsize=(20, 10))
	epochs_range = range(1, len(history["train_loss"]) + 1)

	# 1. Loss
	axes[0, 0].plot(epochs_range, history["train_loss"], label="Train Loss", color="#3498db", lw=2)
	axes[0, 0].plot(epochs_range, history["val_loss"], label="Val Loss", color="#e74c3c", lw=2)
	axes[0, 0].set_xlabel("Epoch")
	axes[0, 0].set_ylabel("Loss")
	axes[0, 0].set_title("Loss Curve")
	axes[0, 0].legend()
	axes[0, 0].grid(alpha=0.3)

	# 2. Recall@1
	axes[0, 1].plot(epochs_range, history["train_recall1"], label="Train R@1", color="#3498db", lw=2)
	axes[0, 1].plot(epochs_range, history["val_recall1"], label="Val R@1", color="#2ecc71", lw=2)
	axes[0, 1].set_xlabel("Epoch")
	axes[0, 1].set_ylabel("Recall@1")
	axes[0, 1].set_title("Recall@1 Curve")
	axes[0, 1].legend()
	axes[0, 1].grid(alpha=0.3)

	# 3. Recall@5
	axes[0, 2].plot(epochs_range, history["train_recall5"], label="Train R@5", color="#3498db", lw=2)
	axes[0, 2].plot(epochs_range, history["val_recall5"], label="Val R@5", color="#2ecc71", lw=2)
	axes[0, 2].set_xlabel("Epoch")
	axes[0, 2].set_ylabel("Recall@5")
	axes[0, 2].set_title("Recall@5 Curve")
	axes[0, 2].legend()
	axes[0, 2].grid(alpha=0.3)

	# 4. Precision@1
	axes[0, 3].plot(epochs_range, history["train_precision1"], label="Train P@1", color="#3498db", lw=2)
	axes[0, 3].plot(epochs_range, history["val_precision1"], label="Val P@1", color="#2ecc71", lw=2)
	axes[0, 3].set_xlabel("Epoch")
	axes[0, 3].set_ylabel("Precision@1")
	axes[0, 3].set_title("Precision@1 Curve")
	axes[0, 3].legend()
	axes[0, 3].grid(alpha=0.3)

	# 5. Precision@5
	axes[1, 0].plot(epochs_range, history["train_precision5"], label="Train P@5", color="#3498db", lw=2)
	axes[1, 0].plot(epochs_range, history["val_precision5"], label="Val P@5", color="#2ecc71", lw=2)
	axes[1, 0].set_xlabel("Epoch")
	axes[1, 0].set_ylabel("Precision@5")
	axes[1, 0].set_title("Precision@5 Curve")
	axes[1, 0].legend()
	axes[1, 0].grid(alpha=0.3)

	# 6. mAP
	axes[1, 1].plot(epochs_range, history["train_map"], label="Train mAP", color="#3498db", lw=2)
	axes[1, 1].plot(epochs_range, history["val_map"], label="Val mAP", color="#2ecc71", lw=2)
	axes[1, 1].set_xlabel("Epoch")
	axes[1, 1].set_ylabel("mAP")
	axes[1, 1].set_title("mAP Curve")
	axes[1, 1].legend()
	axes[1, 1].grid(alpha=0.3)

	# 7. AUC
	axes[1, 2].plot(epochs_range, history["train_auc"], label="Train AUC", color="#3498db", lw=2)
	axes[1, 2].plot(epochs_range, history["val_auc"], label="Val AUC", color="#2ecc71", lw=2)
	axes[1, 2].set_xlabel("Epoch")
	axes[1, 2].set_ylabel("AUC")
	axes[1, 2].set_title("AUC Curve")
	axes[1, 2].legend()
	axes[1, 2].grid(alpha=0.3)

	# 8. ROC Curves
	ax_roc = axes[1, 3]
	def plot_roc_helper(loader, label_str, color):
		embs, lbls = extract_all_embeddings(model, loader, device)
		n = len(lbls)
		dist_matrix = torch.cdist(embs, embs, p=2).numpy()
		pair_labels = []
		pair_scores = []
		for i in range(n):
			for j in range(i + 1, n):
				pair_labels.append(int(lbls[i] == lbls[j]))
				pair_scores.append(-dist_matrix[i, j])
		fpr, tpr, _ = roc_curve(pair_labels, pair_scores)
		roc_auc = auc(fpr, tpr)
		ax_roc.plot(fpr, tpr, color=color, lw=2, label=f"{label_str} (AUC = {roc_auc:.4f})")

	try:
		plot_roc_helper(val_loader, "Val ROC", "#2ecc71")
		plot_roc_helper(test_loader, "Test ROC", "#e74c3c")
	except Exception as e:
		print(f"Error plotting ROC curves: {e}")

	ax_roc.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
	ax_roc.set_xlim([0.0, 1.0])
	ax_roc.set_ylim([0.0, 1.05])
	ax_roc.set_xlabel("False Positive Rate")
	ax_roc.set_ylabel("True Positive Rate")
	ax_roc.set_title("ROC Curves")
	ax_roc.legend(loc="lower right")
	ax_roc.grid(alpha=0.3)

	plt.suptitle("Training Metrics Summary & ROC Analysis", fontsize=16, fontweight='bold', y=0.98)
	plt.tight_layout()
	plt.savefig(output_dir / "metrics_summary.png", dpi=200, bbox_inches="tight")
	plt.close()
	print(f"Metrics summary saved → {output_dir / 'metrics_summary.png'}")


# ============================================================
# Main
# ============================================================

def main() -> None:
	set_seed(SEED)
	device = get_device()
	print(f"Device: {device}")

	output_dir = Path(OUTPUT_DIR)
	output_dir.mkdir(parents=True, exist_ok=True)

	# ── 1. Thu thập ảnh ──
	print("\n[Step 1] Thu thập ảnh...")
	samples = collect_image_samples(ROOT_DIR)
	if not samples:
		raise ValueError(f"Không tìm thấy ảnh nào trong {ROOT_DIR}")

	df = build_dataframe(samples)
	df_filtered = df[df["label"] != "Pterocarpus sp"].reset_index(drop=True)
	print(f"Tổng ảnh: {len(df_filtered)}, Số class: {df_filtered['label'].nunique()}")

	class_names = sorted(df_filtered["label"].unique().tolist())
	class_to_idx = {name: i for i, name in enumerate(class_names)}

	# Lưu class index
	with open(output_dir / "class_indices.json", "w", encoding="utf-8") as f:
		json.dump(class_to_idx, f, indent=2, ensure_ascii=False)

	# ── 2. Compute embeddings cho chia dữ liệu ──
	print("\n[Step 2] Compute embeddings cho chia dữ liệu...")
	embeddings = compute_embeddings(df_filtered, batch_size=EMB_BATCH_SIZE, device=device)
	print(f"Embeddings shape: {embeddings.shape}")

	if device.type == "cuda":
		torch.cuda.empty_cache()

	# ── 3. Chia dữ liệu End Version ──
	print("\n[Step 3] Chia dữ liệu theo End Version Split...")
	df_train, df_val, df_test = end_version_split(
		df_filtered, embeddings,
		train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, seed=SEED,
	)
	validate_split(df_filtered, df_train, df_val, df_test, "Contrastive_EndVersion")
	log_split_summary(df_filtered, df_train, df_val, df_test)
	eda_split_class_distribution(
		df_train, df_val, df_test,
		"Contrastive - Class Distribution",
		output_dir / "eda_split_contrastive.png",
	)

	# Giải phóng embeddings (không cần nữa)
	del embeddings
	gc.collect()
	if device.type == "cuda":
		torch.cuda.empty_cache()

	# ── 4. Khởi tạo model ──
	print("\n[Step 4] Khởi tạo MetricModel...")
	model = MetricModel(embedding_dim=EMBEDDING_DIM, freeze_ratio=FREEZE_RATIO)
	model_info = summarize_model(model)
	print(
		f"Model: {model.model_name}, "
		f"total={model_info['total_params']:,}, "
		f"trainable={model_info['trainable_params']:,}, "
		f"frozen={model_info['frozen_params']:,}"
	)

	# Lấy data config trước khi wrap DataParallel
	cfg = resolve_data_config({}, model=model.backbone)
	img_size = cfg.get("input_size", (3, 224, 224))[-1]
	mean = cfg.get("mean", (0.485, 0.456, 0.406))
	std = cfg.get("std", (0.229, 0.224, 0.225))

	model = model.to(device)

	# ── 5. Transforms ──
	train_tf = transforms.Compose([
		transforms.Resize((img_size, img_size)),
		transforms.RandomRotation(degrees=15),
		transforms.RandomHorizontalFlip(p=0.5),
		transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
		transforms.ToTensor(),
		transforms.Normalize(mean=mean, std=std),
	])
	eval_tf = transforms.Compose([
		transforms.Resize((img_size, img_size)),
		transforms.ToTensor(),
		transforms.Normalize(mean=mean, std=std),
	])

	# ── 6. Datasets & Loaders ──
	train_ds = MetricImageDataset(df_train, class_to_idx, transform=train_tf)
	val_ds = MetricImageDataset(df_val, class_to_idx, transform=eval_tf)
	test_ds = MetricImageDataset(df_test, class_to_idx, transform=eval_tf)

	pk_sampler = PKSampler(train_ds.labels, p=P_CLASSES, k=K_SAMPLES)
	train_loader = DataLoader(train_ds, batch_sampler=pk_sampler, num_workers=0, pin_memory=True)
	val_loader = DataLoader(val_ds, batch_size=P_CLASSES * K_SAMPLES, shuffle=False, num_workers=0, pin_memory=True)
	test_loader = DataLoader(test_ds, batch_size=P_CLASSES * K_SAMPLES, shuffle=False, num_workers=0, pin_memory=True)

	print(f"Train: {len(train_ds)} ảnh | Val: {len(val_ds)} ảnh | Test: {len(test_ds)} ảnh")
	print(f"PK Sampler: P={P_CLASSES}, K={K_SAMPLES} → batch_size={P_CLASSES * K_SAMPLES}")
	print(f"Batches/epoch: {len(pk_sampler)}")

	# ── 7. Loss, Optimizer, Scheduler ──
	criterion = OnlineContrastiveLoss(margin=MARGIN)
	optimizer = torch.optim.AdamW(
		filter(lambda p: p.requires_grad, model.parameters()),
		lr=LR, weight_decay=WEIGHT_DECAY,
	)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

	# ── 8. Trích xuất trạng thái trước training (Pre-training analysis) ──
	print("\n[Analysis] Trích xuất đặc trưng và Grad-CAM trước training...")
	train_eval_loader = DataLoader(train_ds, batch_size=P_CLASSES * K_SAMPLES, shuffle=False, num_workers=0, pin_memory=True)
	
	# Chọn ảnh Grad-CAM
	reps_dict = select_gradcam_representatives(df_filtered, seed=SEED)
	reps_flat = reps_dict['Dalbergia'] + reps_dict['Pterocarpus']
	
	# Tính prototype & CAM trước training
	before_protos = compute_class_prototypes(model, train_eval_loader, device, len(class_names))
	before_cams = generate_gradcam_maps(model, reps_flat, before_protos, class_to_idx, eval_tf, device)
	
	# Tính embeddings validation trước training (cho t-SNE & distance analysis)
	before_val_embs, val_labels = extract_all_embeddings(model, val_loader, device)
	before_val_embs = before_val_embs.numpy()

	# ── 9. Training loop ──
	print(f"\n{'=' * 60}")
	print(f"  Bắt đầu huấn luyện Contrastive Loss — {EPOCHS} epochs")
	print(f"  Margin={MARGIN}, LR={LR}, EarlyStopping patience={PATIENCE}")
	print(f"{'=' * 60}\n")

	history = {
		"train_loss": [], "val_loss": [],
		"train_recall1": [], "val_recall1": [],
		"train_recall5": [], "val_recall5": [],
		"train_precision1": [], "val_precision1": [],
		"train_precision5": [], "val_precision5": [],
		"train_map": [], "val_map": [],
		"train_auc": [], "val_auc": []
	}
	best_recall1 = 0.0
	epochs_no_improve = 0

	for epoch in range(1, EPOCHS + 1):
		# Train
		loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, EPOCHS)
		
		# Val loss
		val_loss = evaluate_loss(model, val_loader, criterion, device)
		
		# Evaluate
		train_results = evaluate_retrieval(model, train_eval_loader, device, class_names, k_values=[1, 5])
		val_results = evaluate_retrieval(model, val_loader, device, class_names, k_values=[1, 5])
		
		val_recall1 = val_results["Recall@1"]
		
		history["train_loss"].append(loss)
		history["val_loss"].append(val_loss)
		history["train_recall1"].append(train_results["Recall@1"])
		history["val_recall1"].append(val_results["Recall@1"])
		history["train_recall5"].append(train_results["Recall@5"])
		history["val_recall5"].append(val_results["Recall@5"])
		history["train_precision1"].append(train_results["Precision@1"])
		history["val_precision1"].append(val_results["Precision@1"])
		history["train_precision5"].append(train_results["Precision@5"])
		history["val_precision5"].append(val_results["Precision@5"])
		history["train_map"].append(train_results["mAP"])
		history["val_map"].append(val_results["mAP"])
		history["train_auc"].append(train_results["AUC"])
		history["val_auc"].append(val_results["AUC"])

		current_lr = optimizer.param_groups[0]["lr"]
		print(
			f"Epoch {epoch}/{EPOCHS} —\n"
			f"  Loss (Train/Val): {loss:.4f} / {val_loss:.4f}\n"
			f"  Recall@1 (Train/Val): {train_results['Recall@1']*100:.2f}% / {val_results['Recall@1']*100:.2f}%\n"
			f"  Recall@5 (Train/Val): {train_results['Recall@5']*100:.2f}% / {val_results['Recall@5']*100:.2f}%\n"
			f"  Precision@1 (Train/Val): {train_results['Precision@1']*100:.2f}% / {val_results['Precision@1']*100:.2f}%\n"
			f"  Precision@5 (Train/Val): {train_results['Precision@5']*100:.2f}% / {val_results['Precision@5']*100:.2f}%\n"
			f"  mAP (Train/Val): {train_results['mAP']*100:.2f}% / {val_results['mAP']*100:.2f}%\n"
			f"  AUC (Train/Val): {train_results['AUC']:.4f} / {val_results['AUC']:.4f}\n"
			f"  LR: {current_lr:.6f}"
		)

		scheduler.step()

		# Checkpoint best model
		if val_recall1 > best_recall1:
			best_recall1 = val_recall1
			epochs_no_improve = 0
			raw_model = model
			torch.save(raw_model.state_dict(), output_dir / "best_model.pth")
			print(f"  → Saved best model (Recall@1={best_recall1:.4f})")
		else:
			epochs_no_improve += 1

		if epochs_no_improve >= PATIENCE:
			print(f"\nEarly stopping tại epoch {epoch} (patience={PATIENCE})")
			break

	# ── 10. Load best & đánh giá cuối ──
	raw_model = model
	best_path = output_dir / "best_model.pth"
	if best_path.exists():
		raw_model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
		print(f"\nĐã load best model từ {best_path}")

	# Đánh giá Val
	print("\n[Đánh giá cuối - Validation]")
	val_results = evaluate_retrieval(model, val_loader, device, class_names)
	val_report = format_retrieval_report(val_results, prefix="Val")
	print(val_report)
	with open(output_dir / "retrieval_report_val.txt", "w", encoding="utf-8") as f:
		f.write(val_report)

	# Đánh giá Test
	print("\n[Đánh giá cuối - Test]")
	test_results = evaluate_retrieval(model, test_loader, device, class_names)
	test_report = format_retrieval_report(test_results, prefix="Test")
	print(test_report)
	with open(output_dir / "retrieval_report_test.txt", "w", encoding="utf-8") as f:
		f.write(test_report)

	# ── 11. Trích xuất trạng thái sau training (Post-training analysis) ──
	print("\n[Analysis] Trích xuất đặc trưng và sinh Grad-CAM sau training...")
	after_protos = compute_class_prototypes(model, train_eval_loader, device, len(class_names))
	after_cams = generate_gradcam_maps(model, reps_flat, after_protos, class_to_idx, eval_tf, device)
	
	# Chia cam cho từng chi
	n_dal = len(reps_dict['Dalbergia'])
	plot_gradcam_comparison(reps_dict['Dalbergia'], before_cams[:n_dal], after_cams[:n_dal], 'Dalbergia', output_dir / "gradcam_dalbergia.png")
	plot_gradcam_comparison(reps_dict['Pterocarpus'], before_cams[n_dal:], after_cams[n_dal:], 'Pterocarpus', output_dir / "gradcam_pterocarpus.png")
	
	# Tính embeddings validation sau training
	after_val_embs, _ = extract_all_embeddings(model, val_loader, device)
	after_val_embs = after_val_embs.numpy()
	
	# Vẽ t-SNE so sánh
	plot_tsne_comparison(before_val_embs, after_val_embs, val_labels, class_names, output_dir / "tsne_comparison.png")
	
	# Phân tích khoảng cách
	plot_distance_analysis(before_val_embs, after_val_embs, val_labels, output_dir / "distance_distribution.png")

	# Vẽ biểu đồ tổng hợp metrics
	plot_metrics_summary(history, model, val_loader, test_loader, device, output_dir)

	summary = {
		"method": "Contrastive Loss (Online Hard Pair Mining)",
		"model": MODEL_NAME,
		"embedding_dim": EMBEDDING_DIM,
		"margin": MARGIN,
		"p_classes": P_CLASSES,
		"k_samples": K_SAMPLES,
		"batch_size": P_CLASSES * K_SAMPLES,
		"epochs_trained": len(history["train_loss"]),
		"best_val_recall1": best_recall1,
		"val_results": {k: round(v, 6) for k, v in val_results.items()},
		"test_results": {k: round(v, 6) for k, v in test_results.items()},
		"train_size": len(df_train),
		"val_size": len(df_val),
		"test_size": len(df_test),
	}
	with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
		json.dump(summary, f, indent=2, ensure_ascii=False)

	# Giải phóng bộ nhớ
	del model, optimizer, criterion
	del train_loader, val_loader, test_loader, train_eval_loader
	del train_ds, val_ds, test_ds
	gc.collect()
	if device.type == "cuda":
		torch.cuda.empty_cache()

	print(f"\n[Hoàn tất] Tất cả kết quả lưu tại: {output_dir}/")


if __name__ == "__main__":
	main()
