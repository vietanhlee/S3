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
from sklearn.metrics import roc_auc_score, roc_curve, auc, silhouette_score, davies_bouldin_score, calinski_harabasz_score, normalized_mutual_info_score
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.manifold import TSNE

# ===== CẤU HÌNH =====
ROOT_DIR = r"/kaggle/input/datasets/b23dckh002lvitanh/s3-origin/S3"
OUTPUT_DIR = "outputs_contrastive"
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
SEED = 42
P_CLASSES = 18          # Số class mỗi batch
K_SAMPLES = 10          # Số ảnh mỗi class trong batch
EPOCHS = 30
PATIENCE = 6           # EarlyStopping patience
CAM_METHODS = ["gradcam", "gradcam++", "xgradcam", "eigencam", "hirescam", "layercam", "eigengradcam", "finercam"]
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
from train_final import end_version_split, compute_embeddings_v2
from split_methods import validate_split


def compute_dunn_index(embeddings: np.ndarray, labels: np.ndarray) -> float:
	"""
	Tính Dunn Index: tỷ lệ giữa khoảng cách liên cụm nhỏ nhất và đường kính nội cụm lớn nhất.
	Dunn = min_inter_dist / max_intra_dist
	"""
	unique_labels = np.unique(labels)
	n_clusters = len(unique_labels)
	if n_clusters <= 1:
		return 0.0

	max_intra_dist = 0.0
	cluster_masks = [labels == label for label in unique_labels]
	cluster_embs = [embeddings[mask] for mask in cluster_masks]

	for embs in cluster_embs:
		if len(embs) > 1:
			dists = cdist(embs, embs, metric='euclidean')
			max_val = dists.max()
			if max_val > max_intra_dist:
				max_intra_dist = max_val

	if max_intra_dist == 0.0:
		return 0.0

	min_inter_dist = float('inf')
	for i in range(n_clusters):
		for j in range(i + 1, n_clusters):
			dists = cdist(cluster_embs[i], cluster_embs[j], metric='euclidean')
			min_val = dists.min()
			if min_val < min_inter_dist:
				min_inter_dist = min_val

	return float(min_inter_dist / max_intra_dist)


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
	Tính toán Recall@K, Precision@K, mAP, AUC toàn cục,
	các chỉ số phân cụm (Silhouette, Davies-Bouldin, Calinski-Harabasz, Dunn, NMI),
	và các chỉ số chi tiết cho từng class.
	"""
	if k_values is None:
		k_values = [1, 5, 10]

	embeddings, labels = extract_all_embeddings(model, loader, device)
	embeddings_np = embeddings.numpy()
	n = len(labels)
	n_classes = len(class_names)

	# Ma trận khoảng cách Euclid (n x n) trên CPU
	dist_matrix = torch.cdist(embeddings, embeddings, p=2).numpy()

	recall_at_k = {k: 0.0 for k in k_values}
	precision_at_k = {k: 0.0 for k in k_values}
	aps = []

	# Lưu metrics cá thể cho từng mẫu để tính per-class
	sample_recall1 = []
	sample_recall5 = []
	sample_map = []

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
			sample_recall1.append(0.0)
			sample_recall5.append(0.0)
			sample_map.append(0.0)
			continue

		# Recall@K và Precision@K cho toàn cục
		for k in k_values:
			top_k_relevant = is_relevant[:k]
			recall_at_k[k] += float(top_k_relevant.any())
			precision_at_k[k] += float(top_k_relevant.sum()) / k

		# Lưu thông tin cho mẫu i
		sample_recall1.append(float(is_relevant[:1].any()))
		sample_recall5.append(float(is_relevant[:5].any()))

		# AP của mẫu i
		cumsum = np.cumsum(is_relevant).astype(np.float64)
		precision_curve = cumsum / np.arange(1, n, dtype=np.float64)
		ap = (precision_curve * is_relevant).sum() / n_positives
		aps.append(ap)
		sample_map.append(ap)

	# Trung bình toàn cục
	n_valid = max(len(aps), 1)
	for k in k_values:
		recall_at_k[k] /= n_valid
		precision_at_k[k] /= n_valid
	mAP = float(np.mean(aps)) if aps else 0.0

	# AUC toàn cục — dùng negative distance làm score
	auc_val = 0.0
	try:
		pair_labels = []
		pair_scores = []
		for i in range(n):
			for j in range(i + 1, n):
				pair_labels.append(int(labels[i] == labels[j]))
				pair_scores.append(-dist_matrix[i][j])
		auc_val = float(roc_auc_score(pair_labels, pair_scores))
	except Exception:
		auc_val = 0.0

	# --- Tính toán các chỉ số phân cụm ---
	try:
		silhouette = float(silhouette_score(embeddings_np, labels))
	except Exception:
		silhouette = 0.0

	try:
		dbi = float(davies_bouldin_score(embeddings_np, labels))
	except Exception:
		dbi = 0.0

	try:
		chi = float(calinski_harabasz_score(embeddings_np, labels))
	except Exception:
		chi = 0.0

	try:
		dunn = float(compute_dunn_index(embeddings_np, labels))
	except Exception:
		dunn = 0.0

	try:
		kmeans = KMeans(n_clusters=n_classes, random_state=42, n_init=10)
		kmeans_labels = kmeans.fit_predict(embeddings_np)
		nmi = float(normalized_mutual_info_score(labels, kmeans_labels))
	except Exception:
		nmi = 0.0

	# --- Tính toán tỷ lệ Intra/Inter Class Distance ---
	try:
		intra_dists = []
		inter_dists = []
		for i in range(n):
			for j in range(i + 1, n):
				d = dist_matrix[i, j]
				if labels[i] == labels[j]:
					intra_dists.append(d)
				else:
					inter_dists.append(d)
		intra_mean = np.mean(intra_dists) if intra_dists else 0.0
		inter_mean = np.mean(inter_dists) if inter_dists else 0.0
		intra_inter_ratio = float(intra_mean / inter_mean) if inter_mean > 0 else 0.0
	except Exception:
		intra_inter_ratio = 0.0

	# --- Tính toán metrics cho từng class ---
	per_class_recall1 = []
	per_class_recall5 = []
	per_class_map = []
	per_class_auc = []

	for c in range(n_classes):
		class_indices = np.where(labels == c)[0]
		if len(class_indices) == 0:
			per_class_recall1.append(0.0)
			per_class_recall5.append(0.0)
			per_class_map.append(0.0)
			per_class_auc.append(0.0)
			continue

		# Lấy trung bình Recall/mAP của các mẫu thuộc class c
		c_recall1 = np.mean([sample_recall1[idx] for idx in class_indices])
		c_recall5 = np.mean([sample_recall5[idx] for idx in class_indices])
		c_map = np.mean([sample_map[idx] for idx in class_indices])

		# Tính AUC riêng cho class c
		c_auc = 0.0
		try:
			pair_labels_c = []
			pair_scores_c = []
			for idx_i in class_indices:
				# Cặp positive (cùng class c)
				for idx_j in class_indices:
					if idx_i < idx_j:
						pair_labels_c.append(1)
						pair_scores_c.append(-dist_matrix[idx_i][idx_j])
				# Cặp negative (khác class c)
				other_indices = np.where(labels != c)[0]
				for idx_k in other_indices:
					pair_labels_c.append(0)
					pair_scores_c.append(-dist_matrix[idx_i][idx_k])
			if len(set(pair_labels_c)) > 1:
				c_auc = float(roc_auc_score(pair_labels_c, pair_scores_c))
			else:
				c_auc = 0.0
		except Exception:
			c_auc = 0.0

		per_class_recall1.append(c_recall1)
		per_class_recall5.append(c_recall5)
		per_class_map.append(c_map)
		per_class_auc.append(c_auc)

	results = {
		"mAP": mAP,
		"AUC": auc_val,
		"Silhouette": silhouette,
		"Davies-Bouldin": dbi,
		"Calinski-Harabasz": chi,
		"Dunn-Index": dunn,
		"NMI": nmi,
		"Intra-Inter-Ratio": intra_inter_ratio,
		"per_class_recall1": per_class_recall1,
		"per_class_recall5": per_class_recall5,
		"per_class_map": per_class_map,
		"per_class_auc": per_class_auc,
	}
	for k in k_values:
		results[f"Recall@{k}"] = recall_at_k[k]
		results[f"Precision@{k}"] = precision_at_k[k]

	return results


def format_retrieval_report(results: dict, class_names: list[str], prefix: str = "") -> str:
	"""Tạo báo cáo chi tiết per-class và global dưới dạng bảng text chuyên nghiệp."""
	lines = []
	lines.append(f"\n=======================================================================")
	lines.append(f"  BÁO CÁO TRUY VẤN CHI TIẾT (RETRIEVAL REPORT) - {prefix.upper()}")
	lines.append(f"=======================================================================")

	# 1. Metrics toàn cục
	lines.append("Chỉ số toàn cục (Global Metrics):")
	global_keys = ["mAP", "AUC", "Recall@1", "Recall@5", "Recall@10", "Precision@1", "Precision@5", "Precision@10"]
	for k in global_keys:
		if k in results:
			val = results[k]
			lines.append(f"  {k:<15}: {val*100:.2f}%" if k != "AUC" else f"  {k:<15}: {val:.4f}")

	# 2. Chỉ số phân cụm
	lines.append("\nChỉ số phân cụm không gian nhúng (Clustering Metrics):")
	clustering_keys = ["Silhouette", "Davies-Bouldin", "Calinski-Harabasz", "Dunn-Index", "NMI", "Intra-Inter-Ratio"]
	for k in clustering_keys:
		if k in results:
			val = results[k]
			lines.append(f"  {k:<20}: {val:.4f}")

	# 3. Bảng chi tiết từng class
	lines.append("\nChi tiết cho từng loài gỗ (Class-wise Metrics):")
	header = f"{'Loài gỗ (Class)':<35} | {'Recall@1':<10} | {'Recall@5':<10} | {'mAP':<10} | {'AUC':<10}"
	lines.append(header)
	lines.append("-" * len(header))

	per_class_recall1 = results.get("per_class_recall1", [])
	per_class_recall5 = results.get("per_class_recall5", [])
	per_class_map = results.get("per_class_map", [])
	per_class_auc = results.get("per_class_auc", [])

	for idx, name in enumerate(class_names):
		r1 = per_class_recall1[idx] * 100 if idx < len(per_class_recall1) else 0.0
		r5 = per_class_recall5[idx] * 100 if idx < len(per_class_recall5) else 0.0
		m = per_class_map[idx] * 100 if idx < len(per_class_map) else 0.0
		a = per_class_auc[idx] if idx < len(per_class_auc) else 0.0
		lines.append(f"{name:<35} | {r1:>8.2f}% | {r5:>8.2f}% | {m:>8.2f}% | {a:>10.4f}")

	lines.append("=======================================================================")
	return "\n".join(lines)


def evaluate_cross_retrieval(
	model: nn.Module,
	query_loader: DataLoader,
	gallery_loader: DataLoader,
	device: torch.device,
	class_names: list[str],
	k_values: list[int] | None = None,
) -> dict:
	"""
	Đánh giá truy vấn chéo (Cross-Retrieval): Query là query_loader, Gallery là gallery_loader.
	Tính Recall@K, Precision@K, mAP, AUC toàn cục và per-class.
	"""
	if k_values is None:
		k_values = [1, 5, 10]

	query_embs, query_labels = extract_all_embeddings(model, query_loader, device)
	gallery_embs, gallery_labels = extract_all_embeddings(model, gallery_loader, device)

	n_query = len(query_labels)
	n_gallery = len(gallery_labels)
	n_classes = len(class_names)

	# Tính khoảng cách Euclidean giữa Query và Gallery (n_query x n_gallery)
	dist_matrix = torch.cdist(query_embs, gallery_embs, p=2).numpy()

	recall_at_k = {k: 0.0 for k in k_values}
	precision_at_k = {k: 0.0 for k in k_values}
	aps = []

	sample_recall1 = []
	sample_recall5 = []
	sample_map = []

	for i in range(n_query):
		dists = dist_matrix[i].copy()
		sorted_indices = np.argsort(dists)

		# Nhãn của gallery được sắp xếp theo khoảng cách
		retrieved_labels = gallery_labels[sorted_indices]
		is_relevant = (retrieved_labels == query_labels[i])
		n_positives = int((gallery_labels == query_labels[i]).sum())

		if n_positives == 0:
			sample_recall1.append(0.0)
			sample_recall5.append(0.0)
			sample_map.append(0.0)
			continue

		# Recall@K và Precision@K
		for k in k_values:
			top_k_relevant = is_relevant[:k]
			recall_at_k[k] += float(top_k_relevant.any())
			precision_at_k[k] += float(top_k_relevant.sum()) / k

		sample_recall1.append(float(is_relevant[:1].any()))
		sample_recall5.append(float(is_relevant[:5].any()))

		# Average Precision (AP)
		cumsum = np.cumsum(is_relevant).astype(np.float64)
		precision_curve = cumsum / np.arange(1, n_gallery + 1, dtype=np.float64)
		ap = (precision_curve * is_relevant).sum() / n_positives
		aps.append(ap)
		sample_map.append(ap)

	# Trung bình toàn cục
	n_valid = max(len(aps), 1)
	for k in k_values:
		recall_at_k[k] /= n_valid
		precision_at_k[k] /= n_valid
	mAP = float(np.mean(aps)) if aps else 0.0

	# AUC chéo
	auc_val = 0.0
	try:
		pair_labels = []
		pair_scores = []
		for i in range(n_query):
			class_i = query_labels[i]
			pos_indices = np.where(gallery_labels == class_i)[0]
			neg_indices = np.where(gallery_labels != class_i)[0]
			
			for pos_idx in pos_indices:
				pair_labels.append(1)
				pair_scores.append(-dist_matrix[i][pos_idx])
			for neg_idx in neg_indices:
				pair_labels.append(0)
				pair_scores.append(-dist_matrix[i][neg_idx])
		if len(set(pair_labels)) > 1:
			auc_val = float(roc_auc_score(pair_labels, pair_scores))
	except Exception:
		auc_val = 0.0

	# --- Tính per-class cho truy vấn chéo ---
	per_class_recall1 = []
	per_class_recall5 = []
	per_class_map = []
	per_class_auc = []

	for c in range(n_classes):
		class_query_indices = np.where(query_labels == c)[0]
		if len(class_query_indices) == 0:
			per_class_recall1.append(0.0)
			per_class_recall5.append(0.0)
			per_class_map.append(0.0)
			per_class_auc.append(0.0)
			continue

		c_recall1 = np.mean([sample_recall1[idx] for idx in class_query_indices])
		c_recall5 = np.mean([sample_recall5[idx] for idx in class_query_indices])
		c_map = np.mean([sample_map[idx] for idx in class_query_indices])

		c_auc = 0.0
		try:
			pair_labels_c = []
			pair_scores_c = []
			for idx_i in class_query_indices:
				pos_indices = np.where(gallery_labels == c)[0]
				neg_indices = np.where(gallery_labels != c)[0]
				for pos_idx in pos_indices:
					pair_labels_c.append(1)
					pair_scores_c.append(-dist_matrix[idx_i][pos_idx])
				for neg_idx in neg_indices:
					pair_labels_c.append(0)
					pair_scores_c.append(-dist_matrix[idx_i][neg_idx])
			if len(set(pair_labels_c)) > 1:
				c_auc = float(roc_auc_score(pair_labels_c, pair_scores_c))
		except Exception:
			c_auc = 0.0

		per_class_recall1.append(c_recall1)
		per_class_recall5.append(c_recall5)
		per_class_map.append(c_map)
		per_class_auc.append(c_auc)

	results = {
		"mAP": mAP,
		"AUC": auc_val,
		"per_class_recall1": per_class_recall1,
		"per_class_recall5": per_class_recall5,
		"per_class_map": per_class_map,
		"per_class_auc": per_class_auc
	}
	for k in k_values:
		results[f"Recall@{k}"] = recall_at_k[k]
		results[f"Precision@{k}"] = precision_at_k[k]

	return results


class MetricGradCAM:
	def __init__(self, model: nn.Module, target_layer: nn.Module, method: str = "gradcam") -> None:
		self.model = model
		self.target_layer = target_layer
		self.method = method.lower()
		self.activations = None
		self.forward_handle = target_layer.register_forward_hook(self._forward_hook)

	def _forward_hook(self, module, inputs, output):
		self.activations = output

	def remove(self) -> None:
		self.forward_handle.remove()

	def __call__(self, input_tensor: torch.Tensor, prototype: torch.Tensor, all_prototypes: torch.Tensor = None, target_class_idx: int = None) -> np.ndarray:
		if self.method == "eigencam":
			self.model.eval()
			with torch.no_grad():
				_ = self.model(input_tensor)
			act = self.activations.squeeze(0).detach().cpu().numpy()
			c, h, w = act.shape
			A = act.reshape(c, h * w).T
			A = A - np.mean(A, axis=0)
			U, S, Vt = np.linalg.svd(A, full_matrices=False)
			projection = (A @ Vt[0, :]).reshape(h, w)
			if np.sum(projection) < 0:
				projection = -projection
			cam = np.maximum(projection, 0)
		else:
			self.model.zero_grad()
			if self.method == "finercam" and all_prototypes is not None and target_class_idx is not None:
				emb = self.model(input_tensor)
				logits = torch.matmul(emb, all_prototypes.to(emb.device).T)
				main_category = target_class_idx
				prob = torch.softmax(logits, dim=-1)
				output_data = logits[0].detach().cpu().numpy()
				target_logit = output_data[main_category]
				
				sorted_indices = np.argsort(np.abs(output_data - target_logit))
				comparison_categories = sorted_indices[1:4]  # 3 lớp gần nhất tiếp theo
				alpha = 1.0
				
				wn = logits[0, main_category]
				weights = [prob[0, idx] for idx in comparison_categories]
				numerator = sum(w * (wn - alpha * logits[0, idx]) for w, idx in zip(weights, comparison_categories))
				denominator = sum(weights)
				score = numerator / (denominator + 1e-9)
			else:
				emb = self.model(input_tensor)
				score = (emb * prototype.unsqueeze(0)).sum()

			if self.activations is None:
				raise RuntimeError("CAM hook did not capture activations")

			grads = torch.autograd.grad(score, self.activations, retain_graph=True)[0]
			
			if self.method == "gradcam++":
				grads_pos = torch.clamp(grads, min=0)
				grads_power_2 = grads_pos ** 2
				grads_power_3 = grads_pos ** 3
				sum_activations = torch.sum(self.activations, dim=(2, 3), keepdim=True)
				eps = 1e-7
				aij = grads_power_2 / (2 * grads_power_2 + sum_activations * grads_power_3 + eps)
				weights = torch.sum(aij * grads_pos, dim=(2, 3), keepdim=True)
				cam = torch.sum(weights * self.activations, dim=1)
				cam = F.relu(cam)
				cam = cam.squeeze().detach().cpu().numpy()
			elif self.method == "xgradcam":
				sum_activations = torch.sum(self.activations, dim=(2, 3), keepdim=True) + 1e-7
				weights = torch.sum(grads * self.activations / sum_activations, dim=(2, 3), keepdim=True)
				cam = torch.sum(weights * self.activations, dim=1)
				cam = F.relu(cam)
				cam = cam.squeeze().detach().cpu().numpy()
			elif self.method == "hirescam":
				cam = torch.sum(grads * self.activations, dim=1)
				cam = F.relu(cam)
				cam = cam.squeeze().detach().cpu().numpy()
			elif self.method == "layercam":
				cam = torch.sum(torch.clamp(grads, min=0) * self.activations, dim=1)
				cam = F.relu(cam)
				cam = cam.squeeze().detach().cpu().numpy()
			elif self.method == "eigengradcam":
				weighted_act = grads * self.activations
				act = weighted_act.squeeze(0).detach().cpu().numpy()
				c, h, w = act.shape
				A = act.reshape(c, h * w).T
				A = A - np.mean(A, axis=0)
				U, S, Vt = np.linalg.svd(A, full_matrices=False)
				projection = (A @ Vt[0, :]).reshape(h, w)
				if np.sum(projection) < 0:
					projection = -projection
				cam = np.maximum(projection, 0)
			else:  # gradcam and finercam use gradcam aggregation
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
	device: torch.device,
	method: str = "gradcam"
) -> list[np.ndarray]:
	target_layer = find_last_conv_layer(model)
	if target_layer is None:
		print("Warning: Không tìm thấy Conv2d layer trong model để vẽ Grad-CAM")
		return [np.zeros((224, 224)) for _ in representatives]
	gradcam = MetricGradCAM(model, target_layer, method=method)
	model.eval()
	cam_maps = []
	for rep in representatives:
		with Image.open(rep['path']) as img:
			img = img.convert("RGB")
		input_tensor = transform(img).unsqueeze(0).to(device)
		class_idx = class_to_idx[rep['label']]
		proto = prototypes[class_idx].to(device)
		cam = gradcam(input_tensor, proto, all_prototypes=prototypes, target_class_idx=class_idx)
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

	# Tính toán giá trị trung bình và tỷ lệ
	intra_mean_bef = np.mean(intra_before)
	inter_mean_bef = np.mean(inter_before)
	ratio_bef = intra_mean_bef / inter_mean_bef if inter_mean_bef > 0 else 0.0

	intra_mean_aft = np.mean(intra_after)
	inter_mean_aft = np.mean(inter_after)
	ratio_aft = intra_mean_aft / inter_mean_aft if inter_mean_aft > 0 else 0.0

	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

	ax1.hist(intra_before, bins=30, alpha=0.6, density=True, label="Intra-class (Same Class)", color="#3498db")
	ax1.hist(inter_before, bins=30, alpha=0.6, density=True, label="Inter-class (Diff Class)", color="#e74c3c")
	ax1.set_xlabel("Euclidean Distance")
	ax1.set_ylabel("Density")
	ax1.set_title("Distance Distribution Before Training", fontsize=11, fontweight='bold')
	ax1.legend()
	ax1.grid(alpha=0.3)
	
	# In thông số lên ax1
	text_bef = f"Intra Mean: {intra_mean_bef:.4f}\nInter Mean: {inter_mean_bef:.4f}\nRatio: {ratio_bef:.4f}"
	ax1.text(0.05, 0.72, text_bef, transform=ax1.transAxes, fontsize=9,
	         bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

	ax2.hist(intra_after, bins=30, alpha=0.6, density=True, label="Intra-class (Same Class)", color="#2ecc71")
	ax2.hist(inter_after, bins=30, alpha=0.6, density=True, label="Inter-class (Diff Class)", color="#e74c3c")
	ax2.set_xlabel("Euclidean Distance")
	ax2.set_ylabel("Density")
	ax2.set_title("Distance Distribution After Training", fontsize=11, fontweight='bold')
	ax2.legend()
	ax2.grid(alpha=0.3)
	
	# In thông số lên ax2
	text_aft = f"Intra Mean: {intra_mean_aft:.4f}\nInter Mean: {inter_mean_aft:.4f}\nRatio: {ratio_aft:.4f}"
	ax2.text(0.05, 0.72, text_aft, transform=ax2.transAxes, fontsize=9,
	         bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

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
# Per-Epoch Metric Plots
# ============================================================

def plot_all_metrics_per_epoch(history: dict, output_dir: Path) -> None:
	"""Vẽ từng metric theo từng epoch — mỗi metric lưu thành một ảnh riêng.
	Các retrieval metrics có val-cross sẽ được tách thành 2 biểu đồ:
	  *_train_val.png  — Train vs Val
	  *_train_cross.png — Train vs Val-Cross
	"""
	print("\n[Plot] Vẽ biểu đồ từng metric theo epoch...")
	epochs_range = range(1, len(history["train_loss"]) + 1)

	def _save_fig(ax, title: str, ylabel: str, filename: str) -> None:
		ax.set_title(title, fontsize=13, fontweight="bold")
		ax.set_xlabel("Epoch")
		ax.set_ylabel(ylabel)
		ax.legend()
		ax.grid(alpha=0.3)
		plt.tight_layout()
		plt.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
		plt.close()
		print(f"  Saved → {filename}")

	def _plot_pair(
		train_key: str, val_key: str, cross_key: str | None,
		label: str, ylabel: str,
		fname_tv: str, fname_tc: str,
		scale: float = 1.0,
	) -> None:
		"""Vẽ 2 biểu đồ riêng: train+val và train+val-cross."""
		tv = [v * scale for v in history[train_key]]
		vv = [v * scale for v in history[val_key]]

		# Biểu đồ 1: Train vs Val
		_, ax = plt.subplots(figsize=(8, 5))
		ax.plot(epochs_range, tv, color="#3498db", lw=2, label="Train")
		ax.plot(epochs_range, vv, color="#2ecc71", lw=2, label="Val")
		_save_fig(ax, f"{label} — Train vs Val", ylabel, fname_tv)

		# Biểu đồ 2: Train vs Val-Cross (chỉ khi có dữ liệu)
		if cross_key and history.get(cross_key):
			vc = [v * scale for v in history[cross_key]]
			_, ax = plt.subplots(figsize=(8, 5))
			ax.plot(epochs_range, tv,   color="#3498db", lw=2, label="Train")
			ax.plot(epochs_range, vc,   color="#e67e22", lw=2, linestyle="--", label="Val-Cross")
			_save_fig(ax, f"{label} — Train vs Val-Cross", ylabel, fname_tc)

	# 1. Loss (không có cross)
	_, ax = plt.subplots(figsize=(8, 5))
	ax.plot(epochs_range, history["train_loss"], color="#3498db", lw=2, label="Train")
	ax.plot(epochs_range, history["val_loss"],   color="#2ecc71", lw=2, label="Val")
	_save_fig(ax, "Loss — Train vs Val", "Loss", "metric_loss.png")

	# 2. Recall@1
	_plot_pair("train_recall1", "val_recall1", "val_cross_recall1",
		"Recall@1", "Recall@1 (%)", "metric_recall1_train_val.png", "metric_recall1_train_cross.png", scale=100)

	# 3. Recall@5
	_plot_pair("train_recall5", "val_recall5", "val_cross_recall5",
		"Recall@5", "Recall@5 (%)", "metric_recall5_train_val.png", "metric_recall5_train_cross.png", scale=100)

	# 4. Precision@1
	_plot_pair("train_precision1", "val_precision1", "val_cross_precision1",
		"Precision@1", "Precision@1 (%)", "metric_precision1_train_val.png", "metric_precision1_train_cross.png", scale=100)

	# 5. Precision@5
	_plot_pair("train_precision5", "val_precision5", "val_cross_precision5",
		"Precision@5", "Precision@5 (%)", "metric_precision5_train_val.png", "metric_precision5_train_cross.png", scale=100)

	# 6. mAP
	_plot_pair("train_map", "val_map", "val_cross_map",
		"mAP", "mAP (%)", "metric_map_train_val.png", "metric_map_train_cross.png", scale=100)

	# 7. AUC
	_plot_pair("train_auc", "val_auc", "val_cross_auc",
		"AUC", "AUC", "metric_auc_train_val.png", "metric_auc_train_cross.png", scale=1)

	# 8–13. Clustering Metrics (Val only — 1 biểu đồ mỗi loại)
	clustering_cfgs = [
		("val_silhouette", "Silhouette Score (Val)",          "Silhouette Score",     "metric_silhouette.png"),
		("val_dbi",        "Davies-Bouldin Index (Val)",       "Davies-Bouldin Index", "metric_dbi.png"),
		("val_chi",        "Calinski-Harabasz Score (Val)",    "CHI Score",            "metric_chi.png"),
		("val_dunn",       "Dunn Index (Val)",                 "Dunn Index",           "metric_dunn.png"),
		("val_nmi",        "NMI (Val)",                        "NMI",                  "metric_nmi.png"),
		("val_ratio",      "Intra/Inter Ratio (Val)",          "Intra/Inter Ratio",    "metric_intra_inter_ratio.png"),
	]
	for key, title, ylabel, filename in clustering_cfgs:
		if not history.get(key):
			continue
		_, ax = plt.subplots(figsize=(8, 5))
		ax.plot(epochs_range, history[key], color="#9b59b6", lw=2, label="Val")
		_save_fig(ax, f"{title} over Epochs", ylabel, filename)

	n_charts = 1 + 6 * 2 + len(clustering_cfgs)
	print(f"[plot_all_metrics_per_epoch] Hoàn tất — {n_charts} biểu đồ đã lưu vào {output_dir}/")


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
	print("Trích xuất embeddings với EfficientNetV2-M...")
	embs_eff = compute_embeddings_v2(df_filtered, "tf_efficientnetv2_m_in21k", batch_size=EMB_BATCH_SIZE, device=device)
	print("Trích xuất embeddings với Swin-Large...")
	embs_swin = compute_embeddings_v2(df_filtered, "swin_large_patch4_window7_224", batch_size=EMB_BATCH_SIZE, device=device)

	# ── 3. Chia dữ liệu End Version ──
	print("\n[Step 3] Chia dữ liệu theo End Version Split...")
	df_train, df_val, df_test = end_version_split(
		df_filtered, embs_eff, embs_swin,
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
	del embs_eff, embs_swin
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
	
	# Tính prototype & CAM trước training cho tất cả các phương pháp
	before_protos = compute_class_prototypes(model, train_eval_loader, device, len(class_names))
	before_cams_dict = {}
	for method in CAM_METHODS:
		before_cams_dict[method] = generate_gradcam_maps(
			model, reps_flat, before_protos, class_to_idx, eval_tf, device, method=method
		)
	
	# Tính embeddings cho tập Test trước training (cho t-SNE & distance analysis)
	print("  Trích xuất đặc trưng của tập Test trước training...")
	before_test_embs, test_labels = extract_all_embeddings(model, test_loader, device)
	before_test_embs = before_test_embs.numpy()

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
		"train_auc": [], "val_auc": [],
		"val_cross_recall1": [], "val_cross_recall5": [],
		"val_cross_precision1": [], "val_cross_precision5": [],
		"val_cross_map": [], "val_cross_auc": [],
		"val_silhouette": [], "val_dbi": [], "val_chi": [], "val_dunn": [], "val_nmi": [], "val_ratio": []
	}
	best_map = 0.0
	epochs_no_improve = 0

	for epoch in range(1, EPOCHS + 1):
		# Train
		loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, EPOCHS)
		
		# Val loss
		val_loss = evaluate_loss(model, val_loader, criterion, device)
		
		# Evaluate
		train_results = evaluate_retrieval(model, train_eval_loader, device, class_names, k_values=[1, 5])
		val_results = evaluate_retrieval(model, val_loader, device, class_names, k_values=[1, 5])
		val_cross_results = evaluate_cross_retrieval(model, val_loader, train_eval_loader, device, class_names, k_values=[1, 5])
		
		val_map = val_results["mAP"]
		
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

		# Lưu thông tin chéo Val vs Train
		history["val_cross_recall1"].append(val_cross_results["Recall@1"])
		history["val_cross_recall5"].append(val_cross_results["Recall@5"])
		history["val_cross_precision1"].append(val_cross_results["Precision@1"])
		history["val_cross_precision5"].append(val_cross_results["Precision@5"])
		history["val_cross_map"].append(val_cross_results["mAP"])
		history["val_cross_auc"].append(val_cross_results["AUC"])

		# Lưu chỉ số phân cụm tập Validation
		history["val_silhouette"].append(val_results["Silhouette"])
		history["val_dbi"].append(val_results["Davies-Bouldin"])
		history["val_chi"].append(val_results["Calinski-Harabasz"])
		history["val_dunn"].append(val_results["Dunn-Index"])
		history["val_nmi"].append(val_results["NMI"])
		history["val_ratio"].append(val_results["Intra-Inter-Ratio"])

		current_lr = optimizer.param_groups[0]["lr"]
		print(
			f"Epoch {epoch}/{EPOCHS} —\n"
			f"  Loss (Train/Val): {loss:.4f} / {val_loss:.4f}\n"
			f"  Recall@1    (Train/Val/Val-Cross): {train_results['Recall@1']*100:.2f}% / {val_results['Recall@1']*100:.2f}% / {val_cross_results['Recall@1']*100:.2f}%\n"
			f"  Recall@5    (Train/Val/Val-Cross): {train_results['Recall@5']*100:.2f}% / {val_results['Recall@5']*100:.2f}% / {val_cross_results['Recall@5']*100:.2f}%\n"
			f"  Precision@1 (Train/Val/Val-Cross): {train_results['Precision@1']*100:.2f}% / {val_results['Precision@1']*100:.2f}% / {val_cross_results['Precision@1']*100:.2f}%\n"
			f"  Precision@5 (Train/Val/Val-Cross): {train_results['Precision@5']*100:.2f}% / {val_results['Precision@5']*100:.2f}% / {val_cross_results['Precision@5']*100:.2f}%\n"
			f"  mAP         (Train/Val/Val-Cross): {train_results['mAP']*100:.2f}% / {val_results['mAP']*100:.2f}% / {val_cross_results['mAP']*100:.2f}%\n"
			f"  AUC         (Train/Val/Val-Cross): {train_results['AUC']:.4f} / {val_results['AUC']:.4f} / {val_cross_results['AUC']:.4f}\n"
			f"  Clustering Metrics (Val): Silhouette: {val_results['Silhouette']:.4f} | DBI: {val_results['Davies-Bouldin']:.4f} | CHI: {val_results['Calinski-Harabasz']:.2f} | Dunn: {val_results['Dunn-Index']:.4f} | NMI: {val_results['NMI']:.4f} | Ratio: {val_results['Intra-Inter-Ratio']:.4f}\n"
			f"  LR: {current_lr:.6f}"
		)

		scheduler.step()

		# Checkpoint best model
		if val_map > best_map:
			best_map = val_map
			epochs_no_improve = 0
			raw_model = model
			torch.save(raw_model.state_dict(), output_dir / "best_model.pth")
			print(f"  → Saved best model (mAP={best_map*100:.2f}%)")
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
	val_report = format_retrieval_report(val_results, class_names, prefix="Val")
	print(val_report)
	print(
		f"  [Clustering Metrics - Val]\n"
		f"    Silhouette Score : {val_results['Silhouette']:.4f}\n"
		f"    Davies-Bouldin   : {val_results['Davies-Bouldin']:.4f}\n"
		f"    Calinski-Harabasz: {val_results['Calinski-Harabasz']:.2f}\n"
		f"    Dunn Index       : {val_results['Dunn-Index']:.4f}\n"
		f"    NMI              : {val_results['NMI']:.4f}\n"
		f"    Intra/Inter Ratio: {val_results['Intra-Inter-Ratio']:.4f}"
	)
	with open(output_dir / "retrieval_report_val.txt", "w", encoding="utf-8") as f:
		f.write(val_report)

	# Đánh giá Test
	print("\n[Đánh giá cuối - Test]")
	test_results = evaluate_retrieval(model, test_loader, device, class_names)
	test_report = format_retrieval_report(test_results, class_names, prefix="Test")
	print(test_report)
	print(
		f"  [Clustering Metrics - Test]\n"
		f"    Silhouette Score : {test_results['Silhouette']:.4f}\n"
		f"    Davies-Bouldin   : {test_results['Davies-Bouldin']:.4f}\n"
		f"    Calinski-Harabasz: {test_results['Calinski-Harabasz']:.2f}\n"
		f"    Dunn Index       : {test_results['Dunn-Index']:.4f}\n"
		f"    NMI              : {test_results['NMI']:.4f}\n"
		f"    Intra/Inter Ratio: {test_results['Intra-Inter-Ratio']:.4f}"
	)
	with open(output_dir / "retrieval_report_test.txt", "w", encoding="utf-8") as f:
		f.write(test_report)

	# Đánh giá truy vấn chéo (Cross-Retrieval): Val/Test làm Query, Train làm Gallery
	print("\n[Đánh giá chéo - Validation Query vs Training Gallery]")
	val_cross_results = evaluate_cross_retrieval(model, val_loader, train_eval_loader, device, class_names)
	val_cross_report = format_retrieval_report(val_cross_results, class_names, prefix="Val Query vs Train Gallery")
	print(val_cross_report)
	print(
		f"  [Cross-Retrieval Summary - Val→Train]\n"
		f"    mAP   : {val_cross_results['mAP']*100:.2f}%\n"
		f"    AUC   : {val_cross_results['AUC']:.4f}\n"
		f"    R@1   : {val_cross_results['Recall@1']*100:.2f}%\n"
		f"    R@5   : {val_cross_results['Recall@5']*100:.2f}%"
	)
	with open(output_dir / "retrieval_report_val_query_train_gallery.txt", "w", encoding="utf-8") as f:
		f.write(val_cross_report)

	print("\n[Đánh giá chéo - Test Query vs Training Gallery]")
	test_cross_results = evaluate_cross_retrieval(model, test_loader, train_eval_loader, device, class_names)
	test_cross_report = format_retrieval_report(test_cross_results, class_names, prefix="Test Query vs Train Gallery")
	print(test_cross_report)
	print(
		f"  [Cross-Retrieval Summary - Test→Train]\n"
		f"    mAP   : {test_cross_results['mAP']*100:.2f}%\n"
		f"    AUC   : {test_cross_results['AUC']:.4f}\n"
		f"    R@1   : {test_cross_results['Recall@1']*100:.2f}%\n"
		f"    R@5   : {test_cross_results['Recall@5']*100:.2f}%"
	)
	with open(output_dir / "retrieval_report_test_query_train_gallery.txt", "w", encoding="utf-8") as f:
		f.write(test_cross_report)

	# ── 11. Trích xuất trạng thái sau training (Post-training analysis) ──
	print("\n[Analysis] Trích xuất đặc trưng và sinh Grad-CAM sau training...")
	after_protos = compute_class_prototypes(model, train_eval_loader, device, len(class_names))
	
	n_dal = len(reps_dict['Dalbergia'])
	for method in CAM_METHODS:
		after_cams = generate_gradcam_maps(
			model, reps_flat, after_protos, class_to_idx, eval_tf, device, method=method
		)
		before_cams = before_cams_dict[method]
		plot_gradcam_comparison(
			reps_dict['Dalbergia'], before_cams[:n_dal], after_cams[:n_dal], 
			f"Dalbergia ({method})", output_dir / f"gradcam_dalbergia_{method}.png"
		)
		plot_gradcam_comparison(
			reps_dict['Pterocarpus'], before_cams[n_dal:], after_cams[n_dal:], 
			f"Pterocarpus ({method})", output_dir / f"gradcam_pterocarpus_{method}.png"
		)
	
	# Tính embeddings sau training cho tập Test
	print("  Trích xuất đặc trưng của tập Test sau training...")
	after_test_embs, _ = extract_all_embeddings(model, test_loader, device)
	after_test_embs = after_test_embs.numpy()
	
	# Vẽ t-SNE so sánh trên tập Test
	plot_tsne_comparison(before_test_embs, after_test_embs, test_labels, class_names, output_dir / "tsne_comparison.png")
	
	# Phân tích khoảng cách trên tập Test
	plot_distance_analysis(before_test_embs, after_test_embs, test_labels, output_dir / "distance_distribution.png")

	# Vẽ biểu đồ tổng hợp metrics
	plot_metrics_summary(history, model, val_loader, test_loader, device, output_dir)

	# Vẽ từng metric riêng biệt theo epoch
	plot_all_metrics_per_epoch(history, output_dir)

	# Lọc các metrics kiểu số để lưu vào summary.json
	val_summary = {}
	for k, v in val_results.items():
		if isinstance(v, (int, float, np.integer, np.floating)):
			val_summary[k] = round(float(v), 6)

	test_summary = {}
	for k, v in test_results.items():
		if isinstance(v, (int, float, np.integer, np.floating)):
			test_summary[k] = round(float(v), 6)

	val_cross_summary = {}
	for k, v in val_cross_results.items():
		if isinstance(v, (int, float, np.integer, np.floating)):
			val_cross_summary[k] = round(float(v), 6)

	test_cross_summary = {}
	for k, v in test_cross_results.items():
		if isinstance(v, (int, float, np.integer, np.floating)):
			test_cross_summary[k] = round(float(v), 6)

	summary = {
		"method": "Contrastive Loss (Online Hard Pair Mining)",
		"model": MODEL_NAME,
		"embedding_dim": EMBEDDING_DIM,
		"margin": MARGIN,
		"p_classes": P_CLASSES,
		"k_samples": K_SAMPLES,
		"batch_size": P_CLASSES * K_SAMPLES,
		"epochs_trained": len(history["train_loss"]),
		"best_val_map": best_map,
		"val_results": val_summary,
		"test_results": test_summary,
		"val_cross_results": val_cross_summary,
		"test_cross_results": test_cross_summary,
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
