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
from sklearn.metrics import roc_auc_score

# ===== CẤU HÌNH =====
ROOT_DIR = r"/kaggle/input/datasets/b23dckh002lvitanh/s3-origin/S3"
OUTPUT_DIR = "outputs_contrastive"
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
SEED = 42
P_CLASSES = 12          # Số class mỗi batch
K_SAMPLES = 8           # Số ảnh mỗi class trong batch
EPOCHS = 200
PATIENCE = 20           # EarlyStopping patience
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


def plot_training_curves(history: dict, output_dir: Path) -> None:
	"""Vẽ biểu đồ Loss và Recall@1 theo epoch."""
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
	epochs_range = range(1, len(history["train_loss"]) + 1)

	ax1.plot(epochs_range, history["train_loss"], label="Train Loss", color="#e74c3c", linewidth=2)
	ax1.set_xlabel("Epoch")
	ax1.set_ylabel("Loss")
	ax1.set_title("Contrastive Loss")
	ax1.legend()
	ax1.grid(alpha=0.3)

	ax2.plot(epochs_range, history["val_recall1"], label="Val Recall@1", color="#2ecc71", linewidth=2)
	ax2.set_xlabel("Epoch")
	ax2.set_ylabel("Recall@1")
	ax2.set_title("Validation Recall@1")
	ax2.legend()
	ax2.grid(alpha=0.3)

	plt.tight_layout()
	plt.savefig(output_dir / "training_curves.png", dpi=200, bbox_inches="tight")
	plt.close()
	print(f"Training curves saved → {output_dir / 'training_curves.png'}")


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
	if device.type == "cuda" and torch.cuda.device_count() > 1:
		print(f"Phát hiện {torch.cuda.device_count()} GPUs → nn.DataParallel")
		model = nn.DataParallel(model)

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

	# ── 8. Training loop ──
	print(f"\n{'=' * 60}")
	print(f"  Bắt đầu huấn luyện Contrastive Loss — {EPOCHS} epochs")
	print(f"  Margin={MARGIN}, LR={LR}, EarlyStopping patience={PATIENCE}")
	print(f"{'=' * 60}\n")

	history: dict[str, list] = {"train_loss": [], "val_recall1": []}
	best_recall1 = 0.0
	epochs_no_improve = 0

	for epoch in range(1, EPOCHS + 1):
		# Train
		loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, EPOCHS)
		history["train_loss"].append(loss)

		# Evaluate trên Val
		val_results = evaluate_retrieval(model, val_loader, device, class_names, k_values=[1, 5, 10])
		val_recall1 = val_results["Recall@1"]
		history["val_recall1"].append(val_recall1)

		current_lr = optimizer.param_groups[0]["lr"]
		print(
			f"Epoch {epoch}/{EPOCHS} — "
			f"loss={loss:.4f}, val_R@1={val_recall1:.4f}, "
			f"val_mAP={val_results['mAP']:.4f}, lr={current_lr:.6f}"
		)

		scheduler.step()

		# Checkpoint best model
		if val_recall1 > best_recall1:
			best_recall1 = val_recall1
			epochs_no_improve = 0
			raw_model = model.module if isinstance(model, nn.DataParallel) else model
			torch.save(raw_model.state_dict(), output_dir / "best_model.pth")
			print(f"  → Saved best model (Recall@1={best_recall1:.4f})")
		else:
			epochs_no_improve += 1

		if epochs_no_improve >= PATIENCE:
			print(f"\nEarly stopping tại epoch {epoch} (patience={PATIENCE})")
			break

	# ── 9. Load best & đánh giá cuối ──
	raw_model = model.module if isinstance(model, nn.DataParallel) else model
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

	# ── 10. Vẽ biểu đồ & lưu summary ──
	plot_training_curves(history, output_dir)

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
	del train_loader, val_loader, test_loader
	del train_ds, val_ds, test_ds
	gc.collect()
	if device.type == "cuda":
		torch.cuda.empty_cache()

	print(f"\n[Hoàn tất] Tất cả kết quả lưu tại: {output_dir}/")


if __name__ == "__main__":
	main()
