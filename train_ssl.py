"""
train_ssl.py
============
Tệp huấn luyện tự giám sát (Self-Supervised Learning) trên bộ dữ liệu ảnh gỗ S3.
Áp dụng khung kiến trúc Barlow Twins (ICML 2021) để học biểu diễn đặc trưng không nhãn.
Sử dụng ConvNeXt-Tiny làm backbone cùng 3 lớp MLP Projection Head.
Tự động đánh giá chất lượng đặc trưng bằng chỉ số Retrieval mAP (KNN) chéo mỗi epoch.

Cách chạy:
python train_ssl.py
"""

import os
import json
import random
from pathlib import Path
from PIL import Image

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from timm.data import resolve_data_config

# Import utilities từ utils
from utils import (
	set_seed,
	get_device,
	collect_image_samples,
	build_dataframe,
	compute_embeddings_v2,
	freeze_model_layers,
	summarize_model,
	log_split_summary,
	eda_split_class_distribution,
	evaluate_cross_retrieval,
	evaluate_retrieval,
	format_retrieval_report,
	build_transforms,
)
from train_final import end_version_split
from split_methods import validate_split

# ===== CẤU HÌNH =====
ROOT_DIR = r"/kaggle/input/datasets/b23dckh002lvitanh/s3-origin/S3"
OUTPUT_DIR = "outputs_ssl"
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
SEED = 42
BATCH_SIZE = 128
EPOCHS = 30
PATIENCE = 10
LR = 5e-4
WEIGHT_DECAY = 1e-4
PROJECTION_DIM = 2048
LAMBD = 0.0051  # Hệ số phạt dư thừa chéo (off-diagonal term)
FREEZE_RATIO = 0.90
MODEL_NAME = "convnext_tiny"
NUM_WORKERS = 8
CALCULATE_CLUSTERING_METRICS = True
# =====================

class DoubleViewTransform:
	"""Tạo hai góc nhìn biến đổi ngẫu nhiên khác nhau từ cùng một ảnh."""
	def __init__(self, transform1, transform2):
		self.transform1 = transform1
		self.transform2 = transform2

	def __call__(self, img):
		return self.transform1(img), self.transform2(img)

class SSLDataset(Dataset):
	"""Dataset phục vụ huấn luyện tự giám sát (trả về 2 views)."""
	def __init__(self, df: pd.DataFrame, transform=None) -> None:
		self.df = df.reset_index(drop=True)
		self.transform = transform

	def __len__(self) -> int:
		return len(self.df)

	def __getitem__(self, idx: int):
		row = self.df.iloc[idx]
		with Image.open(row["path"]) as img:
			img = img.convert("RGB")
		if self.transform:
			img1, img2 = self.transform(img)
		return img1, img2

class SSLEvalDataset(Dataset):
	"""Dataset phục vụ đánh giá đặc trưng (trả về 1 ảnh và label_idx)."""
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

# ============================================================
# Mô Hình Barlow Twins
# ============================================================

class BarlowTwins(nn.Module):
	"""Kiến trúc Barlow Twins: Backbone + MLP Projector Head."""
	def __init__(self, backbone_name: str = "convnext_tiny", projection_dim: int = 2048, freeze_ratio: float = 0.90) -> None:
		super().__init__()
		self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)
		freeze_model_layers(self.backbone, freeze_ratio)

		backbone_dim = self.backbone.num_features  # 768 với convnext_tiny

		# 3-layer Projection Head MLP
		self.projector = nn.Sequential(
			nn.Linear(backbone_dim, projection_dim),
			nn.BatchNorm1d(projection_dim),
			nn.ReLU(inplace=True),
			nn.Linear(projection_dim, projection_dim),
			nn.BatchNorm1d(projection_dim),
			nn.ReLU(inplace=True),
			nn.Linear(projection_dim, projection_dim),
		)
		self.model_name = f"{backbone_name}_barlow_twins"

	def forward(self, x1: torch.Tensor, x2: torch.Tensor = None) -> torch.Tensor:
		if x2 is None:
			# Chế độ đánh giá (trích xuất đặc trưng L2-normalized)
			features = self.backbone(x1)
			return F.normalize(features, p=2, dim=1)

		# Chế độ huấn luyện (dùng projection head)
		z1 = self.projector(self.backbone(x1))
		z2 = self.projector(self.backbone(x2))
		return z1, z2

# ============================================================
# Barlow Twins Loss
# ============================================================

class BarlowTwinsLoss(nn.Module):
	"""Hàm Loss của Barlow Twins tối ưu hóa ma trận tương quan chéo."""
	def __init__(self, lambd: float = 0.0051) -> None:
		super().__init__()
		self.lambd = lambd

	def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
		# Batch normalization cho đặc trưng dọc theo chiều batch
		z1_norm = (z1 - z1.mean(dim=0)) / (z1.std(dim=0) + 1e-9)
		z2_norm = (z2 - z2.mean(dim=0)) / (z2.std(dim=0) + 1e-9)

		batch_size = z1.size(0)
		# Tích vô hướng tính ma trận tương quan chéo C (D x D)
		c = torch.mm(z1_norm.t(), z2_norm) / batch_size

		# Ép các phần tử chéo chính tiến về 1
		on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()

		# Ép các phần tử chéo phụ (off-diagonal) về 0 để triệt tiêu dư thừa đặc trưng
		diag_mask = torch.eye(c.size(0), device=c.device).bool()
		off_diag = c[~diag_mask].pow_(2).sum()

		loss = on_diag + self.lambd * off_diag
		return loss

# ============================================================
# Vòng Lặp Huấn Luyện Từng Epoch
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
	model.train()
	total_loss = 0.0
	n_batches = 0

	pbar = tqdm(loader, desc=f"SSL Train {epoch}/{total_epochs}")
	for x1, x2 in pbar:
		x1 = x1.to(device, non_blocking=True)
		x2 = x2.to(device, non_blocking=True)

		optimizer.zero_grad()
		z1, z2 = model(x1, x2)
		loss = criterion(z1, z2)
		loss.backward()
		optimizer.step()

		total_loss += loss.item()
		n_batches += 1
		pbar.set_postfix(loss=f"{total_loss / n_batches:.4f}")

	return total_loss / max(n_batches, 1)

# ============================================================
# Main
# ============================================================

def main() -> None:
	set_seed(SEED)
	device = get_device()
	print(f"Device: {device}")

	output_dir = Path(OUTPUT_DIR)
	output_dir.mkdir(parents=True, exist_ok=True)

	# 1. Thu thập ảnh
	print("\n[Step 1] Thu thập ảnh...")
	samples = collect_image_samples(ROOT_DIR)
	if not samples:
		raise ValueError(f"Không tìm thấy ảnh nào trong {ROOT_DIR}")

	df = build_dataframe(samples)
	df_filtered = df[df["label"] != "Pterocarpus sp"].reset_index(drop=True)
	print(f"Tổng ảnh: {len(df_filtered)}, Số class: {df_filtered['label'].nunique()}")

	class_names = sorted(df_filtered["label"].unique().tolist())
	class_to_idx = {name: i for i, name in enumerate(class_names)}

	# 2. Compute embeddings cho chia dữ liệu
	print("\n[Step 2] Compute embeddings cho chia dữ liệu...")
	print("Trích xuất embeddings với EfficientNetV2-M...")
	embs_eff = compute_embeddings_v2(df_filtered, "tf_efficientnetv2_m_in21k", batch_size=BATCH_SIZE, device=device)
	print("Trích xuất embeddings với Swin-Large...")
	embs_swin = compute_embeddings_v2(df_filtered, "swin_large_patch4_window7_224", batch_size=BATCH_SIZE, device=device)

	# 3. Chia dữ liệu theo PP chuẩn cuối (End Version Split)
	print("\n[Step 3] Chia dữ liệu theo End Version Split...")
	df_train, df_val, df_test = end_version_split(
		df_filtered, embs_eff, embs_swin,
		train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, seed=SEED,
	)

	validate_split(df_filtered, df_train, df_val, df_test, "SSL_EndVersion")
	log_split_summary(df_filtered, df_train, df_val, df_test)

	# 4. Thiết lập Transforms & Dataloaders
	print("\n[Step 4] Chuẩn bị Dataset và DataLoader...")
	cfg_model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=0)
	cfg = resolve_data_config({}, model=cfg_model)
	img_size = cfg.get("input_size", (3, 224, 224))[-1]
	mean = cfg.get("mean", (0.485, 0.456, 0.406))
	std = cfg.get("std", (0.229, 0.224, 0.225))
	
	# Thiết lập Augment View kép đặc thù cho SSL
	train_tf1 = transforms.Compose([
		transforms.RandomResizedCrop(img_size, scale=(0.3, 1.0)),
		transforms.RandomHorizontalFlip(),
		transforms.RandomVerticalFlip(),
		transforms.RandomRotation(30),
		transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
		transforms.RandomGrayscale(p=0.2),
		transforms.ToTensor(),
		transforms.Normalize(mean, std)
	])
	
	train_tf2 = transforms.Compose([
		transforms.RandomResizedCrop(img_size, scale=(0.3, 1.0)),
		transforms.RandomHorizontalFlip(),
		transforms.RandomVerticalFlip(),
		transforms.RandomRotation(30),
		transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
		transforms.RandomGrayscale(p=0.2),
		transforms.ToTensor(),
		transforms.Normalize(mean, std)
	])
	
	double_transform = DoubleViewTransform(train_tf1, train_tf2)
	_, eval_tf = build_transforms(img_size, mean, std)

	# Dataset huấn luyện SSL (Không dùng nhãn)
	train_ds = SSLDataset(df_train, transform=double_transform)
	
	# Datasets phục vụ đánh giá (Cần nhãn)
	train_eval_ds = SSLEvalDataset(df_train, class_to_idx, transform=eval_tf)
	val_ds = SSLEvalDataset(df_val, class_to_idx, transform=eval_tf)
	test_ds = SSLEvalDataset(df_test, class_to_idx, transform=eval_tf)

	# DataLoaders song song hóa với NUM_WORKERS = 8
	train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
	
	train_eval_loader = DataLoader(train_eval_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
	val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
	test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

	print(f"SSL Train Batch size: {BATCH_SIZE} | Workers: {NUM_WORKERS}")

	# 5. Khởi tạo Model, Loss, Optimizer
	print("\n[Step 5] Khởi tạo model và optimizer...")
	model = BarlowTwins(backbone_name=MODEL_NAME, projection_dim=PROJECTION_DIM, freeze_ratio=FREEZE_RATIO).to(device)

	model_info = summarize_model(model)
	print(
		f"Model: {MODEL_NAME}_barlow_twins, "
		f"total={model_info['total_params']:,}, "
		f"trainable={model_info['trainable_params']:,}, "
		f"frozen={model_info['frozen_params']:,}"
	)

	criterion = BarlowTwinsLoss(lambd=LAMBD)
	optimizer = torch.optim.AdamW(
		filter(lambda p: p.requires_grad, model.parameters()),
		lr=LR, weight_decay=WEIGHT_DECAY
	)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

	# 9. Vòng lặp huấn luyện chính
	print("\n[Step 9] Bắt đầu huấn luyện Barlow Twins SSL...")
	history = {
		"train_loss": [],
		"val_cross_recall1": [],
		"val_cross_recall5": [],
		"val_cross_map": [],
		"val_cross_auc": []
	}
	if CALCULATE_CLUSTERING_METRICS:
		history.update({
			"val_silhouette": [], "val_dbi": [], "val_chi": [], "val_dunn": [], "val_nmi": [], "val_ratio": []
		})
	
	best_map = 0.0
	epochs_no_improve = 0

	for epoch in range(1, EPOCHS + 1):
		# Train SSL
		loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, EPOCHS)
		
		# Đánh giá KNN Retrieval (Val làm Query, Train làm Gallery)
		val_cross_results = evaluate_cross_retrieval(model, val_loader, train_eval_loader, device, class_names)
		
		# Lưu history
		history["train_loss"].append(loss)
		history["val_cross_recall1"].append(val_cross_results["Recall@1"])
		history["val_cross_recall5"].append(val_cross_results["Recall@5"])
		history["val_cross_map"].append(val_cross_results["mAP"])
		history["val_cross_auc"].append(val_cross_results["AUC"])

		# Tính toán clustering metrics nếu được bật
		val_results = None
		if CALCULATE_CLUSTERING_METRICS:
			val_results = evaluate_retrieval(model, val_loader, device, class_names, eval_clustering=True)
			# Lưu history
			history["val_silhouette"].append(val_results["Silhouette"])
			history["val_dbi"].append(val_results["Davies-Bouldin"])
			history["val_chi"].append(val_results["Calinski-Harabasz"])
			history["val_dunn"].append(val_results["Dunn-Index"])
			history["val_nmi"].append(val_results["NMI"])
			history["val_ratio"].append(val_results["Intra-Inter-Ratio"])

		current_lr = optimizer.param_groups[0]["lr"]
		
		log_msg = f"Epoch {epoch}/{EPOCHS} —\n"
		log_msg += f"  Loss (SSL Train)           : {loss:.4f}\n"
		log_msg += f"  Val Cross mAP (Retrieval)  : {val_cross_results['mAP']*100:.2f}%\n"
		log_msg += f"  Val Cross Recall@1         : {val_cross_results['Recall@1']*100:.2f}%\n"
		log_msg += f"  Val Cross AUC              : {val_cross_results['AUC']:.4f}\n"
		
		if val_results is not None:
			log_msg += f"  Val Clustering             : Silhouette: {val_results['Silhouette']:.4f} | DBI: {val_results['Davies-Bouldin']:.4f} | CHI: {val_results['Calinski-Harabasz']:.2f} | Dunn: {val_results['Dunn-Index']:.4f} | NMI: {val_results['NMI']:.4f} | Ratio: {val_results['Intra-Inter-Ratio']:.4f}\n"
			
		log_msg += f"  LR                         : {current_lr:.6f}"
		print(log_msg)

		scheduler.step()

		# Checkpoint model theo mAP
		val_map = val_cross_results["mAP"]
		if val_map > best_map:
			best_map = val_map
			epochs_no_improve = 0
			raw_model = model
			torch.save(raw_model.state_dict(), output_dir / "best_model.pth")
			print(f"  → Saved best model (Val Cross mAP={best_map*100:.2f}%)")
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

	# Đánh giá chéo cuối trên tập Test (Query: Test, Gallery: Train)
	print("\n[Đánh giá cuối - Test Query vs Training Gallery]")
	test_cross_results = evaluate_cross_retrieval(model, test_loader, train_eval_loader, device, class_names)
	test_cross_report = format_retrieval_report(test_cross_results, class_names, prefix="Test Query vs Train Gallery")
	print(test_cross_report)
	
	with open(output_dir / "retrieval_report_test_query_train_gallery.txt", "w", encoding="utf-8") as f:
		f.write(test_cross_report)

	# Ghi summary kết quả
	test_cross_summary = {}
	for k, v in test_cross_results.items():
		if isinstance(v, (int, float, np.integer, np.floating)):
			test_cross_summary[k] = round(float(v), 6)

	summary = {
		"method": "Self-Supervised Learning (Barlow Twins)",
		"model": MODEL_NAME,
		"projection_dim": PROJECTION_DIM,
		"lambd": LAMBD,
		"batch_size": BATCH_SIZE,
		"epochs_trained": len(history["train_loss"]),
		"best_val_map": best_map,
		"test_cross_results": test_cross_summary,
		"train_size": len(df_train),
		"val_size": len(df_val),
		"test_size": len(df_test),
	}
	
	with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
		json.dump(summary, f, indent=2, ensure_ascii=False)

	# Vẽ đồ thị Loss và mAP
	epochs_range = range(1, len(history["train_loss"]) + 1)
	plt.figure(figsize=(12, 5))
	
	plt.subplot(1, 2, 1)
	plt.plot(epochs_range, history["train_loss"], label="SSL Train Loss", color="blue")
	plt.xlabel("Epoch")
	plt.ylabel("Loss")
	plt.title("SSL Train Loss Curve")
	plt.legend()
	
	plt.subplot(1, 2, 2)
	plt.plot(epochs_range, history["val_cross_map"], label="Val Cross mAP", color="green")
	plt.xlabel("Epoch")
	plt.ylabel("mAP")
	plt.title("Val Cross mAP Curve")
	plt.legend()
	
	plt.tight_layout()
	plt.savefig(output_dir / "training_metrics.png", dpi=200)
	plt.close()
	
	print(f"\n[Hoàn tất] Báo cáo và checkpoint đã được lưu tại: {output_dir}")

if __name__ == "__main__":
	main()
