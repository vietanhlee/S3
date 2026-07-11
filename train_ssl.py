"""
train_ssl.py
============
Tệp huấn luyện tự giám sát (Self-Supervised Learning) trên bộ dữ liệu ảnh gỗ S3.
Áp dụng khung kiến trúc Barlow Twins (ICML 2021) để học biểu diễn đặc trưng không nhãn.
Sử dụng ConvNeXt-Tiny làm backbone cùng 3 lớp MLP Projection Head.

Đồng bộ hoàn toàn cấu trúc bộ khung với train_triplet.py và train_contrastive.py:
- Hỗ trợ EVAL_MODE linh hoạt ('self', 'cross', 'both')
- Trích xuất đặc trưng và vẽ t-SNE, phân tích khoảng cách trước/sau training
- Phân tích Grad-CAM trước/sau training trên các mẫu đại diện
- Song song hóa dữ liệu với NUM_WORKERS = 4

Cách chạy:
python train_ssl.py
"""

import os
import gc
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

# Import utilities từ utils.py
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
	extract_all_embeddings,
	evaluate_retrieval,
	format_retrieval_report,
	evaluate_cross_retrieval,
	select_gradcam_representatives,
	compute_class_prototypes,
	generate_gradcam_maps,
	plot_gradcam_comparison,
	plot_tsne_comparison,
	plot_distance_analysis,
	plot_metrics_summary,
	plot_all_metrics_per_epoch,
	build_transforms,
	CAM_METHODS,
)
from train_final import end_version_split
from split_methods import validate_split

# ===== CẤU HÌNH =====
ROOT_DIR = r"/kaggle/input/datasets/b23dckh002lvitanh/s3-origin/S3"
OUTPUT_DIR = "outputs_ssl"
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
SEED = 42
EPOCHS = 60
PATIENCE = 20           # EarlyStopping patience
LR = 5e-4
WEIGHT_DECAY = 1e-4
EMBEDDING_DIM = 256     # Projection head output của SupCon
TEMPERATURE = 0.07      # SupCon Loss temperature
FREEZE_RATIO = 0.90
MODEL_NAME = "convnext_tiny"
EMB_BATCH_SIZE = 128
BATCH_SIZE = 128        # Batch size cho SSL train
NUM_WORKERS = 4
CALCULATE_CLUSTERING_METRICS = True  # Đặt True nếu muốn tính toán clustering metrics mỗi epoch
EVAL_MODE = "cross"                   # Chế độ đánh giá: 'self', 'cross', hoặc 'both'
# =====================

class DoubleViewTransform:
	"""Tạo hai góc nhìn biến đổi ngẫu nhiên khác nhau từ cùng một ảnh."""
	def __init__(self, transform1, transform2):
		self.transform1 = transform1
		self.transform2 = transform2

	def __call__(self, img):
		return self.transform1(img), self.transform2(img)

class SSLDataset(Dataset):
	"""Dataset phục vụ huấn luyện tự giám sát có giám sát (Supervised Contrastive Learning - SupCon)."""
	def __init__(self, df: pd.DataFrame, class_to_idx: dict, transform=None) -> None:
		self.df = df.reset_index(drop=True)
		self.transform = transform
		self.class_to_idx = class_to_idx
		
		# Gán nhãn lớp dạng index cho mỗi hàng
		self.labels = [class_to_idx[lbl] for lbl in self.df["label"]]

	def __len__(self) -> int:
		return len(self.df)

	def __getitem__(self, idx: int):
		row = self.df.iloc[idx]
		with Image.open(row["path"]) as img:
			img = img.convert("RGB")
		
		if self.transform:
			# Sinh 2 views từ cùng một ảnh
			img1, img2 = self.transform(img)
		else:
			img1 = transforms.ToTensor()(img)
			img2 = transforms.ToTensor()(img)

		return img1, img2, self.labels[idx]

class MetricImageDataset(Dataset):
	"""Dataset cho metric learning / evaluation, trả về (image, label_idx)."""
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

class SupConModel(nn.Module):
	"""Kiến trúc SupCon: Backbone + 2-layer Projection Head -> L2 Normalized embeddings."""
	def __init__(self, backbone_name: str = "convnext_tiny", embedding_dim: int = 256, freeze_ratio: float = 0.90) -> None:
		super().__init__()
		self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)
		freeze_model_layers(self.backbone, freeze_ratio)

		backbone_dim = self.backbone.num_features  # 768 với convnext_tiny

		# Projection Head
		self.projector = nn.Sequential(
			nn.Linear(backbone_dim, backbone_dim),
			nn.BatchNorm1d(backbone_dim),
			nn.ReLU(inplace=True),
			nn.Linear(backbone_dim, embedding_dim),
		)
		self.model_name = f"{backbone_name}_supcon"

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		features = self.backbone(x)
		embeddings = self.projector(features)
		return F.normalize(embeddings, p=2, dim=1)


# ============================================================
# Supervised Contrastive Loss (SupCon Loss)
# ============================================================

class SupConLoss(nn.Module):
	"""Supervised Contrastive Learning Loss (NeurIPS 2020)."""
	def __init__(self, temperature: float = 0.07) -> None:
		super().__init__()
		self.temperature = temperature

	def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
		# features: shape (B, N_views, D)
		# labels: shape (B)
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
		mask = torch.eq(labels, labels.t()).float().to(device) # (B, B)
		mask = mask.repeat(n_views, n_views) # (B * V, B * V)
		
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
		loss = - mean_log_prob_pos
		loss = loss.view(n_views, batch_size).mean()
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
	for x1, x2, labels in pbar:
		x1 = x1.to(device, non_blocking=True)
		x2 = x2.to(device, non_blocking=True)
		labels = labels.to(device, non_blocking=True)

		optimizer.zero_grad()
		
		# Trích xuất embeddings của 2 views
		emb1 = model(x1)
		emb2 = model(x2)
		
		# Stack views thành tensor shape (B, 2, D)
		features = torch.stack([emb1, emb2], dim=1)
		
		loss = criterion(features, labels)
		loss.backward()
		optimizer.step()

		total_loss += loss.item()
		n_batches += 1
		pbar.set_postfix(loss=f"{total_loss / n_batches:.4f}")

	return total_loss / max(n_batches, 1)

# ============================================================
# Main Pipeline
# ============================================================

def main() -> None:
	set_seed(SEED)
	device = get_device()
	print(f"Device: {device}")

	output_dir = Path(OUTPUT_DIR)
	output_dir.mkdir(parents=True, exist_ok=True)

	# 1. Thu thập ảnh, build dataframe
	print("\n[Step 1] Thu thập ảnh...")
	samples = collect_image_samples(ROOT_DIR)
	if not samples:
		raise ValueError(f"Không tìm thấy ảnh nào trong {ROOT_DIR}")

	df = build_dataframe(samples)
	df_filtered = df[df["label"] != "Pterocarpus sp"].reset_index(drop=True)
	print(f"Tổng ảnh sau khi lọc bỏ Pterocarpus sp: {len(df_filtered)}, Số class: {df_filtered['label'].nunique()}")

	class_names = sorted(df_filtered["label"].unique().tolist())
	class_to_idx = {name: i for i, name in enumerate(class_names)}

	# 2. Compute embeddings cho chia dữ liệu
	print("\n[Step 2] Compute embeddings cho chia dữ liệu...")
	print("Trích xuất embeddings với EfficientNetV2-M...")
	embs_eff = compute_embeddings_v2(df_filtered, "tf_efficientnetv2_m_in21k", batch_size=EMB_BATCH_SIZE, device=device)
	print("Trích xuất embeddings với Swin-Large...")
	embs_swin = compute_embeddings_v2(df_filtered, "swin_large_patch4_window7_224", batch_size=EMB_BATCH_SIZE, device=device)

	# 3. Chia dữ liệu theo PP chuẩn cuối (End Version Split)
	print("\n[Step 3] Chia dữ liệu theo End Version Split...")
	df_train, df_val, df_test = end_version_split(
		df_filtered, embs_eff, embs_swin,
		train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, seed=SEED,
	)

	validate_split(df_filtered, df_train, df_val, df_test, "SSL_EndVersion")
	log_split_summary(df_filtered, df_train, df_val, df_test)

	# Vẽ biểu đồ phân phối lớp EDA
	eda_split_class_distribution(
		df_train, df_val, df_test,
		"End Version - Class Distribution",
		output_dir / "eda_split_end_version.png",
	)

	# 4. Thiết lập Transforms & Dataloaders
	print("\n[Step 4] Chuẩn bị Dataset và DataLoader...")
	cfg_model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=0)
	cfg = resolve_data_config({}, model=cfg_model)
	img_size = cfg.get("input_size", (3, 224, 224))[-1]
	mean = cfg.get("mean", (0.485, 0.456, 0.406))
	std = cfg.get("std", (0.229, 0.224, 0.225))
	
	# Augment View kép cho SSL training
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

	# Dataset SSL
	train_ds = SSLDataset(df_train, class_to_idx, transform=double_transform)
	
	# Datasets phục vụ đánh giá (MetricImageDataset trả về image, label_idx)
	train_eval_ds = MetricImageDataset(df_train, class_to_idx, transform=eval_tf)
	val_ds = MetricImageDataset(df_val, class_to_idx, transform=eval_tf)
	test_ds = MetricImageDataset(df_test, class_to_idx, transform=eval_tf)

	# DataLoaders song song hóa với NUM_WORKERS = 4
	train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
	
	train_eval_loader = DataLoader(train_eval_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
	val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
	test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

	print(f"SSL Train Batch size: {BATCH_SIZE} | Workers: {NUM_WORKERS}")

	# 5. Khởi tạo Model, Loss, Optimizer
	print("\n[Step 5] Khởi tạo model và optimizer...")
	model = SupConModel(backbone_name=MODEL_NAME, embedding_dim=EMBEDDING_DIM, freeze_ratio=FREEZE_RATIO).to(device)

	model_info = summarize_model(model)
	print(
		f"Model: {MODEL_NAME}_supcon, "
		f"total={model_info['total_params']:,}, "
		f"trainable={model_info['trainable_params']:,}, "
		f"frozen={model_info['frozen_params']:,}"
	)

	criterion = SupConLoss(temperature=TEMPERATURE)
	optimizer = torch.optim.AdamW(
		filter(lambda p: p.requires_grad, model.parameters()),
		lr=LR, weight_decay=WEIGHT_DECAY
	)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

	# 8. Trích xuất trạng thái trước training (Pre-training analysis)
	print("\n[Analysis] Trích xuất đặc trưng và Grad-CAM trước training...")
	
	# Chọn 4 ảnh đại diện cho Dalbergia và 4 ảnh cho Pterocarpus để phân tích Grad-CAM
	representatives = select_gradcam_representatives(df_val, seed=SEED)
	reps_dict = representatives
	reps_flat = representatives['Dalbergia'] + representatives['Pterocarpus']

	before_protos = compute_class_prototypes(model, train_eval_loader, device, len(class_names))
	before_cams_dict = {}
	for method in CAM_METHODS:
		before_cams_dict[method] = generate_gradcam_maps(
			model, reps_flat, before_protos, class_to_idx, eval_tf, device, method=method
		)

	# Tính embeddings trước training cho tập Test
	print("  Trích xuất đặc trưng của tập Test trước training...")
	before_test_embs, test_labels = extract_all_embeddings(model, test_loader, device)
	before_test_embs = before_test_embs.numpy()

	# 9. Vòng lặp huấn luyện chính
	print("\n[Step 9] Bắt đầu huấn luyện Supervised Contrastive Learning (SupCon) SSL...")
	history = {
		"train_loss": [],
	}
	if EVAL_MODE in ["self", "both"]:
		history.update({
			"val_recall1": [], "val_recall5": [],
			"val_precision1": [], "val_precision5": [],
			"val_map": [], "val_auc": [],
		})
	if CALCULATE_CLUSTERING_METRICS:
		history.update({
			"val_silhouette": [], "val_dbi": [], "val_chi": [], "val_dunn": [], "val_nmi": [], "val_ratio": []
		})
	if EVAL_MODE in ["cross", "both"]:
		history.update({
			"val_cross_recall1": [], "val_cross_recall5": [],
			"val_cross_precision1": [], "val_cross_precision5": [],
			"val_cross_map": [], "val_cross_auc": []
		})

	best_map = 0.0
	epochs_no_improve = 0

	for epoch in range(1, EPOCHS + 1):
		# Train SSL
		loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, EPOCHS)
		
		# Lưu history chung
		history["train_loss"].append(loss)

		# Tính toán có điều kiện theo EVAL_MODE hoặc CALCULATE_CLUSTERING_METRICS
		val_results = None
		val_cross_results = None

		if (EVAL_MODE in ["self", "both"]) or CALCULATE_CLUSTERING_METRICS:
			val_results = evaluate_retrieval(model, val_loader, device, class_names, eval_clustering=CALCULATE_CLUSTERING_METRICS)
			
			if EVAL_MODE in ["self", "both"]:
				history["val_recall1"].append(val_results["Recall@1"])
				history["val_recall5"].append(val_results["Recall@5"])
				history["val_precision1"].append(val_results["Precision@1"])
				history["val_precision5"].append(val_results["Precision@5"])
				history["val_map"].append(val_results["mAP"])
				history["val_auc"].append(val_results["AUC"])

			if CALCULATE_CLUSTERING_METRICS:
				history["val_silhouette"].append(val_results["Silhouette"])
				history["val_dbi"].append(val_results["Davies-Bouldin"])
				history["val_chi"].append(val_results["Calinski-Harabasz"])
				history["val_dunn"].append(val_results["Dunn-Index"])
				history["val_nmi"].append(val_results["NMI"])
				history["val_ratio"].append(val_results["Intra-Inter-Ratio"])

		if EVAL_MODE in ["cross", "both"]:
			val_cross_results = evaluate_cross_retrieval(model, val_loader, train_eval_loader, device, class_names)
			history["val_cross_recall1"].append(val_cross_results["Recall@1"])
			history["val_cross_recall5"].append(val_cross_results["Recall@5"])
			history["val_cross_precision1"].append(val_cross_results["Precision@1"])
			history["val_cross_precision5"].append(val_cross_results["Precision@5"])
			history["val_cross_map"].append(val_cross_results["mAP"])
			history["val_cross_auc"].append(val_cross_results["AUC"])

		current_lr = optimizer.param_groups[0]["lr"]

		# Xây dựng log động
		log_msg = f"Epoch {epoch}/{EPOCHS} —\n"
		log_msg += f"  Loss (SSL Train): {loss:.4f}\n"

		if val_results is not None:
			if EVAL_MODE in ["self", "both"]:
				log_msg += f"  Val Self Metrics: R@1: {val_results['Recall@1']*100:.2f}% | R@5: {val_results['Recall@5']*100:.2f}% | P@1: {val_results['Precision@1']*100:.2f}% | mAP: {val_results['mAP']*100:.2f}% | AUC: {val_results['AUC']:.4f}\n"
			if CALCULATE_CLUSTERING_METRICS:
				log_msg += f"  Val Clustering  : Silhouette: {val_results['Silhouette']:.4f} | DBI: {val_results['Davies-Bouldin']:.4f} | CHI: {val_results['Calinski-Harabasz']:.2f} | Dunn: {val_results['Dunn-Index']:.4f} | NMI: {val_results['NMI']:.4f} | Ratio: {val_results['Intra-Inter-Ratio']:.4f}\n"

		if val_cross_results is not None:
			log_msg += f"  Val Cross Metrics: R@1: {val_cross_results['Recall@1']*100:.2f}% | R@5: {val_cross_results['Recall@5']*100:.2f}% | P@1: {val_cross_results['Precision@1']*100:.2f}% | mAP: {val_cross_results['mAP']*100:.2f}% | AUC: {val_cross_results['AUC']:.4f}\n"

		log_msg += f"  LR: {current_lr:.6f}"
		print(log_msg)

		scheduler.step()

		# Chọn mAP để Early Stopping theo chế độ (Cross làm chuẩn nếu có)
		val_map = val_cross_results["mAP"] if val_cross_results is not None else val_results["mAP"]
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

	# Đánh giá cuối có điều kiện theo EVAL_MODE
	val_summary = {}
	test_summary = {}
	val_cross_summary = {}
	test_cross_summary = {}

	if EVAL_MODE in ["self", "both"]:
		# Đánh giá Val
		print("\n[Đánh giá cuối - Validation Self]")
		val_results = evaluate_retrieval(model, val_loader, device, class_names, eval_clustering=True)
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

		for k, v in val_results.items():
			if isinstance(v, (int, float, np.integer, np.floating)):
				val_summary[k] = round(float(v), 6)

		# Đánh giá Test
		print("\n[Đánh giá cuối - Test Self]")
		test_results = evaluate_retrieval(model, test_loader, device, class_names, eval_clustering=True)
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

		for k, v in test_results.items():
			if isinstance(v, (int, float, np.integer, np.floating)):
				test_summary[k] = round(float(v), 6)

	if EVAL_MODE in ["cross", "both"]:
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

		for k, v in val_cross_results.items():
			if isinstance(v, (int, float, np.integer, np.floating)):
				val_cross_summary[k] = round(float(v), 6)

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

		for k, v in test_cross_results.items():
			if isinstance(v, (int, float, np.integer, np.floating)):
				test_cross_summary[k] = round(float(v), 6)

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

	# Vẽ biểu đồ tổng hợp metrics (Tách val_loss bằng cách xấp xỉ do SSL không có val classification loss thực tế)
	# Để hàm plot_metrics_summary không lỗi do thiếu trường val_loss trong history:
	if "val_loss" not in history:
		history["val_loss"] = [0.0] * len(history["train_loss"])
	plot_metrics_summary(history, model, val_loader, test_loader, device, output_dir)

	# Vẽ từng metric riêng biệt theo epoch
	plot_all_metrics_per_epoch(history, output_dir)

	summary = {
		"method": "Supervised Contrastive Learning (SupCon) SSL",
		"model": MODEL_NAME,
		"embedding_dim": EMBEDDING_DIM,
		"temperature": TEMPERATURE,
		"batch_size": BATCH_SIZE,
		"epochs_trained": len(history["train_loss"]),
		"best_val_map": best_map,
		"train_size": len(df_train),
		"val_size": len(df_val),
		"test_size": len(df_test),
	}
	if EVAL_MODE in ["self", "both"]:
		summary.update({
			"val_results": val_summary,
			"test_results": test_summary
		})
	if EVAL_MODE in ["cross", "both"]:
		summary.update({
			"val_cross_results": val_cross_summary,
			"test_cross_results": test_cross_summary
		})
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
