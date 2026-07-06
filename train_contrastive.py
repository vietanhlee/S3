"""
train_contrastive.py
====================
Metric Learning với Contrastive Loss (Online Contrastive Loss với Hard Pair Mining).
- Backbone: ConvNeXt-Tiny (freeze 90%)
- Projection Head: 768 → 256 chiều + L2 Normalize
- PK Sampler: P=12 classes × K=8 samples/class (batch=96)
- 200 epochs, EarlyStopping theo mAP trên tập Val
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

# ===== CẤU HÌNH =====
ROOT_DIR = r"/kaggle/input/datasets/b23dckh002lvitanh/s3-origin/S3"
OUTPUT_DIR = "outputs_contrastive"
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
SEED = 42
P_CLASSES = 18          # Số class mỗi batch
K_SAMPLES = 10          # Số ảnh mỗi class trong batch
EPOCHS = 20
PATIENCE = 6            # EarlyStopping patience
LR = 1e-4
WEIGHT_DECAY = 1e-4
EMBEDDING_DIM = 256     # Projection head output
MARGIN = 0.5            # Contrastive loss margin
FREEZE_RATIO = 0.90
MODEL_NAME = "convnext_tiny"
COSINE_THRESHOLD = 0.92
EMB_BATCH_SIZE = 64
# =====================

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
	evaluate_loss,
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
	"""PK Batch Sampler: mỗi batch chứa P classes × K samples/class."""
	def __init__(self, labels: list, p: int, k: int) -> None:
		self.labels = labels
		self.p = p
		self.k = k

		self.label_to_indices: dict[int, list[int]] = {}
		for idx, lbl in enumerate(labels):
			self.label_to_indices.setdefault(lbl, []).append(idx)

		self.unique_labels = list(self.label_to_indices.keys())
		self.n_batches = max(1, len(labels) // (p * k))

	def __iter__(self):
		for _ in range(self.n_batches):
			p_actual = min(self.p, len(self.unique_labels))
			selected_labels = random.sample(self.unique_labels, p_actual)

			batch = []
			for lbl in selected_labels:
				indices = self.label_to_indices[lbl]
				if len(indices) >= self.k:
					sampled = random.sample(indices, self.k)
				else:
					sampled = random.choices(indices, k=self.k)
				batch.extend(sampled)
			yield batch

	def __len__(self) -> int:
		return self.n_batches


# ============================================================
# Model
# ============================================================

class MetricModel(nn.Module):
	"""ConvNeXt-Tiny backbone + Projection Head → L2 Normalized embeddings."""
	def __init__(self, embedding_dim: int = 256, freeze_ratio: float = 0.90) -> None:
		super().__init__()
		self.backbone = timm.create_model(
			MODEL_NAME, pretrained=True, num_classes=0, global_pool="avg",
		)
		freeze_model_layers(self.backbone, freeze_ratio)

		backbone_dim = self.backbone.num_features  # 768

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
	"""Contrastive Loss với Online Hard Pair Mining."""
	def __init__(self, margin: float = 0.5) -> None:
		super().__init__()
		self.margin = margin

	def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
		dist_matrix = torch.cdist(embeddings, embeddings, p=2)
		n = embeddings.size(0)

		labels_col = labels.unsqueeze(1)
		is_positive = (labels_col == labels_col.t()).float()
		is_negative = 1.0 - is_positive
		mask_diag = 1.0 - torch.eye(n, device=embeddings.device)
		is_positive = is_positive * mask_diag
		is_negative = is_negative * mask_diag

		pos_loss = (is_positive * (dist_matrix ** 2)).sum() / max(1.0, is_positive.sum())

		neg_loss_terms = torch.clamp(self.margin - dist_matrix, min=0.0) ** 2
		neg_loss = (is_negative * neg_loss_terms).sum() / max(1.0, is_negative.sum())

		return pos_loss + neg_loss


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

	# Lưu class index
	with open(output_dir / "class_indices.json", "w", encoding="utf-8") as f:
		json.dump(class_to_idx, f, indent=2, ensure_ascii=False)

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

	validate_split(df_filtered, df_train, df_val, df_test, "Contrastive_EndVersion")
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
	train_tf, eval_tf = build_transforms(img_size, mean, std)

	train_ds = MetricImageDataset(df_train, class_to_idx, transform=train_tf)
	val_ds = MetricImageDataset(df_val, class_to_idx, transform=eval_tf)
	test_ds = MetricImageDataset(df_test, class_to_idx, transform=eval_tf)

	# Dùng PK Sampler cho tập train
	pk_sampler = PKSampler(train_ds.labels, p=P_CLASSES, k=K_SAMPLES)
	train_loader = DataLoader(train_ds, batch_sampler=pk_sampler, num_workers=0, pin_memory=True)

	# DataLoader đánh giá (tuần tự)
	val_loader = DataLoader(val_ds, batch_size=P_CLASSES * K_SAMPLES, shuffle=False, num_workers=0, pin_memory=True)
	test_loader = DataLoader(test_ds, batch_size=P_CLASSES * K_SAMPLES, shuffle=False, num_workers=0, pin_memory=True)

	print(f"Train: {len(train_ds)} ảnh | Val: {len(val_ds)} ảnh | Test: {len(test_ds)} ảnh")

	# 5. Khởi tạo Model, Loss, Optimizer
	print("\n[Step 5] Khởi tạo model và optimizer...")
	model = MetricModel(embedding_dim=EMBEDDING_DIM, freeze_ratio=FREEZE_RATIO).to(device)

	model_info = summarize_model(model)
	print(
		f"Model: {MODEL_NAME}_contrastive, "
		f"total={model_info['total_params']:,}, "
		f"trainable={model_info['trainable_params']:,}, "
		f"frozen={model_info['frozen_params']:,}"
	)

	criterion = OnlineContrastiveLoss(margin=MARGIN)
	optimizer = torch.optim.AdamW(
		filter(lambda p: p.requires_grad, model.parameters()),
		lr=LR, weight_decay=WEIGHT_DECAY
	)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

	# 8. Trích xuất trạng thái trước training (Pre-training analysis)
	print("\n[Analysis] Trích xuất đặc trưng và Grad-CAM trước training...")
	train_eval_loader = DataLoader(train_ds, batch_size=P_CLASSES * K_SAMPLES, shuffle=False, num_workers=0, pin_memory=True)
	
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
	print("\n[Step 9] Bắt đầu huấn luyện Metric Learning...")
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
		
		# Đánh giá retrieval
		train_results = evaluate_retrieval(model, train_eval_loader, device, class_names)
		val_results = evaluate_retrieval(model, val_loader, device, class_names)
		val_cross_results = evaluate_cross_retrieval(model, val_loader, train_eval_loader, device, class_names)

		# Lưu history
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

		# Checkpoint best model theo mAP
		val_map = val_results["mAP"]
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
		"method": "Contrastive Loss (Online Contrastive Loss with Hard Pair Mining)",
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
