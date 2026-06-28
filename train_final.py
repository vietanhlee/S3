"""
train_final.py
==============
Script training model với phương pháp chia dữ liệu chuẩn cuối (End Version).
Kết hợp nhiều phương pháp chia (PP2, PP4, PP5, PP7, PP8, PP9) cho từng class gỗ,
sử dụng embeddings trích xuất từ các model tối ưu tương ứng (EfficientNetV2-M hoặc Swin-Large),
có xử lý hoán đổi Val/Test đối với các class được cấu hình "của val",
loại bỏ hoàn toàn lớp 'Pterocarpus sp'.
"""

import os
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from timm.data import resolve_data_config
from sklearn.metrics import classification_report

# ===== CẤU HÌNH =====
ROOT_DIR = r"/kaggle/input/datasets/b23dckh002lvitanh/s3-origin/S3"
OUTPUT_BASE_DIR = "outputs_final"
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
# TEST_RATIO = 1.0 - TRAIN_RATIO - VAL_RATIO = 0.2
SEED = 42
BATCH_SIZE = 128
EPOCHS = 22
PATIENCE = 50
LR = 5e-4
WEIGHT_DECAY = 1e-2
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.25
MODEL_NAME = "convnext_tiny"
FREEZE_RATIO = 0.90
COSINE_THRESHOLD = 0.92  # Cho PP5
# ====================

from train import (
	set_seed,
	get_device,
	collect_image_samples,
	build_dataframe,
	log_split_summary,
	eda_split_class_distribution,
	ImageListDataset,
	build_transforms,
	FocalLoss,
	accuracy_from_logits,
	train_model,
	plot_training_curves,
	evaluate_and_report,
	summarize_model,
	freeze_model_layers,
	validate_split_minimums,
)

from split_methods import (
	SPLIT_METHODS,
	validate_split,
)

import timm


def build_model(num_classes: int) -> torch.nn.Module:
	"""Tạo model pretrained, freeze theo tỉ lệ."""
	model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=num_classes)
	freeze_model_layers(model, FREEZE_RATIO)
	model.model_name = MODEL_NAME
	return model


def compute_embeddings_v2(
	df: pd.DataFrame,
	model_name: str,
	batch_size: int,
	device: torch.device,
) -> np.ndarray:
	"""Trích xuất embeddings sử dụng model_name cụ thể."""
	print(f"  -> Khởi tạo model embedding: {model_name}...")
	timm_model_name = model_name
	if model_name == "tf_efficientnetv2_m_in21k":
		timm_model_name = "tf_efficientnetv2_m.in21k"

	model = timm.create_model(timm_model_name, pretrained=True, num_classes=0)
	model = model.to(device)
	model.eval()

	cfg = resolve_data_config({}, model=model)
	img_size = cfg.get("input_size", (3, 224, 224))[-1]
	mean = cfg.get("mean", (0.485, 0.456, 0.406))
	std = cfg.get("std", (0.229, 0.224, 0.225))
	transform = transforms.Compose(
		[
			transforms.Resize((img_size, img_size)),
			transforms.ToTensor(),
			transforms.Normalize(mean=mean, std=std),
		]
	)

	from train import ImagePathDataset
	fs = ImagePathDataset(df, transform=transform)
	num_workers = min(4, os.cpu_count() or 1)
	loader = DataLoader(
		fs,
		batch_size=batch_size,
		shuffle=False,
		num_workers=num_workers,
		pin_memory=True,
	)

	features = []
	with torch.no_grad():
		for images in tqdm(loader, desc=f"Embed ({model_name})"):
			images = images.to(device)
			feats = model(images)
			if isinstance(feats, (list, tuple)):
				feats = feats[0]
			features.append(feats.detach().cpu().numpy())

	del model
	if device.type == "cuda":
		torch.cuda.empty_cache()

	if not features:
		return np.empty((0, 0), dtype=np.float32)
	return np.concatenate(features, axis=0)


def end_version_split(
	df: pd.DataFrame,
	embs_eff: np.ndarray,
	embs_swin: np.ndarray,
	train_ratio: float = 0.60,
	val_ratio: float = 0.20,
	seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	"""
	PP Chuẩn cuối (End Version): Kết hợp nhiều phương pháp chia khác nhau cho từng class
	theo cấu hình được định nghĩa trong todo1.md.
	Loại bỏ hoàn toàn class 'Pterocarpus sp'.
	"""
	# 1. Loại bỏ class 'Pterocarpus sp'
	keep_mask = df["label"] != "Pterocarpus sp"
	df_filtered = df[keep_mask].reset_index(drop=True)
	emb_eff_filtered = embs_eff[keep_mask.values]
	emb_swin_filtered = embs_swin[keep_mask.values]

	# 2. Định nghĩa cấu hình chia cho từng class theo todo1.md
	# (PP_Key, Swap_Mode, Embedding_Model)
	split_config = {
		"Afzelia africana": ("PP8", "test", "eff"),
		"Afzelia bella": ("PP4", "val", "swin"),
		"Afzelia pachyloba": ("PP9", "test", "swin"),
		"Afzelia quanzensis": ("PP2", "test", "eff"),
		"Dalbergia cochinchinensis": ("PP9", "val", "eff"),
		"Dalbergia melanoxylon": ("PP2", "val", "eff"),
		"Dalbergia oliveri": ("PP8", "test", "eff"),
		"Dalbergia rimosa": ("PP4", "test", "eff"),
		"Dalbergia tonkinensis": ("PP4", "test", "swin"),
		"Guibourtia arnoldiana": ("PP4", "test", "swin"),
		"Guibourtia coleosperma": ("PP9", "test", "swin"),
		"Guibourtia ehie": ("PP4", "test", "swin"),
		"Peltogyne pubescens": ("PP2", "test", "eff"),
		"Pterocarpus erinaceus": ("PP9", "test", "eff"),
		"Pterocarpus indicus": ("PP9", "test", "eff"),
		"Pterocarpus macrocarpus": ("PP4", "test", "eff"),
		"Pterocarpus soyauxii": ("PP4", "test", "swin"),
		"Sindora cochinchinensis": ("PP2", "test", "swin"),
		"Sindora tonkinensis": ("PP9", "val", "eff"),
	}

	pp_map = {
		"PP2": "PP2_Mahalanobis_Iterative",
		"PP4": "PP4_Hierarchical_Clustering",
		"PP5": "PP5_Cosine_Graph",
		"PP7": "PP7_Adversarial_Validation",
		"PP8": "PP8_StratifiedGroupKFold",
		"PP9": "PP9_Agglom_Stratified",
	}

	train_idx_all = []
	val_idx_all = []
	test_idx_all = []

	# Duyệt qua từng class để chia riêng biệt
	for label, group in df_filtered.groupby("label"):
		# Lấy các hàng và mappings
		indices = group.index.tolist()
		sub_df = group.copy()
		path_to_orig_idx = dict(zip(group["path"], group.index))
		sub_df_reset = sub_df.reset_index(drop=True)

		# Lấy cấu hình chia
		if label in split_config:
			pp_key, mode, model_type = split_config[label]
		else:
			print(f"[Warning] Class '{label}' không có trong cấu hình chia. Sử dụng mặc định PP8 của test.")
			pp_key, mode, model_type = "PP8", "test", "eff"

		# Lấy embeddings tương ứng
		if model_type == "eff":
			sub_emb = emb_eff_filtered[indices]
		else:
			sub_emb = emb_swin_filtered[indices]

		full_pp_name = pp_map[pp_key]
		split_fn = SPLIT_METHODS[full_pp_name]

		# Chạy hàm chia dữ liệu cho riêng class này
		try:
			if pp_key == "PP5":
				tr_df, val_df, te_df = split_fn(
					sub_df_reset, sub_emb,
					train_ratio=train_ratio,
					val_ratio=val_ratio,
					seed=seed,
					cosine_threshold=COSINE_THRESHOLD,
				)
			else:
				tr_df, val_df, te_df = split_fn(
					sub_df_reset, sub_emb,
					train_ratio=train_ratio,
					val_ratio=val_ratio,
					seed=seed,
				)
		except Exception as e:
			print(f"[Error] Lỗi khi chia dữ liệu cho class '{label}' bằng {pp_key}: {e}")
			from split_methods import stratified_random_split
			tr_df, val_df, te_df = stratified_random_split(
				sub_df_reset, sub_emb,
				train_ratio=train_ratio,
				val_ratio=val_ratio,
				seed=seed,
			)

		# Ánh xạ path ngược lại chỉ số index gốc trong df_filtered
		tr_orig_idx = [path_to_orig_idx[p] for p in tr_df["path"]]
		val_orig_idx = [path_to_orig_idx[p] for p in val_df["path"]]
		te_orig_idx = [path_to_orig_idx[p] for p in te_df["path"]]

		# Áp dụng logic hoán đổi nếu mode là 'val'
		if mode == "val":
			train_idx_all.extend(tr_orig_idx)
			val_idx_all.extend(te_orig_idx)
			test_idx_all.extend(val_orig_idx)
		else:
			train_idx_all.extend(tr_orig_idx)
			val_idx_all.extend(val_orig_idx)
			test_idx_all.extend(te_orig_idx)

	df_train = df_filtered.loc[train_idx_all].sample(frac=1, random_state=seed).reset_index(drop=True)
	df_val = df_filtered.loc[val_idx_all].sample(frac=1, random_state=seed).reset_index(drop=True)
	df_test = df_filtered.loc[test_idx_all].sample(frac=1, random_state=seed).reset_index(drop=True)

	return df_train, df_val, df_test


def main() -> None:
	set_seed(SEED)
	device = get_device()
	print(f"Device: {device}")

	output_dir = Path(OUTPUT_BASE_DIR)
	output_dir.mkdir(parents=True, exist_ok=True)

	# 1. Thu thập ảnh, build dataframe
	print("\n[Step 1] Thu thập ảnh...")
	samples = collect_image_samples(ROOT_DIR)
	if not samples:
		raise ValueError(f"Không tìm thấy ảnh nào trong {ROOT_DIR}")

	df = build_dataframe(samples)
	print(f"Tổng ảnh ban đầu: {len(df)}, Số class ban đầu: {df['label'].nunique()}")

	# Lọc bỏ class Pterocarpus sp trước khi trích xuất embeddings để tối ưu hóa
	df_filtered = df[df["label"] != "Pterocarpus sp"].reset_index(drop=True)
	print(f"Tổng ảnh sau khi lọc bỏ Pterocarpus sp: {len(df_filtered)}, Số class: {df_filtered['label'].nunique()}")

	class_names = sorted(df_filtered["label"].unique().tolist())
	class_to_idx = {name: i for i, name in enumerate(class_names)}

	# Ghi danh sách class index ra file JSON để phục vụ infer
	with open(output_dir / "class_indices.json", "w", encoding="utf-8") as f:
		json.dump(class_to_idx, f, indent=2, ensure_ascii=False)

	# 2. Compute embeddings cho cả hai model: EfficientNetV2-M và Swin-Large
	print("\n[Step 2] Compute embeddings...")
	print("Trích xuất embeddings với EfficientNetV2-M...")
	embs_eff = compute_embeddings_v2(df_filtered, "tf_efficientnetv2_m_in21k", batch_size=BATCH_SIZE, device=device)
	print("Trích xuất embeddings với Swin-Large...")
	embs_swin = compute_embeddings_v2(df_filtered, "swin_large_patch4_window7_224", batch_size=BATCH_SIZE, device=device)

	# 3. Thực hiện chia dữ liệu theo phương pháp chuẩn cuối (End Version)
	print("\n[Step 3] Chia dữ liệu theo PP Chuẩn Cuối (End Version)...")
	df_train, df_val, df_test = end_version_split(
		df_filtered, embs_eff, embs_swin,
		train_ratio=TRAIN_RATIO,
		val_ratio=VAL_RATIO,
		seed=SEED,
	)

	# Validate split
	validate_split(df_filtered, df_train, df_val, df_test, "End_Version_Split")
	log_split_summary(df_filtered, df_train, df_val, df_test)

	# Vẽ biểu đồ phân phối lớp EDA
	eda_split_class_distribution(
		df_train, df_val, df_test,
		"End Version - Class Distribution",
		output_dir / "eda_split_end_version.png",
	)

	# 4. Huấn luyện model
	print("\n[Step 4] Khởi tạo model và chuẩn bị training...")
	model = build_model(num_classes=len(class_names))
	model_info = summarize_model(model)
	print(
		f"Model: {MODEL_NAME}, "
		f"total={model_info['total_params']:,}, "
		f"trainable={model_info['trainable_params']:,}, "
		f"frozen={model_info['frozen_params']:,}"
	)
	model = model.to(device)

	# Transforms
	cfg = resolve_data_config({}, model=model)
	img_size = cfg.get("input_size", (3, 224, 224))[-1]
	mean = cfg.get("mean", (0.485, 0.456, 0.406))
	std = cfg.get("std", (0.229, 0.224, 0.225))
	train_tf, eval_tf = build_transforms(img_size, mean, std)

	# Datasets & Loaders
	train_ds = ImageListDataset(df_train, class_to_idx, transform=train_tf)
	val_ds = ImageListDataset(df_val, class_to_idx, transform=eval_tf)
	test_ds = ImageListDataset(df_test, class_to_idx, transform=eval_tf)

	num_workers = 0
	train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=True)
	val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True)
	test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True)

	# Loss & Optimizer
	criterion = FocalLoss(gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA)
	optimizer = torch.optim.AdamW(
		filter(lambda p: p.requires_grad, model.parameters()),
		lr=LR,
		weight_decay=WEIGHT_DECAY,
	)

	# Scheduler
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

	print("\nBắt đầu huấn luyện...")
	history = train_model(
		model, train_loader, val_loader, optimizer, criterion,
		device, epochs=EPOCHS, patience=PATIENCE, output_dir=output_dir,
		scheduler=scheduler,
	)
	plot_training_curves(history, output_dir)

	# Load best model checkpoint để đánh giá
	best_path = output_dir / f"best_model_{MODEL_NAME}.pth"
	if best_path.exists():
		raw_model = model
		raw_model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
		print(f"\nĐã load checkpoint tốt nhất từ {best_path}")

	# Đánh giá trên tập Val và tập Test
	print("\nĐánh giá trên tập Validation...")
	evaluate_and_report(model, val_loader, device, class_names, output_dir, prefix="val")

	print("\nĐánh giá trên tập Test...")
	evaluate_and_report(model, test_loader, device, class_names, output_dir, prefix="test")

	# Lưu file metadata kết quả
	result = {
		"model_name": MODEL_NAME,
		"epochs": EPOCHS,
		"best_val_acc": history.get("best_val_acc", 0.0),
		"train_size": len(df_train),
		"val_size": len(df_val),
		"test_size": len(df_test),
	}
	with open(output_dir / "final_summary.json", "w", encoding="utf-8") as f:
		json.dump(result, f, indent=2, ensure_ascii=False)

	print(f"\n[Hoàn tất] Tất cả kết quả huấn luyện đã lưu tại thư mục: {output_dir}")


if __name__ == "__main__":
	main()
