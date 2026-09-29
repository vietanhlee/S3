"""
02_specimen_variant_gan.partitioning.governed_split
===================================================
Quy trình phân hoạch chuẩn governed (End Version Split):
- Kết hợp đa phương pháp tối ưu riêng biệt cho 18 loài Fabaceae thương mại.
- Trích xuất embedding kép (Swin Transformer + EfficientNetV2).
- Phân bổ mẫu vật có kiểm soát nghiêm ngặt.
"""

import os
from typing import Tuple, Dict
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import timm
from timm.data import resolve_data_config
from tqdm import tqdm

from utils import ImagePathDataset
from .base import _shuffle_df
from .heuristic_splits import (
	mahalanobis_fixed_split,
	mahalanobis_iterative_split,
	agglom_stratified_split,
)
from .graph_splits import (
	hierarchical_clustering_split,
	cosine_graph_split,
)
from .adversarial_and_kfold import (
	stratified_random_split,
	adversarial_validation_split,
	stratified_group_kfold_split,
)


# Cấu hình phân bổ đặc thù cho 18 loài trong bộ dữ liệu
# Format: (Giao thức PP, Target Split Mode, Backbone Embedding)
SPLIT_CONFIG: Dict[str, Tuple[str, str, str]] = {
	"Afzelia africana": ("PP9", "test", "eff"),
	"Afzelia bella": ("PP4", "val", "swin"),
	"Afzelia pachyloba": ("PP4", "val", "eff"),
	"Afzelia quanzensis": ("PP9", "test", "eff"),
	"Dalbergia cochinchinensis": ("PP9", "val", "swin"),
	"Dalbergia melanoxylon": ("PP2", "val", "eff"),
	"Dalbergia oliveri": ("PP1", "test", "swin"),
	"Dalbergia rimosa": ("PP8", "test", "swin"),
	"Dalbergia tonkinensis": ("PP7", "test", "swin"),
	"Guibourtia arnoldiana": ("PP4", "test", "eff"),
	"Guibourtia coleosperma": ("PP1", "test", "eff"),
	"Guibourtia ehie": ("PP5", "test", "swin"),
	"Pterocarpus erinaceus": ("PP2", "test", "swin"),
	"Pterocarpus indicus": ("PP9", "test", "eff"),
	"Pterocarpus macrocarpus": ("PP2", "test", "swin"),
	"Pterocarpus soyauxii": ("PP4", "test", "eff"),
	"Sindora cochinchinensis": ("PP8", "val", "swin"),
	"Sindora tonkinensis": ("PP7", "val", "swin"),
}


def compute_embeddings_v2(
	df: pd.DataFrame,
	model_name: str,
	batch_size: int,
	device: torch.device,
) -> np.ndarray:
	"""Trích xuất embeddings sử dụng backbone timm cụ thể."""
	print(f"  -> Khởi tạo mô hình trích xuất đặc trưng: {model_name}...")
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

	dataset = ImagePathDataset(df, transform=transform)
	num_workers = min(4, os.cpu_count() or 1)
	loader = DataLoader(
		dataset,
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
	cosine_threshold: float = 0.92,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	"""
	PP Chuẩn cuối (End Version Governed Split): Kết hợp nhiều phương pháp chia khác nhau
	cho từng loài dựa trên đặc trưng hình thái IAWA và phân bố mẫu vật vật lý.
	Loại bỏ hoàn toàn 'Pterocarpus sp' và 'Peltogyne pubescens' khỏi benchmark.
	"""
	keep_mask = ~df["label"].isin(["Pterocarpus sp", "Peltogyne pubescens"])
	df_filtered = df[keep_mask].reset_index(drop=True)
	emb_eff_filtered = embs_eff[keep_mask.values]
	emb_swin_filtered = embs_swin[keep_mask.values]

	pp_registry = {
		"PP1": mahalanobis_fixed_split,
		"PP2": mahalanobis_iterative_split,
		"PP4": hierarchical_clustering_split,
		"PP5": cosine_graph_split,
		"PP7": adversarial_validation_split,
		"PP8": stratified_group_kfold_split,
		"PP9": agglom_stratified_split,
	}

	train_idx_all = []
	val_idx_all = []
	test_idx_all = []

	for label, group in df_filtered.groupby("label"):
		indices = group.index.tolist()
		sub_df = group.copy()
		path_to_orig_idx = dict(zip(group["path"], group.index))
		sub_df_reset = sub_df.reset_index(drop=True)

		if label in SPLIT_CONFIG:
			pp_key, mode, model_type = SPLIT_CONFIG[label]
		else:
			print(f"[Warning] Class '{label}' không có trong SPLIT_CONFIG. Fallback dùng PP8.")
			pp_key, mode, model_type = "PP8", "test", "eff"

		sub_emb = emb_eff_filtered[indices] if model_type == "eff" else emb_swin_filtered[indices]
		split_fn = pp_registry.get(pp_key, stratified_group_kfold_split)

		try:
			if pp_key == "PP5":
				tr_df, val_df, te_df = split_fn(
					sub_df_reset,
					sub_emb,
					train_ratio=train_ratio,
					val_ratio=val_ratio,
					seed=seed,
					cosine_threshold=cosine_threshold,
				)
			else:
				tr_df, val_df, te_df = split_fn(
					sub_df_reset,
					sub_emb,
					train_ratio=train_ratio,
					val_ratio=val_ratio,
					seed=seed,
				)
		except Exception as e:
			print(f"[Error] Lỗi khi chia dữ liệu cho loài '{label}' bằng {pp_key}: {e}. Fallback stratified random.")
			tr_df, val_df, te_df = stratified_random_split(
				sub_df_reset,
				sub_emb,
				train_ratio=train_ratio,
				val_ratio=val_ratio,
				seed=seed,
			)

		tr_orig_idx = [path_to_orig_idx[p] for p in tr_df["path"]]
		val_orig_idx = [path_to_orig_idx[p] for p in val_df["path"]]
		te_orig_idx = [path_to_orig_idx[p] for p in te_df["path"]]

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
