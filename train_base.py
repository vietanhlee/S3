"""
train_base.py
=============
Base class (OOP) cho toàn bộ pipeline huấn luyện Metric Learning.
Các script huấn luyện cụ thể kế thừa từ BaseMetricTrainer và chỉ cần
định nghĩa hàm loss đặc trưng + cấu hình siêu tham số.

Pipeline chuẩn:
  1. Thu thập ảnh, lọc bỏ lớp Pterocarpus sp
  2. Trích xuất embeddings nền tảng (EfficientNetV2-M + Swin-Large) cho chia dữ liệu
  3. Chia dữ liệu theo End Version Split
  4. Khởi tạo DataLoader (PK Sampler hoặc Random Sampler)
  5. Khởi tạo Model (MetricModel), Loss, Optimizer
  6. Phân tích trước training (Grad-CAM, t-SNE baseline)
  7. Vòng lặp huấn luyện + đánh giá mỗi epoch
  8. Đánh giá cuối trên tập Val/Test (Retrieval + Clustering Metrics)
  9. Phân tích sau training (Grad-CAM, t-SNE, Distance)
"""

import os
import gc
import json
import random
from abc import ABC, abstractmethod
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

from utils import (
	set_seed, get_device,
	collect_image_samples, build_dataframe,
	compute_embeddings_v2, freeze_model_layers, summarize_model,
	log_split_summary, eda_split_class_distribution,
	extract_all_embeddings, evaluate_retrieval, format_retrieval_report,
	evaluate_cross_retrieval,
	select_gradcam_representatives, compute_class_prototypes,
	generate_gradcam_maps, plot_gradcam_comparison,
	plot_tsne_comparison, plot_distance_analysis,
	plot_metrics_summary, plot_all_metrics_per_epoch,
	evaluate_loss, build_transforms,
	CAM_METHODS,
)
from train_final import end_version_split
from split_methods import validate_split


# ============================================================
# Shared Dataset & PK Sampler
# ============================================================

class MetricImageDataset(Dataset):
	"""Dataset chuẩn cho Metric Learning: trả về (image, label_idx)."""

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
	"""PK Batch Sampler: mỗi batch chứa P classes × K samples/class.
	Đảm bảo duyệt qua toàn bộ ảnh trong tập dữ liệu không trùng lặp trong cùng một epoch,
	chỉ thực hiện lặp lại/tăng cường ảnh khi đã quét qua hết tất cả ảnh gốc của lớp đó.
	"""

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
		# Khởi tạo danh sách chỉ số được trộn ngẫu nhiên và con trỏ tương ứng cho mỗi lớp gỗ ở đầu mỗi epoch
		shuffled_indices = {
			lbl: random.sample(self.label_to_indices[lbl], len(self.label_to_indices[lbl]))
			for lbl in self.unique_labels
		}
		pointers = {lbl: 0 for lbl in self.unique_labels}

		for _ in range(self.n_batches):
			p_actual = min(self.p, len(self.unique_labels))
			selected_labels = random.sample(self.unique_labels, p_actual)
			batch = []
			for lbl in selected_labels:
				indices = shuffled_indices[lbl]
				ptr = pointers[lbl]

				# Nếu số ảnh chưa học của lớp này còn đủ để lấy K ảnh
				if len(indices) - ptr >= self.k:
					sampled = indices[ptr : ptr + self.k]
					pointers[lbl] += self.k
				else:
					# Lấy nốt số ảnh gốc còn lại chưa học của lớp đó
					sampled = indices[ptr:]
					# Trộn và lặp lại lấy ảnh cho đến khi đủ K mẫu
					while len(sampled) < self.k:
						random.shuffle(indices)
						take = min(self.k - len(sampled), len(indices))
						sampled.extend(indices[:take])
					pointers[lbl] = take
				
				batch.extend(sampled)
			yield batch

	def __len__(self) -> int:
		return self.n_batches


# ============================================================
# Shared Model
# ============================================================

class MetricModel(nn.Module):
	"""ConvNeXt-Tiny backbone + Projection Head → L2 Normalized embeddings."""

	def __init__(self, model_name: str = "convnext_tiny",
	             embedding_dim: int = 256, freeze_ratio: float = 0.90) -> None:
		super().__init__()
		self.backbone = timm.create_model(
			model_name, pretrained=True, num_classes=0, global_pool="avg",
		)
		freeze_model_layers(self.backbone, freeze_ratio)
		backbone_dim = self.backbone.num_features  # 768 cho convnext_tiny

		self.projector = nn.Sequential(
			nn.Linear(backbone_dim, backbone_dim),
			nn.BatchNorm1d(backbone_dim),
			nn.ReLU(inplace=True),
			nn.Linear(backbone_dim, embedding_dim),
		)
		self.model_name = model_name

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		features = self.backbone(x)
		embeddings = self.projector(features)
		return F.normalize(embeddings, p=2, dim=1)


# ============================================================
# Abstract Base Trainer
# ============================================================

class BaseMetricTrainer(ABC):
	"""Lớp trừu tượng cơ sở cho toàn bộ pipeline huấn luyện Metric Learning.

	Các lớp con chỉ cần implement 3 abstract methods:
	  - get_method_name()
	  - get_loss_config()
	  - build_loss(num_classes)

	Có thể override thêm:
	  - build_model()
	  - build_train_sampler(labels)
	  - build_train_dataset(df, class_to_idx, transform)
	  - train_one_epoch(...)
	  - get_batch_size()
	"""

	DEFAULT_CONFIG = {
		"ROOT_DIR":   r"/kaggle/input/datasets/b23dckh002lvitanh/s3-origin/S3",
		"OUTPUT_DIR": "outputs_metric",
		"TRAIN_RATIO": 0.7,
		"VAL_RATIO":   0.15,
		"SEED":        42,
		"P_CLASSES":   19,
		"K_SAMPLES":   20,
		"EPOCHS":      30,
		"PATIENCE":    10,
		"LR":          1e-4,
		"WEIGHT_DECAY": 1e-4,
		"EMBEDDING_DIM": 256,
		"FREEZE_RATIO":  0.90,
		"MODEL_NAME":    "convnext_tiny",
		"CALCULATE_CLUSTERING_METRICS": True,
		"EVAL_MODE":     "cross",
		"EMB_BATCH_SIZE": 128,
		"NUM_WORKERS":    4,
	}

	def __init__(self, config: dict | None = None) -> None:
		self.config = {**self.DEFAULT_CONFIG}
		if config:
			self.config.update(config)

	def init_wandb(self, method_name: str) -> None:
		"""Khởi tạo Weights & Biases bằng cách đọc WANDB_API_KEY từ file .env."""
		try:
			from dotenv import load_dotenv
			import wandb
			load_dotenv()
			api_key = os.getenv("WANDB_API_KEY")
			if not api_key:
				print("[Wandb Warning] WANDB_API_KEY không tồn tại trong .env. Chạy không có WandB.")
				self.use_wandb = False
				return

			wandb.login(key=api_key)
			model_prefix = self.config["MODEL_NAME"].lower().replace("_", "-")
			method_slug = method_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
			run_name = f"{model_prefix}-{method_slug}"
			# Khởi tạo project chung và đặt tên run theo phương pháp
			wandb.init(
				project="S3-Wood-Metric-Learning",
				name=run_name,
				config=self.config
			)
			self.use_wandb = True
			self.run_name = run_name
			print(f"[Wandb Info] Khởi tạo thành công: project='S3-Wood-Recognition', run='{run_name}'")
		except Exception as e:
			print(f"[Wandb Warning] Lỗi khởi tạo WandB: {e}. Chạy không có WandB.")
			self.use_wandb = False

	# ── Abstract Methods (BẮT BUỘC implement) ────────────────

	@abstractmethod
	def get_method_name(self) -> str:
		"""Trả về tên đầy đủ của phương pháp (ví dụ 'Multi-Similarity Loss')."""

	@abstractmethod
	def get_loss_config(self) -> dict:
		"""Trả về dict chứa siêu tham số đặc trưng của hàm loss."""

	@abstractmethod
	def build_loss(self, num_classes: int) -> nn.Module:
		"""Khởi tạo và trả về hàm loss. num_classes dùng cho các loss dạng classification."""

	# ── Overridable Methods (TÙY CHỌN override) ─────────────

	def build_model(self) -> nn.Module:
		"""Khởi tạo mô hình. Override cho ArcFace-style models."""
		return MetricModel(
			model_name=self.config["MODEL_NAME"],
			embedding_dim=self.config["EMBEDDING_DIM"],
			freeze_ratio=self.config["FREEZE_RATIO"],
		)

	def build_train_sampler(self, labels: list) -> Sampler | None:
		"""Khởi tạo sampler cho tập train.
		Trả về PKSampler cho pair/tuple loss, None cho classification loss (dùng shuffle)."""
		return PKSampler(labels, p=self.config["P_CLASSES"], k=self.config["K_SAMPLES"])

	def build_train_dataset(self, df: pd.DataFrame, class_to_idx: dict, transform) -> Dataset:
		"""Khởi tạo dataset cho tập train. Override cho SupCon (double view)."""
		return MetricImageDataset(df, class_to_idx, transform=transform)

	def get_batch_size(self) -> int:
		"""Trả về batch size hiệu dụng cho eval loaders."""
		return self.config["P_CLASSES"] * self.config["K_SAMPLES"]

	def train_one_epoch(self, model: nn.Module, loader: DataLoader,
	                    optimizer: torch.optim.Optimizer, criterion: nn.Module,
	                    device: torch.device, epoch: int, total_epochs: int) -> float:
		"""Huấn luyện một epoch. Override cho SupCon hoặc ArcFace nếu cần."""
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

	# ── Core Pipeline ────────────────────────────────────────

	def run(self) -> None:
		"""Thực thi toàn bộ pipeline huấn luyện."""
		cfg = self.config
		set_seed(cfg["SEED"])
		device = get_device()
		method_name = self.get_method_name()
		print(f"Device: {device}")
		print(f"Method: {method_name}")

		# Khởi tạo WandB
		self.init_wandb(method_name)

		output_dir = Path(cfg["OUTPUT_DIR"])
		output_dir.mkdir(parents=True, exist_ok=True)

		# ── 1. Thu thập ảnh ──────────────────────────────────
		print("\n[Step 1] Thu thập ảnh...")
		samples = collect_image_samples(cfg["ROOT_DIR"])
		if not samples:
			raise ValueError(f"Không tìm thấy ảnh nào trong {cfg['ROOT_DIR']}")

		df = build_dataframe(samples)
		EXCLUDED_CLASSES = ["Pterocarpus sp", "Peltogyne pubescens"]
		df_filtered = df[~df["label"].isin(EXCLUDED_CLASSES)].reset_index(drop=True)
		print(f"Tổng ảnh: {len(df_filtered)}, Số class: {df_filtered['label'].nunique()}")

		class_names = sorted(df_filtered["label"].unique().tolist())
		self.class_names = class_names
		num_classes = len(class_names)
		class_to_idx = {name: i for i, name in enumerate(class_names)}

		with open(output_dir / "class_indices.json", "w", encoding="utf-8") as f:
			json.dump(class_to_idx, f, indent=2, ensure_ascii=False)

		# ── 2. Compute embeddings cho chia dữ liệu ──────────
		print("\n[Step 2] Compute embeddings cho chia dữ liệu...")
		print("Trích xuất embeddings với EfficientNetV2-M...")
		embs_eff = compute_embeddings_v2(
			df_filtered, "tf_efficientnetv2_m_in21k",
			batch_size=cfg["EMB_BATCH_SIZE"], device=device,
		)
		print("Trích xuất embeddings với Swin-Large...")
		embs_swin = compute_embeddings_v2(
			df_filtered, "swin_large_patch4_window7_224",
			batch_size=cfg["EMB_BATCH_SIZE"], device=device,
		)

		# ── 3. Chia dữ liệu ─────────────────────────────────
		print("\n[Step 3] Chia dữ liệu theo End Version Split...")
		df_train, df_val, df_test = end_version_split(
			df_filtered, embs_eff, embs_swin,
			train_ratio=cfg["TRAIN_RATIO"], val_ratio=cfg["VAL_RATIO"], seed=cfg["SEED"],
		)
		validate_split(df_filtered, df_train, df_val, df_test, f"{method_name}_EndVersion")
		log_split_summary(df_filtered, df_train, df_val, df_test)
		eda_split_class_distribution(
			df_train, df_val, df_test,
			"End Version - Class Distribution",
			output_dir / "eda_split_end_version.png",
		)

		# Tính toán class weights nghịch đảo tần suất để xử lý mất cân bằng dữ liệu
		class_counts = df_train["label"].value_counts()
		weights = []
		for name in class_names:
			count = class_counts.get(name, 1)
			weights.append(1.0 / count)
		weights = torch.tensor(weights, dtype=torch.float32)
		self.class_weights = (weights / weights.sum() * num_classes).to(device)

		# ── 4. Transforms & DataLoaders ──────────────────────
		print("\n[Step 4] Chuẩn bị Dataset và DataLoader...")
		cfg_model = timm.create_model(cfg["MODEL_NAME"], pretrained=False, num_classes=0)
		data_cfg = resolve_data_config({}, model=cfg_model)
		img_size = data_cfg.get("input_size", (3, 224, 224))[-1]
		mean = data_cfg.get("mean", (0.485, 0.456, 0.406))
		std = data_cfg.get("std", (0.229, 0.224, 0.225))
		train_tf, eval_tf = build_transforms(img_size, mean, std)

		train_ds = self.build_train_dataset(df_train, class_to_idx, train_tf)
		val_ds = MetricImageDataset(df_val, class_to_idx, transform=eval_tf)
		test_ds = MetricImageDataset(df_test, class_to_idx, transform=eval_tf)

		batch_size = self.get_batch_size()
		train_sampler = self.build_train_sampler(train_ds.labels)

		if train_sampler is not None:
			train_loader = DataLoader(
				train_ds, batch_sampler=train_sampler,
				num_workers=cfg["NUM_WORKERS"], pin_memory=True,
			)
		else:
			train_loader = DataLoader(
				train_ds, batch_size=batch_size, shuffle=True,
				num_workers=cfg["NUM_WORKERS"], pin_memory=True,
			)

		val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
		                        num_workers=cfg["NUM_WORKERS"], pin_memory=True)
		test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
		                         num_workers=cfg["NUM_WORKERS"], pin_memory=True)

		# Train eval loader (for retrieval on train set — always uses MetricImageDataset)
		train_eval_ds = MetricImageDataset(df_train, class_to_idx, transform=eval_tf)
		train_eval_loader = DataLoader(train_eval_ds, batch_size=batch_size, shuffle=False,
		                               num_workers=cfg["NUM_WORKERS"], pin_memory=True)

		print(f"Train: {len(train_ds)} ảnh | Val: {len(val_ds)} ảnh | Test: {len(test_ds)} ảnh")

		# ── 5. Model, Loss, Optimizer ────────────────────────
		print("\n[Step 5] Khởi tạo model và optimizer...")
		model = self.build_model().to(device)

		model_info = summarize_model(model)
		print(
			f"Model: {cfg['MODEL_NAME']}_{method_name}, "
			f"total={model_info['total_params']:,}, "
			f"trainable={model_info['trainable_params']:,}, "
			f"frozen={model_info['frozen_params']:,}"
		)

		criterion = self.build_loss(num_classes)
		if isinstance(criterion, nn.Module):
			criterion = criterion.to(device)

		# Thu thập toàn bộ tham số có thể huấn luyện (model + criterion nếu có)
		params = list(filter(lambda p: p.requires_grad, model.parameters()))
		if isinstance(criterion, nn.Module) and any(p.requires_grad for p in criterion.parameters()):
			params += list(filter(lambda p: p.requires_grad, criterion.parameters()))

		optimizer = torch.optim.AdamW(params, lr=cfg["LR"], weight_decay=cfg["WEIGHT_DECAY"])
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
			optimizer, T_max=cfg["EPOCHS"], eta_min=1e-6,
		)

		# ── Pre-training Analysis ────────────────────────────
		print("\n[Analysis] Trích xuất đặc trưng và Grad-CAM trước training...")
		
		# 1. Chọn 5 ảnh không trùng lặp cho mỗi loài trong số 4 loài Afzelia
		afzelia_classes = ["Afzelia africana", "Afzelia bella", "Afzelia pachyloba", "Afzelia quanzensis"]
		reps_by_class = {cls: [] for cls in afzelia_classes}
		for cls in afzelia_classes:
			cls_df_val = df_val[df_val["label"] == cls]
			val_paths = cls_df_val["path"].tolist()
			sampled_paths = []
			if len(val_paths) >= 5:
				sampled_paths = random.sample(val_paths, 5)
			else:
				sampled_paths = list(val_paths)
				cls_df_test = df_test[df_test["label"] == cls]
				test_paths = [p for p in cls_df_test["path"].tolist() if p not in sampled_paths]
				needed = 5 - len(sampled_paths)
				if len(test_paths) >= needed:
					sampled_paths.extend(random.sample(test_paths, needed))
				else:
					sampled_paths.extend(test_paths)
					cls_df_train = df_train[df_train["label"] == cls]
					train_paths = [p for p in cls_df_train["path"].tolist() if p not in sampled_paths]
					needed = 5 - len(sampled_paths)
					if len(train_paths) >= needed:
						sampled_paths.extend(random.sample(train_paths, needed))
					else:
						sampled_paths.extend(train_paths)
			for path in sampled_paths:
				reps_by_class[cls].append({
					"path": path,
					"label": cls,
					"species": cls.replace("Afzelia ", "")
				})
		
		reps_flat = []
		for i in range(5):
			for cls in afzelia_classes:
				if i < len(reps_by_class[cls]):
					reps_flat.append(reps_by_class[cls][i])

		before_protos = compute_class_prototypes(model, train_eval_loader, device, num_classes)
		before_cams_dict = {}
		for cam_method in CAM_METHODS:
			before_cams_dict[cam_method] = generate_gradcam_maps(
				model, reps_flat, before_protos, class_to_idx, eval_tf, device, method=cam_method,
			)

		print("  Trích xuất đặc trưng của tập Test trước training...")
		before_test_embs, test_labels = extract_all_embeddings(model, test_loader, device)
		before_test_embs = before_test_embs.numpy()

		# ── Training Loop ────────────────────────────────────
		print(f"\n[Training] Bắt đầu huấn luyện {method_name}...")
		eval_mode = cfg["EVAL_MODE"]
		calc_clustering = cfg["CALCULATE_CLUSTERING_METRICS"]

		history = self._init_history(eval_mode, calc_clustering)
		best_map = 0.0
		best_stopping_metric = 0.0
		epochs_no_improve = 0

		for epoch in range(1, cfg["EPOCHS"] + 1):
			loss = self.train_one_epoch(
				model, train_loader, optimizer, criterion, device, epoch, cfg["EPOCHS"],
			)
			val_loss = evaluate_loss(model, val_loader, criterion, device)
			train_results = evaluate_retrieval(
				model, train_eval_loader, device, class_names, eval_clustering=False,
			)

			# Ghi history cơ bản
			history["train_loss"].append(loss)
			history["val_loss"].append(val_loss)
			for metric_key, hist_key in [
				("Recall@1", "train_recall1"), ("Recall@5", "train_recall5"),
				("Precision@1", "train_precision1"), ("Precision@5", "train_precision5"),
				("mAP", "train_map"), ("AUC", "train_auc"),
			]:
				history[hist_key].append(train_results[metric_key])

			# Đánh giá val (self/cross) theo cấu hình
			val_results, val_cross_results = self._evaluate_val_epoch(
				model, val_loader, train_eval_loader, device, class_names,
				eval_mode, calc_clustering, history,
			)

			# Log
			self._log_epoch(
				epoch, cfg["EPOCHS"], loss, val_loss, train_results,
				val_results, val_cross_results, eval_mode, calc_clustering,
				optimizer.param_groups[0]["lr"],
			)

			# Log to WandB
			if getattr(self, "use_wandb", False):
				try:
					import wandb
					metrics_to_log = {
						"epoch": epoch,
						"train/loss": loss,
						"val/loss": val_loss,
						"lr": optimizer.param_groups[0]["lr"],
					}
					for k, v in train_results.items():
						metrics_to_log[f"train/{k}"] = v
					
					def _calc_h_mean(vals, eps=0.01):
						if not vals:
							return 0.0
						return len(vals) / sum(1.0 / (val + eps) for val in vals)

					if val_results is not None:
						for k, v in val_results.items():
							if k in ["Recall@1", "Recall@5", "Precision@1", "Precision@5", "mAP", "AUC"]:
								if eval_mode in ("self", "both"):
									metrics_to_log[f"val_self/{k}"] = v
							elif k not in ["per_class_recall1", "per_class_recall5", "per_class_precision1", "per_class_precision5", "per_class_map", "per_class_auc"]:
								metrics_to_log[f"val_clustering/{k}"] = v
						
						# Log Harmonic metrics cho val self
						if eval_mode in ("self", "both"):
							for k, class_key in [
								("Recall@1", "per_class_recall1"),
								("Recall@5", "per_class_recall5"),
								("Precision@1", "per_class_precision1"),
								("Precision@5", "per_class_precision5"),
								("mAP", "per_class_map"),
								("AUC", "per_class_auc")
							]:
								if class_key in val_results:
									metrics_to_log[f"val_self_harmonic/{k}"] = _calc_h_mean(val_results[class_key])

					if val_cross_results is not None:
						for k, v in val_cross_results.items():
							if k in ["Recall@1", "Recall@5", "Precision@1", "Precision@5", "mAP", "AUC"]:
								metrics_to_log[f"val_cross/{k}"] = v
						
						# Log Harmonic metrics cho val cross
						for k, class_key in [
							("Recall@1", "per_class_recall1"),
							("Recall@5", "per_class_recall5"),
							("Precision@1", "per_class_precision1"),
							("Precision@5", "per_class_precision5"),
							("mAP", "per_class_map"),
							("AUC", "per_class_auc")
						]:
							if class_key in val_cross_results:
								metrics_to_log[f"val_cross_harmonic/{k}"] = _calc_h_mean(val_cross_results[class_key])

					wandb.log(metrics_to_log, step=epoch)
				except Exception as e:
					print(f"[Wandb Warning] Lỗi khi log metrics: {e}")

			scheduler.step()

			# Early Stopping dựa trên Harmonic Mean của mAP các lớp để tối ưu hóa lớp khó
			val_results_dict = val_cross_results if val_cross_results is not None else val_results
			if val_results_dict is not None and "per_class_map" in val_results_dict:
				per_class_map = val_results_dict["per_class_map"]
				eps = 0.01
				inv_sum = sum(1.0 / (m + eps) for m in per_class_map)
				val_stopping_metric = len(per_class_map) / inv_sum
				macro_map = val_results_dict["mAP"]
			else:
				val_stopping_metric = (
					val_cross_results["mAP"] if val_cross_results is not None
					else (val_results["mAP"] if val_results is not None else 0.0)
				)
				macro_map = val_stopping_metric

			if val_stopping_metric > best_stopping_metric:
				best_stopping_metric = val_stopping_metric
				best_map = macro_map
				epochs_no_improve = 0
				torch.save(model.state_dict(), output_dir / "best_model.pth")
				print(f"  → Saved best model (Harmonic mAP={val_stopping_metric * 100:.2f}%, Macro mAP={macro_map * 100:.2f}%)")
			else:
				epochs_no_improve += 1

			if epochs_no_improve >= cfg["PATIENCE"]:
				print(f"\nEarly stopping tại epoch {epoch} (patience={cfg['PATIENCE']})")
				break

		# ── Load Best & Final Evaluation ─────────────────────
		best_path = output_dir / "best_model.pth"
		if best_path.exists():
			model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
			print(f"\nĐã load best model từ {best_path}")

		self._run_final_evaluation(
			model, val_loader, test_loader, train_eval_loader,
			device, class_names, output_dir,
		)

		# ── Post-training Analysis ───────────────────────────
		self._run_post_training_analysis(
			model, train_eval_loader, test_loader, val_loader,
			device, class_names, num_classes, class_to_idx,
			reps_by_class, reps_flat, before_cams_dict,
			before_test_embs, test_labels, eval_tf, history, output_dir,
		)

		# ── Save Summary ─────────────────────────────────────
		summary = {
			"method": method_name,
			"model": cfg["MODEL_NAME"],
			"embedding_dim": cfg["EMBEDDING_DIM"],
			**self.get_loss_config(),
			"p_classes": cfg["P_CLASSES"],
			"k_samples": cfg["K_SAMPLES"],
			"batch_size": batch_size,
			"epochs_trained": len(history["train_loss"]),
			"best_val_map": best_map,
			"train_size": len(df_train),
			"val_size": len(df_val),
			"test_size": len(df_test),
		}
		with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
			json.dump(summary, f, indent=2, ensure_ascii=False)

		# ── Upload artifacts lên WandB ───────────────────────
		if getattr(self, "use_wandb", False):
			try:
				import wandb
				artifact_name = f"{self.run_name.replace('_', '-')}-artifacts"
				artifact = wandb.Artifact(name=artifact_name, type="model_and_reports")
				
				# Log best model nếu có
				best_path = output_dir / "best_model.pth"
				if best_path.exists():
					artifact.add_file(str(best_path), name="best_model.pth")
					
				# Log các file report text
				for txt_file in output_dir.glob("*.txt"):
					artifact.add_file(str(txt_file), name=txt_file.name)
				# Log file summary JSON
				summary_json = output_dir / "summary.json"
				if summary_json.exists():
					artifact.add_file(str(summary_json), name="summary.json")
					
				wandb.log_artifact(artifact)
				print(f"[Wandb Info] Đã upload artifact '{artifact_name}' thành công.")
			except Exception as e:
				print(f"[Wandb Warning] Lỗi khi upload artifacts: {e}")
			finally:
				wandb.finish()

		# ── Cleanup ──────────────────────────────────────────
		del model, optimizer, criterion
		del train_loader, val_loader, test_loader, train_eval_loader
		del train_ds, val_ds, test_ds, train_eval_ds
		gc.collect()
		if device.type == "cuda":
			torch.cuda.empty_cache()

		print(f"\n[Hoàn tất] Tất cả kết quả lưu tại: {output_dir}/")

	# ── Private Helper Methods ───────────────────────────────

	@staticmethod
	def _init_history(eval_mode: str, calc_clustering: bool) -> dict:
		"""Khởi tạo dict history phù hợp với EVAL_MODE."""
		history = {
			"train_loss": [], "val_loss": [],
			"train_recall1": [], "train_recall5": [],
			"train_precision1": [], "train_precision5": [],
			"train_map": [], "train_auc": [],
		}
		if eval_mode in ("self", "both"):
			history.update({
				"val_recall1": [], "val_recall5": [],
				"val_precision1": [], "val_precision5": [],
				"val_map": [], "val_auc": [],
			})
		if calc_clustering:
			history.update({
				"val_silhouette": [], "val_dbi": [], "val_chi": [],
				"val_dunn": [], "val_nmi": [], "val_ratio": [],
			})
		if eval_mode in ("cross", "both"):
			history.update({
				"val_cross_recall1": [], "val_cross_recall5": [],
				"val_cross_precision1": [], "val_cross_precision5": [],
				"val_cross_map": [], "val_cross_auc": [],
			})
		return history

	@staticmethod
	def _evaluate_val_epoch(model, val_loader, train_eval_loader, device, class_names,
	                        eval_mode, calc_clustering, history):
		"""Đánh giá retrieval trên tập Validation theo EVAL_MODE."""
		val_results = None
		val_cross_results = None

		if (eval_mode in ("self", "both")) or calc_clustering:
			val_results = evaluate_retrieval(
				model, val_loader, device, class_names, eval_clustering=calc_clustering,
			)
			if eval_mode in ("self", "both"):
				for metric_key, hist_key in [
					("Recall@1", "val_recall1"), ("Recall@5", "val_recall5"),
					("Precision@1", "val_precision1"), ("Precision@5", "val_precision5"),
					("mAP", "val_map"), ("AUC", "val_auc"),
				]:
					history[hist_key].append(val_results[metric_key])
			if calc_clustering:
				history["val_silhouette"].append(val_results["Silhouette"])
				history["val_dbi"].append(val_results["Davies-Bouldin"])
				history["val_chi"].append(val_results["Calinski-Harabasz"])
				history["val_dunn"].append(val_results["Dunn-Index"])
				history["val_nmi"].append(val_results["NMI"])
				history["val_ratio"].append(val_results["Intra-Inter-Ratio"])

		if eval_mode in ("cross", "both"):
			val_cross_results = evaluate_cross_retrieval(
				model, val_loader, train_eval_loader, device, class_names,
			)
			for metric_key, hist_key in [
				("Recall@1", "val_cross_recall1"), ("Recall@5", "val_cross_recall5"),
				("Precision@1", "val_cross_precision1"), ("Precision@5", "val_cross_precision5"),
				("mAP", "val_cross_map"), ("AUC", "val_cross_auc"),
			]:
				history[hist_key].append(val_cross_results[metric_key])

		return val_results, val_cross_results

	@staticmethod
	def _log_epoch(epoch, total_epochs, loss, val_loss, train_results,
	               val_results, val_cross_results, eval_mode, calc_clustering, lr):
		"""In log chi tiết mỗi epoch."""
		log = f"Epoch {epoch}/{total_epochs} —\n"
		log += f"  Loss (Train/Val): {loss:.4f} / {val_loss:.4f}\n"
		log += (
			f"  Train: R@1: {train_results['Recall@1'] * 100:.2f}% | "
			f"R@5: {train_results['Recall@5'] * 100:.2f}% | "
			f"P@1: {train_results['Precision@1'] * 100:.2f}% | "
			f"mAP: {train_results['mAP'] * 100:.2f}% | "
			f"AUC: {train_results['AUC']:.4f}\n"
		)
		if val_results is not None and eval_mode in ("self", "both"):
			log += (
				f"  Val Self: R@1: {val_results['Recall@1'] * 100:.2f}% | "
				f"R@5: {val_results['Recall@5'] * 100:.2f}% | "
				f"mAP: {val_results['mAP'] * 100:.2f}% | "
				f"AUC: {val_results['AUC']:.4f}\n"
			)
		if val_results is not None and calc_clustering:
			log += (
				f"  Val Clustering: Sil: {val_results['Silhouette']:.4f} | "
				f"DBI: {val_results['Davies-Bouldin']:.4f} | "
				f"CHI: {val_results['Calinski-Harabasz']:.2f} | "
				f"Dunn: {val_results['Dunn-Index']:.4f} | "
				f"NMI: {val_results['NMI']:.4f} | "
				f"Ratio: {val_results['Intra-Inter-Ratio']:.4f}\n"
			)
		if val_cross_results is not None:
			log += (
				f"  Val Cross: R@1: {val_cross_results['Recall@1'] * 100:.2f}% | "
				f"R@5: {val_cross_results['Recall@5'] * 100:.2f}% | "
				f"mAP: {val_cross_results['mAP'] * 100:.2f}% | "
				f"AUC: {val_cross_results['AUC']:.4f}\n"
			)
		log += f"  LR: {lr:.6f}"
		print(log)

	def _run_final_evaluation(self, model, val_loader, test_loader,
	                          train_eval_loader, device, class_names, output_dir):
		"""Đánh giá cuối cùng trên tập Val và Test."""
		eval_mode = self.config["EVAL_MODE"]

		if eval_mode in ("self", "both"):
			print(f"\n[Đánh giá cuối - Test Self]")
			results = evaluate_retrieval(model, test_loader, device, class_names, eval_clustering=True)
			report = format_retrieval_report(results, class_names, prefix="Test")
			print(report)
			print(
				f"  [Clustering - Test] Sil: {results['Silhouette']:.4f} | "
				f"DBI: {results['Davies-Bouldin']:.4f} | "
				f"CHI: {results['Calinski-Harabasz']:.2f} | "
				f"Dunn: {results['Dunn-Index']:.4f} | "
				f"NMI: {results['NMI']:.4f} | "
				f"Ratio: {results['Intra-Inter-Ratio']:.4f}"
			)
			with open(output_dir / "retrieval_report_test.txt", "w", encoding="utf-8") as f:
				f.write(report)

			# Log to WandB Summary & History (để vẽ biểu đồ cột so sánh giữa các run)
			if getattr(self, "use_wandb", False):
				try:
					import wandb
					if wandb.run is not None:
						log_dict = {}
						for k, v in results.items():
							if k in ["Recall@1", "Recall@5", "Precision@1", "Precision@5", "mAP", "AUC"]:
								wandb.run.summary[f"final_test_self/{k}"] = v
								log_dict[f"final_test_self/{k}"] = v
							else:
								wandb.run.summary[f"final_test_clustering/{k}"] = v
								log_dict[f"final_test_clustering/{k}"] = v
						wandb.log(log_dict)
				except Exception as e:
					print(f"[Wandb Warning] Lỗi khi log final evaluation self: {e}")

		if eval_mode in ("cross", "both"):
			print(f"\n[Đánh giá chéo - Test Query vs Train Gallery]")
			results = evaluate_cross_retrieval(model, test_loader, train_eval_loader, device, class_names)
			report = format_retrieval_report(results, class_names, prefix="Test Query vs Train Gallery")
			print(report)
			print(
				f"  [Cross - Test→Train] mAP: {results['mAP'] * 100:.2f}% | "
				f"AUC: {results['AUC']:.4f} | "
				f"R@1: {results['Recall@1'] * 100:.2f}% | "
				f"R@5: {results['Recall@5'] * 100:.2f}%"
			)
			with open(output_dir / "retrieval_report_test_cross.txt", "w", encoding="utf-8") as f:
				f.write(report)

			# Log to WandB Summary & History (để vẽ biểu đồ cột so sánh giữa các run)
			if getattr(self, "use_wandb", False):
				try:
					import wandb
					if wandb.run is not None:
						log_dict = {}
						for k, v in results.items():
							wandb.run.summary[f"final_test_cross/{k}"] = v
							log_dict[f"final_test_cross/{k}"] = v
						wandb.log(log_dict)
				except Exception as e:
					print(f"[Wandb Warning] Lỗi khi log final evaluation cross: {e}")

	def _run_post_training_analysis(self, model, train_eval_loader, test_loader,
	                                val_loader, device, class_names, num_classes,
	                                class_to_idx, reps_by_class, reps_flat,
	                                before_cams_dict, before_test_embs, test_labels,
	                                eval_tf, history, output_dir):
		"""Sinh trực quan hóa sau training: Grad-CAM, t-SNE, Distance, Metrics."""
		print("\n[Analysis] Trích xuất đặc trưng và sinh Grad-CAM sau training...")
		after_protos = compute_class_prototypes(model, train_eval_loader, device, num_classes)

		afzelia_classes = ["Afzelia africana", "Afzelia bella", "Afzelia pachyloba", "Afzelia quanzensis"]
		for cam_method in CAM_METHODS:
			after_cams = generate_gradcam_maps(
				model, reps_flat, after_protos, class_to_idx, eval_tf, device, method=cam_method,
			)
			before_cams = before_cams_dict[cam_method]
			
			# Vẽ 5 bức hình, mỗi bức hình gồm 4 loài Afzelia
			for i in range(5):
				fig_reps = []
				fig_before_cams = []
				fig_after_cams = []
				for cls_idx, cls in enumerate(afzelia_classes):
					flat_idx = i * len(afzelia_classes) + cls_idx
					if flat_idx < len(reps_flat):
						fig_reps.append(reps_flat[flat_idx])
						fig_before_cams.append(before_cams[flat_idx])
						fig_after_cams.append(after_cams[flat_idx])
				
				if len(fig_reps) > 0:
					out_path = output_dir / f"gradcam_afzelia_pic_{i+1}_{cam_method}.png"
					plot_gradcam_comparison(
						fig_reps, fig_before_cams, fig_after_cams,
						f"Afzelia Group {i+1} ({cam_method})",
						out_path,
					)
					if getattr(self, "use_wandb", False):
						try:
							import wandb
							if wandb.run is not None:
								wandb.log({f"Analysis/GradCAM_Group_{i+1}_{cam_method}": wandb.Image(str(out_path))})
						except Exception:
							pass

		print("  Trích xuất đặc trưng của tập Test sau training...")
		after_test_embs, _ = extract_all_embeddings(model, test_loader, device)
		after_test_embs = after_test_embs.numpy()

		tsne_path = output_dir / "tsne_comparison.png"
		plot_tsne_comparison(
			before_test_embs, after_test_embs, test_labels, class_names,
			tsne_path,
		)
		if getattr(self, "use_wandb", False):
			try:
				import wandb
				if wandb.run is not None:
					wandb.log({"Analysis/t-SNE_Comparison": wandb.Image(str(tsne_path))})
			except Exception:
				pass

		# Vẽ t-SNE cho từng Chi (Genus) trên tập Test
		genera = sorted(list(set(cls_name.split()[0] for cls_name in class_names)))
		for genus in genera:
			genus_class_indices = [idx for idx, name in enumerate(class_names) if name.startswith(genus)]
			mask = np.isin(test_labels, genus_class_indices)
			if np.sum(mask) >= 5:
				genus_before = before_test_embs[mask]
				genus_after = after_test_embs[mask]
				genus_labels = test_labels[mask]
				
				genus_tsne_path = output_dir / f"tsne_{genus.lower()}.png"
				plot_tsne_comparison(
					genus_before, genus_after, genus_labels, class_names,
					genus_tsne_path,
				)
				if getattr(self, "use_wandb", False):
					try:
						import wandb
						if wandb.run is not None:
							wandb.log({f"Analysis/t-SNE_{genus}": wandb.Image(str(genus_tsne_path))})
					except Exception:
						pass

		dist_path = output_dir / "distance_distribution.png"
		plot_distance_analysis(
			before_test_embs, after_test_embs, test_labels,
			dist_path,
		)
		if getattr(self, "use_wandb", False):
			try:
				import wandb
				if wandb.run is not None:
					wandb.log({"Analysis/Distance_Distribution": wandb.Image(str(dist_path))})
			except Exception:
				pass

		plot_metrics_summary(history, model, val_loader, test_loader, device, output_dir)
		plot_all_metrics_per_epoch(history, output_dir)
