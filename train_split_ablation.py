"""
train_split_ablation.py — So sánh Random Split vs End Version Split
====================================================================
Chứng minh 3 luận điểm:
  1. Random Split overestimate performance (accuracy cao giả tạo)
  2. End Version Split tách tốt hơn (cosine similarity train↔test thấp hơn)
  3. Histogram phân phối khoảng cách embedding minh họa sự khác biệt

Chạy:
  python train_split_ablation.py
"""

import os
import gc
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

import timm
from timm.data import resolve_data_config
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Import từ project ──
from utils import (
	set_seed, get_device,
	collect_image_samples, build_dataframe,
	ImageListDataset, build_transforms,
	freeze_model_layers, summarize_model,
	validate_split_minimums,
)
from train_final import (
	FocalLoss, accuracy_from_logits,
	train_model, build_model,
	compute_embeddings_v2, end_version_split,
	collect_predictions, SPLIT_CONFIG,
)
from split_methods import stratified_random_split, validate_split, compute_split_counts


# ===== CẤU HÌNH =====
ROOT_DIR       = r"/kaggle/input/datasets/b23dckh002lvitanh/s3-origin/S3"
OUTPUT_BASE    = "outputs_split_ablation"
TRAIN_RATIO    = 0.60
VAL_RATIO      = 0.20
BATCH_SIZE     = 128
EPOCHS         = 20
PATIENCE       = 50           # Đảm bảo chạy đủ 20 epochs
LR             = 5e-4
WEIGHT_DECAY   = 1e-2
FOCAL_GAMMA    = 2.0
FOCAL_ALPHA    = 0.25
MODEL_NAME     = "convnext_tiny"
FREEZE_RATIO   = 0.90
COSINE_THRESHOLD = 0.92
SEEDS          = [42, 123, 456]  # 3 random seeds
NUM_WORKERS    = 4
# =====================


def random_split_wrapper(
	df: pd.DataFrame,
	embs_eff: np.ndarray,   # Không dùng, giữ signature nhất quán
	embs_swin: np.ndarray,  # Không dùng
	train_ratio: float = 0.60,
	val_ratio: float = 0.20,
	seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	"""Wrapper cho subfolder-based random split (với fallback sang image-level random split)."""
	# Loại bỏ Pterocarpus sp và Peltogyne pubescens trước
	keep_mask = ~df["label"].isin(["Pterocarpus sp", "Peltogyne pubescens"])
	df_filtered = df[keep_mask].reset_index(drop=True)
	
	rng = random.Random(seed)
	train_idx, val_idx, test_idx = [], [], []

	for label, group in df_filtered.groupby("label"):
		subfolder_groups = group.groupby("subfolder")
		subfolder_names = list(subfolder_groups.groups.keys())
		rng.shuffle(subfolder_names)

		indices = sorted(group.index.tolist())
		n_total = len(indices)

		if len(subfolder_names) < 3:
			# Split ở mức ảnh ngẫu nhiên
			rng.shuffle(indices)
			train_count, val_count, test_count = compute_split_counts(n_total, train_ratio, val_ratio)
			train_idx.extend(indices[:train_count])
			val_idx.extend(indices[train_count:train_count + val_count])
			test_idx.extend(indices[train_count + val_count:])
		else:
			# Split ở mức subfolder ngẫu nhiên
			target_train = int(n_total * train_ratio)
			target_val = int(n_total * val_ratio)

			# Đảm bảo mỗi tập có ít nhất 1 subfolder trước
			test_idx.extend(subfolder_groups.get_group(subfolder_names[0]).index.tolist())
			val_idx.extend(subfolder_groups.get_group(subfolder_names[1]).index.tolist())

			curr_train = 0
			curr_val = len(subfolder_groups.get_group(subfolder_names[1]).index.tolist())
			curr_test = len(subfolder_groups.get_group(subfolder_names[0]).index.tolist())

			for sf_name in subfolder_names[2:]:
				sf_indices = subfolder_groups.get_group(sf_name).index.tolist()
				sf_count = len(sf_indices)

				if curr_train < target_train:
					train_idx.extend(sf_indices)
					curr_train += sf_count
				elif curr_val < target_val:
					val_idx.extend(sf_indices)
					curr_val += sf_count
				else:
					test_idx.extend(sf_indices)
					curr_test += sf_count

	df_train = df_filtered.loc[train_idx].sample(frac=1, random_state=seed).reset_index(drop=True)
	df_val = df_filtered.loc[val_idx].sample(frac=1, random_state=seed).reset_index(drop=True)
	df_test = df_filtered.loc[test_idx].sample(frac=1, random_state=seed).reset_index(drop=True)
	return df_train, df_val, df_test


@torch.no_grad()
def extract_penultimate_embeddings(
	model: nn.Module,
	loader: DataLoader,
	device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
	"""Trích xuất embedding từ lớp trước classifier (penultimate layer) của model classification.
	
	Trả về: (embeddings, labels) dưới dạng numpy arrays.
	"""
	model.eval()
	all_embs = []
	all_labels = []

	# Tách backbone (bỏ head) để lấy feature vector
	backbone = timm.create_model(MODEL_NAME, pretrained=False, num_classes=0, global_pool="avg")
	# Copy weights từ model đã train
	state_dict = model.state_dict()
	backbone_state = {}
	for k, v in state_dict.items():
		# Loại bỏ các key liên quan đến head (classifier)
		if not any(skip in k for skip in ["head", "fc", "classifier"]):
			backbone_state[k] = v
	backbone.load_state_dict(backbone_state, strict=False)
	backbone = backbone.to(device)
	backbone.eval()

	for images, labels in tqdm(loader, desc="Extract embeddings"):
		images = images.to(device)
		features = backbone(images)  # (B, D)
		features = F.normalize(features, p=2, dim=1)
		all_embs.append(features.cpu().numpy())
		all_labels.append(labels.numpy() if isinstance(labels, torch.Tensor) else np.array(labels))

	del backbone
	if device.type == "cuda":
		torch.cuda.empty_cache()

	return np.concatenate(all_embs), np.concatenate(all_labels)


def compute_split_similarity(
	train_embs: np.ndarray,
	test_embs: np.ndarray,
) -> dict:
	"""Tính cosine similarity trung bình và các thống kê giữa train và test embeddings."""
	# Subsample nếu quá lớn (tránh OOM khi tính full matrix)
	max_samples = 2000
	if len(train_embs) > max_samples:
		idx = np.random.choice(len(train_embs), max_samples, replace=False)
		train_sub = train_embs[idx]
	else:
		train_sub = train_embs
	if len(test_embs) > max_samples:
		idx = np.random.choice(len(test_embs), max_samples, replace=False)
		test_sub = test_embs[idx]
	else:
		test_sub = test_embs

	# Cosine similarity matrix: (n_test, n_train)
	sim_matrix = cosine_similarity(test_sub, train_sub)

	# Với mỗi ảnh test, lấy max similarity với tập train
	max_sims = sim_matrix.max(axis=1)
	mean_sims = sim_matrix.mean(axis=1)

	return {
		"mean_cosine_sim": float(np.mean(sim_matrix)),
		"median_cosine_sim": float(np.median(sim_matrix)),
		"max_cosine_sim_per_test_mean": float(np.mean(max_sims)),
		"max_cosine_sim_per_test_std": float(np.std(max_sims)),
		"mean_cosine_sim_per_test_mean": float(np.mean(mean_sims)),
		"all_sims": sim_matrix.flatten(),
		"max_sims": max_sims,
	}


def plot_similarity_histograms(
	random_sims: dict,
	endver_sims: dict,
	output_path: Path,
) -> None:
	"""Vẽ histogram so sánh phân phối cosine similarity train↔test."""
	fig, axes = plt.subplots(1, 2, figsize=(16, 6))

	# Plot 1: Phân phối tất cả cosine similarities
	ax1 = axes[0]
	ax1.hist(random_sims["all_sims"], bins=80, alpha=0.6, label="Random Split", color="#e74c3c", density=True)
	ax1.hist(endver_sims["all_sims"], bins=80, alpha=0.6, label="End Version Split", color="#2ecc71", density=True)
	ax1.axvline(random_sims["mean_cosine_sim"], color="#c0392b", linestyle="--", linewidth=2,
	            label=f"Random mean={random_sims['mean_cosine_sim']:.4f}")
	ax1.axvline(endver_sims["mean_cosine_sim"], color="#27ae60", linestyle="--", linewidth=2,
	            label=f"EndVer mean={endver_sims['mean_cosine_sim']:.4f}")
	ax1.set_xlabel("Cosine Similarity (Train ↔ Test)", fontsize=12)
	ax1.set_ylabel("Density", fontsize=12)
	ax1.set_title("All Pairwise Cosine Similarities", fontsize=13, fontweight="bold")
	ax1.legend(fontsize=9)
	ax1.grid(alpha=0.3)

	# Plot 2: Max similarity per test sample (đo data leakage)
	ax2 = axes[1]
	ax2.hist(random_sims["max_sims"], bins=50, alpha=0.6, label="Random Split", color="#e74c3c", density=True)
	ax2.hist(endver_sims["max_sims"], bins=50, alpha=0.6, label="End Version Split", color="#2ecc71", density=True)
	ax2.axvline(random_sims["max_cosine_sim_per_test_mean"], color="#c0392b", linestyle="--", linewidth=2,
	            label=f"Random mean={random_sims['max_cosine_sim_per_test_mean']:.4f}")
	ax2.axvline(endver_sims["max_cosine_sim_per_test_mean"], color="#27ae60", linestyle="--", linewidth=2,
	            label=f"EndVer mean={endver_sims['max_cosine_sim_per_test_mean']:.4f}")
	ax2.set_xlabel("Max Cosine Similarity (nearest train neighbor)", fontsize=12)
	ax2.set_ylabel("Density", fontsize=12)
	ax2.set_title("Nearest-Neighbor Similarity per Test Sample\n(↑ = higher data leakage risk)", fontsize=13, fontweight="bold")
	ax2.legend(fontsize=9)
	ax2.grid(alpha=0.3)

	plt.suptitle("Data Leakage Analysis: Random Split vs End Version Split",
	             fontsize=15, fontweight="bold", y=1.02)
	plt.tight_layout()
	plt.savefig(output_path, dpi=200, bbox_inches="tight")
	plt.close()
	print(f"  → Histogram saved: {output_path}")


def plot_summary_comparison(all_results: list[dict], output_path: Path) -> None:
	"""Vẽ biểu đồ tổng hợp so sánh 2 phương pháp chia (bar chart + error bars)."""
	# Tổng hợp theo split method
	metrics_keys = ["test_accuracy", "test_f1_macro", "test_f1_weighted",
	                "best_val_acc", "mean_cosine_sim", "max_cosine_sim_per_test_mean"]
	display_names = ["Test Accuracy", "Test F1 (Macro)", "Test F1 (Weighted)",
	                 "Best Val Accuracy", "Mean Cosine Sim\n(Train↔Test)",
	                 "Max Cosine Sim\n(per Test sample)"]

	random_vals = {k: [] for k in metrics_keys}
	endver_vals = {k: [] for k in metrics_keys}

	for r in all_results:
		target = random_vals if r["split_method"] == "Random" else endver_vals
		for k in metrics_keys:
			if k in r:
				target[k].append(r[k])

	fig, axes = plt.subplots(2, 3, figsize=(18, 10))
	axes = axes.flatten()

	x = np.arange(2)
	width = 0.5
	colors = ["#e74c3c", "#2ecc71"]

	for i, (key, name) in enumerate(zip(metrics_keys, display_names)):
		ax = axes[i]
		r_mean = np.mean(random_vals[key]) if random_vals[key] else 0
		r_std = np.std(random_vals[key]) if len(random_vals[key]) > 1 else 0
		e_mean = np.mean(endver_vals[key]) if endver_vals[key] else 0
		e_std = np.std(endver_vals[key]) if len(endver_vals[key]) > 1 else 0

		bars = ax.bar(x, [r_mean, e_mean], width, yerr=[r_std, e_std],
		              color=colors, capsize=8, edgecolor="black", linewidth=0.8, alpha=0.85)
		ax.set_xticks(x)
		ax.set_xticklabels(["Random Split", "End Version Split"], fontsize=10)
		ax.set_title(name, fontsize=12, fontweight="bold")
		ax.grid(axis="y", alpha=0.3)

		# Ghi giá trị lên cột
		for bar, val, std_val in zip(bars, [r_mean, e_mean], [r_std, e_std]):
			ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std_val + 0.005,
			        f"{val:.4f}\n±{std_val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

	plt.suptitle("Random Split vs End Version Split — Classification Baseline Comparison\n"
	             f"(ConvNeXt-Tiny, {EPOCHS} epochs, {len(SEEDS)} seeds)",
	             fontsize=14, fontweight="bold", y=1.02)
	plt.tight_layout()
	plt.savefig(output_path, dpi=200, bbox_inches="tight")
	plt.close()
	print(f"  → Summary chart saved: {output_path}")


def run_single_experiment(
	split_name: str,
	df_train: pd.DataFrame,
	df_val: pd.DataFrame,
	df_test: pd.DataFrame,
	class_names: list[str],
	class_to_idx: dict,
	seed: int,
	device: torch.device,
	output_dir: Path,
) -> dict:
	"""Chạy 1 thí nghiệm: train → evaluate → trả về metrics."""
	set_seed(seed)
	output_dir.mkdir(parents=True, exist_ok=True)

	print(f"\n{'='*70}")
	print(f"  [{split_name}] Seed={seed}")
	print(f"  Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}")
	print(f"{'='*70}")

	# Build model
	model = build_model(num_classes=len(class_names))
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

	train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
	                          num_workers=NUM_WORKERS, pin_memory=True)
	val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
	                        num_workers=NUM_WORKERS, pin_memory=True)
	test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
	                         num_workers=NUM_WORKERS, pin_memory=True)

	# Loss & Optimizer
	criterion = FocalLoss(gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA)
	optimizer = torch.optim.AdamW(
		filter(lambda p: p.requires_grad, model.parameters()),
		lr=LR, weight_decay=WEIGHT_DECAY,
	)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

	# Train
	history = train_model(
		model, train_loader, val_loader, optimizer, criterion,
		device, epochs=EPOCHS, patience=PATIENCE, output_dir=output_dir,
		scheduler=scheduler,
	)

	# Load best model
	best_path = output_dir / f"best_model_{MODEL_NAME}.pth"
	if best_path.exists():
		model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
		print(f"  → Loaded best model from {best_path}")

	# Evaluate Test
	model.eval()
	y_true, y_pred = collect_predictions(model, test_loader, device)
	test_acc = accuracy_score(y_true, y_pred)
	test_f1_macro = f1_score(y_true, y_pred, average="macro")
	test_f1_weighted = f1_score(y_true, y_pred, average="weighted")

	report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
	report_path = output_dir / "classification_report_test.txt"
	with open(report_path, "w", encoding="utf-8") as f:
		f.write(report)
	print(f"\n  Test Accuracy: {test_acc:.4f}")
	print(f"  Test F1 Macro: {test_f1_macro:.4f}")
	print(f"  Test F1 Weighted: {test_f1_weighted:.4f}")

	# Extract embeddings để tính cosine similarity
	print("  Extracting embeddings for similarity analysis...")
	train_embs, _ = extract_penultimate_embeddings(model, train_loader, device)
	test_embs, _ = extract_penultimate_embeddings(model, test_loader, device)

	sim_stats = compute_split_similarity(train_embs, test_embs)

	print(f"  Mean Cosine Sim (Train↔Test): {sim_stats['mean_cosine_sim']:.4f}")
	print(f"  Max Cosine Sim per Test (mean): {sim_stats['max_cosine_sim_per_test_mean']:.4f}")

	# Cleanup
	del model, train_loader, val_loader, test_loader
	del train_ds, val_ds, test_ds
	gc.collect()
	if device.type == "cuda":
		torch.cuda.empty_cache()

	return {
		"split_method": split_name,
		"seed": seed,
		"test_accuracy": test_acc,
		"test_f1_macro": test_f1_macro,
		"test_f1_weighted": test_f1_weighted,
		"best_val_acc": history.get("best_val_acc", 0.0),
		"mean_cosine_sim": sim_stats["mean_cosine_sim"],
		"median_cosine_sim": sim_stats["median_cosine_sim"],
		"max_cosine_sim_per_test_mean": sim_stats["max_cosine_sim_per_test_mean"],
		"max_cosine_sim_per_test_std": sim_stats["max_cosine_sim_per_test_std"],
		"train_size": len(df_train),
		"val_size": len(df_val),
		"test_size": len(df_test),
		"_all_sims": sim_stats["all_sims"],
		"_max_sims": sim_stats["max_sims"],
	}


def print_summary_table(all_results: list[dict]) -> str:
	"""In bảng tổng hợp kết quả tất cả các thí nghiệm."""
	lines = []
	lines.append("\n" + "=" * 100)
	lines.append("  BẢNG TỔNG HỢP: Random Split vs End Version Split")
	lines.append(f"  Model: {MODEL_NAME} | Epochs: {EPOCHS} | Seeds: {SEEDS}")
	lines.append("=" * 100)

	# Header
	header = f"{'Split':<15} {'Seed':<6} {'TestAcc':>9} {'F1-Macro':>9} {'F1-Wt':>9} " \
	         f"{'ValAcc':>9} {'CosSim':>9} {'MaxCosSim':>10}"
	lines.append(header)
	lines.append("-" * 100)

	# Per-run results
	for r in all_results:
		line = (f"{r['split_method']:<15} {r['seed']:<6} "
		        f"{r['test_accuracy']:>9.4f} {r['test_f1_macro']:>9.4f} {r['test_f1_weighted']:>9.4f} "
		        f"{r['best_val_acc']:>9.4f} {r['mean_cosine_sim']:>9.4f} "
		        f"{r['max_cosine_sim_per_test_mean']:>10.4f}")
		lines.append(line)

	lines.append("-" * 100)

	# Aggregated results
	for method in ["Random", "EndVersion"]:
		subset = [r for r in all_results if r["split_method"] == method]
		if not subset:
			continue
		acc_vals = [r["test_accuracy"] for r in subset]
		f1m_vals = [r["test_f1_macro"] for r in subset]
		f1w_vals = [r["test_f1_weighted"] for r in subset]
		val_vals = [r["best_val_acc"] for r in subset]
		cos_vals = [r["mean_cosine_sim"] for r in subset]
		max_vals = [r["max_cosine_sim_per_test_mean"] for r in subset]

		line = (f"{method + ' (avg)':<15} {'---':<6} "
		        f"{np.mean(acc_vals):>9.4f} {np.mean(f1m_vals):>9.4f} {np.mean(f1w_vals):>9.4f} "
		        f"{np.mean(val_vals):>9.4f} {np.mean(cos_vals):>9.4f} "
		        f"{np.mean(max_vals):>10.4f}")
		lines.append(line)

		line_std = (f"{method + ' (std)':<15} {'---':<6} "
		            f"{np.std(acc_vals):>9.4f} {np.std(f1m_vals):>9.4f} {np.std(f1w_vals):>9.4f} "
		            f"{np.std(val_vals):>9.4f} {np.std(cos_vals):>9.4f} "
		            f"{np.std(max_vals):>10.4f}")
		lines.append(line_std)

	lines.append("=" * 100)

	# Delta analysis
	random_acc = [r["test_accuracy"] for r in all_results if r["split_method"] == "Random"]
	endver_acc = [r["test_accuracy"] for r in all_results if r["split_method"] == "EndVersion"]
	if random_acc and endver_acc:
		delta_acc = np.mean(random_acc) - np.mean(endver_acc)
		lines.append(f"\n  Δ Test Accuracy (Random - EndVersion): {delta_acc:+.4f}")
		if delta_acc > 0:
			lines.append(f"  → Random Split OVERESTIMATES accuracy by {delta_acc*100:.2f}%")
		else:
			lines.append(f"  → End Version Split produces higher accuracy by {abs(delta_acc)*100:.2f}%")

	random_cos = [r["mean_cosine_sim"] for r in all_results if r["split_method"] == "Random"]
	endver_cos = [r["mean_cosine_sim"] for r in all_results if r["split_method"] == "EndVersion"]
	if random_cos and endver_cos:
		delta_cos = np.mean(random_cos) - np.mean(endver_cos)
		lines.append(f"  Δ Mean Cosine Similarity: {delta_cos:+.4f}")
		if delta_cos > 0:
			lines.append(f"  → Random Split has HIGHER train-test similarity (data leakage risk)")
		lines.append("")

	# Bảng định dạng Học thuật (mean ± std)
	lines.append("\n" + "=" * 125)
	lines.append("  BẢNG TỔNG HỢP (ĐỊNH DẠNG HỌC THUẬT: MEAN ± STD)")
	lines.append("=" * 125)
	header_academic = f"{'Split Method':<15} {'Test Accuracy':<18} {'F1-Macro':<18} {'F1-Weighted':<18} " \
	                  f"{'Best Val Accuracy':<18} {'Mean Cosine Sim':<18} {'Max Cosine Sim':<18}"
	lines.append(header_academic)
	lines.append("-" * 125)

	for method in ["Random", "EndVersion"]:
		subset = [r for r in all_results if r["split_method"] == method]
		if not subset:
			continue
		acc_vals = [r["test_accuracy"] for r in subset]
		f1m_vals = [r["test_f1_macro"] for r in subset]
		f1w_vals = [r["test_f1_weighted"] for r in subset]
		val_vals = [r["best_val_acc"] for r in subset]
		cos_vals = [r["mean_cosine_sim"] for r in subset]
		max_vals = [r["max_cosine_sim_per_test_mean"] for r in subset]

		acc_str = f"{np.mean(acc_vals):.4f} ± {np.std(acc_vals):.4f}" if len(acc_vals) > 1 else f"{acc_vals[0]:.4f} ± 0.0000"
		f1m_str = f"{np.mean(f1m_vals):.4f} ± {np.std(f1m_vals):.4f}" if len(f1m_vals) > 1 else f"{f1m_vals[0]:.4f} ± 0.0000"
		f1w_str = f"{np.mean(f1w_vals):.4f} ± {np.std(f1w_vals):.4f}" if len(f1w_vals) > 1 else f"{f1w_vals[0]:.4f} ± 0.0000"
		val_str = f"{np.mean(val_vals):.4f} ± {np.std(val_vals):.4f}" if len(val_vals) > 1 else f"{val_vals[0]:.4f} ± 0.0000"
		cos_str = f"{np.mean(cos_vals):.4f} ± {np.std(cos_vals):.4f}" if len(cos_vals) > 1 else f"{cos_vals[0]:.4f} ± 0.0000"
		max_str = f"{np.mean(max_vals):.4f} ± {np.std(max_vals):.4f}" if len(max_vals) > 1 else f"{max_vals[0]:.4f} ± 0.0000"

		line = f"{method:<15} {acc_str:<18} {f1m_str:<18} {f1w_str:<18} {val_str:<18} {cos_str:<18} {max_str:<18}"
		lines.append(line)

	lines.append("=" * 125)

	table_str = "\n".join(lines)
	print(table_str)
	return table_str


def main() -> None:
	device = get_device()
	print(f"Device: {device}")

	output_base = Path(OUTPUT_BASE)
	output_base.mkdir(parents=True, exist_ok=True)

	# ── Step 1: Thu thập dữ liệu & Compute embeddings (chạy 1 lần) ──
	print("\n[Step 1] Thu thập ảnh...")
	samples = collect_image_samples(ROOT_DIR)
	if not samples:
		raise ValueError(f"Không tìm thấy ảnh nào trong {ROOT_DIR}")

	df = build_dataframe(samples)
	EXCLUDED_CLASSES = ["Pterocarpus sp", "Peltogyne pubescens"]
	df_filtered = df[~df["label"].isin(EXCLUDED_CLASSES)].reset_index(drop=True)
	print(f"Tổng ảnh sau lọc: {len(df_filtered)}, Số class: {df_filtered['label'].nunique()}")

	class_names = sorted(df_filtered["label"].unique().tolist())
	class_to_idx = {name: i for i, name in enumerate(class_names)}

	print("\n[Step 2] Compute embeddings (EfficientNetV2-M + Swin-Large)...")
	print("  Trích xuất embeddings với EfficientNetV2-M...")
	embs_eff = compute_embeddings_v2(df_filtered, "tf_efficientnetv2_m_in21k", batch_size=BATCH_SIZE, device=device)
	print("  Trích xuất embeddings với Swin-Large...")
	embs_swin = compute_embeddings_v2(df_filtered, "swin_large_patch4_window7_224", batch_size=BATCH_SIZE, device=device)

	# ── Chạy 3 lần Random Split (3 seeds khác nhau) + 1 lần End Version Split ──
	all_results = []

	for seed in SEEDS:
		print(f"\n{'#'*70}")
		print(f"  RANDOM SPLIT — Seed {seed}")
		print(f"{'#'*70}")

		set_seed(seed)
		df_train_r, df_val_r, df_test_r = random_split_wrapper(
			df, embs_eff, embs_swin,
			train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, seed=seed,
		)

		run_dir = output_base / f"random_seed{seed}"
		result = run_single_experiment(
			"Random", df_train_r, df_val_r, df_test_r,
			class_names, class_to_idx, seed, device, run_dir,
		)
		all_results.append(result)

	# ── Chạy 1 lần End Version Split (seed=42, cấu hình chuẩn từ train_final.py) ──
	endver_seed = 42
	print(f"\n{'#'*70}")
	print(f"  END VERSION SPLIT — Seed {endver_seed} (cấu hình chuẩn train_final.py)")
	print(f"{'#'*70}")

	set_seed(endver_seed)
	df_train_e, df_val_e, df_test_e = end_version_split(
		df_filtered, embs_eff, embs_swin,
		train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, seed=endver_seed,
	)

	run_dir = output_base / f"endversion_seed{endver_seed}"
	result = run_single_experiment(
		"EndVersion", df_train_e, df_val_e, df_test_e,
		class_names, class_to_idx, endver_seed, device, run_dir,
	)
	all_results.append(result)

	# ── Step 3: Tổng hợp & Vẽ biểu đồ ──
	print("\n\n[Step 3] Tổng hợp kết quả...")

	# Bảng tổng hợp
	summary_str = print_summary_table(all_results)
	with open(output_base / "summary_table.txt", "w", encoding="utf-8") as f:
		f.write(summary_str)

	# Lưu JSON chi tiết
	json_results = [{k: v for k, v in r.items() if not k.startswith("_")} for r in all_results]
	with open(output_base / "all_results.json", "w", encoding="utf-8") as f:
		json.dump(json_results, f, indent=2, ensure_ascii=False)

	# Vẽ histogram cosine similarity (dùng seed cuối cùng làm đại diện)
	random_last = [r for r in all_results if r["split_method"] == "Random"][-1]
	endver_last = [r for r in all_results if r["split_method"] == "EndVersion"][-1]
	plot_similarity_histograms(
		{"all_sims": random_last["_all_sims"], "max_sims": random_last["_max_sims"],
		 "mean_cosine_sim": random_last["mean_cosine_sim"],
		 "max_cosine_sim_per_test_mean": random_last["max_cosine_sim_per_test_mean"]},
		{"all_sims": endver_last["_all_sims"], "max_sims": endver_last["_max_sims"],
		 "mean_cosine_sim": endver_last["mean_cosine_sim"],
		 "max_cosine_sim_per_test_mean": endver_last["max_cosine_sim_per_test_mean"]},
		output_base / "histogram_cosine_similarity.png",
	)

	# Vẽ biểu đồ tổng hợp
	plot_summary_comparison(all_results, output_base / "summary_comparison.png")

	print(f"\n[Hoàn tất] Tất cả kết quả lưu tại: {output_base}/")
	print(f"  - summary_table.txt          → Bảng tổng hợp text")
	print(f"  - all_results.json           → JSON chi tiết")
	print(f"  - histogram_cosine_similarity.png → Histogram so sánh")
	print(f"  - summary_comparison.png     → Bar chart tổng hợp")


if __name__ == "__main__":
	main()
