"""
train_split_ablation_v2.py — So sánh Random Split vs End Version Split (V2: Không cần huấn luyện)
============================================================================================
Chứng minh 3 luận điểm khoa học mà không cần huấn luyện mô hình (chỉ dùng đặc trưng nền tảng):
  1. Random Split overestimate performance (accuracy KNN cao giả tạo do rò rỉ dữ liệu)
  2. End Version Split tách tốt hơn (cosine similarity train↔test thấp hơn rõ rệt)
  3. Histogram phân phối khoảng cách embedding train↔test minh họa sự khác biệt

Chạy:
  python train_split_ablation_v2.py
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
from sklearn.neighbors import KNeighborsClassifier

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Import từ project ──
from utils import (
	set_seed, get_device,
	collect_image_samples, build_dataframe,
)
from train_final import (
	compute_embeddings_v2, end_version_split, SPLIT_CONFIG,
)
from split_methods import validate_split, compute_split_counts


# ===== CẤU HÌNH =====
ROOT_DIR       = r"/kaggle/input/datasets/b23dckh002lvitanh/s3-origin/S3"
OUTPUT_BASE    = "outputs_split_ablation_v2"
TRAIN_RATIO    = 0.60
VAL_RATIO      = 0.20
BATCH_SIZE     = 128
SEED_ENDVER    = 42
SEEDS_RANDOM   = [42, 123, 456]  # 3 random seeds cho Random Split
# =====================


def random_split_wrapper(
	df: pd.DataFrame,
	train_ratio: float = 0.60,
	val_ratio: float = 0.20,
	seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	"""Wrapper cho subfolder-based random split (với fallback sang image-level random split)."""
	# Loại bỏ Pterocarpus sp trước
	keep_mask = df["label"] != "Pterocarpus sp"
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


def compute_split_similarity(
	train_embs: np.ndarray,
	test_embs: np.ndarray,
) -> dict:
	"""Tính cosine similarity trung bình và các thống kê giữa train và test embeddings."""
	# Cosine similarity matrix: (n_test, n_train)
	sim_matrix = cosine_similarity(test_embs, train_embs)

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

	plt.suptitle("Data Leakage Analysis: Random Split vs End Version Split (Base Embeddings)",
	             fontsize=15, fontweight="bold", y=1.02)
	plt.tight_layout()
	plt.savefig(output_path, dpi=200, bbox_inches="tight")
	plt.close()
	print(f"  → Histogram saved: {output_path}")


def plot_summary_comparison(all_results: list[dict], output_path: Path) -> None:
	"""Vẽ biểu đồ tổng hợp so sánh 2 phương pháp chia (bar chart + error bars)."""
	metrics_keys = ["test_accuracy", "test_f1_macro", "test_f1_weighted",
	                "mean_cosine_sim", "max_cosine_sim_per_test_mean"]
	display_names = ["KNN Test Accuracy", "KNN Test F1 (Macro)", "KNN Test F1 (Weighted)",
	                 "Mean Cosine Sim\n(Train↔Test)",
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

	for i, key in enumerate(metrics_keys):
		ax = axes[i]
		r_mean = np.mean(random_vals[key]) if random_vals[key] else 0
		r_std = np.std(random_vals[key]) if len(random_vals[key]) > 1 else 0
		e_mean = np.mean(endver_vals[key]) if endver_vals[key] else 0
		e_std = np.std(endver_vals[key]) if len(endver_vals[key]) > 1 else 0

		bars = ax.bar(x, [r_mean, e_mean], width, yerr=[r_std, e_std],
		              color=colors, capsize=8, edgecolor="black", linewidth=0.8, alpha=0.85)
		ax.set_xticks(x)
		ax.set_xticklabels(["Random Split", "End Version Split"], fontsize=10)
		ax.set_title(display_names[i], fontsize=12, fontweight="bold")
		ax.grid(axis="y", alpha=0.3)

		# Ghi giá trị lên cột
		for bar, val, std_val in zip(bars, [r_mean, e_mean], [r_std, e_std]):
			ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std_val + 0.005,
			        f"{val:.4f}\n±{std_val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

	# Hide empty axes
	for i in range(len(metrics_keys), len(axes)):
		fig.delaxes(axes[i])

	plt.suptitle("Random Split vs End Version Split — KNN Classification Analysis (No-Training)",
	             fontsize=14, fontweight="bold", y=1.02)
	plt.tight_layout()
	plt.savefig(output_path, dpi=200, bbox_inches="tight")
	plt.close()
	print(f"  → Summary chart saved: {output_path}")


def evaluate_split_via_knn(
	split_name: str,
	df_train: pd.DataFrame,
	df_test: pd.DataFrame,
	df_filtered: pd.DataFrame,
	embs_swin: np.ndarray,
	class_to_idx: dict,
	seed: int,
) -> dict:
	"""Đánh giá split bằng bộ phân loại KNN trên frozen Swin-Large embeddings."""
	
	# Tạo map từ path đến index của embedding
	path_to_idx = {path: i for i, path in enumerate(df_filtered["path"])}
	
	train_emb_indices = [path_to_idx[p] for p in df_train["path"]]
	test_emb_indices = [path_to_idx[p] for p in df_test["path"]]
	
	# Trích xuất embeddings và nhãn tương ứng
	train_feats = embs_swin[train_emb_indices]
	test_feats = embs_swin[test_emb_indices]
	
	train_labels = np.array([class_to_idx[lbl] for lbl in df_train["label"]])
	test_labels = np.array([class_to_idx[lbl] for lbl in df_test["label"]])
	
	# Bộ phân loại KNN (K=1, metric=cosine) đại diện cho kịch bản Retrieval/Embedding Space
	knn = KNeighborsClassifier(n_neighbors=1, metric="cosine")
	knn.fit(train_feats, train_labels)
	y_pred = knn.predict(test_feats)
	
	# Tính toán metrics
	test_acc = accuracy_score(test_labels, y_pred)
	test_f1_macro = f1_score(test_labels, y_pred, average="macro")
	test_f1_weighted = f1_score(test_labels, y_pred, average="weighted")
	
	# Tính tương đồng hình học
	sim_stats = compute_split_similarity(train_feats, test_feats)
	
	return {
		"split_method": split_name,
		"seed": seed,
		"test_accuracy": test_acc,
		"test_f1_macro": test_f1_macro,
		"test_f1_weighted": test_f1_weighted,
		"mean_cosine_sim": sim_stats["mean_cosine_sim"],
		"median_cosine_sim": sim_stats["median_cosine_sim"],
		"max_cosine_sim_per_test_mean": sim_stats["max_cosine_sim_per_test_mean"],
		"max_cosine_sim_per_test_std": sim_stats["max_cosine_sim_per_test_std"],
		"train_size": len(df_train),
		"test_size": len(df_test),
		"_all_sims": sim_stats["all_sims"],
		"_max_sims": sim_stats["max_sims"],
	}


def print_summary_table(all_results: list[dict]) -> str:
	"""In bảng tổng hợp kết quả tất cả các thí nghiệm."""
	lines = []
	lines.append("\n" + "=" * 110)
	lines.append("  BẢNG TỔNG HỢP (CHI TIẾT RUNS): Random Split vs End Version Split (KNN - Swin-Large)")
	lines.append("=" * 110)

	# Header
	header = f"{'Split':<15} {'Seed':<6} {'KNN-Acc':>9} {'F1-Macro':>9} {'F1-Wt':>9} {'CosSim':>9} {'MaxCosSim':>10} {'TrainSize':>10} {'TestSize':>10}"
	lines.append(header)
	lines.append("-" * 110)

	# Per-run results
	for r in all_results:
		line = (f"{r['split_method']:<15} {r['seed']:<6} "
		        f"{r['test_accuracy']:>9.4f} {r['test_f1_macro']:>9.4f} {r['test_f1_weighted']:>9.4f} "
		        f"{r['mean_cosine_sim']:>9.4f} {r['max_cosine_sim_per_test_mean']:>10.4f} "
		        f"{r['train_size']:>10} {r['test_size']:>10}")
		lines.append(line)

	lines.append("-" * 110)

	# Aggregated results
	for method in ["Random", "EndVersion"]:
		subset = [r for r in all_results if r["split_method"] == method]
		if not subset:
			continue
		acc_vals = [r["test_accuracy"] for r in subset]
		f1m_vals = [r["test_f1_macro"] for r in subset]
		f1w_vals = [r["test_f1_weighted"] for r in subset]
		cos_vals = [r["mean_cosine_sim"] for r in subset]
		max_vals = [r["max_cosine_sim_per_test_mean"] for r in subset]

		line = (f"{method + ' (avg)':<15} {'---':<6} "
		        f"{np.mean(acc_vals):>9.4f} {np.mean(f1m_vals):>9.4f} {np.mean(f1w_vals):>9.4f} "
		        f"{np.mean(cos_vals):>9.4f} {np.mean(max_vals):>10.4f} "
		        f"{'-':>10} {'-':>10}")
		lines.append(line)

		line_std = (f"{method + ' (std)':<15} {'---':<6} "
		            f"{np.std(acc_vals):>9.4f} {np.std(f1m_vals):>9.4f} {np.std(f1w_vals):>9.4f} "
		            f"{np.std(cos_vals):>9.4f} {np.std(max_vals):>10.4f} "
		            f"{'-':>10} {'-':>10}")
		lines.append(line_std)

	lines.append("=" * 110)

	# Delta analysis
	random_acc = [r["test_accuracy"] for r in all_results if r["split_method"] == "Random"]
	endver_acc = [r["test_accuracy"] for r in all_results if r["split_method"] == "EndVersion"]
	if random_acc and endver_acc:
		delta_acc = np.mean(random_acc) - np.mean(endver_acc)
		lines.append(f"\n  Δ KNN Test Accuracy (Random - EndVersion): {delta_acc:+.4f}")
		if delta_acc > 0:
			lines.append(f"  → Random Split OVERESTIMATES baseline performance by {delta_acc*100:.2f}% (Data Leakage)")
		else:
			lines.append(f"  → End Version Split performs better by {abs(delta_acc)*100:.2f}%")

	random_cos = [r["mean_cosine_sim"] for r in all_results if r["split_method"] == "Random"]
	endver_cos = [r["mean_cosine_sim"] for r in all_results if r["split_method"] == "EndVersion"]
	if random_cos and endver_cos:
		delta_cos = np.mean(random_cos) - np.mean(endver_cos)
		lines.append(f"  Δ Mean Cosine Similarity: {delta_cos:+.4f}")
		if delta_cos > 0:
			lines.append(f"  → Random Split has HIGHER train-test similarity (unintentional data leakage)")
		lines.append("")

	# Bảng định dạng Học thuật (mean ± std)
	lines.append("\n" + "=" * 125)
	lines.append("  BẢNG TỔNG HỢP (ĐỊNH DẠNG HỌC THUẬT: MEAN ± STD)")
	lines.append("=" * 125)
	header_academic = f"{'Split Method':<15} {'KNN Test Accuracy':<18} {'F1-Macro':<18} {'F1-Weighted':<18} " \
	                  f"{'Mean Cosine Sim':<18} {'Max Cosine Sim':<18}"
	lines.append(header_academic)
	lines.append("-" * 125)

	for method in ["Random", "EndVersion"]:
		subset = [r for r in all_results if r["split_method"] == method]
		if not subset:
			continue
		acc_vals = [r["test_accuracy"] for r in subset]
		f1m_vals = [r["test_f1_macro"] for r in subset]
		f1w_vals = [r["test_f1_weighted"] for r in subset]
		cos_vals = [r["mean_cosine_sim"] for r in subset]
		max_vals = [r["max_cosine_sim_per_test_mean"] for r in subset]

		acc_str = f"{np.mean(acc_vals):.4f} ± {np.std(acc_vals):.4f}" if len(acc_vals) > 1 else f"{acc_vals[0]:.4f} ± 0.0000"
		f1m_str = f"{np.mean(f1m_vals):.4f} ± {np.std(f1m_vals):.4f}" if len(f1m_vals) > 1 else f"{f1m_vals[0]:.4f} ± 0.0000"
		f1w_str = f"{np.mean(f1w_vals):.4f} ± {np.std(f1w_vals):.4f}" if len(f1w_vals) > 1 else f"{f1w_vals[0]:.4f} ± 0.0000"
		cos_str = f"{np.mean(cos_vals):.4f} ± {np.std(cos_vals):.4f}" if len(cos_vals) > 1 else f"{cos_vals[0]:.4f} ± 0.0000"
		max_str = f"{np.mean(max_vals):.4f} ± {np.std(max_vals):.4f}" if len(max_vals) > 1 else f"{max_vals[0]:.4f} ± 0.0000"

		line = f"{method:<15} {acc_str:<18} {f1m_str:<18} {f1w_str:<18} {cos_str:<18} {max_str:<18}"
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

	# ── Step 1: Thu thập dữ liệu & Trích xuất embeddings ──
	print("\n[Step 1] Thu thập ảnh...")
	samples = collect_image_samples(ROOT_DIR)
	if not samples:
		raise ValueError(f"Không tìm thấy ảnh nào trong {ROOT_DIR}")

	df = build_dataframe(samples)
	df_filtered = df[df["label"] != "Pterocarpus sp"].reset_index(drop=True)
	print(f"Tổng ảnh sau lọc: {len(df_filtered)}, Số class: {df_filtered['label'].nunique()}")

	class_names = sorted(df_filtered["label"].unique().tolist())
	class_to_idx = {name: i for i, name in enumerate(class_names)}

	print("\n[Step 2] Trích xuất embeddings cố định (EfficientNetV2-M + Swin-Large)...")
	print("  Trích xuất embeddings với EfficientNetV2-M (cho thuật toán chia)...")
	embs_eff = compute_embeddings_v2(df_filtered, "tf_efficientnetv2_m_in21k", batch_size=BATCH_SIZE, device=device)
	print("  Trích xuất embeddings với Swin-Large (cho phân tích độ tương đồng)...")
	embs_swin = compute_embeddings_v2(df_filtered, "swin_large_patch4_window7_224", batch_size=BATCH_SIZE, device=device)

	# ── Step 3: Đánh giá các phương pháp chia dữ liệu ──
	all_results = []

	# 1. Đánh giá 3 seeds cho Random Split
	print(f"\n[Step 3] Bắt đầu đánh giá các kịch bản chia...")
	for seed in SEEDS_RANDOM:
		print(f"  -> Đánh giá Random Split (Seed {seed})...")
		df_train_r, df_val_r, df_test_r = random_split_wrapper(
			df, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, seed=seed,
		)
		
		res = evaluate_split_via_knn(
			"Random", df_train_r, df_test_r, df_filtered, embs_swin, class_to_idx, seed
		)
		all_results.append(res)

	# 2. Đánh giá 1 seed cho End Version Split (Seed 42)
	print(f"  -> Đánh giá End Version Split (Seed {SEED_ENDVER})...")
	df_train_e, df_val_e, df_test_e = end_version_split(
		df_filtered, embs_eff, embs_swin,
		train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, seed=SEED_ENDVER,
	)
	
	res = evaluate_split_via_knn(
		"EndVersion", df_train_e, df_test_e, df_filtered, embs_swin, class_to_idx, SEED_ENDVER
	)
	all_results.append(res)

	# ── Step 4: Tổng hợp kết quả ──
	print("\n\n[Step 4] Tổng hợp kết quả...")

	# Bảng tổng hợp
	summary_str = print_summary_table(all_results)
	with open(output_base / "summary_table.txt", "w", encoding="utf-8") as f:
		f.write(summary_str)

	# Lưu JSON chi tiết
	json_results = [{k: v for k, v in r.items() if not k.startswith("_")} for r in all_results]
	with open(output_base / "all_results.json", "w", encoding="utf-8") as f:
		json.dump(json_results, f, indent=2, ensure_ascii=False)

	# Vẽ histogram cosine similarity (dùng seed 42 của cả 2 để so sánh trực tiếp)
	random_42 = [r for r in all_results if r["split_method"] == "Random" and r["seed"] == 42][0]
	endver_42 = [r for r in all_results if r["split_method"] == "EndVersion" and r["seed"] == 42][0]
	
	plot_similarity_histograms(
		{"all_sims": random_42["_all_sims"], "max_sims": random_42["_max_sims"],
		 "mean_cosine_sim": random_42["mean_cosine_sim"],
		 "max_cosine_sim_per_test_mean": random_42["max_cosine_sim_per_test_mean"]},
		{"all_sims": endver_42["_all_sims"], "max_sims": endver_42["_max_sims"],
		 "mean_cosine_sim": endver_42["mean_cosine_sim"],
		 "max_cosine_sim_per_test_mean": endver_42["max_cosine_sim_per_test_mean"]},
		output_base / "histogram_cosine_similarity.png",
	)

	# Vẽ biểu đồ tổng hợp
	plot_summary_comparison(all_results, output_base / "summary_comparison.png")

	print(f"\n[Hoàn tất] Tất cả kết quả lưu tại: {output_base}/")
	print(f"  - summary_table.txt          → Bảng tổng hợp text (chi tiết & học thuật)")
	print(f"  - all_results.json           → JSON chi tiết")
	print(f"  - histogram_cosine_similarity.png → Histogram so sánh đặc trưng")
	print(f"  - summary_comparison.png     → Bar chart so sánh KNN & Cosine Sim")


if __name__ == "__main__":
	main()
