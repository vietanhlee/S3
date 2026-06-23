"""
train_split_comparison.py
=========================
Pipeline chạy 8 phương pháp chia dữ liệu, training CNN (tf_efficientnet_b4) cho mỗi PP,
so sánh kết quả cuối cùng.

Reuse hàm từ train.py, splitting logic từ split_methods.py.
"""

import os
import json
from pathlib import Path

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader

from timm.data import resolve_data_config
from sklearn.metrics import classification_report, precision_recall_fscore_support

# ===== CẤU HÌNH - CHỈNH SỬA TẠI ĐÂY =====
ROOT_DIR = r"/kaggle/input/datasets/canhdoo/s3-data/S3"
OUTPUT_BASE_DIR = "outputs_split_comparison"
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
MODEL_NAME = "tf_efficientnet_b4"
FREEZE_RATIO = 0.90
COSINE_THRESHOLD = 0.92  # Cho PP5 (Cosine Graph)
# ============================================

# Import từ train.py
from train import (
	set_seed,
	get_device,
	collect_image_samples,
	build_dataframe,
	compute_embeddings,
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

# Import từ split_methods_v2.py
from split_methods_v2 import (
	SPLIT_METHODS,
	validate_split,
	cosine_graph_split,
)

import timm


# ============================================================
# Helper: build model
# ============================================================

def build_model(num_classes: int) -> torch.nn.Module:
	"""Tạo tf_efficientnet_b4 pretrained, freeze theo tỉ lệ."""
	model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=num_classes)
	freeze_model_layers(model, FREEZE_RATIO)
	model.model_name = MODEL_NAME
	return model


# ============================================================
# Helper: chạy 1 phương pháp (chia + train + evaluate)
# ============================================================

def run_one_method(
	method_name: str,
	df_train: pd.DataFrame,
	df_val: pd.DataFrame,
	df_test: pd.DataFrame,
	class_names: list[str],
	class_to_idx: dict,
	device: torch.device,
	output_dir: Path,
) -> dict:
	"""Chạy training + evaluation cho 1 split method. Trả về kết quả."""

	output_dir.mkdir(parents=True, exist_ok=True)
	print(f"\n{'=' * 70}")
	print(f"[{method_name}] Starting training...")
	print(f"{'=' * 70}")

	# Log split summary
	df_all = pd.concat([df_train, df_val, df_test], ignore_index=True)
	log_split_summary(df_all, df_train, df_val, df_test)

	# EDA
	eda_split_class_distribution(
		df_train, df_val, df_test,
		f"{method_name} - class distribution",
		output_dir / f"eda_split_{method_name}.png",
	)

	# Build model
	model = build_model(num_classes=len(class_names))
	model_info = summarize_model(model)
	print(
		f"Model: {MODEL_NAME}, "
		f"total={model_info['total_params']:,}, "
		f"trainable={model_info['trainable_params']:,}, "
		f"frozen={model_info['frozen_params']:,}"
	)
	model = model.to(device)
	if device.type == "cuda" and torch.cuda.device_count() > 1:
		print(f"[{method_name}] Phát hiện {torch.cuda.device_count()} GPUs. Sử dụng nn.DataParallel.")
		model = torch.nn.DataParallel(model)

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

	num_workers = min(8, os.cpu_count() or 1)
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

	# Learning Rate Scheduler (Cosine Annealing)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

	# Training
	history = train_model(
		model, train_loader, val_loader, optimizer, criterion,
		device, epochs=EPOCHS, patience=PATIENCE, output_dir=output_dir,
		scheduler=scheduler,
	)
	plot_training_curves(history, output_dir)

	# Load best model
	best_path = output_dir / f"best_model_{MODEL_NAME}.pth"
	if best_path.exists():
		raw_model = model.module if isinstance(model, torch.nn.DataParallel) else model
		raw_model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
		print(f"[{method_name}] Loaded best model from {best_path}")

	# Evaluate
	evaluate_and_report(model, val_loader, device, class_names, output_dir, prefix="val")
	evaluate_and_report(model, test_loader, device, class_names, output_dir, prefix="test")

	# Thu thập per-class accuracy từ test report
	test_report = _get_per_class_acc(model, test_loader, device, class_names)

	# Kết quả
	result = {
		"method": method_name,
		"best_val_acc": history.get("best_val_acc", 0.0),
		"final_train_acc": history["train_acc"][-1] if history["train_acc"] else 0.0,
		"final_val_acc": history["val_acc"][-1] if history["val_acc"] else 0.0,
		"test_acc": test_report["overall_acc"],
		"test_precision": test_report["precision_macro"],
		"test_recall": test_report["recall_macro"],
		"test_f1": test_report["f1_macro"],
		"per_class_test_acc_min": test_report["min_acc"],
		"per_class_test_acc_max": test_report["max_acc"],
		"per_class_test_acc_mean": test_report["mean_acc"],
		"train_size": len(df_train),
		"val_size": len(df_val),
		"test_size": len(df_test),
		"epochs_trained": len(history["train_loss"]),
	}

	# Lưu result JSON
	with open(output_dir / "result.json", "w", encoding="utf-8") as f:
		json.dump(result, f, indent=2, ensure_ascii=False)

	# Giải phóng bộ nhớ
	del model, optimizer, criterion
	if device.type == "cuda":
		torch.cuda.empty_cache()

	return result


@torch.no_grad()
def _get_per_class_acc(
	model: torch.nn.Module,
	loader: DataLoader,
	device: torch.device,
	class_names: list[str],
) -> dict:
	"""Tính accuracy tổng thể, metrics trung bình (precision, recall, f1) và per-class accuracy trên test set."""
	model.eval()
	y_true, y_pred = [], []
	for images, targets in loader:
		images = images.to(device)
		logits = model(images)
		preds = torch.argmax(logits, dim=1).cpu().tolist()
		y_pred.extend(preds)
		y_true.extend(targets.tolist())

	# Overall accuracy
	correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
	overall_acc = correct / len(y_true) if y_true else 0.0

	# Macro metrics
	precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
		y_true, y_pred, average="macro", zero_division=0
	)

	# Per-class accuracy
	class_correct = {}
	class_total = {}
	for t, p in zip(y_true, y_pred):
		class_total[t] = class_total.get(t, 0) + 1
		if t == p:
			class_correct[t] = class_correct.get(t, 0) + 1

	per_class_acc = []
	for idx in range(len(class_names)):
		total = class_total.get(idx, 0)
		correct_c = class_correct.get(idx, 0)
		acc = correct_c / total if total > 0 else 0.0
		per_class_acc.append(acc)

	return {
		"overall_acc": overall_acc,
		"precision_macro": float(precision_macro),
		"recall_macro": float(recall_macro),
		"f1_macro": float(f1_macro),
		"min_acc": min(per_class_acc) if per_class_acc else 0.0,
		"max_acc": max(per_class_acc) if per_class_acc else 0.0,
		"mean_acc": float(np.mean(per_class_acc)) if per_class_acc else 0.0,
		"per_class": per_class_acc,
	}


# ============================================================
# Bảng tổng hợp so sánh
# ============================================================

def print_comparison_table(results: list[dict]) -> str:
	"""In bảng so sánh đẹp và trả về string."""
	header = (
		f"{'Phương pháp':<30} {'Train ACC':>10} {'Val ACC':>10} "
		f"{'Test ACC':>10} {'Test P':>8} {'Test R':>8} {'Test F1':>8} "
		f"{'Min':>7} {'Max':>7} {'Mean':>7} "
		f"{'Train':>6} {'Val':>5} {'Test':>5} {'Ep':>4}"
	)
	separator = "=" * len(header)

	lines = ["\n" + separator, "BẢNG TỔNG HỢP SO SÁNH CÁC PHƯƠNG PHÁP CHIA DỮ LIỆU", separator, header, "-" * len(header)]

	for r in results:
		line = (
			f"{r['method']:<30} "
			f"{r['final_train_acc']*100:>9.2f}% "
			f"{r['final_val_acc']*100:>9.2f}% "
			f"{r['test_acc']*100:>9.2f}% "
			f"{r['test_precision']*100:>7.2f}% "
			f"{r['test_recall']*100:>7.2f}% "
			f"{r['test_f1']*100:>7.2f}% "
			f"{r['per_class_test_acc_min']*100:>6.1f}% "
			f"{r['per_class_test_acc_max']*100:>6.1f}% "
			f"{r['per_class_test_acc_mean']*100:>6.1f}% "
			f"{r['train_size']:>6} "
			f"{r['val_size']:>5} "
			f"{r['test_size']:>5} "
			f"{r['epochs_trained']:>4}"
		)
		lines.append(line)

	lines.append(separator)

	table_str = "\n".join(lines)
	print(table_str)
	return table_str


# ============================================================
# Main
# ============================================================

def main() -> None:
	set_seed(SEED)
	device = get_device()
	print(f"Device: {device}")

	output_base = Path(OUTPUT_BASE_DIR)
	output_base.mkdir(parents=True, exist_ok=True)

	# 1. Thu thập ảnh, build dataframe
	print("\n[Step 1] Thu thập ảnh...")
	samples = collect_image_samples(ROOT_DIR)
	if not samples:
		raise ValueError(f"Không tìm thấy ảnh nào trong {ROOT_DIR}")

	df = build_dataframe(samples)
	print(f"Tổng ảnh: {len(df)}, Số class: {df['label'].nunique()}")

	class_names = sorted(df["label"].unique().tolist())
	class_to_idx = {name: i for i, name in enumerate(class_names)}

	# 2. Compute embeddings (1 lần duy nhất)
	print("\n[Step 2] Compute embeddings...")
	embeddings = compute_embeddings(df, batch_size=BATCH_SIZE, device=device)
	print(f"Embeddings shape: {embeddings.shape}")

	# Giải phóng VRAM sau khi compute embeddings
	if device.type == "cuda":
		torch.cuda.empty_cache()

	# 3. Chạy lần lượt từng phương pháp
	all_results = []
	methods_to_run = list(SPLIT_METHODS.items())

	for method_name, split_fn in methods_to_run:
		print(f"\n{'#' * 70}")
		print(f"# {method_name}")
		print(f"{'#' * 70}")

		method_output_dir = output_base / method_name

		# Gọi hàm chia dữ liệu
		try:
			if method_name == "PP5_Cosine_Graph":
				# PP5 cần thêm tham số cosine_threshold
				df_train, df_val, df_test = split_fn(
					df, embeddings,
					train_ratio=TRAIN_RATIO,
					val_ratio=VAL_RATIO,
					seed=SEED,
					cosine_threshold=COSINE_THRESHOLD,
				)
			else:
				df_train, df_val, df_test = split_fn(
					df, embeddings,
					train_ratio=TRAIN_RATIO,
					val_ratio=VAL_RATIO,
					seed=SEED,
				)
		except Exception as e:
			print(f"[{method_name}] ERROR khi chia dữ liệu: {e}")
			continue

		# Validate split
		is_valid = validate_split(df, df_train, df_val, df_test, method_name)
		if not is_valid:
			print(f"[{method_name}] WARNING: Validation failed, tiếp tục training anyway...")

		# Training + Evaluate
		try:
			result = run_one_method(
				method_name, df_train, df_val, df_test,
				class_names, class_to_idx, device, method_output_dir,
			)
			all_results.append(result)
		except Exception as e:
			print(f"[{method_name}] ERROR khi training: {e}")
			import traceback
			traceback.print_exc()
			continue

	# 4. In bảng tổng hợp
	if all_results:
		table_str = print_comparison_table(all_results)

		# Lưu bảng ra file
		with open(output_base / "comparison_table.txt", "w", encoding="utf-8") as f:
			f.write(table_str)

		# Lưu tất cả results ra JSON
		with open(output_base / "all_results.json", "w", encoding="utf-8") as f:
			json.dump(all_results, f, indent=2, ensure_ascii=False)

		# Vẽ biểu đồ so sánh các phương pháp
		try:
			plot_comparison_chart(all_results, output_base / "comparison_chart.png")
		except Exception as e:
			print(f"ERROR khi vẽ biểu đồ so sánh: {e}")

		print(f"\nKết quả đã lưu tại: {output_base}")
	else:
		print("\nKhông có kết quả nào!")


def plot_comparison_chart(results: list[dict], save_path: Path) -> None:
	"""Vẽ biểu đồ cột so sánh Accuracy, Precision, Recall và F1-Score của các phương pháp."""
	import matplotlib.pyplot as plt

	methods = [r["method"] for r in results]
	test_acc = [r["test_acc"] * 100 for r in results]
	test_precision = [r["test_precision"] * 100 for r in results]
	test_recall = [r["test_recall"] * 100 for r in results]
	test_f1 = [r["test_f1"] * 100 for r in results]

	x = np.arange(len(methods))
	width = 0.2

	fig, ax = plt.subplots(figsize=(15, 8))

	rects1 = ax.bar(x - 1.5 * width, test_acc, width, label="Test Accuracy", color="#3b82f6")
	rects2 = ax.bar(x - 0.5 * width, test_precision, width, label="Test Precision (Macro)", color="#10b981")
	rects3 = ax.bar(x + 0.5 * width, test_recall, width, label="Test Recall (Macro)", color="#f59e0b")
	rects4 = ax.bar(x + 1.5 * width, test_f1, width, label="Test F1-Score (Macro)", color="#ec4899")

	ax.set_ylabel("Phần trăm (%)")
	ax.set_title("So sánh các phương pháp chia dữ liệu trên tập Test", fontsize=14, fontweight="bold", pad=15)
	ax.set_xticks(x)
	ax.set_xticklabels(methods, rotation=30, ha="right")
	ax.set_ylim(0, 110)
	ax.legend(loc="upper right", frameon=True, shadow=True)
	ax.grid(axis="y", linestyle="--", alpha=0.5)

	# Thêm nhãn giá trị trên đầu cột
	def autolabel(rects):
		for rect in rects:
			height = rect.get_height()
			ax.annotate(
				f"{height:.1f}%",
				xy=(rect.get_x() + rect.get_width() / 2, height),
				xytext=(0, 3),  # 3 points vertical offset
				textcoords="offset points",
				ha="center",
				va="bottom",
				fontsize=8,
				rotation=90,
			)

	autolabel(rects1)
	autolabel(rects2)
	autolabel(rects3)
	autolabel(rects4)

	fig.tight_layout()
	plt.savefig(save_path, dpi=300)
	plt.close()
	print(f"Đã lưu biểu đồ so sánh tại: {save_path}")


if __name__ == "__main__":
	main()
