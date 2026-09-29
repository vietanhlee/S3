"""
02_specimen_variant_gan.training.evaluator
==========================================
Đánh giá định lượng & Trực quan hoá sai số: Ma trận nhầm lẫn đa cấp (Loài & Chi), Phổ mẫu nhầm lẫn, Báo cáo F1-macro.
"""

from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from utils import split_genus_species


def plot_training_curves(history: dict, output_dir: Path) -> None:
	"""Vẽ và lưu đồ thị đường cong học tập Loss và Accuracy."""
	epochs = range(1, len(history["train_loss"]) + 1)
	plt.figure(figsize=(8, 4))
	plt.plot(epochs, history["train_loss"], label="train_loss")
	plt.plot(epochs, history["val_loss"], label="val_loss")
	plt.xlabel("Epoch")
	plt.ylabel("Loss")
	plt.legend()
	plt.tight_layout()
	plt.savefig(output_dir / "loss_curve.png", dpi=200)
	plt.close()

	plt.figure(figsize=(8, 4))
	plt.plot(epochs, history["train_acc"], label="train_acc")
	plt.plot(epochs, history["val_acc"], label="val_acc")
	plt.xlabel("Epoch")
	plt.ylabel("Accuracy")
	plt.legend()
	plt.tight_layout()
	plt.savefig(output_dir / "acc_curve.png", dpi=200)
	plt.close()


@torch.no_grad()
def collect_predictions(
	model: nn.Module,
	loader: DataLoader,
	device: torch.device,
) -> Tuple[List[int], List[int]]:
	"""Thu thập toàn bộ nhãn thực tế và nhãn dự đoán từ DataLoader."""
	model.eval()
	y_true, y_pred = [], []
	for images, targets in tqdm(loader, desc="Predict"):
		images = images.to(device)
		logits = model(images)
		preds = torch.argmax(logits, dim=1).cpu().tolist()
		y_pred.extend(preds)
		y_true.extend(targets.tolist())
	return y_true, y_pred


def plot_confusion_matrix(
	y_true: List[int],
	y_pred: List[int],
	labels: List[str],
	title: str,
	save_path: Path,
) -> None:
	"""Vẽ ma trận nhầm lẫn (Confusion Matrix)."""
	cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
	plt.figure(figsize=(10, 8))
	plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(labels))
	plt.xticks(tick_marks, labels, rotation=45, ha="right")
	plt.yticks(tick_marks, labels)
	plt.ylabel("True label")
	plt.xlabel("Predicted label")
	plt.tight_layout()
	plt.savefig(save_path, dpi=200)
	plt.close()


def save_report(report: str, path: Path) -> None:
	"""Lưu văn bản báo cáo ra file txt UTF-8."""
	with open(path, "w", encoding="utf-8") as f:
		f.write(report)


def plot_misclassified_samples(
	loader: DataLoader,
	y_true: List[int],
	y_pred: List[int],
	class_names: List[str],
	output_dir: Path,
	max_images: int = 50,
) -> None:
	"""Vẽ lưới các ảnh dự đoán sai trong tập test kèm nhãn dự đoán và nhãn đúng (lọc trùng cặp lỗi, hàng 3 ảnh)."""
	df = getattr(loader.dataset, "df", None)
	if df is None:
		return

	misclassified_indices = [i for i, (t, p) in enumerate(zip(y_true, y_pred)) if t != p]
	if not misclassified_indices:
		print("  -> Không có ảnh nào bị dự đoán sai trên tập này!")
		return

	seen_errors = set()
	filtered_indices = []
	for idx in misclassified_indices:
		t = y_true[idx]
		p = y_pred[idx]
		error_pair = (t, p)
		if error_pair not in seen_errors:
			seen_errors.add(error_pair)
			filtered_indices.append(idx)

	print(f"  -> Tìm thấy {len(misclassified_indices)} ảnh bị dự đoán sai (sau khi lọc trùng cặp lỗi còn {len(filtered_indices)} ảnh).")

	save_dir = Path(output_dir) / "misclassified_test_samples"
	save_dir.mkdir(parents=True, exist_ok=True)

	selected_indices = filtered_indices[:max_images]
	imgs_per_fig = 15
	num_figs = (len(selected_indices) + imgs_per_fig - 1) // imgs_per_fig

	for fig_idx in range(num_figs):
		fig_indices = selected_indices[fig_idx * imgs_per_fig : (fig_idx + 1) * imgs_per_fig]
		n_imgs = len(fig_indices)

		cols = 3
		rows = (n_imgs + cols - 1) // cols

		fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
		if rows == 1:
			axes = np.expand_dims(axes, axis=0)
		if cols == 1:
			axes = np.expand_dims(axes, axis=-1)

		for idx, sample_idx in enumerate(fig_indices):
			r_idx = idx // cols
			c_idx = idx % cols

			row = df.iloc[sample_idx]
			img_path = row["path"]
			true_lbl = class_names[y_true[sample_idx]]
			pred_lbl = class_names[y_pred[sample_idx]]

			ax = axes[r_idx, c_idx]
			try:
				with Image.open(img_path) as img:
					ax.imshow(img)
			except Exception as e:
				ax.text(0.5, 0.5, f"Error\n{e}", ha="center", va="center")

			ax.axis("off")
			ax.set_title(f"True: {true_lbl}\nPred: {pred_lbl}", fontsize=8, color="red")

		for idx in range(n_imgs, rows * cols):
			r_idx = idx // cols
			c_idx = idx % cols
			axes[r_idx, c_idx].axis("off")

		plt.suptitle(f"Misclassified Test Samples - Part {fig_idx + 1}", fontsize=14, fontweight="bold")
		plt.tight_layout()
		plt.savefig(save_dir / f"misclassified_part_{fig_idx + 1}.png", dpi=200, bbox_inches="tight")
		plt.close()


def evaluate_and_report(
	model: nn.Module,
	loader: DataLoader,
	device: torch.device,
	class_names: List[str],
	output_dir: Path,
	prefix: str,
) -> None:
	"""Báo cáo đánh giá toàn diện cấp loài và cấp chi (Species & Genus Level)."""
	y_true, y_pred = collect_predictions(model, loader, device)
	labels = list(range(len(class_names)))

	if prefix == "test":
		plot_misclassified_samples(loader, y_true, y_pred, class_names, output_dir, max_images=50)

	report = classification_report(
		y_true, y_pred, labels=labels, target_names=class_names, digits=4
	)
	print(f"\n[{prefix.upper()}] Classification Report:\n{report}")
	save_report(report, output_dir / f"report_{prefix}.txt")

	plot_confusion_matrix(
		y_true,
		y_pred,
		class_names,
		f"Confusion Matrix ({prefix})",
		output_dir / f"confusion_matrix_{prefix}.png",
	)

	genus_labels = [split_genus_species(name)[0] for name in class_names]
	genus_names = sorted(list(set(genus_labels)))
	genus_to_idx = {g: i for i, g in enumerate(genus_names)}

	y_true_genus = [genus_to_idx[split_genus_species(class_names[i])[0]] for i in y_true]
	y_pred_genus = [genus_to_idx[split_genus_species(class_names[i])[0]] for i in y_pred]

	genus_report = classification_report(
		y_true_genus, y_pred_genus, target_names=genus_names, digits=4
	)
	print(f"\n[{prefix.upper()}] Genus Report:\n{genus_report}")
	save_report(genus_report, output_dir / f"report_{prefix}_genus.txt")

	plot_confusion_matrix(
		y_true_genus,
		y_pred_genus,
		genus_names,
		f"Confusion Matrix - Genus ({prefix})",
		output_dir / f"confusion_matrix_{prefix}_genus.png",
	)

	# Báo cáo nội bộ theo từng Chi (Intra-genus species breakdown)
	for genus in genus_names:
		indices = [
			i for i, true_idx in enumerate(y_true)
			if split_genus_species(class_names[true_idx])[0] == genus
		]
		if not indices:
			continue
		genus_true = [y_true[i] for i in indices]
		genus_pred = [y_pred[i] for i in indices]
		species_classes = sorted({class_names[idx] for idx in genus_true})
		pred_labels = []
		has_other_genus = False
		for idx in genus_pred:
			pred_label = class_names[idx]
			pred_genus = split_genus_species(pred_label)[0]
			if pred_genus == genus:
				pred_labels.append(pred_label)
			else:
				pred_labels.append("Other Genus")
				has_other_genus = True

		for label in pred_labels:
			if label != "Other Genus" and label not in species_classes:
				species_classes.append(label)

		species_classes = sorted(species_classes)
		if has_other_genus:
			species_classes.append("Other Genus")

		species_to_idx = {name: i for i, name in enumerate(species_classes)}
		mapped_true = [species_to_idx[class_names[idx]] for idx in genus_true]
		mapped_pred = [species_to_idx[label] for label in pred_labels]

		species_report = classification_report(
			mapped_true, mapped_pred, target_names=species_classes, digits=4
		)
		save_report(
			species_report,
			output_dir / f"report_{prefix}_species_{genus}.txt",
		)

		plot_confusion_matrix(
			mapped_true,
			mapped_pred,
			species_classes,
			f"Confusion Matrix - Species ({prefix}, {genus})",
			output_dir / f"confusion_matrix_{prefix}_species_{genus}.png",
		)

	# Log lên WandB nếu đang chạy session
	try:
		import wandb
		if wandb.run is not None:
			cm_path = output_dir / f"confusion_matrix_{prefix}.png"
			if cm_path.exists():
				wandb.log({f"Evaluation/CM_{prefix}": wandb.Image(str(cm_path))})
			cm_genus_path = output_dir / f"confusion_matrix_{prefix}_genus.png"
			if cm_genus_path.exists():
				wandb.log({f"Evaluation/CM_{prefix}_genus": wandb.Image(str(cm_genus_path))})
	except Exception:
		pass
