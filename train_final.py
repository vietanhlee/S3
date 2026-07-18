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
from PIL import Image

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


# SPLIT_CONFIG = {
# 		"Afzelia africana": ("PP8", "val", "eff"),
# 		"Afzelia bella": ("PP4", "val", "swin"),
# 		"Afzelia pachyloba": ("PP9", "test", "swin"),
# 		"Afzelia quanzensis": ("PP2", "val", "eff"),
# 		"Dalbergia cochinchinensis": ("PP9", "val", "eff"),
# 		"Dalbergia melanoxylon": ("PP2", "test", "eff"),
# 		"Dalbergia oliveri": ("PP8", "val", "eff"),
# 		"Dalbergia rimosa": ("PP4", "test", "eff"),
# 		"Dalbergia tonkinensis": ("PP4", "test", "swin"),
# 		"Guibourtia arnoldiana": ("PP4", "test", "swin"),
# 		"Guibourtia coleosperma": ("PP9", "test", "swin"),
# 		"Guibourtia ehie": ("PP4", "test", "swin"),
# 		"Peltogyne pubescens": ("PP2", "test", "eff"),
# 		"Pterocarpus erinaceus": ("PP9", "val", "eff"),
# 		"Pterocarpus indicus": ("PP9", "test", "eff"),
# 		"Pterocarpus macrocarpus": ("PP4", "test", "eff"),
# 		"Pterocarpus soyauxii": ("PP4", "test", "swin"),
# 		"Sindora cochinchinensis": ("PP2", "test", "swin"),
# 		"Sindora tonkinensis": ("PP9", "val", "eff"),
# 	}
	
SPLIT_CONFIG = {
  "Afzelia africana": ["PP8", "val", "eff"],
  "Afzelia bella": ["PP4", "test", "eff"],
  "Afzelia pachyloba": ["PP1", "test", "eff"],
  "Afzelia quanzensis": ["PP2", "val", "eff"],
  "Dalbergia cochinchinensis": ["PP1", "test", "eff"],
  "Dalbergia melanoxylon": ["PP8", "test", "eff"],
  "Dalbergia oliveri": ["PP1", "test", "eff"],
  "Dalbergia rimosa": ["PP1", "test", "eff"],
  "Dalbergia tonkinensis": ["PP4", "test", "swin"],
  "Guibourtia arnoldiana": ["PP4", "test", "eff"],
  "Guibourtia coleosperma": ["PP1", "test", "eff"],
  "Guibourtia ehie": ["PP5", "test", "eff"],
  "Peltogyne pubescens": ["PP1", "test", "eff"],
  "Pterocarpus erinaceus": ["PP9", "val", "eff"],
  "Pterocarpus indicus": ["PP9", "test", "eff"],
  "Pterocarpus macrocarpus": ["PP4", "test", "eff"],
  "Pterocarpus soyauxii": ["PP4", "test", "swin"],
  "Sindora cochinchinensis": ["PP2", "test", "eff"],
  "Sindora tonkinensis": ["PP7", "test", "eff"]
}

from utils import (
	set_seed,
	get_device,
	collect_image_samples,
	build_dataframe,
	log_split_summary,
	eda_split_class_distribution,
	ImageListDataset,
	ImagePathDataset,
	build_transforms,
	summarize_model,
	freeze_model_layers,
	validate_split_minimums,
	split_genus_species,
	CAM_METHODS,
)

from split_methods import (
	SPLIT_METHODS,
	validate_split,
)

import timm
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# ============================================================
# Lớp & Hàm Phục Vụ Huấn Luyện Phân Loại (Classification)
# ============================================================

class FocalLoss(nn.Module):
	def __init__(self, gamma: float = 2.0, alpha: float = 0.25) -> None:
		super().__init__()
		self.gamma = gamma
		self.alpha = alpha

	def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
		ce = F.cross_entropy(logits, targets, reduction="none")
		pt = torch.exp(-ce)
		loss = self.alpha * (1 - pt) ** self.gamma * ce
		return loss.mean()


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
	preds = torch.argmax(logits, dim=1)
	correct = (preds == targets).sum().item()
	return correct / len(targets)


def train_one_epoch(
	model: nn.Module,
	loader: DataLoader,
	criterion: nn.Module,
	optimizer: torch.optim.Optimizer,
	device: torch.device,
	epoch: int,
	epochs: int,
) -> tuple[float, float]:
	model.train()
	running_loss, running_acc, count = 0.0, 0.0, 0
	pbar = tqdm(loader, desc=f"Train {epoch}/{epochs}")
	for images, targets in pbar:
		images = images.to(device)
		targets = targets.to(device)

		optimizer.zero_grad()
		logits = model(images)
		loss = criterion(logits, targets)
		loss.backward()
		optimizer.step()

		batch_size = targets.size(0)
		running_loss += loss.item() * batch_size
		running_acc += accuracy_from_logits(logits, targets) * batch_size
		count += batch_size
		pbar.set_postfix(loss=running_loss / count, acc=running_acc / count)

	return running_loss / count, running_acc / count


@torch.no_grad()
def evaluate_one_epoch(
	model: nn.Module,
	loader: DataLoader,
	criterion: nn.Module,
	device: torch.device,
	epoch: int,
	epochs: int,
) -> tuple[float, float]:
	model.eval()
	running_loss, running_acc, count = 0.0, 0.0, 0
	pbar = tqdm(loader, desc=f"Val {epoch}/{epochs}")
	for images, targets in pbar:
		images = images.to(device)
		targets = targets.to(device)

		logits = model(images)
		loss = criterion(logits, targets)
		batch_size = targets.size(0)
		running_loss += loss.item() * batch_size
		running_acc += accuracy_from_logits(logits, targets) * batch_size
		count += batch_size
		pbar.set_postfix(loss=running_loss / count, acc=running_acc / count)

	return running_loss / count, running_acc / count


def save_checkpoint(
	path: Path,
	model: nn.Module,
	optimizer: torch.optim.Optimizer,
	epoch: int,
	best_val_acc: float,
	history: dict,
) -> None:
	raw_model = model
	payload = {
		"epoch": epoch,
		"model_state": raw_model.state_dict(),
		"optimizer_state": optimizer.state_dict(),
		"best_val_acc": best_val_acc,
		"history": history,
		"model_name": getattr(raw_model, "model_name", "model"),
	}
	torch.save(payload, path)


def train_model(
	model: nn.Module,
	train_loader: DataLoader,
	val_loader: DataLoader,
	optimizer: torch.optim.Optimizer,
	criterion: nn.Module,
	device: torch.device,
	epochs: int,
	patience: int,
	output_dir: Path,
	scheduler=None,
) -> dict:
	history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
	best_val_acc = 0.0
	epochs_no_improve = 0
	raw_model = model
	model_name = getattr(raw_model, "model_name", "model")

	for epoch in range(1, epochs + 1):
		train_loss, train_acc = train_one_epoch(
			model, train_loader, criterion, optimizer, device, epoch, epochs
		)
		val_loss, val_acc = evaluate_one_epoch(
			model, val_loader, criterion, device, epoch, epochs
		)

		history["train_loss"].append(train_loss)
		history["train_acc"].append(train_acc)
		history["val_loss"].append(val_loss)
		history["val_acc"].append(val_acc)

		current_lr = optimizer.param_groups[0]["lr"]
		print(
			f"Epoch {epoch}/{epochs} - "
			f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
			f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, "
			f"lr={current_lr:.6f}"
		)

		# Log to WandB if active
		try:
			import wandb
			if wandb.run is not None:
				wandb.log({
					"epoch": epoch,
					"train/loss": train_loss,
					"train/acc": train_acc,
					"val/loss": val_loss,
					"val/acc": val_acc,
					"lr": current_lr
				}, step=epoch)
		except Exception as e:
			pass

		if scheduler is not None:
			scheduler.step()

		last_path = output_dir / f"last_epoch.pth"
		save_checkpoint(last_path, model, optimizer, epoch, best_val_acc, history)

		if val_acc > best_val_acc:
			best_val_acc = val_acc
			best_path = output_dir / f"best_model_{model_name}.pth"
			torch.save(raw_model.state_dict(), best_path)
			epochs_no_improve = 0
		else:
			epochs_no_improve += 1

		if epochs_no_improve >= patience:
			print(f"Early stopping at epoch {epoch} (patience {patience})")
			break

	history["best_val_acc"] = best_val_acc
	return history


def plot_training_curves(history: dict, output_dir: Path) -> None:
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
	model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[list[int], list[int]]:
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
	y_true: list[int],
	y_pred: list[int],
	labels: list[str],
	title: str,
	save_path: Path,
) -> None:
	from sklearn.metrics import confusion_matrix
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
	with open(path, "w", encoding="utf-8") as f:
		f.write(report)


def plot_misclassified_samples(
	loader: DataLoader,
	y_true: list[int],
	y_pred: list[int],
	class_names: list[str],
	output_dir: Path,
	max_images: int = 50,
) -> None:
	"""Vẽ lưới các ảnh dự đoán sai trong tập test kèm nhãn dự đoán và nhãn đúng (lọc trùng cặp lỗi, hàng 3 ảnh)."""
	df = loader.dataset.df
	misclassified_indices = [i for i, (t, p) in enumerate(zip(y_true, y_pred)) if t != p]

	if not misclassified_indices:
		print("  -> Không có ảnh nào bị dự đoán sai trên tập này!")
		return

	# Lọc để mỗi cặp lỗi (Nhãn đúng, Nhãn dự đoán sai) chỉ xuất hiện tối đa 1 lần
	seen_errors = set()
	filtered_indices = []
	for idx in misclassified_indices:
		t = y_true[idx]
		p = y_pred[idx]
		error_pair = (t, p)
		if error_pair not in seen_errors:
			seen_errors.add(error_pair)
			filtered_indices.append(idx)

	print(f"  -> Tìm thấy {len(misclassified_indices)} ảnh bị dự đoán sai (sau khi lọc trùng cặp lỗi còn {len(filtered_indices)} ảnh). Đang vẽ tối đa {min(max_images, len(filtered_indices))} ảnh...")

	save_dir = Path(output_dir) / "misclassified_test_samples"
	save_dir.mkdir(parents=True, exist_ok=True)

	# Lấy tối đa max_images ảnh bị dự đoán sai sau khi lọc
	selected_indices = filtered_indices[:max_images]
	imgs_per_fig = 15  # 5 hàng x 3 cột
	num_figs = (len(selected_indices) + imgs_per_fig - 1) // imgs_per_fig

	for fig_idx in range(num_figs):
		fig_indices = selected_indices[fig_idx * imgs_per_fig : (fig_idx + 1) * imgs_per_fig]
		n_imgs = len(fig_indices)

		cols = 3
		rows = (n_imgs + cols - 1) // cols

		fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
		# Đảm bảo axes luôn là mảng 2 chiều
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
			# Chú thích nhãn dự đoán và nhãn đúng
			ax.set_title(f"True: {true_lbl}\nPred: {pred_lbl}", fontsize=8, color="red")

		# Ẩn các trục trống nếu lưới không được lấp đầy
		for idx in range(n_imgs, rows * cols):
			r_idx = idx // cols
			c_idx = idx % cols
			axes[r_idx, c_idx].axis("off")

		plt.suptitle(f"Misclassified Test Samples - Part {fig_idx + 1}", fontsize=14, fontweight="bold")
		plt.tight_layout()
		plt.savefig(save_dir / f"misclassified_part_{fig_idx + 1}.png", dpi=200, bbox_inches="tight")
		plt.close()
		print(f"  -> Đã lưu figure ghép tại: {save_dir / f'misclassified_part_{fig_idx + 1}.png'}")


def evaluate_and_report(
	model: nn.Module,
	loader: DataLoader,
	device: torch.device,
	class_names: list[str],
	output_dir: Path,
	prefix: str,
) -> None:
	y_true, y_pred = collect_predictions(model, loader, device)
	labels = list(range(len(class_names)))

	if prefix == "test":
		plot_misclassified_samples(loader, y_true, y_pred, class_names, output_dir, max_images=50)

	report = classification_report(
		y_true, y_pred, labels=labels, target_names=class_names, digits=4
	)
	print(f"\n{prefix} classification report:\n{report}")
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
	print(f"\n{prefix} genus report:\n{genus_report}")
	save_report(genus_report, output_dir / f"report_{prefix}_genus.txt")

	plot_confusion_matrix(
		y_true_genus,
		y_pred_genus,
		genus_names,
		f"Confusion Matrix - Genus ({prefix})",
		output_dir / f"confusion_matrix_{prefix}_genus.png",
	)

	for genus in genus_names:
		indices = [
			i
			for i, true_idx in enumerate(y_true)
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
		print(f"\n{prefix} species report for genus {genus}:\n{species_report}")
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

	# Log confusion matrices to WandB if active
	try:
		import wandb
		if wandb.run is not None:
			# Log main confusion matrix
			cm_path = output_dir / f"confusion_matrix_{prefix}.png"
			if cm_path.exists():
				wandb.log({f"Evaluation/CM_{prefix}": wandb.Image(str(cm_path))})
			# Log genus confusion matrix
			cm_genus_path = output_dir / f"confusion_matrix_{prefix}_genus.png"
			if cm_genus_path.exists():
				wandb.log({f"Evaluation/CM_{prefix}_genus": wandb.Image(str(cm_genus_path))})
			# Log species confusion matrix per genus
			for genus in genus_names:
				cm_spec_path = output_dir / f"confusion_matrix_{prefix}_species_{genus}.png"
				if cm_spec_path.exists():
					wandb.log({f"Evaluation/CM_{prefix}_species_{genus}": wandb.Image(str(cm_spec_path))})
	except Exception:
		pass


class GradCAM:
	def __init__(self, model: nn.Module, target_layer: nn.Module, method: str = "gradcam") -> None:
		self.model = model
		self.target_layer = target_layer
		self.method = method.lower()
		self.activations = None
		self.forward_handle = target_layer.register_forward_hook(self._forward_hook)

	def _forward_hook(self, module, inputs, output):
		self.activations = output

	def remove(self) -> None:
		self.forward_handle.remove()

	def __call__(self, input_tensor: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
		if self.method == "eigencam":
			self.model.eval()
			with torch.no_grad():
				_ = self.model(input_tensor)
			act = self.activations.squeeze(0).detach().cpu().numpy()
			c, h, w = act.shape
			A = act.reshape(c, h * w).T
			A = A - np.mean(A, axis=0)
			U, S, Vt = np.linalg.svd(A, full_matrices=False)
			projection = (A @ Vt[0, :]).reshape(h, w)
			if np.sum(projection) < 0:
				projection = -projection
			cam = np.maximum(projection, 0)
		else:
			self.model.zero_grad()
			if self.method == "finercam":
				output = self.model(input_tensor)
				if class_idx is None:
					class_idx = int(torch.argmax(output, dim=1).item())
				prob = torch.softmax(output, dim=-1)
				output_data = output[0].detach().cpu().numpy()
				target_logit = output_data[class_idx]
				
				sorted_indices = np.argsort(np.abs(output_data - target_logit))
				comparison_categories = sorted_indices[1:4]
				alpha = 1.0
				
				wn = output[0, class_idx]
				weights = [prob[0, idx] for idx in comparison_categories]
				numerator = sum(w * (wn - alpha * output[0, idx]) for w, idx in zip(weights, comparison_categories))
				denominator = sum(weights)
				score = numerator / (denominator + 1e-9)
			else:
				output = self.model(input_tensor)
				if class_idx is None:
					class_idx = int(torch.argmax(output, dim=1).item())
				score = output[:, class_idx].sum()

			if self.activations is None:
				raise RuntimeError("GradCAM hook did not capture activations")

			grads = torch.autograd.grad(score, self.activations, retain_graph=True)[0]
			
			if self.method == "gradcam++":
				grads_pos = torch.clamp(grads, min=0)
				grads_power_2 = grads_pos ** 2
				grads_power_3 = grads_pos ** 3
				sum_activations = torch.sum(self.activations, dim=(2, 3), keepdim=True)
				eps = 1e-7
				aij = grads_power_2 / (2 * grads_power_2 + sum_activations * grads_power_3 + eps)
				weights = torch.sum(aij * grads_pos, dim=(2, 3), keepdim=True)
				cam = torch.sum(weights * self.activations, dim=1)
				cam = F.relu(cam)
				cam = cam.squeeze().detach().cpu().numpy()
			elif self.method == "xgradcam":
				sum_activations = torch.sum(self.activations, dim=(2, 3), keepdim=True) + 1e-7
				weights = torch.sum(grads * self.activations / sum_activations, dim=(2, 3), keepdim=True)
				cam = torch.sum(weights * self.activations, dim=1)
				cam = F.relu(cam)
				cam = cam.squeeze().detach().cpu().numpy()
			elif self.method == "hirescam":
				cam = torch.sum(grads * self.activations, dim=1)
				cam = F.relu(cam)
				cam = cam.squeeze().detach().cpu().numpy()
			elif self.method == "layercam":
				cam = torch.sum(torch.clamp(grads, min=0) * self.activations, dim=1)
				cam = F.relu(cam)
				cam = cam.squeeze().detach().cpu().numpy()
			elif self.method == "eigengradcam":
				weighted_act = grads * self.activations
				act = weighted_act.squeeze(0).detach().cpu().numpy()
				c, h, w = act.shape
				A = act.reshape(c, h * w).T
				A = A - np.mean(A, axis=0)
				U, S, Vt = np.linalg.svd(A, full_matrices=False)
				projection = (A @ Vt[0, :]).reshape(h, w)
				if np.sum(projection) < 0:
					projection = -projection
				cam = np.maximum(projection, 0)
			else:
				weights = grads.mean(dim=(2, 3), keepdim=True)
				cam = (weights * self.activations).sum(dim=1, keepdim=True)
				cam = F.relu(cam)
				cam = cam.squeeze().detach().cpu().numpy()

		if cam.ndim == 0:
			cam = np.array([[float(cam)]])
		elif cam.ndim == 1:
			cam = cam[None, :]
		
		cam -= cam.min()
		if cam.max() > 0:
			cam /= cam.max()
		return cam


def save_gradcam_samples(
	model: nn.Module,
	df: pd.DataFrame,
	eval_tf,
	device: torch.device,
	output_dir: Path,
	num_samples: int = 8,
) -> None:
	from utils import find_last_conv_layer, overlay_cam_on_image
	target_layer = find_last_conv_layer(model)
	if target_layer is None:
		print("GradCAM skipped: no Conv2d layer found in model")
		return

	model.eval()
	count = min(num_samples, len(df))
	if count == 0:
		return
	batch = df.sample(n=count, random_state=SEED).reset_index(drop=True)

	for method in CAM_METHODS:
		method_dir = output_dir / method
		method_dir.mkdir(parents=True, exist_ok=True)

		gradcam = GradCAM(model, target_layer, method=method)
		for i, row in batch.iterrows():
			with Image.open(row["path"]) as img:
				img = img.convert("RGB")
			input_tensor = eval_tf(img).unsqueeze(0).to(device)
			cam = gradcam(input_tensor)
			overlay = overlay_cam_on_image(img, cam)
			label = str(row["label"]).replace(" ", "_")
			out_path = method_dir / f"gradcam_{i}_{label}.png"
			overlay.save(out_path)
		gradcam.remove()
		print(f"Đã lưu ảnh giải thích mô hình cho phương pháp: {method} → {method_dir}/")


# ============================================================
# Các hàm khởi tạo Model cho Classification
# ============================================================

def _create_timm_model(model_name: str, num_classes: int, freeze_ratio: float) -> nn.Module:
	model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
	freeze_model_layers(model, freeze_ratio)
	model.model_name = model_name
	return model

def build_mobilenet_large(num_classes: int) -> nn.Module:
	return _create_timm_model("mobilenetv3_large_100", num_classes, freeze_ratio=0.90)

def build_efficientnet_large(num_classes: int) -> nn.Module:
	return _create_timm_model("tf_efficientnet_b4", num_classes, freeze_ratio=0.90)

def build_resnet_large(num_classes: int) -> nn.Module:
	return _create_timm_model("resnet50", num_classes, freeze_ratio=0.90)

def build_convnext_large(num_classes: int) -> nn.Module:
	return _create_timm_model("convnext_tiny", num_classes, freeze_ratio=0.97)

def build_vit_large(num_classes: int) -> nn.Module:
	return _create_timm_model("vit_base_patch16_224", num_classes, freeze_ratio=0.97)

def build_deit_base(num_classes: int) -> nn.Module:
	return _create_timm_model("deit_base_patch16_224", num_classes, freeze_ratio=0.90)

def build_swin_large(num_classes: int) -> nn.Module:
	return _create_timm_model("swin_large_patch4_window7_224", num_classes, freeze_ratio=0.97)

def build_beit_large(num_classes: int) -> nn.Module:
	return _create_timm_model("beit_large_patch16_224", num_classes, freeze_ratio=0.97)


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
	split_config = SPLIT_CONFIG

	pp_map = {
		"PP1": "PP1_Mahalanobis_Fixed",
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

	# Khởi tạo WandB
	use_wandb = False
	try:
		from dotenv import load_dotenv
		import wandb
		load_dotenv()
		api_key = os.getenv("WANDB_API_KEY")
		if api_key:
			wandb.login(key=api_key)
			model_prefix = MODEL_NAME.lower().replace("_", "-")
			run_name = f"{model_prefix}-classification"
			config_dict = {
				"model_name": MODEL_NAME,
				"epochs": EPOCHS,
				"lr": LR,
				"weight_decay": WEIGHT_DECAY,
				"focal_gamma": FOCAL_GAMMA,
				"focal_alpha": FOCAL_ALPHA,
				"freeze_ratio": FREEZE_RATIO,
				"train_ratio": TRAIN_RATIO,
				"val_ratio": VAL_RATIO,
				"batch_size": BATCH_SIZE,
				"seed": SEED,
				"split_config": SPLIT_CONFIG,
			}
			wandb.init(
				project="S3-Wood-Classification-Base",
				name=run_name,
				config=config_dict
			)
			use_wandb = True
			print(f"[Wandb Info] Khởi tạo thành công: project='S3-Wood-Recognition', run='{run_name}'")
		else:
			print("[Wandb Warning] WANDB_API_KEY không tồn tại trong .env. Chạy không có WandB.")
	except Exception as e:
		print(f"[Wandb Warning] Lỗi khởi tạo WandB: {e}. Chạy không có WandB.")

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

	num_workers = 4
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
	if use_wandb:
		try:
			import wandb
			if wandb.run is not None:
				curves_path = output_dir / "training_curves.png"
				if curves_path.exists():
					wandb.log({"Evaluation/Training_Curves": wandb.Image(str(curves_path))})
		except Exception:
			pass

	# Load best model checkpoint để đánh giá
	best_path = output_dir / f"best_model_{MODEL_NAME}.pth"
	if best_path.exists():
		raw_model = model
		raw_model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
		print(f"\nĐã load checkpoint tốt nhất từ {best_path}")

	# Đánh giá trên tập Val
	print("\nĐánh giá trên tập Val...")
	evaluate_and_report(model, val_loader, device, class_names, output_dir, prefix="val")

	# Đánh giá trên tập Test
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

	# ── Upload artifacts lên WandB ───────────────────────
	if use_wandb:
		try:
			import wandb
			artifact_name = "classification-artifacts"
			artifact = wandb.Artifact(name=artifact_name, type="model_and_reports")
			
			# Log best model
			best_path = output_dir / f"best_model_{MODEL_NAME}.pth"
			if best_path.exists():
				artifact.add_file(str(best_path), name=f"best_model_{MODEL_NAME}.pth")
				
			# Log các file report text
			for txt_file in output_dir.glob("*.txt"):
				artifact.add_file(str(txt_file), name=txt_file.name)
			# Log final summary JSON
			final_summary = output_dir / "final_summary.json"
			if final_summary.exists():
				artifact.add_file(str(final_summary), name="final_summary.json")
				
			wandb.log_artifact(artifact)
			print(f"[Wandb Info] Đã upload artifact '{artifact_name}' thành công.")
		except Exception as e:
			print(f"[Wandb Warning] Lỗi khi upload artifacts: {e}")
		finally:
			wandb.finish()

	print(f"\n[Hoàn tất] Tất cả kết quả huấn luyện đã lưu tại thư mục: {output_dir}")


if __name__ == "__main__":
	main()
