import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.cm as cm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import timm
from timm.data import resolve_data_config

from sklearn.metrics import confusion_matrix, classification_report


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

SEED = 42
BATCH_SIZE = 128
EPOCHS = 20
PATIENCE = 9
LR = 1e-4
WEIGHT_DECAY = 1e-2
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.25
CAM_METHODS = ["gradcam", "gradcam++", "xgradcam", "eigencam", "hirescam", "layercam", "eigengradcam"]


def set_seed(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
	return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def split_genus_species(label: str) -> tuple[str, str]:
	parts = label.split()
	genus = parts[0] if len(parts) > 0 else "Unknown"
	species = parts[1] if len(parts) > 1 else "Unknown"
	return genus, species


def collect_image_samples(root_dir: str) -> list[dict]:
	root = Path(root_dir)
	if not root.exists():
		raise FileNotFoundError(f"Root dir not found: {root_dir}")

	samples = []
	class_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
	for class_dir in tqdm(class_dirs, desc="Scan classes"):
		label = class_dir.name
		files = list(class_dir.rglob("*"))
		for path in tqdm(files, desc=f"Scan {label}", leave=False):
			if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
				samples.append({"path": str(path), "label": label})
	return samples


def build_dataframe(samples: list[dict]) -> pd.DataFrame:
	rows = []
	for sample in samples:
		genus, species = split_genus_species(sample["label"])
		path = Path(sample["path"])
		label = sample["label"]
		parts = path.parts
		try:
			label_idx = parts.index(label)
			# Chỉ lấy ảnh nếu nó nằm trong subfolder thực sự của class
			if label_idx + 1 < len(parts) - 1:
				subfolder = parts[label_idx + 1]
				rows.append(
					{
						"path": sample["path"],
						"label": label,
						"genus": genus,
						"species": species,
						"subfolder": subfolder,
					}
				)
		except ValueError:
			pass
	return pd.DataFrame(rows)


def eda_class_distribution(df: pd.DataFrame, title: str, save_path: Path | None) -> None:
	counts = df["label"].value_counts().sort_index()
	print(f"\n{title} - class counts:\n{counts.to_string()}")
	plt.figure(figsize=(12, 6))
	counts.plot(kind="bar")
	plt.title(title)
	plt.xlabel("Class")
	plt.ylabel("Image count")
	plt.xticks(rotation=45, ha="right")
	plt.tight_layout()
	if save_path:
		plt.savefig(save_path, dpi=200)
	else:
		plt.show()
	plt.close()


def eda_split_class_distribution(
	df_train: pd.DataFrame,
	df_val: pd.DataFrame,
	df_test: pd.DataFrame,
	title: str,
	save_path: Path | None,
) -> None:
	labels = sorted(
		set(df_train["label"]) | set(df_val["label"]) | set(df_test["label"])
	)
	train_counts = df_train["label"].value_counts().reindex(labels, fill_value=0)
	val_counts = df_val["label"].value_counts().reindex(labels, fill_value=0)
	test_counts = df_test["label"].value_counts().reindex(labels, fill_value=0)

	stacked = pd.DataFrame(
		{"train": train_counts, "val": val_counts, "test": test_counts},
		index=labels,
	)

	plt.figure(figsize=(12, 6))
	stacked.plot(
		kind="bar",
		stacked=True,
		ax=plt.gca(),
		color=["#1f77b4", "#ff7f0e", "#2ca02c"],
	)
	ax = plt.gca()
	for i, label in enumerate(labels):
		values = stacked.loc[label]
		total = values.sum()
		if total == 0:
			continue
		bottom = 0.0
		for part in ["train", "val", "test"]:
			value = float(values[part])
			if value > 0:
				percent = value / total * 100
				y = bottom + value / 2
				ax.text(
					i,
					y,
					f"{percent:.1f}%",
					ha="center",
					va="center",
					fontsize=5,
					color="white",
				)
			bottom += value

	# Tính tỉ lệ phần trăm tổng thể Train/Val/Test
	total_train = len(df_train)
	total_val = len(df_val)
	total_test = len(df_test)
	total_all = total_train + total_val + total_test

	p_train = total_train / total_all * 100 if total_all > 0 else 0.0
	p_val = total_val / total_all * 100 if total_all > 0 else 0.0
	p_test = total_test / total_all * 100 if total_all > 0 else 0.0

	overall_text = (
		f"Overall split ratio:\n"
		f"  Train: {total_train} ({p_train:.1f}%)\n"
		f"  Val:   {total_val} ({p_val:.1f}%)\n"
		f"  Test:  {total_test} ({p_test:.1f}%)"
	)

	# Đặt text box ở góc trên bên trái biểu đồ
	props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
	ax.text(
		0.02, 0.95, overall_text, transform=ax.transAxes, fontsize=9,
		verticalalignment='top', bbox=props
	)

	plt.title(f"{title} (Total: {total_all})")
	plt.xlabel("Class")
	plt.ylabel("Image count")
	plt.xticks(rotation=45, ha="right")
	plt.tight_layout()
	if save_path:
		plt.savefig(save_path, dpi=200)
	else:
		plt.show()
	plt.close()


def eda_genus_distribution(df: pd.DataFrame, title: str, save_path: Path | None) -> None:
	counts = df["genus"].value_counts().sort_index()
	print(f"\n{title} - genus counts:\n{counts.to_string()}")

	genus_species = (
		df.groupby(["genus", "label"]).size().unstack(fill_value=0).sort_index()
	)

	# Tạo nhiều màu khác nhau
	n_species = len(genus_species.columns)
	colors = plt.colormaps["tab20"](
		np.linspace(0, 1, n_species)
	)

	plt.figure(figsize=(12, 6))

	genus_species.plot(
		kind="bar",
		stacked=True,
		ax=plt.gca(),
		color=colors
	)
	ax = plt.gca()
	for i, genus in enumerate(genus_species.index):
		values = genus_species.loc[genus]
		total = values.sum()
		if total == 0:
			continue
		bottom = 0.0
		for species in genus_species.columns:
			value = float(values[species])
			if value > 0:
				percent = value / total * 100
				y = bottom + value / 2
				ax.text(
					i,
					y,
					f"{percent:.1f}%",
					ha="center",
					va="center",
					fontsize= 10,
					color="white",
				)
			bottom += value

	plt.title(title)
	plt.xlabel("Genus")
	plt.ylabel("Image count")
	plt.xticks(rotation=45, ha="right")

	plt.legend(
		title="Species",
		bbox_to_anchor=(1.02, 1),
		loc="upper left"
	)

	plt.tight_layout()

	if save_path:
		plt.savefig(save_path, dpi=200)
	else:
		plt.show()

	plt.close()


def stratified_split(
	df: pd.DataFrame,
	train_ratio: float = 0.7,
	val_ratio: float = 0.15,
	seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	train_idx, val_idx, test_idx = [], [], []
	for label, group in df.groupby("label"):
		if "subfolder" not in group.columns:
			raise ValueError("Missing 'subfolder' column; rebuild dataframe with build_dataframe().")
		
		subfolder_groups = group.groupby("subfolder")
		
		# Get counts and sort ascending by count, then ascending by name for determinism
		subfolder_counts = group["subfolder"].value_counts().rename_axis("subfolder").reset_index(name="count")
		subfolder_counts = subfolder_counts.sort_values(by=["count", "subfolder"], ascending=[True, True])
		
		n_total = len(group)
		target_train = int(n_total * train_ratio)
		target_val = int(n_total * val_ratio)
		
		current_train = 0
		current_val = 0
		
		for _, row in subfolder_counts.iterrows():
			subfolder = row["subfolder"]
			count = row["count"]
			indices = subfolder_groups.get_group(subfolder).index.tolist()
			
			if current_train < target_train:
				train_idx.extend(indices)
				current_train += count
			elif current_val < target_val:
				val_idx.extend(indices)
				current_val += count
			else:
				test_idx.extend(indices)

	df_train = df.loc[train_idx].sample(frac=1, random_state=seed).reset_index(drop=True)
	df_val = df.loc[val_idx].sample(frac=1, random_state=seed).reset_index(drop=True)
	df_test = df.loc[test_idx].sample(frac=1, random_state=seed).reset_index(drop=True)
	return df_train, df_val, df_test


def plot_split_distributions(
	df_train: pd.DataFrame,
	df_val: pd.DataFrame,
	df_test: pd.DataFrame,
	output_dir: Path,
) -> None:
	for name, df in [("train", df_train), ("val", df_val), ("test", df_test)]:
		eda_genus_distribution(
			df, f"{name} split genus distribution", output_dir / f"eda_{name}_genus.png"
		)


def log_split_summary(
	df_all: pd.DataFrame,
	df_train: pd.DataFrame,
	df_val: pd.DataFrame,
	df_test: pd.DataFrame,
) -> None:
	print("\nDataset split summary:")
	print(f"  total: {len(df_all)}")
	print(f"  train: {len(df_train)}")
	print(f"  val:   {len(df_val)}")
	print(f"  test:  {len(df_test)}")


class ImageListDataset(Dataset):
	def __init__(self, df: pd.DataFrame, class_to_idx: dict, transform=None) -> None:
		self.df = df.reset_index(drop=True)
		self.class_to_idx = class_to_idx
		self.transform = transform

	def __len__(self) -> int:
		return len(self.df)

	def __getitem__(self, idx: int):
		row = self.df.iloc[idx]
		with Image.open(row["path"]) as img:
			img = img.convert("RGB")
		label = self.class_to_idx[row["label"]]
		if self.transform:
			img = self.transform(img)
		return img, label


class ImagePathDataset(Dataset):
	def __init__(self, df: pd.DataFrame, transform=None) -> None:
		self.df = df.reset_index(drop=True)
		self.transform = transform

	def __len__(self) -> int:
		return len(self.df)

	def __getitem__(self, idx: int):
		row = self.df.iloc[idx]
		with Image.open(row["path"]) as img:
			img = img.convert("RGB")
		if self.transform:
			img = self.transform(img)
		return img


def build_transforms(img_size: int, mean, std):
	train_tf = transforms.Compose(
		[
			transforms.Resize((img_size, img_size)),
			transforms.RandomRotation(degrees=15),
			transforms.RandomHorizontalFlip(p=0.5),
			transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
			transforms.ToTensor(),
			transforms.Normalize(mean=mean, std=std),
		]
	)
	eval_tf = transforms.Compose(
		[
			transforms.Resize((img_size, img_size)),
			transforms.ToTensor(),
			transforms.Normalize(mean=mean, std=std),
		]
	)
	return train_tf, eval_tf


def build_embedding_model() -> nn.Module:
	"""Tạo model embedding (swin_large_patch4_window7_224) phục vụ trích xuất đặc trưng."""
	model = timm.create_model(
		"swin_large_patch4_window7_224",
		pretrained=True,
		num_classes=0,
	)
	model.eval()
	return model

# def build_embedding_model() -> nn.Module:
# 	"""Tạo model embedding (tf_efficientnetv2_m_in21k) phục vụ trích xuất đặc trưng."""
# 	model = timm.create_model(
# 		"tf_efficientnetv2_m_in21k",
# 		pretrained=True,
# 		num_classes=0,
# 		global_pool="avg",
# 	)
# 	model.eval()
# 	return model


def build_embedding_transform(model: nn.Module) -> transforms.Compose:
	cfg = resolve_data_config({}, model=model)
	img_size = cfg.get("input_size", (3, 224, 224))[-1]
	mean = cfg.get("mean", (0.485, 0.456, 0.406))
	std = cfg.get("std", (0.229, 0.224, 0.225))
	return transforms.Compose(
		[
			transforms.Resize((img_size, img_size)),
			transforms.ToTensor(),
			transforms.Normalize(mean=mean, std=std),
		]
	)


def compute_embeddings(
	df: pd.DataFrame,
	batch_size: int,
	device: torch.device,
) -> np.ndarray:
	model = build_embedding_model().to(device)
	model.eval()
	transform = build_embedding_transform(model)

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
		for images in tqdm(loader, desc="Embed"):
			images = images.to(device)
			feats = model(images)
			features.append(feats.detach().cpu().numpy())

	if not features:
		return np.empty((0, 0), dtype=np.float32)
	return np.concatenate(features, axis=0)


def compute_split_counts(
	n_total: int,
	train_ratio: float,
	val_ratio: float,
) -> tuple[int, int, int]:
	if n_total <= 0:
		return 0, 0, 0

	test_ratio = 1.0 - train_ratio - val_ratio
	if test_ratio < 0:
		raise ValueError("train_ratio + val_ratio must be <= 1.0")

	val_count = int(n_total * val_ratio)
	test_count = int(n_total * test_ratio)
	train_count = n_total - val_count - test_count

	if n_total >= 3:
		if val_count == 0:
			if train_count > 1:
				train_count -= 1
				val_count = 1
			elif test_count > 1:
				test_count -= 1
				val_count = 1

		if test_count == 0:
			if train_count > 1:
				train_count -= 1
				test_count = 1
			elif val_count > 1:
				val_count -= 1
				test_count = 1

	return train_count, val_count, test_count


def mahalanobis_distances(embeddings: np.ndarray, eps: float = 1e-6) -> np.ndarray:
	n_samples = embeddings.shape[0]
	if n_samples <= 1:
		return np.zeros(n_samples, dtype=np.float32)

	mean = embeddings.mean(axis=0)
	cov = np.cov(embeddings, rowvar=False)
	cov = np.atleast_2d(cov)
	cov += np.eye(cov.shape[0]) * eps
	cov_inv = np.linalg.pinv(cov)

	diff = embeddings - mean
	d2 = np.einsum("ij,jk,ik->i", diff, cov_inv, diff)
	d2 = np.maximum(d2, 0.0)
	return np.sqrt(d2).astype(np.float32)


def pick_farthest_iterative(
	embeddings: np.ndarray,
	indices: list[int],
	pick_count: int,
	eps: float,
) -> tuple[list[int], list[int]]:
	remaining = list(indices)
	picked = []
	for _ in range(min(pick_count, len(remaining))):
		subset = embeddings[np.array(remaining)]
		dists = mahalanobis_distances(subset, eps=eps)
		max_dist = float(dists.max()) if len(dists) > 0 else 0.0
		candidate_pos = np.flatnonzero(dists == max_dist)
		if len(candidate_pos) > 1:
			candidate_indices = [remaining[pos] for pos in candidate_pos]
			picked_idx = min(candidate_indices)
			pick_pos = candidate_pos[candidate_indices.index(picked_idx)]
		else:
			pick_pos = int(candidate_pos[0]) if len(candidate_pos) else 0
			picked_idx = remaining[pick_pos]

		picked.append(picked_idx)
		remaining.pop(pick_pos)

	return picked, remaining


def mahalanobis_split_by_class(
	df: pd.DataFrame,
	embeddings: np.ndarray,
	train_ratio: float = 0.7,
	val_ratio: float = 0.15,
	seed: int = SEED,
	eps: float = 1e-6,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	if len(df) != embeddings.shape[0]:
		raise ValueError("Embeddings length does not match dataframe length")

	train_idx: list[int] = []
	val_idx: list[int] = []
	test_idx: list[int] = []

	class_count = int(df["label"].nunique())
	for _, group in tqdm(
		df.groupby("label"),
		total=class_count,
		desc="Split by class",
		unit="class",
	):
		indices = sorted(group.index.tolist())
		n_total = len(indices)
		train_count, val_count, test_count = compute_split_counts(
			n_total, train_ratio, val_ratio
		)
		if n_total == 0:
			continue

		picked_test, remaining = pick_farthest_iterative(
			embeddings, indices, test_count, eps
		)
		picked_val, remaining = pick_farthest_iterative(
			embeddings, remaining, val_count, eps
		)

		train_idx.extend(remaining)
		val_idx.extend(picked_val)
		test_idx.extend(picked_test)

	df_train = df.loc[train_idx].sample(frac=1, random_state=seed).reset_index(drop=True)
	df_val = df.loc[val_idx].sample(frac=1, random_state=seed).reset_index(drop=True)
	df_test = df.loc[test_idx].sample(frac=1, random_state=seed).reset_index(drop=True)
	return df_train, df_val, df_test


def validate_split_minimums(
	df_all: pd.DataFrame,
	df_val: pd.DataFrame,
	df_test: pd.DataFrame,
) -> None:
	labels = df_all["label"].value_counts()
	missing = []
	for label, total in labels.items():
		if total < 3:
			continue
		val_count = int((df_val["label"] == label).sum())
		test_count = int((df_test["label"] == label).sum())
		if val_count == 0 or test_count == 0:
			missing.append(label)

	if missing:
		print("Warning: labels missing val or test split with total >= 3:")
		print(", ".join(missing))


def freeze_model_layers(model: nn.Module, ratio: float) -> None:
	params = list(model.named_parameters())
	freeze_count = int(len(params) * ratio)
	keep_keywords = ("head", "classifier", "fc")
	for i, (name, param) in enumerate(params):
		# if any(key in name for key in keep_keywords):
		# 	param.requires_grad = True
		# 	continue
            
		param.requires_grad = i >= freeze_count


def _create_timm_model(model_name: str, num_classes: int, freeze_ratio: float) -> nn.Module:
	model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
	freeze_model_layers(model, freeze_ratio)
	model.model_name = model_name
	return model


def build_mobilenet_large(num_classes: int) -> nn.Module:
	return _create_timm_model("mobilenetv3_large_100", num_classes, freeze_ratio= 0.90)


def build_efficientnet_large(num_classes: int) -> nn.Module:
	return _create_timm_model("tf_efficientnet_b4", num_classes, freeze_ratio= 0.90)


def build_resnet_large(num_classes: int) -> nn.Module:
	return _create_timm_model("resnet50", num_classes, freeze_ratio= 0.90)


def build_convnext_large(num_classes: int) -> nn.Module:
	return _create_timm_model("convnext_tiny", num_classes, freeze_ratio=0.97)


def build_vit_large(num_classes: int) -> nn.Module:
	return _create_timm_model("vit_base_patch16_224", num_classes, freeze_ratio=0.97)


def build_deit_base(num_classes: int) -> nn.Module:
	return _create_timm_model("deit_base_patch16_224", num_classes, freeze_ratio= 0.90)


def build_swin_large(num_classes: int) -> nn.Module:
	return _create_timm_model("swin_large_patch4_window7_224", num_classes, freeze_ratio=0.97)


def build_beit_large(num_classes: int) -> nn.Module:
	return _create_timm_model("beit_large_patch16_224", num_classes, freeze_ratio=0.97)


def summarize_model(model: nn.Module) -> dict:
	total_params = sum(p.numel() for p in model.parameters())
	trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	frozen_params = total_params - trainable_params
	approx_size_mb = total_params * 4 / (1024**2)
	return {
		"total_params": total_params,
		"trainable_params": trainable_params,
		"frozen_params": frozen_params,
		"approx_size_mb": approx_size_mb,
	}


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


def find_last_conv_layer(model: nn.Module) -> nn.Module | None:
	last_conv = None
	for module in model.modules():
		if isinstance(module, nn.Conv2d):
			last_conv = module
	return last_conv


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
			projection = U[:, 0].reshape(h, w)
			if np.sum(projection) < 0:
				projection = -projection
			cam = np.maximum(projection, 0)
		else:
			self.model.zero_grad()
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
				projection = U[:, 0].reshape(h, w)
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


def overlay_cam_on_image(image: Image.Image, cam: np.ndarray, alpha: float = 0.4) -> Image.Image:
	cam_resized = Image.fromarray((cam * 255).astype(np.uint8)).resize(image.size)
	cam_resized = np.array(cam_resized) / 255.0
	color_map = plt.get_cmap("jet")
	heatmap = color_map(cam_resized)[:, :, :3]
	img = np.array(image).astype(np.float32) / 255.0
	overlay = img * (1 - alpha) + heatmap * alpha
	overlay = np.clip(overlay, 0, 1)
	return Image.fromarray((overlay * 255).astype(np.uint8))


def save_gradcam_samples(
	model: nn.Module,
	df: pd.DataFrame,
	eval_tf,
	device: torch.device,
	output_dir: Path,
	num_samples: int = 8,
) -> None:
	target_layer = find_last_conv_layer(model)
	if target_layer is None:
		print("GradCAM skipped: no Conv2d layer found in model")
		return

	model.eval()
	count = min(num_samples, len(df))
	if count == 0:
		return
	batch = df.sample(n=count, random_state=SEED).reset_index(drop=True)

	# Lặp qua tất cả các phương pháp CAM
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


def main() -> None:
	set_seed(SEED)
	device = get_device()
	print(f"Using device: {device}")

	root_dir = r"/kaggle/input/datasets/b23dckh002lvitanh/s3-data/S3"
	output_dir = Path("outputs")
	output_dir.mkdir(parents=True, exist_ok=True)

	samples = collect_image_samples(root_dir)
	if not samples:
		raise ValueError("No images found under root_dir")

	df = build_dataframe(samples)
	eda_genus_distribution(df, "Full dataset genus distribution", output_dir / "eda_all_genus.png")

	print("Computing embeddings for split...")
	embeddings = compute_embeddings(df, batch_size=BATCH_SIZE, device=device)
	df_train, df_test, df_val = mahalanobis_split_by_class(
		df,
		embeddings,
		train_ratio=0.7,
		val_ratio=0.15,
		seed=SEED,
	)
	validate_split_minimums(df, df_val, df_test)
	if device.type == "cuda":
		torch.cuda.empty_cache()
	log_split_summary(df, df_train, df_val, df_test)
	eda_split_class_distribution(
		df_train,
		df_val,
		df_test,
		"All classes distribution (train/val/test)",
		output_dir / "eda_all_classes.png",
	)
	plot_split_distributions(df_train, df_val, df_test, output_dir)

	class_names = sorted(df["label"].unique().tolist())
	class_to_idx = {name: i for i, name in enumerate(class_names)}

	model = build_efficientnet_large(num_classes=len(class_names))
	model_info = summarize_model(model)
	print(
		"Model info - "
		f"name={model.model_name}, "
		f"total_params={model_info['total_params']:,}, "
		f"trainable={model_info['trainable_params']:,}, "
		f"frozen={model_info['frozen_params']:,}, "
		f"approx_size={model_info['approx_size_mb']:.2f} MB"
	)
	
	model = model.to(device)
 
	cfg = resolve_data_config({}, model=model)
	img_size = cfg.get("input_size", (3, 224, 224))[-1]
	mean = cfg.get("mean", (0.485, 0.456, 0.406))
	std = cfg.get("std", (0.229, 0.224, 0.225))
	train_tf, eval_tf = build_transforms(img_size, mean, std)

	train_ds = ImageListDataset(df_train, class_to_idx, transform=train_tf)
	val_ds = ImageListDataset(df_val, class_to_idx, transform=eval_tf)
	test_ds = ImageListDataset(df_test, class_to_idx, transform=eval_tf)

	num_workers = min(5, os.cpu_count() or 1)
	train_loader = DataLoader(
		train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=True
	)
	val_loader = DataLoader(
		val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True
	)
	test_loader = DataLoader(
		test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True
	)

	criterion = FocalLoss(gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA)
	optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

	# Learning Rate Scheduler (Cosine Annealing)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

	history = train_model(
		model,
		train_loader,
		val_loader,
		optimizer,
		criterion,
		device,
		epochs=EPOCHS,
		patience=PATIENCE,
		output_dir=output_dir,
		scheduler=scheduler,
	)
	plot_training_curves(history, output_dir)

	raw_model = model
	best_path = output_dir / f"best_model_{raw_model.model_name}.pth"
	if best_path.exists():
		raw_model.load_state_dict(torch.load(best_path, map_location=device))

	evaluate_and_report(model, val_loader, device, class_names, output_dir, prefix="val")
	evaluate_and_report(model, test_loader, device, class_names, output_dir, prefix="test")

	gradcam_dir = output_dir / "gradcam"
	save_gradcam_samples(model, df_val, eval_tf, device, gradcam_dir, num_samples=8)


if __name__ == "__main__":
	main()
