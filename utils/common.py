import random
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


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
	plt.close()


def eda_genus_distribution(df: pd.DataFrame, title: str, save_path: Path | None) -> None:
	counts = df["genus"].value_counts().sort_index()
	print(f"\n{title} - genus counts:\n{counts.to_string()}")

	genus_species = (
		df.groupby(["genus", "label"]).size().unstack(fill_value=0).sort_index()
	)

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
					fontsize=10,
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
	plt.close()


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
	for i, (name, param) in enumerate(params):
		param.requires_grad = i >= freeze_count


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
