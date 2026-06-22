import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import timm
from timm.data import resolve_data_config

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import colorsys


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def set_seed(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
	return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
	return pd.DataFrame(samples)


def split_genus_species(label: str) -> tuple[str, str]:
	parts = label.split()
	genus = parts[0] if len(parts) > 0 else "Unknown"
	species = " ".join(parts[1:]) if len(parts) > 1 else "Unknown"
	return genus, species


def sample_per_class(df: pd.DataFrame, per_class: int, seed: int) -> pd.DataFrame:
	if per_class <= 0:
		return df
	return (
		df.groupby("label", group_keys=False)
		.apply(lambda g: g.sample(n=min(len(g), per_class), random_state=seed))
		.reset_index(drop=True)
	)


def sample_max_total(df: pd.DataFrame, max_total: int, seed: int) -> pd.DataFrame:
	if max_total <= 0 or len(df) <= max_total:
		return df
	return df.sample(n=max_total, random_state=seed).reset_index(drop=True)


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
		return img, row["label"]


def build_mobilenet_embedding_model() -> torch.nn.Module:
	model = timm.create_model(
		"mobilenetv3_large_100",
		pretrained=True,
		num_classes=0,
		global_pool="avg",
	)
	model.eval()
	return model


def compute_raw_features(
	df: pd.DataFrame,
	image_size: int,
) -> tuple[np.ndarray, list[str]]:
	features = []
	labels = []
	for _, row in tqdm(df.iterrows(), total=len(df), desc="Load raw"):
		with Image.open(row["path"]) as img:
			img = img.convert("RGB").resize((image_size, image_size))
			arr = np.asarray(img, dtype=np.float32) / 255.0
		features.append(arr.reshape(-1))
		labels.append(row["label"])
	return np.stack(features, axis=0), labels


def compute_embedding_features(
	df: pd.DataFrame,
	batch_size: int,
	device: torch.device,
	) -> tuple[np.ndarray, list[str]]:
	model = build_mobilenet_embedding_model().to(device)
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

	ds = ImagePathDataset(df, transform=transform)
	loader = DataLoader(
		ds,
		batch_size=batch_size,
		shuffle=False,
		num_workers=0,
		pin_memory=True,
	)

	all_features = []
	all_labels = []
	with torch.no_grad():
		for images, labels in tqdm(loader, desc="Embed"):
			images = images.to(device)
			feats = model(images).detach().cpu().numpy()
			all_features.append(feats)
			all_labels.extend(list(labels))
	return np.concatenate(all_features, axis=0), all_labels


def run_tsne(
	features: np.ndarray,
	labels: list[str],
	seed: int,
	perplexity: float,
	pca_dim: int,
	early_exaggeration: float,
	max_iter: int,
	n_iter_without_progress: int,
	min_grad_norm: float,
	method: str,
	angle: float,
	metric: str,
	n_jobs: int,
	verbose: int,
	) -> tuple[np.ndarray, list[str]]:
	if features.shape[0] < 3:
		raise ValueError("Need at least 3 samples for t-SNE.")

	if pca_dim > 0 and features.shape[1] > pca_dim:
		features = PCA(n_components=pca_dim, random_state=seed).fit_transform(features)

	max_perp = max(2.0, min(perplexity, features.shape[0] - 1))
	tsne = TSNE(
		n_components=2,
		perplexity=max_perp,
		early_exaggeration=early_exaggeration,
		init="pca",
		learning_rate="auto",
		random_state=seed,
		max_iter=max_iter,
		n_iter_without_progress=n_iter_without_progress,
		min_grad_norm=min_grad_norm,
		method=method,
		angle=angle,
		metric=metric,
		n_jobs=n_jobs,
		verbose=verbose,
	)
	coords = tsne.fit_transform(features)
	return coords, labels


def plot_tsne(coords: np.ndarray, labels: list[str], title: str, out_path: Path) -> None:
	unique_labels = sorted(set(labels))
	genus_to_labels: dict[str, list[str]] = {}
	for label in unique_labels:
		genus, _ = split_genus_species(label)
		genus_to_labels.setdefault(genus, []).append(label)

	genus_names = sorted(genus_to_labels.keys())
	base_cmap = plt.colormaps["tab20"]
	label_to_color: dict[str, tuple[float, float, float]] = {}
	for i, genus in enumerate(genus_names):
		base_rgb = base_cmap(i % base_cmap.N)[:3]
		h, l, s = colorsys.rgb_to_hls(*base_rgb)
		labels_in_genus = sorted(genus_to_labels[genus])
		n_species = len(labels_in_genus)
		lightness_values = np.linspace(0.35, 0.75, max(1, n_species))
		for j, label in enumerate(labels_in_genus):
			lightness = float(lightness_values[j]) if n_species > 1 else l
			rgb = colorsys.hls_to_rgb(h, lightness, s)
			label_to_color[label] = rgb

	point_colors = [label_to_color[l] for l in labels]

	plt.figure(figsize=(10, 8))
	plt.scatter(coords[:, 0], coords[:, 1], c=point_colors, s=12, alpha=0.8)
	plt.title(title)
	plt.xlabel("t-SNE 1")
	plt.ylabel("t-SNE 2")
	plt.tight_layout()
	plt.savefig(out_path, dpi=200)
	plt.close()

	legend_path = out_path.with_suffix(".labels.txt")
	with open(legend_path, "w", encoding="utf-8") as f:
		for label in unique_labels:
			genus, species = split_genus_species(label)
			hex_color = mcolors.to_hex(label_to_color[label])
			f.write(f"{genus}\t{species}\t{label}\t{hex_color}\n")

	legend_fig = plt.figure(figsize=(8, max(4, len(unique_labels) * 0.2)))
	legend_handles = []
	legend_labels = []
	for label in unique_labels:
		color = label_to_color[label]
		legend_handles.append(mpatches.Patch(color=color, label=label))
		legend_labels.append(label)
	legend_fig.legend(
		handles=legend_handles,
		labels=legend_labels,
		loc="center left",
		frameon=False,
	)
	legend_fig.tight_layout()
	legend_fig.savefig(out_path.with_suffix(".legend.png"), dpi=200)
	plt.close(legend_fig)


def main() -> None:
	root_dir = r"/kaggle/input/datasets/huecute/s3-data/S3"
	output_dir = Path("outputs")
	mode = "both"  # raw | embedding | both
	image_size = 64
	batch_size = 128
	per_class = 0
	max_total = 0
	perplexity = 15
	pca_dim = 40
	early_exaggeration = 12.0
	max_iter = 2000
	n_iter_without_progress = 150
	min_grad_norm = 1e-7
	method = "barnes_hut"
	angle = 0.5
	metric = "euclidean"
	n_jobs = -1
	verbose_tsne = 1
	seed = 42

	set_seed(seed)
	output_dir.mkdir(parents=True, exist_ok=True)

	samples = collect_image_samples(root_dir)
	if not samples:
		raise ValueError("No images found under root_dir")

	df = build_dataframe(samples)
	df = sample_per_class(df, per_class, seed)
	df = sample_max_total(df, max_total, seed)

	if mode in ("raw", "both"):
		raw_features, labels = compute_raw_features(df, image_size)
		coords, labels = run_tsne(
			raw_features,
			labels,
			seed=seed,
			perplexity=perplexity,
			pca_dim=pca_dim,
			early_exaggeration=early_exaggeration,
			max_iter=max_iter,
			n_iter_without_progress=n_iter_without_progress,
			min_grad_norm=min_grad_norm,
			method=method,
			angle=angle,
			metric=metric,
			n_jobs=n_jobs,
			verbose=verbose_tsne,
		)
		plot_tsne(coords, labels, "t-SNE (raw pixels)", output_dir / "tsne_raw.png")

	if mode in ("embedding", "both"):
		device = get_device()
		emb_features, labels = compute_embedding_features(df, batch_size, device)
		coords, labels = run_tsne(
			emb_features,
			labels,
			seed=seed,
			perplexity=perplexity,
			pca_dim=pca_dim,
			early_exaggeration=early_exaggeration,
			max_iter=max_iter,
			n_iter_without_progress=n_iter_without_progress,
			min_grad_norm=min_grad_norm,
			method=method,
			angle=angle,
			metric=metric,
			n_jobs=n_jobs,
			verbose=verbose_tsne,
		)
		plot_tsne(coords, labels, "t-SNE (MobileNet embeddings)", output_dir / "tsne_mobilenet.png")


if __name__ == "__main__":
	main()
