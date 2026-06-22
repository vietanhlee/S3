import hashlib
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


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

ROOT_DIR = r"/kaggle/input/datasets/b23dckh002lvitanh/s3-data/S3"
OUTPUT_DIR = "data"
MAPPING_CSV = "split_mapping.csv"
BATCH_SIZE = 128
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
SEED = 42
EPS = 1e-6
NUM_WORKERS = 0


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
				rel_path = path.relative_to(class_dir).as_posix()
				root_rel_path = path.relative_to(root).as_posix()
				samples.append(
					{
						"path": str(path),
						"label": label,
						"rel_path": rel_path,
						"root_rel_path": root_rel_path,
					}
				)
	return samples


def build_dataframe(samples: list[dict]) -> pd.DataFrame:
	return pd.DataFrame(samples)


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


def build_efficientnetb4_embedding_model() -> torch.nn.Module:
	model = timm.create_model(
		"tf_efficientnet_b4",
		pretrained=True,
		num_classes=0,
		global_pool="avg",
	)
	model.eval()
	return model


def build_embedding_transform(model: torch.nn.Module) -> transforms.Compose:
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
	model = build_efficientnetb4_embedding_model().to(device)
	model.eval()
	transform = build_embedding_transform(model)

	ds = ImagePathDataset(df, transform=transform)
	loader = DataLoader(
		ds,
		batch_size=batch_size,
		shuffle=False,
		num_workers=NUM_WORKERS,
		pin_memory=True,
	)

	features = []
	with torch.no_grad():
		progress = tqdm(total=len(ds), desc="Embedding for split", unit="img")
		for images in loader:
			images = images.to(device)
			feats = model(images)
			features.append(feats.detach().cpu().numpy())
			progress.update(images.size(0))
		progress.close()

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
	train_ratio: float,
	val_ratio: float,
	seed: int,
	eps: float,
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


def select_target_relpath(
	dst_rel_dir: Path,
	src_path: Path,
	rel_path: str,
	used_names: set[str],
) -> Path:
	target_name = src_path.name
	if target_name not in used_names:
		used_names.add(target_name)
		return dst_rel_dir / target_name

	digest = hashlib.md5(rel_path.encode("utf-8")).hexdigest()[:8]
	candidate_name = f"{src_path.stem}__{digest}{src_path.suffix}"
	if candidate_name not in used_names:
		used_names.add(candidate_name)
		return dst_rel_dir / candidate_name

	for i in range(1, 10000):
		candidate_name = f"{src_path.stem}__{digest}_{i}{src_path.suffix}"
		if candidate_name not in used_names:
			used_names.add(candidate_name)
			return dst_rel_dir / candidate_name

	raise RuntimeError(f"Too many name collisions in {dst_rel_dir}")


def plan_split_targets(
	df_split: pd.DataFrame,
	split_name: str,
	output_dir: Path,
) -> list[dict]:
	planned = []
	used_by_dir: dict[Path, set[str]] = {}

	progress = tqdm(total=len(df_split), desc=f"Plan {split_name}", unit="img")
	for _, row in df_split.iterrows():
		label = row["label"]
		src = Path(row["path"])
		rel_path = row.get("rel_path", src.name)
		dst_rel_dir = Path(split_name) / label
		used_names = used_by_dir.get(dst_rel_dir)
		if used_names is None:
			used_names = set()
			dst_abs_dir = output_dir / dst_rel_dir
			if dst_abs_dir.exists():
				for existing in dst_abs_dir.iterdir():
					if existing.is_file():
						used_names.add(existing.name)
			used_by_dir[dst_rel_dir] = used_names

		dst_rel_path = select_target_relpath(dst_rel_dir, src, rel_path, used_names)
		original_rel = row.get("root_rel_path")
		if original_rel is None:
			original_rel = src.name

		planned.append(
			{
				"original_path": str(original_rel),
				"final_path": dst_rel_path.as_posix(),
			}
		)
		progress.update(1)
	progress.close()

	return planned


def main() -> None:
	set_seed(SEED)
	device = get_device()
	print(f"Using device: {device}")

	samples = collect_image_samples(ROOT_DIR)
	if not samples:
		raise ValueError("No images found under root_dir")

	df = build_dataframe(samples)

	print("Computing embeddings for split...")
	embeddings = compute_embeddings(df, batch_size=BATCH_SIZE, device=device)
	if device.type == "cuda":
		torch.cuda.empty_cache()

	df_train, df_val, df_test = mahalanobis_split_by_class(
		df,
		embeddings,
		train_ratio=TRAIN_RATIO,
		val_ratio=VAL_RATIO,
		seed=SEED,
		eps=EPS,
	)
	validate_split_minimums(df, df_val, df_test)

	output_dir = Path(OUTPUT_DIR)
	output_dir.mkdir(parents=True, exist_ok=True)

	planned_train = plan_split_targets(df_train, "train", output_dir)
	planned_val = plan_split_targets(df_val, "val", output_dir)
	planned_test = plan_split_targets(df_test, "test", output_dir)

	mapping_df = pd.DataFrame(planned_train + planned_val + planned_test)
	csv_path = output_dir / MAPPING_CSV
	mapping_df.to_csv(csv_path, index=False)

	print(f"Saved CSV mapping to: {csv_path}")


if __name__ == "__main__":
	main()
