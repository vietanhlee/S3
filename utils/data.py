import os
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


def build_embedding_model() -> torch.nn.Module:
	model = timm.create_model("swin_large_patch4_window7_224", pretrained=True, num_classes=0)
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


def compute_embeddings(df: pd.DataFrame, batch_size: int, device: torch.device) -> np.ndarray:
	model = build_embedding_model().to(device)
	transform = build_embedding_transform(model)

	fs = ImagePathDataset(df, transform=transform)
	num_workers = min(4, os.cpu_count() or 1)
	loader = DataLoader(fs, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

	features = []
	with torch.no_grad():
		for images in tqdm(loader, desc="Embed"):
			images = images.to(device)
			feats = model(images)
			features.append(feats.detach().cpu().numpy())

	del model
	if device.type == "cuda":
		torch.cuda.empty_cache()

	if not features:
		return np.empty((0, 0), dtype=np.float32)
	return np.concatenate(features, axis=0)


def compute_embeddings_v2(
	df: pd.DataFrame,
	model_name: str,
	batch_size: int,
	device: torch.device,
) -> np.ndarray:
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
	loader = DataLoader(fs, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

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
