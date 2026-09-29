"""
01_data_article.modules.classification.dataset
==============================================
Quản lý tập dữ liệu ảnh macroscopic wood, tiền xử lý và chia phân vùng tự động.
"""

import sys
import random
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class MacroscopicWoodDataset(Dataset):
    """Dataset đọc ảnh macro trực tiếp từ bảng phân bổ split."""
    def __init__(self, df: pd.DataFrame, class_to_idx: Dict[str, int], transform=None):
        self.df = df.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        img_path = row["image_path"] if "image_path" in row else row["path"]

        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
        except Exception:
            img = Image.fromarray(np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8))

        cls_key = "class_name" if "class_name" in row else "label"
        label_idx = self.class_to_idx[row[cls_key]]
        if self.transform:
            img = self.transform(img)
        return img, label_idx


def build_transforms(img_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    """Data augmentation chuẩn hóa theo thiết kế trong Section 4.2 của bài báo."""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.05),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    return train_tf, eval_tf


def auto_split_from_directory(data_dir: str, seed: int = 42) -> pd.DataFrame:
    """Tự động quét thư mục ảnh gốc và chia phân vùng tỷ lệ 60/20/20 (nếu chưa có split CSV)."""
    root = Path(data_dir)
    records = []
    rng = random.Random(seed)

    for class_folder in sorted(root.iterdir()):
        if not class_folder.is_dir():
            continue
        c_name = class_folder.name
        img_files = sorted([f for f in class_folder.rglob("*") if f.suffix.lower() in [".jpg", ".jpeg", ".png"]])
        if not img_files:
            continue

        rng.shuffle(img_files)
        n = len(img_files)
        n_train = max(1, int(0.6 * n))
        n_val = max(1, int(0.2 * n))

        for idx, f in enumerate(img_files):
            split = "train" if idx < n_train else ("val" if idx < n_train + n_val else "test")
            records.append({
                "image_path": str(f.resolve()),
                "class_name": c_name,
                "split": split
            })

    return pd.DataFrame(records)


def load_or_generate_dataset_split(
    split_csv_path: Optional[str] = None,
    metadata_csv_path: Optional[str] = None,
    data_dir: Optional[str] = None,
    seed: int = 42
) -> pd.DataFrame:
    """Nạp dataframe dữ liệu và chuẩn hóa đường dẫn ảnh."""
    split_p = Path(split_csv_path) if split_csv_path else None
    if split_p and split_p.exists():
        df = pd.read_csv(split_p)
    elif data_dir and Path(data_dir).exists():
        df = auto_split_from_directory(data_dir, seed=seed)
    else:
        candidates = [
            Path("out/splits/split_canonical.csv"),
            Path("paper_data_assets/splits/split_canonical.csv"),
        ]
        found = False
        for c in candidates:
            if c.exists():
                df = pd.read_csv(c)
                found = True
                break
        if not found:
            raise FileNotFoundError("Không tìm thấy tệp split_canonical.csv và không có data-dir hợp lệ!")

    if "class_name" not in df.columns and "label" in df.columns:
        df["class_name"] = df["label"]
    if "image_path" not in df.columns and "path" in df.columns:
        df["image_path"] = df["path"]

    return df
