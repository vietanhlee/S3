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
    splits = ["train", "val", "test"]
    for split in splits:
        split_dir = root / split
        if not split_dir.exists():
            continue
            
        class_dirs = sorted([p for p in split_dir.iterdir() if p.is_dir()])
        for class_dir in tqdm(class_dirs, desc=f"Scan {split} classes"):
            label = class_dir.name
            files = list(class_dir.rglob("*"))
            for path in files:
                if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                    samples.append({"path": str(path), "label": label, "split": split})
    return samples

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
        return img, row["label"], row["split"]

def build_embedding_model() -> torch.nn.Module:
    # EfficientNet-B4 model
    model = timm.create_model(
        "efficientnet_b4",
        pretrained=True,
        num_classes=0,
        global_pool="avg",
    )
    model.eval()
    return model

def compute_embedding_features(
    df: pd.DataFrame,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, list[str], list[str]]:
    model = build_embedding_model().to(device)
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
    all_splits = []
    with torch.no_grad():
        for images, labels, splits in tqdm(loader, desc="Embed"):
            images = images.to(device)
            feats = model(images).detach().cpu().numpy()
            all_features.append(feats)
            all_labels.extend(list(labels))
            all_splits.extend(list(splits))
    return np.concatenate(all_features, axis=0), all_labels, all_splits

def run_tsne(
    features: np.ndarray,
    seed: int,
    perplexity: float = 30.0,
) -> np.ndarray:
    n_samples = features.shape[0]
    
    # Not enough samples for a meaningful t-SNE, fallback to PCA or random
    if n_samples <= 3:
        if n_samples > 1:
            try:
                return PCA(n_components=2, random_state=seed).fit_transform(features)
            except Exception:
                return np.random.randn(n_samples, 2)
        else:
            return np.zeros((n_samples, 2))
            
    # Apply PCA first to reduce noise and speed up if dimensions are very large
    pca_dim = 50
    if features.shape[1] > pca_dim and n_samples > pca_dim:
        features = PCA(n_components=pca_dim, random_state=seed).fit_transform(features)
        
    # Perplexity must be less than n_samples. 
    max_perp = min(perplexity, float(n_samples - 1) / 3.0)
    
    tsne = TSNE(
        n_components=2,
        perplexity=max_perp,
        early_exaggeration=12.0,
        init="pca",
        learning_rate="auto",
        random_state=seed,
        max_iter=1000,
        n_jobs=-1,
    )
    coords = tsne.fit_transform(features)
    return coords

def plot_tsne_splits(coords: np.ndarray, splits: list[str], title: str, out_path: Path) -> None:
    # Colors for train, val, test
    split_colors = {
        "train": "#1f77b4", # blue
        "val": "#ff7f0e",   # orange
        "test": "#2ca02c"   # green
    }
    
    plt.figure(figsize=(10, 8))
    
    # Plot each split
    for split_type in ["train", "val", "test"]:
        indices = [i for i, s in enumerate(splits) if s == split_type]
        if not indices:
            continue
            
        split_coords = coords[indices]
        plt.scatter(split_coords[:, 0], split_coords[:, 1], 
                    c=split_colors[split_type], 
                    label=split_type.capitalize(),
                    s=30, alpha=0.8, edgecolors='none')
                    
    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def main() -> None:
    root_dir = r"G:\S3_final"
    output_dir = Path("outputs_tsne_efficientnet")
    batch_size = 32
    seed = 42
    
    set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate dirs for genus and species
    genus_out_dir = output_dir / "genus_plots"
    genus_out_dir.mkdir(exist_ok=True)
    
    species_out_dir = output_dir / "species_plots"
    species_out_dir.mkdir(exist_ok=True)

    print(f"Collecting image samples from {root_dir}...")
    samples = collect_image_samples(root_dir)
    if not samples:
        raise ValueError(f"No images found under {root_dir}")

    df = pd.DataFrame(samples)
    
    device = get_device()
    print(f"Using device: {device}")
    
    # Compute embeddings
    print("Computing EfficientNetB4 embeddings for all images...")
    emb_features, labels, splits = compute_embedding_features(df, batch_size, device)
    
    df["label"] = labels
    df["split"] = splits
    # Extract genus from label (first word)
    df["genus"] = df["label"].apply(lambda x: x.split()[0])
    
    # 1. Plot per genus
    unique_genera = sorted(df["genus"].unique())
    print(f"\nProcessing {len(unique_genera)} genera...")
    for genus in tqdm(unique_genera, desc="Genera t-SNE"):
        indices = df[df["genus"] == genus].index.tolist()
        if len(indices) == 0:
            continue
            
        sub_features = emb_features[indices]
        sub_splits = [splits[i] for i in indices]
        
        coords = run_tsne(sub_features, seed=seed)
        out_path = genus_out_dir / f"tsne_genus_{genus}.png"
        plot_tsne_splits(coords, sub_splits, f"t-SNE - Chi (Genus): {genus}", out_path)
        
    # 2. Plot per species
    unique_species = sorted(df["label"].unique())
    print(f"\nProcessing {len(unique_species)} species...")
    for species in tqdm(unique_species, desc="Species t-SNE"):
        indices = df[df["label"] == species].index.tolist()
        if len(indices) == 0:
            continue
            
        sub_features = emb_features[indices]
        sub_splits = [splits[i] for i in indices]
        
        coords = run_tsne(sub_features, seed=seed)
        
        # Replace spaces with underscores for filename safety
        safe_species = species.replace(" ", "_")
        out_path = species_out_dir / f"tsne_species_{safe_species}.png"
        plot_tsne_splits(coords, sub_splits, f"t-SNE - Loài (Species): {species}", out_path)

    print(f"\nDone! All plots are saved in {output_dir.absolute()}")

if __name__ == "__main__":
    main()
