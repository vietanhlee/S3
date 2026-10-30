"""
dataset.py
==========
Production-grade PyTorch Dataset and DataLoader loaders for the 11-species timber benchmark.
Strictly loads physical wood micrographs from disk with automatic image root resolution
(supporting Kaggle input directories, server mounts, and local workspaces).
Raises descriptive errors if real images are not mounted, strictly eliminating synthetic fallbacks.
"""

import os
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_train_transforms(image_size: int = 224) -> Callable:
    """Builds standardized macroscopic timber training augmentations."""
    return transforms.Compose([
        transforms.Resize((int(image_size * 1.14), int(image_size * 1.14))),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_eval_transforms(image_size: int = 224) -> Callable:
    """Builds deterministic evaluation transforms."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def resolve_image_root(df: pd.DataFrame, custom_root: Optional[Union[str, Path]] = None) -> Path:
    """
    Tự động nhận diện và xác thực thư mục ảnh gỗ thực tế trên đĩa.
    Ưu tiên quét:
      1. custom_root được truyền vào.
      2. Môi trường Kaggle /kaggle/input/** (nơi dataset s3-origin được mount).
      3. Thư mục cục bộ / server relative paths.
    Nếu không tìm thấy ảnh thật, raise FileNotFoundError có hướng dẫn rõ ràng.
    """
    sample_rel = ""
    if not df.empty and "image_path" in df.columns:
        sample_rel = str(df["image_path"].iloc[0]).replace("\\", "/")

    candidates: List[Path] = []
    if custom_root:
        candidates.append(Path(custom_root))

    # 1. Quét môi trường Kaggle
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        candidates.extend([
            Path("/kaggle/input/datasets/b23dckh002lvitanh/s3-origin/S3"),
            Path("/kaggle/input/datasets/b23dckh002lvitanh/s3-origin"),
            Path("/kaggle/input/s3-origin/S3"),
            Path("/kaggle/input/s3-origin"),
            Path("/kaggle/input/s3/S3"),
            Path("/kaggle/input/s3"),
        ])
        try:
            for sub in kaggle_input.iterdir():
                if sub.is_dir():
                    candidates.append(sub)
                    candidates.append(sub / "S3")
                    candidates.append(sub / "images")
                    candidates.append(sub / "out")
                    candidates.append(sub / "out" / "images")
                    for sub2 in sub.iterdir():
                        if sub2.is_dir():
                            candidates.append(sub2)
                            candidates.append(sub2 / "S3")
                            candidates.append(sub2 / "images")
        except Exception:
            pass

    # 2. Quét môi trường Local và Repo
    current_file_dir = Path(__file__).resolve().parent
    repo_root = current_file_dir.parent

    candidates.extend([
        repo_root / "01_data_paper_forensic_cites" / "out" / "images",
        repo_root / "01_data_paper_forensic_cites" / "out",
        repo_root / "out" / "images",
        repo_root / "out",
        Path("../01_data_paper_forensic_cites/out/images"),
        Path("../01_data_paper_forensic_cites/out"),
        Path("01_data_paper_forensic_cites/out/images"),
        Path("01_data_paper_forensic_cites/out"),
        Path("images"),
        current_file_dir / "images",
    ])

    # Kiểm tra candidate nào chứa sample_rel
    for cand in candidates:
        if cand and cand.exists():
            if sample_rel:
                if (cand / sample_rel).is_file():
                    print(f"[+] [DỮ LIỆU THẬT] Đã nhận diện thư mục ảnh gốc: {cand.resolve()}")
                    print(f"    - Ảnh mẫu xác thực: {cand / sample_rel}")
                    return cand.resolve()
                if (cand / "images" / sample_rel).is_file():
                    print(f"[+] [DỮ LIỆU THẬT] Đã nhận diện thư mục ảnh gốc: {(cand / 'images').resolve()}")
                    return (cand / "images").resolve()

    # Kiểm tra theo thư mục loài đầu tiên
    first_taxon = sample_rel.split("/")[0] if "/" in sample_rel else ""
    if first_taxon:
        for cand in candidates:
            if cand and cand.exists():
                taxon_dir = cand / first_taxon
                if taxon_dir.is_dir():
                    any_img = next(taxon_dir.rglob("*.png"), None) or next(taxon_dir.rglob("*.jpg"), None)
                    if any_img:
                        print(f"[+] [DỮ LIỆU THẬT] Đã nhận diện thư mục ảnh gốc: {cand.resolve()}")
                        print(f"    - Ảnh taxon mẫu xác thực: {any_img}")
                        return cand.resolve()

    # Không tìm thấy bất kỳ ảnh thật nào -> Báo lỗi chi tiết, không chạy giả lập!
    scanned_list = "\n".join(f"    - {str(c)}" for c in candidates[:8] if str(c))
    error_msg = f"""
{'=' * 80}
[LỖI DỮ LIỆU THỰC TẾ]: KHÔNG TÌM THẤY TẬP DỮ LIỆU ẢNH GỖ TRÊN ĐĨA!
{'=' * 80}
Hệ thống KHÔNG CHẠY TRÊN DỮ LIỆU GIẢ LẬP (SYNTHETIC) NỮA.
Để huấn luyện mô hình thật, hệ thống cần 3,702 ảnh hiển vi / macro gốc (.png/.jpg).

Ảnh mẫu đang tìm kiếm:
    '{sample_rel}'

Các đường dẫn đã quét tự động:
{scanned_list}

HƯỚNG DẪN KHẮC PHỤC TRÊN KAGGLE:
  1. Trong giao diện Kaggle Notebook, nhìn sang thanh công cụ bên phải, mục 'Input'.
  2. Bấm vào nút '+ Add Input' (hoặc '+ Add Data').
  3. Tìm kiếm dataset: 'b23dckh002lvitanh/s3-origin' (hoặc từ khóa 's3-origin').
  4. Bấm nút (+) để thêm dataset vào Notebook (dataset sẽ xuất hiện tại /kaggle/input/s3-origin).
  5. Sau khi nạp xong, chạy lại lệnh:
     bash run_all.sh
{'=' * 80}
"""
    print(error_msg)
    raise FileNotFoundError(f"Missing real wood images. Sample '{sample_rel}' not found on disk.")


class Timber11Dataset(Dataset):
    """
    Dataset representing the clean 11-species wood forensics benchmark.
    Yields strictly real physical wood micrographs from disk.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        image_root: Union[str, Path],
        transform: Optional[Callable] = None,
        is_train: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.image_root = Path(image_root)
        self.transform = transform or (build_train_transforms() if is_train else build_eval_transforms())
        self.is_train = is_train

        # Remap specimen IDs in this specific subset to contiguous indices [0 .. N_specimens - 1]
        unique_specimens = sorted(self.df["specimen_id"].unique())
        self.specimen_to_idx = {sid: i for i, sid in enumerate(unique_specimens)}
        self.num_specimens = len(unique_specimens)

    def __len__(self) -> int:
        return len(self.df)

    def _get_image_path(self, row: pd.Series) -> Path:
        rel_path = str(row.get("image_path", "")).replace("\\", "/").strip()
        if rel_path:
            cand = self.image_root / rel_path
            if cand.is_file():
                return cand
            cand_sub = self.image_root / "images" / rel_path
            if cand_sub.is_file():
                return cand_sub

        img_id = str(row.get("image_id", ""))
        for ext in [".png", ".jpg", ".jpeg"]:
            cand_id = self.image_root / f"{img_id}{ext}"
            if cand_id.is_file():
                return cand_id

        return self.image_root / rel_path

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        img_path = self._get_image_path(row)

        if not img_path.is_file():
            raise FileNotFoundError(
                f"[Dữ liệu thiếu] Không tìm thấy file ảnh thực tế: {img_path} "
                f"(image_id: {row.get('image_id')}, specimen: {row.get('specimen_id')})"
            )

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Lỗi đọc file ảnh {img_path}: {e}")

        tensor_img = self.transform(img)

        species_idx = int(row["species_idx"])
        specimen_idx = int(self.specimen_to_idx.get(row["specimen_id"], 0))
        global_specimen_idx = int(row.get("global_specimen_idx", specimen_idx))

        return {
            "image": tensor_img,
            "species_idx": torch.tensor(species_idx, dtype=torch.long),
            "specimen_idx": torch.tensor(specimen_idx, dtype=torch.long),
            "global_specimen_idx": torch.tensor(global_specimen_idx, dtype=torch.long),
            "image_id": str(row["image_id"]),
            "specimen_id": str(row["specimen_id"]),
        }


def create_dataloaders(
    metadata_csv: Union[str, Path],
    image_root: Optional[Union[str, Path]] = None,
    batch_size: int = 64,
    num_workers: int = 2 if os.name != "nt" else 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Creates DataLoaders for (train, test_seen, test_unseen) strictly reading real images.
    Auto-resolves image_root across Kaggle and local environments.
    """
    df = pd.read_csv(metadata_csv)
    resolved_root = resolve_image_root(df, custom_root=image_root)

    df_train = df[df["eval_split"] == "train"].copy().reset_index(drop=True)
    df_seen = df[df["eval_split"] == "test_seen"].copy().reset_index(drop=True)
    df_unseen = df[df["eval_split"] == "test_unseen"].copy().reset_index(drop=True)

    ds_train = Timber11Dataset(df_train, image_root=resolved_root, is_train=True)
    ds_seen = Timber11Dataset(df_seen, image_root=resolved_root, is_train=False)
    ds_unseen = Timber11Dataset(df_unseen, image_root=resolved_root, is_train=False)

    loader_train = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True if len(ds_train) > batch_size else False,
    )

    loader_seen = DataLoader(
        ds_seen,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    loader_unseen = DataLoader(
        ds_unseen,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    info = {
        "num_classes": df["species_idx"].nunique(),
        "train_samples": len(ds_train),
        "seen_samples": len(ds_seen),
        "unseen_samples": len(ds_unseen),
        "train_specimens": ds_train.num_specimens,
        "species_names": sorted(df["taxon_name"].unique()),
        "image_root": str(resolved_root),
    }

    return loader_train, loader_seen, loader_unseen, info
