"""
dataset.py
==========
Production-grade PyTorch Dataset and DataLoader module for the 11-species timber benchmark.
Directly scans and filters physical wood images from the disk directory structure:
/kaggle/input/datasets/b23dckh002lvitanh/s3-origin/S3
without requiring external metadata.csv files.

Implements per-seed dynamic partitioning with 3 evaluation sets and 3 unseen vouchers/species:
  1. Set random seed s.
  2. For each taxon (10 vouchers), partition 3 vouchers to strictly Out-of-Specimen Unseen Physical Test (33 vouchers total).
  3. Partition remaining 7 vouchers (77 vouchers total) into:
     - 70% Train Set
     - 15% In-Specimen Validation Set (Set 1)
     - 15% In-Specimen Seen Test Set (Set 2)
  4. Delivers 3 comprehensive evaluation sets: Val, Seen Test, and Unseen Test (Set 3).
"""

import os
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Đảm bảo console UTF-8 an toàn trên mọi hệ điều hành
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# 11 loài họ Đậu Fabaceae sạch mục tiêu (100% cân bằng 10 phôi/loài, 110 phôi vật lý)
TARGET_SPECIES = [
    "Afzelia bella",
    "Dalbergia melanoxylon",
    "Dalbergia oliveri",
    "Dalbergia rimosa",
    "Dalbergia tonkinensis",
    "Guibourtia arnoldiana",
    "Guibourtia ehie",
    "Peltogyne pubescens",
    "Pterocarpus erinaceus",
    "Pterocarpus indicus",
    "Sindora tonkinensis",
]

# 8 loài bị loại bỏ hoàn toàn do dữ liệu không ổn định hoặc quá ít phôi vật lý
EXCLUDED_SPECIES = [
    "Sindora cochinchinensis",
    "Pterocarpus soyauxii",
    "Afzelia pachyloba",
    "Afzelia quanzensis",
    "Afzelia africana",
    "Guibourtia coleosperma",
    "Pterocarpus macrocarpus",
    "Dalbergia cochinchinensis",
]

# Đường dẫn fix cứng theo yêu cầu người dùng
DEFAULT_IMAGE_ROOT = "/kaggle/input/datasets/b23dckh002lvitanh/s3-origin/S3"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def resolve_image_root(custom_root: Optional[Union[str, Path]] = None) -> Path:
    """
    Xác định thư mục ảnh gốc.
    Ưu tiên cao nhất: đường dẫn fix cứng theo chỉ định /kaggle/input/datasets/b23dckh002lvitanh/s3-origin/S3
    Nếu không có (ví dụ chạy local hoặc mount khác), quét các candidate thông dụng.
    """
    candidates = []
    if custom_root:
        candidates.append(Path(custom_root))

    # 1. Đường dẫn fix cứng do người dùng chỉ định
    candidates.append(Path(DEFAULT_IMAGE_ROOT))

    # 2. Các mount path thay thế trên Kaggle
    candidates.extend([
        Path("/kaggle/input/datasets/b23dckh002lvitanh/s3-origin"),
        Path("/kaggle/input/s3-origin/S3"),
        Path("/kaggle/input/s3-origin"),
        Path("/kaggle/input/s3/S3"),
        Path("/kaggle/input/s3"),
    ])

    # 3. Quét đệ quy trong /kaggle/input nếu tồn tại
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        try:
            for sub in kaggle_input.iterdir():
                if sub.is_dir():
                    candidates.append(sub / "S3")
                    candidates.append(sub)
        except Exception:
            pass

    # 4. Local workspace paths (cho mục đích phát triển / kiểm thử)
    current_dir = Path(__file__).resolve().parent
    repo_root = current_dir.parent
    candidates.extend([
        repo_root / "01_data_paper_forensic_cites" / "out" / "images",
        repo_root / "01_data_paper_forensic_cites" / "out",
        repo_root / "out" / "images",
        current_dir / "images",
        Path("images"),
    ])

    # Kiểm tra candidate nào chứa ít nhất 1 thư mục loài mục tiêu (ví dụ 'Afzelia bella')
    for cand in candidates:
        if cand and cand.exists():
            for sp in TARGET_SPECIES:
                sp_dir = cand / sp
                if sp_dir.is_dir():
                    print(f"[+] [DỮ LIỆU THẬT] Nhận diện thành công thư mục ảnh gốc: {cand.resolve()}")
                    return cand.resolve()

    # Nếu không tìm thấy, thông báo lỗi dừng chương trình
    candidates_str = "\n".join(f"    - {str(c)}" for c in candidates[:6])
    error_msg = f"""
{'=' * 80}
[LỖI DỮ LIỆU]: KHÔNG TÌM THẤY THƯ MỤC ẢNH GỖ TẠI CÁC ĐƯỜNG DẪN KAGGLE!
{'=' * 80}
Đường dẫn fix cứng yêu cầu:
    {DEFAULT_IMAGE_ROOT}

Các đường dẫn đã quét tự động:
{candidates_str}

HƯỚNG DẪN XỬ LÝ TRÊN KAGGLE:
  1. Trong giao diện Kaggle Notebook, nhìn sang thanh bên phải (Input).
  2. Bấm '+ Add Input' (hoặc '+ Add Data').
  3. Tìm kiếm và thêm dataset: 'b23dckh002lvitanh/s3-origin'.
  4. Đảm bảo dataset xuất hiện tại /kaggle/input/datasets/b23dckh002lvitanh/s3-origin/S3
{'=' * 80}
"""
    print(error_msg)
    raise FileNotFoundError(f"Cannot find wood image dataset directory at {DEFAULT_IMAGE_ROOT}")


def scan_and_build_dataset(image_root: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Quét trực tiếp cây thư mục đĩa và xây dựng DataFrame danh mục ảnh:
    Cấu trúc: <image_root>/<Species>/<Voucher>/<image_file>
    Chỉ giữ lại đúng 11 loài mục tiêu, loại bỏ 8 loài không ổn định.
    Không cần phụ thuộc metadata.csv bên ngoài!
    """
    root_path = resolve_image_root(image_root)
    records = []
    img_exts = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"}

    species_mapping = {sp: idx for idx, sp in enumerate(TARGET_SPECIES)}

    for sp_name in TARGET_SPECIES:
        sp_dir = root_path / sp_name
        if not sp_dir.is_dir():
            # Thử tìm case-insensitive
            matched = [d for d in root_path.iterdir() if d.is_dir() and d.name.lower() == sp_name.lower()]
            if matched:
                sp_dir = matched[0]
            else:
                print(f"[!] Cảnh báo: Không tìm thấy thư mục loài '{sp_name}' tại {root_path}")
                continue

        # Quét các subfolder phôi gỗ (vouchers)
        voucher_dirs = [d for d in sp_dir.iterdir() if d.is_dir()]
        
        # Nếu không có subfolder phôi, quét trực tiếp file trong folder loài
        if not voucher_dirs:
            direct_imgs = [f for f in sp_dir.iterdir() if f.is_file() and f.suffix in img_exts]
            for img_f in direct_imgs:
                records.append({
                    "image_path": str(img_f.resolve()),
                    "image_id": img_f.stem,
                    "taxon_name": sp_name,
                    "species_idx": species_mapping[sp_name],
                    "specimen_id": f"{sp_name}_voucher_01",
                })
        else:
            for v_dir in voucher_dirs:
                v_name = f"{sp_name}_{v_dir.name}"
                v_imgs = [f for f in v_dir.iterdir() if f.is_file() and f.suffix in img_exts]
                # Nếu ảnh nằm trong folder con sâu hơn
                if not v_imgs:
                    v_imgs = [f for f in v_dir.rglob("*") if f.is_file() and f.suffix in img_exts]
                for img_f in v_imgs:
                    records.append({
                        "image_path": str(img_f.resolve()),
                        "image_id": img_f.stem,
                        "taxon_name": sp_name,
                        "species_idx": species_mapping[sp_name],
                        "specimen_id": v_name,
                    })

    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError(f"Không tìm thấy bất kỳ file ảnh nào hợp lệ trong 11 loài tại {root_path}")

    # Gán chỉ số index liên tục cho toàn bộ các phôi vật lý
    unique_vouchers = sorted(df["specimen_id"].unique())
    voucher_to_idx = {v: idx for idx, v in enumerate(unique_vouchers)}
    df["global_specimen_idx"] = df["specimen_id"].map(voucher_to_idx)

    print(f"[+] Quét hoàn tất từ đĩa: {len(df):,} ảnh | {df['taxon_name'].nunique()} loài | {len(unique_vouchers)} phôi gỗ vật lý")
    return df


def split_dataset_by_seed(
    df: pd.DataFrame,
    seed: int = 42,
    unseen_vouchers_per_species: int = 3,
    val_ratio: float = 0.15,
    seen_test_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Thực hiện phân vùng dữ liệu theo Seed cụ thể với 3 bộ đánh giá (Val, Seen Test, Unseen Test):
      1. Đặt seed rng = RandomState(seed).
      2. Với mỗi loài (10 phôi): chọn ngẫu nhiên unseen_vouchers_per_species (3 phôi)
         làm tập Unseen Physical Test (33 phôi hoàn toàn mới ngoài hiện trường, 3 phôi/loài).
      3. 7 phôi còn lại đưa vào tập In-Specimen pool (77 phôi).
      4. Trên 77 phôi này, chia ngẫu nhiên ảnh thành:
         - Train Set (70% ảnh trên 77 phôi)
         - In-Specimen Validation Set (15% ảnh trên 77 phôi)
         - In-Specimen Seen Test Set (15% ảnh trên 77 phôi)
    """
    rng = np.random.RandomState(seed)

    unseen_voucher_ids = set()
    in_specimen_voucher_ids = set()

    for sp_name, group in df.groupby("taxon_name"):
        vouchers = sorted(group["specimen_id"].unique())
        shuffled = rng.permutation(vouchers).tolist()
        
        # Chọn ra 3 phôi unseen mỗi loài (tổng 33 phôi)
        unseen_picks = shuffled[:unseen_vouchers_per_species]
        in_picks = shuffled[unseen_vouchers_per_species:]
        
        unseen_voucher_ids.update(unseen_picks)
        in_specimen_voucher_ids.update(in_picks)

    df_unseen = df[df["specimen_id"].isin(unseen_voucher_ids)].copy().reset_index(drop=True)
    df_in_pool = df[df["specimen_id"].isin(in_specimen_voucher_ids)].copy().reset_index(drop=True)

    # Chia train / val / seen_test trên tập in-specimen pool
    # Stratified split theo taxon_name để bảo đảm phân bố loài cân bằng tuyệt đối
    train_indices = []
    val_indices = []
    seen_test_indices = []

    for sp_name, group in df_in_pool.groupby("taxon_name"):
        idx_list = rng.permutation(group.index.values)
        n_val = max(1, int(len(idx_list) * val_ratio))
        n_seen = max(1, int(len(idx_list) * seen_test_ratio))
        
        val_indices.extend(idx_list[:n_val])
        seen_test_indices.extend(idx_list[n_val : n_val + n_seen])
        train_indices.extend(idx_list[n_val + n_seen :])

    df_train = df_in_pool.loc[train_indices].copy().reset_index(drop=True)
    df_val = df_in_pool.loc[val_indices].copy().reset_index(drop=True)
    df_seen_test = df_in_pool.loc[seen_test_indices].copy().reset_index(drop=True)

    # Gán nhãn phân vùng split
    df_train["eval_split"] = "train"
    df_val["eval_split"] = "val"
    df_seen_test["eval_split"] = "test_seen"
    df_unseen["eval_split"] = "test_unseen"

    # Remap lại specimen_idx trong tập train [0 .. N_train_specimens - 1] (77 phôi)
    train_vouchers = sorted(df_train["specimen_id"].unique())
    v_map = {v: idx for idx, v in enumerate(train_vouchers)}
    df_train["specimen_idx"] = df_train["specimen_id"].map(v_map)
    df_val["specimen_idx"] = df_val["specimen_id"].map(lambda x: v_map.get(x, 0))
    df_seen_test["specimen_idx"] = df_seen_test["specimen_id"].map(lambda x: v_map.get(x, 0))

    # Gán nhãn specimen_idx riêng cho tập unseen [0 .. N_unseen_specimens - 1] (33 phôi) để đo SRI Probe
    unseen_vouchers = sorted(df_unseen["specimen_id"].unique())
    unseen_v_map = {v: idx for idx, v in enumerate(unseen_vouchers)}
    df_unseen["specimen_idx"] = df_unseen["specimen_id"].map(unseen_v_map)

    print(f"\n[+] Phân chia dữ liệu hoàn tất cho Seed {seed} (3 bộ đánh giá + 3 phôi test/loài):")
    print(f"    - Train Set            : {len(df_train):,} ảnh (trên {len(train_vouchers)} phôi)")
    print(f"    - Bộ 1: Val Set (Seen) : {len(df_val):,} ảnh (trên cùng {len(train_vouchers)} phôi)")
    print(f"    - Bộ 2: Seen Test Set  : {len(df_seen_test):,} ảnh (trên cùng {len(train_vouchers)} phôi)")
    print(f"    - Bộ 3: Unseen Test Set: {len(df_unseen):,} ảnh (trên {len(unseen_voucher_ids)} phôi CHƯA TỪNG THẤY - 3 phôi/loài)")

    return df_train, df_val, df_seen_test, df_unseen


def build_train_transforms(image_size: int = 224) -> Callable:
    """Augmentation chuẩn hóa cho ảnh hiển vi/macro gỗ cắt ngang."""
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
    """Biến đổi chuẩn hóa xác định cho tập Val và Test."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class TimberDataset(Dataset):
    """
    Dataset đọc ảnh trực tiếp từ file path trên đĩa.
    Cam kết 100% nạp ảnh thật, không dùng bất kỳ fallback giả lập nào.
    """

    def __init__(self, df: pd.DataFrame, transform: Optional[Callable] = None, is_train: bool = False):
        self.df = df.reset_index(drop=True)
        self.transform = transform or (build_train_transforms() if is_train else build_eval_transforms())
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        img_path = Path(str(row["image_path"]))

        if not img_path.is_file():
            raise FileNotFoundError(f"[Thiếu file] Không tìm thấy ảnh thực tế: {img_path}")

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Lỗi đọc file ảnh {img_path}: {e}")

        tensor_img = self.transform(img)

        return {
            "image": tensor_img,
            "species_idx": torch.tensor(int(row["species_idx"]), dtype=torch.long),
            "specimen_idx": torch.tensor(int(row.get("specimen_idx", 0)), dtype=torch.long),
            "global_specimen_idx": torch.tensor(int(row.get("global_specimen_idx", 0)), dtype=torch.long),
            "image_id": str(row.get("image_id", idx)),
            "specimen_id": str(row.get("specimen_id", "unknown")),
        }


def create_seed_dataloaders(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_seen_test: pd.DataFrame,
    df_unseen: pd.DataFrame,
    batch_size: int = 64,
    num_workers: int = 2 if os.name != "nt" else 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader, Dict]:
    """
    Tạo DataLoaders cho 4 tập: Train, In-Specimen Val, In-Specimen Seen Test, Out-of-Specimen Unseen Test.
    """
    ds_train = TimberDataset(df_train, is_train=True)
    ds_val = TimberDataset(df_val, is_train=False)
    ds_seen_test = TimberDataset(df_seen_test, is_train=False)
    ds_unseen = TimberDataset(df_unseen, is_train=False)

    loader_train = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True if len(ds_train) > batch_size else False,
    )

    loader_val = DataLoader(
        ds_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    loader_seen_test = DataLoader(
        ds_seen_test,
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
        "num_classes": len(TARGET_SPECIES),
        "train_samples": len(ds_train),
        "val_samples": len(ds_val),
        "seen_test_samples": len(ds_seen_test),
        "unseen_samples": len(ds_unseen),
        "train_specimens": df_train["specimen_id"].nunique(),
        "unseen_specimens": df_unseen["specimen_id"].nunique(),
        "species_names": TARGET_SPECIES,
    }

    return loader_train, loader_val, loader_seen_test, loader_unseen, info


def create_dataloaders(
    metadata_csv: Optional[Union[str, Path]] = None,
    image_root: Optional[Union[str, Path]] = None,
    batch_size: int = 64,
    num_workers: int = 2 if os.name != "nt" else 0,
    seed: int = 42,
    unseen_vouchers_per_species: int = 3,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Wrapper tương thích ngược cho train_baseline.py, train_gan.py, evaluate_dual.py:
    Quét trực tiếp thư mục ảnh đĩa, phân chia dữ liệu động theo seed, và trả về:
    (loader_train, loader_val, loader_unseen, info).
    """
    df = scan_and_build_dataset(image_root)
    df_train, df_val, df_seen_test, df_unseen = split_dataset_by_seed(
        df, seed=seed, unseen_vouchers_per_species=unseen_vouchers_per_species
    )
    loader_train, loader_val, loader_seen_test, loader_unseen, info = create_seed_dataloaders(
        df_train, df_val, df_seen_test, df_unseen, batch_size=batch_size, num_workers=num_workers
    )
    return loader_train, loader_val, loader_unseen, info
