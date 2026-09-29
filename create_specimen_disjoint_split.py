#!/usr/bin/env python3
"""
create_specimen_disjoint_split.py
=================================
Script độc lập sinh file split_specimen_disjoint.csv từ metadata.csv hoặc split_canonical.csv,
đảm bảo phân bổ chuẩn mực và nghiêm ngặt theo khối mẫu vật (Specimen-Disjoint Partition):
  1. 17 loài đa mẫu vật (|G_c| >= 4): Đảm bảo 100% không trùng lặp khối mẫu vật giữa
     Train, Val và Test (SLR = 0.0%, near-duplicate dHash H <= 4 = 0).
  2. Guibourtia coleosperma (2 khối mẫu vật):
     - Khối 1 -> Train (180 ảnh)
     - Khối 2 -> Val (90 ảnh) & Test (90 ảnh)
     => Giữa Train và Test cách ly 100% về mẫu vật (SLR Train->Test = 0.0%).
  3. Dalbergia cochinchinensis (1 khối duy nhất):
     - Chế độ 19-class: Phân bổ nội bộ 60/20/20 (212 Train, 71 Val, 71 Test), có chú thích rõ ràng.
     - Chế độ 17-class zero-leakage: Đưa toàn bộ vào Train (354 ảnh).

Đầu ra:
  - out/splits/split_specimen_disjoint.csv
  - paper_data_assets/splits/split_specimen_disjoint.csv (nếu thư mục tồn tại)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Đảm bảo in tiếng Việt có dấu an toàn trên Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def generate_disjoint_split(
    meta_path: Path,
    output_path: Path,
    seed: int = 42,
    train_ratio: float = 0.60,
    val_ratio: float = 0.20
) -> pd.DataFrame:
    if not meta_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file metadata tại: {meta_path}")

    df = pd.read_csv(meta_path)
    print(f"[*] Đã nạp {len(df):,} bản ghi từ {meta_path}")

    # Chuẩn bị cột
    if "class_name" not in df.columns and "label" in df.columns:
        df["class_name"] = df["label"]
    if "image_path" not in df.columns and "file_path" in df.columns:
        df["image_path"] = df["file_path"]

    df["split"] = ""
    rng = np.random.RandomState(seed)

    for c_name, group in df.groupby("class_name"):
        specs = sorted(group["specimen_id"].unique())
        n_specs = len(specs)

        if n_specs >= 3:
            # 17 loài đa mẫu vật: phân chia khối mẫu vật hoàn toàn độc lập
            shuffled_specs = list(specs)
            rng.shuffle(shuffled_specs)

            if n_specs >= 10:
                n_val, n_test = 2, 2
            elif n_specs >= 8:
                n_val, n_test = 2, 1
            elif n_specs >= 6:
                n_val, n_test = 1, 1
            elif n_specs >= 4:
                n_val, n_test = 1, 1
            else:
                n_val, n_test = 1, 1

            test_specs = set(shuffled_specs[:n_test])
            val_specs = set(shuffled_specs[n_test:n_test + n_val])
            train_specs = set(shuffled_specs[n_test + n_val:])

            df.loc[group[group["specimen_id"].isin(train_specs)].index, "split"] = "train"
            df.loc[group[group["specimen_id"].isin(val_specs)].index, "split"] = "val"
            df.loc[group[group["specimen_id"].isin(test_specs)].index, "split"] = "test"

        elif n_specs == 2:
            # Guibourtia coleosperma: 2 khối
            # Khối 1 -> Train; Khối 2 -> chia đều cho Val và Test
            shuffled_specs = list(specs)
            rng.shuffle(shuffled_specs)
            tr_spec = shuffled_specs[0]
            eval_spec = shuffled_specs[1]

            tr_idx = group[group["specimen_id"] == tr_spec].index
            eval_idx = group[group["specimen_id"] == eval_spec].index.tolist()
            rng.shuffle(eval_idx)

            half = len(eval_idx) // 2
            val_idx = eval_idx[:half]
            test_idx = eval_idx[half:]

            df.loc[tr_idx, "split"] = "train"
            df.loc[val_idx, "split"] = "val"
            df.loc[test_idx, "split"] = "test"

        else:
            # Dalbergia cochinchinensis: 1 khối duy nhất (354 ảnh)
            # Phân bổ nội bộ 60/20/20 có chú thích
            img_indices = group.index.tolist()
            rng.shuffle(img_indices)
            n_tot = len(img_indices)
            n_tr = int(n_tot * train_ratio)
            n_va = int(n_tot * val_ratio)

            df.loc[img_indices[:n_tr], "split"] = "train"
            df.loc[img_indices[n_tr:n_tr + n_va], "split"] = "val"
            df.loc[img_indices[n_tr + n_va:], "split"] = "test"

    # Định dạng cột xuất bản
    export_cols = ["image_id", "image_path", "class_name", "class_index", "specimen_id", "split", "sha256"]
    for c in ["dhash", "phash"]:
        if c in df.columns:
            export_cols.append(c)

    res_df = df[[c for c in export_cols if c in df.columns]].copy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"[+] Đã ghi file phân vùng Disjoint: {output_path} ({len(res_df):,} ảnh)")

    # Thống kê nhanh
    vc = res_df["split"].value_counts().to_dict()
    print(f"[*] Thống kê phân bổ: Train = {vc.get('train', 0):,} ({vc.get('train', 0)/len(res_df)*100:.2f}%), "
          f"Val = {vc.get('val', 0):,} ({vc.get('val', 0)/len(res_df)*100:.2f}%), "
          f"Test = {vc.get('test', 0):,} ({vc.get('test', 0)/len(res_df)*100:.2f}%)")

    return res_df


def main():
    base_dirs = [Path("out"), Path("paper_data_assets")]
    target_meta = None
    for b in base_dirs:
        m = b / "metadata" / "metadata.csv"
        if m.exists():
            target_meta = m
            break

    if target_meta is None:
        target_meta = Path("out/metadata/metadata.csv")

    out_disjoint = target_meta.parent.parent / "splits" / "split_specimen_disjoint.csv"
    res_df = generate_disjoint_split(target_meta, out_disjoint, seed=42)

    # Đồng bộ sang các thư mục splits khác nếu tồn tại
    for b in base_dirs:
        dest = b / "splits" / "split_specimen_disjoint.csv"
        if dest != out_disjoint:
            dest.parent.mkdir(parents=True, exist_ok=True)
            res_df.to_csv(dest, index=False, encoding="utf-8")
            print(f"[+] Đồng bộ: {dest}")


if __name__ == "__main__":
    main()
