"""
deduplicate_splits.py
=====================
Script kiểm tra và loại bỏ triệt để các ảnh trùng mã băm SHA-256 xuyên split
(Đảm bảo SHA-256 Overlap giữa Train, Val, Test hoàn toàn bằng 0).

Sử dụng:
    python deduplicate_splits.py --split-csv paper_data_assets/splits/split_canonical.csv
"""

import sys
import argparse
from pathlib import Path
import pandas as pd

def deduplicate_splits(split_csv_path: Path):
    if not split_csv_path.exists():
        print(f"[-] Không tìm thấy file: {split_csv_path}")
        return

    print("=" * 70)
    print(f"KIỂM TOÁN VÀ KHỬ TRÙNG LẶP SHA-256 CHO FILE: {split_csv_path.name}")
    print("=" * 70)

    df = pd.read_csv(split_csv_path)
    print(f"[*] Tổng số dòng ban đầu: {len(df):,}")

    if "sha256" not in df.columns:
        print("[!] File CSV không có cột 'sha256'. Vui lòng kiểm tra lại cấu trúc file.")
        return

    # 1. Kiểm tra trùng lặp trong nội bộ từng split
    for s in ["train", "val", "test"]:
        sub = df[df["split"] == s]
        dups = sub.duplicated(subset=["sha256"]).sum()
        if dups > 0:
            print(f"  [!] Phát hiện {dups} ảnh trùng SHA-256 ngay trong tập '{s}'.")

    # 2. Kiểm tra trùng lặp xuyên split
    train_hashes = set(df[df["split"] == "train"]["sha256"])
    val_hashes = set(df[df["split"] == "val"]["sha256"])
    test_hashes = set(df[df["split"] == "test"]["sha256"])

    tv_overlap = train_hashes.intersection(val_hashes)
    tt_overlap = train_hashes.intersection(test_hashes)
    vt_overlap = val_hashes.intersection(test_hashes)

    print(f"\n[*] KẾT QUẢ QUÉT TRÙNG LẶP XUYÊN SPLIT TRƯỚC KHI XỬ LÝ:")
    print(f"  - Trùng lặp Train vs Val : {len(tv_overlap)} mã SHA-256")
    print(f"  - Trùng lặp Train vs Test: {len(tt_overlap)} mã SHA-256")
    print(f"  - Trùng lặp Val vs Test  : {len(vt_overlap)} mã SHA-256")

    if len(tv_overlap) == 0 and len(tt_overlap) == 0 and len(vt_overlap) == 0:
        print("\n[SUCCESS] File split hoàn toàn sạch! Không có bất kỳ ảnh nào bị trùng SHA-256 xuyên split.")
        print("=" * 70)
        return

    # 3. Tiến hành khử trùng lặp (Ưu tiên giữ trong Train, xóa bản sao ở Val và Test để tránh Data Contamination)
    print(f"\n[*] Đang tiến hành loại bỏ các bản sao ở tập Test và Val để bảo vệ tính khách quan...")
    # Ưu tiên thứ tự giữ lại: Train > Val > Test
    # Nếu trùng giữa Train và Test -> xóa ở Test
    # Nếu trùng giữa Train và Val -> xóa ở Val
    # Nếu trùng giữa Val và Test -> xóa ở Test
    
    clean_rows = []
    seen_hashes = set()
    
    # Duyệt Train trước
    train_part = df[df["split"] == "train"].drop_duplicates(subset=["sha256"])
    for h in train_part["sha256"]:
        seen_hashes.add(h)
    clean_rows.append(train_part)

    # Duyệt Val, bỏ qua những hash đã có trong Train
    val_part = df[df["split"] == "val"].drop_duplicates(subset=["sha256"])
    val_clean = val_part[~val_part["sha256"].isin(seen_hashes)]
    for h in val_clean["sha256"]:
        seen_hashes.add(h)
    clean_rows.append(val_clean)

    # Duyệt Test, bỏ qua những hash đã có trong Train và Val
    test_part = df[df["split"] == "test"].drop_duplicates(subset=["sha256"])
    test_clean = test_part[~test_part["sha256"].isin(seen_hashes)]
    clean_rows.append(test_clean)

    df_clean = pd.concat(clean_rows, ignore_index=True)

    backup_path = split_csv_path.with_suffix(".csv.backup")
    df.to_csv(backup_path, index=False)
    print(f"[+] Đã tạo bản sao lưu file gốc tại: {backup_path.name}")

    df_clean.to_csv(split_csv_path, index=False)
    print(f"[+] Đã ghi đè file split sạch tại: {split_csv_path.name}")
    print(f"    - Số dòng mới: {len(df_clean):,} (Giảm {len(df) - len(df_clean)} dòng trùng lặp)")
    print(f"    - Train: {len(df_clean[df_clean['split'] == 'train']):,}")
    print(f"    - Val  : {len(df_clean[df_clean['split'] == 'val']):,}")
    print(f"    - Test : {len(df_clean[df_clean['split'] == 'test']):,}")

    print("\n[SUCCESS] Hoàn tất! Bây giờ SHA-256 Overlap giữa các split bằng 0 tuyệt đối.")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Khử trùng lặp SHA-256 xuyên split")
    parser.add_argument("--split-csv", type=str, default="paper_data_assets/splits/split_canonical.csv",
                        help="Đường dẫn đến file split CSV")
    args = parser.parse_args()
    deduplicate_splits(Path(args.split_csv))
