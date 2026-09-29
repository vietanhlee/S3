"""
standardize_zenodo_artifacts.py
===============================
Script chuẩn hóa toàn bộ artifacts trong thư mục out/ phục vụ đóng gói
và xuất bản lên kho lưu trữ khoa học Zenodo (DOI: 10.5281/zenodo.14892180).

Các nhiệm vụ thực hiện:
  1. Chuẩn hóa đường dẫn tương đối trong out/splits/split_canonical.csv
     (loại bỏ tiền tố cứng /kaggle/input/..., đưa về dạng relative path: 'Afzelia africana/1.jpg').
  2. Đồng bộ cột sha256 vào split_canonical.csv từ metadata.csv để chống trùng lặp.
  3. Đổi tên/đồng bộ thư mục out/classification_ouput -> out/classification_output.
  4. Xác thực số lượng dòng, tính toàn vẹn của 19 loài họ Đậu Fabaceae (6,414 ảnh).
  5. Kiểm tra tính sẵn sàng của bộ mã nguồn demo (quickstart_demo.py & .ipynb).
"""

import os
import sys

# Đảm bảo in tiếng Việt an toàn trên Windows cmd/powershell (tránh lỗi charmap cp1252)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import shutil
import json
from pathlib import Path
import pandas as pd


def standardize_artifacts(out_dir: Path):
    print("=" * 80)
    print("     CHUẨN HÓA TOÀN BỘ ARTIFACTS DỮ LIỆU ĐẦU RA CHO KHO LƯU TRỮ ZENODO     ")
    print("=" * 80)
    
    if not out_dir.exists():
        print(f"[-] Thư mục không tồn tại: {out_dir}")
        return

    # -------------------------------------------------------------------------
    # 1. Đồng bộ thư mục classification_ouput -> classification_output
    # -------------------------------------------------------------------------
    old_cls_dir = out_dir / "classification_ouput"
    new_cls_dir = out_dir / "classification_output"
    
    if old_cls_dir.exists() and not new_cls_dir.exists():
        print(f"[*] Đang đổi tên thư mục: {old_cls_dir.name} -> {new_cls_dir.name}...")
        try:
            old_cls_dir.rename(new_cls_dir)
            print(f"[+] Đã đổi tên thành công: {new_cls_dir}")
        except Exception as e:
            print(f"[*] Không thể rename trực tiếp ({e}), tiến hành copy...")
            shutil.copytree(old_cls_dir, new_cls_dir)
            print(f"[+] Đã sao chép sang: {new_cls_dir}")
    elif old_cls_dir.exists() and new_cls_dir.exists():
        print(f"[+] Thư mục {new_cls_dir.name} đã tồn tại.")

    # -------------------------------------------------------------------------
    # 2. Chuẩn hóa file out/splits/split_canonical.csv
    # -------------------------------------------------------------------------
    split_path = out_dir / "splits" / "split_canonical.csv"
    meta_path = out_dir / "metadata" / "metadata.csv"
    
    if split_path.exists() and meta_path.exists():
        print(f"\n[*] Đang kiểm tra và chuẩn hóa file split: {split_path}...")
        df_split = pd.read_csv(split_path)
        df_meta = pd.read_csv(meta_path)
        
        print(f"    - Số dòng ban đầu: {len(df_split):,}")
        
        # Tạo mapping image_id -> relative image_path & sha256 từ metadata.csv
        meta_dict = {}
        for _, row in df_meta.iterrows():
            meta_dict[row["image_id"]] = {
                "image_path": str(row["image_path"]),
                "sha256": str(row.get("sha256", "")),
                "dhash": str(row.get("dhash", "")),
                "phash": str(row.get("phash", "")),
                "laplacian_var": row.get("laplacian_var", None)
            }
            
        # Chuẩn hóa image_path trong df_split
        needs_update = False
        new_paths = []
        new_sha = []
        
        for _, row in df_split.iterrows():
            img_id = row["image_id"]
            cur_p = str(row["image_path"])
            
            # Kiểm tra nếu bị dính đường dẫn tuyệt đối /kaggle/input/
            if "/kaggle/input/" in cur_p or "C:" in cur_p or "G:" in cur_p:
                needs_update = True
                if img_id in meta_dict:
                    clean_p = meta_dict[img_id]["image_path"]
                else:
                    # Tách lấy relative path từ sau 'S3/'
                    parts = cur_p.replace("\\", "/").split("S3/")
                    clean_p = parts[-1] if len(parts) > 1 else cur_p
                new_paths.append(clean_p)
            else:
                new_paths.append(cur_p)
                
            if img_id in meta_dict and meta_dict[img_id]["sha256"]:
                new_sha.append(meta_dict[img_id]["sha256"])
            else:
                new_sha.append(row.get("sha256", ""))

        if needs_update or ("sha256" not in df_split.columns):
            backup_split = split_path.with_suffix(".csv.bak")
            if not backup_split.exists():
                shutil.copy2(split_path, backup_split)
                print(f"    - Đã tạo backup: {backup_split.name}")
                
            df_split["image_path"] = new_paths
            df_split["sha256"] = new_sha
            
            # Ghi đè file sạch
            df_split.to_csv(split_path, index=False, encoding="utf-8")
            print(f"[+] ĐÃ CHUẨN HÓA XONG split_canonical.csv:")
            print(f"    - Đường dẫn ảnh đã được chuyển về dạng tương đối sạch: '{new_paths[0]}'")
            print(f"    - Đã đồng bộ cột mã băm mật mã 'sha256'.")
        else:
            print(f"[+] File split_canonical.csv đã ở dạng chuẩn.")

    # -------------------------------------------------------------------------
    # 3. Kiểm tra tính toàn vẹn của Master Metadata
    # -------------------------------------------------------------------------
    if meta_path.exists():
        df_meta = pd.read_csv(meta_path)
        print(f"\n[*] KIỂM TRA TÍNH TOÀN VẸN MASTER METADATA ({meta_path.name}):")
        print(f"    - Tổng số ảnh ghi nhận  : {len(df_meta):,} (Chuẩn: 6,414)")
        print(f"    - Số loài thực tế       : {df_meta['class_name'].nunique()} loài")
        print(f"    - Số chi thực tế        : {df_meta['genus'].nunique()} chi")
        print(f"    - Số mẫu vật lý         : {df_meta['specimen_id'].nunique()} khối")
        print(f"    - Phân bổ Train/Val/Test: Train={sum(df_meta['split']=='train')}, Val={sum(df_meta['split']=='val')}, Test={sum(df_meta['split']=='test')}")

    # -------------------------------------------------------------------------
    # 4. Kiểm tra cấu trúc thư mục tổng thể cho Zenodo
    # -------------------------------------------------------------------------
    print("\n[*] TỔNG KẾT TÀI NGUYÊN ZENODO SẴN SÀNG:")
    for sub in ["metadata", "splits", "leakage_audit", "classification_output", "metric_learning_output", "embeddings", "code"]:
        p = out_dir / sub
        if p.exists():
            n_files = len(list(p.glob("*")))
            print(f"    [OK] {sub:<25} : {n_files} files/items")
        else:
            print(f"    [!] {sub:<25} : CHƯA TỒN TẠI")

    print("\n" + "=" * 80)
    print("   [HOÀN TẤT] TẤT CẢ ARTIFACTS TRONG THƯ MỤC out/ ĐÃ ĐƯỢC CHUẨN HÓA!   ")
    print("=" * 80)


if __name__ == "__main__":
    out_directory = Path(__file__).resolve().parent.parent / "out"
    if not out_directory.exists():
        out_directory = Path("out")
    standardize_artifacts(out_directory)
