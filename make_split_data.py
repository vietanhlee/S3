"""
make_split_data.py
==================
Script phân chia và sao chép dữ liệu ảnh vật lý theo các phương pháp được cài đặt trong split_methods.py.
Đầu ra có cấu trúc thư mục: S3_split/{tên pp}/{train,val,test}/{genus}/{species}/ảnh.
"""

import argparse
import hashlib
import os
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from train import (
	set_seed,
	get_device,
	collect_image_samples,
	build_dataframe,
	compute_embeddings,
)
from split_methods import SPLIT_METHODS, validate_split


def ensure_unique_target(dst_dir: Path, src_path: Path, rel_path: str) -> Path:
	"""Tránh ghi đè file khi tên ảnh bị trùng lặp ở các thư mục con khác nhau."""
	target = dst_dir / src_path.name
	if not target.exists():
		return target

	# Sử dụng md5 của relative path để tạo hậu tố duy nhất
	digest = hashlib.md5(rel_path.encode("utf-8")).hexdigest()[:8]
	candidate = dst_dir / f"{src_path.stem}__{digest}{src_path.suffix}"
	if not candidate.exists():
		return candidate

	for i in range(1, 10000):
		candidate = dst_dir / f"{src_path.stem}__{digest}_{i}{src_path.suffix}"
		if not candidate.exists():
			return candidate

	raise RuntimeError(f"Quá nhiều file trùng tên tại thư mục: {dst_dir}")


def materialize_split(
	df_split: pd.DataFrame,
	split_name: str,
	method_output_dir: Path,
	data_dir: Path,
) -> None:
	"""Sao chép vật lý các tệp ảnh vào thư mục cấu trúc {split_name}/{genus}/{species}/ảnh."""
	dst_root = method_output_dir / split_name
	dst_root.mkdir(parents=True, exist_ok=True)

	progress = tqdm(total=len(df_split), desc=f"  Ghi tập {split_name}", unit="ảnh", leave=False)
	for _, row in df_split.iterrows():
		src = Path(row["path"])
		genus = row["genus"]
		species = row["species"]

		# Tính relative path so với data_dir gốc để băm md5 tránh trùng lặp
		try:
			rel_path = str(src.relative_to(data_dir))
		except ValueError:
			rel_path = src.name

		# Cấu trúc thư mục đích: S3_split/{pp}/{split}/{genus}/{species}/
		dst_dir = dst_root / genus / species
		dst_dir.mkdir(parents=True, exist_ok=True)

		dst_path = ensure_unique_target(dst_dir, src, rel_path)
		shutil.copy2(str(src), str(dst_path))
		progress.update(1)
	progress.close()


def main() -> None:
	parser = argparse.ArgumentParser(description="Script phân chia và tạo cấu trúc thư mục dữ liệu vật lý.")
	parser.add_argument("--data_dir", type=str, required=True, help="Đường dẫn đến thư mục dữ liệu gốc (S3).")
	parser.add_argument("--output_dir", type=str, default="S3_split", help="Đường dẫn thư mục đầu ra chứa kết quả chia.")
	parser.add_argument("--train_ratio", type=float, default=0.6, help="Tỷ lệ tập Train (mặc định: 0.6).")
	parser.add_argument("--val_ratio", type=float, default=0.2, help="Tỷ lệ tập Val (mặc định: 0.2).")
	parser.add_argument("--seed", type=int, default=42, help="Random seed (mặc định: 42).")
	parser.add_argument("--batch_size", type=int, default=128, help="Batch size khi extract embeddings (mặc định: 128).")
	parser.add_argument("--cosine_threshold", type=float, default=0.92, help="Cosine threshold cho PP5 (mặc định: 0.92).")
	args = parser.parse_args()

	set_seed(args.seed)
	device = get_device()
	print(f"Sử dụng thiết bị: {device}")

	data_dir = Path(args.data_dir)
	output_base = Path(args.output_dir)
	output_base.mkdir(parents=True, exist_ok=True)

	# 1. Thu thập ảnh
	print("\n[Bước 1] Thu thập và chuẩn bị dữ liệu...")
	samples = collect_image_samples(str(data_dir))
	if not samples:
		raise ValueError(f"Không tìm thấy ảnh nào tại đường dẫn: {data_dir}")

	df = build_dataframe(samples)
	print(f"Tổng số ảnh hợp lệ thu thập được (sau lọc ảnh lẻ): {len(df)}")
	print(f"Số lớp phân loại (class): {df['label'].nunique()}")

	# 2. Extract Embeddings (chỉ chạy 1 lần duy nhất)
	print("\n[Bước 2] Trích xuất đặc trưng (embeddings)...")
	embeddings = compute_embeddings(df, batch_size=args.batch_size, device=device)
	print(f"Kích thước embeddings trích xuất: {embeddings.shape}")

	# Giải phóng VRAM sau khi compute embeddings
	if device.type == "cuda":
		torch.cuda.empty_cache()

	# 3. Chạy phân chia và copy ảnh vật lý cho từng phương pháp
	print("\n[Bước 3] Bắt đầu phân chia và sao chép dữ liệu...")
	for method_name, split_fn in SPLIT_METHODS.items():
		print(f"\n⚡ Đang xử lý phương pháp: {method_name}...")
		method_output_dir = output_base / method_name

		# Xóa thư mục cũ nếu đã tồn tại để tránh rác dữ liệu
		if method_output_dir.exists():
			print(f"  Thư mục {method_output_dir} đã tồn tại. Đang xóa để khởi tạo lại...")
			shutil.rmtree(method_output_dir)
		method_output_dir.mkdir(parents=True, exist_ok=True)

		# Gọi hàm split
		try:
			if method_name == "PP5_Cosine_Graph":
				df_train, df_test, df_val = split_fn(
					df, embeddings,
					train_ratio=args.train_ratio,
					val_ratio=args.val_ratio,
					seed=args.seed,
					cosine_threshold=args.cosine_threshold,
				)
			else:
				df_train, df_test, df_val = split_fn(
					df, embeddings,
					train_ratio=args.train_ratio,
					val_ratio=args.val_ratio,
					seed=args.seed,
				)
		except Exception as e:
			print(f"  ❌ Lỗi khi phân chia phương pháp {method_name}: {e}")
			continue

		# Validate split
		validate_split(df, df_train, df_val, df_test, method_name)

		# Sao chép ảnh vật lý vào thư mục đích
		materialize_split(df_train, "train", method_output_dir, data_dir)
		materialize_split(df_val, "val", method_output_dir, data_dir)
		materialize_split(df_test, "test", method_output_dir, data_dir)
		
		print(f"  ✓ Đã hoàn thành và lưu tại: {method_output_dir}")

	print(f"\n🎉 Hoàn thành toàn bộ quy trình! Kết quả được lưu tại thư mục gốc: {output_base.resolve()}")


if __name__ == "__main__":
	main()
