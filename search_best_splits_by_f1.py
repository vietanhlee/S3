"""
search_best_splits_by_f1.py
===========================
Script tự động hóa tìm kiếm phương pháp chia (PP) tối ưu nhất cho từng class gỗ (19 class)
dựa trên tiêu chí: Đạt giá trị F1-score thấp nhất (khắt khe nhất / thử thách nhất).

Logic hoạt động:
1. Duyệt qua 7 phương pháp chia: PP1, PP2, PP4, PP5, PP7, PP8, PP9.
2. Với mỗi PP, cấu hình tạm thời cho toàn bộ 19 classes sử dụng PP đó, dùng Swin-Large embeddings để chia.
3. Huấn luyện mô hình trong 17 epochs.
4. Sau khi kết thúc training, đánh giá trên tập Val (tương ứng mode "val") và tập Test (tương ứng mode "test")
   để thu thập F1-score của từng class.
5. Sau 7 lần chạy, với mỗi class, chọn ra PP và mode (val/test) có F1-score nhỏ nhất.
6. Lưu cấu hình tối ưu này thành tệp JSON/Python Dict.
7. Chạy lần thứ 8: Huấn luyện mô hình từ đầu với cấu hình tối ưu vừa tìm được trong 17 epochs và báo cáo kết quả cuối cùng.
"""

import os
import gc
import json
import random
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

# Import các hàm tiện ích và biến cấu hình từ train_final
import train_final
from utils import (
	set_seed,
	get_device,
	collect_image_samples,
	build_dataframe,
	validate_split_minimums,
	ImageListDataset,
	build_transforms,
)
from split_methods import validate_split

# ===== CẤU HÌNH TÌM KIẾM =====
ROOT_DIR = r"/kaggle/input/datasets/b23dckh002lvitanh/s3-origin/S3"
OUTPUT_BASE_DIR = "outputs_f1_search"
SEED = 42
EPOCHS = 17
BATCH_SIZE = 64  # Có thể điều chỉnh tùy thuộc vào GPU VRAM (ví dụ 64 hoặc 128)
CLASSIFICATION_MODEL_NAME = "swin_tiny_patch4_window7_224"  # Model dùng để train classification (có thể chỉnh thành convnext_tiny)
EMBEDDING_MODEL_NAME = "swin_large_patch4_window7_224"       # Model dùng để trích xuất embeddings chia dữ liệu
# =============================

# Thiết lập các cấu hình toàn cục trong train_final để đồng bộ
train_final.ROOT_DIR = ROOT_DIR
train_final.SEED = SEED
train_final.EPOCHS = EPOCHS
train_final.BATCH_SIZE = BATCH_SIZE
train_final.MODEL_NAME = CLASSIFICATION_MODEL_NAME

PP_LIST = ["PP1", "PP2", "PP4", "PP5", "PP7", "PP8", "PP9"]


@torch.no_grad()
def get_f1_scores(model, loader, device, class_names):
	"""Đánh giá mô hình và trích xuất F1-score của từng lớp."""
	model.eval()
	y_true, y_pred = [], []
	for images, targets in loader:
		images = images.to(device)
		logits = model(images)
		preds = torch.argmax(logits, dim=1).cpu().tolist()
		y_pred.extend(preds)
		y_true.extend(targets.tolist())
		
	report = classification_report(
		y_true, y_pred, labels=list(range(len(class_names))), target_names=class_names, digits=4, output_dict=True
	)
	
	f1_scores = {}
	for name in class_names:
		# Trích xuất f1-score cho từng class, mặc định là 0.0 nếu có lỗi hoặc không có mẫu
		f1_scores[name] = report.get(name, {}).get("f1-score", 0.0)
	return f1_scores


def main():
	set_seed(SEED)
	device = get_device()
	print(f"Device huấn luyện: {device}")

	output_dir = Path(OUTPUT_BASE_DIR)
	output_dir.mkdir(parents=True, exist_ok=True)

	# 1. Thu thập ảnh
	print("\n[Step 1] Thu thập dữ liệu ảnh...")
	samples = collect_image_samples(ROOT_DIR)
	if not samples:
		raise ValueError(f"Không tìm thấy ảnh nào tại {ROOT_DIR}")

	df = build_dataframe(samples)
	df_filtered = df[df["label"] != "Pterocarpus sp"].reset_index(drop=True)
	print(f"Tổng số ảnh sau khi lọc bỏ Pterocarpus sp: {len(df_filtered)}")

	class_names = sorted(df_filtered["label"].unique().tolist())
	class_to_idx = {name: i for i, name in enumerate(class_names)}

	# 2. Compute embeddings (Swin-Large)
	print(f"\n[Step 2] Compute embeddings bằng {EMBEDDING_MODEL_NAME}...")
	embs_swin = train_final.compute_embeddings_v2(
		df_filtered, EMBEDDING_MODEL_NAME, batch_size=BATCH_SIZE, device=device
	)

	# Lưu trữ kết quả F1-score của từng class ứng với từng PP và mode
	# Cấu trúc: { class_name: [ {"pp": "PP1", "mode": "val", "f1": 0.5}, ... ] }
	f1_results = {name: [] for name in class_names}

	# 3. Chạy huấn luyện 7 lần tương ứng với 7 PP chia dữ liệu
	print("\n[Step 3] Bắt đầu huấn luyện thử nghiệm trên 7 PP khác nhau...")
	for pp in PP_LIST:
		print("\n" + "=" * 80)
		print(f" TIẾN HÀNH THỬ NGHIỆM VỚI PHƯƠNG PHÁP CHIA: {pp}")
		print("=" * 80)

		# Cấu hình split tạm thời: Tất cả các class đều dùng pp này, mặc định test mode và swin embed
		tmp_config = {name: (pp, "test", "swin") for name in class_names}
		train_final.SPLIT_CONFIG = tmp_config

		# Chia dữ liệu (truyền embs_swin cho cả 2 đối số eff và swin để ép dùng Swin embeddings)
		df_train, df_val, df_test = train_final.end_version_split(
			df_filtered, embs_swin, embs_swin,
			train_ratio=0.6, val_ratio=0.2, seed=SEED
		)

		# Chuẩn bị DataLoaders
		model_temp = train_final.build_model(num_classes=len(class_names))
		cfg_model = train_final.resolve_data_config({}, model=model_temp)
		img_size = cfg_model.get("input_size", (3, 224, 224))[-1]
		mean = cfg_model.get("mean", (0.485, 0.456, 0.406))
		std = cfg_model.get("std", (0.229, 0.224, 0.225))
		train_tf, eval_tf = build_transforms(img_size, mean, std)

		train_ds = ImageListDataset(df_train, class_to_idx, transform=train_tf)
		val_ds = ImageListDataset(df_val, class_to_idx, transform=eval_tf)
		test_ds = ImageListDataset(df_test, class_to_idx, transform=eval_tf)

		train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
		val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
		test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

		# Khởi tạo mô hình huấn luyện
		model = train_final.build_model(num_classes=len(class_names)).to(device)
		criterion = train_final.FocalLoss(gamma=train_final.FOCAL_GAMMA, alpha=train_final.FOCAL_ALPHA)
		optimizer = torch.optim.AdamW(
			filter(lambda p: p.requires_grad, model.parameters()),
			lr=train_final.LR,
			weight_decay=train_final.WEIGHT_DECAY,
		)
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

		# Thư mục lưu kết quả tạm thời cho PP này
		pp_output_dir = output_dir / f"run_{pp}"
		pp_output_dir.mkdir(parents=True, exist_ok=True)

		# Huấn luyện
		train_final.train_model(
			model, train_loader, val_loader, optimizer, criterion,
			device, epochs=EPOCHS, patience=train_final.PATIENCE, output_dir=pp_output_dir,
			scheduler=scheduler,
		)

		# Load checkpoint tốt nhất
		best_path = pp_output_dir / f"best_model_{CLASSIFICATION_MODEL_NAME}.pth"
		if best_path.exists():
			model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))

		# Đánh giá lấy F1-score
		print(f"Đang đánh giá kết quả cho phương pháp {pp}...")
		val_f1_scores = get_f1_scores(model, val_loader, device, class_names)
		test_f1_scores = get_f1_scores(model, test_loader, device, class_names)

		# Lưu kết quả
		for name in class_names:
			f1_results[name].append({"pp": pp, "mode": "val", "f1": val_f1_scores[name]})
			f1_results[name].append({"pp": pp, "mode": "test", "f1": test_f1_scores[name]})

		# Dọn dẹp bộ nhớ GPU
		del model, model_temp, optimizer, scheduler, train_loader, val_loader, test_loader
		if device.type == "cuda":
			torch.cuda.empty_cache()
		gc.collect()

	# 4. Xác định cấu hình chia dữ liệu tối ưu nhất (F1 nhỏ nhất) cho từng class
	print("\n[Step 4] Phân tích tìm kiếm cấu hình tối ưu (F1-score nhỏ nhất)...")
	optimal_split_config = {}
	report_lines = []
	report_lines.append("=" * 80)
	report_lines.append(f"{'Class Name':<30} | {'Optimal PP':<12} | {'Mode':<8} | {'Min F1-Score':<12}")
	report_lines.append("-" * 80)

	for name in class_names:
		class_runs = f1_results[name]
		# Tìm run có F1-score nhỏ nhất
		best_run = min(class_runs, key=lambda x: x["f1"])
		best_pp = best_run["pp"]
		best_mode = best_run["mode"]
		min_f1 = best_run["f1"]

		optimal_split_config[name] = [best_pp, best_mode, "swin"]
		report_lines.append(f"{name:<30} | {best_pp:<12} | {best_mode:<8} | {min_f1:<12.4f}")

	report_lines.append("=" * 80)
	report_str = "\n".join(report_lines)
	print("\n" + report_str)

	# Ghi báo cáo ra file text
	with open(output_dir / "optimal_splits_report.txt", "w", encoding="utf-8") as f:
		f.write(report_str)

	# Ghi kết quả cấu hình tối ưu ra file JSON
	with open(output_dir / "optimal_split_config.json", "w", encoding="utf-8") as f:
		json.dump(optimal_split_config, f, indent=2, ensure_ascii=False)

	# 5. Huấn luyện lần thứ 8: Sử dụng cấu hình tối ưu vừa tìm được
	print("\n" + "=" * 80)
	print(" [Step 5] BẮT ĐẦU HUẤN LUYỆN LẦN 8 VỚI CẤU HÌNH TỐI ƯU CUỐI CÙNG")
	print("=" * 80)

	train_final.SPLIT_CONFIG = optimal_split_config

	# Chia dữ liệu theo cấu hình tối ưu
	df_train, df_val, df_test = train_final.end_version_split(
		df_filtered, embs_swin, embs_swin,
		train_ratio=0.6, val_ratio=0.2, seed=SEED
	)

	validate_split(df_filtered, df_train, df_val, df_test, "Optimal_End_Version_Split")
	train_final.log_split_summary(df_filtered, df_train, df_val, df_test)

	# DataLoaders cho lần chạy cuối
	model_temp = train_final.build_model(num_classes=len(class_names))
	cfg_model = train_final.resolve_data_config({}, model=model_temp)
	img_size = cfg_model.get("input_size", (3, 224, 224))[-1]
	mean = cfg_model.get("mean", (0.485, 0.456, 0.406))
	std = cfg_model.get("std", (0.229, 0.224, 0.225))
	train_tf, eval_tf = build_transforms(img_size, mean, std)

	train_ds = ImageListDataset(df_train, class_to_idx, transform=train_tf)
	val_ds = ImageListDataset(df_val, class_to_idx, transform=eval_tf)
	test_ds = ImageListDataset(df_test, class_to_idx, transform=eval_tf)

	train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
	val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
	test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

	# Khởi tạo mô hình
	model = train_final.build_model(num_classes=len(class_names)).to(device)
	criterion = train_final.FocalLoss(gamma=train_final.FOCAL_GAMMA, alpha=train_final.FOCAL_ALPHA)
	optimizer = torch.optim.AdamW(
		filter(lambda p: p.requires_grad, model.parameters()),
		lr=train_final.LR,
		weight_decay=train_final.WEIGHT_DECAY,
	)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

	final_output_dir = output_dir / "final_optimal_run"
	final_output_dir.mkdir(parents=True, exist_ok=True)

	# Ghi danh sách index lớp ra thư mục cuối cùng
	with open(final_output_dir / "class_indices.json", "w", encoding="utf-8") as f:
		json.dump(class_to_idx, f, indent=2, ensure_ascii=False)

	# Huấn luyện mô hình tối ưu
	history = train_final.train_model(
		model, train_loader, val_loader, optimizer, criterion,
		device, epochs=EPOCHS, patience=train_final.PATIENCE, output_dir=final_output_dir,
		scheduler=scheduler,
	)
	train_final.plot_training_curves(history, final_output_dir)

	# Load best model final để đánh giá xuất báo cáo
	best_path = final_output_dir / f"best_model_{CLASSIFICATION_MODEL_NAME}.pth"
	if best_path.exists():
		model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
		print(f"\nĐã load checkpoint tốt nhất cuối cùng từ {best_path}")

	# Đánh giá và ghi báo cáo trên tập Val và tập Test
	print("\nĐánh giá trên tập Val...")
	train_final.evaluate_and_report(model, val_loader, device, class_names, final_output_dir, prefix="val")

	print("\nĐánh giá trên tập Test...")
	train_final.evaluate_and_report(model, test_loader, device, class_names, final_output_dir, prefix="test")

	# Lưu file metadata kết quả
	result_summary = {
		"model_name": CLASSIFICATION_MODEL_NAME,
		"epochs": EPOCHS,
		"best_val_acc": history.get("best_val_acc", 0.0),
		"train_size": len(df_train),
		"val_size": len(df_val),
		"test_size": len(df_test),
		"optimal_split_config": optimal_split_config,
	}
	with open(final_output_dir / "final_summary.json", "w", encoding="utf-8") as f:
		json.dump(result_summary, f, indent=2, ensure_ascii=False)

	print(f"\n[Hoàn tất] Quá trình tìm kiếm và huấn luyện tối ưu đã xong!")
	print(f"  - Kết quả báo cáo so sánh F1: {output_dir / 'optimal_splits_report.txt'}")
	print(f"  - Cấu hình JSON tối ưu được lưu tại: {output_dir / 'optimal_split_config.json'}")
	print(f"  - Kết quả và checkpoint chạy lần cuối lưu tại: {final_output_dir}/")


if __name__ == "__main__":
	main()
