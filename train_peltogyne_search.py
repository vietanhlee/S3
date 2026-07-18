"""
train_peltogyne_search.py
==========================
Pipeline tìm kiếm phương pháp chia dữ liệu (PP) tối ưu nhất cho riêng lớp
'Peltogyne pubescens' (đạt F1-score nhỏ nhất trên tập Test) để đánh giá độ khó của mô hình.
Các lớp khác giữ nguyên cấu hình chia như trong train_final.py.

Cách chạy:
python train_peltogyne_search.py
"""

import os
import gc
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from timm.data import resolve_data_config

# Import các helper dùng chung từ utils
from utils import (
	set_seed,
	get_device,
	collect_image_samples,
	build_dataframe,
	ImageListDataset,
	build_transforms,
	summarize_model,
	freeze_model_layers,
	validate_split_minimums,
)

# Import các hàm từ train_final
from train_final import (
	FocalLoss,
	accuracy_from_logits,
	train_model,
	plot_training_curves,
	collect_predictions,
	save_report,
	compute_embeddings_v2,
	build_model,
)

# Import từ split_methods
from split_methods import (
	SPLIT_METHODS,
	validate_split,
)

# ===== CẤU HÌNH =====
ROOT_DIR = r"/kaggle/input/datasets/b23dckh002lvitanh/s3-origin/S3"
OUTPUT_BASE_DIR = "outputs_peltogyne_search"
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
SEED = 42
BATCH_SIZE = 128
EPOCHS = 10              # Đặt 10 epochs giống train_split_comparison.py để tìm nhanh
PATIENCE = 5
LR = 5e-4
WEIGHT_DECAY = 1e-2
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.25
MODEL_NAME = "convnext_tiny"
FREEZE_RATIO = 0.90
COSINE_THRESHOLD = 0.92  # Dùng cho PP5 Cosine Graph

# ===== CHẾ ĐỘ CHẠY (RUN MODES) =====
# Chế độ chạy: "search" (mặc định - huấn luyện 10 epochs để tìm PP tốt nhất)
#              "infer"  (chỉ load checkpoint đã có và chạy đánh giá, không huấn luyện)
RUN_MODE = "infer"

# Đường dẫn checkpoint để load ở chế độ "infer":
# - Nếu là đường dẫn tới 1 file .pth cụ thể: Dùng chung file này cho tất cả các PP cần đánh giá.
# - Nếu là một thư mục (ví dụ: outputs_peltogyne_search): Tự động tìm model tương ứng trong thư mục con của từng PP.
#   Ví dụ: {INFER_MODEL_PATH}/{pp}/best_model_{MODEL_NAME}.pth
INFER_MODEL_PATH = "/kaggle/input/models/huecute/model-end/pytorch/default/1/best_model_convnext_tiny.pth"
# =====================


def customized_end_version_split(
	df: pd.DataFrame,
	embs_eff: np.ndarray,
	embs_swin: np.ndarray,
	peltogyne_pp_key: str,
	train_ratio: float = 0.70,
	val_ratio: float = 0.15,
	peltogyne_train_ratio: float = 0.70,
	peltogyne_val_ratio: float = 0.15,
	seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	"""Chia dữ liệu End Version, trong đó chỉ riêng Peltogyne pubescens dùng pp_key tùy chọn."""
	# 1. Loại bỏ class 'Pterocarpus sp'
	keep_mask = df["label"] != "Pterocarpus sp"
	df_filtered = df[keep_mask].reset_index(drop=True)
	emb_eff_filtered = embs_eff[keep_mask.values]
	emb_swin_filtered = embs_swin[keep_mask.values]

	# 2. Cấu hình phân chia gốc từ train_final.py
	split_config = {
		"Afzelia africana": ("PP8", "val", "eff"),
		"Afzelia bella": ("PP4", "val", "swin"),
		"Afzelia pachyloba": ("PP9", "test", "swin"),
		"Afzelia quanzensis": ("PP2", "val", "eff"),
		"Dalbergia cochinchinensis": ("PP9", "val", "eff"),
		"Dalbergia melanoxylon": ("PP2", "test", "eff"),
		"Dalbergia oliveri": ("PP8", "val", "eff"),
		"Dalbergia rimosa": ("PP4", "test", "eff"),
		"Dalbergia tonkinensis": ("PP4", "test", "swin"),
		"Guibourtia arnoldiana": ("PP4", "test", "swin"),
		"Guibourtia coleosperma": ("PP9", "test", "swin"),
		"Guibourtia ehie": ("PP4", "test", "swin"),
		"Peltogyne pubescens": (peltogyne_pp_key, "test", "eff"),  # Ghi đè phương pháp tại đây
		"Pterocarpus erinaceus": ("PP9", "val", "eff"),
		"Pterocarpus indicus": ("PP9", "test", "eff"),
		"Pterocarpus macrocarpus": ("PP4", "test", "eff"),
		"Pterocarpus soyauxii": ("PP4", "test", "swin"),
		"Sindora cochinchinensis": ("PP2", "test", "swin"),
		"Sindora tonkinensis": ("PP9", "val", "eff"),
	}

	pp_map = {
		"PP1": "PP1_Mahalanobis_Fixed",
		"PP2": "PP2_Mahalanobis_Iterative",
		"PP4": "PP4_Hierarchical_Clustering",
		"PP5": "PP5_Cosine_Graph",
		"PP7": "PP7_Adversarial_Validation",
		"PP8": "PP8_StratifiedGroupKFold",
		"PP9": "PP9_Agglom_Stratified",
	}

	train_idx_all = []
	val_idx_all = []
	test_idx_all = []

	# Duyệt từng lớp để thực hiện phân chia độc lập
	for label, group in df_filtered.groupby("label"):
		indices = group.index.tolist()
		sub_df = group.copy()
		path_to_orig_idx = dict(zip(group["path"], group.index))
		sub_df_reset = sub_df.reset_index(drop=True)

		pp_key, mode, model_type = split_config[label]

		# Xác định ratio cho từng class
		if label == "Peltogyne pubescens":
			tr_ratio = peltogyne_train_ratio
			va_ratio = peltogyne_val_ratio
		else:
			tr_ratio = train_ratio
			va_ratio = val_ratio

		# Lấy embedding tương ứng
		if model_type == "eff":
			sub_emb = emb_eff_filtered[indices]
		else:
			sub_emb = emb_swin_filtered[indices]

		full_pp_name = pp_map[pp_key]
		split_fn = SPLIT_METHODS[full_pp_name]

		try:
			if pp_key == "PP5":
				tr_df, val_df, te_df = split_fn(
					sub_df_reset, sub_emb,
					train_ratio=tr_ratio,
					val_ratio=va_ratio,
					seed=seed,
					cosine_threshold=COSINE_THRESHOLD,
				)
			else:
				tr_df, val_df, te_df = split_fn(
					sub_df_reset, sub_emb,
					train_ratio=tr_ratio,
					val_ratio=va_ratio,
					seed=seed,
				)
		except Exception as e:
			from split_methods import stratified_random_split
			tr_df, val_df, te_df = stratified_random_split(
				sub_df_reset, sub_emb,
				train_ratio=tr_ratio,
				val_ratio=va_ratio,
				seed=seed,
			)

		tr_orig_idx = [path_to_orig_idx[p] for p in tr_df["path"]]
		val_orig_idx = [path_to_orig_idx[p] for p in val_df["path"]]
		te_orig_idx = [path_to_orig_idx[p] for p in te_df["path"]]

		if mode == "val":
			train_idx_all.extend(tr_orig_idx)
			val_idx_all.extend(te_orig_idx)
			test_idx_all.extend(val_orig_idx)
		else:
			train_idx_all.extend(tr_orig_idx)
			val_idx_all.extend(val_orig_idx)
			test_idx_all.extend(te_orig_idx)

	df_train = df_filtered.loc[train_idx_all].sample(frac=1, random_state=seed).reset_index(drop=True)
	df_val = df_filtered.loc[val_idx_all].sample(frac=1, random_state=seed).reset_index(drop=True)
	df_test = df_filtered.loc[test_idx_all].sample(frac=1, random_state=seed).reset_index(drop=True)

	return df_train, df_val, df_test


def main():
	set_seed(SEED)
	device = get_device()
	print(f"Khởi động tiến trình tìm kiếm PP cho Peltogyne pubescens trên device: {device}")

	output_base = Path(OUTPUT_BASE_DIR)
	output_base.mkdir(parents=True, exist_ok=True)

	# 1. Thu thập ảnh & lọc dữ liệu
	print("\n[Step 1] Thu thập và tiền xử lý ảnh...")
	samples = collect_image_samples(ROOT_DIR)
	if not samples:
		raise ValueError(f"Không tìm thấy ảnh nào tại {ROOT_DIR}")
	df = build_dataframe(samples)
	df_filtered = df[df["label"] != "Pterocarpus sp"].reset_index(drop=True)
	print(f"Tổng số ảnh sau lọc: {len(df_filtered)}, Số loài gỗ: {df_filtered['label'].nunique()}")

	class_names = sorted(df_filtered["label"].unique().tolist())
	class_to_idx = {name: i for i, name in enumerate(class_names)}
	target_class = "Peltogyne pubescens"

	# 2. Trích xuất embeddings cho cả 2 backbone
	print("\n[Step 2] Trích xuất đặc trưng embeddings...")
	print("Đang trích xuất với EfficientNetV2-M...")
	embs_eff = compute_embeddings_v2(df_filtered, "tf_efficientnetv2_m_in21k", batch_size=BATCH_SIZE, device=device)
	print("Đang trích xuất với Swin-Large...")
	embs_swin = compute_embeddings_v2(df_filtered, "swin_large_patch4_window7_224", batch_size=BATCH_SIZE, device=device)

	# Giải phóng bộ nhớ VRAM thừa
	if device.type == "cuda":
		torch.cuda.empty_cache()

	# 3. Lặp qua các tỉ lệ training và các phương pháp chia dữ liệu
	train_ratios = [0.4, 0.5, 0.6, 0.7]
	candidate_pps = ["PP2", "PP4", "PP5", "PP7", "PP8", "PP9"]
	search_results = []

	for ratio in train_ratios:
		# Tính val_ratio tỉ lệ thuận để test_ratio và val_ratio cân bằng nhau
		p_val_ratio = (1.0 - ratio) / 2.0
		
		for pp in candidate_pps:
			print(f"\n======================================================================")
			print(f" Đang huấn luyện với Ratio={ratio} và PP={pp} cho Peltogyne pubescens...")
			print(f"======================================================================")

			pp_output_dir = output_base / f"ratio_{ratio}" / pp
			pp_output_dir.mkdir(parents=True, exist_ok=True)

			# Chia dữ liệu theo PP đang xét và ratio đang xét
			df_train, df_val, df_test = customized_end_version_split(
				df_filtered, embs_eff, embs_swin,
				peltogyne_pp_key=pp,
				train_ratio=TRAIN_RATIO, # giữ nguyên train_ratio mặc định cho các lớp khác
				val_ratio=VAL_RATIO,     # giữ nguyên val_ratio mặc định cho các lớp khác
				peltogyne_train_ratio=ratio,
				peltogyne_val_ratio=p_val_ratio,
				seed=SEED,
			)

			# Validate split
			validate_split(df_filtered, df_train, df_val, df_test, f"End_Version_with_Peltogyne_Ratio_{ratio}_{pp}")

			# Xây dựng model
			model = build_model(num_classes=len(class_names))
			model = model.to(device)

			checkpoint_loaded = True
			if RUN_MODE == "infer":
				checkpoint_path = None
				if INFER_MODEL_PATH is not None:
					p_path = Path(INFER_MODEL_PATH)
					if p_path.is_file():
						checkpoint_path = p_path
					elif p_path.is_dir():
						# Tìm model cụ thể cho Ratio và PP này trong thư mục con
						candidate_path = p_path / f"ratio_{ratio}" / pp / f"best_model_{MODEL_NAME}.pth"
						if not candidate_path.exists():
							# Dự phòng nếu checkpoint cũ đặt phẳng
							candidate_path = p_path / pp / f"best_model_{MODEL_NAME}.pth"
						if candidate_path.exists():
							checkpoint_path = candidate_path

				if checkpoint_path and checkpoint_path.exists():
					print(f"  [Infer Mode] Loading weights from: {checkpoint_path}")
					model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
				else:
					print(f"  [Infer Mode Warning] Không tìm thấy checkpoint tại {checkpoint_path or INFER_MODEL_PATH}! Bỏ qua cấu hình này.")
					checkpoint_loaded = False
					
			if not checkpoint_loaded:
				continue

			# Xây dựng Dataset & Loader
			cfg_model = resolve_data_config({}, model=model)
			img_size = cfg_model.get("input_size", (3, 224, 224))[-1]
			mean = cfg_model.get("mean", (0.485, 0.456, 0.406))
			std = cfg_model.get("std", (0.229, 0.224, 0.225))
			train_tf, eval_tf = build_transforms(img_size, mean, std)

			test_ds = ImageListDataset(df_test, class_to_idx, transform=eval_tf)
			test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

			if RUN_MODE == "search":
				train_ds = ImageListDataset(df_train, class_to_idx, transform=train_tf)
				val_ds = ImageListDataset(df_val, class_to_idx, transform=eval_tf)
				train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
				val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

				criterion = FocalLoss(gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA)
				optimizer = torch.optim.AdamW(
					filter(lambda p: p.requires_grad, model.parameters()),
					lr=LR,
					weight_decay=WEIGHT_DECAY,
				)
				scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

				# Huấn luyện
				history = train_model(
					model, train_loader, val_loader, optimizer, criterion,
					device, epochs=EPOCHS, patience=PATIENCE, output_dir=pp_output_dir,
					scheduler=scheduler,
				)
				plot_training_curves(history, pp_output_dir)

				# Load checkpoint tốt nhất
				best_path = pp_output_dir / f"best_model_{MODEL_NAME}.pth"
				if best_path.exists():
					model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
			else:
				val_loader = test_loader  # Dự phòng nếu len(df_test) == 0

			# Đánh giá trên tập Test
			y_true, y_pred = collect_predictions(model, val_loader if len(df_test) == 0 else test_loader, device)
			
			# Tính classification report
			target_names = class_names
			report_dict = classification_report(
				y_true, y_pred, target_names=target_names, output_dict=True, zero_division=0
			)
			report_str = classification_report(
				y_true, y_pred, target_names=target_names, digits=4, zero_division=0
			)
			
			print(f"\n[Test Report for Ratio {ratio} - {pp}]:")
			print(report_str)

			# Lưu báo cáo dạng text
			save_report(report_str, pp_output_dir / "report_test.txt")

			# Trích xuất F1-Score của class Peltogyne pubescens
			peltogyne_metrics = report_dict.get(target_class, {"precision": 0.0, "recall": 0.0, "f1-score": 0.0})
			f1 = peltogyne_metrics["f1-score"]
			prec = peltogyne_metrics["precision"]
			rec = peltogyne_metrics["recall"]

			print(f"\n>> {target_class} (Ratio: {ratio}, PP: {pp}) -> Precision: {prec:.4f}, Recall: {rec:.4f}, F1-Score: {f1:.4f}")

			search_results.append({
				"ratio": ratio,
				"pp": pp,
				"precision": prec,
				"recall": rec,
				"f1": f1,
				"report_str": report_str
			})

			# Giải phóng VRAM
			if RUN_MODE == "search":
				del optimizer, scheduler, train_loader, val_loader
			del model, test_loader
			if device.type == "cuda":
				torch.cuda.empty_cache()
			gc.collect()

	if not search_results:
		print("\n[Warning] Không có kết quả tìm kiếm nào được thu thập!")
		return

	# 4. Tìm phương pháp tối ưu nhất (đạt F1-score nhỏ nhất)
	best_pp_info = min(search_results, key=lambda x: x["f1"])

	print("\n" + "="*80)
	print(" BẢNG TỔNG HỢP KẾT QUẢ TÌM KIẾM CHO PELTOGYNE PUBESCENS")
	print("="*80)
	print(f"{'Ratio':<8} | {'PP Key':<10} | {'Precision':<12} | {'Recall':<12} | {'F1-Score':<12}")
	print("-" * 65)
	for res in search_results:
		print(f"{res['ratio']:<8} | {res['pp']:<10} | {res['precision']:<12.4f} | {res['recall']:<12.4f} | {res['f1']:<12.4f}")
	print("="*80)

	print(f"\n🏆 CẤU HÌNH TỐI ƯU NHẤT (F1-score nhỏ nhất cho {target_class}): Ratio={best_pp_info['ratio']}, PP={best_pp_info['pp']}")
	print(f" -> Precision: {best_pp_info['precision']:.4f}")
	print(f" -> Recall: {best_pp_info['recall']:.4f}")
	print(f" -> F1-Score: {best_pp_info['f1']:.4f}")

	# Lưu báo cáo tổng kết ra file
	summary_path = output_base / "peltogyne_search_summary.txt"
	with open(summary_path, "w", encoding="utf-8") as f:
		f.write("="*80 + "\n")
		f.write(" BẢNG TỔNG HỢP KẾT QUẢ TÌM KIẾM CHO PELTOGYNE PUBESCENS\n")
		f.write("="*80 + "\n")
		f.write(f"{'Ratio':<8} | {'PP Key':<10} | {'Precision':<12} | {'Recall':<12} | {'F1-Score':<12}\n")
		f.write("-" * 65 + "\n")
		for res in search_results:
			f.write(f"{res['ratio']:<8} | {res['pp']:<10} | {res['precision']:<12.4f} | {res['recall']:<12.4f} | {res['f1']:<12.4f}\n")
		f.write("="*80 + "\n")
		f.write(f"\n🏆 CẤU HÌNH TỐI ƯU NHẤT (F1-score nhỏ nhất): Ratio={best_pp_info['ratio']}, PP={best_pp_info['pp']}\n")
		f.write(f" -> Precision: {best_pp_info['precision']:.4f}\n")
		f.write(f" -> Recall: {best_pp_info['recall']:.4f}\n")
		f.write(f" -> F1-Score: {best_pp_info['f1']:.4f}\n")

	print(f"\nĐã ghi báo cáo tổng hợp tại: {summary_path}")


if __name__ == "__main__":
	main()
