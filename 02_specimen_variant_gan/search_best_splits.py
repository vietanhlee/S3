"""
search_best_splits.py
======================
Script tìm kiếm phương pháp chia (PP) tối ưu nhất trong 9 PP cho từng lớp gỗ (19 lớp).
Tiêu chí tối ưu: Đạt giá trị trung bình Nearest-Neighbor Similarity per Test Sample nhỏ nhất
(tức là giảm thiểu tối đa nguy cơ rò rỉ dữ liệu / tạo ra tập Test thử thách nhất).

Cách chạy:
  python search_best_splits.py
"""

import os
import gc
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from sklearn.metrics.pairwise import cosine_similarity

# Import helpers từ package utils
from utils import (
	set_seed,
	get_device,
	collect_image_samples,
	build_dataframe,
)

# Import các thuật toán trích xuất đặc trưng
from train_final import compute_embeddings_v2

# Import toàn bộ 9 PP chia dữ liệu
from split_methods import (
	mahalanobis_fixed_split,
	mahalanobis_iterative_split,
	group_based_split,
	hierarchical_clustering_split,
	cosine_graph_split,
	stratified_random_split,
	adversarial_validation_split,
	stratified_group_kfold_split,
	agglom_stratified_split,
)

# ===== CẤU HÌNH =====
ROOT_DIR = r"/kaggle/input/datasets/b23dckh002lvitanh/s3-origin/S3"
OUTPUT_DIR = "outputs_split_search"
TRAIN_RATIO = 0.60
VAL_RATIO = 0.20
BATCH_SIZE = 128
SEED = 42
COSINE_THRESHOLD = 0.92
# =====================

# Đăng ký 9 phương pháp chia dữ liệu
PP_METHODS = {
	"PP1_Mahalanobis_Fixed": mahalanobis_fixed_split,
	"PP2_Mahalanobis_Iterative": mahalanobis_iterative_split,
	"PP3_Group_Based": group_based_split,
	"PP4_Hierarchical_Clustering": hierarchical_clustering_split,
	"PP5_Cosine_Graph": cosine_graph_split,
	"PP6_Stratified_Random": stratified_random_split,
	"PP7_Adversarial_Validation": adversarial_validation_split,
	"PP8_StratifiedGroupKFold": stratified_group_kfold_split,
	"PP9_Agglom_Stratified": agglom_stratified_split,
}


def main():
	set_seed(SEED)
	device = get_device()
	print(f"Khởi động script tìm kiếm phương pháp chia tối ưu trên device: {device}")

	output_base = Path(OUTPUT_DIR)
	output_base.mkdir(parents=True, exist_ok=True)

	# 1. Thu thập ảnh
	print("\n[Step 1] Thu thập dữ liệu ảnh...")
	samples = collect_image_samples(ROOT_DIR)
	if not samples:
		raise ValueError(f"Không tìm thấy ảnh tại: {ROOT_DIR}")

	df = build_dataframe(samples)
	df_filtered = df[~df["label"].isin(["Pterocarpus sp", "Peltogyne pubescens"])].reset_index(drop=True)
	print(f"Tổng số ảnh sau khi lọc bỏ Pterocarpus sp và Peltogyne pubescens: {len(df_filtered)}")

	class_names = sorted(df_filtered["label"].unique().tolist())
	print(f"Số lượng lớp gỗ cần duyệt: {len(class_names)}")

	# 2. Trích xuất embeddings cho cả 2 mạng
	print("\n[Step 2] Trích xuất đặc trưng embeddings để phục vụ chia và đánh giá...")
	print("  -> Trích xuất với EfficientNetV2-M (dùng cho thuật toán chia)...")
	embs_eff = compute_embeddings_v2(df_filtered, "tf_efficientnetv2_m_in21k", batch_size=BATCH_SIZE, device=device)
	print("  -> Trích xuất với Swin-Large (dùng để tính toán Nearest-Neighbor Similarity)...")
	embs_swin = compute_embeddings_v2(df_filtered, "swin_large_patch4_window7_224", batch_size=BATCH_SIZE, device=device)

	# Giải phóng VRAM
	if device.type == "cuda":
		torch.cuda.empty_cache()
	gc.collect()

	# Map từ path ảnh sang index trong array embeddings để tra cứu nhanh
	path_to_idx = {path: i for i, path in enumerate(df_filtered["path"])}

	# 3. Quét qua 19 lớp và 9 phương pháp (171 tổ hợp)
	print("\n[Step 3] Bắt đầu tìm kiếm 171 tổ hợp (19 classes * 9 PPs)...")
	search_records = []

	for label in tqdm(class_names, desc="Duyệt từng lớp gỗ"):
		# Lọc dataframe và embeddings riêng cho lớp gỗ này
		class_mask = df_filtered["label"] == label
		sub_df = df_filtered[class_mask].copy()
		sub_df_reset = sub_df.reset_index(drop=True)
		
		# Lấy các chỉ số tương ứng của lớp gỗ này trong toàn bộ dataset
		class_indices = sub_df.index.tolist()
		sub_emb_eff = embs_eff[class_indices]
		
		# Duyệt qua 9 phương pháp chia
		for pp_name, split_fn in PP_METHODS.items():
			try:
				# Chia dữ liệu cho lớp gỗ hiện tại
				if pp_name == "PP5_Cosine_Graph":
					tr_df, val_df, te_df = split_fn(
						sub_df_reset, sub_emb_eff,
						train_ratio=TRAIN_RATIO,
						val_ratio=VAL_RATIO,
						seed=SEED,
						cosine_threshold=COSINE_THRESHOLD,
					)
				else:
					tr_df, val_df, te_df = split_fn(
						sub_df_reset, sub_emb_eff,
						train_ratio=TRAIN_RATIO,
						val_ratio=VAL_RATIO,
						seed=SEED,
					)
				
				# Kiểm tra nếu tập Train hoặc Test bị trống
				if len(tr_df) == 0 or len(te_df) == 0:
					mean_max_sim = 1.0  # Phạt nặng nếu split bị rỗng
				else:
					# Tra cứu embeddings Swin-Large của Train và Test để tính similarity
					tr_indices = [path_to_idx[p] for p in tr_df["path"]]
					te_indices = [path_to_idx[p] for p in te_df["path"]]
					
					train_feats = embs_swin[tr_indices]
					test_feats = embs_swin[te_indices]
					
					# Tính Cosine Similarity matrix (n_test, n_train)
					sim_matrix = cosine_similarity(test_feats, train_feats)
					# Lấy Max similarity với Train cho từng ảnh Test
					max_sims = sim_matrix.max(axis=1)
					mean_max_sim = float(np.mean(max_sims))
					
			except Exception as e:
				# Gán giá trị phạt tối đa nếu có lỗi xảy ra
				mean_max_sim = 1.0
			
			search_records.append({
				"class_name": label,
				"pp_name": pp_name,
				"nearest_neighbor_similarity": mean_max_sim,
				"train_samples": len(tr_df) if 'tr_df' in locals() else 0,
				"test_samples": len(te_df) if 'te_df' in locals() else 0,
			})

	# 4. Phân tích tìm phương pháp tốt nhất cho từng lớp gỗ (Phương hướng 1: Cục bộ / Class-wise)
	df_results = pd.DataFrame(search_records)
	
	best_splits_classwise = {}
	summary_lines = []
	summary_lines.append("=" * 90)
	summary_lines.append(" PHƯƠNG HƯỚNG 1: TỐI ƯU HÓA CỤC BỘ (CLASS-WISE OPTIMIZATION)")
	summary_lines.append(" (Mỗi class chọn phương pháp chia tối ưu nhất có NN-Similarity nhỏ nhất)")
	summary_lines.append("=" * 90)
	summary_lines.append(f"{'Class Name':<30} | {'Best PP':<30} | {'Min NN-Similarity':<20}")
	summary_lines.append("-" * 90)

	best_split_config_classwise = {}
	for label in class_names:
		class_results = df_results[df_results["class_name"] == label]
		best_row = class_results.loc[class_results["nearest_neighbor_similarity"].idxmin()]
		best_pp = best_row["pp_name"]
		min_sim = best_row["nearest_neighbor_similarity"]
		
		short_pp = best_pp.split("_")[0]
		best_split_config_classwise[label] = (short_pp, "test", "swin")
		summary_lines.append(f"{label:<30} | {best_pp:<30} | {min_sim:<20.4f}")

	summary_lines.append("=" * 90)
	summary_lines.append("\n")

	# 5. Phân tích tìm phương pháp tốt nhất chung cho toàn bộ (Phương hướng 2: Toàn cục / Global Unified)
	summary_lines.append("=" * 90)
	summary_lines.append(" PHƯƠNG HƯỚNG 2: TỐI ƯU HÓA TOÀN CỤC (GLOBAL UNIFIED OPTIMIZATION)")
	summary_lines.append(" (Chọn 1 phương pháp chia duy nhất cho toàn bộ các class để đạt mean NN-Similarity nhỏ nhất)")
	summary_lines.append("=" * 90)
	summary_lines.append(f"{'PP Method':<30} | {'Mean NN-Similarity Across All Classes':<45}")
	summary_lines.append("-" * 90)

	global_pp_results = []
	for pp_name in PP_METHODS.keys():
		pp_rows = df_results[df_results["pp_name"] == pp_name]
		mean_similarity = pp_rows["nearest_neighbor_similarity"].mean()
		global_pp_results.append({
			"pp_name": pp_name,
			"mean_similarity": mean_similarity
		})
		summary_lines.append(f"{pp_name:<30} | {mean_similarity:<45.4f}")

	df_global_pp = pd.DataFrame(global_pp_results)
	best_global_row = df_global_pp.loc[df_global_pp["mean_similarity"].idxmin()]
	best_global_pp = best_global_row["pp_name"]
	min_global_similarity = best_global_row["mean_similarity"]

	summary_lines.append("-" * 90)
	summary_lines.append(f"-> PHƯƠNG PHÁP TOÀN CỤC TỐT NHẤT: {best_global_pp} (Mean Similarity: {min_global_similarity:.4f})")
	summary_lines.append("=" * 90)

	best_split_config_global = {}
	short_global_pp = best_global_pp.split("_")[0]
	for label in class_names:
		best_split_config_global[label] = (short_global_pp, "test", "swin")

	summary_str = "\n".join(summary_lines)
	print("\n" + summary_str)

	# Lưu báo cáo dạng text
	with open(output_base / "best_splits_search_report.txt", "w", encoding="utf-8") as f:
		f.write(summary_str)
	
	# Lưu cấu hình Class-wise dạng Dict rút gọn
	with open(output_base / "optimal_split_config_classwise.json", "w", encoding="utf-8") as f:
		f.write("{\n")
		for k, v in best_split_config_classwise.items():
			f.write(f'\t\t"{k}": ("{v[0]}", "{v[1]}", "{v[2]}"),\n')
		f.write("\t}\n")

	# Lưu cấu hình Global dạng Dict rút gọn
	with open(output_base / "optimal_split_config_global.json", "w", encoding="utf-8") as f:
		f.write("{\n")
		for k, v in best_split_config_global.items():
			f.write(f'\t\t"{k}": ("{v[0]}", "{v[1]}", "{v[2]}"),\n')
		f.write("\t}\n")
		
	# Lưu kết quả toàn bộ 171 tổ hợp để vẽ biểu đồ hoặc phân tích thêm
	df_results.to_csv(output_base / "all_combinations_results.csv", index=False)
	
	print(f"\n[Hoàn tất] Kết quả tìm kiếm đã được ghi nhận tại: {output_base}/")
	print(f"  - best_splits_search_report.txt -> Báo cáo tổng hợp chi tiết cả 2 phương hướng.")
	print(f"  - optimal_split_config_classwise.json -> File cấu hình tối ưu cục bộ (class-wise).")
	print(f"  - optimal_split_config_global.json    -> File cấu hình tối ưu toàn cục (global).")
	print(f"  - all_combinations_results.csv   -> Bảng dữ liệu thô của cả 171 tổ hợp.")


if __name__ == "__main__":
	main()
