"""
datasail_benchmark/evaluator.py
===============================
Pipeline quản lý thực thi 9 thuật toán chia, Mục 5 Meta-Selector k^N, tính toán 12 chỉ số đo lường và xuất báo cáo.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import OUTPUT_DIR, BENCHMARK_SEEDS, EXCLUDED_CLASSES
from .metrics import (
	compute_datasail_loss,
	compute_inter_split_cosine_sim,
	compute_intra_split_cosine_sim,
	compute_specimen_leakage_risk,
	compute_pseudoreplication_index,
	compute_silhouette_separation,
	compute_nearest_neighbor_stats,
	compute_wasserstein_divergence,
	compute_knn_metrics,
	compute_leakage_inflation_deltas,
	compute_statistical_significance,
)
from .solvers import ALL_SOLVERS


def run_benchmark_pipeline(
	df_filtered: pd.DataFrame,
	embeddings: np.ndarray,
	class_to_idx: Dict[str, int],
	output_dir: Path = OUTPUT_DIR,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], str]:
	"""
	Thực thi pipeline benchmark toàn diện trên 9 thuật toán chia x 5 seeds + Meta-Selector Mục 5.
	"""
	output_dir.mkdir(parents=True, exist_ok=True)
	path_to_idx = {path: i for i, path in enumerate(df_filtered["path"])}
	class_names = sorted(df_filtered["label"].unique().tolist())

	print(f"\n=========================================================================")
	print(f" KHỞI ĐỘNG BENCHMARK DATA LEAKAGE VÀ TỐI ƯU HÓA DATASAIL k^N (18 SPECIES)")
	print(f"=========================================================================\n")

	raw_benchmark_records = []

	# Duyệt qua 9 phương pháp chia x 5 seeds
	for proto_name, solver_fn in ALL_SOLVERS.items():
		print(f"-> Đang thực thi phương pháp: {proto_name}...")
		for seed in BENCHMARK_SEEDS:
			try:
				df_tr, df_va, df_te = solver_fn(df_filtered, embeddings, seed=seed)

				tr_idx = [path_to_idx[p] for p in df_tr["path"]]
				va_idx = [path_to_idx[p] for p in df_va["path"]]
				te_idx = [path_to_idx[p] for p in df_te["path"]]

				# Tính toán 12 chỉ số đo lường
				datasail_loss = compute_datasail_loss(embeddings, tr_idx, va_idx, te_idx)
				inter_sim = compute_inter_split_cosine_sim(embeddings, tr_idx, va_idx, te_idx)
				intra_sim = compute_intra_split_cosine_sim(embeddings, tr_idx, va_idx, te_idx)
				slr = compute_specimen_leakage_risk(df_tr, df_va, df_te)
				pri = compute_pseudoreplication_index(df_tr, df_te)
				sil_score = compute_silhouette_separation(embeddings, tr_idx, va_idx, te_idx)
				nn_stats = compute_nearest_neighbor_stats(embeddings, tr_idx, te_idx)
				w1_dist = compute_wasserstein_divergence(df_filtered, df_tr, df_va, df_te)
				knn_res = compute_knn_metrics(embeddings, df_tr, df_te, class_to_idx)

				rec = {
					"protocol": proto_name,
					"seed": seed,
					"datasail_loss": datasail_loss,
					"inter_cosine_sim": inter_sim,
					"intra_cosine_sim": intra_sim,
					"slr_percent": slr,
					"pri_percent": pri,
					"silhouette_score": sil_score,
					"nn_sim_mean": nn_stats["nn_sim_mean"],
					"nn_sim_max": nn_stats["nn_sim_max"],
					"nn_sim_p90": nn_stats["nn_sim_p90"],
					"nn_sim_std": nn_stats["nn_sim_std"],
					"wasserstein_dist": w1_dist,
					"knn_accuracy": knn_res["knn_accuracy"],
					"knn_f1_macro": knn_res["knn_f1_macro"],
					"knn_f1_weighted": knn_res["knn_f1_weighted"],
					"num_train": len(df_tr),
					"num_val": len(df_va),
					"num_test": len(df_te),
				}
				raw_benchmark_records.append(rec)
			except Exception as e:
				print(f"  [Lỗi] {proto_name} (Seed {seed}): {str(e)}")

	# =========================================================================
	# MỤC 5: Tối ưu hóa Tổ hợp k^N (Meta-Selector over k^N search space)
	# =========================================================================
	print(f"\n-> Đang thực thi Mục 5: Tối ưu hóa Tổ hợp k^N (Meta-Selector cho {len(class_names)} loài gỗ)...")
	
	optimal_classwise_pp = {}
	meta_tr_indices, meta_va_indices, meta_te_indices = [], [], []
	meta_tr_dfs, meta_va_dfs, meta_te_dfs = [], [], []

	for label in class_names:
		class_mask = df_filtered["label"] == label
		sub_df = df_filtered[class_mask].copy()
		sub_indices = sub_df.index.tolist()
		sub_embs = embeddings[sub_indices]
		sub_path_to_idx = {p: i for i, p in enumerate(sub_df["path"])}

		best_proto = None
		best_class_loss = float("inf")
		best_splits = None

		# Thử tất cả 9 phương pháp ứng viên cho loài gỗ này
		for proto_name, solver_fn in ALL_SOLVERS.items():
			try:
				tr_sub, va_sub, te_sub = solver_fn(sub_df, sub_embs, seed=42)
				tr_i = [sub_path_to_idx[p] for p in tr_sub["path"]]
				va_i = [sub_path_to_idx[p] for p in va_sub["path"]]
				te_i = [sub_path_to_idx[p] for p in te_sub["path"]]

				loss_c = compute_datasail_loss(sub_embs, tr_i, va_i, te_i)
				if loss_c < best_class_loss:
					best_class_loss = loss_c
					best_proto = proto_name
					best_splits = (tr_sub, va_sub, te_sub)
			except Exception:
				pass

		if best_proto is not None and best_splits is not None:
			optimal_classwise_pp[label] = best_proto
			tr_s, va_s, te_s = best_splits
			meta_tr_dfs.append(tr_s)
			meta_va_dfs.append(va_s)
			meta_te_dfs.append(te_s)

	df_meta_tr = pd.concat(meta_tr_dfs).reset_index(drop=True)
	df_meta_va = pd.concat(meta_va_dfs).reset_index(drop=True)
	df_meta_te = pd.concat(meta_te_dfs).reset_index(drop=True)

	meta_tr_idx = [path_to_idx[p] for p in df_meta_tr["path"]]
	meta_va_idx = [path_to_idx[p] for p in df_meta_va["path"]]
	meta_te_idx = [path_to_idx[p] for p in df_meta_te["path"]]

	# Tính toán 12 chỉ số cho Meta-Selector
	meta_datasail_loss = compute_datasail_loss(embeddings, meta_tr_idx, meta_va_idx, meta_te_idx)
	meta_inter_sim = compute_inter_split_cosine_sim(embeddings, meta_tr_idx, meta_va_idx, meta_te_idx)
	meta_intra_sim = compute_intra_split_cosine_sim(embeddings, meta_tr_idx, meta_va_idx, meta_te_idx)
	meta_slr = compute_specimen_leakage_risk(df_meta_tr, df_meta_va, df_meta_te)
	meta_pri = compute_pseudoreplication_index(df_meta_tr, df_meta_te)
	meta_sil = compute_silhouette_separation(embeddings, meta_tr_idx, meta_va_idx, meta_te_idx)
	meta_nn = compute_nearest_neighbor_stats(embeddings, meta_tr_idx, meta_te_idx)
	meta_w1 = compute_wasserstein_divergence(df_filtered, df_meta_tr, df_meta_va, df_meta_te)
	meta_knn = compute_knn_metrics(embeddings, df_meta_tr, df_meta_te, class_to_idx)

	meta_rec = {
		"protocol": "PP10_DataSAIL_Meta_Selector_Optimal_k^N",
		"seed": 42,
		"datasail_loss": meta_datasail_loss,
		"inter_cosine_sim": meta_inter_sim,
		"intra_cosine_sim": meta_intra_sim,
		"slr_percent": meta_slr,
		"pri_percent": meta_pri,
		"silhouette_score": meta_sil,
		"nn_sim_mean": meta_nn["nn_sim_mean"],
		"nn_sim_max": meta_nn["nn_sim_max"],
		"nn_sim_p90": meta_nn["nn_sim_p90"],
		"nn_sim_std": meta_nn["nn_sim_std"],
		"wasserstein_dist": meta_w1,
		"knn_accuracy": meta_knn["knn_accuracy"],
		"knn_f1_macro": meta_knn["knn_f1_macro"],
		"knn_f1_weighted": meta_knn["knn_f1_weighted"],
		"num_train": len(df_meta_tr),
		"num_val": len(df_meta_va),
		"num_test": len(df_meta_te),
	}
	raw_benchmark_records.append(meta_rec)

	# Lưu file dữ liệu thô CSV & JSON
	df_all_results = pd.DataFrame(raw_benchmark_records)
	df_all_results.to_csv(output_dir / "all_splits_results.csv", index=False)

	with open(output_dir / "optimal_k_n_classwise_config.json", "w", encoding="utf-8") as f:
		json.dump(optimal_classwise_pp, f, indent=2, ensure_ascii=False)

	# =========================================================================
	# Tổng hợp Bảng Thống kê Học thuật (Mean +- Std & Stat Tests)
	# =========================================================================
	lines = []
	lines.append("=" * 135)
	lines.append(" BẢNG TỔNG HỢP KẾT QUẢ BENCHMARK DATA LEAKAGE VÀ TỐI ƯU HÓA DATASAIL (MEAN ± STD QUA 5 SEEDS)")
	lines.append("=" * 135)

	header = f"{'Protocol':<32} {'KNN Test Acc':<16} {'F1-Macro':<16} {'DataSAIL Loss':<18} " \
	         f"{'S_inter':<14} {'SLR (%)':<10} {'NN_Sim Mean':<14} {'p-val vs R-Split':<16}"
	lines.append(header)
	lines.append("-" * 135)

	random_acc_scores = [r["knn_accuracy"] for r in raw_benchmark_records if r["protocol"] == "PP1_Image_Random"]

	summary_rows = {}
	all_protocols = sorted(list(set(r["protocol"] for r in raw_benchmark_records)))

	for proto in all_protocols:
		subset = [r for r in raw_benchmark_records if r["protocol"] == proto]
		acc_vals = [r["knn_accuracy"] for r in subset]
		f1_vals = [r["knn_f1_macro"] for r in subset]
		loss_vals = [r["datasail_loss"] for r in subset]
		s_inter_vals = [r["inter_cosine_sim"] for r in subset]
		slr_vals = [r["slr_percent"] for r in subset]
		nn_vals = [r["nn_sim_mean"] for r in subset]

		acc_str = f"{np.mean(acc_vals):.4f} ± {np.std(acc_vals):.4f}" if len(acc_vals) > 1 else f"{acc_vals[0]:.4f}"
		f1_str = f"{np.mean(f1_vals):.4f} ± {np.std(f1_vals):.4f}" if len(f1_vals) > 1 else f"{f1_vals[0]:.4f}"
		loss_str = f"{np.mean(loss_vals):.1f} ± {np.std(loss_vals):.1f}" if len(loss_vals) > 1 else f"{loss_vals[0]:.1f}"
		s_inter_str = f"{np.mean(s_inter_vals):.4f}"
		slr_str = f"{np.mean(slr_vals):.1f}%"
		nn_str = f"{np.mean(nn_vals):.4f}"

		# Tính p-value kiểm định so với PP1_Image_Random
		if proto != "PP1_Image_Random" and len(acc_vals) > 1 and len(random_acc_scores) > 1:
			stat_res = compute_statistical_significance(random_acc_scores, acc_vals)
			pval_str = f"{stat_res['p_value']:.4e}"
		else:
			pval_str = "---"

		lines.append(f"{proto:<32} {acc_str:<16} {f1_str:<16} {loss_str:<18} {s_inter_str:<14} {slr_str:<10} {nn_str:<14} {pval_str:<16}")

		summary_rows[proto] = {
			"acc_mean": float(np.mean(acc_vals)),
			"acc_std": float(np.std(acc_vals)),
			"f1_mean": float(np.mean(f1_vals)),
			"f1_std": float(np.std(f1_vals)),
			"datasail_loss_mean": float(np.mean(loss_vals)),
			"inter_sim_mean": float(np.mean(s_inter_vals)),
			"slr_mean": float(np.mean(slr_vals)),
			"nn_sim_mean": float(np.mean(nn_vals)),
		}

	lines.append("=" * 135)
	summary_txt = "\n".join(lines)
	print("\n" + summary_txt)

	with open(output_dir / "summary_academic_table.txt", "w", encoding="utf-8") as f:
		f.write(summary_txt)

	return raw_benchmark_records, summary_rows, summary_txt
