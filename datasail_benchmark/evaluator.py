"""
datasail_benchmark/evaluator.py
===============================
Pipeline quản lý thực thi các thuật toán chia từ split_methods.py + DataSAIL,
Mục 5A: Class-wise DataSAIL Loss Selector (PP12),
Mục 5B: Multi-Objective Simulated Annealing Selector (PP13),
tính toán các chỉ số đo lường học thuật và xuất báo cáo.
"""

import json
import random
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
	compute_class_coverage_rate,
	compute_silhouette_separation,
	compute_maximum_mean_discrepancy,
	compute_nearest_neighbor_mean_sim,
	compute_wasserstein_divergence,
	compute_knn_metrics,
	compute_statistical_significance,
)
from .solvers import ALL_SOLVERS, SPLIT_METHODS_WRAPPED


def run_classwise_datasail_loss_selector(
	df_filtered: pd.DataFrame,
	embeddings: np.ndarray,
	path_to_idx: Dict[str, int],
) -> Tuple[Dict[str, str], Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
	"""
	Mục 5A: Tối ưu hóa đơn mục tiêu Per-Class DataSAIL Loss:
	Duyệt qua từng loài gỗ c, chọn phương pháp solver cho ra DataSAIL Loss L_c nhỏ nhất cho loài c
	chỉ quét trên không gian các thuật toán từ split_methods.py (SPLIT_METHODS_WRAPPED).
	"""
	class_names = sorted(df_filtered["label"].unique().tolist())
	optimal_classwise_pp = {}
	meta_tr_dfs, meta_va_dfs, meta_te_dfs = [], [], []

	# Chỉ quét không gian các thuật toán từ split_methods.py
	candidate_solvers = SPLIT_METHODS_WRAPPED

	for label in class_names:
		class_mask = df_filtered["label"] == label
		sub_df = df_filtered[class_mask].copy()
		sub_indices = sub_df.index.tolist()
		sub_embs = embeddings[sub_indices]
		sub_path_to_idx = {p: i for i, p in enumerate(sub_df["path"])}

		best_proto = None
		best_class_loss = float("inf")
		best_splits = None

		for proto_name, solver_fn in candidate_solvers.items():
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
			meta_tr_dfs.append(best_splits[0])
			meta_va_dfs.append(best_splits[1])
			meta_te_dfs.append(best_splits[2])

	df_meta_tr = pd.concat(meta_tr_dfs).reset_index(drop=True)
	df_meta_va = pd.concat(meta_va_dfs).reset_index(drop=True)
	df_meta_te = pd.concat(meta_te_dfs).reset_index(drop=True)

	return optimal_classwise_pp, (df_meta_tr, df_meta_va, df_meta_te)


def optimize_multi_objective_datasail_sa(
	df_filtered: pd.DataFrame,
	embeddings: np.ndarray,
	class_to_idx: Dict[str, int],
	path_to_idx: Dict[str, int],
	w_datasail: float = 1.0,
	w_mmd: float = 0.5,
	w_hardest_f1: float = 0.5,
	n_iters: int = 10000,
	seed: int = 42,
) -> Tuple[Dict[str, str], Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
	"""
	Mục 5B: Thuật toán Luyện Kim (Simulated Annealing - SA) Tối Ưu Hóa Đa Mục Tiêu:
	Fitness(m) = w1 * L_DataSAIL / 1000 - w2 * MMD * 10 - w3 * Hardest_F1 * 10
	Chỉ quét chọn trong không gian các thuật toán phân tách từ split_methods.py (SPLIT_METHODS_WRAPPED).
	"""
	rng = random.Random(seed)
	class_names = sorted(df_filtered["label"].unique().tolist())

	# Chỉ quét không gian các thuật toán từ split_methods.py
	candidate_solvers = SPLIT_METHODS_WRAPPED

	class_splits_cache: Dict[str, Dict[str, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]] = {}
	for label in class_names:
		sub_df = df_filtered[df_filtered["label"] == label].copy()
		sub_indices = sub_df.index.tolist()
		sub_embs = embeddings[sub_indices]

		class_splits_cache[label] = {}
		for proto_name, solver_fn in candidate_solvers.items():
			try:
				tr_s, va_s, te_s = solver_fn(sub_df, sub_embs, seed=seed)
				class_splits_cache[label][proto_name] = (tr_s, va_s, te_s)
			except Exception:
				pass

	current_config = {label: rng.choice(list(class_splits_cache[label].keys())) for label in class_names}

	def assemble_and_evaluate(config: Dict[str, str]) -> Tuple[float, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
		tr_dfs, va_dfs, te_dfs = [], [], []
		for lbl in class_names:
			p_name = config[lbl]
			tr_s, va_s, te_s = class_splits_cache[lbl][p_name]
			tr_dfs.append(tr_s)
			va_dfs.append(va_s)
			te_dfs.append(te_s)

		df_tr = pd.concat(tr_dfs).reset_index(drop=True)
		df_va = pd.concat(va_dfs).reset_index(drop=True)
		df_te = pd.concat(te_dfs).reset_index(drop=True)

		tr_i = [path_to_idx[p] for p in df_tr["path"]]
		va_i = [path_to_idx[p] for p in df_va["path"]]
		te_i = [path_to_idx[p] for p in df_te["path"]]

		l_datasail = compute_datasail_loss(embeddings, tr_i, va_i, te_i)
		mmd_val = compute_maximum_mean_discrepancy(embeddings, tr_i, te_i)
		knn_res = compute_knn_metrics(embeddings, df_tr, df_te, class_to_idx, path_to_idx)
		hardest_f1 = knn_res["hardest_class_f1"]

		fitness = w_datasail * (l_datasail / 1000.0) - w_mmd * (mmd_val * 10.0) - w_hardest_f1 * (hardest_f1 * 10.0)
		return float(fitness), (df_tr, df_va, df_te)

	curr_fitness, curr_splits = assemble_and_evaluate(current_config)
	best_config = current_config.copy()
	best_fitness = curr_fitness
	best_splits = curr_splits

	temp = 30.0
	cooling_rate = 0.999



	for _ in range(n_iters):
		target_label = rng.choice(class_names)
		available_protos = list(class_splits_cache[target_label].keys())
		new_proto = rng.choice(available_protos)

		cand_config = current_config.copy()
		cand_config[target_label] = new_proto

		cand_fitness, cand_splits = assemble_and_evaluate(cand_config)
		delta = cand_fitness - curr_fitness

		if delta < 0 or rng.random() < np.exp(-delta / max(1e-5, temp)):
			current_config = cand_config
			curr_fitness = cand_fitness
			curr_splits = cand_splits

			if curr_fitness < best_fitness:
				best_fitness = curr_fitness
				best_config = current_config.copy()
				best_splits = curr_splits

		temp *= cooling_rate

	return best_config, best_splits


def run_benchmark_pipeline(
	df_filtered: pd.DataFrame,
	embeddings: np.ndarray,
	class_to_idx: Dict[str, int],
	output_dir: Path = OUTPUT_DIR,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], str]:
	"""
	Thực thi pipeline benchmark toàn diện trên tất cả các thuật toán chia x 5 seeds
	+ Mục 5A: Per-Class DataSAIL Loss Selector (PP12) - Quét các thuật toán từ split_methods.py
	+ Mục 5B: Multi-Objective SA Selector (PP13) - Quét các thuật toán từ split_methods.py.
	"""
	output_dir.mkdir(parents=True, exist_ok=True)
	path_to_idx = {path: i for i, path in enumerate(df_filtered["path"])}
	class_names = sorted(df_filtered["label"].unique().tolist())

	print(f"\n=========================================================================")
	print(f" KHỞI ĐỘNG BENCHMARK DATA LEAKAGE VÀ TỐI ƯU HÓA DATASAIL k^N (18 SPECIES)")
	print(f"=========================================================================\n")

	raw_benchmark_records = []

	# Duyệt tất cả các solvers x 5 seeds
	for proto_name, solver_fn in ALL_SOLVERS.items():
		print(f"-> Đang thực thi phương pháp: {proto_name}...")
		for seed in BENCHMARK_SEEDS:
			try:
				df_tr, df_va, df_te = solver_fn(df_filtered, embeddings, seed=seed)

				tr_idx = [path_to_idx[p] for p in df_tr["path"]]
				va_idx = [path_to_idx[p] for p in df_va["path"]]
				te_idx = [path_to_idx[p] for p in df_te["path"]]

				datasail_loss = compute_datasail_loss(embeddings, tr_idx, va_idx, te_idx)
				inter_sim = compute_inter_split_cosine_sim(embeddings, tr_idx, va_idx, te_idx)
				intra_sim = compute_intra_split_cosine_sim(embeddings, tr_idx, va_idx, te_idx)
				slr = compute_specimen_leakage_risk(df_tr, df_va, df_te)
				pri = compute_pseudoreplication_index(df_tr, df_te)
				ccr = compute_class_coverage_rate(df_filtered, df_tr, df_va, df_te)
				sil_score = compute_silhouette_separation(embeddings, tr_idx, va_idx, te_idx)
				mmd_dist = compute_maximum_mean_discrepancy(embeddings, tr_idx, te_idx)
				nn_mean = compute_nearest_neighbor_mean_sim(embeddings, tr_idx, te_idx)
				w1_dist = compute_wasserstein_divergence(df_filtered, df_tr, df_va, df_te)
				knn_res = compute_knn_metrics(embeddings, df_tr, df_te, class_to_idx, path_to_idx)

				rec = {
					"protocol": proto_name,
					"seed": seed,
					"datasail_loss": datasail_loss,
					"inter_cosine_sim": inter_sim,
					"intra_cosine_sim": intra_sim,
					"slr_percent": slr,
					"pri_percent": pri,
					"ccr_percent": ccr,
					"silhouette_score": sil_score,
					"mmd_distance": mmd_dist,
					"nn_sim_mean": nn_mean,
					"wasserstein_dist": w1_dist,
					"knn_accuracy": knn_res["knn_accuracy"],
					"knn_top3_accuracy": knn_res["knn_top3_accuracy"],
					"knn_balanced_accuracy": knn_res["knn_balanced_accuracy"],
					"knn_f1_macro": knn_res["knn_f1_macro"],
					"hardest_class_f1": knn_res["hardest_class_f1"],
					"num_train": len(df_tr),
					"num_val": len(df_va),
					"num_test": len(df_te),
				}
				raw_benchmark_records.append(rec)
			except Exception as e:
				print(f"  [Lỗi] {proto_name} (Seed {seed}): {str(e)}")

	# =========================================================================
	# MỤC 5A: Tối ưu hóa Per-Class DataSAIL Loss Selector (PP12) - split_methods.py ONLY
	# =========================================================================
	print(f"\n-> Đang thực thi Mục 5A: Classwise DataSAIL Loss Selector (PP12 - split_methods.py ONLY)...")
	opt_classwise_config, (df_meta_tr_a, df_meta_va_a, df_meta_te_a) = run_classwise_datasail_loss_selector(
		df_filtered, embeddings, path_to_idx
	)

	tr_i_a = [path_to_idx[p] for p in df_meta_tr_a["path"]]
	va_i_a = [path_to_idx[p] for p in df_meta_va_a["path"]]
	te_i_a = [path_to_idx[p] for p in df_meta_te_a["path"]]
	knn_a = compute_knn_metrics(embeddings, df_meta_tr_a, df_meta_te_a, class_to_idx, path_to_idx)

	rec_5a = {
		"protocol": "PP12_DataSAIL_Meta_Selector_Classwise_Loss",
		"seed": 42,
		"datasail_loss": compute_datasail_loss(embeddings, tr_i_a, va_i_a, te_i_a),
		"inter_cosine_sim": compute_inter_split_cosine_sim(embeddings, tr_i_a, va_i_a, te_i_a),
		"intra_cosine_sim": compute_intra_split_cosine_sim(embeddings, tr_i_a, va_i_a, te_i_a),
		"slr_percent": compute_specimen_leakage_risk(df_meta_tr_a, df_meta_va_a, df_meta_te_a),
		"pri_percent": compute_pseudoreplication_index(df_meta_tr_a, df_meta_te_a),
		"ccr_percent": compute_class_coverage_rate(df_filtered, df_meta_tr_a, df_meta_va_a, df_meta_te_a),
		"silhouette_score": compute_silhouette_separation(embeddings, tr_i_a, va_i_a, te_i_a),
		"mmd_distance": compute_maximum_mean_discrepancy(embeddings, tr_i_a, te_i_a),
		"nn_sim_mean": compute_nearest_neighbor_mean_sim(embeddings, tr_i_a, te_i_a),
		"wasserstein_dist": compute_wasserstein_divergence(df_filtered, df_meta_tr_a, df_meta_va_a, df_meta_te_a),
		"knn_accuracy": knn_a["knn_accuracy"],
		"knn_top3_accuracy": knn_a["knn_top3_accuracy"],
		"knn_balanced_accuracy": knn_a["knn_balanced_accuracy"],
		"knn_f1_macro": knn_a["knn_f1_macro"],
		"hardest_class_f1": knn_a["hardest_class_f1"],
		"num_train": len(df_meta_tr_a),
		"num_val": len(df_meta_va_a),
		"num_test": len(df_meta_te_a),
	}
	raw_benchmark_records.append(rec_5a)

	# =========================================================================
	# MỤC 5B: Tối ưu hóa Đa Mục Tiêu Multi-Objective SA (PP13) - split_methods.py ONLY
	# =========================================================================
	print(f"\n-> Đang thực thi Mục 5B: Multi-Objective SA Selector (PP13 - split_methods.py ONLY)...")
	opt_sa_config, (df_meta_tr_b, df_meta_va_b, df_meta_te_b) = optimize_multi_objective_datasail_sa(
		df_filtered, embeddings, class_to_idx, path_to_idx, seed=42
	)

	tr_i_b = [path_to_idx[p] for p in df_meta_tr_b["path"]]
	va_i_b = [path_to_idx[p] for p in df_meta_va_b["path"]]
	te_i_b = [path_to_idx[p] for p in df_meta_te_b["path"]]
	knn_b = compute_knn_metrics(embeddings, df_meta_tr_b, df_meta_te_b, class_to_idx, path_to_idx)

	rec_5b = {
		"protocol": "PP13_DataSAIL_Meta_Selector_Multi_Objective_SA",
		"seed": 42,
		"datasail_loss": compute_datasail_loss(embeddings, tr_i_b, va_i_b, te_i_b),
		"inter_cosine_sim": compute_inter_split_cosine_sim(embeddings, tr_i_b, va_i_b, te_i_b),
		"intra_cosine_sim": compute_intra_split_cosine_sim(embeddings, tr_i_b, va_i_b, te_i_b),
		"slr_percent": compute_specimen_leakage_risk(df_meta_tr_b, df_meta_va_b, df_meta_te_b),
		"pri_percent": compute_pseudoreplication_index(df_meta_tr_b, df_meta_te_b),
		"ccr_percent": compute_class_coverage_rate(df_filtered, df_meta_tr_b, df_meta_va_b, df_meta_te_b),
		"silhouette_score": compute_silhouette_separation(embeddings, tr_i_b, va_i_b, te_i_b),
		"mmd_distance": compute_maximum_mean_discrepancy(embeddings, tr_i_b, te_i_b),
		"nn_sim_mean": compute_nearest_neighbor_mean_sim(embeddings, tr_i_b, te_i_b),
		"wasserstein_dist": compute_wasserstein_divergence(df_filtered, df_meta_tr_b, df_meta_va_b, df_meta_te_b),
		"knn_accuracy": knn_b["knn_accuracy"],
		"knn_top3_accuracy": knn_b["knn_top3_accuracy"],
		"knn_balanced_accuracy": knn_b["knn_balanced_accuracy"],
		"knn_f1_macro": knn_b["knn_f1_macro"],
		"hardest_class_f1": knn_b["hardest_class_f1"],
		"num_train": len(df_meta_tr_b),
		"num_val": len(df_meta_va_b),
		"num_test": len(df_meta_te_b),
	}
	raw_benchmark_records.append(rec_5b)

	df_all_results = pd.DataFrame(raw_benchmark_records)
	df_all_results.to_csv(output_dir / "all_splits_results.csv", index=False)

	configs_dict = {
		"PP12_Classwise_DataSAIL_Loss": opt_classwise_config,
		"PP13_Multi_Objective_SA": opt_sa_config,
	}
	with open(output_dir / "optimal_k_n_classwise_config.json", "w", encoding="utf-8") as f:
		json.dump(configs_dict, f, indent=2, ensure_ascii=False)

	# IN CẤU HÌNH THUẬT TOÁN TỐI ƯU SONG SONG CHO PP12 VÀ PP13 RA TERMINAL
	print("\n" + "=" * 115)
	print(" BẢNG CẤU HÌNH PHÂN BỔ THUẬT TOÁN TỐI ƯU THEO TỪNG LOÀI GỖ (PP12 vs PP13)")
	print("=" * 115)
	print(f"{'Loài Gỗ (Species Label)':<35} | {'Single-Obj Loss Selector (PP12)':<36} | {'Multi-Obj SA Meta-Selector (PP13)':<36}")
	print("-" * 115)
	for sp in class_names:
		m12 = opt_classwise_config.get(sp, "N/A")
		m13 = opt_sa_config.get(sp, "N/A")
		print(f"{sp:<35} | {m12:<36} | {m13:<36}")
	print("=" * 115 + "\n")

	# BẢNG THỐNG KÊ HỌC THUẬT
	lines = []
	lines.append("=" * 155)
	lines.append(" BẢNG TỔNG HỢP KẾT QUẢ BENCHMARK DATA LEAKAGE Q1/Q2 (MEAN ± STD QUA 5 SEEDS, 100% CLASS COVERAGE)")
	lines.append("=" * 155)

	header = f"{'Protocol':<44} {'KNN Test Acc':<16} {'F1-Macro':<16} {'DataSAIL Loss':<18} " \
	         f"{'S_inter':<12} {'SLR (%)':<10} {'CCR (%)':<10} {'NN_Sim Mean':<14} {'p-val vs R-Split':<16}"
	lines.append(header)
	lines.append("-" * 155)

	random_acc_scores = [r["knn_accuracy"] for r in raw_benchmark_records if "Random" in r["protocol"]]

	summary_rows = {}
	all_protocols = sorted(list(set(r["protocol"] for r in raw_benchmark_records)))

	for proto in all_protocols:
		subset = [r for r in raw_benchmark_records if r["protocol"] == proto]
		acc_vals = [r["knn_accuracy"] for r in subset]
		f1_vals = [r["knn_f1_macro"] for r in subset]
		loss_vals = [r["datasail_loss"] for r in subset]
		s_inter_vals = [r["inter_cosine_sim"] for r in subset]
		slr_vals = [r["slr_percent"] for r in subset]
		ccr_vals = [r["ccr_percent"] for r in subset]
		nn_vals = [r["nn_sim_mean"] for r in subset]

		acc_str = f"{np.mean(acc_vals):.4f} ± {np.std(acc_vals):.4f}" if len(acc_vals) > 1 else f"{acc_vals[0]:.4f}"
		f1_str = f"{np.mean(f1_vals):.4f} ± {np.std(f1_vals):.4f}" if len(f1_vals) > 1 else f"{f1_vals[0]:.4f}"
		loss_str = f"{np.mean(loss_vals):.1f} ± {np.std(loss_vals):.1f}" if len(loss_vals) > 1 else f"{loss_vals[0]:.1f}"
		s_inter_str = f"{np.mean(s_inter_vals):.4f}"
		slr_str = f"{np.mean(slr_vals):.1f}%"
		ccr_str = f"{np.mean(ccr_vals):.1f}%"
		nn_str = f"{np.mean(nn_vals):.4f}"

		if "Random" not in proto and len(acc_vals) > 1 and len(random_acc_scores) > 1:
			stat_res = compute_statistical_significance(random_acc_scores, acc_vals)
			pval_str = f"{stat_res['p_value']:.4e}"
		else:
			pval_str = "Baseline"

		lines.append(f"{proto:<44} {acc_str:<16} {f1_str:<16} {loss_str:<18} {s_inter_str:<12} {slr_str:<10} {ccr_str:<10} {nn_str:<14} {pval_str:<16}")

		summary_rows[proto] = {
			"acc_mean": float(np.mean(acc_vals)),
			"acc_std": float(np.std(acc_vals)),
			"f1_mean": float(np.mean(f1_vals)),
			"f1_std": float(np.std(f1_vals)),
			"datasail_loss_mean": float(np.mean(loss_vals)),
			"inter_sim_mean": float(np.mean(s_inter_vals)),
			"slr_mean": float(np.mean(slr_vals)),
			"ccr_mean": float(np.mean(ccr_vals)),
			"nn_sim_mean": float(np.mean(nn_vals)),
		}

	lines.append("=" * 155)
	summary_txt = "\n".join(lines)
	print("\n" + summary_txt)

	with open(output_dir / "summary_academic_table.txt", "w", encoding="utf-8") as f:
		f.write(summary_txt)

	return raw_benchmark_records, summary_rows, summary_txt
