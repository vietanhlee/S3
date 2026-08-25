"""
datasail_benchmark/solvers.py
=============================
Triển khai các thuật toán chia dữ liệu (Splitting Protocols) cho Benchmark Data Leakage & DataSAIL.
Tự động import và đồng bộ hóa trực tiếp từ từ điển `SPLIT_METHODS` trong `split_methods.py`.
Bảo đảm duy trì phân phối class, tỷ lệ target (60/20/20) và TUYỆT ĐỐI NGUYÊN VẸN KHỐI MẪU VẬT (không xé lẻ subfolder).
"""

import random
from typing import Tuple, List, Dict, Callable
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from .config import TRAIN_RATIO, VAL_RATIO

# Import trực tiếp từ điển SPLIT_METHODS từ split_methods.py
import split_methods as sm


def _shuffle_df(df: pd.DataFrame, seed: int) -> pd.DataFrame:
	return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def validate_and_fix_class_coverage_subfolder(
	df_all: pd.DataFrame,
	df_train: pd.DataFrame,
	df_val: pd.DataFrame,
	df_test: pd.DataFrame,
	seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	"""
	Đảm bảo Class Coverage tối đa ở cấp độ MẪU VẬT NGUYÊN VẸN (SUBFOLDER LEVEL).
	TUYỆT ĐỐI KHÔNG XÉ LẺ ẢNH TRONG SUBFOLDER.
	Nếu một tập thiếu class c, điều chuyển TOÀN BỘ 1 SUBFOLDER của class c từ tập có dư sang.
	"""
	all_classes = set(df_all["label"].unique())
	tr_classes = set(df_train["label"].unique())
	va_classes = set(df_val["label"].unique())
	te_classes = set(df_test["label"].unique())

	if tr_classes == all_classes and va_classes == all_classes and te_classes == all_classes:
		return _shuffle_df(df_train, seed), _shuffle_df(df_val, seed), _shuffle_df(df_test, seed)

	df_tr_list, df_va_list, df_te_list = [], [], []

	for cls in all_classes:
		sub_all = df_all[df_all["label"] == cls]
		sub_tr = df_train[df_train["label"] == cls]
		sub_va = df_val[df_val["label"] == cls]
		sub_te = df_test[df_test["label"] == cls]

		sfs_tr = list(sub_tr["subfolder"].unique()) if len(sub_tr) > 0 else []
		sfs_va = list(sub_va["subfolder"].unique()) if len(sub_va) > 0 else []
		sfs_te = list(sub_te["subfolder"].unique()) if len(sub_te) > 0 else []

		if not sfs_tr:
			if len(sfs_te) > 1:
				sfs_tr.append(sfs_te.pop())
			elif len(sfs_va) > 1:
				sfs_tr.append(sfs_va.pop())
		if not sfs_va:
			if len(sfs_tr) > 1:
				sfs_va.append(sfs_tr.pop())
			elif len(sfs_te) > 1:
				sfs_va.append(sfs_te.pop())
		if not sfs_te:
			if len(sfs_tr) > 1:
				sfs_te.append(sfs_tr.pop())
			elif len(sfs_va) > 1:
				sfs_te.append(sfs_va.pop())

		if sfs_tr:
			df_tr_list.append(sub_all[sub_all["subfolder"].isin(sfs_tr)])
		if sfs_va:
			df_va_list.append(sub_all[sub_all["subfolder"].isin(sfs_va)])
		if sfs_te:
			df_te_list.append(sub_all[sub_all["subfolder"].isin(sfs_te)])

	new_tr = pd.concat(df_tr_list).reset_index(drop=True) if df_tr_list else df_train
	new_va = pd.concat(df_va_list).reset_index(drop=True) if df_va_list else df_val
	new_te = pd.concat(df_te_list).reset_index(drop=True) if df_te_list else df_test

	return _shuffle_df(new_tr, seed), _shuffle_df(new_va, seed), _shuffle_df(new_te, seed)


# ============================================================
# Core DataSAIL Optimization Solver Engine
# ============================================================
def _run_datasail_solver(
	item_embeddings: np.ndarray,
	item_weights: np.ndarray,
	train_ratio: float = 0.60,
	val_ratio: float = 0.20,
	seed: int = 42,
	max_iters: int = 1500,
) -> Tuple[List[int], List[int], List[int]]:
	n = len(item_embeddings)
	if n == 0:
		return [], [], []
	if n == 1:
		return [0], [], []
	if n == 2:
		return ([0], [1], []) if val_ratio > 0 else ([0], [], [1])


	sim_matrix = cosine_similarity(item_embeddings)
	np.fill_diagonal(sim_matrix, 0.0)
	sim_matrix = np.maximum(sim_matrix, 0.0)

	total_weight = float(np.sum(item_weights))
	target_tr = total_weight * train_ratio
	target_va = total_weight * val_ratio
	target_te = total_weight * (1.0 - train_ratio - val_ratio)

	k_init = min(n, 3)
	kmeans = KMeans(n_clusters=k_init, random_state=seed, n_init="auto")
	cluster_labels = kmeans.fit_predict(item_embeddings)

	global_centroid = item_embeddings.mean(axis=0)
	cluster_dists = []
	for c in range(k_init):
		c_mask = cluster_labels == c
		if np.any(c_mask):
			c_dist = float(np.linalg.norm(item_embeddings[c_mask].mean(axis=0) - global_centroid))
			cluster_dists.append((c, c_dist))

	cluster_dists.sort(key=lambda x: -x[1])
	split_assignment = np.zeros(n, dtype=int)

	if len(cluster_dists) >= 3:
		split_assignment[cluster_labels == cluster_dists[0][0]] = 2
		split_assignment[cluster_labels == cluster_dists[1][0]] = 1
		split_assignment[cluster_labels == cluster_dists[2][0]] = 0
	elif len(cluster_dists) == 2:
		split_assignment[cluster_labels == cluster_dists[0][0]] = 2
		split_assignment[cluster_labels == cluster_dists[1][0]] = 0

	rng = np.random.RandomState(seed)

	def calc_loss(assign: np.ndarray) -> float:
		diff_mask = assign[:, None] != assign[None, :]
		weight_outer = np.outer(item_weights, item_weights)
		inter_sim_loss = float(np.sum(sim_matrix[diff_mask] * weight_outer[diff_mask]) / 2.0)

		w_tr = float(np.sum(item_weights[assign == 0]))
		w_va = float(np.sum(item_weights[assign == 1]))
		w_te = float(np.sum(item_weights[assign == 2]))

		dev_tr = (w_tr - target_tr) / total_weight
		dev_va = (w_va - target_va) / total_weight
		dev_te = (w_te - target_te) / total_weight

		ratio_penalty = 1e5 * (dev_tr**2 + dev_va**2 + dev_te**2)
		return inter_sim_loss + ratio_penalty

	best_assign = split_assignment.copy()
	best_loss = calc_loss(best_assign)

	for _ in range(max_iters):
		idx_to_move = rng.randint(0, n)
		current_split = best_assign[idx_to_move]
		new_split = rng.choice([s for s in [0, 1, 2] if s != current_split])

		cand_assign = best_assign.copy()
		cand_assign[idx_to_move] = new_split

		w_tr = np.sum(item_weights[cand_assign == 0])
		w_va = np.sum(item_weights[cand_assign == 1])
		w_te = np.sum(item_weights[cand_assign == 2])

		if w_tr == 0 or (val_ratio > 0 and w_va == 0) or w_te == 0:
			continue

		cand_loss = calc_loss(cand_assign)
		if cand_loss < best_loss:
			best_loss = cand_loss
			best_assign = cand_assign

	tr_idx = np.where(best_assign == 0)[0].tolist()
	va_idx = np.where(best_assign == 1)[0].tolist()
	te_idx = np.where(best_assign == 2)[0].tolist()

	return tr_idx, va_idx, te_idx


# ============================================================
# Additional Solvers: DataSAIL Specimen & DataSAIL Image
# ============================================================
def pp10_datasail_specimen_split(df: pd.DataFrame, embeddings: np.ndarray, seed: int = 42):
	train_idx, val_idx, test_idx = [], [], []
	for _, group in df.groupby("label"):
		subfolder_groups = group.groupby("subfolder")
		subfolder_names = list(subfolder_groups.groups.keys())
		sf_embs = [embeddings[subfolder_groups.get_group(sf).index.tolist()].mean(axis=0) for sf in subfolder_names]
		sf_weights = [len(subfolder_groups.get_group(sf)) for sf in subfolder_names]
		sf_embs = np.array(sf_embs)
		sf_weights = np.array(sf_weights, dtype=np.float32)

		tr_sf, va_sf, te_sf = _run_datasail_solver(sf_embs, sf_weights, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, seed=seed)

		for pos in tr_sf:
			train_idx.extend(subfolder_groups.get_group(subfolder_names[pos]).index.tolist())
		for pos in va_sf:
			val_idx.extend(subfolder_groups.get_group(subfolder_names[pos]).index.tolist())
		for pos in te_sf:
			test_idx.extend(subfolder_groups.get_group(subfolder_names[pos]).index.tolist())

	return validate_and_fix_class_coverage_subfolder(df, df.loc[train_idx], df.loc[val_idx], df.loc[test_idx], seed)


def pp11_datasail_image_split(df: pd.DataFrame, embeddings: np.ndarray, seed: int = 42):
	train_idx, val_idx, test_idx = [], [], []
	for _, group in df.groupby("label"):
		indices = sorted(group.index.tolist())
		img_embs = embeddings[indices]
		img_weights = np.ones(len(indices), dtype=np.float32)

		tr_img, va_img, te_img = _run_datasail_solver(img_embs, img_weights, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, seed=seed)

		for pos in tr_img:
			train_idx.append(indices[pos])
		for pos in va_img:
			val_idx.append(indices[pos])
		for pos in te_img:
			test_idx.append(indices[pos])

	return _shuffle_df(df.loc[train_idx], seed), _shuffle_df(df.loc[val_idx], seed), _shuffle_df(df.loc[test_idx], seed)


def wrap_stratified_random(df: pd.DataFrame, embeddings: np.ndarray, seed: int = 42):
	tr, va, te = sm.stratified_random_split(df, embeddings, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, seed=seed)
	return _shuffle_df(tr, seed), _shuffle_df(va, seed), _shuffle_df(te, seed)


# Tự động đóng gói tất cả phương pháp từ từ điển SPLIT_METHODS của split_methods.py
SPLIT_METHODS_WRAPPED: Dict[str, Callable] = {}

for name, fn in sm.SPLIT_METHODS.items():
	def _make_wrapper(func):
		def _wrapper(df: pd.DataFrame, embeddings: np.ndarray, seed: int = 42):
			tr, va, te = func(df, embeddings, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, seed=seed)
			return validate_and_fix_class_coverage_subfolder(df, tr, va, te, seed)
		return _wrapper
	SPLIT_METHODS_WRAPPED[name] = _make_wrapper(fn)


# Registry tổng hợp đầy đủ các phương pháp
ALL_SOLVERS: Dict[str, Callable] = {
	"PP0_Stratified_Random": wrap_stratified_random,
	**SPLIT_METHODS_WRAPPED,
	"PP10_DataSAIL_Specimen": pp10_datasail_specimen_split,
	"PP11_DataSAIL_Image": pp11_datasail_image_split,
}
