"""
02_specimen_variant_gan.partitioning.heuristic_splits
=====================================================
Các giao thức phân hoạch dựa trên Mahalanobis (Fixed, Iterative) và Group-based:
- PP1: Fixed Mahalanobis Distance
- PP2: Iterative Centroid Mahalanobis Distance (cấp subfolder)
- PP3: Group-based Split (subfolder/specimen)
- PP9: Agglomerative Stratified Mahalanobis Split
"""

import random
from typing import Tuple, List
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, fcluster
from tqdm import tqdm

from .base import (
	compute_split_counts,
	_shuffle_df,
	_mahalanobis_distances,
	_mahalanobis_dist_to_centroid,
	find_optimal_clusters_elbow,
	_split_by_groups,
)


def _split_by_mahalanobis_image_level(
	indices: List[int],
	subset_emb: np.ndarray,
	train_ratio: float,
	val_ratio: float,
	seed: int,
	eps: float = 1e-6,
) -> Tuple[List[int], List[int], List[int]]:
	"""Phân chia các ảnh đơn lẻ dựa trên khoảng cách Mahalanobis đến centroid chung của subset."""
	n_total = len(indices)
	train_count, val_count, test_count = compute_split_counts(n_total, train_ratio, val_ratio)

	if n_total == 0:
		return [], [], []

	# Giảm chiều bằng PCA nếu đủ lớn
	pca_emb = subset_emb
	if n_total >= 5:
		d_prime = min(n_total - 2, 128)
		if d_prime >= 2:
			pca = PCA(n_components=d_prime, random_state=seed)
			pca_emb = pca.fit_transform(subset_emb)

	dists = _mahalanobis_distances(pca_emb, eps=eps)
	sorted_positions = np.argsort(-dists)

	test_idx, val_idx, train_idx = [], [], []
	for i, pos in enumerate(sorted_positions):
		orig_idx = indices[pos]
		if i < test_count:
			test_idx.append(orig_idx)
		elif i < test_count + val_count:
			val_idx.append(orig_idx)
		else:
			train_idx.append(orig_idx)

	return train_idx, val_idx, test_idx


def mahalanobis_fixed_split(
	df: pd.DataFrame,
	embeddings: np.ndarray,
	train_ratio: float = 0.60,
	val_ratio: float = 0.20,
	seed: int = 42,
	eps: float = 1e-6,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	"""
	PP1: Tính khoảng cách Mahalanobis đến centroid CỐ ĐỊNH.
	Ảnh xa nhất -> test, tiếp theo -> val, còn lại -> train.
	"""
	if len(df) != embeddings.shape[0]:
		raise ValueError("Embeddings length does not match dataframe length")

	train_idx, val_idx, test_idx = [], [], []

	for _, group in tqdm(df.groupby("label"), desc="PP1-MahalFixed"):
		indices = sorted(group.index.tolist())
		n_total = len(indices)
		train_count, val_count, test_count = compute_split_counts(n_total, train_ratio, val_ratio)

		if n_total == 0:
			continue

		subset_emb = embeddings[np.array(indices)]
		dists = _mahalanobis_distances(subset_emb, eps=eps)
		sorted_positions = np.argsort(-dists)

		for i, pos in enumerate(sorted_positions):
			orig_idx = indices[pos]
			if i < test_count:
				test_idx.append(orig_idx)
			elif i < test_count + val_count:
				val_idx.append(orig_idx)
			else:
				train_idx.append(orig_idx)

	df_train = _shuffle_df(df.loc[train_idx], seed)
	df_val = _shuffle_df(df.loc[val_idx], seed)
	df_test = _shuffle_df(df.loc[test_idx], seed)
	return df_train, df_val, df_test


def mahalanobis_iterative_split(
	df: pd.DataFrame,
	embeddings: np.ndarray,
	train_ratio: float = 0.60,
	val_ratio: float = 0.20,
	seed: int = 42,
	eps: float = 1e-6,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	"""
	PP2: Tính khoảng cách Mahalanobis, TÁI TÍNH centroid sau mỗi lần rút subfolder.
	Đảm bảo KHÔNG rò rỉ dữ liệu mức subfolder bằng cách chia ở cấp độ subfolder.
	Sử dụng PCA 128 chiều.
	"""
	if len(df) != embeddings.shape[0]:
		raise ValueError("Embeddings length does not match dataframe length")

	train_idx, val_idx, test_idx = [], [], []

	for _, group in tqdm(df.groupby("label"), desc="PP2-MahalIter"):
		subfolder_groups = group.groupby("subfolder")
		subfolder_names = list(subfolder_groups.groups.keys())
		n_subfolders = len(subfolder_names)

		if n_subfolders == 0:
			continue

		subfolder_embs = []
		for sf_name in subfolder_names:
			sf_indices = subfolder_groups.get_group(sf_name).index.tolist()
			sf_emb_mean = embeddings[sf_indices].mean(axis=0)
			subfolder_embs.append(sf_emb_mean)

		subfolder_embs = np.array(subfolder_embs)

		if n_subfolders < 3:
			groups = [subfolder_groups.get_group(sf_name).index.tolist() for sf_name in subfolder_names]
			tr, va, te = _split_by_groups(groups, train_ratio, val_ratio, seed, sorted_descending=True)
			train_idx.extend(tr)
			val_idx.extend(va)
			test_idx.extend(te)
			continue

		if n_subfolders >= 5:
			d_prime = min(n_subfolders - 2, 128)
			if d_prime >= 2:
				pca = PCA(n_components=d_prime, random_state=seed)
				subfolder_embs = pca.fit_transform(subfolder_embs)

		train_sf_count, val_sf_count, test_sf_count = compute_split_counts(n_subfolders, train_ratio, val_ratio)

		remaining_pos = list(range(n_subfolders))
		picked_test_pos = []
		picked_val_pos = []

		# Rút test
		for _ in range(min(test_sf_count, len(remaining_pos))):
			cur_embs = subfolder_embs[remaining_pos]
			dists = _mahalanobis_distances(cur_embs, eps=eps)
			max_pos = int(np.argmax(dists))
			picked_test_pos.append(remaining_pos[max_pos])
			remaining_pos.pop(max_pos)

		# Rút val
		for _ in range(min(val_sf_count, len(remaining_pos))):
			cur_embs = subfolder_embs[remaining_pos]
			dists = _mahalanobis_distances(cur_embs, eps=eps)
			max_pos = int(np.argmax(dists))
			picked_val_pos.append(remaining_pos[max_pos])
			remaining_pos.pop(max_pos)

		for pos in picked_test_pos:
			sf_name = subfolder_names[pos]
			test_idx.extend(subfolder_groups.get_group(sf_name).index.tolist())

		for pos in picked_val_pos:
			sf_name = subfolder_names[pos]
			val_idx.extend(subfolder_groups.get_group(sf_name).index.tolist())

		for pos in remaining_pos:
			sf_name = subfolder_names[pos]
			train_idx.extend(subfolder_groups.get_group(sf_name).index.tolist())

	df_train = _shuffle_df(df.loc[train_idx], seed)
	df_val = _shuffle_df(df.loc[val_idx], seed)
	df_test = _shuffle_df(df.loc[test_idx], seed)
	return df_train, df_val, df_test


def group_based_split(
	df: pd.DataFrame,
	embeddings: np.ndarray = None,
	train_ratio: float = 0.60,
	val_ratio: float = 0.20,
	seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	"""
	PP3: Chia theo đơn vị subfolder, KHÔNG bao giờ cắt subfolder.
	Mỗi subfolder đại diện cho 1 nhóm ảnh cùng mẫu gỗ/nguồn thu thập.
	"""
	rng = random.Random(seed)
	train_idx, val_idx, test_idx = [], [], []

	for _, group in tqdm(df.groupby("label"), desc="PP3-GroupBased"):
		subfolder_groups = group.groupby("subfolder")
		subfolder_names = list(subfolder_groups.groups.keys())
		rng.shuffle(subfolder_names)

		n_total = len(group)

		if len(subfolder_names) < 3:
			groups = [subfolder_groups.get_group(sf_name).index.tolist() for sf_name in subfolder_names]
			tr, va, te = _split_by_groups(groups, train_ratio, val_ratio, seed, sorted_descending=True)
			train_idx.extend(tr)
			val_idx.extend(va)
			test_idx.extend(te)
			continue

		target_train = int(n_total * train_ratio)
		target_val = int(n_total * val_ratio)

		test_idx.extend(subfolder_groups.get_group(subfolder_names[0]).index.tolist())
		val_idx.extend(subfolder_groups.get_group(subfolder_names[1]).index.tolist())

		curr_train = 0
		curr_val = len(subfolder_groups.get_group(subfolder_names[1]).index.tolist())
		curr_test = len(subfolder_groups.get_group(subfolder_names[0]).index.tolist())

		for sf_name in subfolder_names[2:]:
			sf_indices = subfolder_groups.get_group(sf_name).index.tolist()
			sf_count = len(sf_indices)

			if curr_train < target_train:
				train_idx.extend(sf_indices)
				curr_train += sf_count
			elif curr_val < target_val:
				val_idx.extend(sf_indices)
				curr_val += sf_count
			else:
				test_idx.extend(sf_indices)
				curr_test += sf_count

	df_train = _shuffle_df(df.loc[train_idx], seed)
	df_val = _shuffle_df(df.loc[val_idx], seed)
	df_test = _shuffle_df(df.loc[test_idx], seed)
	return df_train, df_val, df_test


def agglom_stratified_split(
	df: pd.DataFrame,
	embeddings: np.ndarray,
	train_ratio: float = 0.60,
	val_ratio: float = 0.20,
	seed: int = 42,
	eps: float = 1e-6,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	"""
	PP9: Agglom Stratified — Gom các subfolders bằng AgglomerativeClustering cho mỗi class.
	Xác định số cụm tối ưu qua Elbow Method (3-30).
	Tính khoảng cách cụm subfolders đến centroid chung bằng khoảng cách Mahalanobis (PCA 128 chiều).
	Chia các cụm subfolders thành 3 dải khoảng cách (Gần, Vừa, Xa), phân bổ nguyên vẹn.
	"""
	if len(df) != embeddings.shape[0]:
		raise ValueError("Embeddings length does not match dataframe length")

	train_idx, val_idx, test_idx = [], [], []

	for label, group in tqdm(df.groupby("label"), desc="PP9-AgglomStratified"):
		subfolder_groups = group.groupby("subfolder")
		subfolder_names = list(subfolder_groups.groups.keys())
		n_subfolders = len(subfolder_names)

		if n_subfolders == 0:
			continue

		if n_subfolders < 3:
			groups = [subfolder_groups.get_group(sf_name).index.tolist() for sf_name in subfolder_names]
			tr, va, te = _split_by_groups(groups, train_ratio, val_ratio, seed, sorted_descending=True)
			train_idx.extend(tr)
			val_idx.extend(va)
			test_idx.extend(te)
			continue

		subfolder_embs = []
		for sf_name in subfolder_names:
			sf_indices = subfolder_groups.get_group(sf_name).index.tolist()
			subfolder_embs.append(embeddings[sf_indices].mean(axis=0))
		subfolder_embs = np.array(subfolder_embs)

		pca_emb = subfolder_embs
		if n_subfolders >= 5:
			d_prime = min(n_subfolders - 2, 128)
			if d_prime >= 2:
				pca = PCA(n_components=d_prime, random_state=seed)
				pca_emb = pca.fit_transform(subfolder_embs)

		cov = np.cov(pca_emb, rowvar=False)
		cov = np.atleast_2d(cov)
		cov += np.eye(cov.shape[0]) * eps
		cov_inv = np.linalg.pinv(cov)
		class_centroid = pca_emb.mean(axis=0)

		n_clusters = find_optimal_clusters_elbow(pca_emb, max_k=30, seed=seed)
		n_clusters = min(n_clusters, n_subfolders)

		Z = linkage(pca_emb, method="ward")
		cluster_labels = fcluster(Z, t=n_clusters, criterion="maxclust")

		cluster_ids = sorted(set(cluster_labels))
		clusters_info = []
		for cid in cluster_ids:
			mask = cluster_labels == cid
			comp_emb_pca = pca_emb[mask]
			comp_centroid = comp_emb_pca.mean(axis=0)
			dist = _mahalanobis_dist_to_centroid(comp_centroid, class_centroid, cov_inv)[0]
			comp_sf_names = [subfolder_names[j] for j in range(n_subfolders) if cluster_labels[j] == cid]
			clusters_info.append((dist, comp_sf_names))

		clusters_info.sort(key=lambda x: x[0])
		K = len(clusters_info)

		if K < 3:
			groups = []
			for _, sf_list in clusters_info:
				g_indices = []
				for sf_name in sf_list:
					g_indices.extend(subfolder_groups.get_group(sf_name).index.tolist())
				groups.append(g_indices)
			tr, va, te = _split_by_groups(groups, train_ratio, val_ratio, seed, sorted_descending=False)
			train_idx.extend(tr)
			val_idx.extend(va)
			test_idx.extend(te)
			continue

		near_num = max(1, K // 3)
		mid_num = max(1, (K - near_num) // 2)

		near_clusters = clusters_info[:near_num]
		mid_clusters = clusters_info[near_num:near_num + mid_num]
		far_clusters = clusters_info[near_num + mid_num:]

		train_clusters = [near_clusters[0]]
		val_clusters = [mid_clusters[0]]
		test_clusters = [far_clusters[0]]

		remaining_near = near_clusters[1:]
		remaining_mid = mid_clusters[1:]
		remaining_far = far_clusters[1:]

		train_idx_label = []
		val_idx_label = []
		test_idx_label = []

		for _, sf_list in train_clusters:
			for sf_name in sf_list:
				train_idx_label.extend(subfolder_groups.get_group(sf_name).index.tolist())
		for _, sf_list in val_clusters:
			for sf_name in sf_list:
				val_idx_label.extend(subfolder_groups.get_group(sf_name).index.tolist())
		for _, sf_list in test_clusters:
			for sf_name in sf_list:
				test_idx_label.extend(subfolder_groups.get_group(sf_name).index.tolist())

		curr_train = len(train_idx_label)
		curr_val = len(val_idx_label)
		curr_test = len(test_idx_label)
		n_total = len(group)
		target_train = int(n_total * train_ratio)
		target_val = int(n_total * val_ratio)

		def allocate_band_clusters(band_clusters):
			nonlocal curr_train, curr_val, curr_test
			label_seed = sum(ord(c) for c in label)
			rng_local = random.Random(seed + label_seed)
			shuffled_band = band_clusters.copy()
			rng_local.shuffle(shuffled_band)

			for _, sf_list in shuffled_band:
				c_indices = []
				for sf_name in sf_list:
					c_indices.extend(subfolder_groups.get_group(sf_name).index.tolist())
				c_count = len(c_indices)

				diff_tr = max(0, target_train - curr_train)
				diff_va = max(0, target_val - curr_val)
				diff_te = max(0, (n_total - target_train - target_val) - curr_test)

				total_diff = diff_tr + diff_va + diff_te
				if total_diff == 0:
					min_split = min(curr_train, curr_val, curr_test)
					if min_split == curr_train:
						train_idx_label.extend(c_indices)
						curr_train += c_count
					elif min_split == curr_val:
						val_idx_label.extend(c_indices)
						curr_val += c_count
					else:
						test_idx_label.extend(c_indices)
						curr_test += c_count
					continue

				max_diff = max(diff_tr, diff_va, diff_te)
				if max_diff == diff_tr:
					train_idx_label.extend(c_indices)
					curr_train += c_count
				elif max_diff == diff_va:
					val_idx_label.extend(c_indices)
					curr_val += c_count
				else:
					test_idx_label.extend(c_indices)
					curr_test += c_count

		allocate_band_clusters(remaining_near)
		allocate_band_clusters(remaining_mid)
		allocate_band_clusters(remaining_far)

		train_idx.extend(train_idx_label)
		val_idx.extend(val_idx_label)
		test_idx.extend(test_idx_label)

	df_train = _shuffle_df(df.loc[train_idx], seed)
	df_val = _shuffle_df(df.loc[val_idx], seed)
	df_test = _shuffle_df(df.loc[test_idx], seed)
	return df_train, df_val, df_test
