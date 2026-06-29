"""
split_methods.py
================
8 phương pháp chia dữ liệu để tránh data leakage cho bài toán phân loại ảnh mặt cắt gỗ.

Mỗi hàm nhận:
	- df: pd.DataFrame (cột: path, label, genus, species, subfolder)
	- embeddings: np.ndarray (N x D) — embedding từ tf_efficientnet_b4
	- train_ratio, val_ratio, seed
Và trả về: (df_train, df_val, df_test)
"""

import random
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA


# ============================================================
# Utility: tính số lượng train/val/test cho mỗi class
# ============================================================

def compute_split_counts(
	n_total: int,
	train_ratio: float,
	val_ratio: float,
) -> tuple[int, int, int]:
	"""Tính số lượng mẫu cho mỗi split, đảm bảo val và test >= 1 nếu n >= 3."""
	if n_total <= 0:
		return 0, 0, 0

	test_ratio = 1.0 - train_ratio - val_ratio
	if test_ratio < 0:
		raise ValueError("train_ratio + val_ratio must be <= 1.0")

	val_count = int(n_total * val_ratio)
	test_count = int(n_total * test_ratio)
	train_count = n_total - val_count - test_count

	if n_total >= 3:
		if val_count == 0:
			if train_count > 1:
				train_count -= 1
				val_count = 1
			elif test_count > 1:
				test_count -= 1
				val_count = 1

		if test_count == 0:
			if train_count > 1:
				train_count -= 1
				test_count = 1
			elif val_count > 1:
				val_count -= 1
				test_count = 1

	return train_count, val_count, test_count


# ============================================================
# Utility: validate split
# ============================================================

def validate_split(
	df_all: pd.DataFrame,
	df_train: pd.DataFrame,
	df_val: pd.DataFrame,
	df_test: pd.DataFrame,
	method_name: str,
) -> bool:
	"""Validate: no overlap, completeness, mỗi class có ít nhất 1 mẫu/split."""
	all_paths = set(df_all["path"])
	train_paths = set(df_train["path"])
	val_paths = set(df_val["path"])
	test_paths = set(df_test["path"])

	ok = True

	# No overlap
	if train_paths & val_paths:
		print(f"[{method_name}] ERROR: train ∩ val = {len(train_paths & val_paths)} ảnh")
		ok = False
	if train_paths & test_paths:
		print(f"[{method_name}] ERROR: train ∩ test = {len(train_paths & test_paths)} ảnh")
		ok = False
	if val_paths & test_paths:
		print(f"[{method_name}] ERROR: val ∩ test = {len(val_paths & test_paths)} ảnh")
		ok = False

	# Completeness
	union = train_paths | val_paths | test_paths
	if union != all_paths:
		missing = all_paths - union
		extra = union - all_paths
		print(f"[{method_name}] ERROR: missing={len(missing)}, extra={len(extra)}")
		ok = False

	# Class coverage
	all_labels = set(df_all["label"].unique())
	for split_name, df_split in [("train", df_train), ("val", df_val), ("test", df_test)]:
		missing_labels = all_labels - set(df_split["label"].unique())
		if missing_labels:
			print(f"[{method_name}] WARNING: {split_name} thiếu {len(missing_labels)} class: {missing_labels}")

	if ok:
		print(f"[{method_name}] ✓ Validation passed (train={len(df_train)}, val={len(df_val)}, test={len(df_test)})")

	return ok


def _shuffle_df(df: pd.DataFrame, seed: int) -> pd.DataFrame:
	"""Shuffle dataframe và reset index."""
	return df.sample(frac=1, random_state=seed).reset_index(drop=True)


# ============================================================
# PP1: Mahalanobis Fixed Centroid
# ============================================================

def _mahalanobis_distances(embeddings: np.ndarray, eps: float = 1e-6) -> np.ndarray:
	"""Tính khoảng cách Mahalanobis từ mỗi điểm đến centroid cố định."""
	n_samples = embeddings.shape[0]
	if n_samples <= 1:
		return np.zeros(n_samples, dtype=np.float32)

	mean = embeddings.mean(axis=0)
	cov = np.cov(embeddings, rowvar=False)
	cov = np.atleast_2d(cov)
	cov += np.eye(cov.shape[0]) * eps
	cov_inv = np.linalg.pinv(cov)

	diff = embeddings - mean
	d2 = np.einsum("ij,jk,ik->i", diff, cov_inv, diff)
	d2 = np.maximum(d2, 0.0)
	return np.sqrt(d2).astype(np.float32)


def _mahalanobis_dist_to_centroid(points: np.ndarray, mean: np.ndarray, cov_inv: np.ndarray) -> np.ndarray:
	"""Tính khoảng cách Mahalanobis của một hoặc nhiều điểm đến centroid cho trước."""
	if points.ndim == 1:
		points = points[np.newaxis, :]
	diff = points - mean
	d2 = np.einsum("ij,jk,ik->i", diff, cov_inv, diff)
	d2 = np.maximum(d2, 0.0)
	return np.sqrt(d2).astype(np.float32)


def find_optimal_clusters_elbow(embeddings: np.ndarray, max_k: int = 30, seed: int = 42) -> int:
	"""Tự động xác định số lượng cụm tối ưu từ 3 đến max_k dựa trên Elbow Method."""
	n_samples = embeddings.shape[0]
	if n_samples <= 4:
		return 3
	
	max_k = min(max_k, n_samples - 1)
	if max_k < 3:
		return 3
		
	wcss = []
	k_values = list(range(3, max_k + 1))
	for k in k_values:
		kmeans = KMeans(n_clusters=k, random_state=seed, n_init='auto')
		kmeans.fit(embeddings)
		wcss.append(kmeans.inertia_)
		
	if len(wcss) <= 2:
		return 3
		
	# Chuẩn hóa cả hai trục về [0, 1] để loại bỏ sai lệch tỷ lệ đơn vị (scaling bias) giữa K và WCSS
	k_min, k_max = k_values[0], k_values[-1]
	wcss_min, wcss_max = min(wcss), max(wcss)
	
	k_denom = (k_max - k_min) if (k_max - k_min) > 0 else 1.0
	w_denom = (wcss_max - wcss_min) if (wcss_max - wcss_min) > 0 else 1.0
	
	distances = []
	for i in range(len(k_values)):
		# Chuẩn hóa điểm hiện tại
		x0 = (k_values[i] - k_min) / k_denom
		y0 = (wcss[i] - wcss_min) / w_denom
		
		# Khoảng cách từ (x0, y0) đến đường chéo nối (0, 1) và (1, 0)
		# Phương trình đường chéo: x + y - 1 = 0
		dist = abs(x0 + y0 - 1.0) / np.sqrt(2.0)
		distances.append(dist)
		
	optimal_k = k_values[np.argmax(distances)]
	return max(3, optimal_k)


def _split_by_mahalanobis_image_level(
	indices: list[int],
	subset_emb: np.ndarray,
	train_ratio: float,
	val_ratio: float,
	seed: int,
	eps: float = 1e-6,
) -> tuple[list[int], list[int], list[int]]:
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
	# Sort theo khoảng cách giảm dần (xa nhất lên đầu)
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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	"""
	PP1: Tính khoảng cách Mahalanobis đến centroid CỐ ĐỊNH.
	Ảnh xa nhất → test, tiếp theo → val, còn lại → train.
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

		# Tính khoảng cách Mahalanobis với centroid cố định
		subset_emb = embeddings[np.array(indices)]
		dists = _mahalanobis_distances(subset_emb, eps=eps)

		# Sort theo khoảng cách giảm dần
		sorted_positions = np.argsort(-dists)

		# Xa nhất → test, tiếp → val, còn lại → train
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


# ============================================================
# PP2: Mahalanobis Iterative Centroid
# ============================================================

def mahalanobis_iterative_split(
	df: pd.DataFrame,
	embeddings: np.ndarray,
	train_ratio: float = 0.60,
	val_ratio: float = 0.20,
	seed: int = 42,
	eps: float = 1e-6,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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

		# Gom đặc trưng cho từng subfolder
		subfolder_embs = []
		for sf_name in subfolder_names:
			sf_indices = subfolder_groups.get_group(sf_name).index.tolist()
			sf_emb_mean = embeddings[sf_indices].mean(axis=0)
			subfolder_embs.append(sf_emb_mean)
		
		subfolder_embs = np.array(subfolder_embs)

		# Trường hợp đặc biệt: < 3 subfolders
		if n_subfolders < 3:
			indices = group.index.tolist()
			tr, va, te = _split_by_mahalanobis_image_level(
				indices, embeddings[np.array(indices)], train_ratio, val_ratio, seed, eps
			)
			train_idx.extend(tr)
			val_idx.extend(va)
			test_idx.extend(te)
			continue

		# Giảm chiều bằng PCA nếu số lượng subfolders lớn (128 chiều)
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

		# Phân bổ ngược lại ảnh
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


# ============================================================
# PP3: Group-based Split (Subfolder)
# ============================================================

def group_based_split(
	df: pd.DataFrame,
	embeddings: np.ndarray,  # Không dùng, giữ signature nhất quán
	train_ratio: float = 0.60,
	val_ratio: float = 0.20,
	seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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

		# Phân bổ nguyên vẹn các subfolders
		if len(subfolder_names) < 3:
			indices = group.index.tolist()
			tr, va, te = _split_by_mahalanobis_image_level(
				indices, embeddings[np.array(indices)], train_ratio, val_ratio, seed
			)
			train_idx.extend(tr)
			val_idx.extend(va)
			test_idx.extend(te)
			continue

		# >= 3 subfolders
		target_train = int(n_total * train_ratio)
		target_val = int(n_total * val_ratio)

		# Đảm bảo mỗi tập nhận ít nhất 1 subfolder trước
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


# ============================================================
# PP4: Hierarchical Clustering Split
# ============================================================

def hierarchical_clustering_split(
	df: pd.DataFrame,
	embeddings: np.ndarray,
	train_ratio: float = 0.60,
	val_ratio: float = 0.20,
	seed: int = 42,
	eps: float = 1e-6,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	"""
	PP4: Agglomerative Clustering (Ward) trên centroid embeddings của subfolders.
	Xác định số cụm tối ưu qua Elbow Method (3-30).
	Tính khoảng cách centroid cụm đến centroid chung qua khoảng cách Mahalanobis (PCA 128 chiều).
	Gán nguyên cụm subfolder vào test/val/train, KHÔNG bao giờ chia cắt subfolder.
	"""
	if len(df) != embeddings.shape[0]:
		raise ValueError("Embeddings length does not match dataframe length")

	train_idx, val_idx, test_idx = [], [], []

	for _, group in tqdm(df.groupby("label"), desc="PP4-HierClust"):
		subfolder_groups = group.groupby("subfolder")
		subfolder_names = list(subfolder_groups.groups.keys())
		n_subfolders = len(subfolder_names)

		if n_subfolders == 0:
			continue

		if n_subfolders < 3:
			indices = group.index.tolist()
			tr, va, te = _split_by_mahalanobis_image_level(
				indices, embeddings[np.array(indices)], train_ratio, val_ratio, seed, eps
			)
			train_idx.extend(tr)
			val_idx.extend(va)
			test_idx.extend(te)
			continue

		# Tính subfolder embeddings
		subfolder_embs = []
		for sf_name in subfolder_names:
			sf_indices = subfolder_groups.get_group(sf_name).index.tolist()
			subfolder_embs.append(embeddings[sf_indices].mean(axis=0))
		subfolder_embs = np.array(subfolder_embs)

		# PCA 128 chiều trên subfolder embeddings để tính covariance nghịch đảo
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
		global_centroid = pca_emb.mean(axis=0)

		# Số cụm subfolders bằng Elbow Method
		n_clusters = find_optimal_clusters_elbow(pca_emb, max_k=30, seed=seed)
		n_clusters = min(n_clusters, n_subfolders)

		# Agglomerative clustering ở cấp độ subfolder
		Z = linkage(pca_emb, method="ward")
		cluster_labels = fcluster(Z, t=n_clusters, criterion="maxclust")

		cluster_ids = sorted(set(cluster_labels))
		cluster_info = []

		for cid in cluster_ids:
			mask = cluster_labels == cid
			cluster_centroid = pca_emb[mask].mean(axis=0)
			# Tính khoảng cách Mahalanobis
			dist_to_global = _mahalanobis_dist_to_centroid(cluster_centroid, global_centroid, cov_inv)[0]
			
			# Gom tất cả các index ảnh của các subfolders thuộc cluster này
			cluster_img_indices = []
			for idx_sf, sf_lbl in enumerate(cluster_labels):
				if sf_lbl == cid:
					sf_name = subfolder_names[idx_sf]
					cluster_img_indices.extend(subfolder_groups.get_group(sf_name).index.tolist())
			
			cluster_info.append((cid, dist_to_global, cluster_img_indices))

		# Sort: cụm xa nhất trước
		cluster_info.sort(key=lambda x: -x[1])

		# Đảm bảo tập Train không bao giờ trống và xử lý số lượng cụm < 3
		if len(cluster_info) < 3:
			indices = group.index.tolist()
			tr, va, te = _split_by_mahalanobis_image_level(
				indices, embeddings[np.array(indices)], train_ratio, val_ratio, seed, eps
			)
			train_idx.extend(tr)
			val_idx.extend(va)
			test_idx.extend(te)
			continue

		# >= 3 cụm
		n_total = len(group)
		target_train = int(n_total * train_ratio)
		target_val = int(n_total * val_ratio)

		# Đảm bảo mỗi tập nhận ít nhất 1 cụm subfolder
		test_idx.extend(cluster_info[0][2])
		val_idx.extend(cluster_info[1][2])

		curr_train = 0
		curr_val = len(cluster_info[1][2])
		curr_test = len(cluster_info[0][2])

		for _, _, c_indices in cluster_info[2:]:
			c_count = len(c_indices)
			if curr_train < target_train:
				train_idx.extend(c_indices)
				curr_train += c_count
			elif curr_val < target_val:
				val_idx.extend(c_indices)
				curr_val += c_count
			else:
				test_idx.extend(c_indices)
				curr_test += c_count

	df_train = _shuffle_df(df.loc[train_idx], seed)
	df_val = _shuffle_df(df.loc[val_idx], seed)
	df_test = _shuffle_df(df.loc[test_idx], seed)
	return df_train, df_val, df_test


# ============================================================
# PP5: Cosine Similarity Graph + Connected Components
# ============================================================

def cosine_graph_split(
	df: pd.DataFrame,
	embeddings: np.ndarray,
	train_ratio: float = 0.60,
	val_ratio: float = 0.20,
	seed: int = 42,
	cosine_threshold: float = 0.92,
	eps: float = 1e-6,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	"""
	PP5: Xây đồ thị cosine similarity ở cấp độ subfolder, tìm connected components, chia theo component.
	Sắp xếp các components bằng khoảng cách Mahalanobis (PCA 128 chiều).
	Tuyệt đối KHÔNG chia cắt subfolder để tránh rò rỉ dữ liệu.
	"""
	if len(df) != embeddings.shape[0]:
		raise ValueError("Embeddings length does not match dataframe length")

	train_idx, val_idx, test_idx = [], [], []

	for _, group in tqdm(df.groupby("label"), desc="PP5-CosGraph"):
		subfolder_groups = group.groupby("subfolder")
		subfolder_names = list(subfolder_groups.groups.keys())
		n_subfolders = len(subfolder_names)

		if n_subfolders == 0:
			continue

		if n_subfolders < 3:
			indices = group.index.tolist()
			tr, va, te = _split_by_mahalanobis_image_level(
				indices, embeddings[np.array(indices)], train_ratio, val_ratio, seed, eps
			)
			train_idx.extend(tr)
			val_idx.extend(va)
			test_idx.extend(te)
			continue

		# Tính subfolder embeddings
		subfolder_embs = []
		for sf_name in subfolder_names:
			sf_indices = subfolder_groups.get_group(sf_name).index.tolist()
			subfolder_embs.append(embeddings[sf_indices].mean(axis=0))
		subfolder_embs = np.array(subfolder_embs)

		# PCA 128 chiều trên subfolder embeddings để tính covariance nghịch đảo
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
		global_centroid = pca_emb.mean(axis=0)

		# Tính cosine similarity giữa các subfolder embeddings
		sim_matrix = cosine_similarity(subfolder_embs)
		np.fill_diagonal(sim_matrix, 0.0)

		# Xây adjacency matrix (sparse)
		adj = (sim_matrix >= cosine_threshold).astype(np.float32)
		sparse_adj = csr_matrix(adj)

		# Tìm connected components của subfolders
		n_components, component_labels = connected_components(sparse_adj, directed=False)

		# Gom thành danh sách component subfolders
		components = []

		for cid in range(n_components):
			mask = component_labels == cid
			comp_emb_pca = pca_emb[mask]
			comp_centroid = comp_emb_pca.mean(axis=0)
			# Tính khoảng cách Mahalanobis
			dist = _mahalanobis_dist_to_centroid(comp_centroid, global_centroid, cov_inv)[0]
			
			# Gom tất cả các index ảnh của các subfolders thuộc component này
			comp_img_indices = []
			for idx_sf, comp_lbl in enumerate(component_labels):
				if comp_lbl == cid:
					sf_name = subfolder_names[idx_sf]
					comp_img_indices.extend(subfolder_groups.get_group(sf_name).index.tolist())
			
			components.append((cid, dist, comp_img_indices))

		# Sort: component xa nhất trước
		components.sort(key=lambda x: -x[1])

		# Đảm bảo tập Train không bao giờ trống và xử lý số lượng components < 3
		if len(components) < 3:
			indices = group.index.tolist()
			tr, va, te = _split_by_mahalanobis_image_level(
				indices, embeddings[np.array(indices)], train_ratio, val_ratio, seed, eps
			)
			train_idx.extend(tr)
			val_idx.extend(va)
			test_idx.extend(te)
			continue

		# >= 3 components
		n_total = len(group)
		target_train = int(n_total * train_ratio)
		target_val = int(n_total * val_ratio)

		# Đảm bảo mỗi tập nhận ít nhất 1 component
		test_idx.extend(components[0][2])
		val_idx.extend(components[1][2])

		curr_train = 0
		curr_val = len(components[1][2])
		curr_test = len(components[0][2])

		for _, _, c_indices in components[2:]:
			c_count = len(c_indices)
			if curr_train < target_train:
				train_idx.extend(c_indices)
				curr_train += c_count
			elif curr_val < target_val:
				val_idx.extend(c_indices)
				curr_val += c_count
			else:
				test_idx.extend(c_indices)
				curr_test += c_count

	df_train = _shuffle_df(df.loc[train_idx], seed)
	df_val = _shuffle_df(df.loc[val_idx], seed)
	df_test = _shuffle_df(df.loc[test_idx], seed)
	return df_train, df_val, df_test


# ============================================================
# PP6: Stratified Random Split (Baseline)
# ============================================================

def stratified_random_split(
	df: pd.DataFrame,
	embeddings: np.ndarray,  # Không dùng, giữ signature nhất quán
	train_ratio: float = 0.60,
	val_ratio: float = 0.20,
	seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	"""
	PP6: Baseline — chia ngẫu nhiên giữ tỉ lệ class.
	Được viết lại thủ công để tránh crash sklearn.model_selection.train_test_split
	khi có class có quá ít mẫu vật (ví dụ 1 hoặc 2 mẫu).
	"""
	rng = random.Random(seed)
	train_idx, val_idx, test_idx = [], [], []

	for _, group in df.groupby("label"):
		indices = sorted(group.index.tolist())
		rng.shuffle(indices)
		
		n_total = len(indices)
		train_count, val_count, test_count = compute_split_counts(n_total, train_ratio, val_ratio)
		
		train_idx.extend(indices[:train_count])
		val_idx.extend(indices[train_count:train_count + val_count])
		test_idx.extend(indices[train_count + val_count:])

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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
			indices = group.index.tolist()
			tr, va, te = _split_by_mahalanobis_image_level(
				indices, embeddings[np.array(indices)], train_ratio, val_ratio, seed, eps
			)
			train_idx.extend(tr)
			val_idx.extend(va)
			test_idx.extend(te)
			continue

		# Tính subfolder embeddings
		subfolder_embs = []
		for sf_name in subfolder_names:
			sf_indices = subfolder_groups.get_group(sf_name).index.tolist()
			subfolder_embs.append(embeddings[sf_indices].mean(axis=0))
		subfolder_embs = np.array(subfolder_embs)

		# PCA 128 chiều
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

		# Xác định số cụm bằng Elbow Method
		n_clusters = find_optimal_clusters_elbow(pca_emb, max_k=30, seed=seed)
		n_clusters = min(n_clusters, n_subfolders)

		# Phân cụm Agglomerative các subfolders
		Z = linkage(pca_emb, method="ward")
		cluster_labels = fcluster(Z, t=n_clusters, criterion="maxclust")

		cluster_ids = sorted(set(cluster_labels))
		clusters_info = []
		for cid in cluster_ids:
			mask = cluster_labels == cid
			comp_emb_pca = pca_emb[mask]
			comp_centroid = comp_emb_pca.mean(axis=0)
			# Tính khoảng cách Mahalanobis
			dist = _mahalanobis_dist_to_centroid(comp_centroid, class_centroid, cov_inv)[0]
			
			# Gom các subfolder names thuộc cụm này
			comp_sf_names = [subfolder_names[j] for j in range(n_subfolders) if cluster_labels[j] == cid]
			clusters_info.append((dist, comp_sf_names))

		# Sắp xếp các cụm theo khoảng cách Mahalanobis từ gần đến xa
		clusters_info.sort(key=lambda x: x[0])
		K = len(clusters_info)

		# Đảm bảo tập Train không bao giờ trống và xử lý số lượng cụm < 3
		if K < 3:
			indices = group.index.tolist()
			tr, va, te = _split_by_mahalanobis_image_level(
				indices, embeddings[np.array(indices)], train_ratio, val_ratio, seed, eps
			)
			train_idx.extend(tr)
			val_idx.extend(va)
			test_idx.extend(te)
			continue

		# >= 3 cụm
		# Chia thành 3 dải khoảng cách: Gần, Vừa, Xa
		near_num = max(1, K // 3)
		mid_num = max(1, (K - near_num) // 2)
		
		near_clusters = clusters_info[:near_num]
		mid_clusters = clusters_info[near_num:near_num + mid_num]
		far_clusters = clusters_info[near_num + mid_num:]

		# Biến theo dõi số lượng ảnh đã phân bổ
		curr_train, curr_val, curr_test = 0, 0, 0
		n_total = len(group)
		target_train = int(n_total * train_ratio)
		target_val = int(n_total * val_ratio)

		# Hàm phụ phân bổ các cụm subfolders trong một dải khoảng cách
		def allocate_band_clusters(band_clusters):
			nonlocal curr_train, curr_val, curr_test
			rng_local = random.Random(seed + hash(label))
			shuffled_band = band_clusters.copy()
			rng_local.shuffle(shuffled_band)

			for _, sf_list in shuffled_band:
				# Tính tổng số ảnh trong cụm subfolders này
				c_indices = []
				for sf_name in sf_list:
					c_indices.extend(subfolder_groups.get_group(sf_name).index.tolist())
				c_count = len(c_indices)

				# Quyết định đưa vào split nào đang thiếu nhiều nhất
				diff_tr = max(0, target_train - curr_train)
				diff_va = max(0, target_val - curr_val)
				diff_te = max(0, (n_total - target_train - target_val) - curr_test)

				total_diff = diff_tr + diff_va + diff_te
				if total_diff == 0:
					min_split = min(curr_train, curr_val, curr_test)
					if min_split == curr_train:
						train_idx.extend(c_indices)
						curr_train += c_count
					elif min_split == curr_val:
						val_idx.extend(c_indices)
						curr_val += c_count
					else:
						test_idx.extend(c_indices)
						curr_test += c_count
					continue

				max_diff = max(diff_tr, diff_va, diff_te)
				if max_diff == diff_tr:
					train_idx.extend(c_indices)
					curr_train += c_count
				elif max_diff == diff_va:
					val_idx.extend(c_indices)
					curr_val += c_count
				else:
					test_idx.extend(c_indices)
					curr_test += c_count

		# Phân bổ cho từng dải
		allocate_band_clusters(near_clusters)
		allocate_band_clusters(mid_clusters)
		allocate_band_clusters(far_clusters)

	df_train = _shuffle_df(df.loc[train_idx], seed)
	df_val = _shuffle_df(df.loc[val_idx], seed)
	df_test = _shuffle_df(df.loc[test_idx], seed)
	return df_train, df_val, df_test


# ============================================================
# PP7: Adversarial Validation Split
# ============================================================

def adversarial_validation_split(
	df: pd.DataFrame,
	embeddings: np.ndarray,
	train_ratio: float = 0.60,
	val_ratio: float = 0.20,
	seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	"""
	PP7: Adversarial Validation — train discriminator trên subfolder embeddings để phân biệt 2 pool subfolders.
	Những subfolders mà discriminator tự tin nhất là "khác biệt" → đưa vào test.
	Đảm bảo KHÔNG rò rỉ dữ liệu mức subfolder.
	"""
	if len(df) != embeddings.shape[0]:
		raise ValueError("Embeddings length does not match dataframe length")

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	rng = np.random.RandomState(seed)

	train_idx, val_idx, test_idx = [], [], []

	for label, group in tqdm(df.groupby("label"), desc="PP7-Adversarial"):
		subfolder_groups = group.groupby("subfolder")
		subfolder_names = list(subfolder_groups.groups.keys())
		n_subfolders = len(subfolder_names)

		if n_subfolders == 0:
			continue

		if n_subfolders < 3:
			indices = group.index.tolist()
			tr, va, te = _split_by_mahalanobis_image_level(
				indices, embeddings[np.array(indices)], train_ratio, val_ratio, seed
			)
			train_idx.extend(tr)
			val_idx.extend(va)
			test_idx.extend(te)
			continue

		# Tính subfolder embeddings
		subfolder_embs = []
		for sf_name in subfolder_names:
			sf_indices = subfolder_groups.get_group(sf_name).index.tolist()
			subfolder_embs.append(embeddings[sf_indices].mean(axis=0))
		subfolder_embs = np.array(subfolder_embs)

		# Chia tạm 50/50 các subfolders thành pool_A (label=0) và pool_B (label=1)
		perm = rng.permutation(n_subfolders)
		half = n_subfolders // 2
		pool_a_pos = perm[:half]
		pool_b_pos = perm[half:]

		X = subfolder_embs.copy()
		y = np.zeros(n_subfolders, dtype=np.float32)
		y[pool_b_pos] = 1.0

		# Train MLP nhỏ để phân biệt pool_A vs pool_B ở mức subfolder
		emb_dim = X.shape[1]
		discriminator = nn.Sequential(
			nn.Linear(emb_dim, 128),
			nn.ReLU(),
			nn.Dropout(0.3),
			nn.Linear(128, 1),
			nn.Sigmoid(),
		).to(device)

		X_tensor = torch.from_numpy(X).float().to(device)
		y_tensor = torch.from_numpy(y).float().to(device)

		optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-3)
		criterion = nn.BCELoss()

		# Training nhanh (30 epochs)
		discriminator.train()
		for _ in range(30):
			optimizer.zero_grad()
			pred = discriminator(X_tensor).squeeze()
			loss = criterion(pred, y_tensor)
			loss.backward()
			optimizer.step()

		# Lấy prediction score
		discriminator.eval()
		with torch.no_grad():
			scores = discriminator(X_tensor).squeeze().cpu().numpy()

		# Score = |pred - 0.5|: càng cao = discriminator càng tự tin phân biệt
		# Subfolder "khác biệt nhất" → đưa vào test
		difficulty_scores = np.abs(scores - 0.5)
		sorted_positions = np.argsort(-difficulty_scores)

		train_sf_count, val_sf_count, test_sf_count = compute_split_counts(n_subfolders, train_ratio, val_ratio)

		# Phân bổ subfolders theo độ khó
		for i, pos in enumerate(sorted_positions):
			sf_name = subfolder_names[pos]
			sf_indices = subfolder_groups.get_group(sf_name).index.tolist()
			if i < test_sf_count:
				test_idx.extend(sf_indices)
			elif i < test_sf_count + val_sf_count:
				val_idx.extend(sf_indices)
			else:
				train_idx.extend(sf_indices)

	df_train = _shuffle_df(df.loc[train_idx], seed)
	df_val = _shuffle_df(df.loc[val_idx], seed)
	df_test = _shuffle_df(df.loc[test_idx], seed)
	return df_train, df_val, df_test


# ============================================================
# PP8: StratifiedGroupKFold Split
# ============================================================

def stratified_group_kfold_split(
	df: pd.DataFrame,
	embeddings: np.ndarray,
	train_ratio: float = 0.60,
	val_ratio: float = 0.20,
	seed: int = 42,
	eps: float = 1e-6,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	"""
	PP8: StratifiedGroupKFold — Cân bằng cách ly nhóm subfolder và phân tầng class.
	Đối với các class có ít hơn 3 subfolders, ta phân rã ảnh ra và chia theo khoảng cách Mahalanobis mức ảnh.
	Đối với các class có từ 3 subfolders trở lên, ta chia bằng StratifiedGroupKFold ở mức group.
	"""
	from sklearn.model_selection import StratifiedGroupKFold as SKF_Splitter

	train_idx, val_idx, test_idx = [], [], []

	# Tách dataframe thành các class lớn (>= 3 subfolders) và class nhỏ (< 3 subfolders)
	large_class_groups = []
	
	for label, group in df.groupby("label"):
		subfolder_names = group["subfolder"].unique()
		n_subfolders = len(subfolder_names)
		
		if n_subfolders < 3:
			# Class nhỏ: phân rã ảnh và chia theo Mahalanobis mức ảnh
			indices = group.index.tolist()
			tr, va, te = _split_by_mahalanobis_image_level(
				indices, embeddings[np.array(indices)], train_ratio, val_ratio, seed, eps
			)
			train_idx.extend(tr)
			val_idx.extend(va)
			test_idx.extend(te)
		else:
			large_class_groups.append(group)

	if large_class_groups:
		# Gộp các class lớn lại để chạy StratifiedGroupKFold
		df_large = pd.concat(large_class_groups)
		
		# Tính n_splits sao cho mỗi fold ≈ test_ratio (VD: 0.20 → 5 folds)
		test_ratio = 1.0 - train_ratio - val_ratio
		n_splits_test = max(3, int(round(1.0 / test_ratio)))

		# Tạo group array từ subfolder cho df_large
		groups = (df_large["label"] + "___" + df_large["subfolder"]).values
		labels = df_large["label"].values

		# Bước 1: Tách test fold bằng StratifiedGroupKFold
		sgkf_test = SKF_Splitter(n_splits=n_splits_test, shuffle=False, random_state=None)

		trainval_indices_rel = None
		test_indices_rel = None
		for tv_idx, te_idx in sgkf_test.split(df_large.index, labels, groups):
			trainval_indices_rel = tv_idx
			test_indices_rel = te_idx
			break

		# Lấy indices tuyệt đối của df gốc
		trainval_absolute_indices = df_large.index[trainval_indices_rel].tolist()
		test_absolute_indices = df_large.index[test_indices_rel].tolist()
		
		df_trainval = df.loc[trainval_absolute_indices]
		test_idx.extend(test_absolute_indices)

		# Bước 2: Tách val từ trainval bằng StratifiedGroupKFold
		val_fraction = val_ratio / (train_ratio + val_ratio)
		n_splits_val = max(3, int(round(1.0 / val_fraction)))

		groups_tv = (df_trainval["label"] + "___" + df_trainval["subfolder"]).values
		labels_tv = df_trainval["label"].values

		sgkf_val = SKF_Splitter(n_splits=n_splits_val, shuffle=False, random_state=None)

		train_indices_rel = None
		val_indices_rel = None
		for tr_idx, va_idx in sgkf_val.split(df_trainval.index, labels_tv, groups_tv):
			train_indices_rel = tr_idx
			val_indices_rel = va_idx
			break

		train_absolute_indices = df_trainval.index[train_indices_rel].tolist()
		val_absolute_indices = df_trainval.index[val_indices_rel].tolist()
		
		train_idx.extend(train_absolute_indices)
		val_idx.extend(val_absolute_indices)

	df_train = _shuffle_df(df.loc[train_idx], seed)
	df_val = _shuffle_df(df.loc[val_idx], seed)
	df_test = _shuffle_df(df.loc[test_idx], seed)

	return df_train, df_val, df_test


# ============================================================
# Registry: danh sách tất cả phương pháp
# ============================================================

SPLIT_METHODS = {
	"PP2_Mahalanobis_Iterative": mahalanobis_iterative_split,
	"PP4_Hierarchical_Clustering": hierarchical_clustering_split,
	"PP5_Cosine_Graph": cosine_graph_split,
	"PP7_Adversarial_Validation": adversarial_validation_split,
	"PP8_StratifiedGroupKFold": stratified_group_kfold_split,
	"PP9_Agglom_Stratified": agglom_stratified_split,
}