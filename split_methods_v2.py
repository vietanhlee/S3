"""
split_methods_v2.py
===================
8 phương pháp chia dữ liệu ở cấp độ ảnh đơn lẻ (image-level) chống rò rỉ (hoặc baseline),
được nâng cấp sử dụng Elbow Method (1-30 cụm) và khoảng cách Mahalanobis (PCA 128 chiều).
"""

import random
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
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
# Helper: khoảng cách Mahalanobis
# ============================================================

def _mahalanobis_distances(embeddings: np.ndarray, eps: float = 1e-6) -> np.ndarray:
	"""Tính khoảng cách Mahalanobis từ mỗi điểm đến centroid cố định của tập."""
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


# ============================================================
# Helper: Elbow Method tìm n_clusters tối ưu (1 đến 30)
# ============================================================

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
		
	# Tìm điểm khuỷu tay (elbow point) bằng khoảng cách xa nhất đến đường thẳng nối điểm đầu và điểm cuối
	x1, y1 = k_values[0], wcss[0]
	x2, y2 = k_values[-1], wcss[-1]
	
	distances = []
	for i in range(len(k_values)):
		x0, y0 = k_values[i], wcss[i]
		numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
		denominator = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
		distances.append(numerator / denominator if denominator > 0 else 0.0)
		
	optimal_k = k_values[np.argmax(distances)]
	return max(3, optimal_k)


# ============================================================
# PP2: Mahalanobis Iterative Centroid (Image-level, PCA 128 chiều)
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
	PP2: Tính khoảng cách Mahalanobis và TÁI TÍNH centroid sau mỗi lần rút ảnh đơn lẻ.
	Được tối ưu hóa bằng PCA giảm chiều xuống tối đa 128 chiều.
	"""
	if len(df) != embeddings.shape[0]:
		raise ValueError("Embeddings length does not match dataframe length")

	train_idx, val_idx, test_idx = [], [], []

	for _, group in tqdm(df.groupby("label"), desc="PP2-MahalIter"):
		indices = sorted(group.index.tolist())
		n_total = len(indices)
		train_count, val_count, test_count = compute_split_counts(n_total, train_ratio, val_ratio)

		if n_total == 0:
			continue

		subset_emb = embeddings[np.array(indices)]
		# Áp dụng PCA giảm chiều xuống tối đa 128 chiều
		if n_total >= 5:
			d_prime = min(n_total - 2, 128)
			if d_prime >= 2:
				pca = PCA(n_components=d_prime, random_state=seed)
				subset_emb = pca.fit_transform(subset_emb)

		remaining_pos = list(range(n_total))
		picked_test_pos = []
		picked_val_pos = []

		# Rút test
		for _ in range(min(test_count, len(remaining_pos))):
			cur_embs = subset_emb[remaining_pos]
			dists = _mahalanobis_distances(cur_embs, eps=eps)
			max_pos = int(np.argmax(dists))
			picked_test_pos.append(remaining_pos[max_pos])
			remaining_pos.pop(max_pos)

		# Rút val
		for _ in range(min(val_count, len(remaining_pos))):
			cur_embs = subset_emb[remaining_pos]
			dists = _mahalanobis_distances(cur_embs, eps=eps)
			max_pos = int(np.argmax(dists))
			picked_val_pos.append(remaining_pos[max_pos])
			remaining_pos.pop(max_pos)

		test_idx.extend([indices[pos] for pos in picked_test_pos])
		val_idx.extend([indices[pos] for pos in picked_val_pos])
		train_idx.extend([indices[pos] for pos in remaining_pos])

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
	Triết lý cách ly nguồn mẫu vật lý.
	"""
	rng = random.Random(seed)
	train_idx, val_idx, test_idx = [], [], []

	for _, group in tqdm(df.groupby("label"), desc="PP3-GroupBased"):
		subfolder_groups = group.groupby("subfolder")
		subfolder_names = list(subfolder_groups.groups.keys())
		rng.shuffle(subfolder_names)

		n_total = len(group)

		# Phân bổ nguyên vẹn các subfolders
		if len(subfolder_names) == 1:
			train_idx.extend(subfolder_groups.get_group(subfolder_names[0]).index.tolist())
			continue

		if len(subfolder_names) == 2:
			g0 = subfolder_groups.get_group(subfolder_names[0]).index.tolist()
			g1 = subfolder_groups.get_group(subfolder_names[1]).index.tolist()
			if len(g0) >= len(g1):
				train_idx.extend(g0)
				test_idx.extend(g1)
			else:
				train_idx.extend(g1)
				test_idx.extend(g0)
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
# PP4: Hierarchical Clustering (Ward, Image-level, Elbow, Mahalanobis)
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
	PP4: Agglomerative Clustering (Ward) trên từng ảnh đơn lẻ.
	Xác định số cụm tối ưu qua Elbow Method (1-30).
	Tính khoảng cách centroid cụm đến centroid chung qua khoảng cách Mahalanobis (PCA 128 chiều).
	Gán nguyên cụm ảnh vào test/val/train (xấp xỉ tỷ lệ).
	"""
	if len(df) != embeddings.shape[0]:
		raise ValueError("Embeddings length does not match dataframe length")

	train_idx, val_idx, test_idx = [], [], []

	for _, group in tqdm(df.groupby("label"), desc="PP4-HierClust"):
		indices = sorted(group.index.tolist())
		n_total = len(indices)

		if n_total == 0:
			continue

		if n_total <= 3:
			# Nếu quá ít ảnh, gán toàn bộ vào Train
			train_idx.extend(indices)
			continue

		subset_emb = embeddings[np.array(indices)]
		
		# PCA 128 chiều trên class embeddings để tính covariance nghịch đảo ổn định
		pca_emb = subset_emb
		if n_total >= 5:
			d_prime = min(n_total - 2, 128)
			if d_prime >= 2:
				pca = PCA(n_components=d_prime, random_state=seed)
				pca_emb = pca.fit_transform(subset_emb)

		# Tính ma trận hiệp phương sai nghịch đảo của lớp
		cov = np.cov(pca_emb, rowvar=False)
		cov = np.atleast_2d(cov)
		cov += np.eye(cov.shape[0]) * eps
		cov_inv = np.linalg.pinv(cov)
		global_centroid = pca_emb.mean(axis=0)

		# Xác định số cụm bằng Elbow Method
		n_clusters = find_optimal_clusters_elbow(pca_emb, max_k=30, seed=seed)
		n_clusters = min(n_clusters, n_total)

		# Phân cụm Agglomerative
		Z = linkage(pca_emb, method="ward")
		cluster_labels = fcluster(Z, t=n_clusters, criterion="maxclust")

		cluster_ids = sorted(set(cluster_labels))
		cluster_info = []
		for cid in cluster_ids:
			mask = cluster_labels == cid
			cluster_centroid = pca_emb[mask].mean(axis=0)
			# Tính khoảng cách Mahalanobis từ centroid cụm đến global centroid
			dist = _mahalanobis_dist_to_centroid(cluster_centroid, global_centroid, cov_inv)[0]
			cluster_indices = [indices[j] for j in range(n_total) if cluster_labels[j] == cid]
			cluster_info.append((cid, dist, cluster_indices))

		# Sắp xếp cụm theo khoảng cách Mahalanobis giảm dần (ngoại lai lên trước)
		cluster_info.sort(key=lambda x: -x[1])

		# Phân bổ nguyên cụm (không chia cắt)
		if len(cluster_info) == 1:
			train_idx.extend(cluster_info[0][2])
			continue

		if len(cluster_info) == 2:
			g0 = cluster_info[0][2]
			g1 = cluster_info[1][2]
			if len(g0) >= len(g1):
				train_idx.extend(g0)
				test_idx.extend(g1)
			else:
				train_idx.extend(g1)
				test_idx.extend(g0)
			continue

		target_train = int(n_total * train_ratio)
		target_val = int(n_total * val_ratio)

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
# PP5: Cosine Graph (Connected Components, Image-level, Mahalanobis)
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
	PP5: Connected Components trên đồ thị tương đồng cosine của ảnh đơn lẻ.
	Sắp xếp components bằng khoảng cách Mahalanobis đến centroid chung (PCA 128 chiều).
	Gán nguyên vẹn components (không chia cắt) để tránh rò rỉ ảnh gần như giống hệt.
	"""
	if len(df) != embeddings.shape[0]:
		raise ValueError("Embeddings length does not match dataframe length")

	train_idx, val_idx, test_idx = [], [], []

	for _, group in tqdm(df.groupby("label"), desc="PP5-CosGraph"):
		indices = sorted(group.index.tolist())
		n_total = len(indices)

		if n_total == 0:
			continue

		if n_total <= 3:
			train_idx.extend(indices)
			continue

		subset_emb = embeddings[np.array(indices)]

		# PCA 128 chiều trên class embeddings để tính covariance nghịch đảo ổn định
		pca_emb = subset_emb
		if n_total >= 5:
			d_prime = min(n_total - 2, 128)
			if d_prime >= 2:
				pca = PCA(n_components=d_prime, random_state=seed)
				pca_emb = pca.fit_transform(subset_emb)

		cov = np.cov(pca_emb, rowvar=False)
		cov = np.atleast_2d(cov)
		cov += np.eye(cov.shape[0]) * eps
		cov_inv = np.linalg.pinv(cov)
		global_centroid = pca_emb.mean(axis=0)

		# Tính cosine similarity trên raw embeddings
		sim_matrix = cosine_similarity(subset_emb)
		np.fill_diagonal(sim_matrix, 0.0)

		# Xây adjacency matrix
		adj = (sim_matrix >= cosine_threshold).astype(np.float32)
		sparse_adj = csr_matrix(adj)

		# Tìm connected components
		n_components, component_labels = connected_components(sparse_adj, directed=False)

		components = []
		for cid in range(n_components):
			mask = component_labels == cid
			comp_emb = pca_emb[mask]
			comp_centroid = comp_emb.mean(axis=0)
			# Tính khoảng cách Mahalanobis từ centroid component đến global centroid
			dist = _mahalanobis_dist_to_centroid(comp_centroid, global_centroid, cov_inv)[0]
			comp_indices = [indices[j] for j in range(n_total) if component_labels[j] == cid]
			components.append((cid, dist, comp_indices))

		# Sắp xếp components theo khoảng cách Mahalanobis giảm dần
		components.sort(key=lambda x: -x[1])

		# Phân bổ nguyên vẹn các components (không chia cắt)
		if len(components) == 1:
			train_idx.extend(components[0][2])
			continue

		if len(components) == 2:
			g0 = components[0][2]
			g1 = components[1][2]
			if len(g0) >= len(g1):
				train_idx.extend(g0)
				test_idx.extend(g1)
			else:
				train_idx.extend(g1)
				test_idx.extend(g0)
			continue

		target_train = int(n_total * train_ratio)
		target_val = int(n_total * val_ratio)

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
# PP6: Stratified Random Split (Baseline, Image-level)
# ============================================================

def stratified_random_split(
	df: pd.DataFrame,
	embeddings: np.ndarray,  # Không dùng, giữ signature nhất quán
	train_ratio: float = 0.60,
	val_ratio: float = 0.20,
	seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	"""
	PP6: Baseline — chia ngẫu nhiên giữ tỉ lệ class ở mức ảnh đơn lẻ.
	Được viết thủ công để loại bỏ hoàn toàn lỗi crash khi class có ít mẫu.
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


# ============================================================
# PP7: Adversarial Validation (Image-level)
# ============================================================

def adversarial_validation_split(
	df: pd.DataFrame,
	embeddings: np.ndarray,
	train_ratio: float = 0.60,
	val_ratio: float = 0.20,
	seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	"""
	PP7: Adversarial Validation ở mức ảnh đơn lẻ.
	Chạy discriminator phân biệt 2 pool ảnh để tìm ảnh lệch phân phối đưa vào test.
	"""
	if len(df) != embeddings.shape[0]:
		raise ValueError("Embeddings length does not match dataframe length")

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	rng = np.random.RandomState(seed)

	train_idx, val_idx, test_idx = [], [], []

	for label, group in tqdm(df.groupby("label"), desc="PP7-Adversarial"):
		indices = sorted(group.index.tolist())
		n_total = len(indices)
		train_count, val_count, test_count = compute_split_counts(n_total, train_ratio, val_ratio)

		if n_total <= 5:
			# Quá ít mẫu, chia ngẫu nhiên
			train_idx.extend(indices[:train_count])
			val_idx.extend(indices[train_count:train_count + val_count])
			test_idx.extend(indices[train_count + val_count:])
			continue

		subset_emb = embeddings[np.array(indices)]
		n_samples = len(indices)

		# Chia tạm 50/50 thành pool_A (label=0) và pool_B (label=1)
		perm = rng.permutation(n_samples)
		half = n_samples // 2
		pool_a_pos = perm[:half]
		pool_b_pos = perm[half:]

		X = subset_emb.copy()
		y = np.zeros(n_samples, dtype=np.float32)
		y[pool_b_pos] = 1.0

		# Train MLP nhỏ để phân biệt pool_A vs pool_B
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

		discriminator.train()
		for _ in range(30):
			optimizer.zero_grad()
			pred = discriminator(X_tensor).squeeze()
			loss = criterion(pred, y_tensor)
			loss.backward()
			optimizer.step()

		discriminator.eval()
		with torch.no_grad():
			scores = discriminator(X_tensor).squeeze().cpu().numpy()

		# Ảnh "khác biệt nhất" = dễ phân biệt nhất → đưa vào test
		difficulty_scores = np.abs(scores - 0.5)
		sorted_positions = np.argsort(-difficulty_scores)

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
# PP8: StratifiedGroupKFold Split (Subfolder)
# ============================================================

def stratified_group_kfold_split(
	df: pd.DataFrame,
	embeddings: np.ndarray,  # Không dùng, giữ signature nhất quán
	train_ratio: float = 0.60,
	val_ratio: float = 0.20,
	seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	"""
	PP8: StratifiedGroupKFold — Cân bằng cách ly nhóm subfolder và phân tầng class.
	"""
	from sklearn.model_selection import StratifiedGroupKFold as SKF_Splitter

	test_ratio = 1.0 - train_ratio - val_ratio
	n_splits_test = max(3, int(round(1.0 / test_ratio)))

	groups = (df["label"] + "___" + df["subfolder"]).values
	labels = df["label"].values

	sgkf_test = SKF_Splitter(n_splits=n_splits_test, shuffle=False, random_state=None)

	trainval_indices = None
	test_indices = None
	for tv_idx, te_idx in sgkf_test.split(df.index, labels, groups):
		trainval_indices = tv_idx
		test_indices = te_idx
		break

	df_trainval = df.iloc[trainval_indices].reset_index(drop=True)
	df_test_raw = df.iloc[test_indices]

	val_fraction = val_ratio / (train_ratio + val_ratio)
	n_splits_val = max(3, int(round(1.0 / val_fraction)))

	groups_tv = (df_trainval["label"] + "___" + df_trainval["subfolder"]).values
	labels_tv = df_trainval["label"].values

	sgkf_val = SKF_Splitter(n_splits=n_splits_val, shuffle=False, random_state=None)

	train_indices_final = None
	val_indices_final = None
	for tr_idx, va_idx in sgkf_val.split(df_trainval.index, labels_tv, groups_tv):
		train_indices_final = tr_idx
		val_indices_final = va_idx
		break

	df_train = df_trainval.iloc[train_indices_final].reset_index(drop=True)
	df_val = df_trainval.iloc[val_indices_final].reset_index(drop=True)
	df_test = df_test_raw.reset_index(drop=True)

	return df_train, df_val, df_test


# ============================================================
# PP9: Agglom Stratified (Image-level, Elbow, Mahalanobis)
# ============================================================

def agglom_stratified_split(
	df: pd.DataFrame,
	embeddings: np.ndarray,
	train_ratio: float = 0.60,
	val_ratio: float = 0.20,
	seed: int = 42,
	eps: float = 1e-6,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	"""
	PP9: Phân cụm Agglomerative các ảnh đơn lẻ của từng class.
	Xác định số cụm bằng Elbow Method (1-30).
	Tính khoảng cách centroid cụm đến centroid chung bằng khoảng cách Mahalanobis (PCA 128 chiều).
	Chia cụm ảnh thành 3 dải (Gần, Vừa, Xa), phân bổ đều các cụm vào Train/Val/Test.
	"""
	if len(df) != embeddings.shape[0]:
		raise ValueError("Embeddings length does not match dataframe length")

	train_idx, val_idx, test_idx = [], [], []

	for label, group in tqdm(df.groupby("label"), desc="PP9-AgglomStratified"):
		indices = sorted(group.index.tolist())
		n_total = len(indices)

		if n_total == 0:
			continue

		if n_total <= 5:
			train_count, val_count, test_count = compute_split_counts(n_total, train_ratio, val_ratio)
			rng_local = random.Random(seed + hash(label))
			shuffled = indices.copy()
			rng_local.shuffle(shuffled)
			train_idx.extend(shuffled[:train_count])
			val_idx.extend(shuffled[train_count:train_count + val_count])
			test_idx.extend(shuffled[train_count + val_count:])
			continue

		subset_emb = embeddings[np.array(indices)]

		# PCA 128 chiều trên class embeddings để tính covariance nghịch đảo ổn định
		pca_emb = subset_emb
		if n_total >= 5:
			d_prime = min(n_total - 2, 128)
			if d_prime >= 2:
				pca = PCA(n_components=d_prime, random_state=seed)
				pca_emb = pca.fit_transform(subset_emb)

		cov = np.cov(pca_emb, rowvar=False)
		cov = np.atleast_2d(cov)
		cov += np.eye(cov.shape[0]) * eps
		cov_inv = np.linalg.pinv(cov)
		class_centroid = pca_emb.mean(axis=0)

		# Xác định số cụm bằng Elbow Method
		n_clusters = find_optimal_clusters_elbow(pca_emb, max_k=30, seed=seed)
		n_clusters = min(n_clusters, n_total)

		# Phân cụm Agglomerative
		Z = linkage(pca_emb, method="ward")
		cluster_labels = fcluster(Z, t=n_clusters, criterion="maxclust")

		cluster_ids = sorted(set(cluster_labels))
		clusters_info = []
		for cid in cluster_ids:
			mask = cluster_labels == cid
			comp_emb = pca_emb[mask]
			comp_centroid = comp_emb.mean(axis=0)
			# Tính khoảng cách Mahalanobis từ centroid cụm đến global centroid
			dist = _mahalanobis_dist_to_centroid(comp_centroid, class_centroid, cov_inv)[0]
			comp_indices = [indices[j] for j in range(n_total) if cluster_labels[j] == cid]
			clusters_info.append((dist, comp_indices))

		# Sắp xếp các cụm theo khoảng cách Mahalanobis từ gần đến xa
		clusters_info.sort(key=lambda x: x[0])
		K = len(clusters_info)

		# Chia thành 3 dải khoảng cách: Gần, Vừa, Xa
		near_num = max(1, K // 3)
		mid_num = max(1, (K - near_num) // 2)
		
		near_clusters = clusters_info[:near_num]
		mid_clusters = clusters_info[near_num:near_num + mid_num]
		far_clusters = clusters_info[near_num + mid_num:]

		# Biến theo dõi số lượng ảnh đã phân bổ
		curr_train, curr_val, curr_test = 0, 0, 0
		target_train = int(n_total * train_ratio)
		target_val = int(n_total * val_ratio)

		# Hàm phụ phân bổ các cụm trong một dải khoảng cách
		def allocate_band_clusters(band_clusters):
			nonlocal curr_train, curr_val, curr_test
			rng_local = random.Random(seed + hash(label))
			shuffled_band = band_clusters.copy()
			rng_local.shuffle(shuffled_band)

			for _, c_indices in shuffled_band:
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
# Registry: danh sách tất cả phương pháp
# ============================================================

SPLIT_METHODS = {
	"PP2_Mahalanobis_Iterative": mahalanobis_iterative_split,
	"PP3_Group_Based": group_based_split,
	"PP4_Hierarchical_Clustering": hierarchical_clustering_split,
	"PP5_Cosine_Graph": cosine_graph_split,
	"PP6_Stratified_Random": stratified_random_split,
	"PP7_Adversarial_Validation": adversarial_validation_split,
	"PP8_StratifiedGroupKFold": stratified_group_kfold_split,
	"PP9_Agglom_Stratified": agglom_stratified_split,
}
