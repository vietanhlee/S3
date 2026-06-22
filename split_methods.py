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
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity


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
	PP2: Tính khoảng cách Mahalanobis, nhưng TÁI TÍNH centroid sau mỗi lần rút ảnh.
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

		remaining = list(indices)
		picked_test = []
		picked_val = []

		# Rút test: lặp, mỗi lần tính lại centroid và rút ảnh xa nhất
		for _ in range(min(test_count, len(remaining))):
			subset_emb = embeddings[np.array(remaining)]
			dists = _mahalanobis_distances(subset_emb, eps=eps)
			max_pos = int(np.argmax(dists))
			picked_test.append(remaining[max_pos])
			remaining.pop(max_pos)

		# Rút val: tương tự
		for _ in range(min(val_count, len(remaining))):
			subset_emb = embeddings[np.array(remaining)]
			dists = _mahalanobis_distances(subset_emb, eps=eps)
			max_pos = int(np.argmax(dists))
			picked_val.append(remaining[max_pos])
			remaining.pop(max_pos)

		test_idx.extend(picked_test)
		val_idx.extend(picked_val)
		train_idx.extend(remaining)

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
		target_train = int(n_total * train_ratio)
		target_val = int(n_total * val_ratio)

		current_train, current_val = 0, 0

		for sf_name in subfolder_names:
			sf_indices = subfolder_groups.get_group(sf_name).index.tolist()
			sf_count = len(sf_indices)

			if current_train < target_train:
				train_idx.extend(sf_indices)
				current_train += sf_count
			elif current_val < target_val:
				val_idx.extend(sf_indices)
				current_val += sf_count
			else:
				test_idx.extend(sf_indices)

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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	"""
	PP4: Agglomerative Clustering (Ward) trên embedding.
	Gán nguyên cụm vào test/val/train, ưu tiên cụm xa centroid chung nhất cho test.
	"""
	if len(df) != embeddings.shape[0]:
		raise ValueError("Embeddings length does not match dataframe length")

	train_idx, val_idx, test_idx = [], [], []

	for label, group in tqdm(df.groupby("label"), desc="PP4-HierClust"):
		indices = sorted(group.index.tolist())
		n_total = len(indices)
		train_count, val_count, test_count = compute_split_counts(n_total, train_ratio, val_ratio)

		if n_total <= 3:
			# Quá ít mẫu, chia đơn giản
			train_idx.extend(indices[:train_count])
			val_idx.extend(indices[train_count:train_count + val_count])
			test_idx.extend(indices[train_count + val_count:])
			continue

		subset_emb = embeddings[np.array(indices)]

		# Số clusters = max(3, số subfolder)
		n_subfolders = group["subfolder"].nunique()
		n_clusters = max(3, n_subfolders)
		n_clusters = min(n_clusters, n_total)

		# Agglomerative clustering
		Z = linkage(subset_emb, method="ward")
		cluster_labels = fcluster(Z, t=n_clusters, criterion="maxclust")

		# Tính centroid chung + khoảng cách mỗi cluster đến centroid
		global_centroid = subset_emb.mean(axis=0)
		cluster_ids = sorted(set(cluster_labels))
		cluster_info = []
		for cid in cluster_ids:
			mask = cluster_labels == cid
			cluster_centroid = subset_emb[mask].mean(axis=0)
			dist_to_global = float(np.linalg.norm(cluster_centroid - global_centroid))
			cluster_indices = [indices[j] for j in range(n_total) if cluster_labels[j] == cid]
			cluster_info.append((cid, dist_to_global, cluster_indices))

		# Sort: cụm xa nhất trước
		cluster_info.sort(key=lambda x: -x[1])

		# Gán: cụm xa nhất → test, tiếp → val, còn lại → train
		current_test, current_val = 0, 0
		for _, _, c_indices in cluster_info:
			c_count = len(c_indices)
			if current_test < test_count:
				test_idx.extend(c_indices)
				current_test += c_count
			elif current_val < val_count:
				val_idx.extend(c_indices)
				current_val += c_count
			else:
				train_idx.extend(c_indices)

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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	"""
	PP5: Xây đồ thị cosine similarity, tìm connected components, chia theo component.
	Mỗi component = 1 đơn vị không chia cắt.
	"""
	if len(df) != embeddings.shape[0]:
		raise ValueError("Embeddings length does not match dataframe length")

	rng = random.Random(seed)
	train_idx, val_idx, test_idx = [], [], []

	for label, group in tqdm(df.groupby("label"), desc="PP5-CosGraph"):
		indices = sorted(group.index.tolist())
		n_total = len(indices)
		train_count, val_count, test_count = compute_split_counts(n_total, train_ratio, val_ratio)

		if n_total <= 3:
			train_idx.extend(indices[:train_count])
			val_idx.extend(indices[train_count:train_count + val_count])
			test_idx.extend(indices[train_count + val_count:])
			continue

		subset_emb = embeddings[np.array(indices)]

		# Tính cosine similarity matrix
		sim_matrix = cosine_similarity(subset_emb)
		np.fill_diagonal(sim_matrix, 0.0)

		# Xây adjacency matrix (sparse)
		adj = (sim_matrix >= cosine_threshold).astype(np.float32)
		sparse_adj = csr_matrix(adj)

		# Tìm connected components
		n_components, component_labels = connected_components(sparse_adj, directed=False)

		# Gom thành danh sách component
		components = []
		for cid in range(n_components):
			mask = component_labels == cid
			comp_indices = [indices[j] for j in range(n_total) if component_labels[j] == cid]
			# Tính khoảng cách centroid component đến centroid chung
			comp_emb = subset_emb[mask]
			comp_centroid = comp_emb.mean(axis=0)
			global_centroid = subset_emb.mean(axis=0)
			dist = float(np.linalg.norm(comp_centroid - global_centroid))
			components.append((cid, dist, comp_indices))

		# Sort: component xa nhất trước
		components.sort(key=lambda x: -x[1])

		# Gán
		current_test, current_val = 0, 0
		for _, _, c_indices in components:
			c_count = len(c_indices)

			# Nếu component quá lớn (> 50% class), cho phép chia cắt
			if c_count > n_total * 0.5:
				rng_local = random.Random(seed)
				rng_local.shuffle(c_indices)
				needed_test = max(0, test_count - current_test)
				needed_val = max(0, val_count - current_val)
				test_idx.extend(c_indices[:needed_test])
				val_idx.extend(c_indices[needed_test:needed_test + needed_val])
				train_idx.extend(c_indices[needed_test + needed_val:])
				current_test += needed_test
				current_val += needed_val
				continue

			if current_test < test_count:
				test_idx.extend(c_indices)
				current_test += c_count
			elif current_val < val_count:
				val_idx.extend(c_indices)
				current_val += c_count
			else:
				train_idx.extend(c_indices)

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
	PP6: Baseline — chia ngẫu nhiên giữ tỉ lệ class bằng sklearn stratified split.
	Không chống leakage, dùng để so sánh.
	"""
	test_ratio = 1.0 - train_ratio - val_ratio

	# Bước 1: tách test
	df_trainval, df_test = train_test_split(
		df,
		test_size=test_ratio,
		stratify=df["label"],
		random_state=seed,
	)

	# Bước 2: tách val từ trainval
	val_fraction = val_ratio / (train_ratio + val_ratio)
	df_train, df_val = train_test_split(
		df_trainval,
		test_size=val_fraction,
		stratify=df_trainval["label"],
		random_state=seed,
	)

	df_train = df_train.reset_index(drop=True)
	df_val = df_val.reset_index(drop=True)
	df_test = df_test.reset_index(drop=True)
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
	PP7: Adversarial Validation — train discriminator trên embedding để phân biệt 2 pool.
	Ảnh mà discriminator tự tin nhất là "khác biệt" → đưa vào test.
	Tạo test set khó nhất có thể.
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
			# Quá ít mẫu, chia đơn giản
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
		# Ảnh "khác biệt nhất" = dễ phân biệt nhất → đưa vào test
		difficulty_scores = np.abs(scores - 0.5)
		sorted_positions = np.argsort(-difficulty_scores)

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
# PP8: StratifiedGroupKFold Split
# ============================================================

def stratified_group_kfold_split(
	df: pd.DataFrame,
	embeddings: np.ndarray,  # Không dùng, giữ signature nhất quán
	train_ratio: float = 0.60,
	val_ratio: float = 0.20,
	seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	"""
	PP8: StratifiedGroupKFold — "tiêu chuẩn vàng" cho dữ liệu vi phẫu gỗ.

	Kết hợp 2 yếu tố:
	  - Group isolation: subfolder = group, KHÔNG bao giờ cắt subfolder
	  - Stratification: giữ tỉ lệ class đồng nhất giữa các fold

	Sử dụng sklearn.model_selection.StratifiedGroupKFold.
	Chọn n_splits sao cho test fold ≈ test_ratio, rồi tách val từ train fold.

	Tham khảo: Báo cáo 2.md — StratifiedGroupKFold là phương pháp tối ưu nhất
	khi dữ liệu vừa có cấu trúc nhóm (cùng mẫu gỗ) vừa mất cân bằng lớp.
	"""
	from sklearn.model_selection import StratifiedGroupKFold as SKF_Splitter

	test_ratio = 1.0 - train_ratio - val_ratio

	# Tính n_splits sao cho mỗi fold ≈ test_ratio (VD: 0.20 → 5 folds)
	n_splits_test = max(3, int(round(1.0 / test_ratio)))

	# Tạo group array từ subfolder
	# Mỗi (label, subfolder) là 1 group unique
	groups = (df["label"] + "___" + df["subfolder"]).values
	labels = df["label"].values

	# Bước 1: Tách test fold bằng StratifiedGroupKFold
	sgkf_test = SKF_Splitter(n_splits=n_splits_test, shuffle=False, random_state=None)

	# Lấy fold đầu tiên làm test
	trainval_indices = None
	test_indices = None
	for tv_idx, te_idx in sgkf_test.split(df.index, labels, groups):
		trainval_indices = tv_idx
		test_indices = te_idx
		break

	df_trainval = df.iloc[trainval_indices].reset_index(drop=True)
	df_test_raw = df.iloc[test_indices]

	# Bước 2: Tách val từ trainval bằng StratifiedGroupKFold
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
# Registry: danh sách tất cả phương pháp
# ============================================================

SPLIT_METHODS = {
	# "PP1_Mahalanobis_Fixed": mahalanobis_fixed_split,
	# "PP2_Mahalanobis_Iterative": mahalanobis_iterative_split,
	# "PP3_Group_Based": group_based_split,
	# "PP4_Hierarchical_Clustering": hierarchical_clustering_split,
	"PP5_Cosine_Graph": cosine_graph_split,
	"PP6_Stratified_Random": stratified_random_split,
	"PP7_Adversarial_Validation": adversarial_validation_split,
	"PP8_StratifiedGroupKFold": stratified_group_kfold_split,
}
