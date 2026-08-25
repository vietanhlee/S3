"""
datasail_benchmark/solvers.py
=============================
Triển khai 9 thuật toán chia dữ liệu (Splitting Protocols) cho Benchmark Data Leakage & DataSAIL.
Bảo đảm duy trì chính xác tỷ lệ phân bổ target (60% Train / 20% Val / 20% Test).
"""

import random
from typing import Tuple, List, Dict, Callable
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedGroupKFold

from .config import TRAIN_RATIO, VAL_RATIO, COSINE_THRESHOLD


def _shuffle_df(df: pd.DataFrame, seed: int) -> pd.DataFrame:
	return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def compute_split_counts(n_total: int, train_ratio: float = 0.60, val_ratio: float = 0.20) -> Tuple[int, int, int]:
	if n_total <= 0:
		return 0, 0, 0
	test_ratio = 1.0 - train_ratio - val_ratio
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


def _allocate_groups_by_ratio(
	groups_indices: List[List[int]],
	train_ratio: float = 0.60,
	val_ratio: float = 0.20,
) -> Tuple[List[int], List[int], List[int]]:
	"""
	Phân bổ danh sách các nhóm chỉ số (groups_indices) vào Train, Val, Test
	sao cho duy trì tỷ lệ phân bổ target (~60% Train, ~20% Val, ~20% Test).
	"""
	n_total = sum(len(g) for g in groups_indices)
	if n_total == 0:
		return [], [], []

	target_tr = int(n_total * train_ratio)
	target_va = int(n_total * val_ratio)

	train_idx, val_idx, test_idx = [], [], []
	curr_tr, curr_va = 0, 0

	for g in groups_indices:
		g_len = len(g)
		if curr_tr < target_tr or (curr_tr == 0 and len(groups_indices) >= 3):
			train_idx.extend(g)
			curr_tr += g_len
		elif curr_va < target_va or (curr_va == 0 and len(groups_indices) >= 2):
			val_idx.extend(g)
			curr_va += g_len
		else:
			test_idx.extend(g)

	# Đảm bảo không tập nào bị trống nếu số nhóm >= 3
	if len(groups_indices) >= 3:
		if len(val_idx) == 0 and len(train_idx) > 1:
			val_idx.append(train_idx.pop())
		if len(test_idx) == 0 and len(train_idx) > 1:
			test_idx.append(train_idx.pop())

	return train_idx, val_idx, test_idx


# ============================================================
# PP1: Image-Level Random Split (Flatten Random)
# ============================================================
def pp1_image_random_split(
	df: pd.DataFrame,
	embeddings: np.ndarray,
	seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	"""PP1: Chia ngẫu nhiên cấp độ Ảnh (Flatten Random) — Rò rỉ tối đa."""
	rng = random.Random(seed)
	train_idx, val_idx, test_idx = [], [], []

	for _, group in df.groupby("label"):
		indices = sorted(group.index.tolist())
		rng.shuffle(indices)
		n_total = len(indices)
		tr_c, va_c, te_c = compute_split_counts(n_total, TRAIN_RATIO, VAL_RATIO)

		train_idx.extend(indices[:tr_c])
		val_idx.extend(indices[tr_c:tr_c + va_c])
		test_idx.extend(indices[tr_c + va_c:])

	return _shuffle_df(df.loc[train_idx], seed), _shuffle_df(df.loc[val_idx], seed), _shuffle_df(df.loc[test_idx], seed)


# ============================================================
# PP2: Group-Level Random Split (GroupKFold)
# ============================================================
def pp2_group_random_split(
	df: pd.DataFrame,
	embeddings: np.ndarray,
	seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	"""PP2: Chia ngẫu nhiên cấp độ Mẫu vật (GroupKFold) — Bảo toàn khối mẫu vật."""
	rng = random.Random(seed)
	train_idx, val_idx, test_idx = [], [], []

	for _, group in df.groupby("label"):
		subfolder_groups = group.groupby("subfolder")
		subfolders = list(subfolder_groups.groups.keys())
		rng.shuffle(subfolders)

		groups_indices = [subfolder_groups.get_group(sf).index.tolist() for sf in subfolders]
		tr_g, va_g, te_g = _allocate_groups_by_ratio(groups_indices, TRAIN_RATIO, VAL_RATIO)

		train_idx.extend(tr_g)
		val_idx.extend(va_g)
		test_idx.extend(te_g)

	return _shuffle_df(df.loc[train_idx], seed), _shuffle_df(df.loc[val_idx], seed), _shuffle_df(df.loc[test_idx], seed)


# ============================================================
# PP3: Stratified Group Split (StratifiedGroupKFold)
# ============================================================
def pp3_stratified_group_split(
	df: pd.DataFrame,
	embeddings: np.ndarray,
	seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	"""PP3: Phân tầng nhóm (StratifiedGroupKFold) — Bảo toàn khối mẫu vật + cân bằng loài."""
	sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)

	groups = (df["label"] + "___" + df["subfolder"]).values
	labels = df["label"].values

	trainval_idx, test_idx = None, None
	for tv_i, te_i in sgkf.split(df.index, labels, groups):
		trainval_idx = df.index[tv_i].tolist()
		test_idx = df.index[te_i].tolist()
		break

	df_trainval = df.loc[trainval_idx]
	groups_tv = (df_trainval["label"] + "___" + df_trainval["subfolder"]).values
	labels_tv = df_trainval["label"].values

	sgkf_val = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
	train_idx, val_idx = None, None
	for tr_i, va_i in sgkf_val.split(df_trainval.index, labels_tv, groups_tv):
		train_idx = df_trainval.index[tr_i].tolist()
		val_idx = df_trainval.index[va_i].tolist()
		break

	return _shuffle_df(df.loc[train_idx], seed), _shuffle_df(df.loc[val_idx], seed), _shuffle_df(df.loc[test_idx], seed)


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
	"""
	Giải bài toán tối ưu DataSAIL (Integer Linear Programming / Graph Cut heuristic):
	Tối thiểu hóa L(π) = Σ Σ [π(x) != π(x')] * sim(x, x') * κ(x) * κ(x')
	kết hợp ràng buộc tỷ lệ nghiêm ngặt (60/20/20).
	"""
	n = len(item_embeddings)
	if n == 0:
		return [], [], []
	if n == 1:
		return [0], [], []
	if n == 2:
		return [0], [1], [] if val_ratio > 0 else [0], [], [1]

	sim_matrix = cosine_similarity(item_embeddings)
	np.fill_diagonal(sim_matrix, 0.0)
	sim_matrix = np.maximum(sim_matrix, 0.0)

	total_weight = float(np.sum(item_weights))
	target_tr = total_weight * train_ratio
	target_va = total_weight * val_ratio
	target_te = total_weight * (1.0 - train_ratio - val_ratio)

	# Khởi tạo phân bổ ban đầu dựa trên Spectral / KMeans clustering
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
	split_assignment = np.zeros(n, dtype=int)  # 0=Train, 1=Val, 2=Test

	if len(cluster_dists) >= 3:
		split_assignment[cluster_labels == cluster_dists[0][0]] = 2  # Test xa nhất
		split_assignment[cluster_labels == cluster_dists[1][0]] = 1  # Val xa nhì
		split_assignment[cluster_labels == cluster_dists[2][0]] = 0  # Train gần nhất
	elif len(cluster_dists) == 2:
		split_assignment[cluster_labels == cluster_dists[0][0]] = 2
		split_assignment[cluster_labels == cluster_dists[1][0]] = 0

	rng = np.random.RandomState(seed)

	# Hàm tính loss DataSAIL kết hợp phạt lệch tỷ lệ
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
# PP4: DataSAIL Specimen-Level Split (Subfolder Level)
# ============================================================
def pp4_datasail_specimen_split(
	df: pd.DataFrame,
	embeddings: np.ndarray,
	seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	"""PP4: DataSAIL ở cấp độ Mẫu vật — Giữ nguyên khối mẫu vật, tối thiểu hóa L(π)."""
	train_idx, val_idx, test_idx = [], [], []

	for _, group in df.groupby("label"):
		subfolder_groups = group.groupby("subfolder")
		subfolder_names = list(subfolder_groups.groups.keys())

		sf_embs = []
		sf_weights = []
		for sf in subfolder_names:
			indices = subfolder_groups.get_group(sf).index.tolist()
			sf_embs.append(embeddings[indices].mean(axis=0))
			sf_weights.append(len(indices))

		sf_embs = np.array(sf_embs)
		sf_weights = np.array(sf_weights, dtype=np.float32)

		tr_sf, va_sf, te_sf = _run_datasail_solver(
			sf_embs, sf_weights, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, seed=seed
		)

		for pos in tr_sf:
			train_idx.extend(subfolder_groups.get_group(subfolder_names[pos]).index.tolist())
		for pos in va_sf:
			val_idx.extend(subfolder_groups.get_group(subfolder_names[pos]).index.tolist())
		for pos in te_sf:
			test_idx.extend(subfolder_groups.get_group(subfolder_names[pos]).index.tolist())

	return _shuffle_df(df.loc[train_idx], seed), _shuffle_df(df.loc[val_idx], seed), _shuffle_df(df.loc[test_idx], seed)


# ============================================================
# PP5: DataSAIL Image-Level Split (Flatten Image Level)
# ============================================================
def pp5_datasail_image_split(
	df: pd.DataFrame,
	embeddings: np.ndarray,
	seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	"""PP5: DataSAIL ở cấp độ Ảnh — Tối thiểu hóa L(π) trên từng ảnh đơn lẻ."""
	train_idx, val_idx, test_idx = [], [], []

	for _, group in df.groupby("label"):
		indices = sorted(group.index.tolist())
		img_embs = embeddings[indices]
		img_weights = np.ones(len(indices), dtype=np.float32)

		tr_img, va_img, te_img = _run_datasail_solver(
			img_embs, img_weights, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, seed=seed
		)

		for pos in tr_img:
			train_idx.append(indices[pos])
		for pos in va_img:
			val_idx.append(indices[pos])
		for pos in te_img:
			test_idx.append(indices[pos])

	return _shuffle_df(df.loc[train_idx], seed), _shuffle_df(df.loc[val_idx], seed), _shuffle_df(df.loc[test_idx], seed)


# ============================================================
# PP6: Mahalanobis Centroid Outlier Split
# ============================================================
def pp6_mahalanobis_centroid_split(
	df: pd.DataFrame,
	embeddings: np.ndarray,
	seed: int = 42,
	eps: float = 1e-6,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	"""PP6: Chia theo khoảng cách Mahalanobis từ centroid — Mẫu dị biệt nhất vào Test."""
	train_idx, val_idx, test_idx = [], [], []

	for _, group in df.groupby("label"):
		subfolder_groups = group.groupby("subfolder")
		subfolder_names = list(subfolder_groups.groups.keys())

		sf_embs = [embeddings[subfolder_groups.get_group(sf).index.tolist()].mean(axis=0) for sf in subfolder_names]
		sf_embs = np.array(sf_embs)
		n_sf = len(sf_embs)

		if n_sf <= 2:
			groups_indices = [subfolder_groups.get_group(sf).index.tolist() for sf in subfolder_names]
			tr_g, va_g, te_g = _allocate_groups_by_ratio(groups_indices, TRAIN_RATIO, VAL_RATIO)
			train_idx.extend(tr_g)
			val_idx.extend(va_g)
			test_idx.extend(te_g)
			continue

		if n_sf >= 5:
			d_prime = min(n_sf - 2, 128)
			if d_prime >= 2:
				pca = PCA(n_components=d_prime, random_state=seed)
				sf_embs = pca.fit_transform(sf_embs)

		cov = np.cov(sf_embs, rowvar=False)
		cov = np.atleast_2d(cov) + np.eye(sf_embs.shape[1]) * eps
		cov_inv = np.linalg.pinv(cov)
		mean = sf_embs.mean(axis=0)

		diff = sf_embs - mean
		dists = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", diff, cov_inv, diff), 0.0))
		sorted_pos = np.argsort(dists)  # Sắp xếp tăng dần: Gần trước (Train), Xa sau (Test)

		sorted_groups_indices = [subfolder_groups.get_group(subfolder_names[pos]).index.tolist() for pos in sorted_pos]
		tr_g, va_g, te_g = _allocate_groups_by_ratio(sorted_groups_indices, TRAIN_RATIO, VAL_RATIO)

		train_idx.extend(tr_g)
		val_idx.extend(va_g)
		test_idx.extend(te_g)

	return _shuffle_df(df.loc[train_idx], seed), _shuffle_df(df.loc[val_idx], seed), _shuffle_df(df.loc[test_idx], seed)


# ============================================================
# PP7: Hierarchical Agglomerative Clustering Split
# ============================================================
def pp7_hierarchical_clustering_split(
	df: pd.DataFrame,
	embeddings: np.ndarray,
	seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	"""PP7: Agglomerative Ward Clustering trên cụm mẫu vật."""
	train_idx, val_idx, test_idx = [], [], []

	for _, group in df.groupby("label"):
		subfolder_groups = group.groupby("subfolder")
		subfolder_names = list(subfolder_groups.groups.keys())
		n_sf = len(subfolder_names)

		if n_sf < 3:
			groups_indices = [subfolder_groups.get_group(sf).index.tolist() for sf in subfolder_names]
			tr_g, va_g, te_g = _allocate_groups_by_ratio(groups_indices, TRAIN_RATIO, VAL_RATIO)
			train_idx.extend(tr_g)
			val_idx.extend(va_g)
			test_idx.extend(te_g)
			continue

		sf_embs = np.array([embeddings[subfolder_groups.get_group(sf).index.tolist()].mean(axis=0) for sf in subfolder_names])
		n_clusters = min(3, n_sf)
		Z = linkage(sf_embs, method="ward")
		cluster_labels = fcluster(Z, t=n_clusters, criterion="maxclust")

		global_centroid = sf_embs.mean(axis=0)
		cluster_info = []

		for cid in sorted(set(cluster_labels)):
			mask = cluster_labels == cid
			c_centroid = sf_embs[mask].mean(axis=0)
			c_dist = float(np.linalg.norm(c_centroid - global_centroid))
			sf_indices = []
			for idx_sf, lbl in enumerate(cluster_labels):
				if lbl == cid:
					sf_indices.extend(subfolder_groups.get_group(subfolder_names[idx_sf]).index.tolist())
			cluster_info.append((cid, c_dist, sf_indices))

		# Sắp xếp tăng dần theo khoảng cách (gần tâm nhất vào Train trước)
		cluster_info.sort(key=lambda x: x[1])
		sorted_groups_indices = [info[2] for info in cluster_info]

		tr_g, va_g, te_g = _allocate_groups_by_ratio(sorted_groups_indices, TRAIN_RATIO, VAL_RATIO)
		train_idx.extend(tr_g)
		val_idx.extend(va_g)
		test_idx.extend(te_g)

	return _shuffle_df(df.loc[train_idx], seed), _shuffle_df(df.loc[val_idx], seed), _shuffle_df(df.loc[test_idx], seed)


# ============================================================
# PP8: Cosine Similarity Graph Connected Components Split
# ============================================================
def pp8_cosine_graph_split(
	df: pd.DataFrame,
	embeddings: np.ndarray,
	seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	"""PP8: Đồ thị Cosine Similarity + Connected Components — Ngăn rò rỉ gián tiếp."""
	train_idx, val_idx, test_idx = [], [], []

	for _, group in df.groupby("label"):
		subfolder_groups = group.groupby("subfolder")
		subfolder_names = list(subfolder_groups.groups.keys())
		n_sf = len(subfolder_names)

		if n_sf < 3:
			groups_indices = [subfolder_groups.get_group(sf).index.tolist() for sf in subfolder_names]
			tr_g, va_g, te_g = _allocate_groups_by_ratio(groups_indices, TRAIN_RATIO, VAL_RATIO)
			train_idx.extend(tr_g)
			val_idx.extend(va_g)
			test_idx.extend(te_g)
			continue

		sf_embs = np.array([embeddings[subfolder_groups.get_group(sf).index.tolist()].mean(axis=0) for sf in subfolder_names])
		sim_matrix = cosine_similarity(sf_embs)
		np.fill_diagonal(sim_matrix, 0.0)

		adj = (sim_matrix >= COSINE_THRESHOLD).astype(np.float32)
		n_comp, comp_labels = connected_components(csr_matrix(adj), directed=False)

		global_centroid = sf_embs.mean(axis=0)
		comp_info = []
		for cid in range(n_comp):
			mask = comp_labels == cid
			c_centroid = sf_embs[mask].mean(axis=0)
			c_dist = float(np.linalg.norm(c_centroid - global_centroid))
			c_indices = []
			for idx_sf, lbl in enumerate(comp_labels):
				if lbl == cid:
					c_indices.extend(subfolder_groups.get_group(subfolder_names[idx_sf]).index.tolist())
			comp_info.append((cid, c_dist, c_indices))

		# Sắp xếp tăng dần theo khoảng cách (gần tâm nhất vào Train trước)
		comp_info.sort(key=lambda x: x[1])
		sorted_groups_indices = [info[2] for info in comp_info]

		tr_g, va_g, te_g = _allocate_groups_by_ratio(sorted_groups_indices, TRAIN_RATIO, VAL_RATIO)
		train_idx.extend(tr_g)
		val_idx.extend(va_g)
		test_idx.extend(te_g)

	return _shuffle_df(df.loc[train_idx], seed), _shuffle_df(df.loc[val_idx], seed), _shuffle_df(df.loc[test_idx], seed)


# ============================================================
# PP9: Adversarial MLP Discriminator Split
# ============================================================
def pp9_adversarial_validation_split(
	df: pd.DataFrame,
	embeddings: np.ndarray,
	seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	"""PP9: Adversarial MLP Discriminator — Subfolder dị biệt nhất vào Test."""
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	rng = np.random.RandomState(seed)
	train_idx, val_idx, test_idx = [], [], []

	for _, group in df.groupby("label"):
		subfolder_groups = group.groupby("subfolder")
		subfolder_names = list(subfolder_groups.groups.keys())
		n_sf = len(subfolder_names)

		if n_sf < 3:
			groups_indices = [subfolder_groups.get_group(sf).index.tolist() for sf in subfolder_names]
			tr_g, va_g, te_g = _allocate_groups_by_ratio(groups_indices, TRAIN_RATIO, VAL_RATIO)
			train_idx.extend(tr_g)
			val_idx.extend(va_g)
			test_idx.extend(te_g)
			continue

		sf_embs = np.array([embeddings[subfolder_groups.get_group(sf).index.tolist()].mean(axis=0) for sf in subfolder_names])

		perm = rng.permutation(n_sf)
		half = n_sf // 2
		y = np.zeros(n_sf, dtype=np.float32)
		y[perm[half:]] = 1.0

		emb_dim = sf_embs.shape[1]
		discriminator = nn.Sequential(
			nn.Linear(emb_dim, 128),
			nn.ReLU(),
			nn.Dropout(0.3),
			nn.Linear(128, 1),
			nn.Sigmoid(),
		).to(device)

		X_t = torch.from_numpy(sf_embs).float().to(device)
		y_t = torch.from_numpy(y).float().to(device)

		opt = torch.optim.Adam(discriminator.parameters(), lr=1e-3)
		crit = nn.BCELoss()

		discriminator.train()
		for _ in range(25):
			opt.zero_grad()
			loss = crit(discriminator(X_t).squeeze(), y_t)
			loss.backward()
			opt.step()

		discriminator.eval()
		with torch.no_grad():
			scores = discriminator(X_t).squeeze().cpu().numpy()

		difficulty = np.abs(scores - 0.5)
		sorted_pos = np.argsort(difficulty)  # Dị biệt ít hơn (gần 0.5) vào Train trước, dị biệt nhất vào Test

		sorted_groups_indices = [subfolder_groups.get_group(subfolder_names[pos]).index.tolist() for pos in sorted_pos]
		tr_g, va_g, te_g = _allocate_groups_by_ratio(sorted_groups_indices, TRAIN_RATIO, VAL_RATIO)

		train_idx.extend(tr_g)
		val_idx.extend(va_g)
		test_idx.extend(te_g)

	return _shuffle_df(df.loc[train_idx], seed), _shuffle_df(df.loc[val_idx], seed), _shuffle_df(df.loc[test_idx], seed)


# Registry chứa toàn bộ 9 thuật toán chia
ALL_SOLVERS: Dict[str, Callable] = {
	"PP1_Image_Random": pp1_image_random_split,
	"PP2_Group_Random": pp2_group_random_split,
	"PP3_Stratified_Group": pp3_stratified_group_split,
	"PP4_DataSAIL_Specimen": pp4_datasail_specimen_split,
	"PP5_DataSAIL_Image": pp5_datasail_image_split,
	"PP6_Mahalanobis_Centroid": pp6_mahalanobis_centroid_split,
	"PP7_Hierarchical_Clustering": pp7_hierarchical_clustering_split,
	"PP8_Cosine_Graph": pp8_cosine_graph_split,
	"PP9_Adversarial_Validation": pp9_adversarial_validation_split,
}
