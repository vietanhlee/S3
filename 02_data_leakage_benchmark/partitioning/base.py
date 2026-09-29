"""
02_specimen_variant_gan.partitioning.base
=========================================
Các hàm phụ trợ đo đạc khoảng cách đặc trưng, phân nhóm mẫu vật và kiểm định tính hợp lệ của phân vùng.
"""

import random
from typing import Optional, Tuple, List, Dict
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def compute_split_counts(
	n_total: int,
	train_ratio: float,
	val_ratio: float,
) -> Tuple[int, int, int]:
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


def validate_split(
	df_all: pd.DataFrame,
	df_train: pd.DataFrame,
	df_val: pd.DataFrame,
	df_test: pd.DataFrame,
	method_name: str,
) -> bool:
	"""Kiểm tra không trùng lặp, tính toàn vẹn và mức độ phủ lớp của phân vùng."""
	all_paths = set(df_all["path"])
	train_paths = set(df_train["path"])
	val_paths = set(df_val["path"])
	test_paths = set(df_test["path"])

	ok = True

	if train_paths & val_paths:
		print(f"[{method_name}] ERROR: train ∩ val = {len(train_paths & val_paths)} ảnh")
		ok = False
	if train_paths & test_paths:
		print(f"[{method_name}] ERROR: train ∩ test = {len(train_paths & test_paths)} ảnh")
		ok = False
	if val_paths & test_paths:
		print(f"[{method_name}] ERROR: val ∩ test = {len(val_paths & test_paths)} ảnh")
		ok = False

	union = train_paths | val_paths | test_paths
	if union != all_paths:
		missing = all_paths - union
		extra = union - all_paths
		print(f"[{method_name}] ERROR: missing={len(missing)}, extra={len(extra)}")
		ok = False

	all_labels = set(df_all["label"].unique())
	for split_name, df_split in [("train", df_train), ("val", df_val), ("test", df_test)]:
		missing_labels = all_labels - set(df_split["label"].unique())
		if missing_labels:
			print(f"[{method_name}] WARNING: {split_name} thiếu {len(missing_labels)} class: {missing_labels}")

	if ok:
		print(f"[{method_name}] ✓ Validation passed (train={len(df_train)}, val={len(df_val)}, test={len(df_test)})")
	return ok


def _shuffle_df(df: pd.DataFrame, seed: int) -> pd.DataFrame:
	return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def _mahalanobis_distances(embeddings: np.ndarray, eps: float = 1e-6) -> np.ndarray:
	"""Tính khoảng cách Mahalanobis từ mỗi điểm đến tâm cụm của toàn bộ embedding."""
	mean = np.mean(embeddings, axis=0)
	diff = embeddings - mean
	cov = np.cov(diff, rowvar=False)

	if cov.ndim == 0:
		cov = np.array([[cov]])
	cov += np.eye(cov.shape[0]) * eps

	try:
		cov_inv = np.linalg.inv(cov)
	except np.linalg.LinAlgError:
		cov_inv = np.linalg.pinv(cov)

	left = diff @ cov_inv
	dists_sq = np.sum(left * diff, axis=1)
	return np.sqrt(np.maximum(dists_sq, 0.0))


def _mahalanobis_dist_to_centroid(points: np.ndarray, mean: np.ndarray, cov_inv: np.ndarray) -> np.ndarray:
	diff = points - mean
	left = diff @ cov_inv
	dists_sq = np.sum(left * diff, axis=1)
	return np.sqrt(np.maximum(dists_sq, 0.0))


def find_optimal_clusters_elbow(embeddings: np.ndarray, max_k: int = 30, seed: int = 42) -> int:
	"""Tìm số cụm K tối ưu dựa trên điểm uốn (Elbow method) của đường cong quán tính WCSS."""
	n_samples = len(embeddings)
	if n_samples < 4:
		return 1

	max_k = min(max_k, n_samples - 1)
	k_range = list(range(2, max_k + 1))
	if len(k_range) == 0:
		return 1

	inertias = []
	for k in k_range:
		km = KMeans(n_clusters=k, random_state=seed, n_init=3)
		km.fit(embeddings)
		inertias.append(km.inertia_)

	p1 = np.array([k_range[0], inertias[0]])
	p2 = np.array([k_range[-1], inertias[-1]])
	line_vec = p2 - p1
	line_len = np.linalg.norm(line_vec)

	if line_len < 1e-9:
		return k_range[0]

	line_unit = line_vec / line_len
	max_dist = -1.0
	best_k = k_range[0]

	for k_val, in_val in zip(k_range, inertias):
		pt = np.array([k_val, in_val])
		pt_vec = pt - p1
		proj = np.dot(pt_vec, line_unit) * line_unit
		dist = np.linalg.norm(pt_vec - proj)
		if dist > max_dist:
			max_dist = dist
			best_k = k_val

	return best_k


def _split_by_groups(
	groups: List[List[int]],
	train_ratio: float,
	val_ratio: float,
	seed: int,
	sorted_descending: bool = True,
) -> Tuple[List[int], List[int], List[int]]:
	"""
	Phân chia các nhóm chỉ số (các cụm/components/subfolders) tránh data leakage ở mức nhóm:
	- groups: Danh sách các list chứa chỉ số ảnh của từng nhóm.
	- sorted_descending: Nếu True, groups được sắp xếp giảm dần theo khoảng cách (xa centroid nhất đứng đầu).
	                     Nếu False, groups được sắp xếp tăng dần theo khoảng cách (gần centroid nhất đứng đầu).
	"""
	rng = random.Random(seed)
	train_idx, val_idx, test_idx = [], [], []
	K = len(groups)

	if K == 0:
		return [], [], []

	if K == 1:
		# Chỉ có 1 nhóm: buộc phải chia ở mức ảnh
		indices = groups[0].copy()
		rng.shuffle(indices)
		n_total = len(indices)
		train_count = max(1, int(n_total * train_ratio))
		train_idx = indices[:train_count]
		temp_test_idx = indices[train_count:]

		# Chia đôi Temp-Test vào Val và Test
		mid = len(temp_test_idx) // 2
		val_idx = temp_test_idx[:mid]
		test_idx = temp_test_idx[mid:]

		# Đảm bảo các tập không bị trống
		if n_total >= 3:
			if len(val_idx) == 0:
				val_idx = [test_idx.pop(0)]
			if len(test_idx) == 0:
				test_idx = [val_idx.pop(0)]

	elif K == 2:
		# Có đúng 2 nhóm (cụm):
		if sorted_descending:
			# Cụm gần nhất (index 1) làm Train, cụm xa nhất (index 0) làm Test
			train_group = groups[1].copy()
			test_group = groups[0].copy()
		else:
			# Gần nhất (index 0) làm Train, xa nhất (index 1) làm Test
			train_group = groups[0].copy()
			test_group = groups[1].copy()

		# Xáo trộn train_group để chọn ngẫu nhiên
		rng.shuffle(train_group)

		# Val có số lượng bằng Test và lấy ảnh từ Train (giới hạn tối đa một nửa train_group để tránh làm trống Train)
		val_len = min(len(test_group), len(train_group) // 2)
		if val_len == 0 and len(train_group) > 1:
			val_len = 1

		val_idx = train_group[:val_len]
		train_idx = train_group[val_len:]
		test_idx = test_group.copy()

	else:
		# Có >= 3 nhóm (cụm): Chia theo khoảng cách phân cấp
		if sorted_descending:
			# Sắp xếp giảm dần (xa nhất đứng đầu):
			test_idx = groups[0].copy()
			val_idx = groups[1].copy()
			for g in groups[2:]:
				train_idx.extend(g)
		else:
			# Sắp xếp tăng dần (gần nhất đứng đầu):
			test_idx = groups[-1].copy()
			val_idx = groups[-2].copy()
			for g in groups[:-2]:
				train_idx.extend(g)

	return train_idx, val_idx, test_idx

