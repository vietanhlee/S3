"""
02_specimen_variant_gan.partitioning.graph_splits
=================================================
Các giao thức phân hoạch dựa trên cấu trúc đồ thị tương đồng hoặc phân cấp Ward:
- PP4: Hierarchical Clustering (Ward linkage)
- PP5: Cosine Similarity Graph + Connected Components
"""

from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.cluster.hierarchy import linkage, fcluster
from tqdm import tqdm

from .base import (
	_shuffle_df,
	_mahalanobis_dist_to_centroid,
	find_optimal_clusters_elbow,
	_split_by_groups,
)


def hierarchical_clustering_split(
	df: pd.DataFrame,
	embeddings: np.ndarray,
	train_ratio: float = 0.60,
	val_ratio: float = 0.20,
	seed: int = 42,
	eps: float = 1e-6,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
		global_centroid = pca_emb.mean(axis=0)

		n_clusters = find_optimal_clusters_elbow(pca_emb, max_k=30, seed=seed)
		n_clusters = min(n_clusters, n_subfolders)

		Z = linkage(pca_emb, method="ward")
		cluster_labels = fcluster(Z, t=n_clusters, criterion="maxclust")

		cluster_ids = sorted(set(cluster_labels))
		cluster_info = []

		for cid in cluster_ids:
			mask = cluster_labels == cid
			cluster_centroid = pca_emb[mask].mean(axis=0)
			dist_to_global = _mahalanobis_dist_to_centroid(cluster_centroid, global_centroid, cov_inv)[0]

			cluster_img_indices = []
			for idx_sf, sf_lbl in enumerate(cluster_labels):
				if sf_lbl == cid:
					sf_name = subfolder_names[idx_sf]
					cluster_img_indices.extend(subfolder_groups.get_group(sf_name).index.tolist())

			cluster_info.append((cid, dist_to_global, cluster_img_indices))

		cluster_info.sort(key=lambda x: -x[1])

		if len(cluster_info) < 3:
			groups = [info[2] for info in cluster_info]
			tr, va, te = _split_by_groups(groups, train_ratio, val_ratio, seed, sorted_descending=True)
			train_idx.extend(tr)
			val_idx.extend(va)
			test_idx.extend(te)
			continue

		n_total = len(group)
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


def cosine_graph_split(
	df: pd.DataFrame,
	embeddings: np.ndarray,
	train_ratio: float = 0.60,
	val_ratio: float = 0.20,
	seed: int = 42,
	cosine_threshold: float = 0.92,
	eps: float = 1e-6,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
		global_centroid = pca_emb.mean(axis=0)

		sim_matrix = cosine_similarity(subfolder_embs)
		np.fill_diagonal(sim_matrix, 0.0)

		adj = (sim_matrix >= cosine_threshold).astype(np.float32)
		sparse_adj = csr_matrix(adj)

		n_components, component_labels = connected_components(sparse_adj, directed=False)

		components = []
		for cid in range(n_components):
			mask = component_labels == cid
			comp_emb_pca = pca_emb[mask]
			comp_centroid = comp_emb_pca.mean(axis=0)
			dist = _mahalanobis_dist_to_centroid(comp_centroid, global_centroid, cov_inv)[0]

			comp_img_indices = []
			for idx_sf, comp_lbl in enumerate(component_labels):
				if comp_lbl == cid:
					sf_name = subfolder_names[idx_sf]
					comp_img_indices.extend(subfolder_groups.get_group(sf_name).index.tolist())

			components.append((cid, dist, comp_img_indices))

		components.sort(key=lambda x: -x[1])

		if len(components) < 3:
			groups = [comp[2] for comp in components]
			tr, va, te = _split_by_groups(groups, train_ratio, val_ratio, seed, sorted_descending=True)
			train_idx.extend(tr)
			val_idx.extend(va)
			test_idx.extend(te)
			continue

		n_total = len(group)
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
