"""
datasail_benchmark/metrics.py
=============================
Bộ 16 chỉ số định lượng đánh giá rò rỉ dữ liệu (Data Leakage) và hiệu suất phân loại đạt chuẩn Q1/Q2.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import wasserstein_distance
from sklearn.metrics import (
	classification_report,
	accuracy_score,
	f1_score,
	balanced_accuracy_score,
	silhouette_score,
	top_k_accuracy_score,
)
from sklearn.metrics.pairwise import cosine_similarity, rbf_kernel
from sklearn.neighbors import KNeighborsClassifier


def compute_datasail_loss(
	embeddings: np.ndarray,
	train_indices: List[int],
	val_indices: List[int],
	test_indices: List[int],
	weights: np.ndarray | None = None,
) -> float:
	"""
	Tính hàm phạt DataSAIL Inter-Split Similarity Loss L(π):
	L(π) = Σ Σ [π(x) ≠ π(x')] * sim(x, x') * κ(x) * κ(x')
	"""
	n = len(embeddings)
	if n <= 1:
		return 0.0

	if weights is None:
		weights = np.ones(n, dtype=np.float32)

	split_labels = np.full(n, -1, dtype=int)
	split_labels[train_indices] = 0
	split_labels[val_indices] = 1
	split_labels[test_indices] = 2

	sim_matrix = cosine_similarity(embeddings)
	np.fill_diagonal(sim_matrix, 0.0)
	sim_matrix = np.maximum(sim_matrix, 0.0)

	weight_matrix = np.outer(weights, weights)
	diff_split_mask = split_labels[:, None] != split_labels[None, :]
	valid_mask = (split_labels[:, None] >= 0) & (split_labels[None, :] >= 0)
	mask = diff_split_mask & valid_mask

	total_loss = float(np.sum(sim_matrix[mask] * weight_matrix[mask]) / 2.0)
	return total_loss


def compute_inter_split_cosine_sim(
	embeddings: np.ndarray,
	train_indices: List[int],
	val_indices: List[int],
	test_indices: List[int],
) -> float:
	"""Tính tương đồng Cosine trung bình giữa các ảnh thuộc hai tập khác nhau (S_inter)."""
	n = len(embeddings)
	if n <= 1:
		return 0.0

	split_labels = np.full(n, -1, dtype=int)
	split_labels[train_indices] = 0
	split_labels[val_indices] = 1
	split_labels[test_indices] = 2

	sim_matrix = cosine_similarity(embeddings)
	diff_split_mask = (split_labels[:, None] != split_labels[None, :]) & \
	                  (split_labels[:, None] >= 0) & (split_labels[None, :] >= 0)
	
	if not np.any(diff_split_mask):
		return 0.0
		
	return float(np.mean(sim_matrix[diff_split_mask]))


def compute_intra_split_cosine_sim(
	embeddings: np.ndarray,
	train_indices: List[int],
	val_indices: List[int],
	test_indices: List[int],
) -> float:
	"""Tính tương đồng Cosine trung bình giữa các ảnh thuộc cùng một tập (S_intra)."""
	intra_sims = []
	sim_matrix = cosine_similarity(embeddings)

	for indices in [train_indices, val_indices, test_indices]:
		if len(indices) > 1:
			sub_sim = sim_matrix[np.ix_(indices, indices)]
			triu_indices = np.triu_indices(len(indices), k=1)
			intra_sims.extend(sub_sim[triu_indices])

	if not intra_sims:
		return 0.0
	return float(np.mean(intra_sims))


def compute_specimen_leakage_risk(
	df_train: pd.DataFrame,
	df_val: pd.DataFrame,
	df_test: pd.DataFrame,
) -> float:
	"""
	Tính Specimen Leakage Risk Ratio (SLR):
	Tỷ lệ % mẫu vật lý (subfolder) xuất hiện đồng thời ở nhiều hơn 1 tập.
	"""
	train_subfolders = set(df_train["subfolder"])
	val_subfolders = set(df_val["subfolder"])
	test_subfolders = set(df_test["subfolder"])

	all_subfolders = train_subfolders | val_subfolders | test_subfolders
	if not all_subfolders:
		return 0.0

	leaked_subfolders = (train_subfolders & val_subfolders) | \
	                    (train_subfolders & test_subfolders) | \
	                    (val_subfolders & test_subfolders)

	return float(len(leaked_subfolders) / len(all_subfolders) * 100.0)


def compute_pseudoreplication_index(
	df_train: pd.DataFrame,
	df_test: pd.DataFrame,
) -> float:
	"""
	Tính Pseudoreplication Index (PRI):
	Tỷ lệ cặp ảnh có cùng subfolder giữa Train và Test so với tổng số cặp có thể có.
	"""
	train_sf_counts = df_train["subfolder"].value_counts()
	test_sf_counts = df_test["subfolder"].value_counts()

	common_sfs = set(train_sf_counts.index) & set(test_sf_counts.index)
	if not common_sfs:
		return 0.0

	leaked_pairs = sum(train_sf_counts[sf] * test_sf_counts[sf] for sf in common_sfs)
	total_possible_pairs = len(df_train) * len(df_test)
	if total_possible_pairs == 0:
		return 0.0

	return float((leaked_pairs / total_possible_pairs) * 100.0)


def compute_class_coverage_rate(
	df_all: pd.DataFrame,
	df_train: pd.DataFrame,
	df_val: pd.DataFrame,
	df_test: pd.DataFrame,
) -> float:
	"""
	Tính Class Coverage Rate (CCR):
	Tỷ lệ % các class có mặt đầy đủ ở cả 3 tập (Train, Val, Test).
	"""
	all_classes = set(df_all["label"].unique())
	if not all_classes:
		return 100.0

	tr_classes = set(df_train["label"].unique())
	va_classes = set(df_val["label"].unique())
	te_classes = set(df_test["label"].unique())

	covered_classes = tr_classes & va_classes & te_classes
	return float((len(covered_classes) / len(all_classes)) * 100.0)


def compute_silhouette_separation(
	embeddings: np.ndarray,
	train_indices: List[int],
	val_indices: List[int],
	test_indices: List[int],
) -> float:
	"""Tính chỉ số Silhouette đo độ tách biệt không gian đặc trưng giữa các tập (S_split in [-1, 1])."""
	n = len(embeddings)
	if n <= 3:
		return 0.0

	split_labels = np.full(n, -1, dtype=int)
	split_labels[train_indices] = 0
	split_labels[val_indices] = 1
	split_labels[test_indices] = 2

	valid_mask = split_labels >= 0
	if len(set(split_labels[valid_mask])) < 2:
		return 0.0

	try:
		score = silhouette_score(embeddings[valid_mask], split_labels[valid_mask], metric="cosine")
		return float(score)
	except Exception:
		return 0.0


def compute_maximum_mean_discrepancy(
	embeddings: np.ndarray,
	train_indices: List[int],
	test_indices: List[int],
	gamma: float = 1.0,
) -> float:
	"""
	Tính Maximum Mean Discrepancy (MMD) đo độ lệch phân phối không gian đặc trưng giữa Train và Test.
	"""
	if not train_indices or not test_indices:
		return 0.0

	X_tr = embeddings[train_indices]
	X_te = embeddings[test_indices]

	K_xx = rbf_kernel(X_tr, X_tr, gamma=gamma)
	K_yy = rbf_kernel(X_te, X_te, gamma=gamma)
	K_xy = rbf_kernel(X_tr, X_te, gamma=gamma)

	mmd2 = float(np.mean(K_xx) + np.mean(K_yy) - 2.0 * np.mean(K_xy))
	return float(np.sqrt(max(0.0, mmd2)))


def compute_nearest_neighbor_stats(
	embeddings: np.ndarray,
	train_indices: List[int],
	test_indices: List[int],
) -> Dict[str, float]:
	"""
	Tính chỉ số Nearest-Neighbor Cosine Similarity từ ảnh Test đến ảnh gần nhất trong tập Train (NN_Sim):
	Trả về Mean, Max, 90th percentile, và Standard Deviation.
	"""
	if not train_indices or not test_indices:
		return {"nn_sim_mean": 0.0, "nn_sim_max": 0.0, "nn_sim_p90": 0.0, "nn_sim_std": 0.0}

	train_feats = embeddings[train_indices]
	test_feats = embeddings[test_indices]

	sim_matrix = cosine_similarity(test_feats, train_feats)
	max_sims_per_test = sim_matrix.max(axis=1)

	return {
		"nn_sim_mean": float(np.mean(max_sims_per_test)),
		"nn_sim_max": float(np.max(max_sims_per_test)),
		"nn_sim_p90": float(np.percentile(max_sims_per_test, 90)),
		"nn_sim_std": float(np.std(max_sims_per_test)),
	}


def compute_wasserstein_divergence(
	df_all: pd.DataFrame,
	df_train: pd.DataFrame,
	df_val: pd.DataFrame,
	df_test: pd.DataFrame,
) -> float:
	"""
	Tính Khoảng cách Wasserstein (Earth Mover's Distance) đo độ lệch phân phối loài giữa các tập và dataset gốc.
	"""
	all_labels = sorted(df_all["label"].unique())
	label_to_code = {lbl: i for i, lbl in enumerate(all_labels)}

	p_global = df_all["label"].map(label_to_code).values
	w_total = 0.0
	count = 0

	for df_split in [df_train, df_val, df_test]:
		if len(df_split) > 0:
			p_split = df_split["label"].map(label_to_code).values
			w_dist = wasserstein_distance(p_global, p_split)
			w_total += w_dist
			count += 1

	return float(w_total / max(1, count))


def compute_knn_metrics(
	embeddings: np.ndarray,
	df_train: pd.DataFrame,
	df_test: pd.DataFrame,
	class_to_idx: Dict[str, int],
	path_to_idx: Dict[str, int],
	k_neighbors: int = 1,
) -> Dict[str, float]:
	"""
	Đánh giá khả năng phân loại Zero-Training bằng K-Nearest Neighbors (KNN) trên frozen embeddings.
	Tính Accuracy Top-1, Top-3, Balanced Accuracy, F1 Macro, Weighted-F1, và Hardest Class F1.
	"""
	if len(df_train) == 0 or len(df_test) == 0:
		return {
			"knn_accuracy": 0.0,
			"knn_top3_accuracy": 0.0,
			"knn_balanced_accuracy": 0.0,
			"knn_f1_macro": 0.0,
			"knn_f1_weighted": 0.0,
			"hardest_class_f1": 0.0,
		}

	train_indices = [path_to_idx[p] for p in df_train["path"]]
	test_indices = [path_to_idx[p] for p in df_test["path"]]

	X_train = embeddings[train_indices]
	y_train = np.array([class_to_idx[lbl] for lbl in df_train["label"]])

	X_test = embeddings[test_indices]
	y_test = np.array([class_to_idx[lbl] for lbl in df_test["label"]])

	knn = KNeighborsClassifier(n_neighbors=k_neighbors, metric="cosine", n_jobs=-1)
	knn.fit(X_train, y_train)

	y_pred = knn.predict(X_test)
	probs = knn.predict_proba(X_test)

	acc = float(accuracy_score(y_test, y_pred))
	balanced_acc = float(balanced_accuracy_score(y_test, y_pred))

	# Top-3 Accuracy
	try:
		if probs.shape[1] >= 3:
			top3_acc = float(top_k_accuracy_score(y_test, probs, k=3, labels=np.arange(len(class_to_idx))))
		else:
			top3_acc = acc
	except Exception:
		top3_acc = acc

	f1_macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
	f1_weighted = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

	# Hardest Class F1 (F1 nhỏ nhất trong các loài)
	per_class_f1 = f1_score(y_test, y_pred, average=None, zero_division=0)
	hardest_f1 = float(np.min(per_class_f1)) if len(per_class_f1) > 0 else 0.0

	return {
		"knn_accuracy": acc,
		"knn_top3_accuracy": top3_acc,
		"knn_balanced_accuracy": balanced_acc,
		"knn_f1_macro": f1_macro,
		"knn_f1_weighted": f1_weighted,
		"hardest_class_f1": hardest_f1,
	}


def compute_leakage_inflation_deltas(
	acc_random: float,
	acc_protocol: float,
	f1_random: float,
	f1_protocol: float,
) -> Dict[str, float]:
	"""
	Tính mức độ "bơm phồng" hiệu suất giả tạo do rò rỉ dữ liệu:
	Δ Acc = Acc_Random - Acc_Protocol (pp)
	Δ F1  = F1_Random - F1_Protocol (pp)
	"""
	return {
		"delta_accuracy_pp": float((acc_random - acc_protocol) * 100.0),
		"delta_f1_macro_pp": float((f1_random - f1_protocol) * 100.0),
	}


def compute_statistical_significance(
	scores_group_a: List[float],
	scores_group_b: List[float],
) -> Dict[str, float]:
	"""
	Tính kiểm định ý nghĩa thống kê (Welch's t-test p-value) và Kích thước Hiệu ứng (Cohen's d).
	"""
	if len(scores_group_a) < 2 or len(scores_group_b) < 2:
		return {"p_value": 1.0, "cohens_d": 0.0}

	t_stat, p_val = stats.ttest_ind(scores_group_a, scores_group_b, equal_var=False)

	n1, n2 = len(scores_group_a), len(scores_group_b)
	s1, s2 = np.std(scores_group_a, ddof=1), np.std(scores_group_b, ddof=1)
	s_pooled = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))

	if s_pooled == 0:
		cohen_d = 0.0
	else:
		cohen_d = (np.mean(scores_group_a) - np.mean(scores_group_b)) / s_pooled

	return {
		"p_value": float(p_val if not np.isnan(p_val) else 1.0),
		"cohens_d": float(cohen_d if not np.isnan(cohen_d) else 0.0),
	}
