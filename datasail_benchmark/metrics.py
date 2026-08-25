"""
datasail_benchmark/metrics.py
=============================
Bộ chỉ số định lượng đánh giá rò rỉ dữ liệu (Data Leakage) và hiệu suất phân loại đạt chuẩn Q1/Q2.
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
	"""Tính tương đồng Cosine trung bình giữa các ảnh trong CÙNG một tập (S_intra)."""
	n = len(embeddings)
	if n <= 1:
		return 0.0

	split_labels = np.full(n, -1, dtype=int)
	split_labels[train_indices] = 0
	split_labels[val_indices] = 1
	split_labels[test_indices] = 2

	sim_matrix = cosine_similarity(embeddings)
	same_split_mask = (split_labels[:, None] == split_labels[None, :]) & \
	                  (split_labels[:, None] >= 0)
	np.fill_diagonal(same_split_mask, False)

	if not np.any(same_split_mask):
		return 0.0

	return float(np.mean(sim_matrix[same_split_mask]))


def compute_specimen_leakage_risk(
	df_train: pd.DataFrame,
	df_val: pd.DataFrame,
	df_test: pd.DataFrame,
) -> float:
	"""
	Tính Specimen Leakage Risk Ratio (SLR %):
	Tỷ lệ % subfolder/specimen bị phân tán đồng thời sang nhiều tập khác nhau.
	"""
	tr_sf = set(df_train["subfolder"].unique())
	va_sf = set(df_val["subfolder"].unique())
	te_sf = set(df_test["subfolder"].unique())

	all_sf = tr_sf | va_sf | te_sf
	if not all_sf:
		return 0.0

	leaked_sf = (tr_sf & va_sf) | (tr_sf & te_sf) | (va_sf & te_sf)
	return float((len(leaked_sf) / len(all_sf)) * 100.0)


def compute_pseudoreplication_index(
	df_train: pd.DataFrame,
	df_test: pd.DataFrame,
) -> float:
	"""
	Tính Pseudoreplication Index (PRI %):
	Tỷ lệ % các cặp ảnh cùng subfolder xuất hiện ở cả Train và Test.
	"""
	tr_sf_counts = df_train["subfolder"].value_counts().to_dict()
	te_sf_counts = df_test["subfolder"].value_counts().to_dict()

	shared_sf = set(tr_sf_counts.keys()) & set(te_sf_counts.keys())
	leaked_pairs = sum(tr_sf_counts[sf] * te_sf_counts[sf] for sf in shared_sf)

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
	"""Tính Class Coverage Rate (CCR): Tỷ lệ % các class có mặt ở cả 3 tập (100.0%)."""
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
	"""Tính Maximum Mean Discrepancy (MMD) đo độ lệch phân phối không gian đặc trưng giữa Train và Test."""
	if not train_indices or not test_indices:
		return 0.0

	X_tr = embeddings[train_indices]
	X_te = embeddings[test_indices]

	K_xx = rbf_kernel(X_tr, X_tr, gamma=gamma)
	K_yy = rbf_kernel(X_te, X_te, gamma=gamma)
	K_xy = rbf_kernel(X_tr, X_te, gamma=gamma)

	mmd2 = float(np.mean(K_xx) + np.mean(K_yy) - 2.0 * np.mean(K_xy))
	return float(np.sqrt(max(0.0, mmd2)))


def compute_nearest_neighbor_mean_sim(
	embeddings: np.ndarray,
	train_indices: List[int],
	test_indices: List[int],
) -> float:
	"""Tính chỉ số Nearest-Neighbor Cosine Similarity trung bình từ ảnh Test đến ảnh gần nhất trong Train."""
	if not train_indices or not test_indices:
		return 0.0

	train_feats = embeddings[train_indices]
	test_feats = embeddings[test_indices]

	sim_matrix = cosine_similarity(test_feats, train_feats)
	max_sims_per_test = sim_matrix.max(axis=1)
	return float(np.mean(max_sims_per_test))


def compute_wasserstein_divergence(
	df_all: pd.DataFrame,
	df_train: pd.DataFrame,
	df_val: pd.DataFrame,
	df_test: pd.DataFrame,
) -> float:
	"""Tính khoảng cách Wasserstein Divergence (W1) đo độ lệch phân phối class."""
	all_classes = sorted(df_all["label"].unique().tolist())
	n_classes = len(all_classes)
	if n_classes == 0:
		return 0.0

	cls_to_idx = {c: i for i, c in enumerate(all_classes)}

	def get_dist(df: pd.DataFrame) -> np.ndarray:
		counts = df["label"].value_counts().to_dict()
		vec = np.array([counts.get(c, 0) for c in all_classes], dtype=float)
		total = vec.sum()
		return vec / total if total > 0 else vec

	p_all = get_dist(df_all)
	p_tr = get_dist(df_train)
	p_te = get_dist(df_test)

	w1_tr = wasserstein_distance(np.arange(n_classes), np.arange(n_classes), p_all, p_tr)
	w1_te = wasserstein_distance(np.arange(n_classes), np.arange(n_classes), p_all, p_te)
	return float((w1_tr + w1_te) / 2.0)


def compute_knn_metrics(
	embeddings: np.ndarray,
	df_train: pd.DataFrame,
	df_test: pd.DataFrame,
	class_to_idx: Dict[str, int],
	path_to_idx: Dict[str, int],
	k: int = 1,
) -> Dict[str, float]:
	"""
	Đánh giá hiệu suất phân loại Zero-Training 1-NN Classifier trên tập Test:
	Trả về Top-1 Acc, Top-3 Acc, Balanced Acc, F1-Macro, và Hardest Class F1.
	"""
	tr_indices = [path_to_idx[p] for p in df_train["path"]]
	te_indices = [path_to_idx[p] for p in df_test["path"]]

	X_tr = embeddings[tr_indices]
	y_tr = np.array([class_to_idx[lbl] for lbl in df_train["label"]])

	X_te = embeddings[te_indices]
	y_te = np.array([class_to_idx[lbl] for lbl in df_test["label"]])

	knn = KNeighborsClassifier(n_neighbors=k, metric="cosine")
	knn.fit(X_tr, y_tr)

	preds = knn.predict(X_te)
	probs = knn.predict_proba(X_te)

	acc = float(accuracy_score(y_te, preds))
	bacc = float(balanced_accuracy_score(y_te, preds))
	f1_macro = float(f1_score(y_te, preds, average="macro", zero_division=0))

	# Top-3 Accuracy
	try:
		if len(knn.classes_) >= 3:
			top3_acc = float(np.mean([y_te[i] in np.argsort(probs[i])[-3:] for i in range(len(y_te))]))
		else:
			top3_acc = acc
	except Exception:
		top3_acc = acc

	# Hardest Class F1
	rep = classification_report(y_te, preds, output_dict=True, zero_division=0)
	summary_keys = {"accuracy", "macro avg", "weighted avg", "micro avg"}
	class_f1s = [v["f1-score"] for k_cls, v in rep.items() if isinstance(v, dict) and "f1-score" in v and k_cls not in summary_keys]
	hardest_f1 = float(min(class_f1s)) if class_f1s else 0.0


	return {
		"knn_accuracy": acc,
		"knn_top3_accuracy": top3_acc,
		"knn_balanced_accuracy": bacc,
		"knn_f1_macro": f1_macro,
		"hardest_class_f1": hardest_f1,
	}


def compute_statistical_significance(
	baseline_scores: List[float],
	candidate_scores: List[float],
) -> Dict[str, float]:
	"""Kiểm định ý nghĩa thống kê: Welch's t-test p-value & Cohen's d effect size."""
	if len(baseline_scores) <= 1 or len(candidate_scores) <= 1:
		return {"p_value": 1.0, "cohens_d": 0.0}

	t_stat, p_val = stats.ttest_ind(baseline_scores, candidate_scores, equal_var=False)

	m1, m2 = np.mean(baseline_scores), np.mean(candidate_scores)
	s1, s2 = np.std(baseline_scores, ddof=1), np.std(candidate_scores, ddof=1)

	pooled_std = np.sqrt(((len(baseline_scores) - 1) * s1**2 + (len(candidate_scores) - 1) * s2**2) /
	                     (len(baseline_scores) + len(candidate_scores) - 2))

	cohens_d = float((m1 - m2) / pooled_std) if pooled_std > 0 else 0.0
	return {
		"p_value": float(p_val),
		"cohens_d": float(abs(cohens_d)),
	}
