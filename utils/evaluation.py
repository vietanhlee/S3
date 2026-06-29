import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.spatial.distance import cdist
from sklearn.metrics import (
	roc_auc_score,
	silhouette_score,
	davies_bouldin_score,
	calinski_harabasz_score,
	normalized_mutual_info_score
)
from sklearn.cluster import KMeans


@torch.no_grad()
def extract_all_embeddings(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[torch.Tensor, np.ndarray]:
	model.eval()
	all_embs = []
	all_labels = []
	for images, labels in loader:
		images = images.to(device, non_blocking=True)
		embs = model(images)
		all_embs.append(embs.cpu())
		all_labels.extend(labels.tolist())
	return torch.cat(all_embs, dim=0), np.array(all_labels)


def compute_dunn_index(embeddings: np.ndarray, labels: np.ndarray) -> float:
	n_clusters = len(np.unique(labels))
	if n_clusters <= 1:
		return 0.0

	cluster_embs = [embeddings[labels == i] for i in range(n_clusters)]
	max_intra_dist = 0.0
	for embs in cluster_embs:
		if len(embs) > 1:
			dists = cdist(embs, embs, metric='euclidean')
			max_val = dists.max()
			if max_val > max_intra_dist:
				max_intra_dist = max_val

	if max_intra_dist == 0.0:
		return 0.0

	min_inter_dist = float('inf')
	for i in range(n_clusters):
		for j in range(i + 1, n_clusters):
			dists = cdist(cluster_embs[i], cluster_embs[j], metric='euclidean')
			min_val = dists.min()
			if min_val < min_inter_dist:
				min_inter_dist = min_val

	return float(min_inter_dist / max_intra_dist)


def evaluate_retrieval(model: nn.Module, loader: DataLoader, device: torch.device, class_names: list[str], k_values: list[int] | None = None) -> dict:
	if k_values is None:
		k_values = [1, 5, 10]

	embeddings, labels = extract_all_embeddings(model, loader, device)
	embeddings_np = embeddings.numpy()
	n = len(labels)
	n_classes = len(class_names)

	dist_matrix = torch.cdist(embeddings, embeddings, p=2).numpy()

	recall_at_k = {k: 0.0 for k in k_values}
	precision_at_k = {k: 0.0 for k in k_values}
	aps = []

	sample_recall1 = []
	sample_recall5 = []
	sample_map = []

	for i in range(n):
		dists = dist_matrix[i].copy()
		dists[i] = np.inf
		sorted_indices = np.argsort(dists)[:-1]

		retrieved_labels = labels[sorted_indices]
		is_relevant = (retrieved_labels == labels[i])
		n_positives = int((labels == labels[i]).sum()) - 1

		if n_positives == 0:
			sample_recall1.append(0.0)
			sample_recall5.append(0.0)
			sample_map.append(0.0)
			continue

		for k in k_values:
			top_k_relevant = is_relevant[:k]
			recall_at_k[k] += float(top_k_relevant.any())
			precision_at_k[k] += float(top_k_relevant.sum()) / k

		sample_recall1.append(float(is_relevant[:1].any()))
		sample_recall5.append(float(is_relevant[:5].any()))

		cumsum = np.cumsum(is_relevant).astype(np.float64)
		precision_curve = cumsum / np.arange(1, n, dtype=np.float64)
		ap = (precision_curve * is_relevant).sum() / n_positives
		aps.append(ap)
		sample_map.append(ap)

	n_valid = max(len(aps), 1)
	for k in k_values:
		recall_at_k[k] /= n_valid
		precision_at_k[k] /= n_valid
	mAP = float(np.mean(aps)) if aps else 0.0

	auc_val = 0.0
	try:
		pair_labels = []
		pair_scores = []
		for i in range(n):
			for j in range(i + 1, n):
				pair_labels.append(int(labels[i] == labels[j]))
				pair_scores.append(-dist_matrix[i][j])
		auc_val = float(roc_auc_score(pair_labels, pair_scores))
	except Exception:
		auc_val = 0.0

	# Clustering metrics
	try:
		silhouette = float(silhouette_score(embeddings_np, labels))
	except Exception:
		silhouette = 0.0

	try:
		dbi = float(davies_bouldin_score(embeddings_np, labels))
	except Exception:
		dbi = 0.0

	try:
		chi = float(calinski_harabasz_score(embeddings_np, labels))
	except Exception:
		chi = 0.0

	try:
		dunn = float(compute_dunn_index(embeddings_np, labels))
	except Exception:
		dunn = 0.0

	try:
		kmeans = KMeans(n_clusters=n_classes, random_state=42, n_init=10)
		kmeans_labels = kmeans.fit_predict(embeddings_np)
		nmi = float(normalized_mutual_info_score(labels, kmeans_labels))
	except Exception:
		nmi = 0.0

	# Intra-Inter ratio
	try:
		intra_dists = []
		inter_dists = []
		for i in range(n):
			for j in range(i + 1, n):
				d = dist_matrix[i, j]
				if labels[i] == labels[j]:
					intra_dists.append(d)
				else:
					inter_dists.append(d)
		intra_mean = np.mean(intra_dists) if intra_dists else 0.0
		inter_mean = np.mean(inter_dists) if inter_dists else 0.0
		intra_inter_ratio = float(intra_mean / inter_mean) if inter_mean > 0 else 0.0
	except Exception:
		intra_inter_ratio = 0.0

	# Per class metrics
	per_class_recall1 = []
	per_class_recall5 = []
	per_class_map = []
	per_class_auc = []

	for c in range(n_classes):
		class_indices = np.where(labels == c)[0]
		if len(class_indices) == 0:
			per_class_recall1.append(0.0)
			per_class_recall5.append(0.0)
			per_class_map.append(0.0)
			per_class_auc.append(0.0)
			continue

		c_recall1 = np.mean([sample_recall1[idx] for idx in class_indices])
		c_recall5 = np.mean([sample_recall5[idx] for idx in class_indices])
		c_map = np.mean([sample_map[idx] for idx in class_indices])

		c_auc = 0.0
		try:
			pair_labels_c = []
			pair_scores_c = []
			for idx_i in class_indices:
				for idx_j in class_indices:
					if idx_i < idx_j:
						pair_labels_c.append(1)
						pair_scores_c.append(-dist_matrix[idx_i][idx_j])
				other_indices = np.where(labels != c)[0]
				for idx_k in other_indices:
					pair_labels_c.append(0)
					pair_scores_c.append(-dist_matrix[idx_i][idx_k])
			if len(set(pair_labels_c)) > 1:
				c_auc = float(roc_auc_score(pair_labels_c, pair_scores_c))
		except Exception:
			c_auc = 0.0

		per_class_recall1.append(c_recall1)
		per_class_recall5.append(c_recall5)
		per_class_map.append(c_map)
		per_class_auc.append(c_auc)

	results = {
		"mAP": mAP,
		"AUC": auc_val,
		"Silhouette": silhouette,
		"Davies-Bouldin": dbi,
		"Calinski-Harabasz": chi,
		"Dunn-Index": dunn,
		"NMI": nmi,
		"Intra-Inter-Ratio": intra_inter_ratio,
		"per_class_recall1": per_class_recall1,
		"per_class_recall5": per_class_recall5,
		"per_class_map": per_class_map,
		"per_class_auc": per_class_auc,
	}
	for k in k_values:
		results[f"Recall@{k}"] = recall_at_k[k]
		results[f"Precision@{k}"] = precision_at_k[k]

	return results


def format_retrieval_report(results: dict, class_names: list[str], prefix: str = "") -> str:
	lines = []
	lines.append(f"\n=======================================================================")
	lines.append(f"  BÁO CÁO TRUY VẤN CHI TIẾT (RETRIEVAL REPORT) - {prefix.upper()}")
	lines.append(f"=======================================================================")

	lines.append("Chỉ số toàn cục (Global Metrics):")
	global_keys = ["mAP", "AUC", "Recall@1", "Recall@5", "Recall@10", "Precision@1", "Precision@5", "Precision@10"]
	for k in global_keys:
		if k in results:
			val = results[k]
			lines.append(f"  {k:<15}: {val*100:.2f}%" if k != "AUC" else f"  {k:<15}: {val:.4f}")

	lines.append("\nChỉ số phân cụm không gian nhúng (Clustering Metrics):")
	clustering_keys = ["Silhouette", "Davies-Bouldin", "Calinski-Harabasz", "Dunn-Index", "NMI", "Intra-Inter-Ratio"]
	for k in clustering_keys:
		if k in results:
			val = results[k]
			lines.append(f"  {k:<20}: {val:.4f}")

	lines.append("\nChi tiết cho từng loài gỗ (Class-wise Metrics):")
	header = f"{'Loài gỗ (Class)':<35} | {'Recall@1':<10} | {'Recall@5':<10} | {'mAP':<10} | {'AUC':<10}"
	lines.append(header)
	lines.append("-" * len(header))

	per_class_recall1 = results.get("per_class_recall1", [])
	per_class_recall5 = results.get("per_class_recall5", [])
	per_class_map = results.get("per_class_map", [])
	per_class_auc = results.get("per_class_auc", [])

	for idx, name in enumerate(class_names):
		r1 = per_class_recall1[idx] * 100 if idx < len(per_class_recall1) else 0.0
		r5 = per_class_recall5[idx] * 100 if idx < len(per_class_recall5) else 0.0
		m = per_class_map[idx] * 100 if idx < len(per_class_map) else 0.0
		a = per_class_auc[idx] if idx < len(per_class_auc) else 0.0
		lines.append(f"{name:<35} | {r1:>8.2f}% | {r5:>8.2f}% | {m:>8.2f}% | {a:>10.4f}")

	lines.append("=======================================================================")
	return "\n".join(lines)


def evaluate_cross_retrieval(
	model: nn.Module,
	query_loader: DataLoader,
	gallery_loader: DataLoader,
	device: torch.device,
	class_names: list[str],
	k_values: list[int] | None = None,
) -> dict:
	if k_values is None:
		k_values = [1, 5, 10]

	query_embs, query_labels = extract_all_embeddings(model, query_loader, device)
	gallery_embs, gallery_labels = extract_all_embeddings(model, gallery_loader, device)

	n_query = len(query_labels)
	n_gallery = len(gallery_labels)
	n_classes = len(class_names)

	dist_matrix = torch.cdist(query_embs, gallery_embs, p=2).numpy()

	recall_at_k = {k: 0.0 for k in k_values}
	precision_at_k = {k: 0.0 for k in k_values}
	aps = []

	sample_recall1 = []
	sample_recall5 = []
	sample_precision1 = []
	sample_precision5 = []
	sample_map = []

	for i in range(n_query):
		dists = dist_matrix[i].copy()
		sorted_indices = np.argsort(dists)

		retrieved_labels = gallery_labels[sorted_indices]
		is_relevant = (retrieved_labels == query_labels[i])
		n_positives = int((gallery_labels == query_labels[i]).sum())

		if n_positives == 0:
			sample_recall1.append(0.0)
			sample_recall5.append(0.0)
			sample_precision1.append(0.0)
			sample_precision5.append(0.0)
			sample_map.append(0.0)
			continue

		for k in k_values:
			top_k_relevant = is_relevant[:k]
			recall_at_k[k] += float(top_k_relevant.any())
			precision_at_k[k] += float(top_k_relevant.sum()) / k

		sample_recall1.append(float(is_relevant[:1].any()))
		sample_recall5.append(float(is_relevant[:5].any()))
		sample_precision1.append(float(is_relevant[:1].sum()) / 1.0)
		sample_precision5.append(float(is_relevant[:5].sum()) / 5.0)

		cumsum = np.cumsum(is_relevant).astype(np.float64)
		precision_curve = cumsum / np.arange(1, n_gallery + 1, dtype=np.float64)
		ap = (precision_curve * is_relevant).sum() / n_positives
		aps.append(ap)
		sample_map.append(ap)

	n_valid = max(len(aps), 1)
	for k in k_values:
		recall_at_k[k] /= n_valid
		precision_at_k[k] /= n_valid
	mAP = float(np.mean(aps)) if aps else 0.0

	auc_val = 0.0
	try:
		pair_labels = []
		pair_scores = []
		for i in range(n_query):
			class_i = query_labels[i]
			pos_indices = np.where(gallery_labels == class_i)[0]
			neg_indices = np.where(gallery_labels != class_i)[0]
			
			for pos_idx in pos_indices:
				pair_labels.append(1)
				pair_scores.append(-dist_matrix[i][pos_idx])
			for neg_idx in neg_indices:
				pair_labels.append(0)
				pair_scores.append(-dist_matrix[i][neg_idx])
		if len(set(pair_labels)) > 1:
			auc_val = float(roc_auc_score(pair_labels, pair_scores))
	except Exception:
		auc_val = 0.0

	per_class_recall1 = []
	per_class_recall5 = []
	per_class_precision1 = []
	per_class_precision5 = []
	per_class_map = []
	per_class_auc = []

	for c in range(n_classes):
		class_query_indices = np.where(query_labels == c)[0]
		if len(class_query_indices) == 0:
			per_class_recall1.append(0.0)
			per_class_recall5.append(0.0)
			per_class_precision1.append(0.0)
			per_class_precision5.append(0.0)
			per_class_map.append(0.0)
			per_class_auc.append(0.0)
			continue

		c_recall1 = np.mean([sample_recall1[idx] for idx in class_query_indices])
		c_recall5 = np.mean([sample_recall5[idx] for idx in class_query_indices])
		c_precision1 = np.mean([sample_precision1[idx] for idx in class_query_indices])
		c_precision5 = np.mean([sample_precision5[idx] for idx in class_query_indices])
		c_map = np.mean([sample_map[idx] for idx in class_query_indices])

		c_auc = 0.0
		try:
			pair_labels_c = []
			pair_scores_c = []
			for idx_i in class_query_indices:
				pos_indices = np.where(gallery_labels == c)[0]
				neg_indices = np.where(gallery_labels != c)[0]
				for pos_idx in pos_indices:
					pair_labels_c.append(1)
					pair_scores_c.append(-dist_matrix[idx_i][pos_idx])
				for neg_idx in neg_indices:
					pair_labels_c.append(0)
					pair_scores_c.append(-dist_matrix[idx_i][neg_idx])
			if len(set(pair_labels_c)) > 1:
				c_auc = float(roc_auc_score(pair_labels_c, pair_scores_c))
		except Exception:
			c_auc = 0.0

		per_class_recall1.append(c_recall1)
		per_class_recall5.append(c_recall5)
		per_class_precision1.append(c_precision1)
		per_class_precision5.append(c_precision5)
		per_class_map.append(c_map)
		per_class_auc.append(c_auc)

	results = {
		"mAP": mAP,
		"AUC": auc_val,
		"per_class_recall1": per_class_recall1,
		"per_class_recall5": per_class_recall5,
		"per_class_precision1": per_class_precision1,
		"per_class_precision5": per_class_precision5,
		"per_class_map": per_class_map,
		"per_class_auc": per_class_auc
	}
	for k in k_values:
		results[f"Recall@{k}"] = recall_at_k[k]
		results[f"Precision@{k}"] = precision_at_k[k]

	return results


def evaluate_loss(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
	model.eval()
	total_loss = 0.0
	n_batches = 0
	with torch.no_grad():
		for images, labels in loader:
			images = images.to(device, non_blocking=True)
			labels = labels.to(device, non_blocking=True)
			embeddings = model(images)
			loss = criterion(embeddings, labels)
			total_loss += loss.item()
			n_batches += 1
	return total_loss / max(n_batches, 1)
