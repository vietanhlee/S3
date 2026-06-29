import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.manifold import TSNE

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .evaluation import extract_all_embeddings

CAM_METHODS = ["gradcam", "gradcam++", "xgradcam", "eigencam", "hirescam", "layercam", "eigengradcam", "finercam"]


class MetricGradCAM:
	def __init__(self, model: nn.Module, target_layer: nn.Module, method: str = "gradcam") -> None:
		self.model = model
		self.target_layer = target_layer
		self.method = method.lower()
		self.activations = None
		self.forward_handle = target_layer.register_forward_hook(self._forward_hook)

	def _forward_hook(self, module, inputs, output):
		self.activations = output

	def remove(self) -> None:
		self.forward_handle.remove()

	def __call__(self, input_tensor: torch.Tensor, prototype: torch.Tensor, all_prototypes: torch.Tensor = None, target_class_idx: int = None) -> np.ndarray:
		if self.method == "eigencam":
			self.model.eval()
			with torch.no_grad():
				_ = self.model(input_tensor)
			act = self.activations.squeeze(0).detach().cpu().numpy()
			c, h, w = act.shape
			A = act.reshape(c, h * w).T
			A = A - np.mean(A, axis=0)
			U, S, Vt = np.linalg.svd(A, full_matrices=False)
			projection = (A @ Vt[0, :]).reshape(h, w)
			if np.sum(projection) < 0:
				projection = -projection
			cam = np.maximum(projection, 0)
		else:
			self.model.zero_grad()
			if self.method == "finercam" and all_prototypes is not None and target_class_idx is not None:
				emb = self.model(input_tensor)
				logits = torch.matmul(emb, all_prototypes.to(emb.device).T)
				main_category = target_class_idx
				prob = torch.softmax(logits, dim=-1)
				output_data = logits[0].detach().cpu().numpy()
				target_logit = output_data[main_category]
				
				sorted_indices = np.argsort(np.abs(output_data - target_logit))
				comparison_categories = sorted_indices[1:4]
				alpha = 1.0
				
				wn = logits[0, main_category]
				weights = [prob[0, idx] for idx in comparison_categories]
				numerator = sum(w * (wn - alpha * logits[0, idx]) for w, idx in zip(weights, comparison_categories))
				denominator = sum(weights)
				score = numerator / (denominator + 1e-9)
			else:
				emb = self.model(input_tensor)
				score = (emb * prototype.unsqueeze(0)).sum()

			if self.activations is None:
				raise RuntimeError("CAM hook did not capture activations")

			grads = torch.autograd.grad(score, self.activations, retain_graph=True)[0]
			
			if self.method == "gradcam++":
				grads_pos = torch.clamp(grads, min=0)
				grads_power_2 = grads_pos ** 2
				grads_power_3 = grads_pos ** 3
				sum_activations = torch.sum(self.activations, dim=(2, 3), keepdim=True)
				eps = 1e-7
				aij = grads_power_2 / (2 * grads_power_2 + sum_activations * grads_power_3 + eps)
				weights = torch.sum(aij * grads_pos, dim=(2, 3), keepdim=True)
				cam = torch.sum(weights * self.activations, dim=1)
				cam = F.relu(cam)
				cam = cam.squeeze().detach().cpu().numpy()
			elif self.method == "xgradcam":
				sum_activations = torch.sum(self.activations, dim=(2, 3), keepdim=True) + 1e-7
				weights = torch.sum(grads * self.activations / sum_activations, dim=(2, 3), keepdim=True)
				cam = torch.sum(weights * self.activations, dim=1)
				cam = F.relu(cam)
				cam = cam.squeeze().detach().cpu().numpy()
			elif self.method == "hirescam":
				cam = torch.sum(grads * self.activations, dim=1)
				cam = F.relu(cam)
				cam = cam.squeeze().detach().cpu().numpy()
			elif self.method == "layercam":
				cam = torch.sum(torch.clamp(grads, min=0) * self.activations, dim=1)
				cam = F.relu(cam)
				cam = cam.squeeze().detach().cpu().numpy()
			elif self.method == "eigengradcam":
				weighted_act = grads * self.activations
				act = weighted_act.squeeze(0).detach().cpu().numpy()
				c, h, w = act.shape
				A = act.reshape(c, h * w).T
				A = A - np.mean(A, axis=0)
				U, S, Vt = np.linalg.svd(A, full_matrices=False)
				projection = (A @ Vt[0, :]).reshape(h, w)
				if np.sum(projection) < 0:
					projection = -projection
				cam = np.maximum(projection, 0)
			else:
				weights = grads.mean(dim=(2, 3), keepdim=True)
				cam = (weights * self.activations).sum(dim=1, keepdim=True)
				cam = F.relu(cam)
				cam = cam.squeeze().detach().cpu().numpy()

		if cam.ndim == 0:
			cam = np.array([[float(cam)]])
		elif cam.ndim == 1:
			cam = cam[None, :]
		
		cam -= cam.min()
		if cam.max() > 0:
			cam /= cam.max()
		return cam


def find_last_conv_layer(model: nn.Module) -> nn.Module | None:
	last_conv = None
	for module in model.modules():
		if isinstance(module, nn.Conv2d):
			last_conv = module
	return last_conv


def overlay_cam_on_image(image: Image.Image, cam: np.ndarray, alpha: float = 0.4) -> Image.Image:
	cam_resized = Image.fromarray((cam * 255).astype(np.uint8)).resize(image.size)
	cam_resized = np.array(cam_resized) / 255.0
	color_map = plt.get_cmap("jet")
	heatmap = color_map(cam_resized)[:, :, :3]
	img = np.array(image).astype(np.float32) / 255.0
	overlay = img * (1 - alpha) + heatmap * alpha
	overlay = np.clip(overlay, 0, 1)
	return Image.fromarray((overlay * 255).astype(np.uint8))


def select_gradcam_representatives(df: pd.DataFrame, seed: int = 42) -> dict[str, list[dict]]:
	representatives = {'Dalbergia': [], 'Pterocarpus': []}
	unique_classes = df['label'].unique()
	for genus in ['Dalbergia', 'Pterocarpus']:
		genus_classes = [c for c in unique_classes if c.startswith(genus)]
		genus_classes = sorted(genus_classes)
		for cls in genus_classes:
			cls_df = df[df['label'] == cls]
			if len(cls_df) >= 2:
				sampled = cls_df.sample(n=2, random_state=seed).reset_index(drop=True)
			else:
				sampled = cls_df.sample(n=2, replace=True, random_state=seed).reset_index(drop=True)
			for _, row in sampled.iterrows():
				representatives[genus].append({
					'path': row['path'],
					'label': row['label'],
					'species': cls.replace(genus + " ", "")
				})
	return representatives


@torch.no_grad()
def compute_class_prototypes(model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int) -> torch.Tensor:
	model.eval()
	embeddings = []
	labels = []
	for images, batch_labels in loader:
		images = images.to(device, non_blocking=True)
		embs = model(images)
		embeddings.append(embs.cpu())
		labels.extend(batch_labels.tolist())
	embeddings = torch.cat(embeddings, dim=0)
	labels = np.array(labels)
	embedding_dim = embeddings.shape[1]
	prototypes = torch.zeros(num_classes, embedding_dim)
	for c in range(num_classes):
		idx = np.where(labels == c)[0]
		if len(idx) > 0:
			class_embs = embeddings[idx]
			proto = class_embs.mean(dim=0)
			proto = F.normalize(proto, p=2, dim=0)
			prototypes[c] = proto
		else:
			prototypes[c] = torch.zeros(embedding_dim)
	return prototypes


def generate_gradcam_maps(
	model: nn.Module,
	representatives: list[dict],
	prototypes: torch.Tensor,
	class_to_idx: dict[str, int],
	transform,
	device: torch.device,
	method: str = "gradcam"
) -> list[np.ndarray]:
	target_layer = find_last_conv_layer(model)
	if target_layer is None:
		print("Warning: Không tìm thấy Conv2d layer trong model để vẽ Grad-CAM")
		return [np.zeros((224, 224)) for _ in representatives]
	gradcam = MetricGradCAM(model, target_layer, method=method)
	model.eval()
	cam_maps = []
	for rep in representatives:
		with Image.open(rep['path']) as img:
			img = img.convert("RGB")
		input_tensor = transform(img).unsqueeze(0).to(device)
		class_idx = class_to_idx[rep['label']]
		proto = prototypes[class_idx].to(device)
		cam = gradcam(input_tensor, proto, all_prototypes=prototypes, target_class_idx=class_idx)
		cam_maps.append(cam)
	gradcam.remove()
	return cam_maps


def plot_gradcam_comparison(
	representatives: list[dict],
	before_cams: list[np.ndarray],
	after_cams: list[np.ndarray],
	genus_name: str,
	output_path: Path
) -> None:
	n_samples = len(representatives)
	if n_samples == 0:
		return
	fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))
	if n_samples == 1:
		axes = np.expand_dims(axes, axis=0)
	for i in range(n_samples):
		rep = representatives[i]
		with Image.open(rep['path']) as img:
			img = img.convert("RGB")
		axes[i, 0].imshow(img)
		axes[i, 0].axis('off')
		axes[i, 0].set_title(f"{rep['label']}\nSample {i % 2 + 1}", fontsize=10, fontweight='bold')

		cam_before = before_cams[i]
		overlay_before = overlay_cam_on_image(img, cam_before)
		axes[i, 1].imshow(overlay_before)
		axes[i, 1].axis('off')
		axes[i, 1].set_title("Before Training", fontsize=9)

		cam_after = after_cams[i]
		overlay_after = overlay_cam_on_image(img, cam_after)
		axes[i, 2].imshow(overlay_after)
		axes[i, 2].axis('off')
		axes[i, 2].set_title("After Training", fontsize=9)
	plt.suptitle(f"Grad-CAM Comparison for Genus {genus_name}", fontsize=14, fontweight='bold', y=0.99)
	plt.tight_layout()
	plt.savefig(output_path, dpi=200, bbox_inches='tight')
	plt.close()
	print(f"Grad-CAM comparison saved → {output_path}")


def plot_tsne_comparison(
	before_embs: np.ndarray,
	after_embs: np.ndarray,
	labels: np.ndarray,
	class_names: list[str],
	output_path: Path
) -> None:
	print("Running t-SNE dimensionality reduction...")
	n_samples = len(labels)
	perp = min(30, max(5, n_samples // 4))
	tsne = TSNE(n_components=2, perplexity=perp, random_state=42, n_iter=1000)
	embs_2d_before = tsne.fit_transform(before_embs)
	embs_2d_after = tsne.fit_transform(after_embs)

	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
	unique_labels = np.unique(labels)
	cmap = plt.get_cmap("tab20", len(unique_labels))

	for idx, label_idx in enumerate(unique_labels):
		class_name = class_names[label_idx]
		color = cmap(idx)
		mask = (labels == label_idx)
		ax1.scatter(
			embs_2d_before[mask, 0], embs_2d_before[mask, 1],
			color=color, label=class_name, alpha=0.8, edgecolors='k', s=50
		)
		ax2.scatter(
			embs_2d_after[mask, 0], embs_2d_after[mask, 1],
			color=color, label=class_name, alpha=0.8, edgecolors='k', s=50
		)

	ax1.set_title("Feature Space Before Metric Learning (t-SNE)", fontsize=12, fontweight='bold')
	ax1.grid(alpha=0.3)
	ax1.set_xlabel("t-SNE Dimension 1")
	ax1.set_ylabel("t-SNE Dimension 2")

	ax2.set_title("Feature Space After Metric Learning (t-SNE)", fontsize=12, fontweight='bold')
	ax2.grid(alpha=0.3)
	ax2.set_xlabel("t-SNE Dimension 1")
	ax2.set_ylabel("t-SNE Dimension 2")

	handles, labels_legend = ax2.get_legend_handles_labels()
	fig.legend(handles, labels_legend, loc='center right', bbox_to_anchor=(0.99, 0.5), fontsize=9)
	plt.tight_layout()
	plt.subplots_adjust(right=0.83)
	plt.savefig(output_path, dpi=200, bbox_inches='tight')
	plt.close()
	print(f"t-SNE comparison saved → {output_path}")


def calculate_pairwise_distances(embeddings: np.ndarray, labels: np.ndarray) -> tuple[list[float], list[float]]:
	n = len(labels)
	dist_matrix = np.sqrt(np.maximum(2.0 - 2.0 * np.dot(embeddings, embeddings.T), 0.0))
	intra_dists = []
	inter_dists = []
	for i in range(n):
		for j in range(i + 1, n):
			d = dist_matrix[i, j]
			if labels[i] == labels[j]:
				intra_dists.append(d)
			else:
				inter_dists.append(d)
	return intra_dists, inter_dists


def plot_distance_analysis(
	before_embs: np.ndarray,
	after_embs: np.ndarray,
	labels: np.ndarray,
	output_path: Path
) -> None:
	intra_before, inter_before = calculate_pairwise_distances(before_embs, labels)
	intra_after, inter_after = calculate_pairwise_distances(after_embs, labels)

	intra_mean_bef = np.mean(intra_before)
	inter_mean_bef = np.mean(inter_before)
	ratio_bef = intra_mean_bef / inter_mean_bef if inter_mean_bef > 0 else 0.0

	intra_mean_aft = np.mean(intra_after)
	inter_mean_aft = np.mean(inter_after)
	ratio_aft = intra_mean_aft / inter_mean_aft if inter_mean_aft > 0 else 0.0

	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

	ax1.hist(intra_before, bins=30, alpha=0.6, density=True, label="Intra-class (Same Class)", color="#3498db")
	ax1.hist(inter_before, bins=30, alpha=0.6, density=True, label="Inter-class (Diff Class)", color="#e74c3c")
	ax1.set_xlabel("Euclidean Distance")
	ax1.set_ylabel("Density")
	ax1.set_title("Distance Distribution Before Training", fontsize=11, fontweight='bold')
	ax1.legend()
	ax1.grid(alpha=0.3)
	
	text_bef = f"Intra Mean: {intra_mean_bef:.4f}\nInter Mean: {inter_mean_bef:.4f}\nRatio: {ratio_bef:.4f}"
	ax1.text(0.05, 0.72, text_bef, transform=ax1.transAxes, fontsize=9,
	         bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

	ax2.hist(intra_after, bins=30, alpha=0.6, density=True, label="Intra-class (Same Class)", color="#2ecc71")
	ax2.hist(inter_after, bins=30, alpha=0.6, density=True, label="Inter-class (Diff Class)", color="#e74c3c")
	ax2.set_xlabel("Euclidean Distance")
	ax2.set_ylabel("Density")
	ax2.set_title("Distance Distribution After Training", fontsize=11, fontweight='bold')
	ax2.legend()
	ax2.grid(alpha=0.3)
	
	text_aft = f"Intra Mean: {intra_mean_aft:.4f}\nInter Mean: {inter_mean_aft:.4f}\nRatio: {ratio_aft:.4f}"
	ax2.text(0.05, 0.72, text_aft, transform=ax2.transAxes, fontsize=9,
	         bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

	plt.suptitle("Pairwise Distance Analysis: Intra-class vs Inter-class", fontsize=13, fontweight='bold')
	plt.tight_layout()
	plt.savefig(output_path, dpi=200, bbox_inches='tight')
	plt.close()
	print(f"Distance analysis distribution saved → {output_path}")


def plot_metrics_summary(history: dict, model: nn.Module, val_loader: DataLoader, test_loader: DataLoader, device: torch.device, output_dir: Path) -> None:
	from sklearn.metrics import roc_curve, auc
	fig, axes = plt.subplots(2, 4, figsize=(20, 10))
	epochs_range = range(1, len(history["train_loss"]) + 1)

	# 1. Loss
	axes[0, 0].plot(epochs_range, history["train_loss"], label="Train Loss", color="#3498db", lw=2)
	axes[0, 0].plot(epochs_range, history["val_loss"], label="Val Loss", color="#e74c3c", lw=2)
	axes[0, 0].set_xlabel("Epoch")
	axes[0, 0].set_ylabel("Loss")
	axes[0, 0].set_title("Loss Curve")
	axes[0, 0].legend()
	axes[0, 0].grid(alpha=0.3)

	# 2. Recall@1
	axes[0, 1].plot(epochs_range, history["train_recall1"], label="Train R@1", color="#3498db", lw=2)
	axes[0, 1].plot(epochs_range, history["val_recall1"], label="Val R@1", color="#2ecc71", lw=2)
	axes[0, 1].set_xlabel("Epoch")
	axes[0, 1].set_ylabel("Recall@1")
	axes[0, 1].set_title("Recall@1 Curve")
	axes[0, 1].legend()
	axes[0, 1].grid(alpha=0.3)

	# 3. Recall@5
	axes[0, 2].plot(epochs_range, history["train_recall5"], label="Train R@5", color="#3498db", lw=2)
	axes[0, 2].plot(epochs_range, history["val_recall5"], label="Val R@5", color="#2ecc71", lw=2)
	axes[0, 2].set_xlabel("Epoch")
	axes[0, 2].set_ylabel("Recall@5")
	axes[0, 2].set_title("Recall@5 Curve")
	axes[0, 2].legend()
	axes[0, 2].grid(alpha=0.3)

	# 4. Precision@1
	axes[0, 3].plot(epochs_range, history["train_precision1"], label="Train P@1", color="#3498db", lw=2)
	axes[0, 3].plot(epochs_range, history["val_precision1"], label="Val P@1", color="#2ecc71", lw=2)
	axes[0, 3].set_xlabel("Epoch")
	axes[0, 3].set_ylabel("Precision@1")
	axes[0, 3].set_title("Precision@1 Curve")
	axes[0, 3].legend()
	axes[0, 3].grid(alpha=0.3)

	# 5. Precision@5
	axes[1, 0].plot(epochs_range, history["train_precision5"], label="Train P@5", color="#3498db", lw=2)
	axes[1, 0].plot(epochs_range, history["val_precision5"], label="Val P@5", color="#2ecc71", lw=2)
	axes[1, 0].set_xlabel("Epoch")
	axes[1, 0].set_ylabel("Precision@5")
	axes[1, 0].set_title("Precision@5 Curve")
	axes[1, 0].legend()
	axes[1, 0].grid(alpha=0.3)

	# 6. mAP
	axes[1, 1].plot(epochs_range, history["train_map"], label="Train mAP", color="#3498db", lw=2)
	axes[1, 1].plot(epochs_range, history["val_map"], label="Val mAP", color="#2ecc71", lw=2)
	axes[1, 1].set_xlabel("Epoch")
	axes[1, 1].set_ylabel("mAP")
	axes[1, 1].set_title("mAP Curve")
	axes[1, 1].legend()
	axes[1, 1].grid(alpha=0.3)

	# 7. AUC
	axes[1, 2].plot(epochs_range, history["train_auc"], label="Train AUC", color="#3498db", lw=2)
	axes[1, 2].plot(epochs_range, history["val_auc"], label="Val AUC", color="#2ecc71", lw=2)
	axes[1, 2].set_xlabel("Epoch")
	axes[1, 2].set_ylabel("AUC")
	axes[1, 2].set_title("AUC Curve")
	axes[1, 2].legend()
	axes[1, 2].grid(alpha=0.3)

	# 8. ROC Curves
	ax_roc = axes[1, 3]
	def plot_roc_helper(loader, label_str, color):
		embs, lbls = extract_all_embeddings(model, loader, device)
		n = len(lbls)
		dist_matrix = torch.cdist(embs, embs, p=2).numpy()
		pair_labels = []
		pair_scores = []
		for i in range(n):
			for j in range(i + 1, n):
				pair_labels.append(int(lbls[i] == lbls[j]))
				pair_scores.append(-dist_matrix[i, j])
		fpr, tpr, _ = roc_curve(pair_labels, pair_scores)
		roc_auc = auc(fpr, tpr)
		ax_roc.plot(fpr, tpr, color=color, lw=2, label=f"{label_str} (AUC = {roc_auc:.4f})")

	try:
		plot_roc_helper(val_loader, "Val ROC", "#2ecc71")
		plot_roc_helper(test_loader, "Test ROC", "#e74c3c")
	except Exception as e:
		print(f"Error plotting ROC curves: {e}")

	ax_roc.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
	ax_roc.set_xlim([0.0, 1.0])
	ax_roc.set_ylim([0.0, 1.05])
	ax_roc.set_xlabel("False Positive Rate")
	ax_roc.set_ylabel("True Positive Rate")
	ax_roc.set_title("ROC Curves")
	ax_roc.legend(loc="lower right")
	ax_roc.grid(alpha=0.3)

	plt.suptitle("Training Metrics Summary & ROC Analysis", fontsize=16, fontweight='bold', y=0.98)
	plt.tight_layout()
	plt.savefig(output_dir / "metrics_summary.png", dpi=200, bbox_inches="tight")
	plt.close()
	print(f"Metrics summary saved → {output_dir / 'metrics_summary.png'}")


def plot_all_metrics_per_epoch(history: dict, output_dir: Path) -> None:
	print("\n[Plot] Vẽ biểu đồ từng metric theo epoch...")
	epochs_range = range(1, len(history["train_loss"]) + 1)

	def _save_fig(ax, title: str, ylabel: str, filename: str) -> None:
		ax.set_title(title, fontsize=13, fontweight="bold")
		ax.set_xlabel("Epoch")
		ax.set_ylabel(ylabel)
		ax.legend()
		ax.grid(alpha=0.3)
		plt.tight_layout()
		plt.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
		plt.close()
		print(f"  Saved → {filename}")

	def _plot_pair(
		train_key: str, val_key: str, cross_key: str | None,
		label: str, ylabel: str,
		fname_tv: str, fname_tc: str,
		scale: float = 1.0,
	) -> None:
		tv = [v * scale for v in history[train_key]]
		vv = [v * scale for v in history[val_key]]

		# Biểu đồ 1: Train vs Val
		_, ax = plt.subplots(figsize=(8, 5))
		ax.plot(epochs_range, tv, color="#3498db", lw=2, label="Train")
		ax.plot(epochs_range, vv, color="#2ecc71", lw=2, label="Val")
		_save_fig(ax, f"{label} — Train vs Val", ylabel, fname_tv)

		# Biểu đồ 2: Train vs Val-Cross
		if cross_key and history.get(cross_key):
			vc = [v * scale for v in history[cross_key]]
			_, ax = plt.subplots(figsize=(8, 5))
			ax.plot(epochs_range, tv,   color="#3498db", lw=2, label="Train")
			ax.plot(epochs_range, vc,   color="#e67e22", lw=2, linestyle="--", label="Val-Cross")
			_save_fig(ax, f"{label} — Train vs Val-Cross", ylabel, fname_tc)

	# 1. Loss
	_, ax = plt.subplots(figsize=(8, 5))
	ax.plot(epochs_range, history["train_loss"], color="#3498db", lw=2, label="Train")
	ax.plot(epochs_range, history["val_loss"],   color="#2ecc71", lw=2, label="Val")
	_save_fig(ax, "Loss — Train vs Val", "Loss", "metric_loss.png")

	# 2. Recall@1
	_plot_pair("train_recall1", "val_recall1", "val_cross_recall1",
		"Recall@1", "Recall@1 (%)", "metric_recall1_train_val.png", "metric_recall1_train_cross.png", scale=100)

	# 3. Recall@5
	_plot_pair("train_recall5", "val_recall5", "val_cross_recall5",
		"Recall@5", "Recall@5 (%)", "metric_recall5_train_val.png", "metric_recall5_train_cross.png", scale=100)

	# 4. Precision@1
	_plot_pair("train_precision1", "val_precision1", "val_cross_precision1",
		"Precision@1", "Precision@1 (%)", "metric_precision1_train_val.png", "metric_precision1_train_cross.png", scale=100)

	# 5. Precision@5
	_plot_pair("train_precision5", "val_precision5", "val_cross_precision5",
		"Precision@5", "Precision@5 (%)", "metric_precision5_train_val.png", "metric_precision5_train_cross.png", scale=100)

	# 6. mAP
	_plot_pair("train_map", "val_map", "val_cross_map",
		"mAP", "mAP (%)", "metric_map_train_val.png", "metric_map_train_cross.png", scale=100)

	# 7. AUC
	_plot_pair("train_auc", "val_auc", "val_cross_auc",
		"AUC", "AUC", "metric_auc_train_val.png", "metric_auc_train_cross.png", scale=1)

	# 8–13. Clustering Metrics (Val only)
	clustering_cfgs = [
		("val_silhouette", "Silhouette Score (Val)",          "Silhouette Score",     "metric_silhouette.png"),
		("val_dbi",        "Davies-Bouldin Index (Val)",       "Davies-Bouldin Index", "metric_dbi.png"),
		("val_chi",        "Calinski-Harabasz Score (Val)",    "CHI Score",            "metric_chi.png"),
		("val_dunn",       "Dunn Index (Val)",                 "Dunn Index",           "metric_dunn.png"),
		("val_nmi",        "NMI (Val)",                        "NMI",                  "metric_nmi.png"),
		("val_ratio",      "Intra/Inter Ratio (Val)",          "Intra/Inter Ratio",    "metric_intra_inter_ratio.png"),
	]
	for key, title, ylabel, filename in clustering_cfgs:
		if not history.get(key):
			continue
		_, ax = plt.subplots(figsize=(8, 5))
		ax.plot(epochs_range, history[key], color="#9b59b6", lw=2, label="Val")
		_save_fig(ax, f"{title} over Epochs", ylabel, filename)

	n_charts = 1 + 6 * 2 + len(clustering_cfgs)
	print(f"[plot_all_metrics_per_epoch] Hoàn tất — {n_charts} biểu đồ đã lưu vào {output_dir}/")
