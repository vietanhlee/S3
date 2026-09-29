"""
02_specimen_variant_gan.training.explainability
===============================================
Giải thích quyết định mô hình (XAI) bằng Grad-CAM, HiResCAM, GradCAM++, EigenCAM, FinerCAM.
"""

from pathlib import Path
from typing import Optional, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from utils import find_last_conv_layer, overlay_cam_on_image, CAM_METHODS


class GradCAM:
	"""Trình sinh bản đồ kích hoạt trực quan Grad-CAM đa biến thể."""
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

	def __call__(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
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
			if self.method == "finercam":
				output = self.model(input_tensor)
				if class_idx is None:
					class_idx = int(torch.argmax(output, dim=1).item())
				prob = torch.softmax(output, dim=-1)
				output_data = output[0].detach().cpu().numpy()
				target_logit = output_data[class_idx]

				sorted_indices = np.argsort(np.abs(output_data - target_logit))
				comparison_categories = sorted_indices[1:4]
				alpha = 1.0

				wn = output[0, class_idx]
				weights = [prob[0, idx] for idx in comparison_categories]
				numerator = sum(w * (wn - alpha * output[0, idx]) for w, idx in zip(weights, comparison_categories))
				denominator = sum(weights)
				score = numerator / (denominator + 1e-9)
			else:
				output = self.model(input_tensor)
				if class_idx is None:
					class_idx = int(torch.argmax(output, dim=1).item())
				score = output[:, class_idx].sum()

			if self.activations is None:
				raise RuntimeError("GradCAM hook did not capture activations")

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


def save_gradcam_samples(
	model: nn.Module,
	df,
	eval_tf,
	device: torch.device,
	output_dir: Path,
	num_samples: int = 8,
	seed: int = 42,
) -> None:
	"""Lưu tập mẫu kiểm chứng giải thích mô hình GradCAM cho từng phương pháp CAM."""
	target_layer = find_last_conv_layer(model)
	if target_layer is None:
		print("GradCAM skipped: Không tìm thấy Conv2d layer trong model")
		return

	model.eval()
	count = min(num_samples, len(df))
	if count == 0:
		return
	batch = df.sample(n=count, random_state=seed).reset_index(drop=True)

	for method in CAM_METHODS:
		method_dir = output_dir / method
		method_dir.mkdir(parents=True, exist_ok=True)

		gradcam = GradCAM(model, target_layer, method=method)
		for i, row in batch.iterrows():
			with Image.open(row["path"]) as img:
				img = img.convert("RGB")
			input_tensor = eval_tf(img).unsqueeze(0).to(device)
			cam = gradcam(input_tensor)
			overlay = overlay_cam_on_image(img, cam)
			label = str(row["label"]).replace(" ", "_")
			out_path = method_dir / f"gradcam_{i}_{label}.png"
			overlay.save(out_path)
		gradcam.remove()
		print(f"  -> Lưu kết quả XAI ({method}) vào: {method_dir}/")
