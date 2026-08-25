"""
train_custom_semi_hard_triplet.py — Custom Genus-Aware Semi-hard Triplet Loss
================================================================================
Cải tiến Semi-hard Triplet Loss bằng cách phân tách khoảng cách theo cấp Chi (Genus):
1. Các loài CÙNG CHI (Intra-genus): Margin nhỏ hơn (margin_intra = 0.3), nhưng trọng số phạt CAO HƠN (weight_intra = 2.0)
   vì hai loài cùng chi có cấu trúc gỗ rất tương đồng, nếu xếp sai phải chịu hình phạt nặng hơn.
2. Các loài KHÁC CHI (Inter-genus): Margin tiêu chuẩn (margin_inter = 0.5), trọng số tiêu chuẩn (weight_inter = 1.0).
"""

import torch
import torch.nn as nn
from utils.common import split_genus_species
from train_base import BaseMetricTrainer

# ===== CẤU HÌNH =====
CONFIG = {
	"OUTPUT_DIR": "outputs_custom_semi_hard_triplet",
	"MARGIN_INTRA": 0.3,   # Margin cho loài cùng chi (nhỏ hơn)
	"WEIGHT_INTRA": 2.0,   # Hệ số loss phạt loài cùng chi (phạt nặng hơn)
	"MARGIN_INTER": 0.5,   # Margin cho loài khác chi (tiêu chuẩn)
	"WEIGHT_INTER": 1.0,   # Hệ số loss phạt loài khác chi
	"EPOCHS": 50,
	"PATIENCE": 25,
	"LR": 1e-4,
	"P_CLASSES": 18,
	"K_SAMPLES": 20,
}
# =====================


class CustomGenusSemiHardTripletLoss(nn.Module):
	"""Custom Genus-Aware Semi-hard Triplet Loss.

	- Intra-genus negatives (cùng chi, khác loài): margin = margin_intra, loss_weight = weight_intra
	- Inter-genus negatives (khác chi): margin = margin_inter, loss_weight = weight_inter
	"""

	def __init__(
		self,
		class_names: list[str],
		margin_intra: float = 0.3,
		weight_intra: float = 2.0,
		margin_inter: float = 0.5,
		weight_inter: float = 1.0,
	) -> None:
		super().__init__()
		self.margin_intra = margin_intra
		self.weight_intra = weight_intra
		self.margin_inter = margin_inter
		self.weight_inter = weight_inter

		# Xây dựng ánh xạ class_idx -> genus_idx
		genera = sorted(list(set(split_genus_species(c)[0] for c in class_names)))
		genus_to_idx = {g: i for i, g in enumerate(genera)}
		class_genus_indices = [genus_to_idx[split_genus_species(c)[0]] for c in class_names]
		
		# Đăng ký làm buffer tensor để tự động di chuyển theo thiết bị (GPU/CPU)
		self.register_buffer("class_to_genus", torch.tensor(class_genus_indices, dtype=torch.long))

	def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
		dist_mat = torch.cdist(embeddings, embeddings, p=2).pow(2)
		batch_size = embeddings.size(0)

		# Chuyển đổi nhãn lớp sang nhãn chi
		genus_labels = self.class_to_genus[labels]

		same_species = labels.unsqueeze(0) == labels.unsqueeze(1)
		same_genus = genus_labels.unsqueeze(0) == genus_labels.unsqueeze(1)
		diag = torch.eye(batch_size, dtype=torch.bool, device=embeddings.device)

		pos_mask = same_species & ~diag
		intra_neg_mask = (~same_species) & same_genus       # Cùng chi, khác loài
		inter_neg_mask = (~same_species) & (~same_genus)    # Khác chi

		losses = []
		for i in range(batch_size):
			pos_indices = torch.where(pos_mask[i])[0]
			if len(pos_indices) == 0:
				continue

			intra_neg_indices = torch.where(intra_neg_mask[i])[0]
			inter_neg_indices = torch.where(inter_neg_mask[i])[0]

			for p_idx in pos_indices:
				d_ap = dist_mat[i, p_idx]

				# 1. Intra-genus negative mining (cùng chi)
				if len(intra_neg_indices) > 0:
					d_an_all_intra = dist_mat[i, intra_neg_indices]
					semi_hard_intra = (d_an_all_intra > d_ap) & (d_an_all_intra < d_ap + self.margin_intra)

					if semi_hard_intra.any():
						d_an_intra = d_an_all_intra[semi_hard_intra].min()
					else:
						d_an_intra = d_an_all_intra.min()

					loss_intra = torch.clamp(d_ap - d_an_intra + self.margin_intra, min=0.0)
					if loss_intra > 0:
						losses.append(self.weight_intra * loss_intra)

				# 2. Inter-genus negative mining (khác chi)
				if len(inter_neg_indices) > 0:
					d_an_all_inter = dist_mat[i, inter_neg_indices]
					semi_hard_inter = (d_an_all_inter > d_ap) & (d_an_all_inter < d_ap + self.margin_inter)

					if semi_hard_inter.any():
						d_an_inter = d_an_all_inter[semi_hard_inter].min()
					else:
						d_an_inter = d_an_all_inter.min()

					loss_inter = torch.clamp(d_ap - d_an_inter + self.margin_inter, min=0.0)
					if loss_inter > 0:
						losses.append(self.weight_inter * loss_inter)

		if len(losses) == 0:
			return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
		return torch.stack(losses).mean()


class CustomSemiHardTripletTrainer(BaseMetricTrainer):

	def get_method_name(self) -> str:
		return "Custom Genus-Aware Semi-hard Triplet Loss"

	def get_loss_config(self) -> dict:
		return {
			"margin_intra": self.config.get("MARGIN_INTRA", 0.3),
			"weight_intra": self.config.get("WEIGHT_INTRA", 2.0),
			"margin_inter": self.config.get("MARGIN_INTER", 0.5),
			"weight_inter": self.config.get("WEIGHT_INTER", 1.0),
		}

	def build_loss(self, num_classes: int) -> nn.Module:
		return CustomGenusSemiHardTripletLoss(
			class_names=self.class_names,
			margin_intra=self.config.get("MARGIN_INTRA", 0.3),
			weight_intra=self.config.get("WEIGHT_INTRA", 2.0),
			margin_inter=self.config.get("MARGIN_INTER", 0.5),
			weight_inter=self.config.get("WEIGHT_INTER", 1.0),
		)


if __name__ == "__main__":
	trainer = CustomSemiHardTripletTrainer(CONFIG)
	trainer.run()
