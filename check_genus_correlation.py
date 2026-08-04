"""
check_genus_correlation.py
==========================
Script phân tích thực nghiệm mối tương quan giữa Loài (Species) và Chi (Genus) trong không gian nhúng đặc trưng thô.
Sử dụng Silhouette Score và Khoảng cách Cosine để kiểm chứng giả thuyết:
"Các loài cùng chi có thực sự nằm gần nhau hơn khác chi trong không gian đặc trưng thô không?"

Cách chạy:
python check_genus_correlation.py
"""

import os
import json
import random
from pathlib import Path
from PIL import Image

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from timm.data import resolve_data_config

from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

# ===== CẤU HÌNH =====
ROOT_DIR = r"/kaggle/input/datasets/b23dckh002lvitanh/s3-origin/S3"
CHECKPOINT_PATH = Path("outputs_final/best_model_convnext_tiny.pth")
CLASS_INDICES_PATH = Path("outputs_final/class_indices.json")
OUTPUT_DIR = Path("outputs_analysis")
SEED = 42
BATCH_SIZE = 128
# ====================

def set_seed(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

def split_genus_species(class_name: str) -> tuple[str, str]:
	parts = class_name.strip().split()
	genus = parts[0] if parts else "Unknown"
	species = " ".join(parts[1:]) if len(parts) > 1 else "sp"
	return genus, species

class SimpleImageDataset(Dataset):
	def __init__(self, df: pd.DataFrame, transform=None) -> None:
		self.df = df.reset_index(drop=True)
		self.transform = transform

	def __len__(self) -> int:
		return len(self.df)

	def __getitem__(self, idx: int):
		row = self.df.iloc[idx]
		with Image.open(row["path"]) as img:
			img = img.convert("RGB")
		if self.transform:
			img = self.transform(img)
		return img, row["label"], row["genus"]

@torch.no_grad()
def extract_embeddings(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, list[str], list[str]]:
	model.eval()
	all_embs = []
	all_species = []
	all_genera = []
	
	for images, species, genera in tqdm(loader, desc="Trích xuất Embeddings"):
		images = images.to(device)
		# Trích xuất embeddings đặc trưng (đã pool) trước classifier head
		features = model(images)
		# Chuẩn hóa L2
		features = torch.nn.functional.normalize(features, p=2, dim=1)
		
		all_embs.append(features.cpu().numpy())
		all_species.extend(species)
		all_genera.extend(genera)
		
	return np.concatenate(all_embs, axis=0), all_species, all_genera

def main() -> None:
	set_seed(SEED)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Sử dụng thiết bị: {device}")

	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

	# 1. Quét dữ liệu ảnh
	print("\n[Bước 1] Quét dữ liệu...")
	if not os.path.exists(ROOT_DIR):
		# Fallback nếu chạy ở cục bộ khác
		local_dir = Path("S3")
		if local_dir.exists():
			globals()["ROOT_DIR"] = str(local_dir)
			print(f"Fallback sang thư mục dữ liệu cục bộ: {ROOT_DIR}")
		else:
			raise FileNotFoundError(f"Không tìm thấy thư mục dữ liệu tại: {ROOT_DIR}")

	# Quét các file ảnh
	extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
	samples = []
	for root, _, files in os.walk(ROOT_DIR):
		for f in files:
			if f.lower().endswith(extensions):
				path = Path(root) / f
				label = path.parent.name
				if label not in ["Pterocarpus sp", "Peltogyne pubescens"]:  # Loại bỏ các class không phù hợp theo chuẩn
					genus, species = split_genus_species(label)
					samples.append({"path": str(path.absolute()), "label": label, "genus": genus})

	if not samples:
		raise ValueError("Không tìm thấy mẫu ảnh gỗ hợp lệ nào.")

	df = pd.DataFrame(samples)
	print(f"Tìm thấy: {len(df)} ảnh gỗ.")
	print(f"Số lượng loài (Species): {df['label'].nunique()}")
	print(f"Số lượng chi (Genus): {df['genus'].nunique()}")
	print("Danh sách các Chi gỗ có trong dữ liệu:")
	for g, count in df['genus'].value_counts().items():
		print(f"  - Chi {g}: {count} ảnh")

	# 2. Khởi tạo mô hình
	print("\n[Bước 2] Thiết lập mô hình trích xuất đặc trưng...")
	model_name = "convnext_tiny"
	
	# Xác định số lượng class để build model khớp state_dict
	num_classes = df["label"].nunique()
	if CLASS_INDICES_PATH.exists():
		with open(CLASS_INDICES_PATH, "r", encoding="utf-8") as f:
			class_to_idx = json.load(f)
			num_classes = len(class_to_idx)
			print(f"Load cấu hình: Tìm thấy {num_classes} lớp từ {CLASS_INDICES_PATH}")

	# Thử load checkpoint phân loại thô đã lưu
	loaded_chkpt = False
	if CHECKPOINT_PATH.exists():
		try:
			print(f"Đang cố gắng load checkpoint phân loại thô từ: {CHECKPOINT_PATH}...")
			# Khởi tạo mô hình đầy đủ lớp phân loại để nạp trọng số
			model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
			state_dict = torch.load(CHECKPOINT_PATH, map_location="cpu")
			model.load_state_dict(state_dict)
			
			# Chuyển đổi mô hình thành bộ trích xuất đặc trưng (loại bỏ classifier head)
			model.reset_classifier(0)
			model = model.to(device)
			loaded_chkpt = True
			print("  -> Load checkpoint thành công! Dùng đặc trưng của mô hình Classifier đã huấn luyện.")
		except Exception as e:
			print(f"[Warning] Không thể load checkpoint phân loại: {e}. Fallback sang model ImageNet pre-trained.")
	
	if not loaded_chkpt:
		print("Đang khởi tạo mô hình ConvNeXt-Tiny pre-trained từ ImageNet làm đặc trưng thô đại diện...")
		model = timm.create_model(model_name, pretrained=True, num_classes=0)
		model = model.to(device)

	# Cấu hình transforms phù hợp với model
	cfg = resolve_data_config({}, model=model)
	img_size = cfg.get("input_size", (3, 224, 224))[-1]
	mean = cfg.get("mean", (0.485, 0.456, 0.406))
	std = cfg.get("std", (0.229, 0.224, 0.225))
	
	transform = transforms.Compose([
		transforms.Resize((img_size, img_size)),
		transforms.ToTensor(),
		transforms.Normalize(mean, std)
	])

	dataset = SimpleImageDataset(df, transform=transform)
	loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

	# 3. Trích xuất đặc trưng
	print("\n[Bước 3] Tiến hành trích xuất embeddings...")
	embs, species_list, genera_list = extract_embeddings(model, loader, device)
	print(f"Trích xuất thành công. Shape của embeddings: {embs.shape}")

	# 4. Tính toán Metrics trắc lượng học
	print("\n[Bước 4] Tính toán các chỉ số phân tích hình học...")
	
	# Silhouette Score
	print("  -> Đang tính Silhouette Score theo nhãn Loài (Species)...")
	sil_species = silhouette_score(embs, species_list, metric="cosine")
	
	print("  -> Đang tính Silhouette Score theo nhãn Chi (Genus)...")
	sil_genus = silhouette_score(embs, genera_list, metric="cosine")

	# Tính toán ma trận khoảng cách Cosine
	# Vì embs đã được chuẩn hóa L2, nên similarity = embs @ embs.T
	print("  -> Tính toán ma trận khoảng cách Cosine...")
	similarity_matrix = np.dot(embs, embs.T)
	distance_matrix = 1.0 - similarity_matrix
	
	# Tránh lấy đường chéo chính (bằng 0)
	np.fill_diagonal(distance_matrix, np.nan)

	# Chia nhóm khoảng cách chéo
	intra_genus_dists = []
	inter_genus_dists = []
	
	n_samples = len(df)
	species_arr = np.array(species_list)
	genus_arr = np.array(genera_list)
	
	for i in tqdm(range(n_samples), desc="Phân loại khoảng cách"):
		genus_i = genus_arr[i]
		species_i = species_arr[i]
		
		# Cùng Chi nhưng khác Loài
		mask_intra = (genus_arr == genus_i) & (species_arr != species_i)
		# Khác Chi hoàn toàn
		mask_inter = (genus_arr != genus_i)
		
		dists_i = distance_matrix[i]
		
		# Lọc các khoảng cách hợp lệ
		intra_dists = dists_i[mask_intra]
		inter_dists = dists_i[mask_inter]
		
		intra_genus_dists.extend(intra_dists[~np.isnan(intra_dists)].tolist())
		inter_genus_dists.extend(inter_dists[~np.isnan(inter_dists)].tolist())

	mean_intra = np.mean(intra_genus_dists)
	std_intra = np.std(intra_genus_dists)
	mean_inter = np.mean(inter_genus_dists)
	std_inter = np.std(inter_genus_dists)
	
	# Tỷ lệ khoảng cách chéo
	ratio = mean_intra / (mean_inter + 1e-9)

	print("\n" + "="*50)
	print("KẾT QUẢ PHÂN TÍCH THỰC NGHIỆM:")
	print("="*50)
	print(f"1. Silhouette Score theo Loài (Species) : {sil_species:.4f}")
	print(f"2. Silhouette Score theo Chi (Genus)    : {sil_genus:.4f}")
	print(f"3. Khoảng cách Cosine trung bình:")
	print(f"   - Cùng Chi, Khác Loài (Intra-genus)  : {mean_intra:.4f} ± {std_intra:.4f}")
	print(f"   - Khác Chi hoàn toàn (Inter-genus)   : {mean_inter:.4f} ± {std_inter:.4f}")
	print(f"   - Tỷ số Khoảng cách (Intra / Inter)  : {ratio:.4f}")
	print("="*50)

	# 5. Phân tích đề xuất khoa học tự động
	print("\n[Bước 5] Đánh giá giả thuyết và Đề xuất thiết kế loss:")
	
	# Định nghĩa các ngưỡng trắc lượng để kiểm chứng
	is_clustered_by_genus = (sil_genus > 0.10)
	has_clear_margin = (ratio < 0.90)  # Cùng chi gần nhau hơn khác chi ít nhất 10%
	
	print(f"  - Đánh giá 1: Mức độ gom cụm của Chi tự nhiên: " + ("TỐT" if is_clustered_by_genus else "YẾU"))
	print(f"  - Đánh giá 2: Khoảng cách cùng Chi gần hơn khác Chi: " + (f"ĐÚNG (nhỏ hơn {(1-ratio)*100:.1f}%)" if has_clear_margin else "SAI / KHÔNG RÕ RỆT"))
	
	if is_clustered_by_genus and has_clear_margin:
		print("\n  ==> KẾT LUẬN & ĐỀ XUẤT:")
		print("      [KHUYẾN NGHỊ] GIẢ THUYẾT ĐÚNG. Các loài cùng chi phân bố gần nhau tự nhiên trong không gian thô.")
		print("      Việc áp dụng hàm Loss phân cấp (Hierarchical loss hoặc Genus-based margin) là an toàn.")
		print("      Nó sẽ củng cố xu hướng hình học có sẵn, giúp mô hình phạt nặng hơn lỗi nhầm lẫn khác chi.")
	else:
		print("\n  ==> KẾT LUẬN & ĐỀ XUẤT:")
		print("      [CẢNH BÁO] GIẢ THUYẾT SAI HOẶC TƯƠNG QUAN YẾU.")
		print("      Các loài cùng Chi không thực sự nằm gần nhau tự nhiên trong không gian đặc trưng thô.")
		print("      Nếu ép buộc cấu trúc margin theo Chi cứng nhắc, bạn có thể phá vỡ phân phối tự nhiên,")
		print("      gây quá khớp (overfitting) hoặc làm giảm sụt nghiêm trọng độ chính xác phân loại loài (Species Accuracy).")
		print("      -> Khuyên dùng: Nên giữ nguyên hàm Loss phân loại chuẩn và tăng cường Data Augmentation.")

	# 6. Trực quan hóa bằng t-SNE
	print("\n[Bước 6] Tiến hành giảm chiều t-SNE và vẽ biểu đồ phân bố...")
	# Giảm chiều xuống 2D
	tsne = TSNE(n_components=2, random_state=SEED, metric="cosine")
	embs_2d = tsne.fit_transform(embs)
	
	# Lấy danh sách màu sắc cho Genus và Species
	unique_genera = sorted(list(set(genera_list)))
	genus_color_map = {g: plt.cm.tab10(i % 10) for i, g in enumerate(unique_genera)}
	
	unique_species = sorted(list(set(species_list)))
	species_color_map = {s: plt.cm.tab20(i % 20) for i, s in enumerate(unique_species)}
	
	fig, axes = plt.subplots(1, 2, figsize=(18, 8))
	
	# Đồ thị 1: Tô màu theo Chi (Genus)
	ax1 = axes[0]
	for genus in unique_genera:
		mask = np.array(genera_list) == genus
		ax1.scatter(
			embs_2d[mask, 0], embs_2d[mask, 1],
			label=genus, color=genus_color_map[genus],
			alpha=0.7, edgecolors="none", s=25
		)
	ax1.set_title("Phân bố đặc trưng thô - Tô màu theo Chi (Genus)", fontsize=13, fontweight="bold")
	ax1.legend(loc="best", bbox_to_anchor=(1, 1), title="Genus")
	ax1.grid(True, linestyle="--", alpha=0.5)

	# Đồ thị 2: Tô màu theo Loài (Species)
	ax2 = axes[1]
	for species in unique_species:
		mask = np.array(species_list) == species
		ax2.scatter(
			embs_2d[mask, 0], embs_2d[mask, 1],
			label=species, color=species_color_map[species],
			alpha=0.6, edgecolors="none", s=15
		)
	ax2.set_title("Phân bố đặc trưng thô - Tô màu theo Loài (Species)", fontsize=13, fontweight="bold")
	ax2.grid(True, linestyle="--", alpha=0.5)
	
	plt.suptitle(
		f"Phân Tích t-SNE Đặc Trưng Thô (Classifier Model)\n"
		f"Silhouette Genus: {sil_genus:.4f} | Silhouette Species: {sil_species:.4f}",
		fontsize=15, fontweight="bold", y=0.98
	)
	
	plt.tight_layout()
	out_img_path = OUTPUT_DIR / "genus_correlation_tsne.png"
	plt.savefig(out_img_path, dpi=200, bbox_inches="tight")
	plt.close()
	
	print(f"\n[Hoàn tất] Đã lưu biểu đồ phân bố t-SNE tại: {out_img_path}")
	print("="*80)

if __name__ == "__main__":
	main()
