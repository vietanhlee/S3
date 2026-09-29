"""
02_specimen_variant_gan.train_governed_baseline
===============================================
CLI runner huấn luyện mô hình cơ sở theo phân hoạch chuẩn có kiểm soát (Governed Baseline).
Sử dụng End Version Split (PP Governed), trích xuất đặc trưng Swin + EfficientNetV2,
tối ưu hoá bằng Focal Loss kết hợp Cosine Annealing LR Schedule.
"""

import os
import json
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from timm.data import resolve_data_config

from utils import (
	set_seed,
	get_device,
	collect_image_samples,
	build_dataframe,
	log_split_summary,
	eda_split_class_distribution,
	ImageListDataset,
	build_transforms,
	summarize_model,
)
from partitioning import (
	validate_split,
	compute_embeddings_v2,
	end_version_split,
	SPLIT_CONFIG,
)
from training import (
	FocalLoss,
	train_model,
	build_model,
	plot_training_curves,
	evaluate_and_report,
	save_gradcam_samples,
)


def parse_args():
	parser = argparse.ArgumentParser(description="Huấn luyện Governed Baseline cho phân loại gỗ vi mô")
	parser.add_argument("--data_dir", type=str, default=r"/kaggle/input/datasets/b23dckh002lvitanh/s3-origin/S3",
						help="Đường dẫn đến thư mục chứa dữ liệu ảnh gỗ")
	parser.add_argument("--output_dir", type=str, default="outputs_final",
						help="Thư mục lưu mô hình và kết quả đánh giá")
	parser.add_argument("--model_name", type=str, default="convnext_tiny",
						help="Kiến trúc mạng (từ timm)")
	parser.add_argument("--epochs", type=int, default=22, help="Số epoch huấn luyện tối đa")
	parser.add_argument("--batch_size", type=int, default=128, help="Kích thước batch")
	parser.add_argument("--lr", type=float, default=5e-4, help="Tốc độ học")
	parser.add_argument("--weight_decay", type=float, default=1e-2, help="Hệ số weight decay")
	parser.add_argument("--freeze_ratio", type=float, default=0.90, help="Tỷ lệ đóng băng các lớp đầu của backbone")
	parser.add_argument("--focal_gamma", type=float, default=2.0, help="Tham số gamma cho Focal Loss")
	parser.add_argument("--focal_alpha", type=float, default=0.25, help="Tham số alpha cho Focal Loss")
	parser.add_argument("--train_ratio", type=float, default=0.60, help="Tỷ lệ tập Train")
	parser.add_argument("--val_ratio", type=float, default=0.20, help="Tỷ lệ tập Val")
	parser.add_argument("--seed", type=int, default=42, help="Random seed")
	parser.add_argument("--patience", type=int, default=50, help="Early stopping patience")
	parser.add_argument("--enable_gradcam", action="store_true", help="Sinh các mẫu trực quan hoá Grad-CAM sau khi test")
	return parser.parse_args()


def main():
	args = parse_args()
	set_seed(args.seed)
	device = get_device()
	print(f"Device thực thi: {device}")

	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	# Khởi tạo WandB nếu có cấu hình
	use_wandb = False
	try:
		from dotenv import load_dotenv
		import wandb
		load_dotenv()
		api_key = os.getenv("WANDB_API_KEY")
		if api_key:
			wandb.login(key=api_key)
			run_name = f"{args.model_name.lower().replace('_', '-')}-governed-baseline"
			wandb.init(
				project="S3-Wood-Classification-Base",
				name=run_name,
				config=vars(args),
			)
			use_wandb = True
			print(f"[WandB] Đã kết nối dự án 'S3-Wood-Classification-Base' (run: {run_name})")
	except Exception as e:
		print(f"[WandB Notice] Chạy chế độ offline: {e}")

	# 1. Thu thập dữ liệu và lọc lớp
	print("\n[Bước 1] Thu thập ảnh mẫu và tiền xử lý nhãn...")
	samples = collect_image_samples(args.data_dir)
	if not samples:
		raise ValueError(f"Không tìm thấy ảnh nào trong {args.data_dir}")

	df = build_dataframe(samples)
	excluded_classes = ["Pterocarpus sp", "Peltogyne pubescens"]
	df_filtered = df[~df["label"].isin(excluded_classes)].reset_index(drop=True)
	print(f"Tổng số ảnh đạt chuẩn: {len(df_filtered)} (trên {df_filtered['label'].nunique()} loài)")

	class_names = sorted(df_filtered["label"].unique().tolist())
	class_to_idx = {name: i for i, name in enumerate(class_names)}
	with open(output_dir / "class_indices.json", "w", encoding="utf-8") as f:
		json.dump(class_to_idx, f, indent=2, ensure_ascii=False)

	# 2. Trích xuất đặc trưng phục vụ phân hoạch
	print("\n[Bước 2] Trích xuất đặc trưng kiểm soát (EfficientNetV2-M + Swin-Large)...")
	embs_eff = compute_embeddings_v2(df_filtered, "tf_efficientnetv2_m_in21k", batch_size=args.batch_size, device=device)
	embs_swin = compute_embeddings_v2(df_filtered, "swin_large_patch4_window7_224", batch_size=args.batch_size, device=device)

	# 3. Phân hoạch dữ liệu chuẩn
	print("\n[Bước 3] Thực hiện End Version Governed Split...")
	df_train, df_val, df_test = end_version_split(
		df_filtered,
		embs_eff,
		embs_swin,
		train_ratio=args.train_ratio,
		val_ratio=args.val_ratio,
		seed=args.seed,
	)

	validate_split(df_filtered, df_train, df_val, df_test, "Governed_End_Version")
	log_split_summary(df_filtered, df_train, df_val, df_test)

	eda_split_class_distribution(
		df_train, df_val, df_test,
		"Governed End Version - Class Distribution",
		output_dir / "eda_split_end_version.png",
	)

	# 4. Chuẩn bị mô hình và DataLoader
	print(f"\n[Bước 4] Khởi tạo mô hình {args.model_name}...")
	model = build_model(num_classes=len(class_names), model_name=args.model_name, freeze_ratio=args.freeze_ratio)
	summary = summarize_model(model)
	print(f"  -> Tổng tham số: {summary['total_params']:,} | Trainable: {summary['trainable_params']:,} | Frozen: {summary['frozen_params']:,}")
	model = model.to(device)

	cfg = resolve_data_config({}, model=model)
	img_size = cfg.get("input_size", (3, 224, 224))[-1]
	mean = cfg.get("mean", (0.485, 0.456, 0.406))
	std = cfg.get("std", (0.229, 0.224, 0.225))
	train_tf, eval_tf = build_transforms(img_size, mean, std)

	num_workers = min(4, os.cpu_count() or 1)
	train_loader = DataLoader(ImageListDataset(df_train, class_to_idx, transform=train_tf),
							  batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
	val_loader = DataLoader(ImageListDataset(df_val, class_to_idx, transform=eval_tf),
							batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
	test_loader = DataLoader(ImageListDataset(df_test, class_to_idx, transform=eval_tf),
							 batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

	criterion = FocalLoss(gamma=args.focal_gamma, alpha=args.focal_alpha)
	optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

	# 5. Huấn luyện
	print("\n[Bước 5] Bắt đầu huấn luyện...")
	history = train_model(
		model, train_loader, val_loader, optimizer, criterion, device,
		epochs=args.epochs, patience=args.patience, output_dir=output_dir, scheduler=scheduler,
	)
	plot_training_curves(history, output_dir)

	# Tải checkpoint tốt nhất để đánh giá
	best_ckpt = output_dir / f"best_model_{args.model_name}.pth"
	if best_ckpt.exists():
		model.load_state_dict(torch.load(best_ckpt, map_location=device, weights_only=True))
		print(f"  -> Tải thành công best checkpoint từ {best_ckpt}")

	# 6. Đánh giá & Báo cáo
	print("\n[Bước 6] Đánh giá định lượng trên Validation và Test Sets...")
	evaluate_and_report(model, val_loader, device, class_names, output_dir, prefix="val")
	evaluate_and_report(model, test_loader, device, class_names, output_dir, prefix="test")

	# 7. Trực quan hoá XAI nếu có yêu cầu
	if args.enable_gradcam:
		print("\n[Bước 7] Trực quan hoá giải thích quyết định mô hình (XAI)...")
		save_gradcam_samples(model, df_test, eval_tf, device, output_dir / "xai_gradcam", num_samples=8, seed=args.seed)

	# Lưu tóm tắt kết quả
	summary_result = {
		"model_name": args.model_name,
		"epochs": args.epochs,
		"best_val_acc": history.get("best_val_acc", 0.0),
		"train_size": len(df_train),
		"val_size": len(df_val),
		"test_size": len(df_test),
	}
	with open(output_dir / "final_summary.json", "w", encoding="utf-8") as f:
		json.dump(summary_result, f, indent=2, ensure_ascii=False)

	if use_wandb:
		try:
			import wandb
			artifact = wandb.Artifact(name="governed-baseline-artifacts", type="model_and_reports")
			if best_ckpt.exists():
				artifact.add_file(str(best_ckpt))
			for txt_file in output_dir.glob("*.txt"):
				artifact.add_file(str(txt_file))
			wandb.log_artifact(artifact)
			wandb.finish()
		except Exception:
			pass

	print(f"\n✓ Hoàn tất phiên huấn luyện! Kết quả toàn vẹn được lưu trữ tại: {output_dir.resolve()}")


if __name__ == "__main__":
	main()
