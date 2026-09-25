#!/usr/bin/env python3
"""
run_pipeline.py
===============
Script điều phối (Orchestrator) toàn bộ luồng công việc cho IC4SDMacroWood:
  Bước 1: Sinh metadata, tính Laplacian variance, SHA-256 và thẩm định rò rỉ
  Bước 2: Huấn luyện mô hình phân loại cơ sở ConvNeXt-Tiny + Focal Loss
  Bước 3: Huấn luyện và đánh giá Deep Metric Learning (Contrastive / Triplet / SupCon)

Cách dùng:
  # Chạy toàn bộ quy trình:
  python run_pipeline.py --all --data-dir <đường_dẫn_ảnh>

  # Hoặc chạy từng bước:
  python run_pipeline.py --step assets --data-dir <đường_dẫn_ảnh>
  python run_pipeline.py --step classify --epochs 22
  python run_pipeline.py --step metric --epochs 30
"""

import sys
import argparse
import subprocess
from pathlib import Path


def run_command(cmd: list):
    print(f"\n[EXEC] {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[!] Lỗi khi thực thi lệnh: {' '.join(cmd)}")
        sys.exit(result.returncode)


def find_data_directory(requested_dir: str = None) -> str:
    """Tự động tìm kiếm thư mục chứa dữ liệu ảnh trên Kaggle, Colab hoặc Local."""
    candidate_dirs = []
    if requested_dir:
        candidate_dirs.append(Path(requested_dir))

    candidate_dirs.extend([
        Path("/kaggle/input/datasets/b23dckh002lvitanh/s3-origin/S3"),
        Path("/kaggle/input/datasets/b23dckh002lvitanh/s3-origin"),
        Path("/kaggle/input/s3-origin/S3"),
        Path("/kaggle/input/s3-origin"),
        Path("/kaggle/input/s3/S3"),
        Path("/kaggle/input/s3"),
        Path("./S3"),
        Path("../S3"),
        Path("data/S3")
    ])

    for cand in candidate_dirs:
        if cand.exists() and cand.is_dir():
            subdirs = [p for p in cand.iterdir() if p.is_dir()]
            if len(subdirs) >= 3:
                return str(cand)
    return requested_dir or "/kaggle/input/datasets/b23dckh002lvitanh/s3-origin/S3"


def main():
    parser = argparse.ArgumentParser(description="Điều phối toàn bộ Pipeline thực nghiệm IC4SDMacroWood")
    parser.add_argument("--all", action="store_true",
                        help="Chạy toàn bộ quy trình thực nghiệm (tương đương --step all)")
    parser.add_argument("--step", type=str, default="all", choices=["all", "assets", "classify", "metric"],
                        help="Bước cần thực hiện: assets | classify | metric | all")
    parser.add_argument("--data-dir", type=str, default="/kaggle/input/datasets/b23dckh002lvitanh/s3-origin/S3",
                        help="Đường dẫn đến thư mục chứa 19 lớp ảnh macro")
    parser.add_argument("--classify-epochs", type=int, default=30, help="Số epochs cho bài toán phân loại")
    parser.add_argument("--classify-loss", type=str, default="focal", choices=["focal", "cross_entropy", "both"],
                        help="Hàm mất mát phân loại: 'focal', 'cross_entropy', hoặc 'both' (chạy cả hai để so sánh)")
    parser.add_argument("--metric-epochs", type=int, default=40, help="Số epochs cho Semi-Hard Triplet Loss")
    parser.add_argument("--metric-margin", type=float, default=0.5, help="Margin d^2 cho Semi-Hard Triplet Loss")
    parser.add_argument("--metric-k-samples", type=int, default=4, help="Số mẫu K của từng lớp trong batch (mặc định 4)")
    parser.add_argument("--extract-embeddings", action="store_true", help="Trích xuất convnext_tiny.npy khi tạo assets")
    args = parser.parse_args()

    # Nếu truyền cờ --all, tự động đặt step là 'all'
    if args.all:
        args.step = "all"

    # Tự động dò tìm thư mục dữ liệu thật trên Kaggle/Colab nếu đường dẫn truyền vào chưa khớp
    detected_data_dir = find_data_directory(args.data_dir)
    if Path(detected_data_dir).exists():
        args.data_dir = detected_data_dir
        print(f"[*] Thư mục dữ liệu sử dụng: {args.data_dir}")

    python_bin = sys.executable

    # BƯỚC 1: TẠO ASSETS, TÍNH LAPLACIAN, SHA256 & METADATA
    if args.step in ["all", "assets"]:
        print("\n" + "=" * 70)
        print(" [BƯỚC 1/3] SINH METADATA, TÍNH LAPLACIAN SHARPNESS & THẨM ĐỊNH RÒ RỈ ")
        print("=" * 70)
        cmd_assets = [
            python_bin, "generate_benchmark_assets.py",
            "--data-dir", args.data_dir,
            "--output-dir", "paper_data_assets"
        ]
        if args.extract_embeddings:
            cmd_assets.append("--extract-embeddings")
        run_command(cmd_assets)

    # BƯỚC 2: HUẤN LUYỆN CLASSIFICATION BASELINE (CONVNEXT-TINY)
    if args.step in ["all", "classify"]:
        print("\n" + "=" * 70)
        print(" [BƯỚC 2/3] HUẤN LUYỆN CONVNEXT-TINY BASELINE (SUPERVISED CLASSIFICATION) ")
        print("=" * 70)
        cmd_classify = [
            python_bin, "train_classification_pipeline.py",
            "--split-csv", "paper_data_assets/splits/split_canonical.csv",
            "--epochs", str(args.classify_epochs),
            "--output-dir", "baseline_outputs",
            "--fig-dir", "paper_data/fig"
        ]
        if args.classify_loss == "both":
            cmd_classify.append("--run-both")
        else:
            cmd_classify.extend(["--loss", args.classify_loss])
        run_command(cmd_classify)

    # BƯỚC 3: HUẤN LUYỆN METRIC REPRESENTATION LEARNING (SEMI-HARD TRIPLET LOSS)
    if args.step in ["all", "metric"]:
        print("\n" + "=" * 70)
        print(" [BƯỚC 3/3] HUẤN LUYỆN DEEP METRIC LEARNING (SEMI-HARD TRIPLET LOSS) ")
        print("=" * 70)
        cmd_metric = [
            python_bin, "train_metric_learning_pipeline.py",
            "--split-csv", "paper_data_assets/splits/split_canonical.csv",
            "--loss", "semihard_triplet",
            "--margin", str(args.metric_margin),
            "--epochs", str(args.metric_epochs),
            "--k-samples", str(args.metric_k_samples),
            "--output-dir", "metric_outputs",
            "--fig-dir", "paper_data/fig"
        ]
        run_command(cmd_metric)

    print("\n" + "=" * 70)
    print(" [✓] HOÀN TẤT TOÀN BỘ QUY TRÌNH THỰC NGHIỆM! DỮ LIỆU ĐÃ ĐỒNG BỘ VÀO BÀI BÁO.")
    print("=" * 70)


if __name__ == "__main__":
    main()
