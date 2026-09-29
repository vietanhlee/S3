#!/usr/bin/env python3
"""
run_pipeline.py
===============
Script điều phối (Master Orchestrator) tinh gọn cho ForensicMacroWood-CITES
(Elsevier Data in Brief), hỗ trợ kiểm định thống kê đa hạt giống (5 Random Seeds).

Quy trình tinh gọn (2 bước chính):
  1. 'assets': Sinh metadata, tính độ nét Laplacian (sigma^2 >= 100), mã băm
     mật mã SHA-256 và mã băm tri giác dHash/pHash (khử trùng lặp 2 cấp độ).
  2. 'classify': Huấn luyện baseline phân loại ConvNeXt-Tiny qua 5 random seeds
     (42, 123, 456, 789, 2024) và tự động tính toán kiểm định thống kê (Mean +- Std, 95% CI).
  - 'all': Tự động chạy tuần tự Bước 1 -> Bước 2.

Cách dùng cơ bản:
  # Chạy toàn bộ quy trình với kiểm định thống kê 5 seeds (chuẩn bài báo):
  python run_pipeline.py --all --data-dir <đường_dẫn_ảnh>

  # Chỉ chạy thử nghiệm nhanh với 1 seed:
  python run_pipeline.py --step classify --single-seed

  # Chỉ tạo assets & metadata:
  python run_pipeline.py --step assets --data-dir <đường_dẫn_ảnh>
"""

import os
import sys

# Đảm bảo in tiếng Việt có dấu an toàn tuyệt đối trên Windows terminal (cmd/powershell)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import json
import shutil
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List

from utils.statistical_testing import aggregate_multi_seed_results, print_statistical_summary_table


def run_command(cmd: List[str]) -> None:
    """Thực thi lệnh shell và dừng ngay lập tức nếu gặp lỗi."""
    print(f"\n[EXEC] {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n[!] LỖI THỰC THI: Lệnh sau trả về mã lỗi {result.returncode}:")
        print(f"    {' '.join(cmd)}")
        sys.exit(result.returncode)


def find_data_directory(requested_dir: Optional[str] = None) -> str:
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
        Path("data/S3"),
        Path("out/images")
    ])

    for cand in candidate_dirs:
        if cand.exists() and cand.is_dir():
            subdirs = [p for p in cand.iterdir() if p.is_dir()]
            if len(subdirs) >= 3:
                return str(cand)
    return requested_dir or "/kaggle/input/datasets/b23dckh002lvitanh/s3-origin/S3"


def resolve_assets_dir(requested_assets_dir: str) -> Path:
    """Xác định thư mục assets ưu tiên, tương thích ngược giữa paper_data_assets và out."""
    p = Path(requested_assets_dir)
    if not p.exists() and Path("out").exists() and (Path("out") / "splits").exists():
        return Path("out")
    return p


def run_assets_step(python_bin: str, data_dir: str, assets_dir: Path) -> None:
    """Bước 1: Sinh metadata, mã băm SHA-256, phương sai Laplacian và thẩm định rò rỉ 2 cấp độ."""
    print("\n" + "=" * 78)
    print(" [BƯỚC 1/2] SINH ASSETS, METADATA & KIỂM ĐỊNH RÒ RỈ MẪU VẬT 2 CẤP ĐỘ ")
    print("=" * 78)
    cmd = [
        python_bin, "generate_benchmark_assets.py",
        "--data-dir", data_dir,
        "--output-dir", str(assets_dir)
    ]
    run_command(cmd)


def run_audit_step(python_bin: str, assets_dir: Path) -> None:
    """Bước 1b (Tùy chọn): Kiểm định độc lập 2 cấp độ: Bitwise (SHA-256) & Tri giác (dHash/pHash)."""
    print("\n" + "=" * 80)
    print(" [KIỂM ĐỊNH 2 CẤP ĐỘ] THẨM ĐỊNH TRÙNG LẶP BITWISE (SHA-256) & TRI GIÁC (dHash/pHash) ")
    print("=" * 80)
    split_can = assets_dir / "splits" / "split_canonical.csv"
    split_dis = assets_dir / "splits" / "split_specimen_disjoint.csv"
    meta_csv = assets_dir / "metadata" / "metadata.csv"
    out_json = assets_dir / "leakage_audit" / "perceptual_audit_report.json"

    cmd = [
        python_bin, "audit_perceptual_and_embedding_similarity.py",
        "--split-csv", str(split_can),
        "--metadata-csv", str(meta_csv),
        "--output-json", str(out_json)
    ]
    if split_dis.exists():
        cmd.extend(["--compare-split-csv", str(split_dis)])

    run_command(cmd)


def run_classify_multi_seed(
    python_bin: str,
    split_csv: Path,
    metadata_csv: Path,
    output_dir: Path,
    fig_dir: Path,
    epochs: int,
    loss_mode: str,
    seeds: List[int],
    data_dir: Optional[str] = None,
    batch_size: Optional[int] = None,
    lr: Optional[float] = None
) -> None:
    """Bước 2: Huấn luyện Baseline Phân loại ConvNeXt-Tiny qua nhiều random seeds và kiểm định thống kê."""
    print("\n" + "=" * 82)
    print(f" [BƯỚC 2/2] HUẤN LUYỆN BASELINE CONVNEXT-TINY QUA {len(seeds)} RANDOM SEEDS & KIỂM ĐỊNH THỐNG KÊ ")
    print(f" Seeds: {seeds} | Loss: {loss_mode.upper()} | Epochs: {epochs}")
    print("=" * 82)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    seed_payloads = []
    loss_tag = "focal" if loss_mode in ["focal", "both"] else "ce"

    for idx, seed in enumerate(seeds, 1):
        print(f"\n>>> [SEED {idx}/{len(seeds)}: SEED = {seed}] <<<")
        seed_out_dir = output_dir / f"seed_{seed}"
        seed_out_dir.mkdir(parents=True, exist_ok=True)
        seed_fig_dir = fig_dir / f"seed_{seed}"
        seed_fig_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            python_bin, "train_classification_pipeline.py",
            "--split-csv", str(split_csv),
            "--metadata-csv", str(metadata_csv),
            "--epochs", str(epochs),
            "--output-dir", str(seed_out_dir),
            "--fig-dir", str(seed_fig_dir),
            "--seed", str(seed)
        ]
        if data_dir:
            cmd.extend(["--data-dir", data_dir])
        if batch_size:
            cmd.extend(["--batch-size", str(batch_size)])
        if lr:
            cmd.extend(["--lr", str(lr)])
        if loss_mode == "both":
            cmd.append("--run-both")
        else:
            cmd.extend(["--loss", loss_mode])

        run_command(cmd)

        # Thu thập file kết quả JSON của seed này
        res_file = seed_out_dir / f"classification_results_{loss_tag}.json"
        if res_file.exists():
            try:
                with open(res_file, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                    seed_payloads.append(payload)
            except Exception as e:
                print(f"[!] Lỗi khi nạp file kết quả seed {seed}: {e}")

    # Đồng bộ 1 bản checkpoint và confusion matrix của seed đầu tiên ra thư mục cha (tương thích ngược)
    first_seed_dir = output_dir / f"seed_{seeds[0]}"
    if first_seed_dir.exists():
        for f in first_seed_dir.glob("*.json"):
            shutil.copy(f, output_dir / f.name)
        for f in first_seed_dir.glob("*.pth"):
            shutil.copy(f, output_dir / f.name)
        first_seed_fig = fig_dir / f"seed_{seeds[0]}"
        if first_seed_fig.exists():
            for f in first_seed_fig.glob("*.*"):
                shutil.copy(f, fig_dir / f.name)

    # Tổng hợp phân tích thống kê nếu có >= 2 seeds
    if len(seed_payloads) >= 2:
        print("\n[*] Đang tổng hợp phân tích thống kê (Mean, Std, 95% Student's t CI)...")
        summary_stats = aggregate_multi_seed_results(seed_payloads, confidence=0.95)
        stat_json_path = output_dir / "multi_seed_statistical_summary.json"
        with open(stat_json_path, "w", encoding="utf-8") as f:
            json.dump(summary_stats, f, indent=2)
        print(f"[+] Đã lưu báo cáo thống kê đa hạt giống: {stat_json_path}")

        # Lưu file text Classification Report định dạng Mean +- Std
        report_txt = summary_stats.get("formatted_classification_report", "")
        if report_txt:
            stat_txt_path = output_dir / "multi_seed_classification_report.txt"
            with open(stat_txt_path, "w", encoding="utf-8") as f:
                f.write(report_txt)
            print(f"[+] Đã lưu bảng Classification Report (Mean ± Std): {stat_txt_path}")

        print_statistical_summary_table(summary_stats)
    elif len(seed_payloads) == 1:
        print("\n[+] Đã hoàn thành huấn luyện với 1 seed duy nhất.")


def main():
    parser = argparse.ArgumentParser(
        description="Master Orchestrator Pipeline tinh gọn cho ForensicMacroWood-CITES (Elsevier Data in Brief)"
    )
    parser.add_argument("--all", action="store_true",
                        help="Thực hiện toàn bộ quy trình thực nghiệm (Assets -> Multi-seed Classify)")
    parser.add_argument("--step", type=str, default="all",
                        choices=["all", "assets", "audit", "classify"],
                        help="Bước cần thực hiện: assets | audit | classify | all")
    parser.add_argument("--split-type", type=str, default="canonical",
                        choices=["canonical", "specimen_disjoint"],
                        help="Kiểu phân vùng: 'canonical' (mặc định) hoặc 'specimen_disjoint'")
    parser.add_argument("--data-dir", type=str, default="/kaggle/input/datasets/b23dckh002lvitanh/s3-origin/S3",
                        help="Đường dẫn đến thư mục chứa 19 lớp ảnh macroscopic wood")
    parser.add_argument("--assets-dir", type=str, default="paper_data_assets",
                        help="Thư mục chứa các tệp metadata, splits và leakage audit")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Kích thước batch size cho huấn luyện phân loại (mặc định 64)")
    parser.add_argument("--classify-epochs", type=int, default=22,
                        help="Số epochs cho mô hình phân loại ConvNeXt-Tiny (mặc định 22)")
    parser.add_argument("--classify-loss", type=str, default="focal",
                        choices=["focal", "cross_entropy", "both"],
                        help="Hàm mất mát phân loại: 'focal', 'cross_entropy', hoặc 'both'")
    parser.add_argument("--classify-lr", type=float, default=5e-4,
                        help="Tốc độ học cho phân loại (mặc định 5e-4)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 2024],
                        help="Danh sách các random seeds để chạy kiểm định thống kê (mặc định 5 seeds)")
    parser.add_argument("--single-seed", action="store_true",
                        help="Chỉ chạy duy nhất seed đầu tiên nếu muốn thực nghiệm nhanh")
    args = parser.parse_args()

    if args.all:
        args.step = "all"

    active_seeds = [args.seeds[0]] if args.single_seed else args.seeds

    # Tự động dò tìm thư mục dữ liệu thật trên Kaggle/Colab/Local
    detected_data_dir = find_data_directory(args.data_dir)
    if Path(detected_data_dir).exists():
        args.data_dir = detected_data_dir
        print(f"[*] Thư mục dữ liệu ảnh phát hiện: {args.data_dir}")

    assets_dir = resolve_assets_dir(args.assets_dir)
    print(f"[*] Thư mục assets dữ liệu sử dụng: {assets_dir}")

    python_bin = sys.executable

    # BƯỚC 1: TẠO ASSETS, TÍNH TOÁN SHARPNESS, HASH SHA-256 VÀ THẨM ĐỊNH RÒ RỈ
    if args.step in ["all", "assets"]:
        run_assets_step(python_bin, args.data_dir, assets_dir)

    # BƯỚC 1b: KIỂM ĐỊNH ĐỘC LẬP 2 CẤP ĐỘ (SHA-256 & dHash/pHash)
    if args.step in ["audit"]:
        run_audit_step(python_bin, assets_dir)

    # BƯỚC 2: HUẤN LUYỆN CLASSIFICATION BASELINE (CONVNEXT-TINY QUA CÁC SEEDS)
    if args.step in ["all", "classify"]:
        split_file = "split_canonical.csv" if args.split_type == "canonical" else "split_specimen_disjoint.csv"
        split_csv = assets_dir / "splits" / split_file
        metadata_csv = assets_dir / "metadata" / "metadata.csv"
        out_p = Path("baseline_outputs")
        fig_p = Path("paper_data/fig") / args.split_type

        print(f"\n[*] Đang thực thi phân loại trên split [{args.split_type}] với {len(active_seeds)} seeds -> {out_p}...")
        run_classify_multi_seed(
            python_bin=python_bin,
            split_csv=split_csv,
            metadata_csv=metadata_csv,
            output_dir=out_p,
            fig_dir=fig_p,
            epochs=args.classify_epochs,
            loss_mode=args.classify_loss,
            seeds=active_seeds,
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            lr=args.classify_lr
        )

    print("\n" + "=" * 78)
    print(" [✓] HOÀN TẤT QUY TRÌNH THỰC NGHIỆM! DỮ LIỆU ĐÃ ĐỒNG BỘ CHUẨN PUBLICATION.")
    print("=" * 78)


if __name__ == "__main__":
    main()
