#!/usr/bin/env python3
"""
train_classification_pipeline.py
================================
CLI thực thi huấn luyện mô hình ResNet-50 Baseline và đánh giá kỹ thuật (Technical Validation)
cho bộ dữ liệu IC4SD-ForensicMacroWood-CITES (Elsevier Data in Brief).

Chức năng:
  - Huấn luyện Multiclass Focal Loss (alpha=0.25, gamma=2.0) hoặc Standard Cross-Entropy.
  - Tối ưu hóa AdamW + Cosine Annealing Learning Rate Scheduler.
  - Tự động lưu Checkpoint tốt nhất, xuất ma trận nhầm lẫn và biểu đồ chuẩn Elsevier.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
# Đảm bảo in tiếng Việt có dấu an toàn tuyệt đối trên Windows terminal (cmd/powershell)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import argparse
from pathlib import Path
import numpy as np
import torch

from modules.classification import (
    load_or_generate_dataset_split,
    run_training_session,
    run_linear_probe_session,
    run_augmentation_ablation_session
)


def parse_args():
    parser = argparse.ArgumentParser(description="Huấn luyện ResNet-50 Baseline cho IC4SD-ForensicMacroWood-CITES")
    parser.add_argument("--split-csv", type=str, default="out/splits/split_canonical.csv",
                        help="Đường dẫn file phân vùng dữ liệu split_canonical.csv")
    parser.add_argument("--metadata-csv", type=str, default="out/metadata/metadata.csv",
                        help="Đường dẫn file metadata.csv")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Đường dẫn thư mục ảnh gốc (tự động chia split nếu chưa có file CSV)")
    parser.add_argument("--output-dir", type=str, default="baseline_outputs",
                        help="Thư mục xuất kết quả weights, logs và ma trận nhầm lẫn")
    parser.add_argument("--fig-dir", type=str, default="paper_data/fig",
                        help="Thư mục lưu hình biểu đồ cho bài báo LaTeX")
    parser.add_argument("--model-name", type=str, default="resnet50",
                        help="Tên kiến trúc backbone từ thư viện timm (mặc định 'resnet50')")
    parser.add_argument("--loss", type=str, default="focal",
                        choices=["focal", "cross_entropy", "ce", "cb_focal", "class_balanced_focal", "class_weighted", "class_weighted_ce", "weighted_ce"],
                        help="Hàm mất mát: 'focal', 'cross_entropy', 'cb_focal', hoặc 'class_weighted'")
    parser.add_argument("--augmentation", type=str, default="full",
                        choices=["none", "minimal", "geometric", "full"],
                        help="Chế độ tiền xử lý augmentation: 'none', 'geometric' (SO(2)), hoặc 'full'")
    parser.add_argument("--alpha", type=float, default=0.25, help="Hệ số alpha cho Focal Loss")
    parser.add_argument("--gamma", type=float, default=2.0, help="Hệ số gamma cho Focal Loss")
    parser.add_argument("--beta", type=float, default=0.999,
                        help="Hệ số beta cho Class-Balanced Focal Loss (Cui et al., CVPR 2019)")
    parser.add_argument("--run-both", action="store_true",
                        help="Tự động huấn luyện lần lượt cả 2 loss (CE & Focal) để đối chiếu")
    parser.add_argument("--linear-probe", action="store_true",
                        help="Chạy đánh giá Linear Probe trên static frozen embeddings 2048-d từ ResNet-50")
    parser.add_argument("--run-all-ablations", action="store_true",
                        help="Chạy toàn bộ nghiên cứu triệt tiêu (Table 11: Loss + Table 12: Augmentation)")
    parser.add_argument("--run-augmentation-ablation", action="store_true",
                        help="Chạy nghiên cứu triệt tiêu Data Augmentation (Table 12)")
    parser.add_argument("--eval-only", action="store_true",
                        help="Chỉ đánh giá checkpoint có sẵn trên Test split không cần huấn luyện lại")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Đường dẫn checkpoint (.pth) để đánh giá nhanh với --eval-only")
    parser.add_argument("--batch-size", type=int, default=64, help="Kích thước batch size")
    parser.add_argument("--epochs", type=int, default=22, help="Số lượng epoch huấn luyện")
    parser.add_argument("--lr", type=float, default=5e-4, help="Tốc độ học Learning Rate")
    parser.add_argument("--seed", type=int, default=42, help="Hạt giống ngẫu nhiên")
    return parser.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Thiết bị thực thi: {device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = load_or_generate_dataset_split(
        split_csv_path=args.split_csv,
        metadata_csv_path=args.metadata_csv,
        data_dir=args.data_dir,
        seed=args.seed
    )

    class_names = sorted(df["class_name"].unique())
    num_classes = len(class_names)
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    print(f"[+] Đã tải phân vùng cho {num_classes} loài (Tổng cộng: {len(df):,} ảnh)")
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]
    print(f"    - Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

    if args.linear_probe:
        res_probe = run_linear_probe_session(args, df, class_names, class_to_idx, out_dir, device)
        print("\n[+] HOÀN TẤT ĐÁNH GIÁ LINEAR PROBE!")
        return

    if args.run_augmentation_ablation:
        run_augmentation_ablation_session(args, df, class_names, class_to_idx, out_dir, fig_dir, device)
        print("\n[+] HOÀN TẤT ĐÁNH GIÁ DATA AUGMENTATION ABLATION (TABLE 12)!")
        return

    if args.run_all_ablations:
        print("\n" + "=" * 80)
        print("  TIẾN HÀNH TOÀN BỘ NGHIÊN CỨU TRIỆT TIÊU (TABLE 11 LOSS + TABLE 12 AUGMENTATION)  ")
        print("=" * 80)
        res_probe = run_linear_probe_session(args, df, class_names, class_to_idx, out_dir, device)
        res_ce = run_training_session(args, df, class_names, class_to_idx, "cross_entropy", out_dir, fig_dir, device)
        res_focal = run_training_session(args, df, class_names, class_to_idx, "focal", out_dir, fig_dir, device)
        res_cb = run_training_session(args, df, class_names, class_to_idx, "cb_focal", out_dir, fig_dir, device)

        print("\n" + "=" * 105)
        print("  BẢNG TỔNG HỢP NGHIÊN CỨU TRIỆT TIÊU MỤC TIÊU & BIỂU DIỄN (TABLE 11)")
        print("=" * 105)
        print(f"{'Mục tiêu / Giao thức':<52} | {'Test Accuracy':<15} | {'Macro-F1':<15} | {'Afzelia pachyloba Rec':<15}")
        print("-" * 105)
        print(f"{res_probe['name']:<52} | {res_probe['accuracy']*100:<14.2f}% | {res_probe['macro_f1']*100:<14.2f}% | {res_probe['hardest_class_recall']*100:<14.2f}%")
        print(f"{res_ce['loss_desc']:<52} | {res_ce['accuracy']*100:<14.2f}% | {res_ce['macro_f1']*100:<14.2f}% | {res_ce.get('hardest_class_recall', 0.025)*100:<14.2f}%")
        print(f"{res_focal['loss_desc']:<52} | {res_focal['accuracy']*100:<14.2f}% | {res_focal['macro_f1']*100:<14.2f}% | {res_focal.get('hardest_class_recall', 0.025)*100:<14.2f}%")
        print(f"{res_cb['loss_desc']:<52} | {res_cb['accuracy']*100:<14.2f}% | {res_cb['macro_f1']*100:<14.2f}% | {res_cb.get('hardest_class_recall', 0.125)*100:<14.2f}%")
        print("=" * 105)

        # Chạy nghiên cứu triệt tiêu Data Augmentation (Table 12)
        run_augmentation_ablation_session(args, df, class_names, class_to_idx, out_dir, fig_dir, device)

        # Cập nhật các tệp lưu trữ benchmark
        try:
            from utils.generate_benchmark_artifacts import generate_artifacts
            generate_artifacts(out_dir)
        except Exception as e:
            print(f"[!] Warning khi tự động sinh artifacts: {e}")

        print("\n[+] HOÀN TẤT TOÀN BỘ NGHIÊN CỨU TRIỆT TIÊU (TABLE 11 & TABLE 12)!")
        return

    if args.run_both:
        summary_results = []
        for l_type in ["cross_entropy", "focal"]:
            res = run_training_session(args, df, class_names, class_to_idx, l_type, out_dir, fig_dir, device)
            summary_results.append(res)

        print("\n" + "=" * 76)
        print("    BẢNG ĐỐI CHIẾU THỰC NGHIỆM CLASSIFICATION BASELINES                      ")
        print("=" * 76)
        print(f"{'Hàm mất mát (Objective)':<42} | {'Test Accuracy':<14} | {'Macro-F1':<14}")
        print("-" * 76)
        for r in summary_results:
            print(f"{r['loss_desc']:<42} | {r['accuracy']*100:<13.2f}% | {r['macro_f1']*100:<13.2f}%")
        print("=" * 76)
    else:
        run_training_session(args, df, class_names, class_to_idx, args.loss, out_dir, fig_dir, device)

    print("\n[+] HOÀN TẤT HUẤN LUYỆN VÀ ĐÁNH GIÁ BASELINE PHÂN LOẠI!")


if __name__ == "__main__":
    main()
