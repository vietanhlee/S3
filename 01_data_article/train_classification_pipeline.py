#!/usr/bin/env python3
"""
train_classification_pipeline.py
================================
CLI thực thi huấn luyện mô hình ConvNeXt-Tiny Baseline và đánh giá kỹ thuật (Technical Validation)
cho bộ dữ liệu ForensicMacroWood-CITES (Elsevier Data in Brief).

Chức năng:
  - Huấn luyện Multiclass Focal Loss (alpha=0.25, gamma=2.0) hoặc Standard Cross-Entropy.
  - Tối ưu hóa AdamW + Cosine Annealing Learning Rate Scheduler.
  - Tự động lưu Checkpoint tốt nhất, xuất ma trận nhầm lẫn và biểu đồ chuẩn Elsevier.
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import torch

from modules.classification import (
    load_or_generate_dataset_split,
    run_training_session
)


def parse_args():
    parser = argparse.ArgumentParser(description="Huấn luyện ConvNeXt-Tiny Baseline cho ForensicMacroWood-CITES")
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
    parser.add_argument("--model-name", type=str, default="convnext_tiny",
                        help="Tên kiến trúc backbone từ thư viện timm")
    parser.add_argument("--loss", type=str, default="focal", choices=["focal", "cross_entropy", "ce"],
                        help="Hàm mất mát: 'focal' hoặc 'cross_entropy'")
    parser.add_argument("--alpha", type=float, default=0.25, help="Hệ số alpha cho Focal Loss")
    parser.add_argument("--gamma", type=float, default=2.0, help="Hệ số gamma cho Focal Loss")
    parser.add_argument("--run-both", action="store_true",
                        help="Tự động huấn luyện lần lượt cả 2 loss để đối chiếu")
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
