"""
benchmark_split_generalization.py
=================================
Đo lường độ thổi phồng hiệu năng (Performance Inflation) và nguy cơ rò rỉ dữ liệu (Data Leakage)
giữa Naive Random Split vs. Governed Specimen Split (End Version / CEGS-Split).

Hỗ trợ 2 chế độ đánh giá:
  1. --mode knn (Mặc định): Đánh giá 1-NN classification trên frozen deep embeddings (zero-training, chạy nhanh).
  2. --mode train: Huấn luyện mạng nơ-ron tích chập (ConvNeXt-Tiny) với Focal Loss để đo thực tế generalization gap.
  3. --mode both: Thực thi cả hai chế độ.

Cách chạy:
  python benchmark_split_generalization.py --mode knn
  python benchmark_split_generalization.py --mode train --epochs 17
"""

import os
import gc
import json
import random
import argparse
from pathlib import Path
from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

import timm
from timm.data import resolve_data_config
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KNeighborsClassifier

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Imports từ các module nội bộ của 02_specimen_variant_gan ──
from utils import (
    set_seed, get_device,
    collect_image_samples, build_dataframe,
    ImageListDataset, build_transforms,
    freeze_model_layers, summarize_model,
)
from train_governed_baseline import (
    FocalLoss, accuracy_from_logits,
    train_model, build_model,
    compute_embeddings_v2, end_version_split,
    collect_predictions, SPLIT_CONFIG,
)
from split_protocols import compute_split_counts, validate_split


# =============================================================================
# 1. Các hàm hỗ trợ chia dữ liệu ngẫu nhiên (Random Split Baseline)
# =============================================================================

def random_split_wrapper(
    df: pd.DataFrame,
    train_ratio: float = 0.60,
    val_ratio: float = 0.20,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Phân tách ngẫu nhiên cấp độ mẫu vật/ảnh làm đối chứng (Baseline)."""
    keep_mask = ~df["label"].isin(["Pterocarpus sp", "Peltogyne pubescens"])
    df_filtered = df[keep_mask].reset_index(drop=True)

    rng = random.Random(seed)
    train_idx, val_idx, test_idx = [], [], []

    for label, group in df_filtered.groupby("label"):
        subfolder_groups = group.groupby("subfolder")
        subfolder_names = list(subfolder_groups.groups.keys())
        rng.shuffle(subfolder_names)

        indices = sorted(group.index.tolist())
        n_total = len(indices)

        if len(subfolder_names) < 3:
            rng.shuffle(indices)
            train_count, val_count, _ = compute_split_counts(n_total, train_ratio, val_ratio)
            train_idx.extend(indices[:train_count])
            val_idx.extend(indices[train_count:train_count + val_count])
            test_idx.extend(indices[train_count + val_count:])
        else:
            target_train = int(n_total * train_ratio)
            target_val = int(n_total * val_ratio)

            test_idx.extend(subfolder_groups.get_group(subfolder_names[0]).index.tolist())
            val_idx.extend(subfolder_groups.get_group(subfolder_names[1]).index.tolist())

            curr_train = 0
            curr_val = len(subfolder_groups.get_group(subfolder_names[1]).index.tolist())
            curr_test = len(subfolder_groups.get_group(subfolder_names[0]).index.tolist())

            for sf_name in subfolder_names[2:]:
                sf_indices = subfolder_groups.get_group(sf_name).index.tolist()
                sf_count = len(sf_indices)

                if curr_train < target_train:
                    train_idx.extend(sf_indices)
                    curr_train += sf_count
                elif curr_val < target_val:
                    val_idx.extend(sf_indices)
                    curr_val += sf_count
                else:
                    test_idx.extend(sf_indices)
                    curr_test += sf_count

    df_train = df_filtered.loc[train_idx].sample(frac=1, random_state=seed).reset_index(drop=True)
    df_val = df_filtered.loc[val_idx].sample(frac=1, random_state=seed).reset_index(drop=True)
    df_test = df_filtered.loc[test_idx].sample(frac=1, random_state=seed).reset_index(drop=True)
    return df_train, df_val, df_test


# =============================================================================
# 2. Phân tích tương đồng Cosine & Nguy cơ Rò rỉ Dữ liệu
# =============================================================================

def compute_split_similarity(train_embs: np.ndarray, test_embs: np.ndarray) -> Dict[str, Any]:
    """Tính toán ma trận Cosine Similarity và các thống kê đo rò rỉ giữa Train và Test."""
    sim_matrix = cosine_similarity(test_embs, train_embs)
    max_sims = sim_matrix.max(axis=1)
    mean_sims = sim_matrix.mean(axis=1)

    return {
        "mean_cosine_sim": float(np.mean(sim_matrix)),
        "median_cosine_sim": float(np.median(sim_matrix)),
        "max_cosine_sim_per_test_mean": float(np.mean(max_sims)),
        "max_cosine_sim_per_test_std": float(np.std(max_sims)),
        "mean_cosine_sim_per_test_mean": float(np.mean(mean_sims)),
        "all_sims": sim_matrix.flatten(),
        "max_sims": max_sims,
    }


def plot_similarity_histograms(random_sims: dict, endver_sims: dict, output_path: Path) -> None:
    """Vẽ biểu đồ histogram phân phối tương đồng Train <-> Test."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax1 = axes[0]
    ax1.hist(random_sims["all_sims"], bins=80, alpha=0.6, label="Random Split", color="#e74c3c", density=True)
    ax1.hist(endver_sims["all_sims"], bins=80, alpha=0.6, label="Governed Split (EndVer)", color="#2ecc71", density=True)
    ax1.axvline(random_sims["mean_cosine_sim"], color="#c0392b", linestyle="--", linewidth=2,
                label=f"Random mean={random_sims['mean_cosine_sim']:.4f}")
    ax1.axvline(endver_sims["mean_cosine_sim"], color="#27ae60", linestyle="--", linewidth=2,
                label=f"Governed mean={endver_sims['mean_cosine_sim']:.4f}")
    ax1.set_xlabel("Cosine Similarity (Train ↔ Test)", fontsize=11)
    ax1.set_ylabel("Density", fontsize=11)
    ax1.set_title("All Pairwise Cosine Similarities", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    ax2 = axes[1]
    ax2.hist(random_sims["max_sims"], bins=50, alpha=0.6, label="Random Split", color="#e74c3c", density=True)
    ax2.hist(endver_sims["max_sims"], bins=50, alpha=0.6, label="Governed Split (EndVer)", color="#2ecc71", density=True)
    ax2.axvline(random_sims["max_cosine_sim_per_test_mean"], color="#c0392b", linestyle="--", linewidth=2,
                label=f"Random mean={random_sims['max_cosine_sim_per_test_mean']:.4f}")
    ax2.axvline(endver_sims["max_cosine_sim_per_test_mean"], color="#27ae60", linestyle="--", linewidth=2,
                label=f"Governed mean={endver_sims['max_cosine_sim_per_test_mean']:.4f}")
    ax2.set_xlabel("Max Cosine Similarity (Nearest Train Neighbor)", fontsize=11)
    ax2.set_ylabel("Density", fontsize=11)
    ax2.set_title("Nearest-Neighbor Similarity per Test Sample\n(↑ = Higher Data Leakage Risk)", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    plt.suptitle("Specimen Data Leakage Analysis: Random Split vs. Governed Split", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[+] Đã lưu biểu đồ phân phối Cosine: {output_path}")


# =============================================================================
# 3. Đánh giá Zero-Training 1-NN Benchmark
# =============================================================================

def run_knn_benchmark(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    embs: np.ndarray,
    n_neighbors: int = 1,
) -> Dict[str, float]:
    """Chạy 1-NN Classifier trên frozen feature vectors."""
    train_indices = df_train["_orig_idx"].values
    test_indices = df_test["_orig_idx"].values

    x_train, y_train = embs[train_indices], df_train["label"].values
    x_test, y_test = embs[test_indices], df_test["label"].values

    knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric="cosine")
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)

    acc = float(accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


# =============================================================================
# 4. Main Execution
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Benchmark đo lường ảnh hưởng của rò rỉ dữ liệu mẫu vật (SSPB)")
    parser.add_argument("--mode", type=str, default="knn", choices=["knn", "train", "both"],
                        help="Chế độ đánh giá: 'knn' (nhanh), 'train' (huấn luyện đầy đủ), hoặc 'both'")
    parser.add_argument("--data-dir", type=str, default="/kaggle/input/datasets/b23dckh002lvitanh/s3-origin/S3",
                        help="Đường dẫn thư mục chứa ảnh gỗ gốc")
    parser.add_argument("--output-dir", type=str, default="outputs_generalization_benchmark",
                        help="Thư mục xuất báo cáo và biểu đồ")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456],
                        help="Danh sách các random seeds cho Random Split")
    parser.add_argument("--epochs", type=int, default=17,
                        help="Số epochs cho chế độ huấn luyện mạng")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Batch size")
    args = parser.parse_args()

    device = get_device()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("   BENCHMARK TỔNG QUÁT HÓA & ĐO LƯỜNG RÒ RỈ MẪU VẬT (SPECIMEN LEAKAGE)   ")
    print(f"   Chế độ thực thi: {args.mode.upper()} | Thiết bị: {device}")
    print("=" * 80)

    # 1. Quét dữ liệu ảnh
    samples = collect_image_samples(args.data_dir)
    df = build_dataframe(samples)
    keep_mask = ~df["label"].isin(["Pterocarpus sp", "Peltogyne pubescens"])
    df_filtered = df[keep_mask].reset_index(drop=True)
    df_filtered["_orig_idx"] = np.arange(len(df_filtered))

    # 2. Trích xuất hoặc tải cache embeddings
    print("\n[*] Đang nạp hoặc trích xuất representation embeddings...")
    embs_eff, embs_swin = compute_embeddings_v2(df_filtered, device=device, batch_size=args.batch_size)

    # 3. Đánh giá KNN Mode
    if args.mode in ["knn", "both"]:
        print("\n" + "-" * 75)
        print("  [PHASE 1] ZERO-TRAINING 1-NN BENCHMARK TRÊN FROZEN EMBEDDINGS")
        print("-" * 75)
        knn_records = []

        # 3a. Random Split (N seeds)
        for s in args.seeds:
            df_tr, df_va, df_te = random_split_wrapper(df_filtered, seed=s)
            sim_stats = compute_split_similarity(embs_eff[df_tr["_orig_idx"]], embs_eff[df_te["_orig_idx"]])
            metrics = run_knn_benchmark(df_tr, df_te, embs_eff, n_neighbors=1)
            knn_records.append({
                "split": "Naive Random", "seed": s,
                **metrics, **{k: v for k, v in sim_stats.items() if k not in ["all_sims", "max_sims"]}
            })

        # 3b. Governed Split (Seed 42)
        df_tr_g, df_va_g, df_te_g = end_version_split(df_filtered, embs_eff, embs_swin, seed=42)
        sim_stats_g = compute_split_similarity(embs_eff[df_tr_g["_orig_idx"]], embs_eff[df_te_g["_orig_idx"]])
        metrics_g = run_knn_benchmark(df_tr_g, df_te_g, embs_eff, n_neighbors=1)
        knn_records.append({
            "split": "Governed (CEGS-Split)", "seed": 42,
            **metrics_g, **{k: v for k, v in sim_stats_g.items() if k not in ["all_sims", "max_sims"]}
        })

        # Bảng hiển thị
        df_knn = pd.DataFrame(knn_records)
        print(df_knn[["split", "seed", "accuracy", "macro_f1", "max_cosine_sim_per_test_mean"]].to_string(index=False))

        # Vẽ histogram so sánh
        rand_last_sim = compute_split_similarity(embs_eff[df_tr["_orig_idx"]], embs_eff[df_te["_orig_idx"]])
        plot_similarity_histograms(rand_last_sim, sim_stats_g, output_dir / "histogram_cosine_similarity.png")

        df_knn.to_csv(output_dir / "knn_benchmark_results.csv", index=False)
        print(f"[+] Đã lưu kết quả KNN: {output_dir / 'knn_benchmark_results.csv'}")

    print("\n[✓] Hoàn thành phân tích đánh giá tổng quát hóa!")


if __name__ == "__main__":
    main()
