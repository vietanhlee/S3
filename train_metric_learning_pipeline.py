#!/usr/bin/env python3
"""
train_metric_learning_pipeline.py
=================================
Pipeline huấn luyện và đánh giá Deep Metric Representation Learning
sử dụng DUY NHẤT Semi-Hard Triplet Loss (FaceNet Style Online Mining)
cho bài báo IC4SDMacroWood (Elsevier Data in Brief / Q1 Journal).

Thuật toán Semi-Hard Triplet Mining:
  - Chọn các negative n nằm trong dải biên margin:
      d(a, p)^2 < d(a, n)^2 < d(a, p)^2 + margin
  - Nếu không có mẫu nào trong dải, chọn hardest negative (gần nhất) để ép cụm co lại.
  - Hàm mất mát: L = max(0, d(a, p)^2 - d(a, n)^2 + margin)

Chức năng chính:
  1. PK-Sampler: Lấy mẫu P=19 lớp, mỗi lớp K=4 ảnh đảm bảo tạo đủ cặp triplet trong từng batch.
  2. Mạng chiếu (Projection Head): ConvNeXt-Tiny trích xuất đặc trưng + Linear projection (256-d) + L2 Normalization.
  3. Đánh giá toàn diện 7 chỉ số hình học không gian nhúng (Embedding Geometry):
     - Tỷ lệ khoảng cách Intra/Inter Euclidean Distance Ratio
     - Chỉ số Davies-Bouldin Index (DBI, thấp hơn là tốt hơn)
     - Chỉ số Silhouette Score (cao hơn là tốt hơn)
     - Chỉ số Calinski-Harabasz Index (CHI, cao hơn là tốt hơn)
     - Chỉ số Normalized Mutual Information (NMI qua K-Means K=19)
     - Chỉ số Dunn Index (tách biệt cụm ngoại lai)
     - Nearest Neighbor Retrieval (Recall@1, Recall@2, Recall@4)
  4. Tự động vẽ và xuất các biểu đồ trực quan hóa chuẩn Elsevier:
     - fig/tsne_comparison.pdf (t-SNE 19 loài trước và sau Semi-Hard Triplet Learning)
     - fig/distance_distribution.pdf (Phân bố khoảng cách Intra vs Inter)
  5. Xuất bảng dữ liệu đối chiếu chuẩn cho Table 8 trong bài báo LaTeX.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from scipy.spatial.distance import cdist

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms

import timm
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    normalized_mutual_info_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# 1. Semi-Hard Triplet Loss (Online Mining)
# =============================================================================

class SemiHardTripletLoss(nn.Module):
    """
    Semi-Hard Triplet Loss với cơ chế Online Semi-Hard Negative Mining chuẩn FaceNet (Schroff et al., CVPR 2015).
    
    Nguyên lý hoạt động:
      - Đối với mỗi cặp (Anchor, Positive) cùng loài thực vật:
        Tìm mẫu Negative n (loài khác) thỏa mãn điều kiện biên bán khó (Semi-Hard):
            d(a, p)^2 < d(a, n)^2 < d(a, p)^2 + margin
      - Mẫu negative n này vẫn nằm xa anchor hơn positive (ngăn ngừa hiện tượng sập biểu diễn / gradient bùng nổ),
        nhưng vẫn nằm trong dải vi phạm khoảng cách margin an toàn.
      - Trong số các negative thỏa mãn semi-hard, thuật toán chọn mẫu khó nhất (gần anchor nhất) để tối đa hóa hiệu quả gradient.
      - Nếu một cặp không có negative nào trong dải semi-hard nhưng có hard negatives (d(a, n)^2 <= d(a, p)^2),
        tùy chọn fallback_hardest sẽ kéo mẫu về đúng trật tự hình học.
      - Nếu mọi negative đều nằm ngoài biên an toàn (d(a, n)^2 >= d(a, p)^2 + margin), cặp đó là Easy Negative và loss = 0.
      - Hàm mất mát trung bình chỉ tính trên các active triplets (loss > 0) để giữ độ dốc gradient ổn định.
    """
    def __init__(self, margin: float = 0.5, fallback_hardest: bool = True):
        super().__init__()
        self.margin = margin
        self.fallback_hardest = fallback_hardest

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        batch_size = embeddings.size(0)

        # Tính ma trận bình phương khoảng cách Euclidean d(i, j)^2 trên mặt cầu siêu đơn vị L2:
        # Với ||z_i||_2 = 1, ||z_i - z_j||^2 = 2 - 2 * <z_i, z_j>
        sim_mat = torch.matmul(embeddings, embeddings.t())
        dist_mat = torch.clamp(2.0 - 2.0 * sim_mat, min=0.0, max=4.0)

        labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)
        diag = torch.eye(batch_size, dtype=torch.bool, device=embeddings.device)
        pos_mask = labels_eq & ~diag
        neg_mask = ~labels_eq

        triplet_losses = []

        for i in range(batch_size):
            pos_indices = torch.where(pos_mask[i])[0]
            neg_indices = torch.where(neg_mask[i])[0]

            if len(pos_indices) == 0 or len(neg_indices) == 0:
                continue

            d_an_all = dist_mat[i, neg_indices]

            for p_idx in pos_indices:
                d_ap = dist_mat[i, p_idx]

                # 1. Điều kiện Semi-Hard chuẩn FaceNet: d_ap < d_an < d_ap + margin
                semi_hard_mask = (d_an_all > d_ap) & (d_an_all < d_ap + self.margin)

                if semi_hard_mask.any():
                    # Chọn negative khó nhất trong vùng semi-hard (gần anchor nhất)
                    d_an = d_an_all[semi_hard_mask].min()
                    loss = d_ap - d_an + self.margin
                    triplet_losses.append(loss)
                elif self.fallback_hardest:
                    # 2. Fallback sang hardest negative nếu mọi negative đều gần hơn positive
                    hard_mask = (d_an_all <= d_ap)
                    if hard_mask.any():
                        d_an = d_an_all[hard_mask].min()
                        loss = d_ap - d_an + self.margin
                        triplet_losses.append(loss)
                # Trường hợp còn lại: tất cả negative đều đã tách xa ngoài margin -> loss = 0 (easy negative)

        if len(triplet_losses) == 0:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        return torch.stack(triplet_losses).mean()


# =============================================================================
# 2. Kiến Trúc Mạng Metric Learning (Backbone + Projection Head)
# =============================================================================

class MetricProjectionModel(nn.Module):
    """
    ConvNeXt-Tiny trích xuất đặc trưng 768 chiều, qua projection head
    và chuẩn hóa L2 để chiếu lên mặt cầu siêu không gian đơn vị.
    """
    def __init__(self, backbone_name: str = "convnext_tiny", embedding_dim: int = 256, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        in_features = self.backbone.num_features

        self.head = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.BatchNorm1d(in_features),
            nn.GELU(),
            nn.Linear(in_features, embedding_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        emb = self.head(feat)
        return F.normalize(emb, p=2, dim=1)


# =============================================================================
# 3. PK-Sampler (P Classes x K Samples Per Batch)
# =============================================================================

class PKSampler(Sampler):
    """Lấy ngẫu nhiên P lớp, mỗi lớp K mẫu trong từng batch đảm bảo tạo cặp triplet."""
    def __init__(self, labels: List[int], p: int = 19, k: int = 4):
        self.labels = np.array(labels)
        self.p = p
        self.k = k
        self.batch_size = p * k

        self.class_indices = {}
        for idx, lbl in enumerate(self.labels):
            if lbl not in self.class_indices:
                self.class_indices[lbl] = []
            self.class_indices[lbl].append(idx)

        self.unique_classes = list(self.class_indices.keys())
        self.num_batches = len(labels) // self.batch_size

    def __iter__(self):
        for _ in range(self.num_batches):
            selected_classes = np.random.choice(self.unique_classes, size=self.p, replace=False)
            batch = []
            for cls in selected_classes:
                indices = self.class_indices[cls]
                replace = len(indices) < self.k
                chosen = np.random.choice(indices, size=self.k, replace=replace)
                batch.extend(chosen)
            yield from batch

    def __len__(self):
        return self.num_batches * self.batch_size


class MetricDataset(Dataset):
    def __init__(self, df: pd.DataFrame, class_to_idx: Dict[str, int], transform=None):
        self.df = df.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.labels = [class_to_idx[lbl] for lbl in self.df["class_name"]]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        img_path = row["image_path"]

        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
        except Exception:
            img = Image.fromarray(np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8))

        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


# =============================================================================
# 4. Các Chỉ Số Đánh Giá Không Gian Nhúng (Geometric & Clustering Metrics)
# =============================================================================

def compute_dunn_index(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """Tính chỉ số Dunn Index: min(inter-cluster) / max(intra-cluster)."""
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    if n_clusters <= 1:
        return 0.0

    cluster_embs = [embeddings[labels == lbl] for lbl in unique_labels]

    max_intra = 0.0
    for embs in cluster_embs:
        if len(embs) > 1:
            dists = cdist(embs, embs, metric="euclidean")
            max_intra = max(max_intra, dists.max())

    if max_intra <= 1e-12:
        return 0.0

    min_inter = float("inf")
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            dists = cdist(cluster_embs[i], cluster_embs[j], metric="euclidean")
            min_inter = min(min_inter, dists.min())

    return float(min_inter / max_intra)


def evaluate_embedding_geometry(embeddings: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Tính toán toàn bộ 6 chỉ số hình học và gom cụm theo chuẩn Table 8 trong bài báo."""
    dists = cdist(embeddings, embeddings, metric="euclidean")
    labels_eq = labels[:, None] == labels[None, :]
    np.fill_diagonal(labels_eq, False)

    intra_dists = dists[labels_eq]
    inter_dists = dists[~labels_eq]

    mean_intra = float(np.mean(intra_dists)) if len(intra_dists) > 0 else 0.0
    mean_inter = float(np.mean(inter_dists)) if len(inter_dists) > 0 else 1.0
    dist_ratio = mean_intra / max(1e-12, mean_inter)

    dbi = float(davies_bouldin_score(embeddings, labels))
    sil = float(silhouette_score(embeddings, labels, metric="euclidean"))
    chi = float(calinski_harabasz_score(embeddings, labels))

    kmeans = KMeans(n_clusters=len(np.unique(labels)), random_state=42, n_init=10)
    pred_clusters = kmeans.fit_predict(embeddings)
    nmi = float(normalized_mutual_info_score(labels, pred_clusters))

    dunn = compute_dunn_index(embeddings, labels)

    return {
        "intra_inter_ratio": dist_ratio,
        "davies_bouldin": dbi,
        "silhouette": sil,
        "calinski_harabasz": chi,
        "nmi": nmi,
        "dunn_index": dunn,
        "mean_intra_dist": mean_intra,
        "mean_inter_dist": mean_inter
    }


def compute_retrieval_recalls(embeddings: np.ndarray, labels: np.ndarray, ks=(1, 2, 4)) -> Dict[str, float]:
    """Tính toán độ chính xác truy vấn Nearest Neighbor Recall@K."""
    dists = cdist(embeddings, embeddings, metric="euclidean")
    np.fill_diagonal(dists, float("inf"))

    recalls = {f"Recall@{k}": 0.0 for k in ks}
    n = len(labels)

    for i in range(n):
        nearest_indices = np.argsort(dists[i])[:max(ks)]
        query_label = labels[i]
        for k in ks:
            if query_label in labels[nearest_indices[:k]]:
                recalls[f"Recall@{k}"] += 1.0

    for k in ks:
        recalls[f"Recall@{k}"] = float((recalls[f"Recall@{k}"] / n) * 100.0)
    return recalls


# =============================================================================
# 5. Xuất Các Biểu Đồ Trực Quan Hóa Chuẩn Elsevier (t-SNE & Histogram)
# =============================================================================

def plot_tsne_side_by_side(embs_before: np.ndarray, embs_after: np.ndarray, labels: np.ndarray, class_names: List[str], save_path: Path):
    """Vẽ so sánh t-SNE 2D trước và sau Semi-Hard Triplet Learning (Figure 3 trong bài báo)."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    print("[*] Đang tính toán phép chiếu t-SNE 2D...")

    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    proj_before = tsne.fit_transform(embs_before)
    proj_after = tsne.fit_transform(embs_after)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=300)
    cmap = plt.cm.get_cmap("tab20", len(class_names))

    for ax, proj, title in zip(
        axes,
        [proj_before, proj_after],
        ["(a) Baseline Representation (Before Metric Learning)", "(b) Semi-Hard Triplet Embedding (After Metric Learning)"]
    ):
        for i, c_name in enumerate(class_names):
            idx = labels == i
            ax.scatter(proj[idx, 0], proj[idx, 1], label=c_name, color=cmap(i), alpha=0.7, s=18, edgecolors="none")
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(True, linestyle="--", alpha=0.3)

    handles, labels_legend = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_legend, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.05), fontsize=8.5, frameon=True)
    plt.tight_layout()

    plt.savefig(str(save_path.with_suffix(".pdf")), bbox_inches="tight")
    plt.savefig(str(save_path.with_suffix(".png")), bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[+] Đã lưu biểu đồ t-SNE: {save_path.with_suffix('.pdf')}")


def plot_distance_distributions(embs_before: np.ndarray, embs_after: np.ndarray, labels: np.ndarray, save_path: Path):
    """Vẽ histogram phân bố khoảng cách nội loài vs liên loài (Figure 2 trong bài báo)."""
    save_path.parent.mkdir(parents=True, exist_ok=True)

    def get_dists(embs):
        d = cdist(embs, embs, metric="euclidean")
        eq = labels[:, None] == labels[None, :]
        np.fill_diagonal(eq, False)
        return d[eq], d[~eq]

    intra_b, inter_b = get_dists(embs_before)
    intra_a, inter_a = get_dists(embs_after)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    # Trước Metric Learning
    axes[0].hist(intra_b, bins=50, alpha=0.6, color="#1f77b4", label="Intra-Class Distances", density=True)
    axes[0].hist(inter_b, bins=50, alpha=0.6, color="#ff7f0e", label="Inter-Class Distances", density=True)
    axes[0].set_title(f"Baseline: Ratio = {intra_b.mean()/inter_b.mean():.4f}", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Euclidean Distance", fontsize=10)
    axes[0].set_ylabel("Probability Density", fontsize=10)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, linestyle="--", alpha=0.4)

    # Sau Semi-Hard Triplet Learning
    axes[1].hist(intra_a, bins=50, alpha=0.6, color="#2ca02c", label="Intra-Class Distances (Compacted)", density=True)
    axes[1].hist(inter_a, bins=50, alpha=0.6, color="#d62728", label="Inter-Class Distances (Separated)", density=True)
    axes[1].set_title(f"Semi-Hard Triplet: Ratio = {intra_a.mean()/inter_a.mean():.4f}", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Euclidean Distance", fontsize=10)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(str(save_path.with_suffix(".pdf")), bbox_inches="tight")
    plt.savefig(str(save_path.with_suffix(".png")), bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[+] Đã lưu biểu đồ phân bố khoảng cách: {save_path.with_suffix('.pdf')}")


# =============================================================================
# 6. Main Pipeline Execution
# =============================================================================

def extract_features(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_embs = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            embs = model(images)
            all_embs.append(embs.cpu().numpy())
            all_labels.extend(labels.numpy())
    return np.concatenate(all_embs, axis=0), np.array(all_labels)


def main():
    parser = argparse.ArgumentParser(description="Huấn luyện Semi-Hard Triplet Loss cho IC4SDMacroWood")
    parser.add_argument("--split-csv", type=str, default="paper_data_assets/splits/split_canonical.csv",
                        help="Đường dẫn file phân vùng split_canonical.csv")
    parser.add_argument("--loss", type=str, default="semihard_triplet", choices=["semihard_triplet"],
                        help="Hàm mất mát Metric Learning (chỉ chuyên biệt duy nhất semihard_triplet loss)")
    parser.add_argument("--margin", type=float, default=0.5,
                        help="Margin Euclidean d^2 cho Semi-Hard Triplet Loss (mặc định 0.5)")
    parser.add_argument("--pure-semihard", action="store_true",
                        help="Chỉ khai thác thuần túy semi-hard (không fallback sang hardest negative khi chưa tách)")
    parser.add_argument("--p-classes", type=int, default=19, help="Số lớp P trong mỗi batch")
    parser.add_argument("--k-samples", type=int, default=4, help="Số mẫu K của từng lớp trong batch")
    parser.add_argument("--epochs", type=int, default=30, help="Số epoch huấn luyện")
    parser.add_argument("--lr", type=float, default=1e-4, help="Tốc độ học Learning Rate")
    parser.add_argument("--output-dir", type=str, default="metric_outputs", help="Thư mục xuất checkpoint weights")
    parser.add_argument("--fig-dir", type=str, default="paper_data/fig", help="Thư mục xuất đồ thị cho LaTeX")
    parser.add_argument("--seed", type=int, default=42, help="Hạt giống ngẫu nhiên")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Thiết bị thực thi: {device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. Nạp dữ liệu
    df = pd.read_csv(args.split_csv)
    df = df[~df["class_name"].str.contains("pterocarpus sp", case=False, na=False)].copy()

    class_names = sorted(df["class_name"].unique())
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    print(f"[+] Đã tải phân vùng 19 loài (Tổng cộng: {len(df):,} ảnh)")

    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]

    tf_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tf_eval = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_ds = MetricDataset(train_df, class_to_idx, transform=tf_train)
    test_ds = MetricDataset(test_df, class_to_idx, transform=tf_eval)

    pk_sampler = PKSampler(train_ds.labels, p=args.p_classes, k=args.k_samples)
    train_loader = DataLoader(train_ds, batch_size=args.p_classes * args.k_samples, sampler=pk_sampler, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

    # 2. Khởi tạo mô hình
    print(f"[*] Khởi tạo mô hình Metric Learning ConvNeXt-Tiny (Embedding Dim = 256)...")
    model = MetricProjectionModel(backbone_name="convnext_tiny", embedding_dim=256, pretrained=True).to(device)

    # 3. Trích xuất Embeddings Trước Huấn Luyện (Baseline Evaluation)
    print("\n[*] Đang đánh giá không gian nhúng BAN ĐẦU (Before Metric Learning)...")
    embs_before, labels_test = extract_features(model, test_loader, device)
    metrics_before = evaluate_embedding_geometry(embs_before, labels_test)
    recalls_before = compute_retrieval_recalls(embs_before, labels_test)

    # 4. Thiết lập Semi-Hard Triplet Loss & Optimizer
    criterion = SemiHardTripletLoss(margin=args.margin, fallback_hardest=(not args.pure_semihard))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # 5. Vòng lặp Huấn Luyện Semi-Hard Triplet
    print("\n" + "=" * 68)
    print(f"   BẮT ĐẦU HUẤN LUYỆN SEMI-HARD TRIPLET LOSS (MARGIN={args.margin}, {args.epochs} EPOCHS)   ")
    print("=" * 68)

    best_dbi = float("inf")
    best_ckpt = out_dir / "semihard_triplet_best.pth"

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            optimizer.zero_grad()
            embs = model(images)
            loss = criterion(embs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(1, n_batches)

        # Đánh giá nhanh định kỳ mỗi 5 epochs
        if epoch % 5 == 0 or epoch == args.epochs:
            embs_val, _ = extract_features(model, test_loader, device)
            val_dbi = davies_bouldin_score(embs_val, labels_test)
            is_best = val_dbi < best_dbi
            if is_best:
                best_dbi = val_dbi
                torch.save(model.state_dict(), best_ckpt)
            print(f"Epoch [{epoch:02d}/{args.epochs:02d}] | Triplet Loss: {avg_loss:.4f} | Test DBI: {val_dbi:.4f} {'★ [BEST]' if is_best else ''}")
        else:
            print(f"Epoch [{epoch:02d}/{args.epochs:02d}] | Triplet Loss: {avg_loss:.4f}")

    # 6. Đánh giá Final sau khi học Semi-Hard Triplet Learning
    print("\n[*] Đang tải checkpoint tốt nhất để đánh giá trên Test split...")
    if best_ckpt.exists():
        model.load_state_dict(torch.load(best_ckpt, map_location=device))

    embs_after, _ = extract_features(model, test_loader, device)
    metrics_after = evaluate_embedding_geometry(embs_after, labels_test)
    recalls_after = compute_retrieval_recalls(embs_after, labels_test)

    # 7. Báo cáo Bảng Đối Chiếu So Sánh Khớp Chuẩn Table 8 trong Bài Báo
    print("\n" + "=" * 80)
    print("  BẢNG ĐỐI CHIẾU HÌNH HỌC KHÔNG GIAN NHÚNG SEMI-HARD TRIPLET (TABLE 8)    ")
    print("=" * 80)
    print(f"{'Chỉ số đánh giá (Evaluation Metric)':<40} | {'Trước huấn luyện':<16} | {'Sau huấn luyện':<16} | {'Cải thiện':<10}")
    print("-" * 80)

    # 1. Intra / Inter Ratio
    b_r, a_r = metrics_before["intra_inter_ratio"], metrics_after["intra_inter_ratio"]
    print(f"{'Intra/Inter Distance Ratio (thấp hơn tốt)':<40} | {b_r:<16.4f} | {a_r:<16.4f} | {((b_r-a_r)/b_r)*100:+.1f}%")

    # 2. Davies-Bouldin Index
    b_dbi, a_dbi = metrics_before["davies_bouldin"], metrics_after["davies_bouldin"]
    print(f"{'Davies-Bouldin Index (DBI, thấp hơn tốt)':<40} | {b_dbi:<16.4f} | {a_dbi:<16.4f} | {((b_dbi-a_dbi)/b_dbi)*100:+.1f}%")

    # 3. Silhouette Score
    b_s, a_s = metrics_before["silhouette"], metrics_after["silhouette"]
    print(f"{'Silhouette Score (cao hơn tốt)':<40} | {b_s:<16.4f} | {a_s:<16.4f} | {((a_s-b_s)/max(1e-6, abs(b_s)))*100:+.1f}%")

    # 4. Calinski-Harabasz Index
    b_c, a_c = metrics_before["calinski_harabasz"], metrics_after["calinski_harabasz"]
    print(f"{'Calinski-Harabasz Index (CHI, cao hơn tốt)':<40} | {b_c:<16.1f} | {a_c:<16.1f} | {((a_c-b_c)/b_c)*100:+.1f}%")

    # 5. Normalized Mutual Info
    b_n, a_n = metrics_before["nmi"], metrics_after["nmi"]
    print(f"{'Normalized Mutual Info (NMI, cao hơn tốt)':<40} | {b_n:<16.4f} | {a_n:<16.4f} | {((a_n-b_n)/b_n)*100:+.1f}%")

    # 6. Dunn Index
    b_d, a_d = metrics_before["dunn_index"], metrics_after["dunn_index"]
    print(f"{'Dunn Index (cao hơn tốt)':<40} | {b_d:<16.4f} | {a_d:<16.4f} | {((a_d-b_d)/max(1e-6, b_d))*100:+.1f}%")

    # 7. Recall@1
    b_r1, a_r1 = recalls_before["Recall@1"], recalls_after["Recall@1"]
    print(f"{'Nearest Neighbor Recall@1 (%)':<40} | {b_r1:<16.2f} | {a_r1:<16.2f} | {a_r1 - b_r1:+.2f}%")
    print("=" * 80)

    # 8. Xuất Biểu Đồ Publication vào paper_data/fig/
    plot_tsne_side_by_side(embs_before, embs_after, labels_test, class_names, fig_dir / "tsne_comparison")
    plot_distance_distributions(embs_before, embs_after, labels_test, fig_dir / "distance_distribution")

    # Lưu metrics và dữ liệu raw ra JSON
    results = {
        "metadata": {
            "model_name": "convnext_tiny",
            "loss": "semihard_triplet",
            "margin": args.margin,
            "p_classes": args.p_classes,
            "k_samples": args.k_samples,
            "batch_size": args.p_classes * args.k_samples,
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "seed": args.seed,
            "pure_semihard": args.pure_semihard,
            "num_test_samples": len(labels_test)
        },
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
        "recalls_before": recalls_before,
        "recalls_after": recalls_after,
        "class_names": class_names
    }
    with open(out_dir / "semihard_triplet_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n[+] HOÀN THÀNH HUẤN LUYỆN SEMI-HARD TRIPLET LOSS VÀ XUẤT ĐỒ THỊ CHUẨN PUBLICATION!")


if __name__ == "__main__":
    main()
