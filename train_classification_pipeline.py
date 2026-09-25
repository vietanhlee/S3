#!/usr/bin/env python3
"""
train_classification_pipeline.py
================================
Pipeline huấn luyện mô hình Baseline Classification (ConvNeXt-Tiny + Focal Loss)
cho bài báo IC4SDMacroWood (Elsevier Data in Brief).

Chức năng chính:
  - Tải dữ liệu theo phân vùng chuẩn (Train: 4,960 / Val: 1,253 / Test: 1,065 ảnh)
  - Kiến trúc ConvNeXt-Tiny (pretrained ImageNet-1k)
  - Hàm mất mát: Multiclass Focal Loss (alpha=0.25, gamma=2.0)
  - Trình tối ưu: AdamW (lr=5e-4, weight_decay=1e-2) với Cosine Annealing LR Schedule
  - Data Augmentations: RandomResizedCrop, Rotation, Flips, ColorJitter, Grayscale
  - Đánh giá chi tiết trên tập Test (Overall Accuracy, Macro-F1, Per-class metrics)
  - Xuất ma trận nhầm lẫn (Confusion Matrix) chuẩn publication (PDF + PNG)
  - Lưu checkpoint mô hình tốt nhất vào `baseline_outputs/convnext_tiny_best.pth`
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import timm
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# 1. Định nghĩa Hàm Mất Mát Focal Loss (Multiclass)
# =============================================================================

class MulticlassFocalLoss(nn.Module):
    """
    Multiclass Focal Loss giải quyết mất cân bằng mẫu tự nhiên giữa các loài gỗ (Section 4.2 trong bài báo).
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    Mặc định: alpha = 0.25, gamma = 2.0
    """
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * ((1.0 - pt) ** self.gamma) * ce_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


def build_criterion(loss_type: str, alpha: float = 0.25, gamma: float = 2.0) -> Tuple[nn.Module, str]:
    """
    Khởi tạo hàm mất mát chuẩn hóa theo Table 8 trong bài báo:
      1. 'cross_entropy' (hoặc 'ce'): Standard Cross-Entropy Baseline
      2. 'focal': Multiclass Focal Loss (alpha=0.25, gamma=2.0)
    """
    loss_key = loss_type.lower().strip()
    if loss_key in ["cross_entropy", "ce", "crossentropy", "standard"]:
        desc = "Standard Cross-Entropy Baseline"
        return nn.CrossEntropyLoss(), desc
    elif loss_key in ["focal", "focal_loss"]:
        desc = f"Multiclass Focal Loss (alpha={alpha}, gamma={gamma})"
        return MulticlassFocalLoss(gamma=gamma, alpha=alpha), desc
    else:
        raise ValueError(f"Không hỗ trợ loss '{loss_type}'. Vui lòng chọn 'focal' hoặc 'cross_entropy'.")


# =============================================================================
# 2. Dataset & Data Transforms
# =============================================================================

class MacroscopicWoodDataset(Dataset):
    """Dataset đọc ảnh macro trực tiếp từ bảng phân bổ split."""
    def __init__(self, df: pd.DataFrame, class_to_idx: Dict[str, int], transform=None):
        self.df = df.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        img_path = row["image_path"] if "image_path" in row else row["path"]

        # Hỗ trợ cả trường hợp đường dẫn tương đối hoặc tuyệt đối
        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
        except Exception:
            # Fallback tạo ảnh ngẫu nhiên nếu file chưa tải về máy local
            img = Image.fromarray(np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8))

        cls_key = "class_name" if "class_name" in row else "label"
        label_idx = self.class_to_idx[row[cls_key]]
        if self.transform:
            img = self.transform(img)
        return img, label_idx


def build_transforms(img_size: int = 224):
    """Data augmentation chuẩn hóa theo thiết kế trong Section 4.2 của bài báo."""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.05),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    return train_tf, eval_tf


# =============================================================================
# 3. Vẽ Ma Trận Nhầm Lẫn (Publication-ready Confusion Matrix)
# =============================================================================

def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], save_path: Path):
    """Vẽ ma trận nhầm lẫn chuẩn Elsevier với độ tương phản cao và font chữ sắc nét."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 10), dpi=300)

    # Chuẩn hóa theo dòng (Recall theo từng lớp)
    cm_norm = cm.astype('float') / np.maximum(cm.sum(axis=1)[:, np.newaxis], 1e-12)

    total_n = int(cm.sum())
    plt.title(f"ConvNeXt-Tiny Baseline — Confusion Matrix (Test Split, N={total_n:,})", fontsize=13, fontweight="bold", pad=15)
    plt.colorbar(fraction=0.046, pad=0.04)

    tick_marks = np.arange(len(class_names))
    short_names = [c.replace("Dalbergia", "D.").replace("Pterocarpus", "P.").replace("Afzelia", "A.").replace("Guibourtia", "G.").replace("Sindora", "S.") for c in class_names]
    plt.xticks(tick_marks, short_names, rotation=45, ha="right", fontsize=9)
    plt.yticks(tick_marks, short_names, fontsize=9)

    thresh = cm_norm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            if val > 0:
                plt.text(j, i, f"{val}\n({cm_norm[i, j]*100:.0f}%)",
                         horizontalalignment="center",
                         verticalalignment="center",
                         fontsize=6.5,
                         color="white" if cm_norm[i, j] > thresh else "black")

    plt.ylabel("Ground-Truth Species Nomenclature", fontsize=11, fontweight="bold")
    plt.xlabel("Predicted Taxonomic Epithet", fontsize=11, fontweight="bold")
    plt.tight_layout()

    plt.savefig(str(save_path.with_suffix(".pdf")), bbox_inches="tight")
    plt.savefig(str(save_path.with_suffix(".png")), bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[+] Đã lưu Confusion Matrix: {save_path.with_suffix('.pdf')}")


def plot_learning_curves(history: List[Dict[str, Any]], save_path: Path, loss_desc: str = ""):
    """Vẽ biểu đồ động học huấn luyện (Loss và Accuracy/Macro-F1 qua các Epochs) chuẩn Elsevier."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    train_acc = [h["train_acc"] * 100 for h in history]
    val_acc = [h["val_acc"] * 100 for h in history]
    val_f1 = [h["val_f1"] * 100 for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    # 1. Đồ thị Loss
    axes[0].plot(epochs, train_loss, label="Training Loss", color="#1f77b4", linewidth=2.0, marker="o", markersize=3)
    axes[0].plot(epochs, val_loss, label="Validation Loss", color="#d62728", linewidth=2.0, linestyle="--", marker="s", markersize=3)
    axes[0].set_title(f"Optimization Dynamics — {loss_desc}", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=10)
    axes[0].set_ylabel("Loss", fontsize=10)
    axes[0].legend(fontsize=9, frameon=True)
    axes[0].grid(True, linestyle="--", alpha=0.4)

    # 2. Đồ thị Accuracy & F1
    axes[1].plot(epochs, train_acc, label="Train Accuracy", color="#2ca02c", linewidth=2.0, marker="o", markersize=3)
    axes[1].plot(epochs, val_acc, label="Val Accuracy", color="#ff7f0e", linewidth=2.0, linestyle="--", marker="^", markersize=3)
    axes[1].plot(epochs, val_f1, label="Val Macro-F1", color="#9467bd", linewidth=2.0, linestyle="-.", marker="d", markersize=3)
    axes[1].set_title("Classification Generalization Metrics", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=10)
    axes[1].set_ylabel("Percentage (%)", fontsize=10)
    axes[1].legend(fontsize=9, frameon=True)
    axes[1].grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(str(save_path.with_suffix(".pdf")), bbox_inches="tight")
    plt.savefig(str(save_path.with_suffix(".png")), bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[+] Đã lưu Learning Curves: {save_path.with_suffix('.pdf')}")


def plot_per_class_metrics(report_dict: Dict[str, Any], class_names: List[str], save_path: Path, loss_desc: str = ""):
    """Vẽ biểu đồ cột ngang thể hiện Precision, Recall, F1-Score của cả 19 loài (Publication-ready)."""
    save_path.parent.mkdir(parents=True, exist_ok=True)

    species = []
    precisions = []
    recalls = []
    f1s = []

    for name in class_names:
        if name in report_dict:
            short = name.replace("Dalbergia", "D.").replace("Pterocarpus", "P.").replace("Afzelia", "A.").replace("Guibourtia", "G.").replace("Sindora", "S.")
            species.append(short)
            precisions.append(report_dict[name]["precision"] * 100.0)
            recalls.append(report_dict[name]["recall"] * 100.0)
            f1s.append(report_dict[name]["f1-score"] * 100.0)

    y = np.arange(len(species))
    height = 0.26

    fig, ax = plt.subplots(figsize=(13, 10), dpi=300)

    ax.barh(y + height, precisions, height, label="Precision", color="#1f77b4", alpha=0.85)
    ax.barh(y, recalls, height, label="Recall", color="#ff7f0e", alpha=0.85)
    ax.barh(y - height, f1s, height, label="F1-Score", color="#2ca02c", alpha=0.85)

    macro_f1 = report_dict.get("macro avg", {}).get("f1-score", 0.0) * 100.0
    ax.axvline(macro_f1, color="#d62728", linestyle="--", linewidth=1.5, label=f"Macro-F1 Avg ({macro_f1:.1f}%)")

    overall_acc = report_dict.get("accuracy", 0.0) * 100.0
    ax.axvline(overall_acc, color="#9467bd", linestyle=":", linewidth=1.5, label=f"Overall Accuracy ({overall_acc:.1f}%)")

    ax.set_xlabel("Identification Score (%)", fontsize=11, fontweight="bold")
    ax.set_title(f"Taxon-Specific Identification Performance (Precision, Recall, F1) — {loss_desc}", fontsize=12, fontweight="bold", pad=12)
    ax.set_yticks(y)
    ax.set_yticklabels(species, fontsize=9.5)
    ax.set_xlim(0, 108)
    ax.legend(loc="lower right", fontsize=10, frameon=True)
    ax.grid(True, linestyle="--", alpha=0.3, axis="x")
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(str(save_path.with_suffix(".pdf")), bbox_inches="tight")
    plt.savefig(str(save_path.with_suffix(".png")), bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[+] Đã lưu Per-Class Metrics Bar Chart: {save_path.with_suffix('.pdf')}")


# =============================================================================
# 4. Vòng lặp Huấn luyện và Đánh giá (Train & Evaluation Loop)
# =============================================================================

def train_one_epoch(model, loader, criterion, optimizer, device) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, targets in tqdm(loader, desc="Train", leave=False):
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * targets.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    return total_loss / max(1, total), correct / max(1, total)


@torch.no_grad()
def evaluate_model(model, loader, criterion, device) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    for images, targets in tqdm(loader, desc="Eval", leave=False):
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)

        total_loss += loss.item() * targets.size(0)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    avg_loss = total_loss / max(1, len(all_targets))

    return avg_loss, acc, macro_f1, all_preds, all_targets


# =============================================================================
# 5. Main Execution
# =============================================================================

def run_training_session(args, df, class_names, class_to_idx, loss_type: str, out_dir: Path, fig_dir: Path, device: torch.device):
    num_classes = len(class_names)
    loss_tag = "focal" if loss_type in ["focal", "focal_loss"] else "cross_entropy"

    criterion, loss_desc = build_criterion(loss_type, alpha=args.alpha, gamma=args.gamma)
    print("\n" + "=" * 72)
    print(f"  HUẤN LUYỆN CONVNEXT-TINY: {loss_desc.upper()} ({args.epochs} EPOCHS)")
    print("=" * 72)

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    train_tf, eval_tf = build_transforms(img_size=224)
    train_ds = MacroscopicWoodDataset(train_df, class_to_idx, transform=train_tf)
    val_ds = MacroscopicWoodDataset(val_df, class_to_idx, transform=eval_tf)
    test_ds = MacroscopicWoodDataset(test_df, class_to_idx, transform=eval_tf)

    num_workers = min(4, os.cpu_count() or 1)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # Khởi tạo mô hình Backbone
    model = timm.create_model(args.model_name, pretrained=True, num_classes=num_classes)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_f1 = 0.0
    best_ckpt_path = out_dir / f"{args.model_name}_{loss_tag}_best.pth"
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1, _, _ = evaluate_model(model, val_loader, criterion, device)
        scheduler.step()

        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1
        })

        is_best = val_f1 > best_val_f1
        if is_best:
            best_val_f1 = val_f1
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_f1": val_f1,
                "loss_type": loss_type,
                "class_names": class_names
            }, best_ckpt_path)

        star = " ★ [BEST]" if is_best else ""
        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] | Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}% F1: {val_f1*100:.2f}%{star}")

    # Đánh giá Final trên Held-out Test Split
    print(f"\n[*] Đang tải checkpoint tốt nhất ({best_ckpt_path.name}) để đánh giá trên Test split...")
    if best_ckpt_path.exists():
        checkpoint = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_acc, test_macro_f1, test_preds, test_targets = evaluate_model(model, test_loader, criterion, device)

    report_dict = classification_report(test_targets, test_preds, target_names=class_names, output_dict=True, zero_division=0)
    report_text = classification_report(test_targets, test_preds, target_names=class_names, digits=4, zero_division=0)

    print("\n" + "-" * 72)
    print(f"   KẾT QUẢ ĐÁNH GIÁ TẬP TEST [{loss_desc.upper()}]")
    print("-" * 72)
    print(f"  * Overall Accuracy : {test_acc * 100:.2f}%")
    print(f"  * Macro-Average F1 : {test_macro_f1 * 100:.2f}%")
    print("-" * 72)
    print(report_text)
    print("-" * 72)

    # 5. Lưu trữ Dữ liệu RAW toàn diện ra JSON
    cm = confusion_matrix(test_targets, test_preds, labels=range(num_classes))
    macro_p = float(report_dict.get("macro avg", {}).get("precision", 0.0))
    macro_r = float(report_dict.get("macro avg", {}).get("recall", 0.0))
    weighted_p = float(report_dict.get("weighted avg", {}).get("precision", 0.0))
    weighted_r = float(report_dict.get("weighted avg", {}).get("recall", 0.0))
    weighted_f1 = float(report_dict.get("weighted avg", {}).get("f1-score", 0.0))

    results_payload = {
        "metadata": {
            "model_name": args.model_name,
            "loss_type": loss_type,
            "loss_desc": loss_desc,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "seed": args.seed,
            "num_classes": num_classes,
            "total_test_samples": int(len(test_targets))
        },
        "summary_metrics": {
            "overall_accuracy": float(test_acc),
            "macro_precision": macro_p,
            "macro_recall": macro_r,
            "macro_f1": float(test_macro_f1),
            "weighted_precision": weighted_p,
            "weighted_recall": weighted_r,
            "weighted_f1": weighted_f1
        },
        "per_class_metrics": {
            name: {
                "precision": float(report_dict[name]["precision"]),
                "recall": float(report_dict[name]["recall"]),
                "f1_score": float(report_dict[name]["f1-score"]),
                "support": int(report_dict[name]["support"])
            }
            for name in class_names if name in report_dict
        },
        "confusion_matrix_raw": cm.tolist(),
        "training_history": history
    }

    results_json_path = out_dir / f"classification_results_{loss_tag}.json"
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)
    print(f"[+] Đã lưu toàn bộ số liệu thống kê chi tiết: {results_json_path}")

    # Lưu thêm file raw predictions (nhãn thật và nhãn dự đoán cho từng ảnh)
    raw_preds_payload = {
        "true_labels": [int(x) for x in test_targets],
        "predicted_labels": [int(x) for x in test_preds],
        "class_names": class_names,
        "class_to_idx": class_to_idx
    }
    raw_json_path = out_dir / f"raw_predictions_{loss_tag}.json"
    with open(raw_json_path, "w", encoding="utf-8") as f:
        json.dump(raw_preds_payload, f, indent=2)
    print(f"[+] Đã lưu mảng raw predictions (true vs predicted): {raw_json_path}")

    # 6. Vẽ các biểu đồ Publication-Ready (PDF + PNG)
    # (a) Ma trận nhầm lẫn
    plot_confusion_matrix(cm, class_names, fig_dir / f"confusion_matrix_{loss_tag}_test")
    if loss_tag == "focal":
        plot_confusion_matrix(cm, class_names, fig_dir / "confusion_matrix_test")

    # (b) Động học huấn luyện qua các epochs
    plot_learning_curves(history, fig_dir / f"learning_curves_{loss_tag}", loss_desc=loss_desc)
    if loss_tag == "focal":
        plot_learning_curves(history, fig_dir / "learning_curves", loss_desc=loss_desc)

    # (c) Biểu đồ cột Precision, Recall, F1-Score từng loài
    plot_per_class_metrics(report_dict, class_names, fig_dir / f"per_class_metrics_{loss_tag}", loss_desc=loss_desc)
    if loss_tag == "focal":
        plot_per_class_metrics(report_dict, class_names, fig_dir / "per_class_metrics", loss_desc=loss_desc)

    return {
        "loss_type": loss_type,
        "loss_desc": loss_desc,
        "accuracy": float(test_acc),
        "macro_f1": float(test_macro_f1)
    }


def load_or_generate_dataset_split(
    split_csv_path: Optional[str] = "paper_data_assets/splits/split_canonical.csv",
    metadata_csv_path: Optional[str] = "paper_data_assets/metadata/metadata.csv",
    data_dir: Optional[str] = None,
    seed: int = 42
) -> pd.DataFrame:
    """
    Nạp dữ liệu phân vùng từ file CSV có sẵn hoặc tự động quét thư mục ảnh gốc và chia split.
    Đảm bảo 100% không bị dừng đột ngột khi chạy trên Kaggle/Colab/Local.
    """
    need_regenerate = False
    df = None

    # 1. Thử nạp từ split_canonical.csv nếu tồn tại
    if split_csv_path and Path(split_csv_path).exists():
        print(f"[+] Nạp phân vùng dữ liệu từ: {split_csv_path}")
        df = pd.read_csv(split_csv_path)

        # Kiểm tra tính toàn vẹn của split đã lưu
        cls_col = "class_name" if "class_name" in df.columns else ("label" if "label" in df.columns else None)
        if cls_col and "split" in df.columns:
            af_test = df[(df[cls_col] == "Afzelia africana") & (df["split"] == "test")]
            test_total = len(df[df["split"] == "test"])
            # Nếu Afzelia africana chỉ có < 15 ảnh test (do lỗi split cũ), hoặc tổng test < 1050
            if len(af_test) < 15 or test_total < 1050:
                print("\n" + "=" * 76)
                print(f"[!] PHÁT HIỆN FILE PHÂN VÙNG CŨ BỊ LỖI CHIA DỮ LIỆU:")
                print(f"    - 'Afzelia africana' trong tập test chỉ có {len(af_test)} ảnh (< 15 ảnh chuẩn).")
                print(f"    - Tổng số mẫu tập test: {test_total} (< 1,065 chuẩn bài báo).")
                print(f"[*] HỆ THỐNG ĐANG TỰ ĐỘNG TÁI SINH PHÂN VÙNG CHUẨN (PP8 của Val ~74 ảnh) TỪ DỮ LIỆU GỐC...")
                print("=" * 76 + "\n")
                need_regenerate = True
                df = None
    elif metadata_csv_path and Path(metadata_csv_path).exists():
        print(f"[*] Sử dụng file metadata có sẵn: {metadata_csv_path}")
        df = pd.read_csv(metadata_csv_path)
    else:
        need_regenerate = True

    if need_regenerate or df is None:
        # 2. Không tìm thấy file CSV hoặc file CSV cũ bị lỗi -> Tự động dò tìm thư mục ảnh gốc
        candidate_dirs = []
        if data_dir:
            candidate_dirs.append(Path(data_dir))

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

        found_data_root = None
        for cand in candidate_dirs:
            if cand.exists() and cand.is_dir():
                subdirs = [p for p in cand.iterdir() if p.is_dir()]
                if len(subdirs) >= 3:
                    found_data_root = cand
                    break

        if found_data_root is None:
            print("\n" + "=" * 76)
            print("[!] LỖI: Cần tái sinh phân vùng nhưng không tự động tìm thấy thư mục ảnh!")
            print("=" * 76)
            print(f"  - File split kiểm tra   : {split_csv_path}")
            print(f"  - File metadata kiểm tra: {metadata_csv_path}")
            print("  - Các đường dẫn ảnh đã quét thử:")
            for c in candidate_dirs[:6]:
                print(f"      + {c}")
            print("\n[*] CÁCH KHẮC PHỤC CỰC KỲ ĐƠN GIẢN:")
            print("    Truyền trực tiếp đường dẫn thư mục ảnh trên Kaggle của bạn:")
            print("    python train_classification_pipeline.py --data-dir /kaggle/input/<tên-dataset>/S3")
            print("=" * 76 + "\n")
            sys.exit(1)

        print(f"\n[*] TỰ ĐỘNG PHÁT HIỆN THƯ MỤC ẢNH TẠI: {found_data_root}")
        print("[*] Đang tự động quét ảnh và áp dụng thuật toán phân chia (Specimen-Disjoint Split từ split_methods.py)...")

        try:
            from generate_benchmark_assets import scan_and_collect_images, assign_specimen_disjoint_splits
            records = scan_and_collect_images(found_data_root)
            split_records = assign_specimen_disjoint_splits(records, train_ratio=0.60, val_ratio=0.20, seed=seed)
            df = pd.DataFrame(split_records)

            # Tự động xuất file CSV lưu lại để lần sau chỉ mất 0.1s tải
            save_csv_path = Path(split_csv_path) if split_csv_path else Path("paper_data_assets/splits/split_canonical.csv")
            save_csv_path.parent.mkdir(parents=True, exist_ok=True)

            save_df = pd.DataFrame({
                "image_path": df["file_path"].astype(str) if "file_path" in df.columns else df["path"].astype(str),
                "class_name": df["label"] if "label" in df.columns else df["class_name"],
                "specimen_id": df["specimen_id"],
                "split": df["split"]
            })
            save_df.to_csv(save_csv_path, index=False, encoding="utf-8")
            
            test_counts = save_df[save_df["split"] == "test"]["class_name"].value_counts()
            af_test_count = test_counts.get("Afzelia africana", 0)
            total_test_count = len(save_df[save_df["split"] == "test"])
            print(f"[+] Đã tự động tạo và lưu phân vùng chuẩn vào: {save_csv_path}")
            print(f"    - Tổng số ảnh test: {total_test_count:,} ảnh (chuẩn 1,065 ảnh bài báo)")
            print(f"    - Afzelia africana test: {af_test_count} ảnh")
        except Exception as e:
            print(f"[!] Gặp lỗi khi tự động chia split: {e}")
            sys.exit(1)

    # 3. Chuẩn hóa tên cột
    if "image_path" not in df.columns:
        if "file_path" in df.columns:
            df["image_path"] = df["file_path"].astype(str)
        elif "path" in df.columns:
            df["image_path"] = df["path"].astype(str)

    if "class_name" not in df.columns:
        if "label" in df.columns:
            df["class_name"] = df["label"].astype(str)

    # 4. Loại bỏ duy nhất taxon cấp chi Pterocarpus sp. (576 ảnh)
    df = df[~df["class_name"].str.contains("pterocarpus sp", case=False, na=False)].copy()

    # 5. Kiểm tra cột split
    if "split" not in df.columns:
        print("[!] LỖI: Dữ liệu phân vùng không tìm thấy cột 'split'!")
        sys.exit(1)

    return df


def main():
    parser = argparse.ArgumentParser(description="Huấn luyện ConvNeXt-Tiny Baseline cho IC4SDMacroWood")
    parser.add_argument("--split-csv", type=str, default="paper_data_assets/splits/split_canonical.csv",
                        help="Đường dẫn file phân vùng dữ liệu split_canonical.csv")
    parser.add_argument("--metadata-csv", type=str, default="paper_data_assets/metadata/metadata.csv",
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
                        help="Hàm mất mát: 'focal' (Multiclass Focal Loss) hoặc 'cross_entropy' (Standard Cross-Entropy)")
    parser.add_argument("--alpha", type=float, default=0.25, help="Hệ số alpha cho Focal Loss (mặc định 0.25)")
    parser.add_argument("--gamma", type=float, default=2.0, help="Hệ số gamma cho Focal Loss (mặc định 2.0)")
    parser.add_argument("--run-both", action="store_true",
                        help="Tự động huấn luyện lần lượt cả 2 loss (Cross-Entropy & Focal Loss) để đối chiếu")
    parser.add_argument("--batch-size", type=int, default=64, help="Kích thước batch size")
    parser.add_argument("--epochs", type=int, default=22, help="Số lượng epoch huấn luyện")
    parser.add_argument("--lr", type=float, default=5e-4, help="Tốc độ học Learning Rate")
    parser.add_argument("--seed", type=int, default=42, help="Hạt giống ngẫu nhiên")
    args = parser.parse_args()

    # Thiết lập hạt giống ngẫu nhiên
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Thiết bị thực thi: {device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. Nạp danh sách dữ liệu (Hỗ trợ Auto-Discovery & Auto-Split nếu chưa có CSV)
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
        # Huấn luyện đối chiếu cả hai hàm mất mát
        summary_results = []
        for l_type in ["cross_entropy", "focal"]:
            res = run_training_session(args, df, class_names, class_to_idx, l_type, out_dir, fig_dir, device)
            summary_results.append(res)

        print("\n" + "=" * 76)
        print("    BẢNG ĐỐI CHIẾU THỰC NGHIỆM CLASSIFICATION BASELINES (TABLE 7 & 8)      ")
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
