"""
01_data_article.modules.classification.metrics
==============================================
Hỗ trợ trực quan hóa kết quả kiểm định kỹ thuật (Confusion Matrix, Learning Curves, Per-Class Bar Chart).
"""

from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], save_path: Path) -> None:
    """Vẽ ma trận nhầm lẫn chuẩn Elsevier với độ tương phản cao và font chữ sắc nét."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 10), dpi=300)

    cm_norm = cm.astype('float') / np.maximum(cm.sum(axis=1)[:, np.newaxis], 1e-12)

    im = plt.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
    im.set_clim(0, 1.0)
    total_n = int(cm.sum())
    plt.title(f"ConvNeXt-Tiny Baseline — Confusion Matrix (Test Split, N={total_n:,})", fontsize=13, fontweight="bold", pad=15)
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label("Normalized Ratio (Recall)", fontsize=10)

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


def plot_learning_curves(history: List[Dict[str, Any]], save_path: Path, loss_desc: str = "") -> None:
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


def plot_per_class_metrics(report_dict: Dict[str, Any], class_names: List[str], save_path: Path, loss_desc: str = "") -> None:
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
