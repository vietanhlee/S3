"""
01_data_article.modules.classification.engine
=============================================
Động cơ huấn luyện, đánh giá mô hình phân loại và lưu trữ kết quả thực nghiệm.
"""

import os
import json
from pathlib import Path
from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import timm
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from .dataset import MacroscopicWoodDataset, build_transforms
from .losses import build_criterion
from .metrics import plot_confusion_matrix, plot_learning_curves, plot_per_class_metrics


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) -> Tuple[float, float]:
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
def evaluate_model(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
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


def run_training_session(
    args: Any,
    df: pd.DataFrame,
    class_names: List[str],
    class_to_idx: Dict[str, int],
    loss_type: str,
    out_dir: Path,
    fig_dir: Path,
    device: torch.device
) -> Dict[str, Any]:
    """Thực thi một phiên huấn luyện mô hình ConvNeXt-Tiny hoàn chỉnh."""
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

    print(f"\n[*] Đang nạp checkpoint tốt nhất ({best_ckpt_path.name}) để đánh giá trên Test split...")
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

    cm = confusion_matrix(test_targets, test_preds, labels=range(num_classes))
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
            "macro_precision": float(report_dict.get("macro avg", {}).get("precision", 0.0)),
            "macro_recall": float(report_dict.get("macro avg", {}).get("recall", 0.0)),
            "macro_f1": float(test_macro_f1),
            "weighted_precision": float(report_dict.get("weighted avg", {}).get("precision", 0.0)),
            "weighted_recall": float(report_dict.get("weighted avg", {}).get("recall", 0.0)),
            "weighted_f1": float(report_dict.get("weighted avg", {}).get("f1-score", 0.0)),
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

    raw_json_path = out_dir / f"classification_results_{loss_tag}.json"
    with open(raw_json_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2, ensure_ascii=False)
    print(f"[+] Đã xuất kết quả JSON: {raw_json_path}")

    # Vẽ biểu đồ
    plot_confusion_matrix(cm, class_names, fig_dir / f"confusion_matrix_{loss_tag}_test")
    plot_learning_curves(history, fig_dir / f"learning_curves_{loss_tag}", loss_desc=loss_desc)
    plot_per_class_metrics(report_dict, class_names, fig_dir / f"per_class_metrics_{loss_tag}", loss_desc=loss_desc)

    return {
        "loss_type": loss_type,
        "loss_desc": loss_desc,
        "accuracy": test_acc,
        "macro_f1": test_macro_f1,
        "report_dict": report_dict,
        "raw_json_path": raw_json_path
    }
