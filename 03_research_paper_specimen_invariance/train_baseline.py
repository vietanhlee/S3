"""
train_baseline.py
=================
Trains the Standard Baseline Model (ConvNeXt-Tiny + ERM / Focal Loss) without adversarial regularization.
Saves the best checkpoint to outputs/baseline/best_baseline.pth.
"""

import os
import sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTHONUNBUFFERED"] = "1"

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

from dataset import create_dataloaders
from models_gan import StandardBaselineModel

CURRENT_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Train Standard Baseline Model")
    parser.add_argument("--backbone", type=str, default="convnext_tiny", help="Visual backbone architecture")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--max_batches", type=int, default=None, help="Max batches per epoch for quick testing")
    parser.add_argument("--lr", type=float, default=3e-4, help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--metadata_csv", type=str, default=str(CURRENT_DIR / "data" / "metadata_11species.csv"))
    parser.add_argument("--image_root", type=str, default=None, help="Root folder of real images (auto-detected if None)")
    parser.add_argument("--output_dir", type=str, default=str(CURRENT_DIR / "outputs" / "baseline"))
    return parser.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train_one_epoch(model, loader, optimizer, criterion, scaler, device, max_batches=None):
    model.train()
    total_loss = 0.0
    all_preds, all_targets = [], []

    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["species_idx"].to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.amp.autocast(device_type, enabled=torch.cuda.is_available()):
            out = model(images)
            loss = criterion(out["species_logits"], targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        preds = torch.argmax(out["species_logits"], dim=-1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_targets.extend(targets.detach().cpu().numpy())

    n_samples = len(all_targets) if all_targets else 1
    avg_loss = total_loss / n_samples
    acc = accuracy_score(all_targets, all_preds) * 100.0 if all_targets else 0.0
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0) * 100.0 if all_targets else 0.0
    return avg_loss, acc, macro_f1


@torch.no_grad()
def evaluate(model, loader, criterion, device, max_batches=None):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["species_idx"].to(device, non_blocking=True)

        with torch.amp.autocast(device_type, enabled=torch.cuda.is_available()):
            out = model(images)
            loss = criterion(out["species_logits"], targets)

        total_loss += loss.item() * images.size(0)
        preds = torch.argmax(out["species_logits"], dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

    n_samples = len(all_targets) if all_targets else 1
    avg_loss = total_loss / n_samples
    acc = accuracy_score(all_targets, all_preds) * 100.0 if all_targets else 0.0
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0) * 100.0 if all_targets else 0.0
    return avg_loss, acc, macro_f1


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "#" * 70)
    print("  TRAINING STANDARD BASELINE MODEL (ConvNeXt-Tiny - ERM)")
    print(f"  Device: {device} | Epochs: {args.epochs} | Batch Size: {args.batch_size}")
    print("#" * 70 + "\n")

    loader_train, loader_seen, loader_unseen, info = create_dataloaders(
        metadata_csv=args.metadata_csv,
        image_root=args.image_root,
        batch_size=args.batch_size,
    )

    print(f"[+] Loaded dataset: {info['num_classes']} species, {info['train_samples']} train images, {info['seen_samples']} seen test images")

    model = StandardBaselineModel(
        backbone_name=args.backbone,
        num_classes=info["num_classes"],
        pretrained=True
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    scaler = torch.amp.GradScaler(device_type, enabled=torch.cuda.is_available())

    best_val_f1 = 0.0
    history = []
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc, train_f1 = train_one_epoch(model, loader_train, optimizer, criterion, scaler, device, max_batches=args.max_batches)
        val_loss, val_acc, val_f1 = evaluate(model, loader_seen, criterion, device, max_batches=args.max_batches)
        scheduler.step()
        elapsed = time.time() - t0

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({elapsed:.1f}s) | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, F1: {train_f1:.2f}% | "
              f"Val (Seen) Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, F1: {val_f1:.2f}%")

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_f1": train_f1,
            "val_seen_loss": val_loss,
            "val_seen_acc": val_acc,
            "val_seen_f1": val_f1,
        }
        history.append(record)

        if val_f1 >= best_val_f1 or epoch == 1:
            best_val_f1 = val_f1
            ckpt_path = out_dir / "best_baseline.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_f1": val_f1,
                "val_acc": val_acc,
                "args": vars(args),
            }, ckpt_path)

    total_time = time.time() - start_time
    print(f"\n[+] Baseline Training Completed in {total_time / 60:.2f} min! Best Val Seen F1: {best_val_f1:.2f}%")
    print(f"[+] Saved checkpoint to: {out_dir / 'best_baseline.pth'}")

    with open(out_dir / "train_history.json", "w", encoding="utf-8") as f:
        json.dump({"history": history, "total_time_sec": total_time, "best_val_f1": best_val_f1}, f, indent=2)


if __name__ == "__main__":
    main()
