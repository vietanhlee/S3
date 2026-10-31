"""
train_gan.py
============
Trains the Proposed Adversarial Invariant 'GAN' Model:
Feature Extractor + Species Classifier + Specimen Discriminator (via GRL) + CLUB Bottleneck.
Saves the best checkpoint to outputs/gan/best_gan.pth.
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
from models_gan import AdversarialInvariantGAN

CURRENT_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Train Adversarial Invariant GAN Framework")
    parser.add_argument("--backbone", type=str, default="resnet50", help="Visual backbone")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--max_batches", type=int, default=None, help="Max batches per epoch for quick testing")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--adv_weight", type=float, default=0.05, help="Adversarial loss weight beta")
    parser.add_argument("--club_weight", type=float, default=0.005, help="CLUB MI compression weight mu")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--metadata_csv", type=str, default=str(CURRENT_DIR / "data" / "metadata_11species.csv"))
    parser.add_argument("--image_root", type=str, default=None, help="Root folder of real images (auto-detected if None)")
    parser.add_argument("--output_dir", type=str, default=str(CURRENT_DIR / "outputs" / "gan"))
    return parser.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def compute_grl_alpha(epoch: int, total_epochs: int) -> float:
    """Standard dynamic sigmoid schedule for GRL alpha in [0, 1]."""
    p = float(epoch) / float(max(1, total_epochs))
    return float(2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)


def train_one_epoch(model, loader, optimizer, criterion_species, criterion_specimen, scaler, alpha, adv_weight, club_weight, device, max_batches=None):
    model.train()
    model.set_alpha(alpha)

    total_loss, total_species_loss, total_disc_loss, total_club_loss = 0.0, 0.0, 0.0, 0.0
    all_preds, all_targets = [], []

    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        images = batch["image"].to(device, non_blocking=True)
        species_targets = batch["species_idx"].to(device, non_blocking=True)
        specimen_targets = batch["specimen_idx"].to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.amp.autocast(device_type, enabled=torch.cuda.is_available()):
            out = model(images, specimen_labels=specimen_targets, species_labels=species_targets)
            
            loss_species = criterion_species(out["species_logits"], species_targets)
            loss_specimen = criterion_specimen(out["specimen_logits"], specimen_targets)
            loss_club = out.get("club_loss", torch.tensor(0.0, device=device))

            loss_total = loss_species + (adv_weight * alpha * loss_specimen) + (club_weight * loss_club)

        scaler.scale(loss_total).backward()
        scaler.step(optimizer)
        scaler.update()

        n = images.size(0)
        total_loss += loss_total.item() * n
        total_species_loss += loss_species.item() * n
        total_disc_loss += loss_specimen.item() * n
        total_club_loss += loss_club.item() * n

        preds = torch.argmax(out["species_logits"], dim=-1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_targets.extend(species_targets.detach().cpu().numpy())

    n_samples = len(all_targets) if all_targets else 1
    avg_total = total_loss / n_samples
    avg_species = total_species_loss / n_samples
    avg_disc = total_disc_loss / n_samples
    avg_club = total_club_loss / n_samples
    acc = accuracy_score(all_targets, all_preds) * 100.0 if all_targets else 0.0
    f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0) * 100.0 if all_targets else 0.0

    return avg_total, avg_species, avg_disc, avg_club, acc, f1


@torch.no_grad()
def evaluate(model, loader, criterion_species, device, max_batches=None):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        images = batch["image"].to(device, non_blocking=True)
        species_targets = batch["species_idx"].to(device, non_blocking=True)

        with torch.amp.autocast(device_type, enabled=torch.cuda.is_available()):
            out = model(images)
            loss = criterion_species(out["species_logits"], species_targets)

        total_loss += loss.item() * images.size(0)
        preds = torch.argmax(out["species_logits"], dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(species_targets.cpu().numpy())

    n_samples = len(all_targets) if all_targets else 1
    avg_loss = total_loss / n_samples
    acc = accuracy_score(all_targets, all_preds) * 100.0 if all_targets else 0.0
    f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0) * 100.0 if all_targets else 0.0
    return avg_loss, acc, f1


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "#" * 70)
    print("  TRAINING ADVERSARIAL INVARIANT 'GAN' FRAMEWORK (Minimax GRL + CLUB)")
    print(f"  Device: {device} | Epochs: {args.epochs} | Beta: {args.adv_weight} | Mu: {args.club_weight}")
    print("#" * 70 + "\n")

    loader_train, loader_seen, loader_unseen, info = create_dataloaders(
        metadata_csv=args.metadata_csv,
        image_root=args.image_root,
        batch_size=args.batch_size,
    )

    print(f"[+] Loaded dataset: {info['num_classes']} species, {info['train_samples']} train images, {info['train_specimens']} train vouchers")

    model = AdversarialInvariantGAN(
        backbone_name=args.backbone,
        num_classes=info["num_classes"],
        num_specimens=info["train_specimens"],
        pretrained=True,
        adv_weight=args.adv_weight,
        club_weight=args.club_weight,
    ).to(device)

    criterion_species = nn.CrossEntropyLoss()
    criterion_specimen = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    scaler = torch.amp.GradScaler(device_type, enabled=torch.cuda.is_available())

    best_val_f1 = 0.0
    history = []
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        alpha = compute_grl_alpha(epoch, args.epochs)

        train_loss, loss_sp, loss_disc, loss_club, train_acc, train_f1 = train_one_epoch(
            model, loader_train, optimizer, criterion_species, criterion_specimen, scaler,
            alpha, args.adv_weight, args.club_weight, device, max_batches=args.max_batches
        )

        val_loss, val_acc, val_f1 = evaluate(model, loader_seen, criterion_species, device, max_batches=args.max_batches)
        scheduler.step()
        elapsed = time.time() - t0

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] (alpha={alpha:.2f}) | "
              f"Loss: {train_loss:.3f} (Sp: {loss_sp:.3f}, Disc: {loss_disc:.3f}, CLUB: {loss_club:.3f}) | "
              f"Train F1: {train_f1:.2f}% | Val (Seen) F1: {val_f1:.2f}%")

        record = {
            "epoch": epoch,
            "alpha": alpha,
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
            ckpt_path = out_dir / "best_gan.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_f1": val_f1,
                "val_acc": val_acc,
                "args": vars(args),
            }, ckpt_path)

    total_time = time.time() - start_time
    print(f"\n[+] GAN Training Completed in {total_time / 60:.2f} min! Best Val Seen F1: {best_val_f1:.2f}%")
    print(f"[+] Saved checkpoint to: {out_dir / 'best_gan.pth'}")

    with open(out_dir / "train_history.json", "w", encoding="utf-8") as f:
        json.dump({"history": history, "total_time_sec": total_time, "best_val_f1": best_val_f1}, f, indent=2)


if __name__ == "__main__":
    main()
