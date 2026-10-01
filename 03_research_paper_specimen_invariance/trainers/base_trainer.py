"""
specimen_invariance_framework/trainers/base_trainer.py
======================================================
Base training loop supporting differential learning rates (backbone < head),
cosine annealing with warmup, and strict model selection on specimen-disjoint validation sets.
"""

import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import asdict, is_dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

try:
    from config import TrainingConfig, ModelConfig, safe_load_checkpoint
    from datasets.augmentations import FourierAmplitudeMixing
except (ImportError, ValueError):
    try:
        from ..config import TrainingConfig, ModelConfig, safe_load_checkpoint
        from ..datasets.augmentations import FourierAmplitudeMixing
    except (ImportError, ValueError):
        from config import TrainingConfig, ModelConfig, safe_load_checkpoint
        from augmentations import FourierAmplitudeMixing


class BaseTrainer:
    """
    Core Trainer enforcing strict validation on specimen-disjoint splits.
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        config: TrainingConfig,
        model_config: ModelConfig,
        device: torch.device,
        output_dir: str,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.model_config = model_config
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Style randomization
        self.fourier_mixer = FourierAmplitudeMixing(alpha=0.5, p=0.5).to(device)
        
        # Optimizer with parameter groups: backbone lr < head lr
        self.optimizer = self._build_optimizer()
        
        # Cosine annealing scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.epochs,
            eta_min=config.min_lr
        )
        
        self.best_val_metric = -1.0
        self.best_epoch = -1
        self.history: List[Dict[str, float]] = []

    def _build_optimizer(self) -> torch.optim.Optimizer:
        backbone_params = []
        head_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "backbone" in name:
                backbone_params.append(param)
            else:
                head_params.append(param)
                
        param_groups = [
            {"params": backbone_params, "lr": self.config.lr_backbone, "weight_decay": self.config.weight_decay},
            {"params": head_params, "lr": self.config.lr_head, "weight_decay": self.config.weight_decay},
        ]
        return torch.optim.AdamW(param_groups)

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Subclasses implement specific loss computation."""
        raise NotImplementedError

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        """Evaluates classification performance on a given split."""
        self.model.eval()
        all_preds = []
        all_targets = []
        total_loss = 0.0
        n_samples = 0
        
        ce_fn = nn.CrossEntropyLoss(reduction="sum")
        
        for batch in loader:
            images = batch["image"].to(self.device)
            targets = batch["species_idx"].to(self.device)
            
            out = self.model(images)
            logits = out["species_logits"]
            loss = ce_fn(logits, targets)
            
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            total_loss += loss.item()
            n_samples += targets.size(0)
            
        acc = accuracy_score(all_targets, all_preds) * 100.0
        macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0) * 100.0
        weighted_f1 = f1_score(all_targets, all_preds, average="weighted", zero_division=0) * 100.0
        avg_loss = total_loss / max(1, n_samples)
        
        return {
            "loss": avg_loss,
            "accuracy": acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
        }

    def train(self) -> Dict[str, Any]:
        """
        Executes end-to-end training loop.
        Checkpointing strictly adheres to specimen-disjoint validation metrics.
        """
        best_ckpt_path = self.output_dir / "best_model.pth"
        print(f"[*] Starting training ({self.config.method}) for {self.config.epochs} epochs...", flush=True)
        print(f"[*] Strict checkpoint selection on specimen-disjoint validation set.", flush=True)
        
        for epoch in range(1, self.config.epochs + 1):
            lr_current = self.optimizer.param_groups[0]["lr"]
            print(f"\n>>> Epoch [{epoch:02d}/{self.config.epochs:02d}] (Current LR: {lr_current:.2e})", flush=True)
            t0 = time.time()
            train_metrics = self.train_epoch(epoch)
            print(f"  [*] Validating on strict specimen-disjoint set...", flush=True)
            val_metrics = self.evaluate(self.val_loader)
            self.scheduler.step()
            elapsed = time.time() - t0
            
            current_metric = val_metrics["macro_f1"] if self.config.metric_for_best == "macro_f1" else val_metrics["accuracy"]
            is_best = current_metric > self.best_val_metric
            
            if is_best:
                self.best_val_metric = current_metric
                self.best_epoch = epoch
                cfg_payload = asdict(self.config) if is_dataclass(self.config) else (
                    self.config.__dict__ if hasattr(self.config, "__dict__") else self.config
                )
                raw_model = self.model.module if hasattr(self.model, "module") else self.model
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": raw_model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "config": cfg_payload,
                }, best_ckpt_path)
                
            log_entry = {
                "epoch": epoch,
                "train_loss": train_metrics.get("loss", 0.0),
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "is_best": is_best,
            }
            self.history.append(log_entry)
            
            print(f"  [=] Epoch [{epoch:02d}/{self.config.epochs:02d}] Summary | "
                  f"Train Loss: {train_metrics.get('loss', 0.0):.4f} | "
                  f"Val Acc: {val_metrics['accuracy']:.2f}% | "
                  f"Val Macro-F1: {val_metrics['macro_f1']:.2f}% "
                  f"{'(*) NEW BEST' if is_best else ''} ({elapsed:.1f}s)", flush=True)
            
            # Periodic memory garbage collection to avoid accumulation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc
            gc.collect()
                  
        # Load best checkpoint for final held-out test evaluation
        if best_ckpt_path.exists():
            checkpoint = safe_load_checkpoint(best_ckpt_path, map_location=self.device)
            raw_model = self.model.module if hasattr(self.model, "module") else self.model
            raw_model.load_state_dict(checkpoint["model_state_dict"])
            print(f"\n[+] Loaded best checkpoint from Epoch {self.best_epoch} (Val Macro-F1: {self.best_val_metric:.2f}%)", flush=True)
            
        print(f"[*] Evaluating on strictly held-out test set...", flush=True)
        test_metrics = self.evaluate(self.test_loader)
        print(f"[SUCCESS] Held-Out Strict Evaluation Test Acc: {test_metrics['accuracy']:.2f}% | Macro-F1: {test_metrics['macro_f1']:.2f}%\n", flush=True)
        
        return {
            "best_epoch": self.best_epoch,
            "best_val_metric": self.best_val_metric,
            "test_metrics": test_metrics,
            "history": self.history,
        }
