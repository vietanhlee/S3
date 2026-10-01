"""
specimen_invariance_framework/trainers/trainer_adversarial.py
============================================================
Comprehensive trainer implementing the proposed species-conditioned GRL framework
and all 6 groups of specimen-invariance baselines:
1. No intervention: CE, Focal, ArcFace, SupCon, Semi-Hard Triplet
2. General regularization: Strong Reg (Label Smoothing/Weight Decay), Mixup
3. Group robustness: GroupDRO, IRM
4. Adversarial: DANN Unconditional, Conditional GRL (Proposed)
5. Mutual Information: CLUB
6. Strong-simple: Frozen Linear
"""

from typing import Dict, Any, Optional
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

try:
    from trainers.base_trainer import BaseTrainer
    from losses.species_losses import FocalLoss, LabelSmoothingCrossEntropy
    from losses.contrastive_losses import SupConLoss, SemiHardTripletLoss
    from losses.group_robust_losses import GroupDROLoss, IRMLoss
except (ImportError, ValueError):
    try:
        from .base_trainer import BaseTrainer
        from ..losses.species_losses import FocalLoss, LabelSmoothingCrossEntropy
        from ..losses.contrastive_losses import SupConLoss, SemiHardTripletLoss
        from ..losses.group_robust_losses import GroupDROLoss, IRMLoss
    except (ImportError, ValueError):
        from base_trainer import BaseTrainer
        from species_losses import FocalLoss, LabelSmoothingCrossEntropy
        from contrastive_losses import SupConLoss, SemiHardTripletLoss
        from group_robust_losses import GroupDROLoss, IRMLoss


class InvarianceTrainer(BaseTrainer):
    """
    Unified Trainer executing either Proposed Conditional GRL or comparative baselines.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize loss modules
        self.focal_loss = FocalLoss(alpha=self.config.focal_alpha, gamma=self.config.focal_gamma)
        self.label_smoothing_loss = LabelSmoothingCrossEntropy(smoothing=self.config.label_smoothing)
        self.supcon_loss = SupConLoss()
        self.triplet_loss = SemiHardTripletLoss(margin=0.5)
        self.group_dro = GroupDROLoss(num_groups=148, step_size=self.config.group_dro_step_size).to(self.device)
        self.irm_loss = IRMLoss(penalty_weight=self.config.irm_penalty_weight)
        
        # Handle frozen linear baseline
        if self.config.method == "frozen_linear":
            print("[*] Frozen Linear Baseline: Freezing all visual backbone weights.")
            actual_model = self.model.module if hasattr(self.model, "module") else self.model
            for param in actual_model.backbone.parameters():
                param.requires_grad = False

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        total_batches = len(self.train_loader)
        total_steps = self.config.epochs * total_batches
        log_interval = max(1, total_batches // 5)
        t_epoch_start = time.time()
        
        for batch_idx, batch in enumerate(self.train_loader):
            self.optimizer.zero_grad(set_to_none=True)
            images = batch["image"].to(self.device)
            species_targets = batch["species_idx"].to(self.device)
            local_specimen_targets = batch["local_specimen_idx"].to(self.device)
            global_specimen_targets = batch["global_specimen_idx"].to(self.device)
            
            # Apply Fourier style-randomization (preserves anatomy, perturbs lighting/texture)
            if self.config.method not in ("frozen_linear",):
                images = self.fourier_mixer(images)
                
            # Current global training progress p in [0, 1]
            current_step = (epoch - 1) * total_batches + batch_idx
            progress = current_step / max(1, total_steps)
            
            # =========================================================================
            # METHOD DISPATCH
            # =========================================================================
            method = self.config.method.lower()
            
            if method == "conditional_grl":
                # PROPOSED METHOD: Masked Softmax Conditioned on Species
                # L = L_species + lambda_adv(p) * L_specimen|species (via GRL)
                out = self.model(
                    images,
                    species_targets=species_targets,
                    local_specimen_targets=local_specimen_targets,
                    training_progress=progress,
                )
                logits = out["species_logits"]
                loss_species = F.cross_entropy(logits, species_targets)
                loss_adv = out.get("loss_adv_cond", torch.tensor(0.0, device=self.device))
                if isinstance(loss_adv, torch.Tensor):
                    loss_adv = loss_adv.mean()
                lambda_adv = out.get("lambda_adv", 0.0)
                if isinstance(lambda_adv, torch.Tensor):
                    lambda_adv = lambda_adv.mean().item()
                
                # In code, always use '+', GRL automatically reverses gradient to backbone
                loss = loss_species + lambda_adv * loss_adv

            elif method == "dann_unconditional":
                # BASELINE: Unconditional DANN predicting global specimen identity
                out = self.model(
                    images,
                    global_specimen_targets=global_specimen_targets,
                    training_progress=progress,
                )
                logits = out["species_logits"]
                loss_species = F.cross_entropy(logits, species_targets)
                loss_adv = out.get("loss_adv_uncond", torch.tensor(0.0, device=self.device))
                if isinstance(loss_adv, torch.Tensor):
                    loss_adv = loss_adv.mean()
                lambda_adv = out.get("lambda_adv", 0.0)
                if isinstance(lambda_adv, torch.Tensor):
                    lambda_adv = lambda_adv.mean().item()
                loss = loss_species + lambda_adv * loss_adv

            elif method == "club":
                # BASELINE: CLUB Mutual Information Minimization
                out = self.model(
                    images,
                    global_specimen_targets=global_specimen_targets,
                    training_progress=progress,
                )
                logits = out["species_logits"]
                loss_species = F.cross_entropy(logits, species_targets)
                mi_bound = out.get("club_mi_bound", torch.tensor(0.0, device=self.device))
                if isinstance(mi_bound, torch.Tensor):
                    mi_bound = mi_bound.mean()
                var_loss = out.get("club_var_loss", torch.tensor(0.0, device=self.device))
                if isinstance(var_loss, torch.Tensor):
                    var_loss = var_loss.mean()
                
                # Jointly minimize species loss, MI upper bound, and train variational net
                loss = loss_species + 0.1 * mi_bound + var_loss

            elif method == "group_dro":
                # BASELINE: GroupDRO (groups = physical specimens)
                out = self.model(images)
                logits = out["species_logits"]
                loss = self.group_dro(logits, species_targets, global_specimen_targets)

            elif method == "irm":
                # BASELINE: Invariant Risk Minimization across specimens
                out = self.model(images)
                logits = out["species_logits"]
                loss, _ = self.irm_loss(logits, species_targets, global_specimen_targets)

            elif method == "focal":
                # BASELINE: Multiclass Focal Loss
                out = self.model(images)
                loss = self.focal_loss(out["species_logits"], species_targets)

            elif method == "arcface":
                # BASELINE: ArcFace
                out = self.model(images, species_targets=species_targets)
                loss = F.cross_entropy(out["species_logits"], species_targets)

            elif method == "strong_reg":
                # BASELINE: Strong Regularization (Label Smoothing + Dropout)
                out = self.model(images)
                loss = self.label_smoothing_loss(out["species_logits"], species_targets)

            elif method == "mixup":
                # BASELINE: Mixup Data Augmentation
                lam = np.random.beta(self.config.mixup_alpha, self.config.mixup_alpha)
                perm = torch.randperm(images.size(0))
                mixed_images = lam * images + (1.0 - lam) * images[perm]
                
                out = self.model(mixed_images)
                logits = out["species_logits"]
                loss = lam * F.cross_entropy(logits, species_targets) + (1.0 - lam) * F.cross_entropy(logits, species_targets[perm])

            elif method == "supcon":
                # BASELINE: Supervised Contrastive Loss
                out = self.model(images)
                loss_ce = F.cross_entropy(out["species_logits"], species_targets)
                loss_sc = self.supcon_loss(out["projected"], species_targets)
                loss = loss_ce + 0.5 * loss_sc

            elif method == "semihard_triplet":
                # BASELINE: Semi-Hard Triplet Loss
                out = self.model(images)
                loss_ce = F.cross_entropy(out["species_logits"], species_targets)
                loss_triplet = self.triplet_loss(out["projected"], species_targets)
                loss = loss_ce + 0.5 * loss_triplet

            else:
                # Default: Standard Cross Entropy
                out = self.model(images)
                loss = F.cross_entropy(out["species_logits"], species_targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
            
            if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == total_batches:
                elapsed_b = time.time() - t_epoch_start
                running_loss = total_loss / max(1, n_batches)
                cur_loss = loss.item()
                print(
                    f"  [Epoch {epoch:02d}/{self.config.epochs:02d}] "
                    f"Step [{batch_idx + 1:02d}/{total_batches:02d}] "
                    f"| Batch Loss: {cur_loss:.4f} "
                    f"| Running Avg: {running_loss:.4f} "
                    f"| Elapsed: {elapsed_b:.1f}s",
                    flush=True
                )
            
            del images, species_targets, local_specimen_targets, global_specimen_targets, out, loss

        avg_loss = total_loss / max(1, n_batches)
        return {"loss": avg_loss}
