"""
02_specimen_variant_gan.training.engine
=======================================
Hạ tầng huấn luyện phân loại gỗ vi mô: Loss functions, Optimizer loop, Early stopping, Model builders.
"""

from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import timm
from tqdm import tqdm

from utils import freeze_model_layers


class FocalLoss(nn.Module):
	"""Focal Loss cho phân loại đa lớp không cân bằng."""
	def __init__(self, gamma: float = 2.0, alpha: float = 0.25) -> None:
		super().__init__()
		self.gamma = gamma
		self.alpha = alpha

	def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
		ce = F.cross_entropy(logits, targets, reduction="none")
		pt = torch.exp(-ce)
		loss = self.alpha * (1 - pt) ** self.gamma * ce
		return loss.mean()


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
	preds = torch.argmax(logits, dim=1)
	correct = (preds == targets).sum().item()
	return correct / max(len(targets), 1)


def train_one_epoch(
	model: nn.Module,
	loader: DataLoader,
	criterion: nn.Module,
	optimizer: torch.optim.Optimizer,
	device: torch.device,
	epoch: int,
	epochs: int,
) -> Tuple[float, float]:
	model.train()
	running_loss, running_acc, count = 0.0, 0.0, 0
	pbar = tqdm(loader, desc=f"Train {epoch}/{epochs}")
	for images, targets in pbar:
		images = images.to(device)
		targets = targets.to(device)

		optimizer.zero_grad()
		logits = model(images)
		loss = criterion(logits, targets)
		loss.backward()
		optimizer.step()

		batch_size = targets.size(0)
		running_loss += loss.item() * batch_size
		running_acc += accuracy_from_logits(logits, targets) * batch_size
		count += batch_size
		pbar.set_postfix(loss=running_loss / count, acc=running_acc / count)

	return running_loss / count, running_acc / count


@torch.no_grad()
def evaluate_one_epoch(
	model: nn.Module,
	loader: DataLoader,
	criterion: nn.Module,
	device: torch.device,
	epoch: int,
	epochs: int,
) -> Tuple[float, float]:
	model.eval()
	running_loss, running_acc, count = 0.0, 0.0, 0
	pbar = tqdm(loader, desc=f"Val {epoch}/{epochs}")
	for images, targets in pbar:
		images = images.to(device)
		targets = targets.to(device)

		logits = model(images)
		loss = criterion(logits, targets)
		batch_size = targets.size(0)
		running_loss += loss.item() * batch_size
		running_acc += accuracy_from_logits(logits, targets) * batch_size
		count += batch_size
		pbar.set_postfix(loss=running_loss / count, acc=running_acc / count)

	return running_loss / count, running_acc / count


def save_checkpoint(
	path: Path,
	model: nn.Module,
	optimizer: torch.optim.Optimizer,
	epoch: int,
	best_val_acc: float,
	history: dict,
) -> None:
	payload = {
		"epoch": epoch,
		"model_state": model.state_dict(),
		"optimizer_state": optimizer.state_dict(),
		"best_val_acc": best_val_acc,
		"history": history,
		"model_name": getattr(model, "model_name", "model"),
	}
	torch.save(payload, path)


def train_model(
	model: nn.Module,
	train_loader: DataLoader,
	val_loader: DataLoader,
	optimizer: torch.optim.Optimizer,
	criterion: nn.Module,
	device: torch.device,
	epochs: int,
	patience: int,
	output_dir: Path,
	scheduler: Optional[Any] = None,
) -> dict:
	history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
	best_val_acc = 0.0
	epochs_no_improve = 0
	model_name = getattr(model, "model_name", "model")

	for epoch in range(1, epochs + 1):
		train_loss, train_acc = train_one_epoch(
			model, train_loader, criterion, optimizer, device, epoch, epochs
		)
		val_loss, val_acc = evaluate_one_epoch(
			model, val_loader, criterion, device, epoch, epochs
		)

		history["train_loss"].append(train_loss)
		history["train_acc"].append(train_acc)
		history["val_loss"].append(val_loss)
		history["val_acc"].append(val_acc)

		current_lr = optimizer.param_groups[0]["lr"]
		print(
			f"Epoch {epoch}/{epochs} - "
			f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
			f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, "
			f"lr={current_lr:.6f}"
		)

		try:
			import wandb
			if wandb.run is not None:
				wandb.log({
					"epoch": epoch,
					"train/loss": train_loss,
					"train/acc": train_acc,
					"val/loss": val_loss,
					"val/acc": val_acc,
					"lr": current_lr
				}, step=epoch)
		except Exception:
			pass

		if scheduler is not None:
			scheduler.step()

		last_path = output_dir / "last_epoch.pth"
		save_checkpoint(last_path, model, optimizer, epoch, best_val_acc, history)

		if val_acc > best_val_acc:
			best_val_acc = val_acc
			best_path = output_dir / f"best_model_{model_name}.pth"
			torch.save(model.state_dict(), best_path)
			epochs_no_improve = 0
		else:
			epochs_no_improve += 1

		if epochs_no_improve >= patience:
			print(f"Early stopping kích hoạt tại epoch {epoch} (patience={patience})")
			break

	history["best_val_acc"] = best_val_acc
	return history


def build_model(
	num_classes: int,
	model_name: str = "convnext_tiny",
	freeze_ratio: float = 0.7,
) -> nn.Module:
	"""Khởi tạo mô hình thị giác pretrained từ timm với tỷ lệ đóng băng backbone xác định."""
	model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
	freeze_model_layers(model, freeze_ratio)
	model.model_name = model_name
	return model
