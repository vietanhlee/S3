"""Training modules and loops."""
from .base_trainer import BaseTrainer
from .trainer_adversarial import InvarianceTrainer

__all__ = ["BaseTrainer", "InvarianceTrainer"]
