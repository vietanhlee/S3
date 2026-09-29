"""
01_data_article.modules.classification
======================================
Module phân loại gỗ macroscopic và kiểm định kỹ thuật (Technical Validation).
"""

from .dataset import MacroscopicWoodDataset, build_transforms, load_or_generate_dataset_split
from .losses import MulticlassFocalLoss, build_criterion
from .metrics import plot_confusion_matrix, plot_learning_curves, plot_per_class_metrics
from .engine import train_one_epoch, evaluate_model, run_training_session

__all__ = [
    "MacroscopicWoodDataset",
    "build_transforms",
    "load_or_generate_dataset_split",
    "MulticlassFocalLoss",
    "build_criterion",
    "plot_confusion_matrix",
    "plot_learning_curves",
    "plot_per_class_metrics",
    "train_one_epoch",
    "evaluate_model",
    "run_training_session"
]
