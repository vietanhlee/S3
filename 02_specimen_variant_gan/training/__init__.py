"""
02_specimen_variant_gan.training
================================
Package quản lý toàn bộ quy trình huấn luyện, đánh giá, kiểm thử và giải thích mô hình phân loại.
"""

from .engine import (
	FocalLoss,
	accuracy_from_logits,
	train_one_epoch,
	evaluate_one_epoch,
	save_checkpoint,
	train_model,
	build_model,
)

from .evaluator import (
	plot_training_curves,
	collect_predictions,
	plot_confusion_matrix,
	save_report,
	plot_misclassified_samples,
	evaluate_and_report,
)

from .explainability import (
	GradCAM,
	save_gradcam_samples,
)

__all__ = [
	"FocalLoss",
	"accuracy_from_logits",
	"train_one_epoch",
	"evaluate_one_epoch",
	"save_checkpoint",
	"train_model",
	"build_model",
	"plot_training_curves",
	"collect_predictions",
	"plot_confusion_matrix",
	"save_report",
	"plot_misclassified_samples",
	"evaluate_and_report",
	"GradCAM",
	"save_gradcam_samples",
]
