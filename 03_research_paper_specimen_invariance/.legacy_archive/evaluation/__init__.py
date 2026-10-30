"""Evaluation modules: probe, calibration, metrics, visualization."""
from .specimen_probe import evaluate_specimen_recoverability, extract_all_embeddings
from .calibration import compute_calibration_metrics, plot_reliability_diagram
from .metrics import compute_ggsl, summarize_paired_folds
from .visualizer import plot_tsne_species_vs_specimen, plot_pareto_curve, plot_correlation_sri_vs_ggsl

__all__ = [
    "evaluate_specimen_recoverability",
    "extract_all_embeddings",
    "compute_calibration_metrics",
    "plot_reliability_diagram",
    "compute_ggsl",
    "summarize_paired_folds",
    "plot_tsne_species_vs_specimen",
    "plot_pareto_curve",
    "plot_correlation_sri_vs_ggsl",
]
