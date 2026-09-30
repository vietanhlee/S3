"""
split_methods.py (Root Facade)
==============================
Tương thích ngược cho các lệnh import từ root của package 01_data_paper_forensic_cites.
"""

from modules.curation.split_methods import (
    validate_split,
    compute_split_counts,
    group_based_split,
    stratified_group_kfold_split,
    mahalanobis_fixed_split,
    mahalanobis_iterative_split,
    hierarchical_clustering_split,
    cosine_graph_split,
    stratified_random_split,
    adversarial_validation_split,
    agglom_stratified_split,
    SPLIT_METHODS,
)

__all__ = [
    "validate_split",
    "compute_split_counts",
    "group_based_split",
    "stratified_group_kfold_split",
    "mahalanobis_fixed_split",
    "mahalanobis_iterative_split",
    "hierarchical_clustering_split",
    "cosine_graph_split",
    "stratified_random_split",
    "adversarial_validation_split",
    "agglom_stratified_split",
    "SPLIT_METHODS",
]
