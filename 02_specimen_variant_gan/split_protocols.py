"""
02_specimen_variant_gan.split_protocols
=======================================
Backward-compatibility facade for 02_specimen_variant_gan.partitioning.
"""

from partitioning import (
	compute_split_counts,
	validate_split,
	find_optimal_clusters_elbow,
	mahalanobis_fixed_split,
	mahalanobis_iterative_split,
	group_based_split,
	hierarchical_clustering_split,
	cosine_graph_split,
	stratified_random_split,
	adversarial_validation_split,
	stratified_group_kfold_split,
	agglom_stratified_split,
	SPLIT_CONFIG,
	compute_embeddings_v2,
	end_version_split,
	SPLIT_METHODS,
)

__all__ = [
	"compute_split_counts",
	"validate_split",
	"find_optimal_clusters_elbow",
	"mahalanobis_fixed_split",
	"mahalanobis_iterative_split",
	"group_based_split",
	"hierarchical_clustering_split",
	"cosine_graph_split",
	"stratified_random_split",
	"adversarial_validation_split",
	"stratified_group_kfold_split",
	"agglom_stratified_split",
	"SPLIT_CONFIG",
	"compute_embeddings_v2",
	"end_version_split",
	"SPLIT_METHODS",
]