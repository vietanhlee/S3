"""
02_specimen_variant_gan.partitioning
====================================
Gói phân hoạch dữ liệu khoa học, hỗ trợ các giao thức giảm rò rỉ mẫu vật (Specimen-Disjoint, Governed, Graph, Adversarial).
"""

from .base import (
	compute_split_counts,
	validate_split,
	find_optimal_clusters_elbow,
)

from .heuristic_splits import (
	mahalanobis_fixed_split,
	mahalanobis_iterative_split,
	group_based_split,
	agglom_stratified_split,
)

from .graph_splits import (
	hierarchical_clustering_split,
	cosine_graph_split,
)

from .adversarial_and_kfold import (
	stratified_random_split,
	adversarial_validation_split,
	stratified_group_kfold_split,
)

from .governed_split import (
	SPLIT_CONFIG,
	compute_embeddings_v2,
	end_version_split,
)

SPLIT_METHODS = {
	"PP1_Mahalanobis_Fixed": mahalanobis_fixed_split,
	"PP2_Mahalanobis_Iterative": mahalanobis_iterative_split,
	"PP3_Group_Based": group_based_split,
	"PP4_Hierarchical_Clustering": hierarchical_clustering_split,
	"PP5_Cosine_Graph": cosine_graph_split,
	"PP6_Stratified_Random": stratified_random_split,
	"PP7_Adversarial_Validation": adversarial_validation_split,
	"PP8_StratifiedGroupKFold": stratified_group_kfold_split,
	"PP9_Agglom_Stratified": agglom_stratified_split,
}

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
