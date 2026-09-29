"""
datasail_benchmark package
==========================
Package benchmark rò rỉ dữ liệu (Data Leakage) và tối ưu hóa DataSAIL.
"""

from .config import (
	CANDIDATE_DATASET_PATHS,
	EXCLUDED_CLASSES,
	FEATURE_EXTRACTOR_NAME,
	TRAIN_RATIO,
	VAL_RATIO,
	TEST_RATIO,
	BENCHMARK_SEEDS,
	BATCH_SIZE,
	COSINE_THRESHOLD,
	OUTPUT_DIR,
	resolve_dataset_root,
)

__all__ = [
	"CANDIDATE_DATASET_PATHS",
	"EXCLUDED_CLASSES",
	"FEATURE_EXTRACTOR_NAME",
	"TRAIN_RATIO",
	"VAL_RATIO",
	"TEST_RATIO",
	"BENCHMARK_SEEDS",
	"BATCH_SIZE",
	"COSINE_THRESHOLD",
	"OUTPUT_DIR",
	"resolve_dataset_root",
]
