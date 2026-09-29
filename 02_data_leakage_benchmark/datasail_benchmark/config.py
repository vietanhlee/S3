"""
datasail_benchmark/config.py
============================
Cấu hình hệ thống cho DataSAIL Benchmark & Data Leakage Quantification Pipeline.
"""

from pathlib import Path

# Các đường dẫn dữ liệu ưu tiên (tự động phát hiện)
CANDIDATE_DATASET_PATHS = [
	r"/kaggle/input/datasets/b23dckh002lvitanh/s3-origin/S3",
	r"S3",
	r"g:/S3_paper/S3",
	r"./S3",
	r"../S3",
]

# Các lớp cần loại bỏ theo chuẩn SCDP governance
EXCLUDED_CLASSES = ["Pterocarpus sp", "Peltogyne pubescens"]

# Mô hình trích xuất đặc trưng cố định
FEATURE_EXTRACTOR_NAME = "tf_efficientnetv2_m_in21k"

# Thiết lập tỷ lệ chia dữ liệu chuẩn
TRAIN_RATIO = 0.60
VAL_RATIO = 0.20
TEST_RATIO = 0.20  # 1.0 - TRAIN_RATIO - VAL_RATIO

# 5 Random Seeds cho thử nghiệm thống kê
BENCHMARK_SEEDS = [42, 123, 456, 789, 2024]

# Tham số mặc định khác
BATCH_SIZE = 128
COSINE_THRESHOLD = 0.92
OUTPUT_DIR = Path("outputs_datasail_benchmark")


def resolve_dataset_root() -> Path:
	"""Tự động tìm kiếm thư mục dữ liệu tồn tại trong danh sách ứng viên."""
	for p_str in CANDIDATE_DATASET_PATHS:
		p = Path(p_str)
		if p.exists() and p.is_dir():
			print(f"[Config] Đã phát hiện bộ dữ liệu S3 tại: {p.resolve()}")
			return p
	raise FileNotFoundError(
		f"Không tìm thấy thư mục dữ liệu S3 tại bất kỳ đường dẫn nào trong {CANDIDATE_DATASET_PATHS}"
	)
