#!/usr/bin/env python3
"""
audit_perceptual_and_embedding_similarity.py
============================================
Công cụ kiểm định độc lập kiểm tra trùng lặp ảnh tri giác (Perceptual Hashing: dHash & pHash)
và độ tương đồng đặc trưng không gian nhúng (Deep Feature Embedding Cosine Similarity)
giữa các phân vùng dữ liệu (Train, Val, Test) cho IC4SDMacroWood (Elsevier Data in Brief / Q1 Journal).

Bối cảnh khoa học:
  - Mã băm mật mã học (SHA-256) chỉ phát hiện được các file trùng lặp chính xác từng bit (exact bitwise duplication).
  - SHA-256 hoàn toàn "mù" trước các ô cắt chồng lấn (overlapping crops), ảnh dịch chuyển vài pixel (sub-pixel shift),
    hoặc các lát cắt liền kề trên cùng một khối gỗ (same-specimen adjacent cuts).
  - Mô-đun này cung cấp kiểm định 3 cấp độ (Three-Tier Leakage Audit):
      Cấp độ 1: Trùng lặp Bitwise Cryptographic (SHA-256).
      Cấp độ 2: Trùng lặp Tri giác (Perceptual Hashing: dHash & pHash, tính khoảng cách Hamming).
      Cấp độ 3: Trùng lặp Không gian Biểu diễn Sâu (ConvNeXt-Tiny Cosine Similarity, ngưỡng >= 0.95).

Cách dùng:
  python audit_perceptual_and_embedding_similarity.py --split-csv paper_data_assets/splits/split_canonical.csv
  python audit_perceptual_and_embedding_similarity.py --split-csv paper_data_assets/splits/split_specimen_disjoint.csv
"""

import os
import sys

# Đảm bảo in tiếng Việt có dấu an toàn tuyệt đối trên Windows terminal (cmd/powershell)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
try:
    import scipy.fftpack
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def _dct2_numpy(matrix: np.ndarray) -> np.ndarray:
    """2D DCT trực giao (Orthonormal 2D DCT) thuần NumPy làm phương án dự phòng chuẩn xác khi không có SciPy."""
    N = matrix.shape[0]
    n = np.arange(N)
    k = np.arange(N)[:, None]
    basis = np.cos(np.pi * (2 * n + 1) * k / (2 * N))
    basis[0, :] *= 1.0 / np.sqrt(2.0)
    basis *= np.sqrt(2.0 / N)
    return np.dot(basis, np.dot(matrix, basis.T))


# =============================================================================
# 1. Các thuật toán Perceptual Hashing (dHash & pHash)
# =============================================================================

def compute_dhash_bool(image_input: Any, hash_size: int = 8) -> np.ndarray:
    """
    Tính dHash (Difference Hash) 64-bit:
    Thu nhỏ ảnh về kích thước (hash_size + 1, hash_size) ở không gian thang độ xám (grayscale).
    So sánh độ sáng giữa 2 pixel liền kề theo từng hàng: pixel[x + 1] > pixel[x].
    Trả về mảng boolean độ dài 64.
    """
    if isinstance(image_input, (str, Path)):
        with Image.open(image_input) as img:
            return compute_dhash_bool(img, hash_size=hash_size)

    gray = image_input.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = np.array(gray, dtype=np.float32)
    diff = pixels[:, 1:] > pixels[:, :-1]
    return diff.flatten()


def compute_phash_bool(image_input: Any, hash_size: int = 8, highfreq_factor: int = 4) -> np.ndarray:
    """
    Tính pHash (Perceptual Hash) 64-bit dựa trên Biến đổi Cosin rời rạc (2D DCT):
    Thu nhỏ ảnh về kích thước (32, 32), tính 2D DCT trực giao,
    lấy 64 hệ số tần số thấp ở góc trên bên trái 8x8 (bỏ qua thành phần DC tại [0,0]),
    so sánh các hệ số với trung vị (median).
    Trả về mảng boolean độ dài 64.
    """
    if isinstance(image_input, (str, Path)):
        with Image.open(image_input) as img:
            return compute_phash_bool(img, hash_size=hash_size, highfreq_factor=highfreq_factor)

    img_size = hash_size * highfreq_factor  # 32x32
    gray = image_input.convert("L").resize((img_size, img_size), Image.Resampling.LANCZOS)
    pixels = np.array(gray, dtype=np.float32)

    if HAS_SCIPY:
        dct = scipy.fftpack.dct(scipy.fftpack.dct(pixels.T, norm="ortho").T, norm="ortho")
    else:
        dct = _dct2_numpy(pixels)

    dct_low = dct[:hash_size, :hash_size]
    med = np.median(dct_low.flatten()[1:])
    return (dct_low > med).flatten()


def bool_array_to_hex(bool_arr: np.ndarray) -> str:
    """Chuyển mảng 64 boolean thành chuỗi 16 ký tự thập lục phân (hex string)."""
    packed = np.packbits(bool_arr)
    return packed.tobytes().hex()


def hex_to_bool_array(hex_str: str) -> np.ndarray:
    """Chuyển chuỗi 16 ký tự thập lục phân thành mảng 64 boolean."""
    byte_vals = bytes.fromhex(hex_str)
    return np.unpackbits(np.frombuffer(byte_vals, dtype=np.uint8)).astype(bool)


def compute_dhash_hex(image_input: Any, hash_size: int = 8) -> str:
    """Tính chuỗi 16 ký tự hex biểu diễn dHash 64-bit."""
    return bool_array_to_hex(compute_dhash_bool(image_input, hash_size=hash_size))


def compute_phash_hex(image_input: Any, hash_size: int = 8, highfreq_factor: int = 4) -> str:
    """Tính chuỗi 16 ký tự hex biểu diễn pHash 64-bit."""
    return bool_array_to_hex(compute_phash_bool(image_input, hash_size=hash_size, highfreq_factor=highfreq_factor))


# =============================================================================
# 2. Tính toán Khoảng cách Hamming và Phân tích Trùng lặp Tri giác
# =============================================================================

def compute_pairwise_hamming_stats(
    query_hashes: np.ndarray,
    gallery_hashes: np.ndarray,
    chunk_size: int = 1000
) -> Dict[str, Any]:
    """
    Tính toán khoảng cách Hamming tối thiểu và thống kê phân bố giữa tập Query và tập Gallery.
    - query_hashes: ma trận boolean (N_query, 64)
    - gallery_hashes: ma trận boolean (N_gallery, 64)
    """
    n_query = len(query_hashes)
    min_distances = np.zeros(n_query, dtype=np.int32)

    # Đếm số cặp có khoảng cách <= các ngưỡng tới hạn
    le_0 = 0  # Giống hệt về mặt tri giác (identical perceptual hash)
    le_2 = 0  # Trùng lặp cục bộ cao (overlapping crop / very near-duplicate)
    le_4 = 0  # Độ tương đồng tri giác đáng kể (subtle shift / texture overlap)
    le_6 = 0  # Tương đồng mức trung bình

    for i in range(0, n_query, chunk_size):
        chunk_q = query_hashes[i:i + chunk_size]
        # Phép XOR giữa hai mảng boolean: True nếu bit khác nhau
        diff = chunk_q[:, None, :] ^ gallery_hashes[None, :, :]
        dist_mat = np.sum(diff, axis=-1)  # Khoảng cách Hamming từ 0 đến 64

        min_dists = np.min(dist_mat, axis=1)
        min_distances[i:i + chunk_size] = min_dists

        le_0 += int(np.sum(min_dists == 0))
        le_2 += int(np.sum(min_dists <= 2))
        le_4 += int(np.sum(min_dists <= 4))
        le_6 += int(np.sum(min_dists <= 6))

    return {
        "num_query": n_query,
        "num_gallery": len(gallery_hashes),
        "min_hamming_distance": int(np.min(min_distances)),
        "mean_min_hamming_distance": round(float(np.mean(min_distances)), 2),
        "median_min_hamming_distance": float(np.median(min_distances)),
        "near_duplicate_count_dist_0": le_0,
        "near_duplicate_count_dist_le_2": le_2,
        "near_duplicate_count_dist_le_4": le_4,
        "near_duplicate_count_dist_le_6": le_6,
        "near_duplicate_ratio_le_4_pct": round(float(le_4 / max(1, n_query) * 100.0), 2),
    }


# =============================================================================
# 3. Phân tích Tương đồng Cosine trên Không gian Nhúng Sâu (Deep Embeddings)
# =============================================================================

def compute_cross_split_embedding_similarity(
    query_embs: np.ndarray,
    gallery_embs: np.ndarray,
    chunk_size: int = 1000
) -> Dict[str, Any]:
    """
    Tính toán độ tương đồng Cosine cực đại giữa tập Query và tập Gallery.
    Giả định các vector đặc trưng đã được chuẩn hóa L2 (||z||_2 = 1.0).
    """
    n_query = len(query_embs)
    max_similarities = np.zeros(n_query, dtype=np.float32)

    gt_98 = 0  # Rất gần trùng lặp (Near-Duplicate representation, cos >= 0.98)
    gt_95 = 0  # Tương đồng cực cao (High overlap representation, cos >= 0.95)
    gt_90 = 0  # Tương đồng cao (cos >= 0.90)

    for i in range(0, n_query, chunk_size):
        chunk_q = query_embs[i:i + chunk_size]
        sim_mat = np.dot(chunk_q, gallery_embs.T)  # Cosine similarity in [-1, 1]

        max_sims = np.max(sim_mat, axis=1)
        max_similarities[i:i + chunk_size] = max_sims

        gt_98 += int(np.sum(max_sims >= 0.98))
        gt_95 += int(np.sum(max_sims >= 0.95))
        gt_90 += int(np.sum(max_sims >= 0.90))

    return {
        "num_query": n_query,
        "num_gallery": len(gallery_embs),
        "max_cosine_similarity": round(float(np.max(max_similarities)), 4),
        "mean_max_cosine_similarity": round(float(np.mean(max_similarities)), 4),
        "median_max_cosine_similarity": round(float(np.median(max_similarities)), 4),
        "percentile_95_cosine": round(float(np.percentile(max_similarities, 95)), 4),
        "percentile_99_cosine": round(float(np.percentile(max_similarities, 99)), 4),
        "count_similarity_ge_0_98": gt_98,
        "count_similarity_ge_0_95": gt_95,
        "count_similarity_ge_0_90": gt_90,
        "ratio_ge_0_95_pct": round(float(gt_95 / max(1, n_query) * 100.0), 2),
        "ratio_ge_0_90_pct": round(float(gt_90 / max(1, n_query) * 100.0), 2)
    }


# =============================================================================
# 4. Trích xuất đặc trưng ConvNeXt-Tiny nếu cần
# =============================================================================

def extract_convnext_embeddings_from_paths(
    image_paths: List[str],
    batch_size: int = 64,
    device: str = "cuda"
) -> np.ndarray:
    """Trích xuất ma trận biểu diễn 768 chiều L2-normalized từ ConvNeXt-Tiny."""
    import torch
    import torchvision.transforms as transforms
    import timm

    print(f"[*] Khởi tạo ConvNeXt-Tiny để trích xuất embeddings trên: {device}...")
    model = timm.create_model("convnext_tiny", pretrained=True, num_classes=0).to(device)
    model.eval()

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    all_embs = []
    use_cuda = (device == "cuda" and torch.cuda.is_available())

    with torch.no_grad():
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            tensors = []
            for p in batch_paths:
                try:
                    img = Image.open(p).convert("RGB")
                    tensors.append(tf(img))
                except Exception:
                    tensors.append(torch.zeros(3, 224, 224))
            batch_tensor = torch.stack(tensors).to(device)
            embs = model(batch_tensor)
            embs = torch.nn.functional.normalize(embs, p=2, dim=-1)
            all_embs.append(embs.cpu().numpy())

    return np.concatenate(all_embs, axis=0)


# =============================================================================
# 5. Pipeline Điều phối Kiểm định
# =============================================================================

def audit_perceptual_and_embeddings(
    df: pd.DataFrame,
    embeddings: Optional[np.ndarray] = None,
    protocol_name: str = "Protocol"
) -> Dict[str, Any]:
    """
    Thực hiện kiểm định toàn diện 3 cấp độ trên DataFrame phân vùng:
      1. Cryptographic Hash (SHA-256)
      2. Perceptual Hashing (dHash & pHash)
      3. Deep Feature Cosine Similarity
    """
    train_mask = (df["split"] == "train").values
    val_mask = (df["split"] == "val").values
    test_mask = (df["split"] == "test").values

    audit_report = {
        "protocol": protocol_name,
        "total_samples": len(df),
        "split_counts": {
            "train": int(np.sum(train_mask)),
            "val": int(np.sum(val_mask)),
            "test": int(np.sum(test_mask))
        }
    }

    # 1. SHA-256 Audit
    if "sha256" in df.columns:
        train_sha = set(df[train_mask]["sha256"])
        val_sha = set(df[val_mask]["sha256"])
        test_sha = set(df[test_mask]["sha256"])
        audit_report["sha256_bitwise_overlap"] = {
            "train_vs_val": len(train_sha & val_sha),
            "train_vs_test": len(train_sha & test_sha),
            "val_vs_test": len(val_sha & test_sha)
        }

    # 2. Perceptual Hashing Audit (dHash & pHash)
    has_dhash = "dhash" in df.columns
    has_phash = "phash" in df.columns

    if has_dhash:
        dhash_arr = np.array([hex_to_bool_array(h) for h in df["dhash"]])
        train_dhash = dhash_arr[train_mask]
        val_dhash = dhash_arr[val_mask]
        test_dhash = dhash_arr[test_mask]

        audit_report["dhash_perceptual_audit"] = {
            "test_against_train": compute_pairwise_hamming_stats(test_dhash, train_dhash),
            "val_against_train": compute_pairwise_hamming_stats(val_dhash, train_dhash),
            "test_against_val": compute_pairwise_hamming_stats(test_dhash, val_dhash),
        }

    if has_phash:
        phash_arr = np.array([hex_to_bool_array(h) for h in df["phash"]])
        train_phash = phash_arr[train_mask]
        val_phash = phash_arr[val_mask]
        test_phash = phash_arr[test_mask]

        audit_report["phash_perceptual_audit"] = {
            "test_against_train": compute_pairwise_hamming_stats(test_phash, train_phash),
            "val_against_train": compute_pairwise_hamming_stats(val_phash, train_phash),
            "test_against_val": compute_pairwise_hamming_stats(test_phash, val_phash),
        }

    # 3. Deep Feature Similarity Audit
    if embeddings is not None and len(embeddings) == len(df):
        train_embs = embeddings[train_mask]
        val_embs = embeddings[val_mask]
        test_embs = embeddings[test_mask]

        audit_report["embedding_similarity_audit"] = {
            "test_against_train": compute_cross_split_embedding_similarity(test_embs, train_embs),
            "val_against_train": compute_cross_split_embedding_similarity(val_embs, train_embs),
            "test_against_val": compute_cross_split_embedding_similarity(test_embs, val_embs),
        }

    return audit_report


# Bí danh tương thích ngược cho module curation
run_leakage_audit = audit_perceptual_and_embeddings


def print_audit_comparison_table(report_can: Dict[str, Any], report_dis: Optional[Dict[str, Any]] = None):
    """In bảng đối chiếu báo cáo kiểm định tri giác và đặc trưng chuẩn Elsevier."""
    print("\n" + "=" * 94)
    print("      BÁO CÁO KIỂM ĐỊNH TRÙNG LẶP TRI GIÁC VÀ TƯƠNG ĐỒNG ĐẶC TRƯNG ĐA TẦNG       ")
    print("=" * 94)
    col2_title = report_dis.get("protocol", "Specimen-Disjoint") if report_dis else "N/A"
    print(f"{'Tiêu chí kiểm định bảo mật dữ liệu':<46} | {report_can.get('protocol', 'Canonical'):<20} | {col2_title:<20}")
    print("-" * 94)

    # 1. SHA-256
    sha_can = report_can.get("sha256_bitwise_overlap", {}).get("train_vs_test", 0)
    sha_dis = report_dis.get("sha256_bitwise_overlap", {}).get("train_vs_test", 0) if report_dis else 0
    print(f"{'1. Trùng lặp chính xác bitwise (SHA-256 Overlap)':<46} | {sha_can:<20} | {sha_dis:<20}")

    # 2. dHash
    d_can = report_can.get("dhash_perceptual_audit", {}).get("test_against_train", {})
    d_dis = report_dis.get("dhash_perceptual_audit", {}).get("test_against_train", {}) if report_dis else {}

    if d_can:
        d0_c, d0_d = d_can.get("near_duplicate_count_dist_0", 0), d_dis.get("near_duplicate_count_dist_0", 0)
        print(f"{'2. Trùng lặp tri giác tuyệt đối (dHash Hamming = 0)':<46} | {d0_c:<20} | {d0_d:<20}")

        d4_c, d4_d = d_can.get("near_duplicate_count_dist_le_4", 0), d_dis.get("near_duplicate_count_dist_le_4", 0)
        d4_c_pct = d_can.get("near_duplicate_ratio_le_4_pct", 0.0)
        d4_d_pct = d_dis.get("near_duplicate_ratio_le_4_pct", 0.0)
        print(f"{'3. Tương đồng tri giác cao (dHash Hamming <= 4)':<46} | {f'{d4_c} ({d4_c_pct}%)':<20} | {f'{d4_d} ({d4_d_pct}%)':<20}")

    # 3. Embedding Similarity
    e_can = report_can.get("embedding_similarity_audit", {}).get("test_against_train", {})
    e_dis = report_dis.get("embedding_similarity_audit", {}).get("test_against_train", {}) if report_dis else {}

    if e_can:
        max_c, max_d = e_can.get("max_cosine_similarity", 0.0), e_dis.get("max_cosine_similarity", 0.0)
        print(f"{'4. Tương đồng Cosine cực đại (Max Cosine Sim)':<46} | {max_c:<20.4f} | {max_d:<20.4f}")

        c95_c, c95_d = e_can.get("count_similarity_ge_0_95", 0), e_dis.get("count_similarity_ge_0_95", 0)
        c95_c_pct = e_can.get("ratio_ge_0_95_pct", 0.0)
        c95_d_pct = e_dis.get("ratio_ge_0_95_pct", 0.0)
        print(f"{'5. Trùng lặp biểu diễn sâu (Cosine Sim >= 0.95)':<46} | {f'{c95_c} ({c95_c_pct}%)':<20} | {f'{c95_d} ({c95_d_pct}%)':<20}")

        p95_c, p95_d = e_can.get("percentile_95_cosine", 0.0), e_dis.get("percentile_95_cosine", 0.0)
        print(f"{'6. Phân vị 95% tương đồng Cosine (P95 Sim)':<46} | {p95_c:<20.4f} | {p95_d:<20.4f}")

    print("=" * 94)


def main():
    parser = argparse.ArgumentParser(
        description="Kiểm định Trùng lặp Tri giác (pHash/dHash) và Biểu diễn Sâu (Cosine Similarity) cho IC4SDMacroWood"
    )
    parser.add_argument("--split-csv", type=str, default="paper_data_assets/splits/split_canonical.csv",
                        help="Đường dẫn file split CSV cần kiểm định")
    parser.add_argument("--compare-split-csv", type=str, default=None,
                        help="Đường dẫn file split CSV thứ 2 để đối chiếu (ví dụ: split_specimen_disjoint.csv)")
    parser.add_argument("--metadata-csv", type=str, default="paper_data_assets/metadata/metadata.csv",
                        help="Đường dẫn file metadata.csv")
    parser.add_argument("--embeddings-npy", type=str, default=None,
                        help="Đường dẫn file ma trận embedding ConvNeXt-Tiny (.npy)")
    parser.add_argument("--output-json", type=str, default="paper_data_assets/leakage_audit/perceptual_audit_report.json",
                        help="Đường dẫn file JSON lưu kết quả kiểm định")
    args = parser.parse_args()

    split_path = Path(args.split_csv)
    if not split_path.exists():
        if Path("out/splits/split_canonical.csv").exists():
            split_path = Path("out/splits/split_canonical.csv")
        else:
            print(f"[!] Không tìm thấy file split tại: {split_path}")
            sys.exit(1)

    print(f"[*] Nạp phân vùng dữ liệu chính từ: {split_path}")
    df1 = pd.read_csv(split_path)

    # Nạp metadata nếu có để lấy dhash/phash hoặc image_path
    meta_path = Path(args.metadata_csv)
    if not meta_path.exists() and Path("out/metadata/metadata.csv").exists():
        meta_path = Path("out/metadata/metadata.csv")

    if meta_path.exists():
        meta_df = pd.read_csv(meta_path)
        for col in ["dhash", "phash", "sha256"]:
            if col in meta_df.columns and col not in df1.columns:
                df1[col] = meta_df[col]

    # Kiểm tra xem có ảnh trên đĩa để tính dHash/pHash on the fly không
    img_col = "image_path" if "image_path" in df1.columns else ("file_path" if "file_path" in df1.columns else None)
    if img_col and ("dhash" not in df1.columns or "phash" not in df1.columns):
        first_path = Path(df1[img_col].iloc[0])
        if first_path.exists():
            print("[*] Đang tính toán dHash & pHash cho toàn bộ tập ảnh trên đĩa...")
            dhashes = []
            phashes = []
            for p_str in tqdm(df1[img_col], desc="Perceptual Hashing", leave=False):
                try:
                    img = Image.open(p_str)
                    d_bool = compute_dhash_bool(img)
                    p_bool = compute_phash_bool(img)
                    dhashes.append(bool_array_to_hex(d_bool))
                    phashes.append(bool_array_to_hex(p_bool))
                except Exception:
                    dhashes.append("0" * 16)
                    phashes.append("0" * 16)
            df1["dhash"] = dhashes
            df1["phash"] = phashes

    # Nạp embeddings nếu có
    embs = None
    if args.embeddings_npy and Path(args.embeddings_npy).exists():
        embs = np.load(args.embeddings_npy)
    elif Path("out/embeddings/convnext_tiny.npy").exists():
        embs = np.load("out/embeddings/convnext_tiny.npy")
    elif Path("paper_data_assets/embeddings/convnext_tiny.npy").exists():
        embs = np.load("paper_data_assets/embeddings/convnext_tiny.npy")

    report1 = audit_perceptual_and_embeddings(df1, embeddings=embs, protocol_name=split_path.stem)

    report2 = None
    if args.compare_split_csv and Path(args.compare_split_csv).exists():
        df2 = pd.read_csv(args.compare_split_csv)
        if meta_path.exists():
            meta_df = pd.read_csv(meta_path)
            for col in ["dhash", "phash", "sha256"]:
                if col in meta_df.columns and col not in df2.columns:
                    df2[col] = meta_df[col]
        report2 = audit_perceptual_and_embeddings(df2, embeddings=embs, protocol_name=Path(args.compare_split_csv).stem)

    print_audit_comparison_table(report1, report2)

    # Lưu kết quả JSON
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_payload = {
        "primary_audit": report1,
        "comparison_audit": report2
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2)
    print(f"\n[+] Đã lưu báo cáo kiểm định tri giác toàn diện vào: {out_json}")


if __name__ == "__main__":
    main()
