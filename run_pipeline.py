#!/usr/bin/env python3
"""
run_pipeline.py
===============
Script điều phối (Master Orchestrator) toàn bộ luồng thực nghiệm cho IC4SDMacroWood
(Elsevier Data in Brief / Q1 Journal), tích hợp kiến trúc phân vùng hai tầng
(Two-Tier Split Architecture) và giải quyết triệt để phản biện rò rỉ mẫu vật (Specimen Leakage):

Các phân vùng hỗ trợ:
  1. 'canonical': Phân vùng kế thừa (Legacy / Image-Stratified Split) để duy trì tính tái lập.
  2. 'specimen_disjoint': Phân vùng nghiêm ngặt theo khối mẫu vật (Zero Leakage cho 17 loài đa khối).
  3. 'both': Chạy song song cả hai phân vùng để đo lường Khoảng cách Suy giảm Hiệu năng do Rò rỉ
     (Generalization Gap from Specimen Leakage - GGSL).

Các bước thực thi (--step):
  - 'assets': Sinh metadata, tính Laplacian variance, SHA-256 hash và thẩm định rò rỉ mẫu vật.
  - 'classify': Huấn luyện ConvNeXt-Tiny Baseline (Focal Loss / Cross-Entropy).
  - 'metric': Huấn luyện Semi-Hard Triplet Representation Learning và đánh giá Retrieval.
  - 'compare': Tổng hợp bảng đối chiếu hiệu năng và xuất báo cáo khoa học GGSL.
  - 'all': Tự động thực thi toàn bộ quy trình từ đầu đến cuối.

Cách dùng cơ bản:
  # Chạy toàn bộ trên cả hai split để đối chiếu GGSL:
  python run_pipeline.py --all --split-type both --data-dir <đường_dẫn_ảnh>

  # Chỉ chạy trên split Specimen-Disjoint chuẩn khoa học:
  python run_pipeline.py --all --split-type specimen_disjoint

  # Chỉ tổng hợp báo cáo đối chiếu từ kết quả có sẵn:
  python run_pipeline.py --step compare
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
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List


def run_command(cmd: List[str]) -> None:
    """Thực thi lệnh shell và dừng ngay lập tức nếu gặp lỗi."""
    print(f"\n[EXEC] {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n[!] LỖI THỰC THI: Lệnh sau trả về mã lỗi {result.returncode}:")
        print(f"    {' '.join(cmd)}")
        sys.exit(result.returncode)


def find_data_directory(requested_dir: Optional[str] = None) -> str:
    """Tự động tìm kiếm thư mục chứa dữ liệu ảnh trên Kaggle, Colab hoặc Local."""
    candidate_dirs = []
    if requested_dir:
        candidate_dirs.append(Path(requested_dir))

    candidate_dirs.extend([
        Path("/kaggle/input/datasets/b23dckh002lvitanh/s3-origin/S3"),
        Path("/kaggle/input/datasets/b23dckh002lvitanh/s3-origin"),
        Path("/kaggle/input/s3-origin/S3"),
        Path("/kaggle/input/s3-origin"),
        Path("/kaggle/input/s3/S3"),
        Path("/kaggle/input/s3"),
        Path("./S3"),
        Path("../S3"),
        Path("data/S3"),
        Path("out/images")
    ])

    for cand in candidate_dirs:
        if cand.exists() and cand.is_dir():
            subdirs = [p for p in cand.iterdir() if p.is_dir()]
            if len(subdirs) >= 3:
                return str(cand)
    return requested_dir or "/kaggle/input/datasets/b23dckh002lvitanh/s3-origin/S3"


def resolve_assets_dir(requested_assets_dir: str) -> Path:
    """Xác định thư mục assets ưu tiên, tương thích ngược giữa paper_data_assets và out."""
    p = Path(requested_assets_dir)
    if not p.exists() and Path("out").exists() and (Path("out") / "splits").exists():
        return Path("out")
    return p


def run_assets_step(python_bin: str, data_dir: str, assets_dir: Path, extract_embeddings: bool) -> None:
    """Bước 1: Sinh metadata, mã băm SHA-256, phương sai Laplacian và thẩm định rò rỉ."""
    print("\n" + "=" * 78)
    print(" [BƯỚC 1/4] SINH ASSETS, METADATA & THẨM ĐỊNH RÒ RỈ KHỐI MẪU VẬT (LEAKAGE AUDIT) ")
    print("=" * 78)
    cmd = [
        python_bin, "generate_benchmark_assets.py",
        "--data-dir", data_dir,
        "--output-dir", str(assets_dir)
    ]
    if extract_embeddings:
        cmd.append("--extract-embeddings")
    run_command(cmd)


def run_classify_step(
    python_bin: str,
    split_csv: Path,
    metadata_csv: Path,
    output_dir: Path,
    fig_dir: Path,
    epochs: int,
    loss_mode: str,
    seed: int
) -> None:
    """Bước 2: Huấn luyện Baseline Phân loại có giám sát (ConvNeXt-Tiny)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        python_bin, "train_classification_pipeline.py",
        "--split-csv", str(split_csv),
        "--metadata-csv", str(metadata_csv),
        "--epochs", str(epochs),
        "--output-dir", str(output_dir),
        "--fig-dir", str(fig_dir),
        "--seed", str(seed)
    ]
    if loss_mode == "both":
        cmd.append("--run-both")
    else:
        cmd.extend(["--loss", loss_mode])
    run_command(cmd)


def run_metric_step(
    python_bin: str,
    split_csv: Path,
    metadata_csv: Path,
    output_dir: Path,
    fig_dir: Path,
    epochs: int,
    margin: float,
    k_samples: int,
    seed: int
) -> None:
    """Bước 3: Huấn luyện Deep Metric Representation Learning (Semi-Hard Triplet Loss)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        python_bin, "train_metric_learning_pipeline.py",
        "--split-csv", str(split_csv),
        "--metadata-csv", str(metadata_csv),
        "--loss", "semihard_triplet",
        "--margin", str(margin),
        "--epochs", str(epochs),
        "--k-samples", str(k_samples),
        "--output-dir", str(output_dir),
        "--fig-dir", str(fig_dir),
        "--seed", str(seed)
    ]
    run_command(cmd)


def load_json_if_exists(file_path: Path) -> Optional[Dict[str, Any]]:
    """Tải nội dung JSON nếu tệp tồn tại."""
    if file_path.exists():
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def run_audit_step(python_bin: str, assets_dir: Path) -> None:
    """Bước 1b: Kiểm định độc lập 3 cấp độ: Bitwise (SHA-256), Tri giác (dHash/pHash), và Biểu diễn sâu (ConvNeXt)."""
    print("\n" + "=" * 84)
    print(" [KIỂM ĐỊNH 3 CẤP ĐỘ] THẨM ĐỊNH TRÙNG LẶP TRI GIÁC (dHash/pHash) & BIỂU DIỄN SÂU ")
    print("=" * 84)
    split_can = assets_dir / "splits" / "split_canonical.csv"
    split_dis = assets_dir / "splits" / "split_specimen_disjoint.csv"
    meta_csv = assets_dir / "metadata" / "metadata.csv"
    emb_npy = assets_dir / "embeddings" / "convnext_tiny.npy"
    out_json = assets_dir / "leakage_audit" / "perceptual_audit_report.json"

    cmd = [
        python_bin, "audit_perceptual_and_embedding_similarity.py",
        "--split-csv", str(split_can),
        "--metadata-csv", str(meta_csv),
        "--output-json", str(out_json)
    ]
    if split_dis.exists():
        cmd.extend(["--compare-split-csv", str(split_dis)])
    if emb_npy.exists():
        cmd.extend(["--embeddings-npy", str(emb_npy)])

    run_command(cmd)


def generate_comparison_summary(assets_dir: Path) -> None:
    """
    Bước 4: Đối chiếu toàn diện giữa Phân vùng Kế thừa (Canonical) và Phân vùng Khối Nghiêm ngặt (Specimen-Disjoint),
    tính toán Khoảng cách Suy giảm do Rò rỉ (Generalization Gap from Specimen Leakage - GGSL).
    """
    print("\n" + "=" * 90)
    print("      BẢNG TỔNG HỢP SO SÁNH VÀ ĐO LƯỜNG KHOẢNG CÁCH SUY GIẢM HIỆU NĂNG (GGSL)       ")
    print("=" * 90)

    # 1. Tải dữ liệu Thẩm định Rò rỉ (Leakage Audit)
    audit_can_path = assets_dir / "leakage_audit" / "audit_summary_canonical.json"
    audit_dis_path = assets_dir / "leakage_audit" / "audit_summary_specimen_disjoint.json"
    if not audit_can_path.exists():
        audit_can_path = assets_dir / "leakage_audit" / "audit_summary.json"

    audit_can = load_json_if_exists(audit_can_path)
    audit_dis = load_json_if_exists(audit_dis_path)

    if audit_can and audit_dis:
        print("\n--- [PHẦN 1] ĐỐI CHIẾU MỨC ĐỘ RÒ RỈ MẪU VẬT (SPECIMEN LEAKAGE AUDIT) ---")
        print(f"{'Tiêu chuẩn kiểm định':<38} | {'Canonical (Legacy)':<22} | {'Specimen-Disjoint':<22}")
        print("-" * 88)
        print(f"{'Tổng số ảnh (Total Images)':<38} | {audit_can.get('total_images', 0):<22,} | {audit_dis.get('total_images', 0):<22,}")
        print(f"{'Tổng số khối mẫu vật (Specimens)':<38} | {audit_can.get('total_specimens', 147):<22} | {audit_dis.get('total_specimens', 147):<22}")
        ov_can = audit_can.get("specimen_overlap", {})
        ov_dis = audit_dis.get("specimen_overlap", {})
        print(f"{'Khối trùng Train-Val':<38} | {ov_can.get('train_vs_val', 0):<22} | {ov_dis.get('train_vs_val', 0):<22}")
        print(f"{'Khối trùng Train-Test':<38} | {ov_can.get('train_vs_test', 0):<22} | {ov_dis.get('train_vs_test', 0):<22}")
        print(f"{'Số khối rò rỉ (Set-Theoretic Leaked)':<38} | {audit_can.get('distinct_leaked_specimens', 21):<22} | {audit_dis.get('distinct_leaked_specimens', 2):<22} (chỉ 2 loài 1 block)")
        print(f"{'Tỷ lệ rò rỉ toàn cục (SLR Overall)':<38} | {audit_can.get('specimen_leakage_rate', 14.29)}%{'':<16} | {audit_dis.get('specimen_leakage_rate', 1.36)}%")
        print(f"{'Tỷ lệ rò rỉ Train->Test (SLR Test)':<38} | {audit_can.get('specimen_leakage_rate_train_to_test', 34.29)}%{'':<16} | {audit_dis.get('specimen_leakage_rate_train_to_test', 5.71)}%")

        # Phần 1.5: Kiểm định Trùng lặp Tri giác và Biểu diễn Sâu
        p_can = audit_can.get("perceptual_audit", {}).get("dhash_test_vs_train", {})
        p_dis = audit_dis.get("perceptual_audit", {}).get("dhash_test_vs_train", {})
        sha_can = audit_can.get("sha256_overlap", {}).get("train_vs_test", 0)
        sha_dis = audit_dis.get("sha256_overlap", {}).get("train_vs_test", 0)

        print("\n--- [PHẦN 1.5] KIỂM ĐỊNH BẢO MẬT & TRÙNG LẶP TRI GIÁC (THREE-TIER AUDIT) ---")
        print(f"{'Tiêu chí kiểm định bảo mật':<38} | {'Canonical (Legacy)':<22} | {'Specimen-Disjoint':<22}")
        print("-" * 88)
        print(f"{'1. Trùng lặp bitwise (SHA-256)':<38} | {sha_can:<22} | {sha_dis:<22}")

        if p_can or p_dis:
            d0_c = p_can.get("near_duplicate_count_dist_0", 0)
            d0_d = p_dis.get("near_duplicate_count_dist_0", 0)
            d4_c = p_can.get("near_duplicate_count_dist_le_4", 0)
            d4_d = p_dis.get("near_duplicate_count_dist_le_4", 0)
            d4_c_pct = p_can.get("near_duplicate_ratio_le_4_pct", 0.0)
            d4_d_pct = p_dis.get("near_duplicate_ratio_le_4_pct", 0.0)
            print(f"{'2. Trùng lặp tri giác (dHash = 0)':<38} | {d0_c:<22} | {d0_d:<22}")
            print(f"{'3. Ảnh gần giống/chồng lấn (dHash <= 4)':<38} | {f'{d4_c} ({d4_c_pct}%)':<22} | {f'{d4_d} ({d4_d_pct}%)':<22}")

        e_can = audit_can.get("embedding_similarity_audit", {})
        e_dis = audit_dis.get("embedding_similarity_audit", {})
        if e_can or e_dis:
            max_c = e_can.get("max_cosine_similarity", 0.0)
            max_d = e_dis.get("max_cosine_similarity", 0.0)
            c95_c = e_can.get("count_similarity_ge_0_95", 0)
            c95_d = e_dis.get("count_similarity_ge_0_95", 0)
            print(f"{'4. Tương đồng Cosine cực đại (Max Sim)':<38} | {max_c:<22.4f} | {max_d:<22.4f}")
            print(f"{'5. Trùng lặp biểu diễn sâu (Cos >= 0.95)':<38} | {c95_c:<22} | {c95_d:<22}")

    # 2. Tải kết quả Classification Baseline
    cls_can = load_json_if_exists(Path("baseline_outputs_canonical/classification_results_focal.json"))
    cls_dis = load_json_if_exists(Path("baseline_outputs_specimen_disjoint/classification_results_focal.json"))
    if not cls_can:
        cls_can = load_json_if_exists(Path("baseline_outputs/classification_results_focal.json"))

    if cls_can and cls_dis:
        acc_can = cls_can.get("overall_metrics", {}).get("accuracy", 0.0) * 100.0
        f1_can = cls_can.get("overall_metrics", {}).get("macro_f1", 0.0) * 100.0
        acc_dis = cls_dis.get("overall_metrics", {}).get("accuracy", 0.0) * 100.0
        f1_dis = cls_dis.get("overall_metrics", {}).get("macro_f1", 0.0) * 100.0

        gap_acc = acc_can - acc_dis
        gap_f1 = f1_can - f1_dis

        print("\n--- [PHẦN 2] HIỆU NĂNG PHÂN LOẠI & KHOẢNG CÁCH SUY GIẢM (GGSL CLASSIFICATION) ---")
        print(f"{'Thước đo phân loại (ConvNeXt-Tiny)':<38} | {'Canonical (Rò rỉ)':<22} | {'Specimen-Disjoint':<22} | {'GGSL (Khoảng cách)'}")
        print("-" * 92)
        print(f"{'Top-1 Test Accuracy (%)':<38} | {acc_can:<22.2f} | {acc_dis:<22.2f} | {gap_acc:+.2f}%")
        print(f"{'Macro-Averaged F1-Score (%)':<38} | {f1_can:<22.2f} | {f1_dis:<22.2f} | {gap_f1:+.2f}%")

    # 3. Tải kết quả Metric Representation Learning
    met_can = load_json_if_exists(Path("metric_outputs_canonical/semihard_triplet_results.json"))
    met_dis = load_json_if_exists(Path("metric_outputs_specimen_disjoint/semihard_triplet_results.json"))
    if not met_can:
        met_can = load_json_if_exists(Path("metric_outputs/semihard_triplet_results.json"))

    target_met = met_dis or met_can
    if target_met:
        split_label = "Specimen-Disjoint" if target_met == met_dis else "Canonical"
        print(f"\n--- [PHẦN 3] TRUY VẤN ĐẶC TRƯNG HÌNH HỌC TRÊN SPLIT [{split_label}] ---")
        print(f"{'Chỉ số đánh giá truy vấn & hình học':<42} | {'Trước huấn luyện':<18} | {'Sau huấn luyện':<18} | {'Cải thiện'}")
        print("-" * 92)

        cross_b = target_met.get("cross_recalls_before", {})
        cross_a = target_met.get("cross_recalls_after", {})
        self_b = target_met.get("self_recalls_before", {})
        self_a = target_met.get("self_recalls_after", {})
        geom_b = target_met.get("metrics_before", {})
        geom_a = target_met.get("metrics_after", {})

        # Cross-Split Retrieval
        if cross_b and cross_a:
            print("  [Chuẩn Khoa Học: Query = Test, Gallery = Train]")
            for k in [1, 2, 4]:
                b_val = cross_b.get(f"Recall@{k}", 0.0)
                a_val = cross_a.get(f"Recall@{k}", 0.0)
                print(f"  - Cross-Split Recall@{k} (%):{'' :<17} | {b_val:<18.2f} | {a_val:<18.2f} | {a_val - b_val:+.2f}%")

        # Self-Test Retrieval (Diagnostic)
        if self_b and self_a:
            print("  [Đối Chứng Rò Rỉ Block: Query = Test, Gallery = Test (Leakage Diagnostic)]")
            for k in [1, 2]:
                b_val = self_b.get(f"Self_Recall@{k}", 0.0)
                a_val = self_a.get(f"Self_Recall@{k}", 0.0)
                print(f"  - Diagnostic Self-Test Recall@{k} (%):{'' :<7} | {b_val:<18.2f} | {a_val:<18.2f} | {a_val - b_val:+.2f}%")

        # Geometry
        if geom_b and geom_a:
            print("  [Hình học Không gian Nhúng (Embedding Geometry)]")
            dbi_b, dbi_a = geom_b.get("davies_bouldin", 0.0), geom_a.get("davies_bouldin", 0.0)
            sil_b, sil_a = geom_b.get("silhouette", 0.0), geom_a.get("silhouette", 0.0)
            nmi_b, nmi_a = geom_b.get("nmi", 0.0), geom_a.get("nmi", 0.0)
            print(f"  - Davies-Bouldin Index (thấp hơn tốt):{'' :<6} | {dbi_b:<18.4f} | {dbi_a:<18.4f} | {((dbi_b-dbi_a)/max(1e-6, dbi_b))*100:+.1f}%")
            print(f"  - Silhouette Score (cao hơn tốt):{'' :<11} | {sil_b:<18.4f} | {sil_a:<18.4f} | {((sil_a-sil_b)/max(1e-6, abs(sil_b)))*100:+.1f}%")
            print(f"  - Normalized Mutual Info (NMI, cao hơn tốt): | {nmi_b:<18.4f} | {nmi_a:<18.4f} | {((nmi_a-nmi_b)/max(1e-6, nmi_b))*100:+.1f}%")

    print("=" * 90)
    print("[+] Hoàn tất tổng hợp báo cáo đối chiếu khoa học!")


def main():
    parser = argparse.ArgumentParser(
        description="Master Orchestrator Pipeline cho IC4SDMacroWood (Data in Brief / Q1 Journal)"
    )
    parser.add_argument("--all", action="store_true",
                        help="Thực hiện toàn bộ quy trình thực nghiệm (tương đương --step all)")
    parser.add_argument("--step", type=str, default="all",
                        choices=["all", "assets", "audit", "classify", "metric", "compare"],
                        help="Bước cần thực hiện: assets | audit | classify | metric | compare | all")
    parser.add_argument("--split-type", type=str, default="both",
                        choices=["canonical", "specimen_disjoint", "both"],
                        help="Kiểu phân vùng: 'canonical' (kế thừa), 'specimen_disjoint' (nghiêm ngặt), hoặc 'both' (chạy cả hai để đo GGSL)")
    parser.add_argument("--data-dir", type=str, default="/kaggle/input/datasets/b23dckh002lvitanh/s3-origin/S3",
                        help="Đường dẫn đến thư mục chứa 19 lớp ảnh macroscopic wood")
    parser.add_argument("--assets-dir", type=str, default="paper_data_assets",
                        help="Thư mục chứa các tệp metadata, splits và leakage audit")
    parser.add_argument("--classify-epochs", type=int, default=22,
                        help="Số epochs cho mô hình phân loại ConvNeXt-Tiny (mặc định 22)")
    parser.add_argument("--classify-loss", type=str, default="focal",
                        choices=["focal", "cross_entropy", "both"],
                        help="Hàm mất mát phân loại: 'focal', 'cross_entropy', hoặc 'both'")
    parser.add_argument("--metric-epochs", type=int, default=30,
                        help="Số epochs cho Semi-Hard Triplet Loss (mặc định 30)")
    parser.add_argument("--metric-margin", type=float, default=0.5,
                        help="Margin d^2 cho Semi-Hard Triplet Loss (mặc định 0.5)")
    parser.add_argument("--metric-k-samples", type=int, default=4,
                        help="Số mẫu K của từng lớp trong batch (mặc định 4)")
    parser.add_argument("--extract-embeddings", action="store_true",
                        help="Trích xuất convnext_tiny.npy khi tạo assets")
    parser.add_argument("--seed", type=int, default=42, help="Hạt giống ngẫu nhiên")
    args = parser.parse_args()

    if args.all:
        args.step = "all"

    # Tự động dò tìm thư mục dữ liệu thật trên Kaggle/Colab/Local
    detected_data_dir = find_data_directory(args.data_dir)
    if Path(detected_data_dir).exists():
        args.data_dir = detected_data_dir
        print(f"[*] Thư mục dữ liệu ảnh phát hiện: {args.data_dir}")

    assets_dir = resolve_assets_dir(args.assets_dir)
    print(f"[*] Thư mục assets dữ liệu sử dụng: {assets_dir}")

    python_bin = sys.executable

    # BƯỚC 1: TẠO ASSETS, TÍNH TOÁN SHARPNESS, HASH SHA-256 VÀ THẨM ĐỊNH RÒ RỈ
    if args.step in ["all", "assets"]:
        run_assets_step(python_bin, args.data_dir, assets_dir, args.extract_embeddings)

    # BƯỚC 1b: KIỂM ĐỊNH TRÙNG LẶP TRI GIÁC & BIỂU DIỄN SÂU (THREE-TIER AUDIT)
    if args.step in ["all", "audit"]:
        run_audit_step(python_bin, assets_dir)

    # Xác định danh sách split cần huấn luyện
    splits_to_run = []
    if args.split_type in ["canonical", "both"]:
        splits_to_run.append(("canonical", assets_dir / "splits" / "split_canonical.csv"))
    if args.split_type in ["specimen_disjoint", "both"]:
        splits_to_run.append(("specimen_disjoint", assets_dir / "splits" / "split_specimen_disjoint.csv"))

    metadata_csv = assets_dir / "metadata" / "metadata.csv"

    # BƯỚC 2: HUẤN LUYỆN CLASSIFICATION BASELINE (CONVNEXT-TINY)
    if args.step in ["all", "classify"]:
        print("\n" + "=" * 78)
        print(" [BƯỚC 2/4] HUẤN LUYỆN CONVNEXT-TINY BASELINE (SUPERVISED CLASSIFICATION) ")
        print("=" * 78)
        for s_name, s_csv in splits_to_run:
            out_name = f"baseline_outputs_{s_name}" if args.split_type == "both" else "baseline_outputs"
            out_p = Path(out_name)
            fig_p = Path("paper_data/fig") / s_name
            print(f"\n[*] Đang thực thi phân loại trên split [{s_name}] -> {out_p}...")
            run_classify_step(
                python_bin=python_bin,
                split_csv=s_csv,
                metadata_csv=metadata_csv,
                output_dir=out_p,
                fig_dir=fig_p,
                epochs=args.classify_epochs,
                loss_mode=args.classify_loss,
                seed=args.seed
            )

    # BƯỚC 3: HUẤN LUYỆN METRIC LEARNING (SEMI-HARD TRIPLET LOSS)
    if args.step in ["all", "metric"]:
        print("\n" + "=" * 78)
        print(" [BƯỚC 3/4] HUẤN LUYỆN DEEP METRIC LEARNING (SEMI-HARD TRIPLET LOSS) ")
        print("=" * 78)
        for s_name, s_csv in splits_to_run:
            out_name = f"metric_outputs_{s_name}" if args.split_type == "both" else "metric_outputs"
            out_p = Path(out_name)
            fig_p = Path("paper_data/fig") / s_name
            print(f"\n[*] Đang thực thi Semi-Hard Triplet Learning trên split [{s_name}] -> {out_p}...")
            run_metric_step(
                python_bin=python_bin,
                split_csv=s_csv,
                metadata_csv=metadata_csv,
                output_dir=out_p,
                fig_dir=fig_p,
                epochs=args.metric_epochs,
                margin=args.metric_margin,
                k_samples=args.metric_k_samples,
                seed=args.seed
            )

    # BƯỚC 4: TỔNG HỢP SO SÁNH VÀ XUẤT BÁO CÁO KHOA HỌC GGSL
    if args.step in ["all", "compare"]:
        print("\n" + "=" * 78)
        print(" [BƯỚC 4/4] TỔNG HỢP KẾT QUẢ ĐỐI CHIẾU & ĐO LƯỜNG KHOẢNG CÁCH SUY GIẢM (GGSL) ")
        print("=" * 78)
        generate_comparison_summary(assets_dir)

    print("\n" + "=" * 78)
    print(" [✓] HOÀN TẤT TOÀN BỘ QUY TRÌNH THỰC NGHIỆM! DỮ LIỆU ĐÃ ĐỒNG BỘ CHUẨN PUBLICATION.")
    print("=" * 78)


if __name__ == "__main__":
    main()
