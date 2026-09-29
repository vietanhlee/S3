#!/usr/bin/env python3
"""
run_pipeline.py
===============
Script điều phối (Master Orchestrator) tinh gọn cho ForensicMacroWood-CITES
(Elsevier Data in Brief), hỗ trợ kiểm định thống kê đa hạt giống (5 Random Seeds).

Quy trình tinh gọn (2 bước chính):
  1. 'assets': Sinh metadata, tính độ nét Laplacian (sigma^2 >= 100), mã băm
     mật mã SHA-256 và mã băm tri giác dHash/pHash (khử trùng lặp 2 cấp độ).
  2. 'classify': Huấn luyện baseline phân loại ConvNeXt-Tiny qua 5 random seeds
     (42, 123, 456, 789, 2024) và tự động tính toán kiểm định thống kê (Mean +- Std, 95% CI).
  - 'all': Tự động chạy tuần tự Bước 1 -> Bước 2.

Cách dùng cơ bản:
  # Chạy toàn bộ quy trình với kiểm định thống kê 5 seeds (chuẩn bài báo):
  python run_pipeline.py --all --data-dir <đường_dẫn_ảnh>

  # Chỉ chạy thử nghiệm nhanh với 1 seed:
  python run_pipeline.py --step classify --single-seed

  # Chỉ tạo assets & metadata:
  python run_pipeline.py --step assets --data-dir <đường_dẫn_ảnh>
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
import shutil
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List

from utils.statistical_testing import (
    aggregate_multi_seed_results,
    print_statistical_summary_table,
    plot_averaged_confusion_matrix
)


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


def run_assets_step(python_bin: str, data_dir: str, assets_dir: Path) -> None:
    """Bước 1: Sinh metadata, mã băm SHA-256, phương sai Laplacian và thẩm định rò rỉ 2 cấp độ."""
    print("\n" + "=" * 78)
    print(" [BƯỚC 1/2] SINH ASSETS, METADATA & KIỂM ĐỊNH RÒ RỈ MẪU VẬT 2 CẤP ĐỘ ")
    print("=" * 78)
    cmd = [
        python_bin, "generate_benchmark_assets.py",
        "--data-dir", data_dir,
        "--output-dir", str(assets_dir)
    ]
    run_command(cmd)


def run_audit_step(python_bin: str, assets_dir: Path) -> None:
    """Bước 1b (Tùy chọn): Kiểm định độc lập 2 cấp độ: Bitwise (SHA-256) & Tri giác (dHash/pHash)."""
    print("\n" + "=" * 80)
    print(" [KIỂM ĐỊNH 2 CẤP ĐỘ] THẨM ĐỊNH TRÙNG LẶP BITWISE (SHA-256) & TRI GIÁC (dHash/pHash) ")
    print("=" * 80)
    split_can = assets_dir / "splits" / "split_canonical.csv"
    split_dis = assets_dir / "splits" / "split_specimen_disjoint.csv"
    meta_csv = assets_dir / "metadata" / "metadata.csv"
    out_json = assets_dir / "leakage_audit" / "perceptual_audit_report.json"

    cmd = [
        python_bin, "audit_perceptual_and_embedding_similarity.py",
        "--split-csv", str(split_can),
        "--metadata-csv", str(meta_csv),
        "--output-json", str(out_json)
    ]
    if split_dis.exists():
        cmd.extend(["--compare-split-csv", str(split_dis)])

    run_command(cmd)


def run_classify_multi_seed(
    python_bin: str,
    split_csv: Path,
    metadata_csv: Path,
    output_dir: Path,
    fig_dir: Path,
    epochs: int,
    loss_mode: str,
    seeds: List[int],
    data_dir: Optional[str] = None,
    batch_size: Optional[int] = None,
    lr: Optional[float] = None
) -> None:
    """Bước 2: Huấn luyện Baseline Phân loại ConvNeXt-Tiny qua nhiều random seeds và kiểm định thống kê."""
    print("\n" + "=" * 82)
    print(f" [BƯỚC 2/2] HUẤN LUYỆN BASELINE CONVNEXT-TINY QUA {len(seeds)} RANDOM SEEDS & KIỂM ĐỊNH THỐNG KÊ ")
    print(f" Seeds: {seeds} | Loss: {loss_mode.upper()} | Epochs: {epochs}")
    print("=" * 82)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    seed_payloads = []
    loss_tag = "focal" if loss_mode in ["focal", "both"] else "ce"

    for idx, seed in enumerate(seeds, 1):
        print(f"\n>>> [SEED {idx}/{len(seeds)}: SEED = {seed}] <<<")
        seed_out_dir = output_dir / f"seed_{seed}"
        seed_out_dir.mkdir(parents=True, exist_ok=True)
        seed_fig_dir = fig_dir / f"seed_{seed}"
        seed_fig_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            python_bin, "train_classification_pipeline.py",
            "--split-csv", str(split_csv),
            "--metadata-csv", str(metadata_csv),
            "--epochs", str(epochs),
            "--output-dir", str(seed_out_dir),
            "--fig-dir", str(seed_fig_dir),
            "--seed", str(seed)
        ]
        if data_dir:
            cmd.extend(["--data-dir", data_dir])
        if batch_size:
            cmd.extend(["--batch-size", str(batch_size)])
        if lr:
            cmd.extend(["--lr", str(lr)])
        if loss_mode == "both":
            cmd.append("--run-both")
        else:
            cmd.extend(["--loss", loss_mode])

        run_command(cmd)

        # Thu thập file kết quả JSON của seed này
        res_file = seed_out_dir / f"classification_results_{loss_tag}.json"
        if res_file.exists():
            try:
                with open(res_file, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                    seed_payloads.append(payload)
            except Exception as e:
                print(f"[!] Lỗi khi nạp file kết quả seed {seed}: {e}")

    # Đồng bộ 1 bản checkpoint và confusion matrix của seed đầu tiên ra thư mục cha (tương thích ngược)
    first_seed_dir = output_dir / f"seed_{seeds[0]}"
    if first_seed_dir.exists():
        for f in first_seed_dir.glob("*.json"):
            shutil.copy(f, output_dir / f.name)
        for f in first_seed_dir.glob("*.pth"):
            shutil.copy(f, output_dir / f.name)
        first_seed_fig = fig_dir / f"seed_{seeds[0]}"
        if first_seed_fig.exists():
            for f in first_seed_fig.glob("*.*"):
                shutil.copy(f, fig_dir / f.name)

    # Tổng hợp phân tích thống kê nếu có >= 2 seeds
    if len(seed_payloads) >= 2:
        print("\n[*] Đang tổng hợp phân tích thống kê (Mean, Std, 95% Student's t CI)...")
        summary_stats = aggregate_multi_seed_results(seed_payloads, confidence=0.95)
        stat_json_path = output_dir / "multi_seed_statistical_summary.json"
        with open(stat_json_path, "w", encoding="utf-8") as f:
            json.dump(summary_stats, f, indent=2)
        print(f"[+] Đã lưu báo cáo thống kê đa hạt giống: {stat_json_path}")

        # Lưu file text Classification Report định dạng Mean +- Std
        report_txt = summary_stats.get("formatted_classification_report", "")
        if report_txt:
            stat_txt_path = output_dir / "multi_seed_classification_report.txt"
            with open(stat_txt_path, "w", encoding="utf-8") as f:
                f.write(report_txt)
            print(f"[+] Đã lưu bảng Classification Report (Mean ± Std): {stat_txt_path}")

        print_statistical_summary_table(summary_stats)

        # Tự động vẽ và lưu Averaged Confusion Matrix & Representative Seed Confusion Matrix
        cm_stats = summary_stats.get("confusion_matrix_statistics", {})
        if "averaged_confusion_matrix" in cm_stats and "class_names" in cm_stats:
            avg_cm = cm_stats["averaged_confusion_matrix"]
            c_names = cm_stats["class_names"]
            rep_seed = cm_stats.get("representative_seed", seeds[0])

            # 1. Vẽ ma trận trung bình chuẩn hóa qua các seeds
            plot_averaged_confusion_matrix(avg_cm, c_names, fig_dir / "confusion_matrix_focal_test_averaged", num_seeds=len(seed_payloads))
            plot_averaged_confusion_matrix(avg_cm, c_names, fig_dir / "confusion_matrix_focal_test", num_seeds=len(seed_payloads))

            # Đồng bộ sang paper_data/fig nếu có
            paper_fig_dir = Path("paper_data/fig")
            if paper_fig_dir.exists():
                plot_averaged_confusion_matrix(avg_cm, c_names, paper_fig_dir / "confusion_matrix_focal_test", num_seeds=len(seed_payloads))

            # 2. Đồng bộ ma trận của Representative Seed để người dùng có cả 2 lựa chọn
            rep_fig_dir = fig_dir / f"seed_{rep_seed}"
            if rep_fig_dir.exists():
                for ext in [".pdf", ".png"]:
                    rep_src = rep_fig_dir / f"confusion_matrix_focal_test{ext}"
                    if rep_src.exists():
                        shutil.copy(rep_src, fig_dir / f"confusion_matrix_focal_test_representative{ext}")
                        if paper_fig_dir.exists():
                            shutil.copy(rep_src, paper_fig_dir / f"confusion_matrix_focal_test_representative{ext}")
            print(f"[+] Đã tạo thành công: (1) Averaged Confusion Matrix và (2) Representative Seed ({rep_seed}) Matrix.")
    elif len(seed_payloads) == 1:
        print("\n[+] Đã hoàn thành huấn luyện với 1 seed duy nhất.")


def ensure_disjoint_split_exists(assets_dir: Path) -> Path:
    """Tự động kiểm tra và sinh tệp split_specimen_disjoint.csv nếu chưa tồn tại."""
    disjoint_path = assets_dir / "splits" / "split_specimen_disjoint.csv"
    if not disjoint_path.exists():
        print(f"[*] Chưa phát hiện '{disjoint_path}'. Đang tự động khởi tạo từ metadata.csv...")
        try:
            from create_specimen_disjoint_split import generate_disjoint_split
            meta_csv = assets_dir / "metadata" / "metadata.csv"
            if not meta_csv.exists() and (Path("out") / "metadata" / "metadata.csv").exists():
                meta_csv = Path("out") / "metadata" / "metadata.csv"
            generate_disjoint_split(meta_csv, disjoint_path, seed=42)
            print(f"[+] Đã tự động sinh thành công: {disjoint_path}")
        except Exception as e:
            print(f"[!] Lỗi khi tự động sinh split_specimen_disjoint.csv: {e}")
    return disjoint_path


def run_split_comparison_summary(
    canonical_dir: Path,
    disjoint_dir: Path,
    output_dir: Path,
    loss_tag: str = "focal"
) -> Dict[str, Any]:
    """Tổng hợp đối chiếu kết quả giữa Canonical Split và Specimen-Disjoint Split."""
    print("\n" + "=" * 90)
    print("      BẢNG ĐỐI CHIẾU ĐÁNH GIÁ: CANONICAL SPLIT vs. STRICT SPECIMEN-DISJOINT SPLIT       ")
    print("=" * 90)

    def load_metrics(d: Path) -> Dict[str, float]:
        # Ưu tiên multi_seed summary nếu có, nếu không lấy kết quả của seed đầu
        multi_p = d / "multi_seed_statistical_summary.json"
        if multi_p.exists():
            try:
                with open(multi_p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    agg = data.get("aggregate_metrics", {})
                    return {
                        "accuracy": agg.get("accuracy", {}).get("mean", 0.0),
                        "macro_precision": agg.get("macro_precision", {}).get("mean", 0.0),
                        "macro_recall": agg.get("macro_recall", {}).get("mean", 0.0),
                        "macro_f1": agg.get("macro_f1", {}).get("mean", 0.0),
                        "weighted_f1": agg.get("weighted_f1", {}).get("mean", 0.0),
                    }
            except Exception:
                pass

        single_p = d / f"classification_results_{loss_tag}.json"
        if not single_p.exists():
            for sub in d.glob("seed_*"):
                cand = sub / f"classification_results_{loss_tag}.json"
                if cand.exists():
                    single_p = cand
                    break

        if single_p.exists():
            try:
                with open(single_p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return {
                        "accuracy": data.get("accuracy", 0.0),
                        "macro_precision": data.get("macro_precision", 0.0),
                        "macro_recall": data.get("macro_recall", 0.0),
                        "macro_f1": data.get("macro_f1", 0.0),
                        "weighted_f1": data.get("weighted_f1", 0.0),
                    }
            except Exception:
                pass
        return {"accuracy": 0.0, "macro_precision": 0.0, "macro_recall": 0.0, "macro_f1": 0.0, "weighted_f1": 0.0}

    m_can = load_metrics(canonical_dir)
    m_dis = load_metrics(disjoint_dir)

    comparison = {
        "canonical_split": m_can,
        "specimen_disjoint_split": m_dis,
        "gap_delta": {
            k: round(m_dis.get(k, 0.0) - m_can.get(k, 0.0), 4) for k in m_can
        }
    }

    print(f"{'Chỉ số đánh giá (Evaluation Metric)':<36} | {'Canonical Split':<18} | {'Specimen-Disjoint':<18} | {'Gap (Delta)':<12}")
    print("-" * 90)
    for k, label in [
        ("accuracy", "Overall Top-1 Accuracy"),
        ("macro_precision", "Macro Precision"),
        ("macro_recall", "Macro Recall"),
        ("macro_f1", "Macro F1-Score"),
        ("weighted_f1", "Weighted F1-Score")
    ]:
        v_can = m_can.get(k, 0.0) * 100
        v_dis = m_dis.get(k, 0.0) * 100
        delta = (v_dis - v_can)
        delta_str = f"{delta:+.2f}%"
        print(f"{label:<36} | {v_can:>6.2f}%{'':<11} | {v_dis:>6.2f}%{'':<11} | {delta_str:>8}")
    print("=" * 90)

    # Lưu JSON & Markdown
    out_json = output_dir / "split_comparison_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)
    print(f"[+] Đã lưu bản ghi so sánh: {out_json}")

    out_md = output_dir / "split_comparison_summary.md"
    md_content = f"""# Báo cáo Đối chiếu Phân vùng: Canonical vs. Specimen-Disjoint

| Chỉ số đánh giá | Canonical Governed Split | Strict Specimen-Disjoint Split | Generalization Gap (Delta) |
| :--- | :---: | :---: | :---: |
| **Overall Top-1 Accuracy** | **{m_can.get('accuracy',0)*100:.2f}%** | **{m_dis.get('accuracy',0)*100:.2f}%** | **{(m_dis.get('accuracy',0)-m_can.get('accuracy',0))*100:+.2f}%** |
| **Macro Precision** | {m_can.get('macro_precision',0)*100:.2f}% | {m_dis.get('macro_precision',0)*100:.2f}% | {(m_dis.get('macro_precision',0)-m_can.get('macro_precision',0))*100:+.2f}% |
| **Macro Recall** | {m_can.get('macro_recall',0)*100:.2f}% | {m_dis.get('macro_recall',0)*100:.2f}% | {(m_dis.get('macro_recall',0)-m_can.get('macro_recall',0))*100:+.2f}% |
| **Macro F1-Score** | **{m_can.get('macro_f1',0)*100:.2f}%** | **{m_dis.get('macro_f1',0)*100:.2f}%** | **{(m_dis.get('macro_f1',0)-m_can.get('macro_f1',0))*100:+.2f}%** |
| **Weighted F1-Score** | {m_can.get('weighted_f1',0)*100:.2f}% | {m_dis.get('weighted_f1',0)*100:.2f}% | {(m_dis.get('weighted_f1',0)-m_can.get('weighted_f1',0))*100:+.2f}% |
"""
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[+] Đã lưu báo cáo Markdown: {out_md}")

    return comparison


def main():
    parser = argparse.ArgumentParser(
        description="Master Orchestrator Pipeline tinh gọn cho ForensicMacroWood-CITES (Elsevier Data in Brief)"
    )
    parser.add_argument("--all", action="store_true",
                        help="Thực hiện toàn bộ quy trình thực nghiệm (Assets -> Multi-seed Classify)")
    parser.add_argument("--step", type=str, default="all",
                        choices=["all", "assets", "audit", "classify"],
                        help="Bước cần thực hiện: assets | audit | classify | all")
    parser.add_argument("--split-type", type=str, default="canonical",
                        choices=["canonical", "specimen_disjoint"],
                        help="Kiểu phân vùng: 'canonical' (mặc định) hoặc 'specimen_disjoint'")
    parser.add_argument("--compare-splits", action="store_true",
                        help="Kích hoạt cờ này để chạy và đối chiếu cả 2 split: Canonical vs. Specimen-Disjoint")
    parser.add_argument("--data-dir", type=str, default="/kaggle/input/datasets/b23dckh002lvitanh/s3-origin/S3",
                        help="Đường dẫn đến thư mục chứa 19 lớp ảnh macroscopic wood")
    parser.add_argument("--assets-dir", type=str, default="paper_data_assets",
                        help="Thư mục chứa các tệp metadata, splits và leakage audit")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Kích thước batch size cho huấn luyện phân loại (mặc định 64)")
    parser.add_argument("--classify-epochs", type=int, default=22,
                        help="Số epochs cho mô hình phân loại ConvNeXt-Tiny (mặc định 22)")
    parser.add_argument("--classify-loss", type=str, default="focal",
                        choices=["focal", "cross_entropy", "both"],
                        help="Hàm mất mát phân loại: 'focal', 'cross_entropy', hoặc 'both'")
    parser.add_argument("--classify-lr", type=float, default=5e-4,
                        help="Tốc độ học cho phân loại (mặc định 5e-4)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 2024],
                        help="Danh sách các random seeds để chạy kiểm định thống kê (mặc định 5 seeds)")
    parser.add_argument("--single-seed", action="store_true",
                        help="Chỉ chạy duy nhất seed đầu tiên nếu muốn thực nghiệm nhanh")
    args = parser.parse_args()

    if args.all:
        args.step = "all"

    active_seeds = [args.seeds[0]] if args.single_seed else args.seeds

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
        run_assets_step(python_bin, args.data_dir, assets_dir)

    # BƯỚC 1b: KIỂM ĐỊNH ĐỘC LẬP 2 CẤP ĐỘ (SHA-256 & dHash/pHash)
    if args.step in ["audit"]:
        run_audit_step(python_bin, assets_dir)

    # BƯỚC 2: HUẤN LUYỆN CLASSIFICATION BASELINE (CONVNEXT-TINY)
    if args.step in ["all", "classify"]:
        metadata_csv = assets_dir / "metadata" / "metadata.csv"
        ensure_disjoint_split_exists(assets_dir)

        if args.compare_splits:
            print("\n" + "=" * 80)
            print(" [CHẾ ĐỘ ĐỐI CHIẾU PHÂN VÙNG] HUẤN LUYỆN CANONICAL & SPECIMEN-DISJOINT ")
            print("=" * 80)
            # 1. Chạy trên Canonical Split
            can_split_csv = assets_dir / "splits" / "split_canonical.csv"
            can_out = Path("baseline_outputs") / "canonical"
            can_fig = Path("paper_data/fig") / "canonical"
            print("\n>>> (A) HUẤN LUYỆN TRÊN CANONICAL SPLIT <<<")
            run_classify_multi_seed(
                python_bin=python_bin,
                split_csv=can_split_csv,
                metadata_csv=metadata_csv,
                output_dir=can_out,
                fig_dir=can_fig,
                epochs=args.classify_epochs,
                loss_mode=args.classify_loss,
                seeds=active_seeds,
                data_dir=args.data_dir,
                batch_size=args.batch_size,
                lr=args.classify_lr
            )

            # 2. Chạy trên Specimen-Disjoint Split (chạy đầy đủ các seeds như Canonical)
            dis_split_csv = assets_dir / "splits" / "split_specimen_disjoint.csv"
            dis_out = Path("baseline_outputs") / "specimen_disjoint"
            dis_fig = Path("paper_data/fig") / "specimen_disjoint"
            dis_seeds = active_seeds
            print(f"\n>>> (B) HUẤN LUYỆN TRÊN SPECIMEN-DISJOINT SPLIT (Seeds: {dis_seeds}) <<<")
            run_classify_multi_seed(
                python_bin=python_bin,
                split_csv=dis_split_csv,
                metadata_csv=metadata_csv,
                output_dir=dis_out,
                fig_dir=dis_fig,
                epochs=args.classify_epochs,
                loss_mode=args.classify_loss,
                seeds=dis_seeds,
                data_dir=args.data_dir,
                batch_size=args.batch_size,
                lr=args.classify_lr
            )

            # 3. Tổng hợp bảng so sánh đối chiếu
            run_split_comparison_summary(can_out, dis_out, Path("baseline_outputs"), loss_tag=args.classify_loss)

        else:
            split_file = "split_canonical.csv" if args.split_type == "canonical" else "split_specimen_disjoint.csv"
            split_csv = assets_dir / "splits" / split_file
            out_p = Path("baseline_outputs") / args.split_type
            fig_p = Path("paper_data/fig") / args.split_type

            print(f"\n[*] Đang thực thi phân loại trên split [{args.split_type}] với {len(active_seeds)} seeds -> {out_p}...")
            run_classify_multi_seed(
                python_bin=python_bin,
                split_csv=split_csv,
                metadata_csv=metadata_csv,
                output_dir=out_p,
                fig_dir=fig_p,
                epochs=args.classify_epochs,
                loss_mode=args.classify_loss,
                seeds=active_seeds,
                data_dir=args.data_dir,
                batch_size=args.batch_size,
                lr=args.classify_lr
            )

    print("\n" + "=" * 78)
    print(" [✓] HOÀN TẤT QUY TRÌNH THỰC NGHIỆM! DỮ LIỆU ĐÃ ĐỒNG BỘ CHUẨN PUBLICATION.")
    print("=" * 78)


if __name__ == "__main__":
    main()

