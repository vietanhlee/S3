#!/usr/bin/env python3
"""
utils/statistical_testing.py
============================
Bộ công cụ phân tích và kiểm định thống kê đa hạt giống (Multi-Seed Statistical Analysis)
cho mô hình phân loại gỗ IC4SD-ForensicMacroWood-CITES (Elsevier Data in Brief).

Chức năng:
  1. Tính toán thống kê mô tả: Mean, Standard Deviation (ddof=1), Standard Error (SEM).
  2. Tính toán khoảng tin cậy 95% Confidence Interval (95% CI) theo phân phối Student's t.
  3. Tổng hợp kết quả từ 5 random seeds (Accuracy, Macro-F1, Weighted-F1, per-class metrics).
  4. Xuất báo cáo thống kê định dạng JSON và in bảng đối chiếu chuẩn publication.
"""

import math
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import numpy as np

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# Bảng tra giá trị tới hạn Student's t hai phía (Two-tailed critical t-value) ở mức ý nghĩa alpha = 0.05 (95% CI)
# t_table[df] = t_critical
STUDENT_T_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,  # df = 5 - 1 = 4 (cho 5 seeds)
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
}


def compute_sample_stats(values: List[float], confidence: float = 0.95) -> Dict[str, float]:
    """
    Tính toán các chỉ số thống kê mô tả và khoảng tin cậy cho một mẫu phân phối:
      - mean: Giá trị trung bình
      - std: Độ lệch chuẩn mẫu (ddof=1)
      - sem: Sai số chuẩn của giá trị trung bình (Standard Error of the Mean)
      - ci_lower, ci_upper: Khoảng tin cậy (Confidence Interval)
    """
    n = len(values)
    if n == 0:
        return {"mean": 0.0, "std": 0.0, "sem": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "min": 0.0, "max": 0.0}

    mean_val = float(sum(values) / n)
    if n == 1:
        return {
            "mean": mean_val, "std": 0.0, "sem": 0.0,
            "ci_lower": mean_val, "ci_upper": mean_val,
            "min": mean_val, "max": mean_val
        }

    variance = sum((x - mean_val) ** 2 for x in values) / (n - 1)
    std_val = math.sqrt(variance)
    sem_val = std_val / math.sqrt(n)

    df = n - 1
    if HAS_SCIPY:
        t_crit = float(stats.t.ppf((1 + confidence) / 2.0, df))
    else:
        t_crit = STUDENT_T_95.get(df, 2.776 if df == 4 else 1.96)

    margin = t_crit * sem_val
    return {
        "mean": mean_val,
        "std": std_val,
        "sem": sem_val,
        "t_crit": t_crit,
        "ci_margin": margin,
        "ci_lower": mean_val - margin,
        "ci_upper": mean_val + margin,
        "min": float(min(values)),
        "max": float(max(values)),
        "sample_size": n
    }


def aggregate_multi_seed_results(seed_payloads: List[Dict[str, Any]], confidence: float = 0.95) -> Dict[str, Any]:
    """
    Tổng hợp và kiểm định thống kê kết quả phân loại qua nhiều random seeds.
    Mỗi phần tử trong seed_payloads là một dict chứa 'summary_metrics' và 'per_class_metrics'.
    """
    n_seeds = len(seed_payloads)
    if n_seeds == 0:
        return {}

    seeds_list = [p.get("metadata", {}).get("seed", i) for i, p in enumerate(seed_payloads)]

    # 1. Thu thập các chỉ số Overall
    metric_keys = [
        "overall_accuracy",
        "macro_precision", "macro_recall", "macro_f1",
        "weighted_precision", "weighted_recall", "weighted_f1"
    ]

    raw_overall: Dict[str, List[float]] = {k: [] for k in metric_keys}
    for p in seed_payloads:
        sm = p.get("summary_metrics", {})
        for k in metric_keys:
            raw_overall[k].append(float(sm.get(k, 0.0)))

    overall_stats = {k: compute_sample_stats(raw_overall[k], confidence) for k in metric_keys}

    # 2. Thu thập các chỉ số Per-class (19 loài)
    class_names = sorted(list(seed_payloads[0].get("per_class_metrics", {}).keys()))
    per_class_stats: Dict[str, Dict[str, Any]] = {}

    for c_name in class_names:
        c_f1s = []
        c_precs = []
        c_recs = []
        support = 0

        for p in seed_payloads:
            pcm = p.get("per_class_metrics", {}).get(c_name, {})
            c_f1s.append(float(pcm.get("f1_score", 0.0)))
            c_precs.append(float(pcm.get("precision", 0.0)))
            c_recs.append(float(pcm.get("recall", 0.0)))
            support = int(pcm.get("support", support))

        per_class_stats[c_name] = {
            "f1_score": compute_sample_stats(c_f1s, confidence),
            "precision": compute_sample_stats(c_precs, confidence),
            "recall": compute_sample_stats(c_recs, confidence),
            "support": support
        }

    # 3. Tổng hợp ma trận nhầm lẫn (Confusion Matrix) qua các seeds
    cm_list = [p["confusion_matrix_raw"] for p in seed_payloads if "confusion_matrix_raw" in p]
    cm_stats = {}
    if len(cm_list) > 0:
        cm_arr = np.array(cm_list, dtype=float)
        cm_mean = np.mean(cm_arr, axis=0)
        cm_std = np.std(cm_arr, axis=0, ddof=1) if len(cm_list) > 1 else np.zeros_like(cm_mean)

        # Xác định seed đại diện (Representative Median Seed) có Macro-F1 tiệm cận giá trị trung bình nhất
        macro_f1_mean = overall_stats["macro_f1"]["mean"]
        diffs = [abs(p.get("summary_metrics", {}).get("macro_f1", 0.0) - macro_f1_mean) for p in seed_payloads]
        rep_idx = int(np.argmin(diffs))
        rep_seed = seeds_list[rep_idx]

        cm_stats = {
            "averaged_confusion_matrix": cm_mean.tolist(),
            "std_confusion_matrix": cm_std.tolist(),
            "representative_seed": rep_seed,
            "representative_seed_macro_f1": float(seed_payloads[rep_idx].get("summary_metrics", {}).get("macro_f1", 0.0)),
            "class_names": class_names
        }

    summary_dict = {
        "metadata": {
            "num_seeds": n_seeds,
            "seeds": seeds_list,
            "confidence_level": confidence,
            "degrees_of_freedom": n_seeds - 1
        },
        "overall_statistics": overall_stats,
        "per_class_statistics": per_class_stats,
        "confusion_matrix_statistics": cm_stats,
        "raw_per_seed": seed_payloads
    }

    # Sinh báo cáo phân loại định dạng Mean +- Std chuẩn sklearn
    summary_dict["formatted_classification_report"] = format_multi_seed_classification_report(summary_dict)
    return summary_dict


def plot_averaged_confusion_matrix(
    cm_mean: Any,
    class_names: List[str],
    save_path: Path,
    num_seeds: int = 3,
    model_name: Optional[str] = None
) -> None:
    """
    Vẽ ma trận nhầm lẫn trung bình qua nhiều hạt giống (Averaged Confusion Matrix) chuẩn Elsevier.
    Mỗi ô hiển thị số lượng mẫu trung bình (làm tròn số nguyên) và tỷ lệ Recall trung bình (%).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 10), dpi=300)

    cm = np.array(cm_mean, dtype=float)
    cm_norm = cm / np.maximum(cm.sum(axis=1)[:, np.newaxis], 1e-12)

    im = plt.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
    im.set_clim(0, 1.0)
    total_n = int(round(cm.sum()))
    
    display_title = (model_name or "Baseline").replace("_", " ").title()
    plt.title(f"{display_title} — Averaged Confusion Matrix ({num_seeds} Seeds, Test N={total_n:,})", fontsize=13, fontweight="bold", pad=15)
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label("Mean Normalized Ratio (Recall)", fontsize=10)

    tick_marks = np.arange(len(class_names))
    short_names = [c.replace("Dalbergia", "D.").replace("Pterocarpus", "P.").replace("Afzelia", "A.").replace("Guibourtia", "G.").replace("Sindora", "S.") for c in class_names]
    plt.xticks(tick_marks, short_names, rotation=45, ha="right", fontsize=9)
    plt.yticks(tick_marks, short_names, fontsize=9)

    thresh = cm_norm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            if val >= 0.5:
                cnt = int(round(val))
                pct = cm_norm[i, j] * 100.0
                plt.text(j, i, f"{cnt}\n({pct:.0f}%)",
                         horizontalalignment="center",
                         verticalalignment="center",
                         fontsize=6.5,
                         color="white" if cm_norm[i, j] > thresh else "black")

    plt.ylabel("Ground-Truth Species Nomenclature", fontsize=11, fontweight="bold")
    plt.xlabel("Predicted Taxonomic Epithet", fontsize=11, fontweight="bold")
    plt.tight_layout()

    plt.savefig(str(save_path.with_suffix(".pdf")), bbox_inches="tight")
    plt.savefig(str(save_path.with_suffix(".png")), bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[+] Đã lưu Averaged Confusion Matrix ({num_seeds} Seeds): {save_path.with_suffix('.pdf')}")


def format_multi_seed_classification_report(summary: Dict[str, Any]) -> str:
    """
    Sinh chuỗi Classification Report hoàn chỉnh với tất cả các chỉ số (Precision, Recall, F1-Score)
    ở định dạng Mean +- Std chuẩn publication.
    """
    meta = summary.get("metadata", {})
    overall = summary.get("overall_statistics", {})
    per_class = summary.get("per_class_statistics", {})
    n_seeds = meta.get("num_seeds", 0)
    seeds = meta.get("seeds", [])

    lines = []
    width = 104
    lines.append("=" * width)
    lines.append(f"             MULTI-SEED CLASSIFICATION REPORT (N={n_seeds} SEEDS: {seeds})")
    lines.append("=" * width)
    lines.append(f"{'Botanical Species':<32} {'Precision (Mean±Std)':<24} {'Recall (Mean±Std)':<22} {'F1-Score (Mean±Std)':<22} {'Support':>6}")
    lines.append("-" * width)

    total_support = 0
    for c_name, stats_dict in per_class.items():
        prec = stats_dict["precision"]
        rec = stats_dict["recall"]
        f1 = stats_dict["f1_score"]
        supp = stats_dict["support"]
        total_support += supp

        p_str = f"{prec['mean']:.4f} ± {prec['std']:.4f}"
        r_str = f"{rec['mean']:.4f} ± {rec['std']:.4f}"
        f_str = f"{f1['mean']:.4f} ± {f1['std']:.4f}"
        lines.append(f"{c_name:<32} {p_str:<24} {r_str:<22} {f_str:<22} {supp:>6}")

    lines.append("-" * width)
    # Hàng Accuracy
    acc = overall.get("overall_accuracy", {})
    acc_str = f"{acc['mean']:.4f} ± {acc['std']:.4f}"
    lines.append(f"{'accuracy':<32} {'':<24} {'':<22} {acc_str:<22} {total_support:>6}")

    # Hàng Macro Average
    m_p = overall.get("macro_precision", {})
    m_r = overall.get("macro_recall", {})
    m_f = overall.get("macro_f1", {})
    mp_str = f"{m_p['mean']:.4f} ± {m_p['std']:.4f}"
    mr_str = f"{m_r['mean']:.4f} ± {m_r['std']:.4f}"
    mf_str = f"{m_f['mean']:.4f} ± {m_f['std']:.4f}"
    lines.append(f"{'macro avg':<32} {mp_str:<24} {mr_str:<22} {mf_str:<22} {total_support:>6}")

    # Hàng Weighted Average
    w_p = overall.get("weighted_precision", {})
    w_r = overall.get("weighted_recall", {})
    w_f = overall.get("weighted_f1", {})
    wp_str = f"{w_p['mean']:.4f} ± {w_p['std']:.4f}"
    wr_str = f"{w_r['mean']:.4f} ± {w_r['std']:.4f}"
    wf_str = f"{w_f['mean']:.4f} ± {w_f['std']:.4f}"
    lines.append(f"{'weighted avg':<32} {wp_str:<24} {wr_str:<22} {wf_str:<22} {total_support:>6}")
    lines.append("=" * width)

    return "\n".join(lines)


def print_statistical_summary_table(summary: Dict[str, Any]) -> None:
    """
    In bảng tổng hợp kiểm định thống kê và classification report đa hạt giống ra terminal.
    """
    meta = summary.get("metadata", {})
    overall = summary.get("overall_statistics", {})
    n_seeds = meta.get("num_seeds", 0)
    seeds = meta.get("seeds", [])

    print("\n" + "=" * 90)
    print(f"      BÁO CÁO KIỂM ĐỊNH THỐNG KÊ BASELINE PHÂN LOẠI ({n_seeds} RANDOM SEEDS)       ")
    print(f"      Seeds: {seeds} | Bậc tự do (df) = {meta.get('degrees_of_freedom')} | Độ tin cậy = 95% (Student's t)")
    print("=" * 90)
    print(f"{'Chỉ số đánh giá (Metric)':<28} | {'Mean ± Std (%)':<20} | {'95% Confidence Interval':<22} | {'Min - Max (%)'}")
    print("-" * 90)

    acc = overall.get("overall_accuracy", {})
    f1_m = overall.get("macro_f1", {})
    f1_w = overall.get("weighted_f1", {})
    prec_m = overall.get("macro_precision", {})
    rec_m = overall.get("macro_recall", {})

    print(f"{'Overall Top-1 Accuracy':<28} | {acc['mean']*100:.2f} ± {acc['std']*100:.2f}%{'':<4} | [{acc['ci_lower']*100:.2f}%, {acc['ci_upper']*100:.2f}%]{'':<6} | {acc['min']*100:.2f}% - {acc['max']*100:.2f}%")
    print(f"{'Macro-Averaged F1-Score':<28} | {f1_m['mean']*100:.2f} ± {f1_m['std']*100:.2f}%{'':<4} | [{f1_m['ci_lower']*100:.2f}%, {f1_m['ci_upper']*100:.2f}%]{'':<6} | {f1_m['min']*100:.2f}% - {f1_m['max']*100:.2f}%")
    print(f"{'Weighted-Averaged F1-Score':<28} | {f1_w['mean']*100:.2f} ± {f1_w['std']*100:.2f}%{'':<4} | [{f1_w['ci_lower']*100:.2f}%, {f1_w['ci_upper']*100:.2f}%]{'':<6} | {f1_w['min']*100:.2f}% - {f1_w['max']*100:.2f}%")
    print(f"{'Macro-Averaged Precision':<28} | {prec_m['mean']*100:.2f} ± {prec_m['std']*100:.2f}%{'':<4} | [{prec_m['ci_lower']*100:.2f}%, {prec_m['ci_upper']*100:.2f}%]{'':<6} | {prec_m['min']*100:.2f}% - {prec_m['max']*100:.2f}%")
    print(f"{'Macro-Averaged Recall':<28} | {rec_m['mean']*100:.2f} ± {rec_m['std']*100:.2f}%{'':<4} | [{rec_m['ci_lower']*100:.2f}%, {rec_m['ci_upper']*100:.2f}%]{'':<6} | {rec_m['min']*100:.2f}% - {rec_m['max']*100:.2f}%")
    print("=" * 90 + "\n")

    # In Classification Report dạng Mean +- Std đầy đủ
    report_str = summary.get("formatted_classification_report") or format_multi_seed_classification_report(summary)
    print(report_str + "\n")
