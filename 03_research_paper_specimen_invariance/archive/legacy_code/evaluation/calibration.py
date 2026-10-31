"""
specimen_invariance_framework/evaluation/calibration.py
=======================================================
Model calibration metrics:
- Expected Calibration Error (ECE)
- Maximum Calibration Error (MCE)
- Reliability diagram generator
"""

from typing import Dict, Tuple, Any
import numpy as np
import matplotlib.pyplot as plt


def compute_calibration_metrics(
    confidences: np.ndarray,
    predictions: np.ndarray,
    targets: np.ndarray,
    num_bins: int = 15,
) -> Dict[str, Any]:
    """
    Computes ECE and MCE across confidence bins.
    
    Args:
        confidences: (N,) maximum predicted softmax probabilities
        predictions: (N,) predicted class indices
        targets: (N,) ground-truth class indices
        num_bins: number of equal-width confidence bins in [0, 1]
    """
    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    accuracies = (predictions == targets)

    ece = 0.0
    mce = 0.0

    bin_accs = []
    bin_confs = []
    bin_counts = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        count = int(np.sum(in_bin))

        if count > 0:
            accuracy_in_bin = float(np.mean(accuracies[in_bin]))
            avg_confidence_in_bin = float(np.mean(confidences[in_bin]))
            gap = abs(avg_confidence_in_bin - accuracy_in_bin)

            ece += gap * prop_in_bin
            mce = max(mce, gap)

            bin_accs.append(accuracy_in_bin)
            bin_confs.append(avg_confidence_in_bin)
            bin_counts.append(count)
        else:
            bin_accs.append(0.0)
            bin_confs.append((bin_lower + bin_upper) / 2.0)
            bin_counts.append(0)

    return {
        "ece": float(ece * 100.0),  # In percentage
        "mce": float(mce * 100.0),
        "bin_accs": bin_accs,
        "bin_confs": bin_confs,
        "bin_counts": bin_counts,
        "bin_edges": bin_boundaries.tolist(),
    }


def plot_reliability_diagram(
    calib_data: Dict[str, Any],
    save_path: str,
    title: str = "Reliability Diagram",
):
    """Plots and saves the reliability diagram."""
    bin_edges = np.array(calib_data["bin_edges"])
    bin_accs = np.array(calib_data["bin_accs"])
    bin_confs = np.array(calib_data["bin_confs"])
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    fig, ax = plt.subplots(figsize=(6, 6))

    # Perfect calibration diagonal
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect Calibration")

    # Accuracy bars
    ax.bar(
        bin_centers,
        bin_accs,
        width=1.0 / len(bin_centers),
        alpha=0.6,
        color="#2b5c8f",
        edgecolor="black",
        label="Observed Accuracy",
    )

    # Gap bars
    gap = np.abs(bin_confs - bin_accs)
    ax.bar(
        bin_centers,
        gap,
        bottom=np.minimum(bin_accs, bin_confs),
        width=1.0 / len(bin_centers),
        alpha=0.4,
        color="#d95f02",
        edgecolor="black",
        hatch="//",
        label=f"Calibration Gap (ECE = {calib_data['ece']:.2f}%)",
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
