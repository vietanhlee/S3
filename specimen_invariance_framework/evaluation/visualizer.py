"""
specimen_invariance_framework/evaluation/visualizer.py
======================================================
Visualization utilities:
1. Dual t-SNE/UMAP plots: colored by Species (left) vs. colored by Specimen (right).
2. Pareto Frontier: Strict Validation Accuracy vs. Specimen Recoverability Index.
3. Correlation Analysis: Specimen Recoverability Index vs. Generalization Gap (GGSL).
"""

from typing import List, Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.stats import pearsonr


def plot_tsne_species_vs_specimen(
    embeddings: np.ndarray,
    species_labels: np.ndarray,
    specimen_labels: np.ndarray,
    save_path: str,
    title_prefix: str = "ConvNeXt-Tiny",
):
    """
    Generates side-by-side t-SNE projections:
    - Left: Colored by species (evaluating taxonomic separability).
    - Right: Colored by physical specimen (evaluating specimen shortcut mixing).
    """
    # Downsample if too large for responsive t-SNE
    max_samples = 1500
    if len(embeddings) > max_samples:
        indices = np.random.choice(len(embeddings), max_samples, replace=False)
        embs = embeddings[indices]
        sp_lbls = species_labels[indices]
        spec_lbls = specimen_labels[indices]
    else:
        embs = embeddings
        sp_lbls = species_labels
        spec_lbls = specimen_labels

    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    coords = tsne.fit_transform(embs)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Colored by species
    scatter_sp = axes[0].scatter(
        coords[:, 0], coords[:, 1],
        c=sp_lbls, cmap="tab20", s=20, alpha=0.8, edgecolors="none"
    )
    axes[0].set_title(f"{title_prefix}: Colored by Species (Target Semantics)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("t-SNE Dimension 1")
    axes[0].set_ylabel("t-SNE Dimension 2")
    axes[0].grid(True, linestyle=":", alpha=0.5)

    # 2. Colored by physical specimen
    scatter_spec = axes[1].scatter(
        coords[:, 0], coords[:, 1],
        c=spec_lbls, cmap="nipy_spectral", s=20, alpha=0.8, edgecolors="none"
    )
    axes[1].set_title(f"{title_prefix}: Colored by Specimen (Spurious Shortcuts)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("t-SNE Dimension 1")
    axes[1].set_ylabel("t-SNE Dimension 2")
    axes[1].grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_pareto_curve(
    sweep_results: List[Dict[str, float]],
    save_path: str,
):
    """
    Plots Pareto Frontier: Strict Val Accuracy vs. Specimen Recoverability Index (SRI).
    sweep_results: list of dicts with 'lambda_adv', 'val_acc', 'sri'.
    """
    sri_vals = [r["sri"] for r in sweep_results]
    acc_vals = [r["val_acc"] for r in sweep_results]
    lambdas = [r["lambda_adv"] for r in sweep_results]

    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(sri_vals, acc_vals, c=lambdas, cmap="viridis", s=100, zorder=3, edgecolors="black")

    for i, lam in enumerate(lambdas):
        ax.annotate(
            f"λ={lam:.2f}",
            (sri_vals[i], acc_vals[i]),
            textcoords="offset points",
            xytext=(7, 5),
            fontsize=9
        )

    # Sort for connecting line
    sorted_pairs = sorted(zip(sri_vals, acc_vals))
    ax.plot([p[0] for p in sorted_pairs], [p[1] for p in sorted_pairs], "--", color="gray", alpha=0.7, zorder=2)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Adversarial Weight ($\lambda_{adv}$)", fontsize=11)

    ax.set_xlabel("Specimen Recoverability Index (SRI) $\downarrow$", fontsize=11)
    ax.set_ylabel("Strict Val Accuracy (%) $\uparrow$", fontsize=11)
    ax.set_title("Pareto Trade-off: Accuracy vs. Specimen Invariance", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_correlation_sri_vs_ggsl(
    method_data: List[Dict[str, Any]],
    save_path: str,
):
    """
    Scatter plot and regression line between Specimen Recoverability Index (SRI)
    and Generalization Gap from Specimen Leakage (GGSL).
    """
    sri_vals = np.array([m["sri"] for m in method_data])
    ggsl_vals = np.array([m["ggsl_acc"] for m in method_data])
    names = [m["method_name"] for m in method_data]

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.scatter(sri_vals, ggsl_vals, color="#c51b7d", s=90, edgecolors="black", zorder=3)

    for i, name in enumerate(names):
        ax.annotate(
            name,
            (sri_vals[i], ggsl_vals[i]),
            textcoords="offset points",
            xytext=(6, 5),
            fontsize=9
        )

    # Fit linear regression line
    m, b = np.polyfit(sri_vals, ggsl_vals, 1)
    ax.plot(sri_vals, m * sri_vals + b, "-", color="#4d9221", label=f"Fit (Slope: {m:.2f})")

    r, p = pearsonr(sri_vals, ggsl_vals)
    ax.set_title(f"Correlation: SRI vs. GGSL (Pearson r = {r:.3f}, p = {p:.3e})", fontsize=12, fontweight="bold")
    ax.set_xlabel("Specimen Recoverability Index (SRI)", fontsize=11)
    ax.set_ylabel("Generalization Gap GGSL (Acc %)", fontsize=11)
    ax.legend(loc="upper left")
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
