"""
specimen_invariance_framework/evaluation/metrics.py
===================================================
Evaluation metrics:
- Generalization Gap from Specimen Leakage (GGSL)
- Paired round-robin fold summarizer
"""

from typing import Dict, List, Any
import numpy as np
import pandas as pd
from scipy import stats


def compute_ggsl(
    acc_leaky: float,
    acc_loso: float,
    f1_leaky: float,
    f1_loso: float,
) -> Dict[str, float]:
    """
    Computes the Generalization Gap from Specimen Leakage (GGSL).
    
    GGSL = Performance_leaky - Performance_LOSO
    Higher GGSL indicates greater vulnerability to specimen shortcut memorization.
    """
    return {
        "ggsl_acc": float(acc_leaky - acc_loso),
        "ggsl_macro_f1": float(f1_leaky - f1_loso),
    }


def summarize_paired_folds(
    baseline_results: List[Dict[str, float]],
    proposed_results: List[Dict[str, float]],
    metric_key: str = "accuracy",
) -> Dict[str, Any]:
    """
    Computes paired statistics across round-robin folds.
    """
    base_scores = np.array([r[metric_key] for r in baseline_results])
    prop_scores = np.array([r[metric_key] for r in proposed_results])
    
    diffs = prop_scores - base_scores
    t_stat, p_val = stats.ttest_rel(prop_scores, base_scores)
    
    return {
        "baseline_mean": float(np.mean(base_scores)),
        "baseline_std": float(np.std(base_scores)),
        "proposed_mean": float(np.mean(prop_scores)),
        "proposed_std": float(np.std(prop_scores)),
        "mean_gain": float(np.mean(diffs)),
        "p_value": float(p_val),
    }
