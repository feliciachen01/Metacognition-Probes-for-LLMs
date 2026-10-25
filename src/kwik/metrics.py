"""Metrics for a risk score r in [0, 1] predicting y = 1 (the answer is wrong)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score


def auroc(y: np.ndarray, r: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, r))


def ece(y: np.ndarray, r: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error with equal-width bins."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(r, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(y[m].mean() - r[m].mean())
    return float(total)


def brier(y: np.ndarray, r: np.ndarray) -> float:
    return float(brier_score_loss(y, np.clip(r, 0, 1)))


def risk_coverage(y: np.ndarray, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Answer only the least risky questions first. Returns coverage and error rate among answered."""
    order = np.argsort(r, kind="stable")
    errs = np.cumsum(y[order])
    n = np.arange(1, len(y) + 1)
    return n / len(y), errs / n


def aurc(y: np.ndarray, r: np.ndarray) -> float:
    """Area under the risk-coverage curve. Lower is better."""
    _, risk = risk_coverage(y, r)
    return float(risk.mean())


def accuracy_at_coverage(y: np.ndarray, r: np.ndarray, coverage: float = 0.8) -> float:
    cov, risk = risk_coverage(y, r)
    i = max(int(np.ceil(coverage * len(y))) - 1, 0)
    return float(1 - risk[i])


def _boot_indices(n: int, n_boot: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def bootstrap_auroc(y: np.ndarray, r: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% percentile interval for AUROC, resampling questions."""
    vals = [auroc(y[i], r[i]) for i in _boot_indices(len(y), n_boot, seed) if len(np.unique(y[i])) == 2]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired_auroc_diff(y: np.ndarray, r1: np.ndarray, r2: np.ndarray, n_boot: int = 2000,
                      seed: int = 0) -> tuple[float, float, float]:
    """AUROC(r1) - AUROC(r2) on the same questions, with a 95% interval from the same resamples.
    Paired, so it is tighter than comparing two separate intervals."""
    diffs = [auroc(y[i], r1[i]) - auroc(y[i], r2[i])
             for i in _boot_indices(len(y), n_boot, seed) if len(np.unique(y[i])) == 2]
    return auroc(y, r1) - auroc(y, r2), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def summarize(y: np.ndarray, r: np.ndarray) -> dict:
    return {"n": int(len(y)), "error_rate": float(y.mean()), "auroc": auroc(y, r), "ece": ece(y, r),
            "brier": brier(y, r), "aurc": aurc(y, r), "acc@80%": accuracy_at_coverage(y, r, 0.8)}
