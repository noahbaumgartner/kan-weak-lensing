"""Semantic-error diagnostics for the weak-lensing (Omega_m, S8) regression.

Complements the scalar metrics (MSE, R^2, coverage) with plots that a
number alone can hide: whether the predicted distribution matches the
test set, and whether predictions track the 1:1 line.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset


@torch.no_grad()
def collect_predictions(model, val_input, val_label, device, batch_size=512):
    """Run ``val_input`` through ``model`` and collect (y_true, y_pred).

    Both are returned in the same (standardized or not) space the labels
    were already in — callers unstandardize afterwards if needed. Only the
    mean head is kept for models that also predict an uncertainty (e.g.
    the score objective's (mu, log_sigma) output).
    """
    if isinstance(val_input, Dataset):
        val_ds = val_input
    else:
        val_ds = TensorDataset(val_input, torch.as_tensor(val_label, dtype=torch.float32))

    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    model.get_model().eval()

    trues, preds = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        mu = model.predict(x)[:, : y.shape[1]]
        trues.append(y.numpy())
        preds.append(mu.cpu().numpy())

    return np.concatenate(trues, axis=0), np.concatenate(preds, axis=0)


def compute_bias_metrics(y_true: np.ndarray, y_pred: np.ndarray, target_names) -> dict:
    """Mean signed residual per target — the direct under-/overestimation check.

    bias > 0 -> im Mittel Überschätzung, bias < 0 -> Unterschätzung.
    """
    metrics = {}
    for i, name in enumerate(target_names):
        resid = y_pred[:, i] - y_true[:, i]
        metrics[f"bias_{name}"] = float(resid.mean())
        metrics[f"mae_{name}"] = float(np.abs(resid).mean())
    return metrics


def semantic_error_figures(y_true: np.ndarray, y_pred: np.ndarray, target_names) -> dict:
    """Build the diagnostic figures, keyed by artifact name."""
    import matplotlib.pyplot as plt

    n = len(target_names)
    figs = {}

    # 1) Testset-Histogramm: Verteilung Ground Truth vs. Vorhersage
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4.2))
    axes = np.atleast_1d(axes)
    for i, name in enumerate(target_names):
        axes[i].hist(y_true[:, i], bins=40, alpha=0.6, label="Ground Truth", color="steelblue")
        axes[i].hist(y_pred[:, i], bins=40, alpha=0.6, label="Vorhersage", color="indianred")
        axes[i].set(xlabel=name, ylabel="Anzahl", title=f"Verteilung {name}")
        axes[i].legend()
    fig.suptitle("Testset-Histogramm: Ground Truth vs. Vorhersage")
    fig.tight_layout()
    figs["histogram_pred_vs_groundtruth"] = fig

    # 2) Predicted vs. Groundtruth mit 1:1-Referenzlinie
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5))
    axes = np.atleast_1d(axes)
    for i, name in enumerate(target_names):
        lo = min(y_true[:, i].min(), y_pred[:, i].min())
        hi = max(y_true[:, i].max(), y_pred[:, i].max())
        axes[i].scatter(y_true[:, i], y_pred[:, i], s=10, alpha=0.5, color="steelblue")
        axes[i].plot([lo, hi], [lo, hi], "k--", linewidth=1, label="y = x (perfekt)")
        axes[i].set(
            xlabel=f"{name} (Ground Truth)",
            ylabel=f"{name} (Vorhersage)",
            title=f"Predicted vs. Groundtruth: {name}",
        )
        axes[i].legend()
    fig.tight_layout()
    figs["predicted_vs_groundtruth"] = fig

    return figs
