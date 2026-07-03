import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
import matplotlib.pyplot as plt


def _drop_non_finite(true_col: np.ndarray, pred_col: np.ndarray, name: str):
    """Filter rows where either column is NaN/Inf (e.g. a diverged trial).

    Plain min/max/histogram over non-finite values raise or produce garbage
    ranges, so this keeps plotting from crashing while surfacing the
    divergence instead of silently hiding it.
    """
    mask = np.isfinite(true_col) & np.isfinite(pred_col)
    if not mask.all():
        warnings.warn(
            f"{name}: dropping {(~mask).sum()}/{mask.size} non-finite "
            "predictions from plot (model likely diverged)."
        )
    return true_col[mask], pred_col[mask]


@torch.no_grad()
def collect_predictions(model, val_input, val_label, device, batch_size=512):
    if isinstance(val_input, Dataset):
        val_ds = val_input
    else:
        val_ds = TensorDataset(
            val_input, torch.as_tensor(val_label, dtype=torch.float32)
        )

    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    model.get_model().eval()

    trues, preds = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        mu = model.predict(x)[:, : y.shape[1]]
        trues.append(y.numpy())
        preds.append(mu.cpu().numpy())

    return np.concatenate(trues, axis=0), np.concatenate(preds, axis=0)


def plot_histogram_pred_vs_groundtruth(
    y_true: np.ndarray, y_pred: np.ndarray, target_names
):
    import matplotlib.pyplot as plt

    n = len(target_names)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4.2))
    axes = np.atleast_1d(axes)
    for i, name in enumerate(target_names):
        true_col, pred_col = _drop_non_finite(y_true[:, i], y_pred[:, i], name)
        axes[i].hist(
            true_col, bins=40, alpha=0.6, label="Ground Truth", color="steelblue"
        )
        axes[i].hist(
            pred_col, bins=40, alpha=0.6, label="Vorhersage", color="indianred"
        )
        axes[i].set(xlabel=name, ylabel="Anzahl", title=f"Verteilung {name}")
        axes[i].legend()
    fig.suptitle("Testset-Histogramm: Ground Truth vs. Vorhersage")
    fig.tight_layout()
    return fig


def plot_predicted_vs_groundtruth(y_true: np.ndarray, y_pred: np.ndarray, target_names):
    n = len(target_names)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5))
    axes = np.atleast_1d(axes)
    for i, name in enumerate(target_names):
        true_col, pred_col = _drop_non_finite(y_true[:, i], y_pred[:, i], name)
        lo = min(true_col.min(), pred_col.min())
        hi = max(true_col.max(), pred_col.max())
        axes[i].scatter(true_col, pred_col, s=10, alpha=0.5, color="steelblue")
        axes[i].plot([lo, hi], [lo, hi], "k--", linewidth=1, label="y = x (perfekt)")
        axes[i].set(
            xlabel=f"{name} (Ground Truth)",
            ylabel=f"{name} (Vorhersage)",
            title=f"Predicted vs. Groundtruth: {name}",
        )
        axes[i].legend()
    fig.tight_layout()
    return fig
