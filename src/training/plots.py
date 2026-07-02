import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
import matplotlib.pyplot as plt


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
        axes[i].hist(
            y_true[:, i], bins=40, alpha=0.6, label="Ground Truth", color="steelblue"
        )
        axes[i].hist(
            y_pred[:, i], bins=40, alpha=0.6, label="Vorhersage", color="indianred"
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
    return fig
