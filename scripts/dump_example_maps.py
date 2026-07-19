"""Dump a handful of example weak-lensing convergence maps as MLflow
artifacts -- for use as an illustrative figure in the thesis (Experiments,
Section on Weak-Lensing Cosmology), analogous to the MNIST/FashionMNIST and
Gaussian-blob example figures already used for Phases 1/2.

Builds the real dataset via the same Hydra config as training runs
(dataset=weak_lensing), but does no training: it just reconstructs a few
dense maps from the training split -- including the shape noise that is
actually added during training -- and logs each as its own PNG artifact
under a new, dedicated MLflow experiment.

In addition to the N_EXAMPLES noisy gallery maps, it logs one noiseless/noisy
comparison pair for the same underlying map (same colour scale in both
panels), since shape noise at this survey's ng/pixel-size is not obviously
negligible by eye: sigma = 0.4/sqrt(2*ng*pixel_size_arcmin^2) ~= 0.026 in
kappa units (Section~\ref{sec:weak_lensing}), comparable to the signal RMS
itself -- this is the standard weak-lensing "shape noise dominates the
per-pixel signal" situation that motivates learning calibrated uncertainties
(Section~\ref{sec:uq}) rather than just point estimates.

Standardisation (input z-scoring, label z-scoring) is disabled here so the
saved images show the raw kappa scale and the plot titles show the actual
(Omega_m, S8) values, rather than normalised training-time values.

Run directly (not via main.py), e.g. via scripts/dump_example_maps.submit:
    uv run python scripts/dump_example_maps.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.modules.reduction import reduced_dim
from src.training.utils import generate_run_name

N_EXAMPLES = 10
SEED = 0
TRACKING_URI_DEFAULT = "http://127.0.0.1:9299"
EXPERIMENT_NAME = "weak_lensing_example_maps"


def _make_width(in_d, out_d, n_h, *hidden_widths):
    n_h = int(n_h)
    return [int(in_d)] + [int(w) for w in hidden_widths[:n_h]] + [int(out_d)]


OmegaConf.register_new_resolver("make_width", _make_width, replace=True)
OmegaConf.register_new_resolver("reduced_dim", reduced_dim, replace=True)


def _to_strip(dense: np.ndarray) -> np.ndarray:
    """(H, W) -> transposed wide strip, matching the WIDE12H survey field;
    far more legible than the native tall H x W = 1424 x 176 orientation."""
    return dense.T


def _dense_noiseless(map_ds, i: int) -> np.ndarray:
    """Reconstruct the same map map_ds[i] would return, but without the
    shape-noise step of _MapDataset.__getitem__ (map_ds.add_noise is a
    per-dataset flag, always True for the training split -- this bypasses it
    for a direct noiseless/noisy comparison)."""
    dense = np.zeros(map_ds.mask.shape, dtype=np.float32)
    dense[map_ds.mask] = map_ds.kappa_flat[i].astype(np.float32, copy=False)
    return dense


def _plot_map(img: np.ndarray, title: str, vmax: float):
    fig, ax = plt.subplots(figsize=(12, 12 * img.shape[0] / img.shape[1]))
    im = ax.imshow(img, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label=r"$\kappa$")
    fig.tight_layout()
    return fig


def main():
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(
            config_name="config",
            overrides=[
                "dataset=weak_lensing",
                # Raw scale for display, not the training-time normalised one.
                "dataset.standardize=false",
                "dataset.standardize_labels=false",
            ],
        )

    dataset = instantiate(cfg.dataset)
    data = dataset.create()
    train_ds = data["train_input"]
    target_names = list(cfg.dataset.get("target_names", ["Omega_m", "S8"]))

    rng = np.random.default_rng(SEED)
    n = min(N_EXAMPLES, len(train_ds))
    idx = rng.choice(len(train_ds), size=n, replace=False)

    mlflow.set_tracking_uri(cfg.get("mlflow_tracking_uri", TRACKING_URI_DEFAULT))
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=generate_run_name()):
        mlflow.log_param("n_examples", int(n))
        mlflow.log_param("dataset", "weak_lensing")
        mlflow.log_param("source_split", "train (with shape noise)")

        for rank, i in enumerate(idx):
            x, y = train_ds[int(i)]
            img = _to_strip(x.squeeze(0).numpy())

            vmax = float(np.nanpercentile(np.abs(img), 99))
            vmax = vmax if vmax > 0 else 1.0

            title = ", ".join(f"{n_}={v:.3f}" for n_, v in zip(target_names, y.numpy()))
            fig = _plot_map(img, title, vmax)
            mlflow.log_figure(fig, f"example_maps/map_{rank:02d}.png")
            plt.close(fig)

        # Noiseless/noisy comparison pair for the first sampled map, same
        # colour scale in both panels (see module docstring for why this is
        # worth checking rather than assuming noise is negligible by eye).
        i0 = int(idx[0])
        noisy_img = _to_strip(train_ds[i0][0].squeeze(0).numpy())
        noiseless_img = _to_strip(_dense_noiseless(train_ds, i0))
        vmax = float(np.nanpercentile(np.abs(noisy_img), 99))
        vmax = vmax if vmax > 0 else 1.0

        fig = _plot_map(noiseless_img, "Without shape noise", vmax)
        mlflow.log_figure(fig, "comparison/noiseless.png")
        plt.close(fig)

        fig = _plot_map(noisy_img, "With shape noise", vmax)
        mlflow.log_figure(fig, "comparison/noisy.png")
        plt.close(fig)

    print(f"Logged {n} example maps and a noiseless/noisy pair to MLflow experiment '{EXPERIMENT_NAME}'.")


if __name__ == "__main__":
    main()
