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


def _mask_bbox(mask_strip: np.ndarray):
    """Tight (row_slice, col_slice) bounding box around the valid (True)
    region of a 2D boolean mask, already in strip orientation -- the WIDE12H
    footprint does not span the full 1424-pixel length, so without this the
    figure has a large blank/masked (white, since masked pixels are 0 and 0
    is white on RdBu_r) margin on one side."""
    rows = np.any(mask_strip, axis=1)
    cols = np.any(mask_strip, axis=0)
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    return slice(int(r0), int(r1) + 1), slice(int(c0), int(c1) + 1)


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


def _plot_noise_comparison(
    noiseless_img: np.ndarray,
    noisy_img: np.ndarray,
    noise_sigma: float,
    vmax: float,
):
    """Single figure, noiseless map stacked above the noisy one (matches the
    top/bottom convention already used for Figure~\\ref{fig:mnist_fashionmnist}).
    No suptitle: like the existing MNIST/FashionMNIST and Gaussian-blob example
    figures, the overall description belongs in the LaTeX \\caption{}, not
    baked into the image -- only the per-panel condition label is kept here,
    analogous to their per-panel "MNIST: 7"-style titles."""
    # Panels are very short relative to their width (WIDE12H strip); a fixed
    # inch budget is reserved for the suptitle + per-panel titles. aspect
    # is left to "auto" (not "equal") and layout to "constrained" (not a
    # manual subplots_adjust) so the image always fills its axes box exactly
    # -- with "equal" + manual spacing, a mismatch between the axes box and
    # the image's true pixel aspect ratio left a blank margin on one side.
    h, w = noiseless_img.shape
    panel_h = 12 * h / w
    fig, axes = plt.subplots(
        2, 1, figsize=(12, 2 * panel_h + 1.3), sharex=True, sharey=True,
        layout="constrained",
    )
    panels = [
        (axes[0], noiseless_img, "Simulated convergence map (noise-free)"),
        (axes[1], noisy_img, rf"With shape noise ($\sigma_\kappa \approx {noise_sigma:.3f}$)"),
    ]
    for ax, img, subtitle in panels:
        im = ax.imshow(img, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_title(subtitle, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label=r"$\kappa$")
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
    bbox = _mask_bbox(_to_strip(train_ds.mask))

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
            img = _to_strip(x.squeeze(0).numpy())[bbox]

            vmax = float(np.nanpercentile(np.abs(img), 99))
            vmax = vmax if vmax > 0 else 1.0

            title = ", ".join(f"{n_}={v:.3f}" for n_, v in zip(target_names, y.numpy()))
            fig = _plot_map(img, title, vmax)
            mlflow.log_figure(fig, f"example_maps/map_{rank:02d}.png")
            plt.close(fig)

        # Combined noiseless/noisy comparison figure for the first sampled
        # map, one shared colour scale (see module docstring for why this is
        # worth checking rather than assuming noise is negligible by eye).
        i0 = int(idx[0])
        y0 = train_ds[i0][1].numpy()
        om, s8 = float(y0[0]), float(y0[1])
        noisy_img = _to_strip(train_ds[i0][0].squeeze(0).numpy())[bbox]
        noiseless_img = _to_strip(_dense_noiseless(train_ds, i0))[bbox]
        vmax = float(np.nanpercentile(np.abs(noisy_img), 99))
        vmax = vmax if vmax > 0 else 1.0

        ng = float(cfg.dataset.ng)
        pixel_size_arcmin = float(cfg.dataset.pixel_size_arcmin)
        noise_sigma = 0.4 / (2.0 * ng * pixel_size_arcmin**2) ** 0.5

        # Logged, not drawn into the figure -- these belong in the LaTeX
        # \caption{}, matching how fig:mnist_fashionmnist/fig:gaussian_blob
        # keep their overall description out of the image itself.
        mlflow.log_param("comparison_om", round(om, 4))
        mlflow.log_param("comparison_s8", round(s8, 4))
        mlflow.log_param("comparison_noise_sigma", round(noise_sigma, 4))

        fig = _plot_noise_comparison(noiseless_img, noisy_img, noise_sigma, vmax)
        mlflow.log_figure(fig, "comparison/noise_comparison.png")
        plt.close(fig)

    print(f"Logged {n} example maps and a combined noiseless/noisy comparison to MLflow experiment '{EXPERIMENT_NAME}'.")


if __name__ == "__main__":
    main()
