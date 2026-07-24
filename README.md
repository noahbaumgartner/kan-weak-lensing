# KAN Weak Lensing

This is part of my specialization project "Weighing the Universe: Interpretable Machine Learning with Kolmogorov-Arnold Networks" in the MSc in Engineering.

The project systematically benchmarks Kolmogorov-Arnold Network variants across tasks of increasing complexity, in three phases:

- **Phase 1** — ten functional/Feynman-equation regression datasets.
- **Phase 2** — image datasets: MNIST, FashionMNIST, and a synthetic Gaussian-blobs regression benchmark.
- **Phase 3** — inferring the cosmological parameters $\Omega_m$ and $S_8$ from weak-lensing convergence maps (this repository).

Phases 1 and 2 are developed in [kan-lab](https://github.com/noahbaumgartner/kan-lab); this repository was factored out to develop the weak-lensing phase independently, so as not to inflate the scope of `kan-lab`. Six architectures are carried over: **fastkan**, **fasterkan**, **efficientkan**, **wavkan**, **kkan**, **kat** (see [Models](#models) below).

## Setup

This repository works with `uv`. To set it up locally:

1. Clone this repository: `git clone <repo-url>`
2. Run `uv sync` to install the needed Python version and dependencies (includes `kat-rational`, pulled from Git)

Set the path to the `.npy` map files via `data_dir` in `configs/dataset/weak_lensing.yaml`.

Runs log to the MLflow tracking server at `mlflow_tracking_uri` in `configs/config.yaml` (`http://127.0.0.1:9299` by default, same as `kan-lab`). Start it locally before training:

```bash
./scripts/mlflow_server.sh
# or on the cluster: sbatch scripts/mlflow_server.submit
```

## Usage

The following commands show typical uses of this lab environment.

```bash
# run default experiment (model=fastkan, objective=score, see configs/config.yaml)
uv run main.py

# run with another model, another training objective
uv run main.py model=wavkan
uv run main.py model=fastkan objective=mse

# choose a dimensionality-reduction method (default avgpool; see Dimensionality Reduction below)
uv run main.py model=wavkan dataset.reduction=conv

# overwrite parameters
uv run main.py model=fasterkan training.epochs=5 model.num_grids=16

# hyperparameter sweep (Optuna, via a config under configs/sweep/)
uv run main.py --multirun +sweep=image/tune_fastkan dataset=weak_lensing

# MLflow UI
uv run mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db

# notebooks
uv run jupyter notebook notebooks/
```

## Training Objectives

Two training strategies produce the final $(\hat\Omega_m, \hat S_8, \hat\sigma_{\Omega_m}, \hat\sigma_{S_8})$ estimate, chosen via the Hydra group `configs/objective/`:

| Strategy           | Hydra objective(s)                       | Outputs | Loss             | Description                                                                                                                              |
| ------------------- | ------------------------------------------- | ------- | ----------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| **MSE + Ensemble**  | Stage 1/2: `mse`, then Stage 3: `ensemble`  | 2 -> 4\* | MSE (per stage)   | Stage 1/2 tune a point-estimate model $(\hat\Omega_m, \hat S_8)$ under `objective=mse`. Stage 3 retrains the winning configuration $M$ times on bootstrapped subsets (bagging); mean/std over members give $(\hat\mu, \hat\sigma)$. |
| **Score**           | `score`                                     | 4       | score loss         | Point estimate + uncertainty, trained jointly in a single two-stage sweep against the negative FAIR-Universe leaderboard score. No ensemble stage. |

\* Each ensemble member predicts 2 values; the ensemble's $(\mu, \log\sigma)$ (4 values) come from the mean/spread across members.

```bash
# MSE + Ensemble, stage 1/2: point-estimate regression on (Om, S8)
uv run main.py model=fastkan objective=mse

# MSE + Ensemble, stage 3: deep ensemble (mean/std over N members)
uv run main.py objective=ensemble                       # 8 fastkan members
uv run main.py objective=ensemble model.n_members=12
uv run main.py objective=ensemble member@model.member=wavkan

# Score (default)
uv run main.py model=fastkan objective=score
```

## Dimensionality Reduction

`fastkan`, `fasterkan`, `efficientkan`, and `wavkan` are MLP-style KANs and need the 1424x176 convergence map reduced to a flat vector first (`dataset.reduction`, part of the Stage-2 sweep below):

| Method       | Learnable | Description                                                                             |
| ------------ | --------- | ------------------------------------------------------------------------------------------ |
| `avgpool`    | No        | Fixed-stride average pooling, then flatten (dataset default, stride 8).                     |
| `conv`       | Yes       | Small strided-convolution encoder + global average pooling.                                 |
| `scattering` | No        | 2D wavelet scattering transform ([kymatio](https://www.kymat.io/)) + global average pooling. |

`kkan` and `kat` consume the 2D map directly and do not use any of these; instead their model wrapper bilinearly resizes the native map to `dataset.img_size_h`x`img_size_w` (default 178x22, an exact /8 downscale that preserves the native ~8:1 aspect ratio without distortion).

## Staged Hyperparameter Search

A single sweep over architecture, optimizer, model-specific parameters, and reduction method at once has too many combinations to converge within the cluster's time limit. Search therefore proceeds in stages per model, each fixing the best configuration from the previous stage:

- **Stage 1** (`configs/sweep/image/tune_<model>_arch.yaml`) sweeps only architecture (`model.n_hidden_layers`/`hidden_width_*`, or the transformer/capacity knobs for `kkan`/`kat`) and optimizer (`lr`, `weight_decay`). Reduction is pinned to `conv` instead of the dataset default `avgpool` — `conv` retains more small-scale, non-Gaussian information, needed to break the $\Omega_m$-$S_8$ degeneracy (not applicable to `kkan`/`kat`, which have no reduction stage). Model-specific parameters (`grid_size`, `num_grids`, `wavelet_type`, ...) stay at their config default.
- **Stage 2** (`tune_<model>_reduction.yaml`, or `tune_<model>_model.yaml` for `kkan`/`kat`) fixes the Stage-1 winner and sweeps model-specific parameters together with the reduction method, run separately per objective (`_mse`/`_score` suffix). Architecture and optimizer are **not** swept again — the Stage-1 winner is passed in as fixed Hydra overrides.
- **Stage 3** (`objective=ensemble` only) retrains the best Stage-1/2 configuration $M$ times to form the deep ensemble; no further hyperparameter search.

```bash
# Stage 1: sweep architecture + optimizer
sbatch --export=ALL,EXPERIMENT=wl_efficientkan,SWEEP=image/tune_efficientkan_arch \
  scripts/tune_efficientkan.submit

# inspect the best Stage-1 trial in MLflow, then fix its architecture/optimizer overrides

# Stage 2: sweep model-specific params + reduction, Stage-1 winner passed in explicitly
sbatch --export=ALL,EXPERIMENT=wl_efficientkan,SWEEP=image/tune_efficientkan_reduction,\
EXTRA_weak_lensing="model.n_hidden_layers=2 model.hidden_width_0=494 model.hidden_width_1=409 optimizer.lr=8e-4 optimizer.weight_decay=1e-5" \
  scripts/tune_efficientkan.submit

# all models at once
./scripts/submit_all.sh <experiment_name>
```

The Stage-1 -> Stage-2 handoff is deliberately manual (no automatic best-trial lookup) — simple to control, no extra code.

## Dataset

| Name                            | Task       | Input                                | Output                                                      | Description                                                                                             |
| -------------------------------- | ---------- | -------------------------------------- | -------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| **FAIR-Universe Weak Lensing**   | Regression | 1424x176 (noisy convergence map)      | $\Omega_m$, $S_8$ (+ uncertainty under `score`/`ensemble`)      | Infer cosmological parameters from a weak-lensing map ([FAIR Universe](https://fair-universe.lbl.gov/)) |

101 cosmologies x 256 nuisance-parameter realisations, split 80/20 along the realisation axis (`dataset.val_fraction`). Every map is perturbed with fresh, simulated shape noise each epoch.

## Models

The following KAN variants are implemented in this project, carried over from [kan-lab](https://github.com/noahbaumgartner/kan-lab). The module code in `src/modules/` is copied from the respective repositories.

| Model        | Config name    | Repository                                                                                 | Paper                                                |
| ------------ | -------------- | -------------------------------------------------------------------------------------------- | ------------------------------------------------------- |
| EfficientKAN | `efficientkan` | [Blealtan/efficient-kan](https://github.com/Blealtan/efficient-kan)                          | -                                                     |
| FastKAN      | `fastkan`      | [ZiyaoLi/fast-kan](https://github.com/ZiyaoLi/fast-kan)                                      | [arXiv:2405.06721](https://arxiv.org/abs/2405.06721) |
| FasterKAN    | `fasterkan`    | [AthanasiosDelis/faster-kan](https://github.com/AthanasiosDelis/faster-kan)                  | -                                                     |
| WavKAN       | `wavkan`       | [zavareh1/Wav-KAN](https://github.com/zavareh1/Wav-KAN)                                      | [arXiv:2405.12832](https://arxiv.org/abs/2405.12832) |
| KKAN         | `kkan`         | [AntonioTepsich/Convolutional-KANs](https://github.com/AntonioTepsich/Convolutional-KANs)    | [arXiv:2406.13155](https://arxiv.org/abs/2406.13155) |
| KAT          | `kat`          | [Adamdad/kat](https://github.com/Adamdad/kat)                                                | [arXiv:2409.10594](https://arxiv.org/abs/2409.10594) |
