# KAN Weak Lensing

This is part of my specialization project "Weighing the Universe: Interpretable Machine Learning with Kolmogorov-Arnold Networks" in the MSc in Engineering.

The project systematically benchmarks Kolmogorov-Arnold Network variants across tasks of increasing complexity, in three phases:

- **Phase 1**: ten functional/Feynman-equation regression datasets.
- **Phase 2**: image datasets, MNIST, FashionMNIST, and a synthetic Gaussian-blobs regression benchmark.
- **Phase 3**: inferring the cosmological parameters $\Omega_m$ and $S_8$ from weak-lensing convergence maps (this repository).

Phases 1 and 2 are developed in [kan-lab](https://github.com/noahbaumgartner/kan-lab). This repository was factored out to develop the weak-lensing phase independently, so as not to inflate the scope of `kan-lab`. Six architectures are carried over: **fastkan**, **fasterkan**, **efficientkan**, **wavkan**, **kkan**, **kat** (see [Models](#models) below).

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

# choose a dimensionality-reduction method (default avgpool, see Dimensionality Reduction below)
uv run main.py model=wavkan dataset.reduction=conv

# overwrite parameters
uv run main.py model=fasterkan training.epochs=5 model.num_grids=16

# hyperparameter sweep (Optuna, via a config under configs/sweep/), here Stage 1 (architecture)
# under the MSE objective (Stage 1 must be run once per objective, see Staged Hyperparameter Search)
uv run main.py --multirun +sweep=tune_fastkan_stage1 dataset=weak_lensing objective=mse

# MLflow UI
uv run mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db
```

## Training Objectives

Two training strategies produce the final $(\hat\Omega_m, \hat S_8, \hat\sigma_{\Omega_m}, \hat\sigma_{S_8})$ estimate, chosen via the Hydra group `configs/objective/`:

| Strategy           | Hydra objective(s)                       | Outputs | Loss             | Description                                                                                                                              |
| ------------------- | ------------------------------------------- | ------- | ----------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| **MSE + Ensemble**  | Stage 1/2: `mse`, then Stage 3: `ensemble`  | 2 -> 4\* | MSE (per stage)   | Stage 1/2 tune a point-estimate model $(\hat\Omega_m, \hat S_8)$ under `objective=mse`. Stage 3 retrains the winning configuration $M$ times on bootstrapped subsets (bagging), mean/std over members give $(\hat\mu, \hat\sigma)$. |
| **Score**           | `score`                                     | 4       | score loss         | Point estimate + uncertainty, trained jointly in a single two-stage sweep against the negative FAIR-Universe leaderboard score. No ensemble stage. |

\* Each ensemble member predicts 2 values. The ensemble's $(\mu, \log\sigma)$ (4 values) come from the mean/spread across members.

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

`kkan` and `kat` consume the 2D map directly and do not use any of these. Instead they get a small learnable Conv-Stem (`src/modules/convstem.py`, strided Conv2d + BatchNorm + SiLU layers, where `model.stem_channels` is part of the Stage-2 sweep for these two models and `model.stem_layers` stays pinned) that downscales the native 1424x176 map to its target resolution: `kkan` (`stem_layers: 2`) to 356x44, `kat` (`stem_layers: 3`) to 178x22, both exact power-of-2 downscales that preserve the native ~8:1 aspect ratio without distortion.

## Staged Hyperparameter Search

A single sweep over architecture, optimizer, model-specific parameters, and reduction method at once has too many combinations to converge within the cluster's time limit. Search therefore proceeds in stages per model, each fixing the best configuration from the previous stage. Stage 1 uses the same architecture search space for both objectives (see [Training Objectives](#training-objectives)), but since MSE and Score train under different loss functions and output dimensions, it still needs to be run once per objective, producing two independent Stage-1 winners that each feed their own Stage-2 sweep. Stage 3 (deep ensemble) exists only for the MSE track, since Score already learns its own uncertainty jointly.

- **Stage 1, architecture** (`configs/sweep/tune_<model>_stage1.yaml`) sweeps only architecture and optimizer (`lr`, `weight_decay`): `model.n_hidden_layers`/`hidden_width_*` for the MLP-style models, the transformer capacity knobs (`embed_dim`, `depth`, `num_heads`, `mlp_ratio`) plus `training.batch_size` for `kat`, and just `training.batch_size` for `kkan` (its first KAN-convolution layer alone needs ~26GB at batch size 128, so batch size is the binding capacity knob there). Reduction is pinned to `conv` instead of the dataset default `avgpool`, since `conv` retains more small-scale, non-Gaussian information, needed to break the $\Omega_m$-$S_8$ degeneracy (not applicable to `kkan`/`kat`, which have no reduction stage). Model-specific parameters (`grid_size`, `num_grids`, `wavelet_type`, ...) stay at their config default. One config per model, but **run twice**, once with `objective=mse` and once with `objective=score`, since the trained loss differs even though the search space doesn't.
- **Stage 2, reduction/model params** (`tune_<model>_stage2_mse.yaml`/`tune_<model>_stage2_score.yaml`) fixes the matching Stage-1 winner (the one trained under the same objective) and sweeps model-specific parameters together with the reduction method (where applicable, not for `kkan`/`kat`). Run separately per objective, the `_mse`/`_score` suffix picks which and already bakes in the corresponding `objective=` override. Architecture and optimizer are **not** swept again. Every `_stage2_*` config is fully self-contained: it pins the fixed architecture, optimizer, and (for `kkan`/`kat`) batch size directly, so no extra overrides are needed at submit time.
- **Stage 3, ensemble** (MSE track only, no `_score` counterpart) retrains the best Stage-1/2 (MSE) configuration $M$ times with different seeds/data subsets to form the deep ensemble. No further hyperparameter search, a fixed recipe under `configs/ensemble/<model>.yaml` replays the winning config directly instead of running another Optuna sweep, see [Ensemble Runs](#ensemble-runs) below.

```bash
# Stage 1: sweep architecture + optimizer, once per objective (same search space, different loss)
sbatch --export=ALL,EXPERIMENT=wl_efficientkan_mse,SWEEP_SUFFIX=_stage1,OBJECTIVE=mse \
  scripts/tune_efficientkan.submit
sbatch --export=ALL,EXPERIMENT=wl_efficientkan_score,SWEEP_SUFFIX=_stage1,OBJECTIVE=score \
  scripts/tune_efficientkan.submit

# inspect the best Stage-1 trial per objective in MLflow, then pin its winning
# values into the corresponding _stage2_* config

# Stage 2: sweep model-specific params + reduction. Architecture, optimizer, and
# objective are all pinned inside the _stage2_mse config, nothing else to pass
sbatch --export=ALL,EXPERIMENT=wl_efficientkan,SWEEP_SUFFIX=_stage2_mse \
  scripts/tune_efficientkan.submit

# all models at once, per objective (submit_all.sh needs SWEEP_SUFFIX, there's no unstaged default anymore)
SWEEP_SUFFIX=_stage1 OBJECTIVE=mse ./scripts/submit_all.sh wl_stage1_mse
SWEEP_SUFFIX=_stage1 OBJECTIVE=score ./scripts/submit_all.sh wl_stage1_score

# kkan/kat use the same submit_all.sh for both stages, just not in the default MODELS list
MODELS="kat kkan" SWEEP_SUFFIX=_stage2_mse ./scripts/submit_all.sh wl_kkan_kat_mse
MODELS="kat kkan" SWEEP_SUFFIX=_stage2_score ./scripts/submit_all.sh wl_kkan_kat_score
```

The Stage-1 to Stage-2 handoff is deliberately manual (no automatic best-trial lookup), simple to control and no extra code needed.

## Ensemble Runs

Stage 3 of the MSE + Ensemble track is a fixed recipe, not a sweep. Each `configs/ensemble/<model>.yaml` pins architecture, reduction, optimizer, and training settings to that model's best Stage-1/2 (MSE) configuration, sets `objective: ensemble`, and picks the member architecture via `configs/member/<model>.yaml`. Running it retrains that exact configuration `model.n_members` times on bootstrapped/subsampled data (see [Training Objectives](#training-objectives)), and the mean/std across members give $(\hat\mu, \hat\sigma)$.

```bash
# one architecture directly
sbatch --export=ALL,CONFIG=efficientkan,EXPERIMENT=wl_ensemble scripts/run_ensemble.submit

# every configs/ensemble/*.yaml recipe at once
./scripts/run_ensemble_all.sh <experiment_name>

# same thing locally, without SLURM
uv run main.py +ensemble=efficientkan dataset=weak_lensing
```

Adding a new architecture's ensemble recipe means copying its best Stage-1/2 (MSE) overrides by hand into a new `configs/ensemble/<model>.yaml`, the same manual handoff as Stage 1 to Stage 2.

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
