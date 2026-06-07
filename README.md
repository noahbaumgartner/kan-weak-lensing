# kan-weak-lensing

KAN-Architekturvergleich auf dem FAIR-Universe **Weak Lensing** Datensatz.
Ausgelagert aus `kan-lab`, damit der Weak-Lensing-Teil eigenständig
weiterentwickelt werden kann.

Übernommene Modelle: **fastkan**, **fasterkan**, **wavkan**, **kkan**, **kat**.

## Setup (uv)

Identisch zu `kan-lab`: uv-managed Python 3.12, `kat-rational` aus Git.

```bash
uv sync
```

## Lokaler Lauf

```bash
# fastkan auf weak_lensing, lokaler MLflow-File-Store (./mlruns)
./scripts/run_local.sh

# anderes Modell / Hydra-Overrides
MODEL=fasterkan ./scripts/run_local.sh training.epochs=5 model.num_grids=16
```

Direkt über Hydra:

```bash
uv run python main.py model=wavkan dataset=weak_lensing training=adam
```

Der Pfad zu den `.npy`-Dateien wird in `configs/dataset/weak_lensing.yaml`
(`data_dir`) gesetzt.

## MLflow

Setup identisch zu `kan-lab`. Tracking-URI Default: `http://127.0.0.1:9299`
(siehe `configs/config.yaml`).

```bash
# Tracking-Server starten (sqlite backend + artifacts unter ./mlflow)
./scripts/mlflow_server.sh
# oder auf dem Cluster: sbatch scripts/mlflow_server.submit
```

## Sweeping (Optuna)

Sweep-Konfigurationen liegen unter `configs/sweep/`. `tune_base.yaml` definiert
den Optuna/TPE-Sweeper; die modell-spezifischen Sweeps erben davon.

```bash
# Einzelnen Sweep lokal laufen lassen
uv run python main.py --multirun +sweep=image/tune_fastkan_wl dataset=weak_lensing

# Alle MLP-artigen KANs auf dem Cluster submitten (fastkan, fasterkan, wavkan)
./scripts/submit_all_wl.sh <experiment_name>
```

## Struktur

```
main.py                     Hydra-Entrypoint
configs/                    Hydra-Configs (config, model, dataset, optimizer, training, sweep)
src/
  dataset.py                WeakLensingDataset (einziger Datensatz)
  models/                   Modell-Wrapper (fastkan, fasterkan, wavkan, kkan, kat) + base
  modules/                  KAN-Implementierungen + reduction (image -> vector)
  optimizers/               Adam, AdamW, SGD, LBFGS
  training/                 Trainer + Weak-Lensing-Scoring (metrics.py)
scripts/                    MLflow-Server + SLURM/lokale Tuning-Jobs
```

## Hinweis zu kkan / kat

`fastkan`, `fasterkan` und `wavkan` reduzieren die 1424×176-Maps (avgpool /
kymatio) und laufen direkt auf Weak Lensing — fertige Sweeps:
`configs/sweep/image/tune_{fastkan,fasterkan,wavkan}_wl.yaml`.

`kkan` (Conv-KAN) und `kat` (KAN-ViT) wurden in `kan-lab` auf **quadratischen
Klassifikations-Bildern** (MNIST etc.) verwendet. Sie sind hier vollständig
übernommen, benötigen für Weak Lensing aber noch Anpassung (quadratischer
Input / `dataset.img_size`, bei kat zusätzlich `dataset.patch_size`, sowie ein
Regressions-Head). Ihre Sweep- und Submit-Configs sind als Ausgangspunkt
enthalten.
