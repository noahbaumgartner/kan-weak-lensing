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

## Versuche (objective group)

Ein *Versuch* = Trainings-Objective = Kombination aus Loss-Funktion und
Modell-Head-Breite. Gewählt über die Hydra-Gruppe `configs/objective/`:

| objective | Outputs | Loss             | Trainer-Pfad / Logging          |
|-----------|---------|------------------|---------------------------------|
| `score`   | 4       | `score_inference`| Score + 68%-Coverage + MSE(mu)  |
| `mse`     | 2       | MSE              | MSE / RMSE / R²                 |

```bash
# Versuch 1 — reine MSE-Regression auf (Om, S8)
uv run python main.py model=fastkan objective=mse

# Versuch 2 — score-as-loss (Default)
uv run python main.py model=fastkan objective=score
```

Der Datensatz behält immer die 2 Labels (Om, S8); pro Versuch ändert sich nur
der Head (`output_dim`) und der Loss — der Trainer wählt den passenden
Eval-/Logging-Pfad automatisch anhand von `dataset.loss`.

**Weiteren Versuch hinzufügen:** neue Datei `configs/objective/<name>.yaml` mit
`# @package _global_` anlegen, darin `dataset.loss`, `dataset.output_dim`,
`dataset.num_targets` (und ggf. `training.*`) setzen. Bei einem neuen Loss
zusätzlich `src/training/trainer.py::_create_loss_fn` erweitern. Lauf:
`objective=<name>`.

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
# Einzelnen Sweep lokal laufen lassen (Default-objective aus config.yaml = score)
uv run python main.py --multirun +sweep=image/tune_fastkan_wl dataset=weak_lensing

# Sweep mit anderem Versuch
uv run python main.py --multirun +sweep=image/tune_fastkan_wl objective=mse

# Alle MLP-artigen KANs auf dem Cluster submitten (fastkan, fasterkan, wavkan)
./scripts/submit_all_wl.sh <experiment_name>
# anderer Versuch im Sweep: OBJECTIVE-Env an den Submit-Job geben, z.B.
sbatch --export=ALL,EXPERIMENT=wl_mse,OBJECTIVE=mse scripts/tune_fastkan_wl.submit
```

## Struktur

```
main.py                     Hydra-Entrypoint
configs/                    Hydra-Configs (config, model, dataset, objective, optimizer, training, sweep)
  objective/                Versuche: score (4 out, score-loss) / mse (2 out, MSE)
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
