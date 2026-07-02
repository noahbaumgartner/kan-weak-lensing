# kan-weak-lensing

KAN-Architekturvergleich auf dem FAIR-Universe **Weak Lensing** Datensatz.
Ausgelagert aus `kan-lab`, damit der Weak-Lensing-Teil eigenständig
weiterentwickelt werden kann.

Übernommene Modelle: **fastkan**, **fasterkan**, **efficientkan**, **wavkan**, **kkan**, **kat**.

## Setup (uv)

Identisch zu `kan-lab`: uv-managed Python 3.12, `kat-rational` aus Git.

```bash
uv sync
```

## Lokaler Lauf

Direkt über Hydra:

```bash
# fastkan auf weak_lensing (Default-objective score)
uv run python main.py model=fastkan dataset=weak_lensing training=adam

# anderes Modell / Hydra-Overrides
uv run python main.py model=fasterkan dataset=weak_lensing training=adam training.epochs=5 model.num_grids=16

# Reduction waehlen (Default avgpool)
uv run python main.py model=wavkan dataset=weak_lensing training=adam dataset.reduction=conv
```

Der Pfad zu den `.npy`-Dateien wird in `configs/dataset/weak_lensing.yaml`
(`data_dir`) gesetzt.

## Versuche (objective group)

Ein *Versuch* = Trainings-Objective = Kombination aus Loss-Funktion und
Modell-Head-Breite. Gewählt über die Hydra-Gruppe `configs/objective/`:

| objective   | Outputs | Loss             | Trainer-Pfad / Logging          |
|-------------|---------|------------------|---------------------------------|
| `score`     | 4       | `score_loss_fn`  | Score + 68%-Coverage + MSE(mu)  |
| `mse`       | 2       | MSE              | MSE / RMSE / R²                 |
| `ensemble`  | 4*      | MSE pro Member   | Score + 68%-Coverage + MSE(mu)  |

\* Beim Ensemble sagt jeder Member 2 Werte voraus; mu/log_sigma (4 Werte)
entstehen aus Mittelwert/Streuung über die Member.

```bash
# Versuch 1 — reine MSE-Regression auf (Om, S8)
uv run python main.py model=fastkan objective=mse

# Versuch 2 — score-as-loss (Default)
uv run python main.py model=fastkan objective=score

# Versuch 3 — Deep Ensemble (mean/std über N Member = Gauss-Posterior)
uv run python main.py objective=ensemble                       # 8 fastkan-Member
uv run python main.py objective=ensemble model.n_members=12
uv run python main.py objective=ensemble member@model.member=wavkan
```

### Ensemble-Versuch im Detail

Mehrere KANs mit unterschiedlicher Init werden je auf MSE trainiert; die
Streuung zwischen den Membern liefert σ (Gauss-Annahme), der Mittelwert den
Punktschätzer. So bekommt der Ensemble-Versuch dieselben Score-/Coverage-
Metriken wie `score`, ohne σ direkt zu lernen.

Stellschrauben: `model.n_members` (8, 12, …), `model.seed_base`,
Member-Architektur via `member@model.member=<fastkan|fasterkan|efficientkan|wavkan>`
(`configs/member/`), Member-Hyperparameter via `model.member.*`. Sweep:
`configs/sweep/image/tune_ensemble.yaml` (sweept u.a. `n_members`).

Noch *nicht* enthalten ist der dritte Vorschlag (Samples direkt in den
Probability-Space mappen via spatial NN / Normalizing Flow) — das ist ein
eigener, größerer Versuch.

Der Datensatz behält immer die 2 Labels (Om, S8); pro Versuch ändert sich nur
der Head (`output_dim`) und der Loss — der Trainer wählt den passenden
Eval-/Logging-Pfad automatisch anhand des top-level `objective`-Felds
(`src/training/trainer.py::_SCORE_OBJECTIVES`).

**Weiteren Versuch hinzufügen:** neue Datei `configs/objective/<name>.yaml` mit
`# @package _global_` anlegen, darin `objective: <name>`, `dataset.output_dim`,
`dataset.num_targets` (und ggf. `training.*`) setzen. Bei einem neuen Loss
zusätzlich `src/training/trainer.py::_create_loss_fn` und `_SCORE_OBJECTIVES`
erweitern. Lauf: `objective=<name>`.

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
den Optuna/TPE-Sweeper; die modell-spezifischen Sweeps erben davon. Die
Reduction (`avgpool` | `conv`) ist **nicht** Teil des Sweeps, sondern wird
fix pro Job über `REDUCTION` mitgegeben (Default `avgpool`).

```bash
# Einzelnen Sweep lokal laufen lassen (Default-objective aus config.yaml = score)
uv run python main.py --multirun +sweep=image/tune_fastkan dataset=weak_lensing

# Sweep mit anderem Versuch / anderer Reduction
uv run python main.py --multirun +sweep=image/tune_fastkan objective=mse dataset.reduction=conv

# Alle Modelle auf dem Cluster submitten (fastkan, fasterkan, efficientkan,
# wavkan, kkan, kat) — ein Job pro Modell
./scripts/submit_all.sh <experiment_name>
# Versuch / Reduction im Sweep: per Env an den Submit-Job geben, z.B.
sbatch --export=ALL,EXPERIMENT=wl_mse,OBJECTIVE=mse scripts/tune_fastkan.submit
sbatch --export=ALL,EXPERIMENT=wl_conv,REDUCTION=conv scripts/tune_fastkan.submit
```

## Struktur

```
main.py                     Hydra-Entrypoint
configs/                    Hydra-Configs (config, model, dataset, objective, optimizer, training, sweep)
  objective/                Versuche: score (4 out, score-loss) / mse (2 out, MSE)
src/
  dataset.py                WeakLensingDataset (einziger Datensatz)
  models/                   Modell-Wrapper (fastkan, fasterkan, efficientkan, wavkan, kkan, kat) + base
  modules/                  KAN-Implementierungen + reduction (image -> vector)
  optimizers/               Adam
  training/                 Trainer + Weak-Lensing-Scoring (metrics.py)
scripts/                    MLflow-Server + SLURM-Tuning-Jobs
```

## Hinweis zu kkan / kat

`fastkan`, `fasterkan`, `efficientkan` und `wavkan` reduzieren die 1424×176-Maps
(avgpool / conv) und laufen direkt auf Weak Lensing —
fertige Sweeps: `configs/sweep/image/tune_{fastkan,fasterkan,efficientkan,wavkan}.yaml`.

`kkan` (Conv-KAN) und `kat` (KAN-ViT) wurden in `kan-lab` auf **quadratischen
Klassifikations-Bildern** (MNIST etc.) verwendet. Für Weak Lensing wird die
1424×176-Map im Modell-Wrapper bilinear auf `dataset.img_size_h`×`img_size_w`
resized — Default **178×22**, also ein exakter /8-Downscale, der das native
8:1-Seitenverhältnis **ohne Verzerrung** beibehält (beide Dims durch
`dataset.patch_size`=2 teilbar). Beide Modelle sind rechteck-fähig; die
`num_classes`=`output_dim` rohen Outputs dienen direkt als (Om, S8)-Regressions-
Head unter `objective=mse`. Damit laufen beide jetzt end-to-end.
