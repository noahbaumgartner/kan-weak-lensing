#!/usr/bin/env bash
# Run a weak-lensing KAN experiment locally without SLURM / a remote MLflow.
#
# Logs to a local MLflow file store (./mlruns) by default, so no `mlflow server`
# is needed. Set MLFLOW_TRACKING_URI to log to a running server instead.
#
# Knobs (all optional, set as env vars):
#   MODEL      single-model architecture (fastkan | fasterkan | efficientkan | wavkan | kkan | kat)
#   OBJECTIVE  Versuch (score | mse | ensemble)
#   REDUCTION  image->vector reduction, avgpool | kymatio | conv | powerspectrum (default: avgpool)
#   DATASET    dataset config name (default: weak_lensing)
#
# IMPORTANT: for objective=ensemble do NOT set MODEL — the ensemble objective
# selects the model itself; an explicit MODEL would override it.
#
# Usage:
#   ./scripts/run_local.sh                                  # default model + objective (fastkan / score)
#   MODEL=fasterkan ./scripts/run_local.sh                  # single fasterkan, score
#   MODEL=wavkan OBJECTIVE=mse ./scripts/run_local.sh       # single wavkan, MSE regression
#   OBJECTIVE=ensemble ./scripts/run_local.sh               # deep ensemble (8 members)
#   OBJECTIVE=ensemble ./scripts/run_local.sh model.n_members=12
#   ./scripts/run_local.sh training.epochs=5                # forward any Hydra override
#
# Any extra args are forwarded to Hydra.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

DATASET="${DATASET:-weak_lensing}"
REDUCTION="${REDUCTION:-avgpool}"
EXPERIMENT="${EXPERIMENT:-weak_lensing_local}"
TRACKING_URI="${MLFLOW_TRACKING_URI:-file://${REPO_ROOT}/mlruns}"

# Only pass model= / objective= when explicitly set, so they don't clobber the
# objective-driven model selection (e.g. objective=ensemble).
extra_args=()
if [[ -n "${MODEL:-}" ]]; then
  extra_args+=("model=${MODEL}")
fi
if [[ -n "${OBJECTIVE:-}" ]]; then
  extra_args+=("objective=${OBJECTIVE}")
fi

echo "Running ${MODEL:-<default model>} / objective=${OBJECTIVE:-<default>} on ${DATASET} (experiment=${EXPERIMENT})"
echo "MLflow tracking URI: ${TRACKING_URI}"

exec uv run python main.py \
  "${extra_args[@]}" \
  dataset="${DATASET}" \
  dataset.reduction="${REDUCTION}" \
  training=adam \
  experiment="${EXPERIMENT}" \
  mlflow_tracking_uri="${TRACKING_URI}" \
  "$@"
