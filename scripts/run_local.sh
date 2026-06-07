#!/usr/bin/env bash
# Run a weak-lensing KAN experiment locally without SLURM / a remote MLflow.
#
# By default, trains fastkan on weak_lensing and logs to a local MLflow file
# store (./mlruns), so no `mlflow server` is needed.
#
# Usage:
#   ./scripts/run_local.sh                                 # fastkan on weak_lensing
#   MODEL=fasterkan ./scripts/run_local.sh
#   ./scripts/run_local.sh training.epochs=5 model.num_grids=16
#
# Any extra args are forwarded to Hydra. Set MLFLOW_TRACKING_URI to log to a
# running server instead of the local file store.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

MODEL="${MODEL:-fastkan}"
DATASET="${DATASET:-weak_lensing}"
EXPERIMENT="${EXPERIMENT:-weak_lensing_local}"
TRACKING_URI="${MLFLOW_TRACKING_URI:-file://${REPO_ROOT}/mlruns}"

echo "Running ${MODEL} on ${DATASET} (experiment=${EXPERIMENT})"
echo "MLflow tracking URI: ${TRACKING_URI}"

exec uv run python main.py \
  model="${MODEL}" \
  dataset="${DATASET}" \
  training=adam \
  experiment="${EXPERIMENT}" \
  mlflow_tracking_uri="${TRACKING_URI}" \
  "$@"
