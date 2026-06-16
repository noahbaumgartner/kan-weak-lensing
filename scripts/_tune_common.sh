#!/usr/bin/env bash
# Common logic for per-model GPU tuning jobs. Sources expects $MODEL to be set.
set -euo pipefail

: "${MODEL:?MODEL must be set by the caller (e.g. MODEL=fastkan)}"
: "${EXPERIMENT:?EXPERIMENT must be set by the caller (MLflow experiment name)}"

# Optional overrides:
#   SWEEP     — Hydra sweep name including subgroup (default: image/tune_${MODEL}_wl)
#   DATASETS  — space-separated list of dataset names
#   OBJECTIVE — Versuch / training objective (e.g. score | mse). If set, passed
#               as objective=${OBJECTIVE}; otherwise the config.yaml default is used.
#   REDUCTION — image->vector reduction, avgpool | conv (default: avgpool).
#               Not swept — fixed for the whole job.
SWEEP="${SWEEP:-image/tune_${MODEL}_wl}"
OBJECTIVE_ARG=()
if [[ -n "${OBJECTIVE:-}" ]]; then
  OBJECTIVE_ARG=("objective=${OBJECTIVE}")
fi
REDUCTION="${REDUCTION:-avgpool}"
if [[ -n "${DATASETS:-}" ]]; then
  read -r -a DATASETS <<< "${DATASETS}"
else
  DATASETS=(
    weak_lensing
  )
fi

export PYTHONUNBUFFERED=1
module load uv/0.10.10
cd /cluster/home/baumgnoa/kan-weak-lensing

export UV_PROJECT_ENVIRONMENT=/cluster/home/baumgnoa/kan-weak-lensing/.venv

# Ensure a uv-managed Python is available (ships with headers Triton needs).
export UV_PYTHON_DOWNLOADS=manual
export UV_PYTHON_PREFERENCE=only-managed
uv python install 3.12

# If an existing venv is built against system Python, recreate it so Triton can find Python.h.
if [[ -d "${UV_PROJECT_ENVIRONMENT}" ]]; then
  base_prefix=$("${UV_PROJECT_ENVIRONMENT}/bin/python" -c "import sys; print(sys.base_prefix)" 2>/dev/null || echo "")
  if [[ "${base_prefix}" != *"/uv/python/"* ]]; then
    echo "Recreating venv with uv-managed Python (was: ${base_prefix:-missing})"
    rm -rf "${UV_PROJECT_ENVIRONMENT}"
  fi
fi

uv sync

# Wait for the mlflow server job to publish its URL.
URL_FILE=/cluster/home/baumgnoa/kan-weak-lensing/mlflow/server_url.txt
for i in $(seq 1 60); do
  if [[ -s "${URL_FILE}" ]]; then break; fi
  echo "Waiting for MLflow URL at ${URL_FILE} (${i}/60)..."
  sleep 10
done
if [[ ! -s "${URL_FILE}" ]]; then
  echo "MLflow URL file never appeared; is the mlflow_server job running?" >&2
  exit 1
fi
export MLFLOW_TRACKING_URI=$(cat "${URL_FILE}")
echo "Using MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI}"

for dataset in "${DATASETS[@]}"; do
  echo "=== Tuning ${MODEL} on ${dataset} ==="
  per_dataset_extra="EXTRA_${dataset}"
  extra=()
  if [[ -n "${!per_dataset_extra:-}" ]]; then
    # shellcheck disable=SC2206
    extra=(${!per_dataset_extra})
  fi
  HYDRA_FULL_ERROR=1 uv run main.py --multirun \
    +sweep="${SWEEP}" \
    dataset="${dataset}" \
    dataset.reduction="${REDUCTION}" \
    "${OBJECTIVE_ARG[@]}" \
    "${extra[@]}" \
    mlflow_tracking_uri="${MLFLOW_TRACKING_URI}" \
    +experiment="${EXPERIMENT}"
done

echo "=== ${MODEL} tuning complete ==="
