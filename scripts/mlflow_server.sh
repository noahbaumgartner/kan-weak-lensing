#!/usr/bin/env bash
# Standalone MLflow tracking server. Mirrors scripts/mlflow_server.submit
# but runs locally instead of under SLURM. Advertises its URL via a shared
# file so that per-model tuning jobs can discover it.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONUNBUFFERED=1
uv sync

MLFLOW_HOST=$(hostname -f)
MLFLOW_PORT=9299
MLFLOW_DIR="${REPO_ROOT}/mlflow"
URL_FILE=${MLFLOW_DIR}/server_url.txt
mkdir -p "${MLFLOW_DIR}"

TRACKING_URI="http://${MLFLOW_HOST}:${MLFLOW_PORT}"
echo "${TRACKING_URI}" > "${URL_FILE}"
echo "Wrote tracking URI to ${URL_FILE}: ${TRACKING_URI}"

export MLFLOW_SERVER_ALLOWED_HOSTS="*"

# Clean up URL file on exit so stale URLs don't mislead workers.
trap 'rm -f "${URL_FILE}"' EXIT

exec uv run mlflow server \
  --host 0.0.0.0 \
  --port ${MLFLOW_PORT} \
  --backend-store-uri "sqlite:///${MLFLOW_DIR}/mlflow.db" \
  --default-artifact-root "${MLFLOW_DIR}/artifacts"
