#!/usr/bin/env bash
# Smoke-test: train every model briefly to check it runs end-to-end locally.
#
#   * MLP-style KANs (fastkan, fasterkan, efficientkan, wavkan) are run once per
#     reduction (avgpool, conv, none) -> 4 models x 3 reductions.
#   * kkan + kat are run once each WITHOUT a reduction (they consume the image
#     directly). NOTE: per README "Hinweis zu kkan / kat" these are still the
#     kan-lab classifiers and are NOT yet adapted for the 1424x176 weak-lensing
#     maps (square input / dataset.img_size / regression head). They are
#     included here as-is and are EXPECTED to FAIL until adapted.
#
# Every run uses a short training budget (2 epochs by default). Runs continue
# on error; a PASS/FAIL summary is printed at the end and the script exits
# non-zero if anything failed.
#
# Knobs (env vars, all optional):
#   OBJECTIVE  Versuch to test (default: mse)
#   EPOCHS     epochs per run    (default: 2)
#   BATCH_SIZE batch size        (default: 16 — small so wavkan/none, which
#              expands the full 1424x176 input, fits in GPU memory)
#   DATASET    dataset config    (default: weak_lensing)
#   EXPERIMENT MLflow experiment (default: smoke_test)
#   MLFLOW_TRACKING_URI  default: local file store ./mlruns
#
# Any extra args are forwarded verbatim to every `main.py` call, e.g.
#   ./scripts/smoke_test_all.sh training.batch_size=64
#
# Usage:
#   ./scripts/smoke_test_all.sh
#   OBJECTIVE=score EPOCHS=1 ./scripts/smoke_test_all.sh

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

DATASET="${DATASET:-weak_lensing}"
OBJECTIVE="${OBJECTIVE:-mse}"
EPOCHS="${EPOCHS:-2}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EXPERIMENT="${EXPERIMENT:-smoke_test}"
TRACKING_URI="${MLFLOW_TRACKING_URI:-file://${REPO_ROOT}/mlruns}"

MLP_MODELS=(fastkan fasterkan efficientkan wavkan)
REDUCTIONS=(avgpool conv none)
IMAGE_MODELS=(kkan kat)

EXTRA_ARGS=("$@")
results=()

run_one() {
  local label="$1"; shift
  echo ""
  echo "=============================================================="
  echo ">>> ${label}"
  echo "=============================================================="
  if uv run python main.py \
      "$@" \
      dataset="${DATASET}" \
      objective="${OBJECTIVE}" \
      training=adam \
      training.epochs="${EPOCHS}" \
      training.batch_size="${BATCH_SIZE}" \
      +experiment="${EXPERIMENT}" \
      mlflow_tracking_uri="${TRACKING_URI}" \
      "${EXTRA_ARGS[@]}"; then
    results+=("PASS  ${label}")
  else
    results+=("FAIL  ${label}")
  fi
}

echo "Smoke test: objective=${OBJECTIVE}, epochs=${EPOCHS}, dataset=${DATASET}"
echo "MLflow tracking URI: ${TRACKING_URI}"

# --- MLP-style KANs x every reduction ---
for model in "${MLP_MODELS[@]}"; do
  for red in "${REDUCTIONS[@]}"; do
    run_one "${model} / reduction=${red}" \
      "model=${model}" "dataset.reduction=${red}"
  done
done

# --- image models (kkan, kat), no reduction, as currently implemented ---
for model in "${IMAGE_MODELS[@]}"; do
  run_one "${model} / no reduction (as-is, may fail)" \
    "model=${model}" "dataset.reduction=none"
done

# --- summary ---
echo ""
echo "=============================================================="
echo " SMOKE TEST SUMMARY (objective=${OBJECTIVE}, epochs=${EPOCHS})"
echo "=============================================================="
for r in "${results[@]}"; do echo "  ${r}"; done

if printf '%s\n' "${results[@]}" | grep -q '^FAIL'; then
  echo ""
  echo "Some runs FAILED (see above)."
  exit 1
fi
echo ""
echo "All runs passed."
