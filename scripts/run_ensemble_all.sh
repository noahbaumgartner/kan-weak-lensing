#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <experiment_name>" >&2
  echo "  Submits one SLURM bagging-ensemble job per configs/ensemble/*.yaml recipe" >&2
  echo "  (each recipe pins one architecture's best-MSE Stage-2 winner, see README" >&2
  echo "  \"Ensemble-Versuch im Detail\"). experiment_name is the MLflow experiment" >&2
  echo "  for all runs." >&2
  echo "" >&2
  echo "  Env overrides:" >&2
  echo "    CONFIGS=\"efficientkan fastkan\"  which configs/ensemble/*.yaml to submit" >&2
  echo "                                     (default: all files in configs/ensemble/)" >&2
  exit 1
fi

EXPERIMENT="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENSEMBLE_DIR="${SCRIPT_DIR}/../configs/ensemble"

if [[ -n "${CONFIGS:-}" ]]; then
  read -r -a CONFIG_LIST <<< "${CONFIGS}"
else
  CONFIG_LIST=()
  for f in "${ENSEMBLE_DIR}"/*.yaml; do
    CONFIG_LIST+=("$(basename "${f}" .yaml)")
  done
fi

if [[ ${#CONFIG_LIST[@]} -eq 0 ]]; then
  echo "No configs found under ${ENSEMBLE_DIR} (and CONFIGS was not set)." >&2
  exit 1
fi

for cfg in "${CONFIG_LIST[@]}"; do
  jobname="kan_ensemble_${cfg}"
  echo "Submitting ensemble run for ${cfg} as ${jobname} (experiment=${EXPERIMENT})..."
  sbatch --job-name="${jobname}" --export=ALL,CONFIG="${cfg}",EXPERIMENT="${EXPERIMENT}" \
    "${SCRIPT_DIR}/run_ensemble.submit"
done

echo "All ${#CONFIG_LIST[@]} ensemble jobs submitted. Check with: squeue -u \$USER"
