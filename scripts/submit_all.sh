#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <experiment_name>" >&2
  echo "  Submits one SLURM tuning job per KAN model on the weak_lensing dataset." >&2
  echo "  experiment_name is the MLflow experiment for all runs." >&2
  echo "" >&2
  echo "  Env overrides (forwarded to the jobs via --export=ALL):" >&2
  echo "    MODELS=\"fastkan wavkan\"  models to submit (default: fastkan fasterkan efficientkan wavkan)" >&2
  echo "    OBJECTIVE=mse|score      training objective (default: config.yaml = score)" >&2
  echo "    REDUCTION=avgpool|conv   image->vector reduction for MLP models (default: avgpool)" >&2
  echo "" >&2
  echo "  Reduction is fixed per job, not swept. kkan/kat are excluded by default" >&2
  echo "  (not yet adapted to weak lensing, tend to OOM); add them back via MODELS=." >&2
  exit 1
fi

EXPERIMENT="$1"
# Models to submit. Default: the MLP-style models only (kkan/kat are not yet
# adapted to weak lensing and tend to OOM). Add them back with e.g.
# MODELS="fastkan fasterkan efficientkan wavkan kkan kat".
if [[ -n "${MODELS:-}" ]]; then
  read -r -a MODEL_LIST <<< "${MODELS}"
else
  MODEL_LIST=(fastkan fasterkan efficientkan wavkan)
fi
JOBS=()
for m in "${MODEL_LIST[@]}"; do
  JOBS+=("tune_${m}.submit")
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for job_name in "${JOBS[@]}"; do
  job="${SCRIPT_DIR}/${job_name}"
  # tune_fastkan.submit -> fastkan; tag the SLURM job name with the objective
  # (score | mse | default) so squeue distinguishes runs of different Versuche.
  model="${job_name#tune_}"; model="${model%.submit}"
  jobname="kan_tune_${model}_${OBJECTIVE:-default}"
  echo "Submitting ${job} as ${jobname} (experiment=${EXPERIMENT})..."
  sbatch --job-name="${jobname}" --export=ALL,EXPERIMENT="${EXPERIMENT}" "${job}"
done

echo "All ${#JOBS[@]} weak-lensing tuning jobs submitted. Check with: squeue -u \$USER"
