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
  echo "    SWEEP_SUFFIX=_stage1     required (no unstaged sweep exists to fall back to)." >&2
  echo "                             Selects the staged-sweep variant for every submitted" >&2
  echo "                             model: _stage1 (architecture) or _stage2_mse/_stage2_score" >&2
  echo "                             (reduction + model-specific params, Stage-1 winner pinned)." >&2
  echo "                             See README \"Staged Hyperparameter Search\". _stage1 is the" >&2
  echo "                             same config for both objectives but still needs one call" >&2
  echo "                             per OBJECTIVE (different loss/output_dim, see below)." >&2
  echo "" >&2
  echo "  Image->vector reduction (avgpool|conv) is swept per trial by Optuna for" >&2
  echo "  MLP models, not fixed per job. kkan/kat use this same script for both stages," >&2
  echo "  just not in the default MODELS list (heavier, longer-running); add them via" >&2
  echo "  e.g. MODELS=\"kat kkan\" SWEEP_SUFFIX=_stage1, or _stage2_mse/_stage2_score for Stage 2." >&2
  exit 1
fi

EXPERIMENT="$1"
# Models to submit. Default: the MLP-style/reduction models only, see usage
# note above for adding kkan/kat back, e.g.
# MODELS="fastkan fasterkan efficientkan wavkan kkan kat" SWEEP_SUFFIX=_stage1.
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
