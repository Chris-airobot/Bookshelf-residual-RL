#!/bin/bash
#SBATCH --partition=milan-gpu
#SBATCH --job-name=bookshelf_select
#SBATCH --output=/fred/oz430/tliu/logs/bookshelf_model_selection/%x_%A_%a.out
#SBATCH --error=/fred/oz430/tliu/logs/bookshelf_model_selection/%x_%A_%a.err
#SBATCH --array=0-62%24
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --gres=gpu:1

set -euo pipefail
REPO=/fred/oz430/tliu/code/bookshelf-targeted-dr-20260902
SIF=/fred/oz430/tliu/containers/isaac-lab-bookshelf.sif
CACHE=/fred/oz430/tliu/isaac_cache
MANIFEST=$REPO/scripts/hpc/bookshelf_model_selection_manifest.tsv
PANDA_USD=/fred/oz430/tliu/assets/isaac_5_1/IsaacLab/Robots/FrankaEmika/panda_instanceable.usd
BANK_ROOT=/fred/oz430/tliu/evaluation_models/bookshelf
RESULT_ROOT=/fred/oz430/tliu/evaluation_results/bookshelf/model_selection_20260902
PROFILES=(nominal physical_like gripper_only observation_moderate combined_moderate strong_tail combined_moderate_actuation)
BANKS=(bookshelf_selection_nominal_512.json bookshelf_selection_nominal_512.json bookshelf_selection_nominal_512.json bookshelf_selection_nominal_512.json bookshelf_selection_combined_512.json bookshelf_selection_strong_512.json bookshelf_selection_combined_512.json)

POLICY_INDEX=$((SLURM_ARRAY_TASK_ID / 3))
PROFILE_GROUP=$((SLURM_ARRAY_TASK_ID % 3))
IFS=$'\t' read -r LABEL CHECKPOINT < <(sed -n "$((POLICY_INDEX + 1))p" "$MANIFEST")
mkdir -p /fred/oz430/tliu/logs/bookshelf_model_selection "$RESULT_ROOT/$LABEL"
test -s "$CHECKPOINT"
CHECKPOINT_DIR=${CHECKPOINT%/*}
CHECKPOINT_NAME=${CHECKPOINT##*/}
VECNORMALIZE=$CHECKPOINT_DIR/${CHECKPOINT_NAME/model/model_vecnormalize}
VECNORMALIZE=${VECNORMALIZE%.zip}.pkl
test -s "$VECNORMALIZE"
module load apptainer

case "$PROFILE_GROUP" in
  0) PROFILE_INDICES=(0 1 2) ;;
  1) PROFILE_INDICES=(3 4) ;;
  2) PROFILE_INDICES=(5 6) ;;
esac

for PROFILE_INDEX in "${PROFILE_INDICES[@]}"; do
  PROFILE=${PROFILES[$PROFILE_INDEX]}
  SCENARIO_BANK=$BANK_ROOT/${BANKS[$PROFILE_INDEX]}
  OUTPUT=$RESULT_ROOT/$LABEL/$PROFILE
  TEMP_OUTPUT=$RESULT_ROOT/.partial_${LABEL}_${PROFILE}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
  if [[ -s "$OUTPUT/summary.json" && -s "$OUTPUT/policy_robustness_metrics.csv" ]]; then
    echo "[SKIP] completed output exists: $OUTPUT"
    continue
  fi
  test -s "$SCENARIO_BANK"
  apptainer exec --nv \
    -B /fred:/fred \
    -B "$CACHE/kit_cache:/isaac-sim/kit/cache" \
    -B "$CACHE/kit_data:/isaac-sim/kit/data" \
    -B "$CACHE/ov:/home/tliu/.cache/ov" \
    -B "$CACHE/nvidia:/home/tliu/.cache/nvidia" \
    -B "$CACHE/documents:/home/tliu/Documents" \
    "$SIF" bash -s -- "$REPO" "$CHECKPOINT" "$PANDA_USD" "$PROFILE" "$SCENARIO_BANK" "$TEMP_OUTPUT" <<'INNER'
set -euo pipefail
REPO=$1
CHECKPOINT=$2
PANDA_USD=$3
PROFILE=$4
SCENARIO_BANK=$5
OUTPUT=$6
export ACCEPT_EULA=Y HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 HOME=/home/tliu
export PYTHONPATH="$REPO/source/bookshelf:${PYTHONPATH:-}"
cd "$REPO"
/isaac-sim/python.sh scripts/sb3/play.py \
  --task Bookshelf-Residual-Direct-v0 \
  --checkpoint "$CHECKPOINT" \
  --robot_usd_path "$PANDA_USD" \
  --evaluation_dr_profile "$PROFILE" \
  --eval_slot_clearance 0.003 \
  --eval_scenario_bank "$SCENARIO_BANK" \
  --eval_episodes 512 \
  --eval_output_dir "$OUTPUT" \
  --num_envs 512 \
  --seed 20260902 \
  --headless
test -s "$OUTPUT/policy_robustness_metrics.csv"
INNER

  test ! -e "$OUTPUT"
  mv "$TEMP_OUTPUT" "$OUTPUT"
done
