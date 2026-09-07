#!/bin/bash
#SBATCH --partition=milan-gpu
#SBATCH --job-name=bookshelf_july8_eval
#SBATCH --output=/fred/oz430/tliu/logs/bookshelf_targeted_dr/%x_%A_%a.out
#SBATCH --error=/fred/oz430/tliu/logs/bookshelf_targeted_dr/%x_%A_%a.err
#SBATCH --array=0-6
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1

set -euo pipefail
REPO=/fred/oz430/tliu/code/bookshelf-targeted-dr-20260902
SIF=/fred/oz430/tliu/containers/isaac-lab-bookshelf.sif
CACHE=/fred/oz430/tliu/isaac_cache
CHECKPOINT=${BOOKSHELF_EVAL_CHECKPOINT:-/fred/oz430/tliu/evaluation_models/bookshelf/residual_seed42/model.zip}
LABEL=${BOOKSHELF_EVAL_LABEL:-original_july8}
EPISODES=${BOOKSHELF_EVAL_EPISODES:-2048}
NUM_ENVS=${BOOKSHELF_EVAL_NUM_ENVS:-256}
SCENARIO_BANK=${BOOKSHELF_EVAL_SCENARIO_BANK:-}
PANDA_USD=/fred/oz430/tliu/assets/isaac_5_1/IsaacLab/Robots/FrankaEmika/panda_instanceable.usd
PROFILES=(nominal observation_moderate strong_tail physical_like gripper_only combined_moderate combined_moderate_actuation)
PROFILE=${PROFILES[$SLURM_ARRAY_TASK_ID]}
OUTPUT_ROOT=/fred/oz430/tliu/evaluation_results/bookshelf/targeted_dr_20260902/$LABEL
OUTPUT=$OUTPUT_ROOT/$PROFILE
mkdir -p /fred/oz430/tliu/logs/bookshelf_targeted_dr "$OUTPUT_ROOT"
module load apptainer

apptainer exec --nv \
  -B /fred:/fred \
  -B "$CACHE/kit_cache:/isaac-sim/kit/cache" \
  -B "$CACHE/kit_data:/isaac-sim/kit/data" \
  -B "$CACHE/ov:/home/tliu/.cache/ov" \
  -B "$CACHE/nvidia:/home/tliu/.cache/nvidia" \
  -B "$CACHE/documents:/home/tliu/Documents" \
  "$SIF" bash -s -- "$REPO" "$CHECKPOINT" "$PANDA_USD" "$PROFILE" "$OUTPUT" "$EPISODES" "$NUM_ENVS" "$SCENARIO_BANK" <<'INNER'
set -euo pipefail
REPO=$1
CHECKPOINT=$2
PANDA_USD=$3
PROFILE=$4
OUTPUT=$5
EPISODES=$6
NUM_ENVS=$7
SCENARIO_BANK=$8
SCENARIO_ARGS=()
if [[ -n "$SCENARIO_BANK" ]]; then
  SCENARIO_ARGS=(--eval_scenario_bank "$SCENARIO_BANK")
fi
export ACCEPT_EULA=Y HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 HOME=/home/tliu
export PYTHONPATH="$REPO/source/bookshelf:${PYTHONPATH:-}"
cd "$REPO"
/isaac-sim/python.sh scripts/sb3/play.py \
  --task Bookshelf-Residual-Direct-v0 \
  --checkpoint "$CHECKPOINT" \
  --robot_usd_path "$PANDA_USD" \
  --evaluation_dr_profile "$PROFILE" \
  --eval_slot_clearance 0.003 \
  --eval_episodes "$EPISODES" \
  --eval_output_dir "$OUTPUT" \
  "${SCENARIO_ARGS[@]}" \
  --num_envs "$NUM_ENVS" \
  --seed 4200 \
  --headless
test -s "$OUTPUT/policy_robustness_metrics.csv"
INNER
