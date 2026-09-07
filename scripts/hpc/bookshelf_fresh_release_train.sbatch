#!/bin/bash
#SBATCH --partition=milan-gpu
#SBATCH --output=/fred/oz430/tliu/logs/bookshelf_fresh_release/%x_%j.out
#SBATCH --error=/fred/oz430/tliu/logs/bookshelf_fresh_release/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=14:00:00
#SBATCH --gres=gpu:1

set -euo pipefail

: "${BOOKSHELF_FRESH_VARIANT:?set BOOKSHELF_FRESH_VARIANT}"
: "${BOOKSHELF_FRESH_PROFILE:?set BOOKSHELF_FRESH_PROFILE}"
: "${BOOKSHELF_RELEASE_OBJECTIVE:?set BOOKSHELF_RELEASE_OBJECTIVE}"
: "${BOOKSHELF_FRESH_SEED:?set BOOKSHELF_FRESH_SEED}"

REPO=/fred/oz430/tliu/code/bookshelf-targeted-dr-20260902
SIF=/fred/oz430/tliu/containers/isaac-lab-bookshelf.sif
CACHE=/fred/oz430/tliu/isaac_cache
PANDA_USD=/fred/oz430/tliu/assets/isaac_5_1/IsaacLab/Robots/FrankaEmika/panda_instanceable.usd
RUN_NAME=fresh_release_20260902_${BOOKSHELF_FRESH_VARIANT}

mkdir -p /fred/oz430/tliu/logs/bookshelf_fresh_release
module load apptainer

apptainer exec --nv \
  -B /fred:/fred \
  -B "$CACHE/kit_cache:/isaac-sim/kit/cache" \
  -B "$CACHE/kit_data:/isaac-sim/kit/data" \
  -B "$CACHE/ov:/home/tliu/.cache/ov" \
  -B "$CACHE/nvidia:/home/tliu/.cache/nvidia" \
  -B "$CACHE/documents:/home/tliu/Documents" \
  "$SIF" bash -s -- "$REPO" "$PANDA_USD" "$RUN_NAME" "$BOOKSHELF_FRESH_PROFILE" "$BOOKSHELF_RELEASE_OBJECTIVE" "$BOOKSHELF_FRESH_SEED" <<'INNER'
set -euo pipefail
REPO=$1
PANDA_USD=$2
RUN_NAME=$3
PROFILE=$4
OBJECTIVE=$5
SEED=$6
export ACCEPT_EULA=Y HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 HOME=/home/tliu
export PYTHONPATH="$REPO/source/bookshelf:${PYTHONPATH:-}"
cd "$REPO"
/isaac-sim/python.sh scripts/sb3/train.py \
  --task Bookshelf-Residual-Direct-v0 \
  --robot_usd_path "$PANDA_USD" \
  --targeted_dr_profile "$PROFILE" \
  --release_training_objective "$OBJECTIVE" \
  --targeted_dr_curriculum_steps 30000000 \
  --run_name "$RUN_NAME" \
  --num_envs 256 \
  --max_iterations 4883 \
  --training_checkpoint_interval_steps 10000000 \
  --log_interval 100000 \
  --seed "$SEED" \
  --headless

OUT="$REPO/logs/sb3/Bookshelf-Residual-Direct-v0/$RUN_NAME"
test -s "$OUT/model.zip"
test -s "$OUT/model_vecnormalize.pkl"
test -s "$OUT/model_fresh_10000000_steps.zip"
test -s "$OUT/model_fresh_20000000_steps.zip"
test -s "$OUT/model_fresh_30000000_steps.zip"
INNER
