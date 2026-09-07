#!/bin/bash
#SBATCH --partition=milan-gpu
#SBATCH --output=/fred/oz430/tliu/logs/bookshelf_targeted_dr/%x_%j.out
#SBATCH --error=/fred/oz430/tliu/logs/bookshelf_targeted_dr/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1

set -euo pipefail

: "${BOOKSHELF_DR_VARIANT:?set BOOKSHELF_DR_VARIANT}"
: "${BOOKSHELF_DR_PROFILE:?set BOOKSHELF_DR_PROFILE}"
: "${BOOKSHELF_DR_SEED:?set BOOKSHELF_DR_SEED}"

REPO=/fred/oz430/tliu/code/bookshelf-targeted-dr-20260902
SIF=/fred/oz430/tliu/containers/isaac-lab-bookshelf.sif
CACHE=/fred/oz430/tliu/isaac_cache
CHECKPOINT=/fred/oz430/tliu/evaluation_models/bookshelf/residual_seed42/model.zip
PANDA_USD=/fred/oz430/tliu/assets/isaac_5_1/IsaacLab/Robots/FrankaEmika/panda_instanceable.usd

mkdir -p /fred/oz430/tliu/logs/bookshelf_targeted_dr
module load apptainer

test "$(sha256sum "$CHECKPOINT" | cut -d' ' -f1)" = "80f7aa2d6675a99f3965b2479bc0b62f5f3320e724a6f3399efacc1640b3b4ed"
test "$(sha256sum "${CHECKPOINT/model.zip/model_vecnormalize.pkl}" | cut -d' ' -f1)" = "88670e59194fa5d70743872ea02232acdbea93f827fbce116d1fcb4a0745a635"

apptainer exec --nv \
  -B /fred:/fred \
  -B "$CACHE/kit_cache:/isaac-sim/kit/cache" \
  -B "$CACHE/kit_data:/isaac-sim/kit/data" \
  -B "$CACHE/ov:/home/tliu/.cache/ov" \
  -B "$CACHE/nvidia:/home/tliu/.cache/nvidia" \
  -B "$CACHE/documents:/home/tliu/Documents" \
  "$SIF" bash -s -- "$REPO" "$CHECKPOINT" "$PANDA_USD" "$BOOKSHELF_DR_VARIANT" "$BOOKSHELF_DR_PROFILE" "$BOOKSHELF_DR_SEED" <<'INNER'
set -euo pipefail
REPO=$1
CHECKPOINT=$2
PANDA_USD=$3
VARIANT=$4
PROFILE=$5
SEED=$6
export ACCEPT_EULA=Y HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1
export HOME=/home/tliu
export PYTHONPATH="$REPO/source/bookshelf:${PYTHONPATH:-}"
cd "$REPO"
/isaac-sim/python.sh scripts/sb3/train.py \
  --task Bookshelf-Residual-Direct-v0 \
  --checkpoint "$CHECKPOINT" \
  --resume \
  --robot_usd_path "$PANDA_USD" \
  --targeted_dr_profile "$PROFILE" \
  --run_name "targeted_dr_20260902_${VARIANT}" \
  --num_envs 256 \
  --max_iterations 1221 \
  --finetune_checkpoint_interval_steps 2500000 \
  --log_interval 100000 \
  --seed "$SEED" \
  --headless
test -s "$REPO/logs/sb3/Bookshelf-Residual-Direct-v0/targeted_dr_20260902_${VARIANT}/model.zip"
test -s "$REPO/logs/sb3/Bookshelf-Residual-Direct-v0/targeted_dr_20260902_${VARIANT}/model_vecnormalize.pkl"
INNER
