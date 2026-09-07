#!/bin/bash
#SBATCH --partition=milan-gpu
#SBATCH --job-name=bookshelf_dr_smoke
#SBATCH --output=/fred/oz430/tliu/logs/bookshelf_targeted_dr/%x_%j.out
#SBATCH --error=/fred/oz430/tliu/logs/bookshelf_targeted_dr/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:1

set -euo pipefail
REPO=/fred/oz430/tliu/code/bookshelf-targeted-dr-20260902
SIF=/fred/oz430/tliu/containers/isaac-lab-bookshelf.sif
CACHE=/fred/oz430/tliu/isaac_cache
CHECKPOINT=/fred/oz430/tliu/evaluation_models/bookshelf/residual_seed42/model.zip
PANDA_USD=/fred/oz430/tliu/assets/isaac_5_1/IsaacLab/Robots/FrankaEmika/panda_instanceable.usd
SMOKE_SEED=${BOOKSHELF_SMOKE_SEED:-9001}
SMOKE_NAME=${BOOKSHELF_SMOKE_NAME:-targeted_dr_20260902_smoke}
SMOKE_ITERATIONS=${BOOKSHELF_SMOKE_ITERATIONS:-1}
mkdir -p /fred/oz430/tliu/logs/bookshelf_targeted_dr
module load apptainer

apptainer exec --nv \
  -B /fred:/fred \
  -B "$CACHE/kit_cache:/isaac-sim/kit/cache" \
  -B "$CACHE/kit_data:/isaac-sim/kit/data" \
  -B "$CACHE/ov:/home/tliu/.cache/ov" \
  -B "$CACHE/nvidia:/home/tliu/.cache/nvidia" \
  -B "$CACHE/documents:/home/tliu/Documents" \
  "$SIF" bash -s -- "$REPO" "$CHECKPOINT" "$PANDA_USD" "$SMOKE_SEED" "$SMOKE_NAME" "$SMOKE_ITERATIONS" <<'INNER'
set -euo pipefail
REPO=$1
CHECKPOINT=$2
PANDA_USD=$3
SMOKE_SEED=$4
SMOKE_NAME=$5
SMOKE_ITERATIONS=$6
export ACCEPT_EULA=Y HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 HOME=/home/tliu
export PYTHONPATH="$REPO/source/bookshelf:${PYTHONPATH:-}"
cd "$REPO"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /isaac-sim/python.sh -m pytest \
  source/bookshelf/test/test_targeted_dr_profiles.py \
  source/bookshelf/test/test_xarm7_residual_task_static.py -q
/isaac-sim/python.sh scripts/sb3/train.py \
  --task Bookshelf-Residual-Direct-v0 \
  --checkpoint "$CHECKPOINT" \
  --resume \
  --robot_usd_path "$PANDA_USD" \
  --targeted_dr_profile obs_moderate \
  --run_name "$SMOKE_NAME" \
  --num_envs 16 \
  --max_iterations "$SMOKE_ITERATIONS" \
  --seed "$SMOKE_SEED" \
  --headless
test -s "$REPO/logs/sb3/Bookshelf-Residual-Direct-v0/$SMOKE_NAME/model.zip"
test -s "$REPO/logs/sb3/Bookshelf-Residual-Direct-v0/$SMOKE_NAME/model_vecnormalize.pkl"
INNER
