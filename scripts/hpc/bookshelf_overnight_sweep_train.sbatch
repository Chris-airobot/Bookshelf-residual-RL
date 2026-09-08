#!/bin/bash
#SBATCH --partition=milan-gpu
#SBATCH --output=/fred/oz430/tliu/logs/bookshelf_overnight_sweep/%x_%j.out
#SBATCH --error=/fred/oz430/tliu/logs/bookshelf_overnight_sweep/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=18:00:00
#SBATCH --gres=gpu:1

set -euo pipefail

: "${BOOKSHELF_SWEEP_VARIANT:?set BOOKSHELF_SWEEP_VARIANT}"
: "${BOOKSHELF_SWEEP_MODE:?set BOOKSHELF_SWEEP_MODE (finetune|scratch)}"
: "${BOOKSHELF_SWEEP_SEED:?set BOOKSHELF_SWEEP_SEED}"
: "${BOOKSHELF_SWEEP_RUNTAG:?set BOOKSHELF_SWEEP_RUNTAG}"

case "$BOOKSHELF_SWEEP_VARIANT" in
  july|lat4|wall05|lat4_yaw075|quad_match|sc_lat4|lat4_sat001|wall_depth)
    ;;
  *)
    echo "ERROR: Unknown BOOKSHELF_SWEEP_VARIANT '$BOOKSHELF_SWEEP_VARIANT'" >&2
    exit 1
    ;;
esac

if [ "$BOOKSHELF_SWEEP_VARIANT" = "sc_lat4" ] && [ "$BOOKSHELF_SWEEP_MODE" != "scratch" ]; then
  echo "ERROR: Variant sc_lat4 requires BOOKSHELF_SWEEP_MODE=scratch (got '$BOOKSHELF_SWEEP_MODE')" >&2
  exit 1
fi
if [ "$BOOKSHELF_SWEEP_VARIANT" != "sc_lat4" ] && [ "$BOOKSHELF_SWEEP_MODE" != "finetune" ]; then
  echo "ERROR: Variant '$BOOKSHELF_SWEEP_VARIANT' requires BOOKSHELF_SWEEP_MODE=finetune (got '$BOOKSHELF_SWEEP_MODE')" >&2
  exit 1
fi

REPO=/fred/oz430/tliu/code/bookshelf-targeted-dr-20260902
SIF=/fred/oz430/tliu/containers/isaac-lab-bookshelf.sif
CACHE=/fred/oz430/tliu/isaac_cache
PANDA_USD=/fred/oz430/tliu/assets/isaac_5_1/IsaacLab/Robots/FrankaEmika/panda_instanceable.usd
DEFAULT_CHECKPOINT=/fred/oz430/tliu/evaluation_models/bookshelf/residual_seed42/model.zip

mkdir -p /fred/oz430/tliu/logs/bookshelf_overnight_sweep
module load apptainer

DATE_TAG="$(date +%Y%m%d)"
RUN_NAME="overnight_${DATE_TAG}_${BOOKSHELF_SWEEP_RUNTAG}"
OUT="$REPO/logs/sb3/Bookshelf-Residual-Direct-v0/$RUN_NAME"

# Never overwrite existing run directory
if [ -e "$OUT" ]; then
  echo "ERROR: Target run directory already exists: $OUT" >&2
  exit 1
fi

CHECKPOINT=""
if [ "$BOOKSHELF_SWEEP_MODE" = "finetune" ]; then
  CHECKPOINT="${BOOKSHELF_SWEEP_CHECKPOINT:-$DEFAULT_CHECKPOINT}"
  VECNORM="${CHECKPOINT/model.zip/model_vecnormalize.pkl}"

  if [ "$CHECKPOINT" = "$DEFAULT_CHECKPOINT" ]; then
    test "$(sha256sum "$CHECKPOINT" | cut -d' ' -f1)" = "80f7aa2d6675a99f3965b2479bc0b62f5f3320e724a6f3399efacc1640b3b4ed"
    test "$(sha256sum "$VECNORM" | cut -d' ' -f1)" = "88670e59194fa5d70743872ea02232acdbea93f827fbce116d1fcb4a0745a635"
  else
    if [ -n "${BOOKSHELF_SWEEP_CHECKPOINT_SHA:-}" ]; then
      test "$(sha256sum "$CHECKPOINT" | cut -d' ' -f1)" = "$BOOKSHELF_SWEEP_CHECKPOINT_SHA"
    fi
    if [ -n "${BOOKSHELF_SWEEP_VECNORM_SHA:-}" ]; then
      test "$(sha256sum "$VECNORM" | cut -d' ' -f1)" = "$BOOKSHELF_SWEEP_VECNORM_SHA"
    fi
  fi
elif [ "$BOOKSHELF_SWEEP_MODE" != "scratch" ]; then
  echo "ERROR: Unknown BOOKSHELF_SWEEP_MODE '$BOOKSHELF_SWEEP_MODE' (must be finetune or scratch)" >&2
  exit 1
fi

apptainer exec --nv \
  -B /fred:/fred \
  -B "$CACHE/kit_cache:/isaac-sim/kit/cache" \
  -B "$CACHE/kit_data:/isaac-sim/kit/data" \
  -B "$CACHE/ov:/home/tliu/.cache/ov" \
  -B "$CACHE/nvidia:/home/tliu/.cache/nvidia" \
  -B "$CACHE/documents:/home/tliu/Documents" \
  "$SIF" bash -s -- "$REPO" "$PANDA_USD" "$RUN_NAME" "$BOOKSHELF_SWEEP_MODE" "$BOOKSHELF_SWEEP_VARIANT" "$BOOKSHELF_SWEEP_SEED" "$CHECKPOINT" <<'INNER'
set -euo pipefail
REPO=$1
PANDA_USD=$2
RUN_NAME=$3
MODE=$4
VARIANT=$5
SEED=$6
CHECKPOINT=$7

export ACCEPT_EULA=Y HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 HOME=/home/tliu
export PYTHONPATH="$REPO/source/bookshelf:${PYTHONPATH:-}"
cd "$REPO"

OUT="$REPO/logs/sb3/Bookshelf-Residual-Direct-v0/$RUN_NAME"

if [ "$MODE" = "finetune" ]; then
  /isaac-sim/python.sh scripts/sb3/train.py \
    --task Bookshelf-Residual-Direct-v0 \
    --checkpoint "$CHECKPOINT" \
    --resume \
    --robot_usd_path "$PANDA_USD" \
    --overnight_variant "$VARIANT" \
    --run_name "$RUN_NAME" \
    --num_envs 256 \
    --max_iterations 2442 \
    --finetune_checkpoint_milestones 1000000,2500000,5000000,10000000,15000000,20000000 \
    --log_interval 100000 \
    --seed "$SEED" \
    --headless \
    agent.learning_rate=5e-5 agent.clip_range=0.1 agent.n_epochs=5 agent.ent_coef=0.003

  test -s "$OUT/model.zip"
  test -s "$OUT/model_vecnormalize.pkl"
  for m in 1000000 2500000 5000000 10000000 15000000 20000000; do
    test -s "$OUT/model_finetune_${m}_steps.zip"
    test -s "$OUT/model_vecnormalize_finetune_${m}_steps.pkl"
  done
elif [ "$MODE" = "scratch" ]; then
  /isaac-sim/python.sh scripts/sb3/train.py \
    --task Bookshelf-Residual-Direct-v0 \
    --robot_usd_path "$PANDA_USD" \
    --overnight_variant "$VARIANT" \
    --run_name "$RUN_NAME" \
    --num_envs 256 \
    --max_iterations 3663 \
    --training_checkpoint_interval_steps 10000000 \
    --log_interval 100000 \
    --seed "$SEED" \
    --headless

  test -s "$OUT/model.zip"
  test -s "$OUT/model_vecnormalize.pkl"
  for m in 10000000 20000000 30000000; do
    test -s "$OUT/model_fresh_${m}_steps.zip"
  done
fi
INNER
