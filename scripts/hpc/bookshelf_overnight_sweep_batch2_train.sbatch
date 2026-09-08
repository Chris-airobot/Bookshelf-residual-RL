#!/bin/bash
#SBATCH --partition=milan-gpu
#SBATCH --output=/fred/oz430/tliu/logs/bookshelf_overnight_sweep/%x_%j.out
#SBATCH --error=/fred/oz430/tliu/logs/bookshelf_overnight_sweep/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1

set -euo pipefail

: "${BOOKSHELF_SWEEP_VARIANT:?set BOOKSHELF_SWEEP_VARIANT}"
: "${BOOKSHELF_SWEEP_SEED:?set BOOKSHELF_SWEEP_SEED}"
: "${BOOKSHELF_SWEEP_RUNTAG:?set BOOKSHELF_SWEEP_RUNTAG}"

case "$BOOKSHELF_SWEEP_VARIANT" in
  lat4|wall05|lat4_yaw075|quad_match|wall_depth|lat4_sat001|july)
    echo "ERROR: Variant '$BOOKSHELF_SWEEP_VARIANT' is a fine-tune variant; batch-2 is scratch-only; use bookshelf_overnight_sweep_train.sbatch for finetune" >&2
    exit 1
    ;;
  sc_lat4|sc_lat4_yaw075|sc_wall05|sc_quad_match|sc_lat2_softwall_dr|sc_wall05_ramp|sc_lat4_sat001)
    ;;
  *)
    echo "ERROR: Unknown BOOKSHELF_SWEEP_VARIANT '$BOOKSHELF_SWEEP_VARIANT'" >&2
    exit 1
    ;;
esac

REPO=/fred/oz430/tliu/code/bookshelf-targeted-dr-20260902
SIF=/fred/oz430/tliu/containers/isaac-lab-bookshelf.sif
CACHE=/fred/oz430/tliu/isaac_cache
PANDA_USD=/fred/oz430/tliu/assets/isaac_5_1/IsaacLab/Robots/FrankaEmika/panda_instanceable.usd

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

apptainer exec --nv \
  -B /fred:/fred \
  -B "$CACHE/kit_cache:/isaac-sim/kit/cache" \
  -B "$CACHE/kit_data:/isaac-sim/kit/data" \
  -B "$CACHE/ov:/home/tliu/.cache/ov" \
  -B "$CACHE/nvidia:/home/tliu/.cache/nvidia" \
  -B "$CACHE/documents:/home/tliu/Documents" \
  "$SIF" bash -s -- "$REPO" "$PANDA_USD" "$RUN_NAME" "$BOOKSHELF_SWEEP_VARIANT" "$BOOKSHELF_SWEEP_SEED" <<'INNER'
set -euo pipefail
REPO=$1
PANDA_USD=$2
RUN_NAME=$3
VARIANT=$4
SEED=$5

export ACCEPT_EULA=Y HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 HOME=/home/tliu
export PYTHONPATH="$REPO/source/bookshelf:${PYTHONPATH:-}"
cd "$REPO"

OUT="$REPO/logs/sb3/Bookshelf-Residual-Direct-v0/$RUN_NAME"

/isaac-sim/python.sh scripts/sb3/train.py \
  --task Bookshelf-Residual-Direct-v0 \
  --robot_usd_path "$PANDA_USD" \
  --overnight_variant "$VARIANT" \
  --run_name "$RUN_NAME" \
  --num_envs 256 \
  --max_iterations 3663 \
  --scratch_checkpoint_milestones 10000000,20000000,25000000,30000000 \
  --log_interval 100000 \
  --seed "$SEED" \
  --headless

test -s "$OUT/model.zip"
test -s "$OUT/model_vecnormalize.pkl"
for m in 10000000 20000000 25000000 30000000; do
  test -s "$OUT/model_fresh_${m}_steps.zip"
  test -s "$OUT/model_vecnormalize_fresh_${m}_steps.pkl"
done
INNER
