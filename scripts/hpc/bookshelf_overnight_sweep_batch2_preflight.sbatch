#!/bin/bash
#SBATCH --partition=milan-gpu
#SBATCH --output=/fred/oz430/tliu/logs/bookshelf_overnight_sweep/%x_%j.out
#SBATCH --error=/fred/oz430/tliu/logs/bookshelf_overnight_sweep/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=00:25:00
#SBATCH --gres=gpu:1

set -euo pipefail

REPO=/fred/oz430/tliu/code/bookshelf-targeted-dr-20260902
SIF=/fred/oz430/tliu/containers/isaac-lab-bookshelf.sif
CACHE=/fred/oz430/tliu/isaac_cache
PANDA_USD=/fred/oz430/tliu/assets/isaac_5_1/IsaacLab/Robots/FrankaEmika/panda_instanceable.usd

mkdir -p /fred/oz430/tliu/logs/bookshelf_overnight_sweep
module load apptainer

apptainer exec --nv \
  -B /fred:/fred \
  -B "$CACHE/kit_cache:/isaac-sim/kit/cache" \
  -B "$CACHE/kit_data:/isaac-sim/kit/data" \
  -B "$CACHE/ov:/home/tliu/.cache/ov" \
  -B "$CACHE/nvidia:/home/tliu/.cache/nvidia" \
  -B "$CACHE/documents:/home/tliu/Documents" \
  "$SIF" bash -s -- "$REPO" "$PANDA_USD" <<'INNER'
set -euo pipefail
REPO=$1
PANDA_USD=$2

export ACCEPT_EULA=Y HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 HOME=/home/tliu
export PYTHONPATH="$REPO/source/bookshelf:${PYTHONPATH:-}"
cd "$REPO"

echo "=== 1. Asserting torch import and running test suite inside container ==="
/isaac-sim/python.sh -c "import torch; print(f'PyTorch {torch.__version__} successfully imported in container')"

echo "=== 1a. Running torch-gated numeric reward tests (must not skip) ==="
TORCH_TEST_OUTPUT=$(/isaac-sim/python.sh -m pytest -q -p no:cacheprovider \
  source/bookshelf/test/test_overnight_sweep_reward.py \
  -k "yaw or saturation" -rN)
echo "$TORCH_TEST_OUTPUT"
echo "$TORCH_TEST_OUTPUT" | grep -q "2 passed"
if echo "$TORCH_TEST_OUTPUT" | grep -iq "skipped"; then
  echo "ERROR: Torch-gated tests were skipped in container!" >&2
  exit 1
fi

echo "=== 1b. Running complete test suite inside container ==="
/isaac-sim/python.sh -m pytest -q -p no:cacheprovider \
  scripts/sb3/test_overnight_sweep_batch2_variants.py \
  scripts/sb3/test_overnight_sweep_variants.py \
  scripts/sb3/test_overnight_sweep_train_wiring.py \
  source/bookshelf/test/test_overnight_sweep_reward.py \
  source/bookshelf/test/test_overnight_sweep_insert_only.py \
  source/bookshelf/test/test_targeted_dr_profiles.py

echo "=== 2. Running K6 (sc_lat2_softwall_dr) scratch smoke test ==="
PREFLIGHT_K6_RUN="preflight_batch2_$(date +%Y%m%d_%H%M%S)"
K6_OUT="$REPO/logs/sb3/Bookshelf-Residual-Direct-v0/$PREFLIGHT_K6_RUN"
K6_LOG="/tmp/${PREFLIGHT_K6_RUN}.log"

/isaac-sim/python.sh scripts/sb3/train.py \
  --task Bookshelf-Residual-Direct-v0 \
  --robot_usd_path "$PANDA_USD" \
  --overnight_variant sc_lat2_softwall_dr \
  --run_name "$PREFLIGHT_K6_RUN" \
  --num_envs 16 \
  --max_iterations 1 \
  --scratch_checkpoint_milestones 16 \
  --log_interval 16 \
  --seed 2026 \
  --headless \
  2>&1 | tee "$K6_LOG"

grep -q '\[OVERNIGHT_VARIANT\]' "$K6_LOG"
grep -q '"slot_lateral_clearance_min": 0.003' "$K6_LOG"
grep -q '"slot_lateral_clearance_max": 0.0045' "$K6_LOG"

test -s "$K6_OUT/model.zip"
test -s "$K6_OUT/model_vecnormalize.pkl"
test -s "$K6_OUT/model_fresh_16_steps.zip"
test -s "$K6_OUT/model_vecnormalize_fresh_16_steps.pkl"

echo "=== 3. Running K7 (sc_wall05_ramp) ramp smoke test ==="
PREFLIGHT_K7_RUN="preflight_batch2_k7_$(date +%Y%m%d_%H%M%S)"
K7_OUT="$REPO/logs/sb3/Bookshelf-Residual-Direct-v0/$PREFLIGHT_K7_RUN"
K7_LOG="/tmp/${PREFLIGHT_K7_RUN}.log"

/isaac-sim/python.sh scripts/sb3/train.py \
  --task Bookshelf-Residual-Direct-v0 \
  --robot_usd_path "$PANDA_USD" \
  --overnight_variant sc_wall05_ramp \
  --run_name "$PREFLIGHT_K7_RUN" \
  --num_envs 16 \
  --max_iterations 1 \
  --scratch_checkpoint_milestones 16 \
  --log_interval 16 \
  --seed 2026 \
  --headless \
  2>&1 | tee "$K7_LOG"

grep -q '\[OVERNIGHT_VARIANT\]' "$K7_LOG"
grep -q '"insert_variant_ramp_transitions": 5000000' "$K7_LOG"
test -s "$K7_OUT/model.zip"
test -s "$K7_OUT/model_vecnormalize.pkl"
test -s "$K7_OUT/model_fresh_16_steps.zip"
test -s "$K7_OUT/model_vecnormalize_fresh_16_steps.pkl"

echo "=== Batch 2 preflight completed successfully ==="
INNER
