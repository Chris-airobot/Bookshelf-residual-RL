#!/bin/bash
#SBATCH --partition=milan-gpu
#SBATCH --output=/fred/oz430/tliu/logs/bookshelf_overnight_sweep/%x_%j.out
#SBATCH --error=/fred/oz430/tliu/logs/bookshelf_overnight_sweep/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:25:00
#SBATCH --gres=gpu:1

set -euo pipefail

REPO=/fred/oz430/tliu/code/bookshelf-targeted-dr-20260902
SIF=/fred/oz430/tliu/containers/isaac-lab-bookshelf.sif
CACHE=/fred/oz430/tliu/isaac_cache
CHECKPOINT=/fred/oz430/tliu/evaluation_models/bookshelf/residual_seed42/model.zip
PANDA_USD=/fred/oz430/tliu/assets/isaac_5_1/IsaacLab/Robots/FrankaEmika/panda_instanceable.usd

mkdir -p /fred/oz430/tliu/logs/bookshelf_overnight_sweep
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
  "$SIF" bash -s -- "$REPO" "$CHECKPOINT" "$PANDA_USD" <<'INNER'
set -euo pipefail
REPO=$1
CHECKPOINT=$2
PANDA_USD=$3

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
  source/bookshelf/test/test_overnight_sweep_reward.py \
  source/bookshelf/test/test_overnight_sweep_insert_only.py \
  scripts/sb3/test_overnight_sweep_variants.py \
  scripts/sb3/test_overnight_sweep_train_wiring.py \
  source/bookshelf/test/test_targeted_dr_profiles.py \
  source/bookshelf/test/test_bookshelf_drop_logic.py \
  scripts/sb3/test_evaluation_scenarios.py

echo "=== 2. Running fine-tune smoke test ==="
PREFLIGHT_FT_RUN="preflight_ft_$(date +%Y%m%d_%H%M%S)"
FT_OUT="$REPO/logs/sb3/Bookshelf-Residual-Direct-v0/$PREFLIGHT_FT_RUN"
FT_LOG="/tmp/${PREFLIGHT_FT_RUN}.log"

/isaac-sim/python.sh scripts/sb3/train.py \
  --task Bookshelf-Residual-Direct-v0 \
  --checkpoint "$CHECKPOINT" \
  --resume \
  --robot_usd_path "$PANDA_USD" \
  --overnight_variant wall_depth \
  --run_name "$PREFLIGHT_FT_RUN" \
  --num_envs 16 \
  --max_iterations 1 \
  --finetune_checkpoint_milestones 16 \
  --log_interval 16 \
  --seed 42 \
  --headless \
  agent.learning_rate=5e-5 agent.clip_range=0.1 agent.n_epochs=5 agent.ent_coef=0.003 \
  2>&1 | tee "$FT_LOG"

grep -q "\[RESUME_HYPERPARAMS\] effective n_epochs=5" "$FT_LOG"
test -s "$FT_OUT/model.zip"
test -s "$FT_OUT/model_vecnormalize.pkl"
test -s "$FT_OUT/model_finetune_16_steps.zip"
test -s "$FT_OUT/model_vecnormalize_finetune_16_steps.pkl"

echo "=== 3. Running scratch smoke test ==="
PREFLIGHT_SC_RUN="preflight_sc_$(date +%Y%m%d_%H%M%S)"
SC_OUT="$REPO/logs/sb3/Bookshelf-Residual-Direct-v0/$PREFLIGHT_SC_RUN"

/isaac-sim/python.sh scripts/sb3/train.py \
  --task Bookshelf-Residual-Direct-v0 \
  --robot_usd_path "$PANDA_USD" \
  --overnight_variant sc_lat4 \
  --run_name "$PREFLIGHT_SC_RUN" \
  --num_envs 16 \
  --max_iterations 1 \
  --training_checkpoint_interval_steps 16 \
  --log_interval 16 \
  --seed 42 \
  --headless

test -s "$SC_OUT/model.zip"
test -s "$SC_OUT/model_vecnormalize.pkl"

echo "=== Preflight completed successfully ==="
INNER
