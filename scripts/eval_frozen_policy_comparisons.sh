#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

BANK="${BANK:-/home/chris/BookshelfFiles/evaluation_results/frozen_banks/current_noise_3mm_2000_seed20260812.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/chris/BookshelfFiles/evaluation_results/frozen_comparisons_$(date +%Y-%m-%d_%H-%M-%S)}"
PANDA_USD="${PANDA_USD:-/home/chris/Downloads/FrankaEmika/panda_instanceable.usd}"
NUM_ENVS="${NUM_ENVS:-256}"
SEED="${SEED:-42}"
METHODS="${METHODS:-nominal_only residual_ppo}"

RESIDUAL_CHECKPOINT="${RESIDUAL_CHECKPOINT:-/home/chris/BookshelfFiles/trained_models/residual_ppo_3mm_latest/model.zip}"
PPO_CHECKPOINT="${PPO_CHECKPOINT:-}"
BC_PPO_CHECKPOINT="${BC_PPO_CHECKPOINT:-}"
BC_PPO_TASK="${BC_PPO_TASK:-Bookshelf-PPO-Direct-v0}"

test -s "$BANK" || { echo "ERROR: frozen bank missing: $BANK" >&2; exit 1; }
test -s "$PANDA_USD" || { echo "ERROR: local Franka USD missing: $PANDA_USD" >&2; exit 1; }
mkdir -p "$OUTPUT_ROOT"
export BOOKSHELF_PANDA_USD_PATH="$PANDA_USD"

summaries=()

vecnormalize_path() {
  local checkpoint="$1"
  local directory basename
  directory="$(dirname "$checkpoint")"
  basename="$(basename "$checkpoint")"
  basename="${basename/model/model_vecnormalize}"
  basename="${basename%.zip}.pkl"
  printf '%s/%s\n' "$directory" "$basename"
}

require_checkpoint() {
  local method="$1"
  local checkpoint="$2"
  if [[ -z "$checkpoint" || ! -s "$checkpoint" ]]; then
    echo "ERROR: $method checkpoint is missing: ${checkpoint:-<unset>}" >&2
    exit 1
  fi
  local vecnormalize
  vecnormalize="$(vecnormalize_path "$checkpoint")"
  if [[ ! -s "$vecnormalize" ]]; then
    echo "ERROR: $method VecNormalize state is missing: $vecnormalize" >&2
    exit 1
  fi
}

run_nominal_only() {
  local output="$OUTPUT_ROOT/nominal_only"
  PYTHONPATH="$ROOT_DIR/source/bookshelf" \
  "$HOME/isaacsim/python.sh" scripts/sb3/play.py \
    --task Bookshelf-Residual-Direct-v0 \
    --num_envs "$NUM_ENVS" \
    --headless \
    --seed "$SEED" \
    --eval_slot_clearance 0.003 \
    --eval_scenario_bank "$BANK" \
    --eval_output_dir "$output" \
    --eval_nominal_only \
    2>&1 | tee "$OUTPUT_ROOT/nominal_only.log"
  summaries+=("$output/summary.json")
}

run_ppo_method() {
  local name="$1"
  local task="$2"
  local checkpoint="$3"
  local output="$OUTPUT_ROOT/$name"
  require_checkpoint "$name" "$checkpoint"
  PYTHONPATH="$ROOT_DIR/source/bookshelf" \
  "$HOME/isaacsim/python.sh" scripts/sb3/play.py \
    --task "$task" \
    --num_envs "$NUM_ENVS" \
    --headless \
    --seed "$SEED" \
    --eval_slot_clearance 0.003 \
    --eval_scenario_bank "$BANK" \
    --eval_output_dir "$output" \
    --checkpoint "$checkpoint" \
    2>&1 | tee "$OUTPUT_ROOT/$name.log"
  summaries+=("$output/summary.json")
}

for method in $METHODS; do
  case "$method" in
    nominal_only)
      run_nominal_only
      ;;
    residual_ppo)
      run_ppo_method "$method" Bookshelf-Residual-Direct-v0 "$RESIDUAL_CHECKPOINT"
      ;;
    ppo_only)
      run_ppo_method "$method" Bookshelf-PPO-Direct-v0 "$PPO_CHECKPOINT"
      ;;
    bc_ppo)
      run_ppo_method "$method" "$BC_PPO_TASK" "$BC_PPO_CHECKPOINT"
      ;;
    *)
      echo "ERROR: unknown method '$method'" >&2
      exit 1
      ;;
  esac
done

python3 scripts/audit_frozen_scenario_replays.py \
  --output "$OUTPUT_ROOT/frozen_replay_audit.json" \
  "$BANK" "${summaries[@]}"

python3 - "$OUTPUT_ROOT" "${summaries[@]}" <<'PY'
import csv
import json
import sys
from pathlib import Path

output_root = Path(sys.argv[1])
rows = []
for value in sys.argv[2:]:
    path = Path(value)
    summary = json.loads(path.read_text(encoding="utf-8"))
    outcomes = summary.get("outcomes", {})
    episodes = int(summary.get("episode_count", 0))
    success = int(outcomes.get("success", 0))
    rows.append(
        {
            "method": path.parent.name,
            "episodes": episodes,
            "success": success,
            "success_pct": 100.0 * success / episodes if episodes else 0.0,
            "drop": int(outcomes.get("drop", 0)),
            "timeout": int(outcomes.get("timeout", 0)),
            "scenario_trace_sha256": summary.get("scenario_sha256"),
            "checkpoint_sha256": summary.get("metadata", {}).get("checkpoint_sha256"),
            "summary": str(path.resolve()),
        }
    )

destination = output_root / "comparison_summary.csv"
with destination.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(f"Comparison summary: {destination}")
for row in rows:
    print(
        f"{row['method']:16s} success={row['success']:4d}/{row['episodes']} "
        f"({row['success_pct']:.2f}%) drop={row['drop']} timeout={row['timeout']}"
    )
PY

echo "Evaluation directory: $OUTPUT_ROOT"
