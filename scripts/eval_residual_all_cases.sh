#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CHECKPOINT="${CHECKPOINT:-$ROOT_DIR/logs/sb3/Bookshelf-Residual-Direct-v0/2026-07-07_00-33-09/model.zip}"
TASK="${TASK:-Bookshelf-Residual-Direct-v0}"
NUM_ENVS="${NUM_ENVS:-256}"
EVAL_EPISODES="${EVAL_EPISODES:-2000}"
SLOT_CLEARANCE="${SLOT_CLEARANCE:-0.003}"
SEEDS="${SEEDS:-42 123 456}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/logs/eval_residual/$(date +%Y-%m-%d_%H-%M-%S)}"

mkdir -p "$OUT_DIR"

SUMMARY_CSV="$OUT_DIR/summary.csv"
cat > "$SUMMARY_CSV" <<'CSV'
case,seed,slot_clearance,episodes,success,success_pct,drop,drop_pct,timeout,timeout_pct,scenario_sha256,trace_summary,log_path
CSV

run_case() {
  local case_name="$1"
  local seed="$2"
  local extra_args=()

  if [[ "$case_name" == "old_reset_noise" ]]; then
    extra_args+=(--eval_old_reset_noise)
  fi

  local log_path="$OUT_DIR/${case_name}_seed${seed}.log"
  local trace_dir="$OUT_DIR/${case_name}_seed${seed}_trace"
  local trace_summary="$trace_dir/summary.json"

  echo "============================================================"
  echo "case=$case_name seed=$seed clearance=$SLOT_CLEARANCE checkpoint=$CHECKPOINT"
  echo "log=$log_path"
  echo "============================================================"

  PYTHONPATH="$ROOT_DIR/source/bookshelf" "$HOME/isaacsim/python.sh" scripts/sb3/play.py \
    --task "$TASK" \
    --num_envs "$NUM_ENVS" \
    --headless \
    --seed "$seed" \
    --eval_slot_clearance "$SLOT_CLEARANCE" \
    --eval_episodes "$EVAL_EPISODES" \
    --eval_output_dir "$trace_dir" \
    --checkpoint "$CHECKPOINT" \
    "${extra_args[@]}" \
    2>&1 | tee "$log_path"

  python3 - "$case_name" "$seed" "$SLOT_CLEARANCE" "$log_path" "$trace_summary" "$SUMMARY_CSV" <<'PY'
import csv
import json
import re
import sys
from pathlib import Path

case_name, seed, slot_clearance, log_path, trace_summary, summary_csv = sys.argv[1:]
text = Path(log_path).read_text(errors="replace")
trace = json.loads(Path(trace_summary).read_text(encoding="utf-8"))
if not trace.get("scenario_trace_complete", False):
    raise SystemExit(f"Incomplete scenario trace: {trace_summary}")

patterns = {
    "episodes": r"Episodes\s*:\s*(\d+)",
    "success": r"Success\s*:\s*(\d+)\s*/\s*(\d+)\s*\((\d+)%",
    "drop": r"Drop\s*:\s*(\d+)\s*/\s*(\d+)\s*\((\d+)%",
    "timeout": r"Timeout\s*:\s*(\d+)\s*/\s*(\d+)\s*\((\d+)%",
}

episodes_m = re.search(patterns["episodes"], text)
success_m = re.search(patterns["success"], text)
drop_m = re.search(patterns["drop"], text)
timeout_m = re.search(patterns["timeout"], text)

if not all([episodes_m, success_m, drop_m, timeout_m]):
    raise SystemExit(f"Could not parse final summary from {log_path}")

row = {
    "case": case_name,
    "seed": seed,
    "slot_clearance": slot_clearance,
    "episodes": episodes_m.group(1),
    "success": success_m.group(1),
    "success_pct": success_m.group(3),
    "drop": drop_m.group(1),
    "drop_pct": drop_m.group(3),
    "timeout": timeout_m.group(1),
    "timeout_pct": timeout_m.group(3),
    "scenario_sha256": trace["scenario_sha256"],
    "trace_summary": trace_summary,
    "log_path": log_path,
}

with open(summary_csv, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(row))
    writer.writerow(row)
PY
}

for seed in $SEEDS; do
  run_case "current_noise" "$seed"
done

for seed in $SEEDS; do
  run_case "old_reset_noise" "$seed"
done

echo "Done."
echo "Raw logs: $OUT_DIR"
echo "Summary:  $SUMMARY_CSV"
cat "$SUMMARY_CSV"
