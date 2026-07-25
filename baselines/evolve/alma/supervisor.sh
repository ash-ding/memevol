#!/bin/bash
# Alma search supervisor — launches run.py, auto-resumes from checkpoint
# if it exits before reaching the target number of evolution steps.
#
# The underlying `--steps N` is now idempotent ("target total N"), so we just
# relaunch with `--history_ckpt_path <latest>` until the ckpt's completed_steps
# reaches N (or we exceed MAX_REATTEMPTS as a safety cap).
#
# Usage: run from the project root:
#   bash baselines/evolve/alma/supervisor.sh [TARGET_STEPS]
# or via nohup for full detachment:
#   setsid nohup bash baselines/evolve/alma/supervisor.sh 10 > /tmp/alma_sup.out 2>&1 &
set -u

TARGET_STEPS="${1:-10}"
MAX_REATTEMPTS=20

PROJECT_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PYTHON="$PROJECT_ROOT/baselines/venv/bin/python"
CKPT_DIR="$PROJECT_ROOT/baselines/evolve/alma/logs"
SUP_TS="$(date +%Y%m%d_%H%M%S)"
SUP_LOG="$CKPT_DIR/supervisor_${SUP_TS}.log"

cd "$PROJECT_ROOT"

log() {
  local msg="$(date '+%Y-%m-%d %H:%M:%S') [supervisor] $*"
  echo "$msg" | tee -a "$SUP_LOG"
}

read_completed_steps() {
  # Emits the completed_steps int from the given ckpt path (0 if missing/malformed).
  local ckpt="$1"
  [ -f "$ckpt" ] || { echo 0; return; }
  "$PYTHON" -c "
import json, sys
try:
    d = json.load(open('$ckpt'))
    print(int(d.get('completed_steps', 0)))
except Exception:
    print(0)
" 2>/dev/null || echo 0
}

find_latest_ckpt() {
  # Returns the most recently modified alma checkpoint (or empty string).
  ls -t "$CKPT_DIR"/check_dynamicmem_*.json 2>/dev/null | head -1
}

# Evaluation SIZES no longer live on flat --eval_n_*/--check_n_* flags (removed).
# Each candidate is scored through the shared staged gauntlet (--progressive);
# sizes come from the family DEFAULT_STAGES (override with --stages '<json>').
COMMON_ARGS=(
  --config baselines/evolve/alma/config.example.yaml
  --status search
  --dataset dynamicmem
  --progressive
  --steps "$TARGET_STEPS"
)

log "=== Supervisor start | target_steps=$TARGET_STEPS | max_reattempts=$MAX_REATTEMPTS ==="
log "Project root: $PROJECT_ROOT"
log "Python:       $PYTHON"
log "Supervisor log: $SUP_LOG"

attempt=0
while :; do
  attempt=$((attempt + 1))
  if [ "$attempt" -gt "$MAX_REATTEMPTS" ]; then
    log "FATAL: exhausted max reattempts ($MAX_REATTEMPTS); giving up"
    exit 1
  fi

  CKPT="$(find_latest_ckpt)"
  CKPT_NAME=""
  if [ -n "$CKPT" ]; then
    CKPT_NAME="$(basename "$CKPT")"
    COMPLETED="$(read_completed_steps "$CKPT")"
    log "attempt $attempt/$MAX_REATTEMPTS | existing ckpt: $CKPT_NAME | completed_steps=$COMPLETED / $TARGET_STEPS"
    if [ "$COMPLETED" -ge "$TARGET_STEPS" ]; then
      log "Target reached ($COMPLETED >= $TARGET_STEPS); done."
      break
    fi
  else
    log "attempt $attempt/$MAX_REATTEMPTS | no ckpt yet, starting fresh"
  fi

  CHILD_OUT="/tmp/alma_search_attempt${attempt}_${SUP_TS}.out"
  if [ -n "$CKPT_NAME" ]; then
    log "resuming from ckpt=$CKPT_NAME ; child stdout/stderr -> $CHILD_OUT"
    "$PYTHON" "$PROJECT_ROOT/baselines/evolve/alma/run.py" \
      "${COMMON_ARGS[@]}" \
      --history_ckpt_path "$CKPT_NAME" \
      > "$CHILD_OUT" 2>&1
  else
    log "starting fresh ; child stdout/stderr -> $CHILD_OUT"
    "$PYTHON" "$PROJECT_ROOT/baselines/evolve/alma/run.py" \
      "${COMMON_ARGS[@]}" \
      > "$CHILD_OUT" 2>&1
  fi
  child_rc=$?
  log "attempt $attempt child exited with rc=$child_rc"

  # Re-read ckpt to check progress. If completed_steps didn't advance
  # over several attempts, something is badly stuck -- MAX_REATTEMPTS cap
  # prevents an infinite crash loop.
  sleep 2
done

log "=== Supervisor done | final completed_steps=$(read_completed_steps "$(find_latest_ckpt)") ==="
