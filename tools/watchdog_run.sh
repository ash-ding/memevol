#!/usr/bin/env bash
# Auto-resume a forge search run if the orchestrator dies due to fail-fast
# infra abort. Stops when frontier reaches target_steps OR after MAX_RESUMES.
#
# Usage: tools/watchdog_run.sh <run_name> <config_path> [target_steps] [max_resumes]
# Example:
#   tools/watchdog_run.sh bm25_ablation configs/search_example.yaml 10 10

set -u

PROJECT_ROOT="/export/scratch_large/ding/code/memevol"
RUN_NAME="${1:?usage: watchdog_run.sh <run_name> <config> [target_steps] [max_resumes]}"
CONFIG="${2:?usage: watchdog_run.sh <run_name> <config> [target_steps] [max_resumes]}"
TARGET_STEPS="${3:-10}"
MAX_RESUMES="${4:-10}"
RESUME_COOLDOWN_S=90      # wait before resuming, lets transient API issues settle

WORKSPACE_DIR="$PROJECT_ROOT/workspace/$RUN_NAME"
FRONTIER="$WORKSPACE_DIR/frontier.json"
WATCHDOG_LOG="$PROJECT_ROOT/workspace/_watchdog_${RUN_NAME}.log"

cd "$PROJECT_ROOT" || { echo "cannot cd $PROJECT_ROOT" >&2; exit 1; }

log() {
    printf "[watchdog %s] %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >> "$WATCHDOG_LOG"
}

count_entries() {
    [ -f "$FRONTIER" ] || { echo 0; return; }
    jq '.entries | length' "$FRONTIER" 2>/dev/null || echo 0
}

orchestrator_alive() {
    pgrep -f "forge.orchestrator.*--run-name $RUN_NAME" >/dev/null 2>&1
}

log "start; run=$RUN_NAME, target=$TARGET_STEPS, max_resumes=$MAX_RESUMES"
resume_count=0

while true; do
    if orchestrator_alive; then
        sleep 45
        continue
    fi

    done_count=$(count_entries)

    if [ "$done_count" -ge "$TARGET_STEPS" ]; then
        log "DONE: frontier=$done_count/$TARGET_STEPS (target reached)"
        break
    fi

    if [ "$resume_count" -ge "$MAX_RESUMES" ]; then
        log "GIVE_UP: hit max_resumes=$MAX_RESUMES at frontier=$done_count/$TARGET_STEPS"
        break
    fi

    resume_count=$((resume_count + 1))
    log "orchestrator dead, frontier=$done_count/$TARGET_STEPS; resume #$resume_count in ${RESUME_COOLDOWN_S}s"
    sleep "$RESUME_COOLDOWN_S"

    TS=$(date +%H%M%S)
    OUT="$PROJECT_ROOT/workspace/_launch_${RUN_NAME}.resume${resume_count}_${TS}.out"

    nohup venv/bin/python -m forge.orchestrator \
        --config "$CONFIG" \
        --run-name "$RUN_NAME" \
        > "$OUT" 2>&1 &

    local_pid=$!
    log "resumed pid=$local_pid, output=$OUT"
    sleep 25  # let it spin up before next health check
done

log "watchdog exit"
