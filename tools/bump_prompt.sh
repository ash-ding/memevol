#!/usr/bin/env bash
# Versioned-prompt bookkeeping. Does NOT edit prompt content — that's your job.
#
# A prompt version is FIVE files under forge/prompts/parts/ sharing one stem:
#   evolving_<stem>.md   how to run the search (mechanical)
#   design_<stem>.md     how to design a memory system (the search direction)
#   task_<stem>.md       per-iteration task prompt
#   fix_<stem>.md        post-sanity-failure fix prompt
#   fragments_<stem>.py  keyed substitution data
#
#   tools/bump_prompt.sh new [--from <stem>]
#       Copy the current default (or --from <stem>) parts to _draft_* files.
#       Then edit whichever halves you mean to change — editing only
#       _draft_design.md is a supported, common case.
#
#   tools/bump_prompt.sh finalize [--set-default] [--git-add]
#       Hash the draft set, rename every file to <kind>_<YYYYMMDD>_<HHMM>_<hash8>,
#       patch PROMPT_VERSION inside fragments, and smoke-load via the loader.
#       With --set-default, also overwrite _default.
#       With --git-add, stage the new files (and _default if changed).
#
#   tools/bump_prompt.sh status
#       Print resolved default stem, draft state, and available stems.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PARTS_DIR="${ROOT}/forge/prompts/parts"
DEFAULT_FILE="${PARTS_DIR}/_default"

# kind:extension — the full set that makes one version.
KINDS=(evolving:md design:md task:md fix:md fragments:py)

STEM_RX='^[0-9]{8}_[0-9]{4}_[0-9a-f]{8}$'

die() { echo "bump_prompt: $*" >&2; exit 1; }

draft_path() { echo "${PARTS_DIR}/_draft_${1}.${2}"; }
part_path()  { echo "${PARTS_DIR}/${1}_${3}.${2}"; }

resolve_default() {
    [ -f "$DEFAULT_FILE" ] || die "missing $DEFAULT_FILE"
    local stem
    stem=$(<"$DEFAULT_FILE")
    stem="${stem//[$'\t\r\n ']/}"
    [ -n "$stem" ] || die "$DEFAULT_FILE is empty"
    echo "$stem"
}

existing_drafts() {
    local kind ext found=()
    for pair in "${KINDS[@]}"; do
        kind="${pair%%:*}"; ext="${pair##*:}"
        [ -f "$(draft_path "$kind" "$ext")" ] && found+=("_draft_${kind}.${ext}")
    done
    echo "${found[@]:-}"
}

cmd_new() {
    local src_stem=""
    while [ $# -gt 0 ]; do
        case "$1" in
            --from) src_stem="$2"; shift 2;;
            *) die "unknown option: $1";;
        esac
    done
    [ -n "$src_stem" ] || src_stem=$(resolve_default)
    [[ "$src_stem" =~ $STEM_RX ]] || die "source stem '$src_stem' doesn't match $STEM_RX"

    local existing
    existing=$(existing_drafts)
    [ -z "$existing" ] || die "draft already exists ($existing) — finalize or rm it first"

    local kind ext src dst
    for pair in "${KINDS[@]}"; do
        kind="${pair%%:*}"; ext="${pair##*:}"
        src=$(part_path "$kind" "$ext" "$src_stem")
        [ -f "$src" ] || die "source file not found: $src"
        dst=$(draft_path "$kind" "$ext")
        cp "$src" "$dst"
    done
    echo "created draft set from $src_stem:"
    for pair in "${KINDS[@]}"; do
        kind="${pair%%:*}"; ext="${pair##*:}"
        echo "  $(basename "$(draft_path "$kind" "$ext")")"
    done
    echo "next: edit the halves you mean to change, then:  tools/bump_prompt.sh finalize [--set-default]"
}

cmd_finalize() {
    local set_default=0 git_add=0
    while [ $# -gt 0 ]; do
        case "$1" in
            --set-default) set_default=1; shift;;
            --git-add)     git_add=1;     shift;;
            *) die "unknown option: $1";;
        esac
    done

    local kind ext d
    for pair in "${KINDS[@]}"; do
        kind="${pair%%:*}"; ext="${pair##*:}"
        d=$(draft_path "$kind" "$ext")
        [ -f "$d" ] || die "incomplete draft: missing $(basename "$d") — run 'new' first"
    done

    # Hash over the whole set, so any half changing yields a new stem.
    local ts hash8 new_stem
    ts=$(date -u +%Y%m%d_%H%M)
    hash8=$(for pair in "${KINDS[@]}"; do
                kind="${pair%%:*}"; ext="${pair##*:}"
                cat "$(draft_path "$kind" "$ext")"
            done | sha256sum | cut -c1-8)
    new_stem="${ts}_${hash8}"

    for pair in "${KINDS[@]}"; do
        kind="${pair%%:*}"; ext="${pair##*:}"
        [ ! -e "$(part_path "$kind" "$ext" "$new_stem")" ] \
            || die "target $(part_path "$kind" "$ext" "$new_stem") already exists (rare collision — wait 60s and retry)"
    done

    local target
    for pair in "${KINDS[@]}"; do
        kind="${pair%%:*}"; ext="${pair##*:}"
        target=$(part_path "$kind" "$ext" "$new_stem")
        mv "$(draft_path "$kind" "$ext")" "$target"
    done

    # Patch PROMPT_VERSION inside fragments. `sed -i` wants a backup suffix on
    # BSD/macOS and rejects one on GNU, so write through a temp file instead.
    local frag
    frag=$(part_path fragments py "$new_stem")
    sed -E "s/^PROMPT_VERSION = \".*\"$/PROMPT_VERSION = \"${new_stem}\"/" "$frag" > "${frag}.tmp"
    mv "${frag}.tmp" "$frag"
    grep -q "^PROMPT_VERSION = \"${new_stem}\"$" "$frag" \
        || die "failed to update PROMPT_VERSION inside $frag (no matching line?)"

    # Smoke-load via the real loader so a missing sentinel, missing export, or
    # syntax error surfaces here rather than mid-run. `uv` is how this repo runs
    # python, but the tool has to work on a laptop that only has python3 — the
    # loader itself has no third-party imports.
    local runner="python3"
    command -v uv >/dev/null 2>&1 && runner="uv run python"
    ( cd "$ROOT" && $runner -c "
from forge.prompts import load_prompt_parts
parts = load_prompt_parts('${new_stem}')
assert parts.PROMPT_VERSION == '${new_stem}', parts.PROMPT_VERSION
assert parts.SYSTEM_TEMPLATE and parts.TASK_PROMPT_TEMPLATE and parts.FIX_PROMPT_TEMPLATE
print('  load OK — parts assembled, PROMPT_VERSION matches stem')
" ) || die "version $new_stem failed to load (see error above) — fix and retry"

    echo "finalized: $new_stem"

    if [ "$set_default" -eq 1 ]; then
        echo "$new_stem" > "$DEFAULT_FILE"
        echo "set _default → $new_stem"
    else
        echo "_default unchanged (still $(resolve_default)).  Re-run with --set-default to switch."
    fi

    if [ "$git_add" -eq 1 ]; then
        ( cd "$ROOT" && for pair in "${KINDS[@]}"; do
              kind="${pair%%:*}"; ext="${pair##*:}"
              git add "forge/prompts/parts/${kind}_${new_stem}.${ext}" 2>/dev/null || true
          done
          git add "forge/prompts/parts/_default" 2>/dev/null || true )
        echo "git: staged new parts + _default"
    fi
}

cmd_status() {
    local stem
    stem=$(resolve_default)
    echo "default version: $stem"
    echo "parts dir:       $PARTS_DIR"
    local kind ext p
    for pair in "${KINDS[@]}"; do
        kind="${pair%%:*}"; ext="${pair##*:}"
        p=$(part_path "$kind" "$ext" "$stem")
        printf "  %-10s %s\n" "$kind" "$([ -f "$p" ] && echo "$(basename "$p")" || echo "MISSING: $(basename "$p")")"
    done
    local existing
    existing=$(existing_drafts)
    if [ -n "$existing" ]; then
        echo "draft:           $existing  (finalize or rm)"
    else
        echo "draft:           (none)"
    fi
    echo "available stems:"
    ( cd "$PARTS_DIR" && ls evolving_*.md 2>/dev/null \
        | sed -E 's/^evolving_(.*)\.md$/  \1/' )
}

[ $# -gt 0 ] || die "usage: bump_prompt.sh {new|finalize|status} [opts]   (see comments at top)"
case "$1" in
    new)      shift; cmd_new "$@";;
    finalize) shift; cmd_finalize "$@";;
    status)   shift; cmd_status "$@";;
    -h|--help) sed -n '2,26p' "$0"; exit 0;;
    *) die "unknown subcommand: $1";;
esac
