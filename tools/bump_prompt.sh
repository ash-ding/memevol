#!/usr/bin/env bash
# Versioned-prompt bookkeeping. Does NOT edit prompt content — that's your job.
#
#   tools/bump_prompt.sh new [--from <stem>]
#       Copy the current default (or --from <stem>) template to
#       forge/prompts/templates/_draft.py. Then edit _draft.py yourself.
#
#   tools/bump_prompt.sh finalize [--set-default] [--git-add]
#       Read _draft.py, compute sha256[:8] of the file, rename to
#       <YYYYMMDD>_<HHMM>_<hash8>.py, and update the PROMPT_VERSION constant
#       inside to match. With --set-default, also overwrite _default.
#       With --git-add, stage the new file (and _default if changed).
#
#   tools/bump_prompt.sh status
#       Print resolved default stem + whether a _draft.py exists.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TPL_DIR="${ROOT}/forge/prompts/templates"
DEFAULT_FILE="${TPL_DIR}/_default"
DRAFT_FILE="${TPL_DIR}/_draft.py"

STEM_RX='^[0-9]{8}_[0-9]{4}_[0-9a-f]{8}$'

die() { echo "bump_prompt: $*" >&2; exit 1; }

resolve_default() {
    [ -f "$DEFAULT_FILE" ] || die "missing $DEFAULT_FILE"
    local stem
    stem=$(<"$DEFAULT_FILE")
    [ -n "$stem" ] || die "$DEFAULT_FILE is empty"
    echo "$stem"
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
    local src_file="${TPL_DIR}/${src_stem}.py"
    [ -f "$src_file" ] || die "source file not found: $src_file"
    [ ! -f "$DRAFT_FILE" ] || die "draft already exists at $DRAFT_FILE — finalize or rm it first"
    cp "$src_file" "$DRAFT_FILE"
    echo "created draft: $DRAFT_FILE  (from $src_stem)"
    echo "next: edit $DRAFT_FILE, then run:  tools/bump_prompt.sh finalize [--set-default]"
}

cmd_finalize() {
    local set_default=0
    local git_add=0
    while [ $# -gt 0 ]; do
        case "$1" in
            --set-default) set_default=1; shift;;
            --git-add)     git_add=1;     shift;;
            *) die "unknown option: $1";;
        esac
    done
    [ -f "$DRAFT_FILE" ] || die "no draft at $DRAFT_FILE — run 'new' first"

    local ts hash8 new_stem new_file
    ts=$(date -u +%Y%m%d_%H%M)
    hash8=$(sha256sum "$DRAFT_FILE" | cut -c1-8)
    new_stem="${ts}_${hash8}"
    new_file="${TPL_DIR}/${new_stem}.py"

    [ ! -e "$new_file" ] || die "target $new_file already exists (rare collision — wait 60s and retry, or edit one more char)"

    mv "$DRAFT_FILE" "$new_file"
    # Patch the PROMPT_VERSION constant inside. Tolerant of any prior value.
    sed -i -E "s/^PROMPT_VERSION = \".*\"$/PROMPT_VERSION = \"${new_stem}\"/" "$new_file"
    grep -q "^PROMPT_VERSION = \"${new_stem}\"$" "$new_file" \
        || die "failed to update PROMPT_VERSION inside $new_file (no matching line?)"

    # Smoke-load via the existing loader so missing exports / syntax errors
    # surface here, not later in a search run.
    ( cd "$ROOT" && uv run python -c "
from forge.prompts import load_template_module
mod = load_template_module('${new_stem}')
assert mod.PROMPT_VERSION == '${new_stem}', mod.PROMPT_VERSION
print('  load OK — exports present, PROMPT_VERSION matches stem')
" ) || die "module $new_file failed to load (see error above) — fix and retry"

    echo "finalized: ${new_stem}.py"

    if [ "$set_default" -eq 1 ]; then
        echo -n "$new_stem" > "$DEFAULT_FILE"
        echo "set _default → $new_stem"
    else
        echo "_default unchanged (still $(resolve_default)).  Re-run with --set-default to switch."
    fi

    if [ "$git_add" -eq 1 ]; then
        ( cd "$ROOT" && git add "forge/prompts/templates/${new_stem}.py" \
                              "forge/prompts/templates/_default" 2>/dev/null || true )
        echo "git: staged new template + _default"
    fi
}

cmd_status() {
    local stem
    stem=$(resolve_default)
    echo "default version: $stem"
    echo "default path:    ${TPL_DIR}/${stem}.py"
    if [ -f "$DRAFT_FILE" ]; then
        echo "draft:           $DRAFT_FILE  (exists — finalize or rm)"
    else
        echo "draft:           (none)"
    fi
    echo "available stems:"
    ( cd "$TPL_DIR" && ls *.py 2>/dev/null | grep -E '^[0-9]{8}_[0-9]{4}_[0-9a-f]{8}\.py$' | sed 's/^/  /' )
}

[ $# -gt 0 ] || die "usage: bump_prompt.sh {new|finalize|status} [opts]   (see comments at top)"
case "$1" in
    new)      shift; cmd_new "$@";;
    finalize) shift; cmd_finalize "$@";;
    status)   shift; cmd_status "$@";;
    -h|--help) sed -n '2,18p' "$0"; exit 0;;
    *) die "unknown subcommand: $1";;
esac
