"""The prompt split: evolving / design / task / fix / fragments.

Locks in the two properties the split has to keep true:
  1. assembly is exact — SYSTEM_TEMPLATE is the evolving half with the design
     half spliced into its sentinel, nothing else;
  2. rendering leaves no sentinel behind, on every switch the renderer has.

Run:  uv run python tests/test_prompt_parts.py
"""

import itertools
import re
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forge.prompts import (  # noqa: E402
    PromptVersionError,
    build_proposer_system,
    load_prompt_parts,
    proposer_fix_prompt,
    proposer_task_prompt,
    resolve_version,
)
from forge.prompts.loader import DESIGN_SENTINEL, _PARTS_DIR  # noqa: E402

STEM = resolve_version(None)
DATASETS = ["dynamicmem", "locomo", "longmemeval_s"]
METRIC_SETS = [None, ("accuracy",), ("accuracy", "efficiency")]


def _dataset_combos():
    for r in range(len(DATASETS) + 1):
        for c in itertools.combinations(DATASETS, r):
            yield list(c) if c else None


def test_default_version_has_all_five_parts():
    for kind, ext in (("evolving", "md"), ("design", "md"), ("task", "md"),
                      ("fix", "md"), ("fragments", "py")):
        path = _PARTS_DIR / f"{kind}_{STEM}.{ext}"
        assert path.is_file(), f"missing {path}"
        assert path.stat().st_size > 0, f"empty {path}"


def test_evolving_half_carries_exactly_one_design_sentinel():
    evolving = (_PARTS_DIR / f"evolving_{STEM}.md").read_text(encoding="utf-8")
    n = evolving.count(DESIGN_SENTINEL)
    assert n == 1, f"expected 1 design sentinel in evolving_{STEM}.md, found {n}"


def test_system_template_is_exactly_evolving_plus_design():
    evolving = (_PARTS_DIR / f"evolving_{STEM}.md").read_text(encoding="utf-8")
    design = (_PARTS_DIR / f"design_{STEM}.md").read_text(encoding="utf-8")
    parts = load_prompt_parts(None)
    assert parts.SYSTEM_TEMPLATE == evolving.replace(DESIGN_SENTINEL, design)


def test_design_half_survives_into_every_rendered_system_prompt():
    """The design knowledge is the point of the split — a rendering path that
    drops it would be silently shipping a proposer with no search direction."""
    design = (_PARTS_DIR / f"design_{STEM}.md").read_text(encoding="utf-8")
    probe = "# MISSION"
    assert probe in design
    for sanity, ds, metrics in itertools.product(
        (True, False), _dataset_combos(), METRIC_SETS
    ):
        out = build_proposer_system(
            sanity_enabled=sanity, active_datasets=ds, metrics=metrics
        )
        assert probe in out, f"design block missing: sanity={sanity} ds={ds} m={metrics}"


def test_no_sentinel_survives_rendering():
    """Any `<<NAME>>` left in a rendered prompt is an unsubstituted hole."""
    leftover = re.compile(r"<<[A-Z_]+>>")
    for sanity, ds, metrics in itertools.product(
        (True, False), _dataset_combos(), METRIC_SETS
    ):
        out = build_proposer_system(
            sanity_enabled=sanity, active_datasets=ds, metrics=metrics
        )
        found = leftover.findall(out)
        assert not found, f"unsubstituted {found}: sanity={sanity} ds={ds} m={metrics}"
    for text in (proposer_task_prompt("cand_0001"),
                 proposer_fix_prompt("cand_0001", "boom")):
        assert not leftover.findall(text), leftover.findall(text)


def test_fragments_declare_matching_version_and_required_exports():
    parts = load_prompt_parts(None)
    assert parts.PROMPT_VERSION == STEM
    assert parts.DATASET_RENDER_ORDER, "DATASET_RENDER_ORDER is empty"
    for key in parts.DATASET_RENDER_ORDER:
        assert key in parts.DATASET_INFO, f"{key} missing from DATASET_INFO"
    assert ("accuracy",) in parts.OBJECTIVE_AXES_SUBS, "accuracy-only axes block missing"
    assert set(parts.SANITY_ON_SUBS) == set(parts.SANITY_OFF_SUBS), \
        "sanity on/off must cover the same sentinels"


def test_task_and_fix_keep_their_format_slots():
    parts = load_prompt_parts(None)
    assert "{new_dir_rel}" in parts.TASK_PROMPT_TEMPLATE
    assert "{new_dir_rel}" in parts.FIX_PROMPT_TEMPLATE
    assert "{error_trace}" in parts.FIX_PROMPT_TEMPLATE
    assert "cand_0001" in proposer_task_prompt("cand_0001")
    assert "kaboom" in proposer_fix_prompt("cand_0001", "kaboom")


def test_unknown_or_malformed_version_raises_actionably():
    for bad in ("nope", "2026_0514_b959e7ee", "20260918_0514_ZZZZZZZZ"):
        try:
            load_prompt_parts(bad)
        except PromptVersionError:
            pass
        else:
            raise AssertionError(f"{bad!r} should have raised PromptVersionError")
    try:
        load_prompt_parts("20991231_2359_deadbeef")
    except PromptVersionError as exc:
        assert "not found" in str(exc), str(exc)
    else:
        raise AssertionError("a well-formed but absent stem should raise")


def main():
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = []
    for name, fn in tests:
        try:
            fn(); print(f"  PASS  {name}")
        except Exception:
            print(f"  FAIL  {name}"); traceback.print_exc(); failed.append(name)
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    if failed:
        print("failed:", ", ".join(failed)); sys.exit(1)


if __name__ == "__main__":
    main()
