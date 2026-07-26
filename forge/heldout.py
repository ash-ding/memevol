"""Dedicated held-out test entry point — evaluate specific harness(es) on the
test split, WITHOUT the search loop's proposer / sanity gate / frontier.

    # Config-first (see configs/test_example.yaml for the annotated schema —
    # `harnesses:` lists the target dirs right in the YAML):
    venv/bin/python -m forge.heldout --config configs/test_example.yaml

    # CLI overrides: --harness (repeatable) REPLACES the YAML `harnesses:`
    # list; every orchestrator flag (--datasets, ...) works too.
    venv/bin/python -m forge.heldout --config configs/test_example.yaml \\
        --harness workspace/<run>/harnesses/3_9f00aa11

Flow per harness:
  1. Copy the harness dir into this run's workspace (source stays untouched;
     per-run isolation, same principle as the search loop).
  2. ensure_image (per-harness delta resolved from its requirements.txt).
  3. evaluate_harness(split="test", coverage="full") — held-out evaluation
     is ALWAYS ONE whole-test-split pass per benchmark (coverage="full",
     equivalently progressive=false). Requesting the staged gauntlet
     (progressive=true / coverage="sample") is REJECTED with an error
     (exit 2, before any harness runs) — see
     `_reject_progressive_on_heldout` — because held-out numbers must come
     from a uniform full evaluation, not a subset that can eliminate
     candidates early.

Outputs under workspace/<run_name>/:
  heldout_results.json   {harness_id: {objectives (accuracy_<ds>,
                          stage_<ds>, tokens_total, ...), per_ds}}
  harnesses/<id>/<ds>/...  the usual per-stage artifacts (single/ for coverage=full,
                          sized by each dataset's REQUIRED single_stage config block)
  config.yaml, orchestrator.log — same bookkeeping as any run.

The datasets / model / judge_model / llm-transport / gpu configuration is
inherited from --config (plus the usual CLI overrides); search-loop-only
settings (steps, k_per_step, sanity, seed, proposer.*) are ignored.
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as _dt
import json
import logging
import shutil
import sys

import yaml
from pathlib import Path
from typing import Any, Dict, List

from forge.env_builder import EnvBuildError, ensure_image
from forge.orchestrator import (
    HELDOUT_REQUIRED_SCHEMA,
    _attach_run_log,
    _build_objectives,
    _resolve_config,
    _setup_logging,
    _sync_coverage_progressive,
    _write_resolved_config,
    build_arg_parser,
    evaluate_harness,
)
from forge.paths import paths

log = logging.getLogger("forge.heldout")


def _heldout_arg_parser() -> argparse.ArgumentParser:
    """The orchestrator's parser (so every config override keeps working)
    plus the heldout-specific --harness flag and a full-coverage default."""
    parser = build_arg_parser()
    parser.add_argument(
        "--harness", action="append", dest="harnesses", metavar="DIR",
        help="Path to a harness directory (must contain harness.py). Repeat "
             "for several. Overrides (replaces) the config's `harnesses:` list.",
    )
    return parser


def _stage_harness(src: Path) -> str:
    """Copy a source harness dir into this run's workspace; return its id."""
    src = src.resolve()
    if not (src / "harness.py").exists():
        raise SystemExit(
            f"--harness {src}: no harness.py found — pass a harness directory "
            f"(e.g. workspace/<run>/harnesses/<int>_<hash8>)"
        )
    dst = paths.harnesses_dir / src.name
    if dst.exists():
        raise SystemExit(
            f"duplicate harness id {src.name!r} in this heldout run — "
            f"pass directories with distinct names"
        )
    shutil.copytree(src, dst)
    return src.name


async def _run(cfg: Dict[str, Any], harness_paths: List[str]) -> None:
    paths.harnesses_dir.mkdir(parents=True, exist_ok=True)
    ids = [_stage_harness(Path(p)) for p in harness_paths]

    results: Dict[str, Any] = {}
    for hid in ids:
        harness_dir = paths.harnesses_dir / hid
        try:
            image_path = await ensure_image(harness_dir)
        except EnvBuildError as exc:
            log.error(f"heldout: image build failed for {hid}: {exc}")
            results[hid] = {"error": f"image build failed: {exc}"}
            continue

        log.info(f"heldout: evaluating {hid} (coverage={cfg['coverage']}, "
                 f"datasets={list(cfg['datasets'])})")
        per_ds = await evaluate_harness(
            hid, image_path,
            datasets_config=cfg["datasets"],
            split="test",
            model=cfg["model"], judge_model=cfg["judge_model"],
            max_sample_concurrent=cfg["max_sample_concurrent"],
            memory_cache=cfg.get("memory_cache", True),
            gpu=cfg["gpu"]["enabled"],
            llm_cfg=cfg.get("llm"),
            coverage=cfg.get("coverage", "sample"),
        )
        results[hid] = {
            "objectives": _build_objectives(per_ds, harness_dir),
            "per_ds": per_ds,
        }

    out_path = paths.workspace / "heldout_results.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    log.info(f"heldout: results written to {out_path}")
    for hid, r in results.items():
        if "error" in r:
            log.info(f"  {hid}: ERROR — {r['error']}")
            continue
        obj = r["objectives"]
        per_ds_str = "  ".join(
            f"{k[len('accuracy_'):]}={v:.3f}" for k, v in sorted(obj.items())
            if k.startswith("accuracy_")
        )
        log.info(
            f"  {hid}: {per_ds_str}  tokens={obj.get('tokens_total', 0)}"
        )


def _yaml_raw(args) -> Dict[str, Any]:
    """The --config YAML as written (pre-merge) — for heldout-specific keys
    (`harnesses:`) and for distinguishing explicit YAML values from
    DEFAULT_CONFIG fallbacks (`coverage`)."""
    if not getattr(args, "config", None):
        return {}
    with open(args.config) as f:
        return yaml.safe_load(f) or {}


def _apply_heldout_coverage_default(
    cfg: Dict[str, Any], args, raw_yaml: Dict[str, Any]
) -> None:
    """Held-out evaluation is a final-numbers flow — default to FULL coverage
    when NEITHER `coverage` NOR `progressive` is given anywhere (CLI or
    YAML). `progressive` is the canonical knob (`coverage` is only a
    back-compat alias), so an explicit `progressive:` in the YAML must win
    over the "default to full" override even when the YAML omits
    `coverage:` — forcing `coverage="full"` in that case used to silently
    flip `progressive` back to False, discarding the user's explicit intent
    (`progressive: true` + no `coverage:` was resolved to `progressive=False,
    coverage="full"` instead of `progressive=True, coverage="sample"`).

    Mutates `cfg` in place. Precedence:
      1. Neither `coverage` nor `progressive` given anywhere → heldout's own
         default: `coverage="full"` (DEFAULT_CONFIG's `progressive=True` /
         `coverage="sample"` is a search-loop default, not a heldout one).
      2. `progressive` given explicitly (YAML key or --progressive/
         --no-progressive) → respect it as authoritative and (re-)derive
         `coverage` from it, matching `_resolve_config`'s own precedence
         (progressive wins over the coverage alias) — regardless of whether
         `coverage:` also happens to be present.
      3. Otherwise (`coverage` given, `progressive` not) → `_resolve_config`
         already derived `progressive` from `coverage` consistently; re-sync
         here is idempotent (safety net, not a behavior change).
    """
    coverage_given = getattr(args, "coverage", None) is not None or "coverage" in raw_yaml
    progressive_given = getattr(args, "progressive", None) is not None or "progressive" in raw_yaml
    if not coverage_given and not progressive_given:
        cfg["coverage"] = "full"
    if progressive_given:
        _sync_coverage_progressive(cfg, source="progressive")
    else:
        _sync_coverage_progressive(cfg, source="coverage")


def _reject_progressive_on_heldout(cfg: Dict[str, Any]) -> None:
    """Fail fast if the resolved heldout config still requests the staged
    gauntlet after `_apply_heldout_coverage_default` has run.

    Held-out test evaluation exists to produce final, comparable numbers —
    that requires ONE full, uniform pass over the whole test split. The
    staged gauntlet (`progressive=True`, equivalently the back-compat
    `coverage="sample"` alias) evaluates small per-stage subsets and
    ELIMINATES candidates early on a threshold check: a harness cut at
    stage1 only ever gets a ~20-item score, never the full-split score.
    That's semantically invalid as a held-out result, so we refuse to run
    rather than silently producing a partial/eliminated number.

    Checks BOTH spellings defensively — `progressive` is the canonical
    knob, but `coverage` is also checked directly in case some future code
    path leaves the two inconsistent (`_sync_coverage_progressive` is
    supposed to prevent that, but this guard does not rely on that
    invariant holding). Must be called BEFORE any harness is staged or
    evaluated (see `main()` — right after `_apply_heldout_coverage_default`,
    before the workspace/run_name setup and `_run(...)`).
    """
    if cfg.get("progressive") is True or cfg.get("coverage") == "sample":
        msg = (
            "forge.heldout: refusing to run — this config requests the "
            "staged gauntlet (progressive=true / coverage=sample), which is "
            "not valid for held-out test evaluation.\n"
            "Held-out numbers must come from ONE full, uniform pass over the "
            "whole test split. The staged gauntlet samples small per-stage "
            "subsets and ELIMINATES candidates early via threshold checks — "
            "a harness cut at stage1 only ever gets a ~20-item score, not "
            "the full-split score — which invalidates final held-out "
            "numbers.\n"
            "Fix: set `progressive: false` (or the back-compat "
            "`coverage: full`) in your heldout config, or drop both fields "
            "entirely (heldout defaults to full coverage when neither is "
            "given anywhere)."
        )
        log.warning(msg)
        sys.exit(2)


def _resolve_harnesses(args, raw_yaml: Dict[str, Any]) -> List[str]:
    """CLI --harness (repeatable) REPLACES the YAML `harnesses:` list."""
    if getattr(args, "harnesses", None):
        return list(args.harnesses)
    yaml_harnesses = raw_yaml.get("harnesses") or []
    if not isinstance(yaml_harnesses, list) or not yaml_harnesses:
        raise SystemExit(
            "no harnesses to evaluate — pass --harness <dir> (repeatable) or "
            "set a `harnesses:` list in the config "
            "(see configs/test_example.yaml)"
        )
    return [str(h) for h in yaml_harnesses]


def main() -> None:
    args = _heldout_arg_parser().parse_args()
    _setup_logging(getattr(args, "verbose", False))

    cfg = _resolve_config(args, required_schema=HELDOUT_REQUIRED_SCHEMA)
    raw_yaml = _yaml_raw(args)
    harnesses = _resolve_harnesses(args, raw_yaml)
    _apply_heldout_coverage_default(cfg, args, raw_yaml)
    _reject_progressive_on_heldout(cfg)

    run_name = cfg.get("run_name") or _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    if not str(run_name).startswith("heldout"):
        run_name = f"heldout_{run_name}"
    paths.set_run_id(str(run_name))
    paths.workspace.mkdir(parents=True, exist_ok=True)
    _attach_run_log()
    _write_resolved_config(cfg)
    log.info(f"heldout run: {paths.workspace} (test split, coverage={cfg['coverage']})")

    asyncio.run(_run(cfg, harnesses))


if __name__ == "__main__":
    main()
