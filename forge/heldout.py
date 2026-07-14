"""Dedicated held-out test entry point — evaluate specific harness(es) on the
test split, WITHOUT the search loop's proposer / sanity gate / frontier.

    # Config-first (see configs/test_example.yaml for the annotated schema —
    # `harnesses:` lists the target dirs right in the YAML):
    venv/bin/python -m forge.heldout --config configs/test_example.yaml

    # CLI overrides: --harness (repeatable) REPLACES the YAML `harnesses:`
    # list; every orchestrator flag (--datasets, --coverage, ...) works too.
    venv/bin/python -m forge.heldout --config configs/test_example.yaml \\
        --harness workspace/<run>/harnesses/3_9f00aa11 [--coverage sample]

Flow per harness:
  1. Copy the harness dir into this run's workspace (source stays untouched;
     per-run isolation, same principle as the search loop).
  2. ensure_image (per-harness delta resolved from its requirements.txt).
  3. evaluate_harness(split="test", coverage=...) — coverage="full" (the
     default here) runs ONE whole-test-split pass per benchmark; "sample"
     runs the staged gauntlet at the configured stage sizes.

Outputs under workspace/<run_name>/:
  heldout_results.json   {harness_id: {objectives (accuracy_<ds>,
                          stage_<ds>, tokens_total, ...), per_ds}}
  harnesses/<id>/<ds>/...  the usual per-stage artifacts (full/ for coverage=full)
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

import yaml
from pathlib import Path
from typing import Any, Dict, List

from forge.env_builder import EnvBuildError, ensure_image
from forge.orchestrator import (
    _attach_run_log,
    _build_objectives,
    _resolve_config,
    _setup_logging,
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
            update_type=cfg["update_type"],
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

    cfg = _resolve_config(args)
    raw_yaml = _yaml_raw(args)
    harnesses = _resolve_harnesses(args, raw_yaml)
    # Held-out evaluation is a final-numbers flow — default to FULL coverage.
    # Precedence: --coverage CLI > explicit `coverage:` in the YAML > "full"
    # (DEFAULT_CONFIG's "sample" is a search-loop default, not a heldout one).
    if getattr(args, "coverage", None) is None:
        cfg["coverage"] = raw_yaml.get("coverage") or "full"

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
