"""The ONE entrypoint for the harness baselines (amem, hipporag2, lightmem,
mem0, memoryos, simplemem, zep). A harness directory provides only its
MemoClass (memo.py + vendored src/ + its own uv project); this module supplies
everything else — CLI, config resolution, run directory, and the adapter onto
the shared, execution-independent `common.evaluate.evaluate_memo` (the SAME
function forge's container and alma's subprocess run), so a baseline's score
is identical-by-construction to the main method's data path.

    # from the repo root, inside the TARGET baseline's own uv project
    uv run --project baselines/harness/zep python -m baselines.harness.eval_harness \\
        --config baselines/harness/config.example.yaml
    uv run --project baselines/harness/zep python -m baselines.harness.eval_harness \\
        --describe zep      # print the memo class's method defaults (both arms)

Config = the EVALUATION FRAME only (see config.example.yaml): which harness,
which arm, dataset/split/sizing, the shared QA + judge models. Every
method-specific knob is a default on the memo class (`CONFIG_DEFAULTS`, with
`UNIFIED_OVERRIDES` applied for `arm: unified`) — see
common.memo_class.MemoClass. An optional `memo:` block overrides individual
defaults (validated against the class, so a typo still aborts); it is the
ablation hatch, not a documented surface. The fully merged result is written
to the run directory as config.resolved.yaml — the only record of what the
method actually ran with.

Run layout (one directory per run; nothing is ever overwritten):

    baselines/harness/<harness>/runs/
    ├── index.jsonl                  one line per run (appended after it ends)
    ├── latest                       the most recent run_id
    ├── memory_cache/<dataset>/<split>/   cross-run Phase-1 memory cache
    └── <run_id>/                    <YYYYMMDD_HHMMSS>_<dataset>_<split> | run_name
        ├── config.resolved.yaml     frame + fully expanded memo config
        ├── run.json                 harness, memo class, git sha, timing, status
        ├── run.log                  the logger tape for this run
        └── score.json  token_usage.json  stages.json  traces/  stage*/   (evaluate_memo, unchanged)
"""
from __future__ import annotations

import importlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Type

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
try:
    from dotenv import load_dotenv; load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

from common.memo_class import MemoClass

HARNESS_DIR = Path(__file__).resolve().parent

# name → "module:Class". Imported LAZILY, only for the selected harness: each
# baseline has its own venv, so a module-level import of all seven would make
# this file unimportable in every one of them.
MEMOS: Dict[str, str] = {
    "amem":      "baselines.harness.amem.memo:AMemMemo",
    "hipporag2": "baselines.harness.hipporag2.memo:HippoRAGMemo",
    "lightmem":  "baselines.harness.lightmem.memo:LightMemMemo",
    "mem0":      "baselines.harness.mem0.memo:Mem0Memo",
    "memoryos":  "baselines.harness.memoryos.memo:MemoryOSMemo",
    "simplemem": "baselines.harness.simplemem.memo:SimpleMemMemo",
    "zep":       "baselines.harness.zep.memo:ZepMemo",
}

# The evaluation frame: the config file must list EXACTLY these keys (a null
# value counts as listed; sizing is checked to the leaf) plus optionally
# `run_name` and `memo`. No method knob is ever listed here.
FRAME_KEYS = frozenset({
    "harness", "arm", "dataset", "split", "progressive", "sampling_seed",
    "single_stage", "stages", "memory_cache", "llm_model", "judge_model",
    "max_sample_concurrent",
})
OPTIONAL_KEYS = frozenset({"run_name", "memo"})


def load_memo(name: str) -> Type[MemoClass]:
    """Import the registered MemoClass for `name` (lazily — see MEMOS)."""
    if name not in MEMOS:
        raise KeyError(f"unknown harness {name!r}; known: {sorted(MEMOS)}")
    module, _, cls = MEMOS[name].partition(":")
    try:
        return getattr(importlib.import_module(module), cls)
    except ImportError as e:
        raise ImportError(
            f"harness {name!r} could not be imported ({e}) — its dependencies live "
            f"in its own venv: run with `uv run --project baselines/harness/{name}`"
        ) from e


def load_frame_config(path) -> Dict[str, Any]:
    """Load + validate the frame config (exact keys, sizing to the leaf)."""
    from common.config import load_config_file, validate_exact_config
    cfg = load_config_file(path) or {}
    core = {k: v for k, v in cfg.items() if k not in OPTIONAL_KEYS}
    validate_exact_config(core, FRAME_KEYS, context="harness config")
    if cfg["arm"] not in ("faithful", "unified"):
        raise ValueError(f"arm must be 'faithful' or 'unified', got {cfg['arm']!r}")
    if cfg.get("memo") is not None and not isinstance(cfg["memo"], dict):
        raise ValueError("`memo:` must be a mapping of <memo key>: <value>")
    return cfg


async def run_baseline(
    *,
    dataset: str,
    split: str,
    single_stage: Optional[Dict[str, Any]] = None,
    memo_class: Type[MemoClass],
    memo_config: Optional[Dict[str, Any]] = None,
    qa_model: str,
    judge_model: str,
    out_dir: Path,
    max_sample_concurrent: int = 3,
    progressive: bool = False,
    sampling_seed: int = 42,
    stages: Optional[Dict[str, Any]] = None,
    memory_cache: bool = True,
    memcache_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Evaluate one fixed memory system on a split.

    Sizing is config-driven (no sizing CLI flags — config file only):

    progressive=False (default): ONE single-stage pass sized by the REQUIRED
    `single_stage` block (a null/omitted field = the whole split; absent block
    raises ValueError — no silent whole-split). progressive=True: the staged
    stage1→2→3 gauntlet with promotion thresholds (`stages` overrides the family
    DEFAULT_STAGES). Both are one call into common.evaluate.evaluate_memo — the
    shared, execution-independent evaluator (see its docstring for the artifact
    layout: out_dir/<stage>/ + stages.json + reached-stage root copies). The
    per-run sample_seed is the fixed step-0 derivation from `sampling_seed`
    (no search steps here); a no-op at whole-split n=None. `memcache_dir`
    (default out_dir/memory_cache) is where the Phase-1 memory cache lives —
    main() passes a run-independent one so `memory_cache: true` still reuses
    a previous run's build now that every run has its own directory.

    Returns {dataset: metrics} where metrics is evaluate_memo's dict
    {raw_score, score_max, per_user_stddev, tokens, stage, eliminated}."""
    # One call into the shared, execution-independent evaluate_memo (which runs
    # the whole gauntlet — or the single pass — right here in-process). The
    # per-run sample seed is the fixed step-0 derivation (no search steps here).
    from common.evaluate import evaluate_memo
    from common.sampling import derive_sample_seed

    metrics = await evaluate_memo(
        memo_class=memo_class, memo_config=memo_config,
        dataset=dataset, split=split,
        progressive=progressive, out_dir=out_dir,
        qa_model=qa_model, judge_model=judge_model,
        stages=stages, single_stage=single_stage,
        max_sample_concurrent=max_sample_concurrent,
        sample_seed=derive_sample_seed(sampling_seed, 0, dataset),
        memory_cache=memory_cache, memcache_dir=memcache_dir,
    )
    return {dataset: metrics}


def print_result(dataset: str, progressive: bool, result: Dict[str, Any], out_dir: Path) -> None:
    """One-line run summary. run_baseline returns {dataset: metrics} for both
    the progressive gauntlet and the single-stage pass — read it uniformly."""
    m = result.get(dataset, {})
    label = "gauntlet" if progressive else "single"
    print(
        f"[{dataset}] {label} stage={m.get('stage')} "
        f"raw_score={m.get('raw_score')} eliminated={m.get('eliminated')} -> {out_dir}"
    )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _git_sha() -> Optional[str]:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def describe(name: str) -> str:
    """The memo class's method config as YAML, both arms — what replaces the
    per-baseline example files (generated from the one declaration, so it
    cannot drift). The justification for each value sits next to it in memo.py."""
    import yaml
    cls = load_memo(name)
    out = f"# {name} — {MEMOS[name]}\n# faithful arm (CONFIG_DEFAULTS):\n"
    out += yaml.safe_dump(cls.resolve_config("faithful"), sort_keys=False)
    if cls.UNIFIED_OVERRIDES:
        out += "# unified arm changes (UNIFIED_OVERRIDES):\n"
        out += yaml.safe_dump(dict(cls.UNIFIED_OVERRIDES), sort_keys=False)
    return out


def main(argv=None) -> None:
    import argparse, asyncio
    import yaml
    p = argparse.ArgumentParser(description="Harness-baseline evaluation (shared entrypoint)")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--config", help="frame YAML — the only parameter surface (see config.example.yaml)")
    g.add_argument("--describe", metavar="HARNESS", choices=sorted(MEMOS),
                   help="print that memo class's method defaults (both arms) and exit")
    a = p.parse_args(argv)
    if a.describe:
        print(describe(a.describe), end="")
        return

    cfg = load_frame_config(a.config)
    name, dataset, split = cfg["harness"], cfg["dataset"], cfg["split"]
    run_id = cfg.get("run_name") or f"{datetime.now():%Y%m%d_%H%M%S}_{dataset}_{split}"
    runs_dir = HARNESS_DIR / name / "runs"
    run_dir = runs_dir / run_id
    if run_dir.exists():
        raise FileExistsError(f"run directory already exists: {run_dir} (pick another run_name)")
    run_dir.mkdir(parents=True)
    # common.logger resolves these lazily at first get_logger — set before
    # anything that logs is imported, so the tape lands in the run directory.
    os.environ["EVALS_LOG_DIR"] = str(run_dir)
    os.environ["MEMEVOL_LOG_FILE"] = "run.log"

    memo_class = load_memo(name)
    memo_config = memo_class.resolve_config(cfg["arm"], cfg.get("memo"))
    with (run_dir / "config.resolved.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump({**{k: v for k, v in cfg.items() if k != "memo"}, "memo": memo_config},
                       f, sort_keys=False)

    record: Dict[str, Any] = {
        "run_id": run_id, "harness": name, "memo_class": MEMOS[name], "arm": cfg["arm"],
        "dataset": dataset, "split": split, "git_sha": _git_sha(),
        "python": platform.python_version(),
        "started": datetime.now().isoformat(timespec="seconds"), "status": "running",
    }
    _write_json(run_dir / "run.json", record)

    t0 = time.time()
    try:
        result = asyncio.run(run_baseline(
            dataset=dataset, split=split,
            single_stage=cfg["single_stage"], stages=cfg["stages"],
            memo_class=memo_class, memo_config=memo_config,
            qa_model=cfg["llm_model"], judge_model=cfg["judge_model"],
            out_dir=run_dir, memcache_dir=runs_dir / "memory_cache" / dataset / split,
            max_sample_concurrent=cfg["max_sample_concurrent"],
            progressive=cfg["progressive"], sampling_seed=cfg["sampling_seed"],
            memory_cache=cfg["memory_cache"],
        ))
        record["status"] = "ok"
    except BaseException as e:
        record["status"] = f"failed: {type(e).__name__}: {e}"
        raise
    finally:
        record["ended"] = datetime.now().isoformat(timespec="seconds")
        record["wall_clock_s"] = round(time.time() - t0, 1)
        _write_json(run_dir / "run.json", record)

    m = result[dataset]
    with (runs_dir / "index.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "run_id": run_id, "harness": name, "arm": cfg["arm"], "dataset": dataset,
            "split": split, "raw_score": m.get("raw_score"), "stage": m.get("stage"),
            "tokens": m.get("tokens"), "wall_clock_s": record["wall_clock_s"],
            "git_sha": record["git_sha"],
        }) + "\n")
    (runs_dir / "latest").write_text(run_id + "\n", encoding="utf-8")
    print_result(dataset, cfg["progressive"], result, run_dir)


if __name__ == "__main__":
    main()
