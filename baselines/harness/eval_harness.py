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
        --describe zep      # print the method defaults and what `arm: unified` sets

Configuration is explicit end to end — nothing here, and nothing in a memo.py,
reads or writes a setting through the environment (API keys are the one
exception: they come from `.env` / the environment and are never written into
a config or a run record).

  Config file = the EVALUATION FRAME (see config.example.yaml): harness, arm,
  unified_models, dataset/split/sizing, the shared QA + judge models, and an
  optional `memo:` block.

  The method config a memo instance receives is resolved here, in order:
    1. `CONFIG_DEFAULTS` — module-level data in the harness's memo.py: the
       faithful arm (paper / upstream values, each with its justification).
    2. `arm: unified` only — `unified_models.llm` / `.embedding` written into
       the keys that memo.py's `UNIFIED_MODEL_KEYS` names (plus the embedder's
       width where the baseline needs it told).
    3. `memo:` — method hyper-parameters (retrieve_k, window size, ...). It may
       NOT set a key the arm controls: models change only through
       `arm` / `unified_models`, so a config can be read at a glance.

Run layout (one directory per run; nothing is ever overwritten):

    baselines/harness/<harness>/runs/
    ├── index.jsonl                  one line per run (appended after it ends)
    ├── latest                       the most recent run_id
    └── <run_id>/                    <YYYYMMDD_HHMMSS>_<dataset>_<split> | run_name
        ├── config.yaml              the config file as given — copy it to re-run (e.g. flip `arm`)
        ├── memo_config.resolved.yaml  the fully resolved method config — a record, not an input
        ├── run.json                 harness, memo class, git sha, timing, status
        ├── run.log                  the logger tape for this run
        └── score.json  token_usage.json  stages.json  traces/  stage*/   (evaluate_memo, unchanged)
"""
from __future__ import annotations

import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Type

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
try:
    # API keys only — see check_environment for the settings that must NOT arrive this way.
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
    # The calibration floor: stores nothing, retrieves nothing. Needs no
    # venv of its own — run it from the repo-root one.
    "no_memory": "baselines.harness.no_memory.memo:NoMemoryMemo",
    "simplemem": "baselines.harness.simplemem.memo:SimpleMemMemo",
    "zep":       "baselines.harness.zep.memo:ZepMemo",
}

# The evaluation frame: the config file must list EXACTLY these keys (a null
# value counts as listed; sizing is checked to the leaf) plus optionally
# `run_name` and `memo`. No method knob is ever listed here.
FRAME_KEYS = frozenset({
    "harness", "arm", "unified_models", "dataset", "split", "progressive",
    "sampling_seed", "single_stage", "stages", "llm_model", "judge_model",
    "max_sample_concurrent",
})
OPTIONAL_KEYS = frozenset({"run_name", "memo"})

ARMS = ("faithful", "unified")
UNIFIED_MODEL_FIELDS = ("llm", "embedding")
# Roles a memo.py's UNIFIED_MODEL_KEYS may map: the two models, plus the
# embedder's width for a baseline that must be told it (derived, never configured).
MODEL_ROLES = frozenset({"llm", "embedding", "embedding_dims"})

# Environment variables that vendored code (or the OpenAI SDK it builds on)
# reads to change behaviour, and that memo.py cannot override explicitly. Set,
# they would silently diverge a run from what its config and CONFIG_DEFAULTS
# say — so a run refuses to start instead. Checked BEFORE the memo is imported:
# some are read at import time.
_SHARED_ENV_KNOBS: Tuple[str, ...] = (
    "OPENAI_BASE_URL", "OPENAI_API_BASE",                               # redirect every OpenAI client
    "OPENROUTER_API_KEY", "OPENROUTER_API_BASE", "OPENROUTER_BASE_URL",  # lightmem/mem0 switch to OpenRouter
)
SRC_ENV_KNOBS: Dict[str, Tuple[str, ...]] = {
    "zep": (   # graphiti_core module constants, read at import
        "SEMAPHORE_LIMIT", "USE_PARALLEL_RUNTIME",
        "CHUNK_TOKEN_SIZE", "CHUNK_OVERLAP_TOKENS", "CHUNK_MIN_TOKENS", "CHUNK_DENSITY_THRESHOLD",
        "ENTITY_INDEX_NAME", "EPISODE_INDEX_NAME", "COMMUNITY_INDEX_NAME", "ENTITY_EDGE_INDEX_NAME",
        "GRAPHITI_ATTRIBUTE_MAX_LENGTH",
    ),
}


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


def memo_spec(name: str) -> Tuple[Type[MemoClass], Dict[str, Any], Dict[str, Sequence[str]]]:
    """(MemoClass, CONFIG_DEFAULTS, UNIFIED_MODEL_KEYS) for `name` — the two
    constants are module-level data in the harness's memo.py."""
    cls = load_memo(name)
    module = sys.modules[cls.__module__]
    try:
        return cls, module.CONFIG_DEFAULTS, module.UNIFIED_MODEL_KEYS
    except AttributeError as e:
        raise AttributeError(
            f"{module.__name__} must define module-level CONFIG_DEFAULTS and UNIFIED_MODEL_KEYS"
        ) from e


def check_environment(name: str, environ: Mapping[str, str] = os.environ) -> None:
    """Refuse to run when an environment variable would silently change `name`'s behaviour."""
    found = sorted(k for k in _SHARED_ENV_KNOBS + SRC_ENV_KNOBS.get(name, ()) if k in environ)
    if found:
        raise RuntimeError(
            f"harness {name!r}: environment variable(s) {found} would silently change this run "
            f"(vendored code or the OpenAI SDK reads them). Unset them (check .env too) — every "
            f"setting must come from the config file; a baseline's internal-LLM endpoint is its "
            f"`base_url` method key."
        )


def validate_arm(arm: Any, unified_models: Any) -> None:
    """`arm` is faithful|unified; `unified_models` is null for faithful and
    {llm, embedding} (an API embedding model) for unified."""
    from baselines.harness.model_config import api_embedding_dims, is_api_embedding_model
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {list(ARMS)}, got {arm!r}")
    if arm == "faithful":
        if unified_models is not None:
            raise ValueError("`unified_models` is only read under `arm: unified` — set it to null "
                             "for `arm: faithful` (the faithful arm uses each method's own models)")
        return
    if not isinstance(unified_models, dict) or set(unified_models) != set(UNIFIED_MODEL_FIELDS):
        raise ValueError(f"`arm: unified` needs `unified_models` with exactly {list(UNIFIED_MODEL_FIELDS)}, "
                         f"got {unified_models!r}")
    for field in UNIFIED_MODEL_FIELDS:
        if not isinstance(unified_models[field], str) or not unified_models[field]:
            raise ValueError(f"`unified_models.{field}` must be a model name, got {unified_models[field]!r}")
    if not is_api_embedding_model(unified_models["embedding"]):
        raise ValueError(f"`unified_models.embedding` must be an API embedding model (text-embedding-*), "
                         f"got {unified_models['embedding']!r}")
    api_embedding_dims(unified_models["embedding"])   # raises for a width we don't know


def resolve_memo_config(
    defaults: Mapping[str, Any],
    model_keys: Mapping[str, Sequence[str]],
    *,
    arm: str,
    unified_models: Optional[Mapping[str, str]] = None,
    overrides: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """The complete method config a memo instance receives: `defaults`, then
    (arm: unified) the unified models written into `model_keys`, then
    `overrides` (the config's `memo:` block).

    An override naming a key the defaults lack aborts (a typo must never become
    a silent no-op); one naming a key the arm controls aborts too — models
    change only through `arm` / `unified_models`."""
    from baselines.harness.model_config import api_embedding_dims
    validate_arm(arm, unified_models)
    bad_roles = sorted(set(model_keys) - MODEL_ROLES)
    if bad_roles:
        raise ValueError(f"UNIFIED_MODEL_KEYS has unknown role(s) {bad_roles}; allowed: {sorted(MODEL_ROLES)}")
    controlled = {key for keys in model_keys.values() for key in keys}
    undeclared = sorted(controlled - set(defaults))
    if undeclared:
        raise ValueError(f"UNIFIED_MODEL_KEYS names key(s) missing from CONFIG_DEFAULTS: {undeclared}")

    overrides = dict(overrides or {})
    unknown = sorted(set(overrides) - set(defaults))
    if unknown:
        raise KeyError(f"unknown memo config key(s) {unknown}; known: {sorted(defaults)}")
    touched = sorted(set(overrides) & controlled)
    if touched:
        raise ValueError(f"`memo:` cannot set {touched} — models are controlled by `arm` / "
                         f"`unified_models`, not the `memo:` block")

    config = dict(defaults)
    if arm == "unified":
        values = {
            "llm": unified_models["llm"],
            "embedding": unified_models["embedding"],
            "embedding_dims": api_embedding_dims(unified_models["embedding"]),
        }
        for role, keys in model_keys.items():
            for key in keys:
                config[key] = values[role]
    config.update(overrides)
    return config


def load_frame_config(path) -> Dict[str, Any]:
    """Load + validate the frame config (exact keys, sizing to the leaf, arm)."""
    from common.config import load_config_file, reject_removed_keys, validate_exact_config
    cfg = load_config_file(path) or {}
    reject_removed_keys(cfg, "harness config")
    core = {k: v for k, v in cfg.items() if k not in OPTIONAL_KEYS}
    validate_exact_config(core, FRAME_KEYS, context="harness config")
    validate_arm(cfg["arm"], cfg["unified_models"])
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
    (no search steps here); a no-op at whole-split n=None.

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
    """The faithful method config as YAML, and which keys `arm: unified` sets.
    The justification for each value sits next to it in memo.py."""
    import yaml
    _, defaults, model_keys = memo_spec(name)
    out = f"# {name} — {MEMOS[name]}\n# faithful arm (CONFIG_DEFAULTS in memo.py):\n"
    out += yaml.safe_dump(dict(defaults), sort_keys=False)
    out += "# arm: unified writes `unified_models` into these keys (UNIFIED_MODEL_KEYS):\n"
    for role, keys in model_keys.items():
        source = ("width of unified_models.embedding" if role == "embedding_dims"
                  else f"unified_models.{role}")
        out += f"#   {source} -> {', '.join(keys)}\n"
    return out


def main(argv=None) -> None:
    import argparse, asyncio
    import yaml
    from common import logger as common_logger
    p = argparse.ArgumentParser(description="Harness-baseline evaluation (shared entrypoint)")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--config", help="frame YAML — the only parameter surface (see config.example.yaml)")
    g.add_argument("--describe", metavar="HARNESS", choices=sorted(MEMOS),
                   help="print that harness's method defaults and what `arm: unified` sets, and exit")
    a = p.parse_args(argv)
    if a.describe:
        print(describe(a.describe), end="")
        return

    cfg = load_frame_config(a.config)
    name, dataset, split = cfg["harness"], cfg["dataset"], cfg["split"]
    check_environment(name)
    run_id = cfg.get("run_name") or f"{datetime.now():%Y%m%d_%H%M%S}_{dataset}_{split}"
    runs_dir = HARNESS_DIR / name / "runs"
    run_dir = runs_dir / run_id
    if run_dir.exists():
        raise FileExistsError(f"run directory already exists: {run_dir} (pick another run_name)")
    run_dir.mkdir(parents=True)
    # Before the memo is imported: modules that log grab their file handler at import.
    common_logger.configure(run_dir, "run.log")

    try:
        memo_class, defaults, model_keys = memo_spec(name)
        memo_config = resolve_memo_config(
            defaults, model_keys, arm=cfg["arm"],
            unified_models=cfg["unified_models"], overrides=cfg.get("memo"),
        )
    except BaseException:
        # Nothing ran: leave no half-made run directory behind (a retry under the
        # same run_name would otherwise hit FileExistsError).
        shutil.rmtree(run_dir, ignore_errors=True)
        raise
    shutil.copyfile(a.config, run_dir / "config.yaml")
    with (run_dir / "memo_config.resolved.yaml").open("w", encoding="utf-8") as f:
        f.write("# RECORD ONLY — the method config this run's memo instances received\n"
                "# (CONFIG_DEFAULTS -> unified_models -> memo:). Not a --config input:\n"
                "# re-run from config.yaml.\n")
        yaml.safe_dump(memo_config, f, sort_keys=False)

    record: Dict[str, Any] = {
        "run_id": run_id, "harness": name, "memo_class": MEMOS[name], "arm": cfg["arm"],
        "unified_models": cfg["unified_models"],
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
            out_dir=run_dir,
            max_sample_concurrent=cfg["max_sample_concurrent"],
            progressive=cfg["progressive"], sampling_seed=cfg["sampling_seed"],
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
            "run_id": run_id, "harness": name, "arm": cfg["arm"],
            "unified_models": cfg["unified_models"], "dataset": dataset,
            "split": split, "raw_score": m.get("raw_score"), "stage": m.get("stage"),
            "tokens": m.get("tokens"), "wall_clock_s": record["wall_clock_s"],
            "git_sha": record["git_sha"],
        }) + "\n")
    (runs_dir / "latest").write_text(run_id + "\n", encoding="utf-8")
    print_result(dataset, cfg["progressive"], result, run_dir)


if __name__ == "__main__":
    main()
