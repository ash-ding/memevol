"""Mem0 baseline — evaluate on one benchmark's split, comparable to the main
method (same split/judge/scoring via the per-dataset workflow).

Sizing is config-file only (no sizing CLI flags): `single_stage` (progressive:
false) or `stages` (progressive: true) — see config.example.yaml.

    uv run --project baselines/harness/mem0 python baselines/harness/mem0/run.py --config baselines/harness/mem0/config.example.yaml
    uv run --project baselines/harness/mem0 python baselines/harness/mem0/run.py --config baselines/harness/mem0/config.example.yaml --progressive
"""
from __future__ import annotations
import argparse, asyncio, sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
try:
    from dotenv import load_dotenv; load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass
from baselines.harness.eval_utility import run_baseline, print_result
from baselines.harness.mem0.memo import Mem0Memo
from common.config import load_config_file, validate_exact_config

# The config file must list EXACTLY these keys (a null value counts as
# listed; sizing leaves are checked separately) — no CLI overrides, no
# built-in defaults. Copy config.example.yaml and edit.
REQUIRED_KEYS = frozenset({
    "dataset",
    "split",
    "progressive",
    "sampling_seed",
    "single_stage",
    "stages",
    "memory_cache",
    "mem0_llm_model",
    "embedding_model",
    "base_url",
    "add_batch_size",
    "infer",
    "top_k",
    "threshold",
    "llm_model",
    "judge_model",
    "max_sample_concurrent",
})


def main():
    p = argparse.ArgumentParser(description="Mem0 baseline — multi-dataset")
    p.add_argument("--config", required=True,
                   help="YAML config file — the ONLY parameter surface "
                        "(no CLI overrides). Copy config.example.yaml and edit.")
    a = p.parse_args()

    cfg = validate_exact_config(load_config_file(a.config) or {},
                                REQUIRED_KEYS, context="mem0 config")

    memo_config = dict(
        mem0_llm_model=cfg["mem0_llm_model"], embedding_model=cfg["embedding_model"],
        base_url=cfg["base_url"], add_batch_size=cfg["add_batch_size"], infer=cfg["infer"],
        top_k=cfg["top_k"], threshold=cfg["threshold"],
    )
    out_dir = Path(__file__).resolve().parent / "results" / cfg["dataset"] / cfg["split"]
    result = asyncio.run(run_baseline(
        dataset=cfg["dataset"], split=cfg["split"],
        single_stage=cfg["single_stage"],
        memo_class=Mem0Memo, memo_config=memo_config,
        qa_model=cfg["llm_model"], judge_model=cfg["judge_model"],
        out_dir=out_dir,
        max_sample_concurrent=cfg["max_sample_concurrent"],
        progressive=cfg["progressive"],
        sampling_seed=cfg["sampling_seed"],
        stages=cfg["stages"],
        memory_cache=cfg["memory_cache"],
    ))
    print_result(cfg["dataset"], cfg["progressive"], result, out_dir)


if __name__ == "__main__":
    main()
