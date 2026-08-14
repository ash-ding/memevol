"""Score an evolved theta through the SHARED evaluation path.

    cd baselines/evolve/evolvemem && uv run python run.py --config config.example.yaml

This is the half of the baseline that must NOT be the paper's: issue #21 point 4
and `baselines/README.md` both require the final number to come from
`common.evaluate.evaluate_memo`, so it sits on the same axis as forge's and every
harness baseline's. `baselines.harness.eval_utility.run_baseline` is that call's
thin, dataset-generic wrapper (sizing → `evaluate_memo`, artifact layout,
`invalid_users`); an evolve baseline reusing it is the point, not a shortcut —
the scoring path is shared by construction, only the ARTIFACT differs.

The artifact here is `EvolveMemMemo` loaded with a theta (`theta_path`), i.e.
whatever `evolve.py` produced. With `theta_path: null` it scores the named
`initial_config` instead, which is how the paper's R0 baseline is reproduced.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

from baselines.evolve.evolvemem.config_schema import load_and_validate, memo_config  # noqa: E402
from baselines.evolve.evolvemem.memo import EvolveMemMemo  # noqa: E402
from baselines.harness.eval_utility import print_result, run_baseline  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="EvolveMem — score an evolved theta")
    p.add_argument("--config", required=True,
                   help="YAML config file — copy config.example.yaml and edit")
    p.add_argument("--theta", default=None,
                   help="override theta_path (runtime knob: score a specific "
                        "archived theta without editing the config)")
    args = p.parse_args()

    cfg = load_and_validate(args.config)
    if args.theta is not None:
        cfg["theta_path"] = args.theta

    out_dir = Path(__file__).resolve().parent / "results" / cfg["dataset"] / cfg["split"]
    result = asyncio.run(run_baseline(
        dataset=cfg["dataset"],
        split=cfg["split"],
        single_stage=cfg["single_stage"],
        stages=cfg["stages"],
        progressive=cfg["progressive"],
        memo_class=EvolveMemMemo,
        memo_config=memo_config(cfg),
        qa_model=cfg["llm_model"],
        judge_model=cfg["judge_model"],
        out_dir=out_dir,
        max_sample_concurrent=cfg["max_sample_concurrent"],
        sampling_seed=cfg["sampling_seed"],
        memory_cache=cfg["memory_cache"],
    ))
    print_result(cfg["dataset"], cfg["progressive"], result, out_dir)


if __name__ == "__main__":
    main()
