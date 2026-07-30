"""
EvolveMem CLI entry point.

Invoke from the project root:

    # Self-evolution on the SEARCH split (EVALUATE–DIAGNOSE–PROPOSE–GUARD)
    python baselines/evolve/evolvemem/run_main.py --status search \
        --dataset dynamicmem --rounds 8

    # Held-out TEST evaluation of the best evolved θ*
    python baselines/evolve/evolvemem/run_main.py --status test \
        --dataset dynamicmem

Search-split rounds and the final frozen-θ* test run mirror the alma /
forge split discipline: every evolution iteration evaluates on `search`;
`test` is touched once, with the best config frozen.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")


def parse_args():
    from baselines.registry import DATASETS
    parser = argparse.ArgumentParser(
        description="EvolveMem — self-evolving retrieval configuration (baseline); "
                    "one benchmark per run via --dataset.")
    parser.add_argument("--status", type=str, default="search", choices=["search", "test"])
    parser.add_argument("--dataset", type=str, default="dynamicmem", choices=DATASETS)
    parser.add_argument("--tag", type=str, default="default",
                        help="Evolution-state tag (separate concurrent experiments).")
    parser.add_argument("--substrate", type=str, default="native",
                        choices=["native", "simplemem"],
                        help="Which memory system θ drives: native = EvolveMemMemo "
                             "(paper-description approximation); simplemem = the "
                             "vendored upstream substrate (paper-faithful setup; "
                             "NOTE its read path makes 2-4 internal LLM calls per "
                             "query — each round is several× the native cost).")

    # Evolution loop
    parser.add_argument("--rounds", type=int, default=8,
                        help="Target TOTAL rounds (resumable; re-running with the "
                             "same value is a no-op).")
    parser.add_argument("--guard", type=str, default="elitist",
                        choices=["elitist", "paper"],
                        help="elitist = official-code hill-climbing (incumbent, "
                             "acceptance threshold, change cap; DEFAULT); "
                             "paper = the paper's Eq.4 revert/explore/apply.")
    parser.add_argument("--init", type=str, default="weak",
                        choices=["weak", "default"],
                        help="Round-0 θ: weak = official weak_initial_config "
                             "(BM25-leaning minimal start; DEFAULT, matches the "
                             "official evolution setup); default = the space's "
                             "full defaults (strong start).")
    # elitist-guard knobs (official EvolutionConfig defaults)
    parser.add_argument("--acceptance_threshold", type=float, default=0.003)
    parser.add_argument("--max_changes_per_round", type=int, default=2)
    parser.add_argument("--max_consec_noaccept", type=int, default=5)
    # paper-guard knobs
    parser.add_argument("--tau_rev", type=float, default=0.05,
                        help="paper guard: revert threshold τ_rev on score drop.")
    parser.add_argument("--epsilon", type=float, default=0.01,
                        help="paper guard: stagnation/convergence threshold ε.")
    parser.add_argument("--meta_model", type=str, default="gpt-5",
                        help="Diagnosis LLM.")

    # Evaluation pass (mirrors alma's knobs)
    parser.add_argument("--execution_model", type=str, default="gpt-5-mini")
    parser.add_argument("--judge_model", type=str, default="gpt-5-mini")
    parser.add_argument("--eval_n_samples", type=int, default=6)
    parser.add_argument("--eval_n_qa", type=int, default=20)
    parser.add_argument("--max_logs", type=int, default=None)
    parser.add_argument("--max_sample_concurrent", type=int, default=3)

    # Test-time config source (default: best round from the evolution log)
    parser.add_argument("--config_round", type=int, default=None,
                        help="test only: evaluate a specific round's θ instead of best.")
    return parser.parse_args()


async def search(args) -> None:
    from common.logger import get_logger
    from common.tokens import init_global_tracker
    from baselines.evolve.evolvemem.diagnosis import build_failure_log, diagnose
    from baselines.evolve.evolvemem.eval_runner import read_score, run_evaluation
    from baselines.evolve.evolvemem.evolution import (
        EvolutionState, elitist_update, guarded_update,
    )

    log = get_logger("main")
    tracker = init_global_tracker()
    if args.substrate == "simplemem":
        from baselines.evolve.evolvemem import action_space_simplemem as space
    else:
        from baselines.evolve.evolvemem import action_space as space
    state = EvolutionState(args.dataset, tag=args.tag)

    if state.completed >= args.rounds:
        log.info(f"Evolution log already has {state.completed} rounds ≥ target "
                 f"--rounds {args.rounds}; nothing to do.")
        return
    if state.completed:
        log.info(f"Resuming from round {state.completed} (target {args.rounds}).")

    for r in range(state.completed, args.rounds):
        if state.rounds:
            config = state.next_config()
        elif args.init == "weak":
            config = space.weak_initial_config()
        else:
            config = space.clamp_config({})
        log.info(f"[blue]━━━━━━━ EVOLUTION ROUND {r}/{args.rounds - 1} ━━━━━━━[/blue]")

        # EVALUATE
        run_id = f"{args.tag}_r{r}"
        run_dir = await run_evaluation(
            run_id=run_id, config=config, dataset=args.dataset, status="search",
            model=args.execution_model, judge_model=args.judge_model,
            eval_n_samples=args.eval_n_samples, eval_n_qa=args.eval_n_qa,
            max_logs=args.max_logs, max_sample_concurrent=args.max_sample_concurrent,
            substrate=args.substrate,
        )
        score = read_score(run_dir)
        log.info(f"[ROUND {r}] score={score:.3f}")

        # DIAGNOSE
        failure_log = build_failure_log(run_dir, args.dataset)
        try:
            proposal = await diagnose(
                failure_log, config, state.rounds, args.dataset,
                meta_model=args.meta_model,
                space=space if args.substrate != "native" else None,
                max_adjustments=(args.max_changes_per_round
                                 if args.guard == "elitist" else None),
            )
        except Exception as exc:
            log.warning(f"[ROUND {r}] diagnosis failed ({exc}); recording empty proposal")
            proposal = {"root_causes": [], "adjustments": [],
                        "summary": f"diagnosis failed: {exc}"}

        # PROPOSE + GUARD
        space_arg = space if args.substrate != "native" else None
        if args.guard == "elitist":
            next_config, action, converged = elitist_update(
                state, config, score, proposal,
                acceptance_threshold=args.acceptance_threshold,
                max_changes_per_round=args.max_changes_per_round,
                max_consec_noaccept=args.max_consec_noaccept,
                space=space_arg,
            )
        else:
            next_config, action, converged = guarded_update(
                state, config, score, proposal,
                tau_rev=args.tau_rev, epsilon=args.epsilon, explore_seed=r,
                space=space_arg,
            )
        state.record_round(config, score, proposal, action, next_config, run_dir)

        best = state.best()
        log.info(f"[ROUND {r}] action={action} best={best['score']:.3f}@r{best['round']}")
        if converged:
            log.info(f"[ROUND {r}] converged (Δf < ε with no pending adjustments); stopping.")
            break

    tracker.print_summary()
    best = state.best()
    if best:
        log.info(f"Best θ*: round {best['round']} score={best['score']:.3f} "
                 f"(evolution log: {state.path})")


async def test(args) -> None:
    from common.logger import get_logger
    from baselines.evolve.evolvemem.eval_runner import read_score, run_evaluation
    from baselines.evolve.evolvemem.evolution import EvolutionState

    log = get_logger("main")
    state = EvolutionState(args.dataset, tag=args.tag)
    if args.config_round is not None:
        match = [r for r in state.rounds if r["round"] == args.config_round]
        if not match:
            raise SystemExit(f"round {args.config_round} not in evolution log {state.path}")
        chosen = match[0]
    else:
        chosen = state.best()
        if chosen is None:
            raise SystemExit(f"no completed evolution rounds in {state.path}; run --status search first")

    log.info(f"TEST: frozen θ* from round {chosen['round']} "
             f"(search score {chosen['score']:.3f})")
    run_dir = await run_evaluation(
        run_id=f"{args.tag}_best_r{chosen['round']}",
        config=chosen["config"], dataset=args.dataset, status="test",
        model=args.execution_model, judge_model=args.judge_model,
        eval_n_samples=args.eval_n_samples, eval_n_qa=args.eval_n_qa,
        max_logs=args.max_logs, max_sample_concurrent=args.max_sample_concurrent,
        substrate=args.substrate,
    )
    log.info(f"TEST score: {read_score(run_dir):.3f} → {run_dir}")


if __name__ == "__main__":
    args = parse_args()

    logs_dir = PROJECT_ROOT / "baselines" / "evolve" / "evolvemem" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.environ["MEMEVOL_LOG_FILE"] = f"{args.status}_{args.dataset}_{timestamp}.log"
    os.environ.setdefault("EVALS_LOG_DIR", str(logs_dir))

    if args.status == "search":
        asyncio.run(search(args))
    else:
        asyncio.run(test(args))
