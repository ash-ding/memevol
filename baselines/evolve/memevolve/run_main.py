"""
MemEvolve CLI entry point — dual-evolution loop (paper §4).

    # Meta-evolution on the SEARCH split
    python baselines/evolve/memevolve/run_main.py --status search \
        --dataset dynamicmem --iterations 3

    # Held-out TEST evaluation of a frozen genotype (default: best perf)
    python baselines/evolve/memevolve/run_main.py --status test \
        --dataset dynamicmem [--memo_SHA <sha>]

Each iteration k:
  INNER  every candidate Ω_j is evaluated from an empty memory on the
         search split → feedback F_j = (perf, −cost, −delay)
  OUTER  Pareto top-K parents → per parent: DIAGNOSE (trace evidence →
         defect profile) then DESIGN S variants (constrained to the four-
         operator design space), each sanity-checked with repair retries.
         Next candidate set = parents (elitism) + surviving variants.

Split discipline mirrors alma/forge: all inner-loop evaluation on `search`;
`test` is touched once per reported number, with the genotype frozen.
"""

from __future__ import annotations

import argparse
import asyncio
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
        description="MemEvolve — meta-evolution of memory architectures (baseline); "
                    "one benchmark per run via --dataset.")
    parser.add_argument("--status", type=str, default="search", choices=["search", "test"])
    parser.add_argument("--dataset", type=str, default="dynamicmem", choices=DATASETS)
    parser.add_argument("--tag", type=str, default="default",
                        help="Population-state tag (separate concurrent experiments).")

    # Dual-evolution shape (paper defaults: K_max=3, survivor budget small)
    parser.add_argument("--iterations", type=int, default=3,
                        help="Target TOTAL outer iterations (resumable).")
    parser.add_argument("--top_k", type=int, default=2, help="Survivor/parent budget K.")
    parser.add_argument("--variants_per_parent", type=int, default=2, help="S descendants per parent.")
    parser.add_argument("--meta_model", type=str, default="gpt-5",
                        help="Diagnose/design/repair LLM.")

    # Evaluation pass (mirrors alma's knobs)
    parser.add_argument("--execution_model", type=str, default="gpt-5-mini")
    parser.add_argument("--judge_model", type=str, default="gpt-5-mini")
    parser.add_argument("--eval_n_samples", type=int, default=6)
    parser.add_argument("--eval_n_qa", type=int, default=20)
    parser.add_argument("--max_logs", type=int, default=None)
    parser.add_argument("--max_sample_concurrent", type=int, default=3)
    parser.add_argument("--check_n_samples", type=int, default=2)
    parser.add_argument("--check_n_qa", type=int, default=3)

    # Test-time genotype source
    parser.add_argument("--memo_SHA", type=str, default=None,
                        help="test only: evaluate this archived genotype instead of best-perf.")
    return parser.parse_args()


async def search(args) -> None:
    from common.logger import get_logger
    from common.tokens import init_global_tracker
    from baselines.evolve.memevolve.eval_runner import read_feedback, run_evaluation
    from baselines.evolve.memevolve.genotype_manager import GenotypeArchive, PopulationState
    from baselines.evolve.memevolve.meta_evolver import (
        build_failure_log, design_variant, diagnose, sanity_check_with_repair,
        select_parents,
    )
    from baselines.evolve.memevolve.seed_genotypes import SEED_GENOTYPES

    log = get_logger("main")
    tracker = init_global_tracker()
    archive = GenotypeArchive(args.dataset)
    state = PopulationState(args.dataset, tag=args.tag)

    if state.completed >= args.iterations:
        log.info(f"Population log already has {state.completed} iterations ≥ target "
                 f"--iterations {args.iterations}; nothing to do.")
        return

    # Candidate set for the next iteration: seeds at k=0, else previous
    # parents (elitism) + variants archived under the last iteration.
    if state.completed == 0:
        candidate_shas = [
            archive.save(ops, {"parent": None, "iteration": 0, "seed_name": name,
                               "design_rationale": f"seed genotype `{name}`"})
            for name, ops in SEED_GENOTYPES.items()
        ]
        log.info(f"Iteration 0 seeds: {candidate_shas}")
    else:
        candidate_shas = state.iterations[-1].get("next_candidates") or state.last_parents()

    evaluated = state.all_evaluated()

    for k in range(state.completed, args.iterations):
        log.info(f"[blue]━━━━━━━ META-EVOLUTION ITERATION {k}/{args.iterations - 1} ━━━━━━━[/blue]")

        # ---- INNER LOOP: evaluate each candidate from empty memory ------
        feedback_by_sha = {}
        for sha in candidate_shas:
            if sha in evaluated:
                # Elite carried over — reuse its feedback (same split/config).
                feedback_by_sha[sha] = evaluated[sha]
                log.info(f"[k={k}] {sha}: reusing prior feedback perf={evaluated[sha]['perf']:.3f}")
                continue
            run_dir = await run_evaluation(
                sha=sha, module_path=archive.assembled_path(sha),
                dataset=args.dataset, mode="eval", status="search",
                model=args.execution_model, judge_model=args.judge_model,
                eval_n_samples=args.eval_n_samples, eval_n_qa=args.eval_n_qa,
                max_logs=args.max_logs,
                max_sample_concurrent=args.max_sample_concurrent,
            )
            fb = read_feedback(run_dir)
            feedback_by_sha[sha] = {"sha": sha, **fb,
                                    "parent": archive.read_meta(sha).get("parent")}
            log.info(f"[k={k}] {sha}: perf={fb['perf']:.3f} cost={fb['cost']:.0f} "
                     f"delay={fb['delay']:.0f}s")

        candidates = [{"sha": sha, **{key: val for key, val in fb.items() if key != "sha"}}
                      for sha, fb in feedback_by_sha.items()]
        evaluated.update(feedback_by_sha)

        # ---- OUTER LOOP: select + diagnose-and-design -------------------
        parents = select_parents(candidates, args.top_k)
        parent_shas = [p["sha"] for p in parents]

        async def _run_check(sha: str, module_path: Path):
            run_dir = await run_evaluation(
                sha=sha, module_path=module_path, dataset=args.dataset,
                mode="check", status="search",
                model=args.execution_model, judge_model=args.judge_model,
                eval_n_samples=args.eval_n_samples,
                max_logs=args.max_logs,
                max_sample_concurrent=args.max_sample_concurrent,
                check_n_samples=args.check_n_samples, check_n_qa=args.check_n_qa,
            )
            return read_feedback(run_dir)

        next_candidates = list(parent_shas)  # elitism
        if k < args.iterations - 1:          # last iteration: no design spend
            for parent in parents:
                p_ops = archive.load_operators(parent["sha"])
                failure_log = build_failure_log(Path(parent["run_dir"]), args.dataset)
                try:
                    profile = await diagnose(p_ops, parent, failure_log,
                                             meta_model=args.meta_model)
                except Exception as exc:
                    log.warning(f"diagnosis failed for {parent['sha']}: {exc}; skipping parent")
                    continue

                rationales: list = []
                for s in range(args.variants_per_parent):
                    try:
                        ops, rationale = await design_variant(
                            p_ops, profile, s, args.variants_per_parent, rationales,
                            meta_model=args.meta_model)
                    except Exception as exc:
                        log.warning(f"design s={s} failed for {parent['sha']}: {exc}")
                        continue
                    rationales.append(rationale)
                    sha = await sanity_check_with_repair(
                        ops, archive, args.dataset,
                        meta={"parent": parent["sha"], "iteration": k + 1,
                              "design_rationale": rationale,
                              "defect_profile": profile},
                        run_check=_run_check, meta_model=args.meta_model)
                    if sha and sha not in next_candidates:
                        next_candidates.append(sha)
                        log.info(f"[k={k}] variant accepted: {parent['sha']} → {sha}")

        state.iterations.append({
            "k": k,
            "candidates": [{key: c.get(key) for key in
                            ("sha", "parent", "perf", "cost", "delay", "run_dir")}
                           for c in candidates],
            "parents": parent_shas,
            "next_candidates": next_candidates,
        })
        state.save()
        candidate_shas = next_candidates

    tracker.print_summary()
    best = state.best()
    if best:
        log.info(f"Best genotype: {best['sha']} perf={best['perf']:.3f} "
                 f"(population log: {state.path})")


async def test(args) -> None:
    from common.logger import get_logger
    from baselines.evolve.memevolve.eval_runner import read_feedback, run_evaluation
    from baselines.evolve.memevolve.genotype_manager import GenotypeArchive, PopulationState

    log = get_logger("main")
    archive = GenotypeArchive(args.dataset)
    if args.memo_SHA:
        sha = args.memo_SHA
    else:
        state = PopulationState(args.dataset, tag=args.tag)
        best = state.best()
        if best is None:
            raise SystemExit(f"no evaluated genotypes in {state.path}; run --status search first")
        sha = best["sha"]
        log.info(f"TEST: best-perf genotype {sha} (search perf {best['perf']:.3f})")

    run_dir = await run_evaluation(
        sha=sha, module_path=archive.assembled_path(sha), dataset=args.dataset,
        mode="eval", status="test",
        model=args.execution_model, judge_model=args.judge_model,
        eval_n_samples=args.eval_n_samples, eval_n_qa=args.eval_n_qa,
        max_logs=args.max_logs, max_sample_concurrent=args.max_sample_concurrent,
    )
    fb = read_feedback(run_dir)
    log.info(f"TEST: perf={fb['perf']:.3f} cost={fb['cost']:.0f} "
             f"delay={fb['delay']:.0f}s → {run_dir}")


if __name__ == "__main__":
    args = parse_args()

    logs_dir = PROJECT_ROOT / "baselines" / "evolve" / "memevolve" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.environ["MEMEVOL_LOG_FILE"] = f"{args.status}_{args.dataset}_{timestamp}.log"
    os.environ.setdefault("EVALS_LOG_DIR", str(logs_dir))

    if args.status == "search":
        asyncio.run(search(args))
    else:
        asyncio.run(test(args))
