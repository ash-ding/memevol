"""EvolveMem's self-evolution loop, run on THIS repo's search split.

    cd baselines/evolve/evolvemem && uv run python evolve.py --config config.example.yaml

What this file is and is not
----------------------------
It is a data adapter plus a runner. The loop itself — EVALUATE → DIAGNOSE →
PROPOSE → GUARD, elitist acceptance, revert-on-regression, explore-on-stagnation,
convergence — is the vendored `EvolutionEngine`'s (`src/evolvemem/evolution.py`,
byte-identical @ db80b6a). Nothing here reimplements or second-guesses it: the
unit of comparison against forge is the paper's search method, so the search
method has to be the paper's.

What this file decides, per `baselines/README.md` § "Adding an evolve baseline":

* **Split discipline.** Evolution runs on `split: search`, full stop — a
  non-search split is a hard error, not a warning. Held-out numbers come only
  from a frozen theta, through `run.py`, with manager authorization.
* **Data.** Units come from the shared registry (`baselines.registry.resolve`)
  and each dataset's own `env.load_user_data`, never from a private loader — so
  the loop sees exactly the samples the eval path would.
* **Ingestion parity.** Sessions are built with `memo.init_to_sessions`, the
  SAME function the scored artifact uses, so what the loop optimises over is
  what the artifact will later ingest.
* **Archive.** Every round's theta plus the run summary is written under
  `memo_archive/<dataset>/`, so a reported number is traceable to a theta.

Internal evaluation stays the paper's own token-F1 over the search split (that
is the diagnosis module's input signal and its convergence test). This repo's
LLM judge scores only the FINAL artifact, via `run.py` → `evaluate_memo`.

dynamicmem note: the loop's internal signal uses `env.load_user_data`, which for
dynamicmem is the two-phase compat shim (last checkpoint's items). Final scoring
still goes through the official TCE checkpoint protocol. Recorded in README.md.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

from baselines.evolve.evolvemem.config_schema import load_and_validate  # noqa: E402
from baselines.evolve.evolvemem.llm_bridge import make_llm_call  # noqa: E402
from baselines.evolve.evolvemem.memo import (  # noqa: E402
    _ADAPTER_FOR_DATASET,
    _shared_embedder,
    init_to_sessions,
    load_theta,
)
from baselines.registry import resolve  # noqa: E402

HERE = Path(__file__).resolve().parent
ARCHIVE = HERE / "memo_archive"

# Which recorder.init key each dataset's `load_user_data` init_data belongs
# under — the same keys the per-dataset workflows use in Phase 1.
_INIT_KEY = {"locomo": "conversation", "longmemeval_s": "sessions",
             "dynamicmem": "app_logs"}


def _unit_size_field(dataset: str) -> str:
    """The family's native per-unit sizing field (see baselines/README.md)."""
    return {"locomo": "n_conversations", "longmemeval_s": "n_questions",
            "dynamicmem": "n_users"}[dataset]


def collect_search_split(cfg: Dict[str, Any]) -> Tuple[List[Tuple[str, str, List[Dict]]], List[Dict]]:
    """Merge the sampled search-split units into one (sessions, qa_pairs) pair.

    Mirrors upstream's `_merge_samples_for_evolution`: one evolution run sees
    the whole sampled split at once and produces ONE theta. Session ids are
    prefixed per unit so ids never collide across units.
    """
    dataset = cfg["dataset"]
    _workflow_cls, env, _recorder_cls = resolve(dataset)

    sizing = cfg.get("single_stage") or {}
    n_units = sizing.get(_unit_size_field(dataset))
    n_qa = sizing.get("n_qa")

    unit_ids = env.get_task_list("search", n_units)
    all_sessions: List[Tuple[str, str, List[Dict]]] = []
    all_qa: List[Dict] = []

    for unit_id in unit_ids:
        init_data, _profile, qa_pairs = env.load_user_data(unit_id, n_qa)
        sessions = init_to_sessions({_INIT_KEY[dataset]: init_data})
        for session_id, date_str, turns in sessions:
            all_sessions.append((f"{unit_id}::{session_id}", date_str, turns))
        for qa in qa_pairs:
            meta = qa.get("metadata") or {}
            extras = {}
            for key in ("question_date", "question_time"):
                if meta.get(key):
                    extras[key] = meta[key]
            all_qa.append({
                # Gold IS needed here: the loop's EVALUATE step scores its own
                # answers to compute the failure logs diagnosis reads. This is
                # the search split, which is what the search split is for.
                "question": qa["query"],
                "answer": qa.get("reference", ""),
                "category": int(meta.get("category", 0) or 0),
                "meta": {"extras": extras, "unit_id": unit_id},
            })

    return all_sessions, all_qa


async def run_evolution(cfg: Dict[str, Any]) -> Dict[str, Any]:
    if cfg["split"] != "search":
        raise ValueError(
            f"evolve.py runs on the search split only (got split={cfg['split']!r}). "
            "Held-out numbers come from a frozen theta via run.py, and only with "
            "manager authorization — see baselines/README.md on split discipline."
        )

    from evolvemem.evolution import EvolutionConfig, EvolutionEngine
    from evolvemem.extractor import ExtractionConfig

    dataset = cfg["dataset"]
    sessions, qa_pairs = collect_search_split(cfg)
    if not sessions or not qa_pairs:
        raise ValueError(f"search split for {dataset} yielded no data "
                         f"({len(sessions)} sessions, {len(qa_pairs)} QA)")

    print(f"[evolvemem] {dataset} search split: {len(sessions)} sessions, "
          f"{len(qa_pairs)} QA, max_rounds={cfg['max_rounds']}, "
          f"initial={cfg['initial_config']}")

    adapter = None
    adapter_name = cfg.get("benchmark_adapter") or _ADAPTER_FOR_DATASET.get(dataset)
    if adapter_name:
        from evolvemem.benchmarks import get_adapter
        adapter = get_adapter(adapter_name)

    # Starting point: an archived theta if one is given, else the named initial
    # config. Seeding from a prior theta is upstream's own `--prior` flow (used
    # for the paper's cross-benchmark transfer result) and is what lets a run be
    # extended without paying for the rounds already done.
    #
    # CAVEAT, state this wherever a resumed number is reported: resuming is NOT
    # identical to having run the rounds continuously. The engine restarts its
    # round counter and loses `attempt_history`, so diagnosis cannot see which
    # proposals were already tried and rejected; and memories added by targeted
    # re-extraction in the earlier run are not in the extraction cache, so the
    # store restarts from the base extraction.
    embedder = None
    resumed = bool(cfg.get("theta_path") or cfg.get("theta"))
    theta0 = load_theta(cfg) if resumed else load_theta({"initial_config": cfg["initial_config"]})
    if resumed:
        print(f"[evolvemem] RESUMING from theta {cfg.get('theta_path')} — round counter and "
              f"attempt_history restart; see README before reporting a resumed number")
    if theta0.fusion_mode != "keyword_only" and theta0.semantic_top_k > 0:
        embedder = _shared_embedder(cfg["embedding_model"])

    # One directory per RUN, not per dataset. The engine numbers its rounds from
    # zero every time, so a second run — a resume, an ablation, a different
    # max_rounds — would otherwise overwrite the first run's round_N.json and
    # silently destroy the trajectory it recorded. The stamp carries what
    # actually distinguishes runs; `cache/` stays shared on purpose, since
    # extraction is keyed by session id and is the expensive part to redo.
    stamp = (f"{cfg['initial_config']}_r{cfg['max_rounds']}"
             + ("_resumed" if resumed else "")
             + f"_{cfg['evolve_llm_model'].replace('/', '-')}")
    run_dir = ARCHIVE / dataset / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = ARCHIVE / dataset / "cache"

    engine = EvolutionEngine(
        llm_call=make_llm_call(cfg["evolve_llm_model"]),
        embedder=embedder,
        adapter=adapter,
        config=EvolutionConfig(
            max_rounds=int(cfg["max_rounds"]),
            convergence_threshold=float(cfg["convergence_threshold"]),
            initial_retrieval_config=theta0,
            extraction_config=ExtractionConfig(
                window_size=int(cfg["extraction_window_size"]),
                overlap=int(cfg["extraction_overlap"]),
            ),
            cache_dir=str(cache_dir),
            results_dir=str(run_dir / "rounds"),
        ),
    )

    # A search loop's cost is the thing you most need to see before scaling it,
    # and this baseline is the one that CAN see it (llm_bridge routes every
    # internal call through common.llm). `evaluate_memo` sets this up on the
    # scoring path; the search path has to do it itself.
    from common.tokens import init_global_tracker
    tracker = init_global_tracker()

    started = time.time()
    # The engine is synchronous; keep the event loop free so `llm_bridge`'s
    # calls can be scheduled back onto it.
    result = await asyncio.to_thread(engine.evolve, sessions, qa_pairs, None)
    token_summary = tracker.summary()

    summary = {
        "dataset": dataset,
        "split": "search",
        "initial_config": cfg["initial_config"],
        "max_rounds": cfg["max_rounds"],
        "evolve_llm_model": cfg["evolve_llm_model"],
        "n_sessions": len(sessions),
        "n_qa": len(qa_pairs),
        "duration_seconds": round(time.time() - started, 1),
        "token_usage": token_summary,
        "final_config": result.final_config,
        # Field names follow upstream's RoundResult exactly (f1 / zero_f1_count /
        # category_f1 / retrieval_config): guessing at them silently writes nulls.
        "rounds": [
            {
                "round_id": r.round_id,
                "f1": r.f1,
                "zero_f1_count": r.zero_f1_count,
                "total_questions": r.total_questions,
                "category_f1": r.category_f1,
                "memory_count": r.memory_count,
                "all_metrics": r.all_metrics,
                "improvements_applied": list(r.improvements_applied or []),
                "duration_seconds": r.duration_seconds,
                "retrieval_config": r.retrieval_config,
            }
            for r in result.rounds
        ],
    }

    (run_dir / "theta.json").write_text(
        json.dumps(result.final_config, indent=2), encoding="utf-8")
    (run_dir / "evolution_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    print(result.trajectory())
    for model, stats in (token_summary or {}).items():
        if isinstance(stats, dict):
            print(f"[evolvemem] tokens {model}: prompt={stats.get('prompt_tokens', 0):,} "
                  f"completion={stats.get('completion_tokens', 0):,} "
                  f"total={stats.get('total_tokens', 0):,}")
    print(f"[evolvemem] theta -> {run_dir / 'theta.json'}")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="EvolveMem self-evolution on the search split")
    p.add_argument("--config", required=True,
                   help="YAML config file — copy config.example.yaml and edit")
    p.add_argument("--max-rounds", type=int, default=None,
                   help="override max_rounds (a genuine runtime knob, as in alma)")
    args = p.parse_args()

    cfg = load_and_validate(args.config)
    if args.max_rounds is not None:
        cfg["max_rounds"] = args.max_rounds

    asyncio.run(run_evolution(cfg))


if __name__ == "__main__":
    main()
