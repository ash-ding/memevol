"""
HippoRAG2 baseline evaluation on DynamicMem benchmark.

Uses HippoRAG2's graph-based RAG pipeline:
  Phase 1 (Index): app_logs → OpenIE (NER + triples) → knowledge graph + embeddings
  Phase 2 (QA):    query → fact retrieval → reranking → PPR → top-k passages → LLM answer

Same judge as meta-learning pipeline for fair comparison.

Usage:
    # OpenAI API embedding (no GPU needed)
    python baselines/hipporag2/eval_hipporag2.py --embedding text-embedding-3-small --dry_run
    python baselines/hipporag2/eval_hipporag2.py --embedding text-embedding-3-small

    # Local GPU embedding (requires CUDA)
    python baselines/hipporag2/eval_hipporag2.py --embedding nvidia/NV-Embed-v2 --dry_run
    python baselines/hipporag2/eval_hipporag2.py --embedding nvidia/NV-Embed-v2 --embedding_batch_size 2 --embedding_dtype float16
"""

from __future__ import annotations

# Mock heavy optional deps before any hipporag import
import sys
from unittest.mock import MagicMock
sys.modules.setdefault('vllm', MagicMock())
sys.modules.setdefault('outlines.generate', MagicMock())

import argparse
import asyncio
import json
import logging
import os
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.dynamicmem.env import get_task_list, load_user_data, judge_answer
from hipporag import HippoRAG
from hipporag.utils.config_utils import BaseConfig

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

log = logging.getLogger("hipporag2_eval")
log.setLevel(logging.DEBUG)
log.propagate = False

_console = logging.StreamHandler(sys.stdout)
_console.setLevel(logging.INFO)
_console.setFormatter(logging.Formatter("%(message)s"))
log.addHandler(_console)

_file_handler: logging.FileHandler | None = None


def _init_file_logger(embedding_short: str, dry_run: bool):
    global _file_handler
    if _file_handler:
        log.removeHandler(_file_handler)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "_dryrun" if dry_run else ""
    log_path = LOG_DIR / f"hipporag2_{embedding_short}_{ts}{suffix}.log"
    _file_handler = logging.FileHandler(log_path, encoding="utf-8")
    _file_handler.setLevel(logging.DEBUG)
    _file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    log.addHandler(_file_handler)
    log.info(f"Log file: {log_path}")

    # Also route HippoRAG's internal logger to our log file
    # so NER/triple extraction progress appears in the log
    hippo_logger = logging.getLogger("hipporag")
    hippo_logger.setLevel(logging.INFO)
    hippo_logger.addHandler(_file_handler)


# ---------------------------------------------------------------------------
# App log → passage conversion
# ---------------------------------------------------------------------------

def app_log_to_passage(log_entry: dict) -> str:
    """Convert a single app_log entry to a text passage for HippoRAG indexing."""
    ts = log_entry.get("timestamp", "")
    app = log_entry.get("app_name", "")
    api = log_entry.get("api_name", "")
    req = json.dumps(log_entry.get("request", {}), ensure_ascii=False)
    resp = json.dumps(log_entry.get("response", {}), ensure_ascii=False)
    domain = log_entry.get("metadata", {}).get("domain", "")

    return (
        f"[{ts}] App: {app}, Action: {api}\n"
        f"Domain: {domain}\n"
        f"Request: {req}\n"
        f"Response: {resp}"
    )


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"


async def evaluate(
    embedding_model: str,
    llm_model: str,
    judge_model: str,
    dry_run: bool = False,
    embedding_batch_size: int | None = None,
    embedding_dtype: str | None = None,
):
    embedding_short = embedding_model.replace("/", "_").replace("nvidia_", "")
    _init_file_logger(embedding_short, dry_run)

    task_list = get_task_list(status="test", eval_n_users=4)

    log.info(f"\n{'='*60}")
    log.info(f"HippoRAG2 Baseline Evaluation")
    log.info(f"LLM: {llm_model}")
    log.info(f"Embedding: {embedding_model}")
    log.info(f"Users: {[t.split('/')[-1] for t in task_list]}")
    log.info(f"{'='*60}")

    all_results = []
    per_user = {}

    for user_dir in task_list:
        user_id = user_dir.split("/")[-1]
        user_short = user_id.split("_")[0]

        app_logs, user_profile, qa_pairs = load_user_data(user_dir, eval_n_qa=None)
        if dry_run:
            app_logs = app_logs[:50]
            qa_pairs = qa_pairs[:1]

        log.info(f"\n--- User {user_id}: {len(app_logs)} logs, {len(qa_pairs)} QA ---")

        # Convert app_logs to text passages
        passages = [app_log_to_passage(entry) for entry in app_logs]
        log.info(f"  Converted {len(passages)} app_logs to passages")

        # Per-user HippoRAG instance with isolated save_dir
        save_dir = str(OUTPUTS_DIR / f"{user_short}_{embedding_short}")
        log.info(f"  HippoRAG save_dir: {save_dir}")

        t_index_start = time.time()
        # Local GPU models (e.g. NV-Embed-v2) need smaller batch size and explicit dtype;
        # API models (text-embedding-*) use larger batches and don't need dtype.
        is_local_model = "text-embedding" not in embedding_model
        batch_size = embedding_batch_size or (4 if is_local_model else 16)
        dtype = embedding_dtype or ("float16" if is_local_model else "auto")

        config = BaseConfig(
            llm_name=llm_model,
            embedding_model_name=embedding_model,
            save_dir=save_dir,
            response_format=None,  # Avoid JSON mode issues with newer models
            temperature=1,  # gpt-5-mini only supports temperature=1
            seed=None,  # gpt-5-mini does not support seed
            embedding_batch_size=batch_size,
            embedding_model_dtype=dtype,
        )
        hippo = HippoRAG(global_config=config)

        # Phase 1: Index
        log.info(f"  Phase 1: Indexing {len(passages)} passages...")
        hippo.index(docs=passages)
        t_index = time.time() - t_index_start
        log.info(f"  Phase 1 complete: {t_index:.1f}s")

        # Phase 2: QA
        log.info(f"  Phase 2: Answering {len(qa_pairs)} questions...")
        user_scores = []

        for i, qa in enumerate(qa_pairs):
            t0 = time.time()

            try:
                # Use HippoRAG's native rag_qa pipeline
                query_solutions, response_msgs, metadata = hippo.rag_qa(
                    queries=[qa["query"]]
                )
                answer = query_solutions[0].answer if query_solutions else ""
                retrieved_docs = query_solutions[0].docs[:5] if query_solutions else []
            except Exception as e:
                log.warning(f"  [Q{i+1}] HippoRAG error: {e}")
                answer = ""
                retrieved_docs = []

            # Judge with same judge as meta-learning
            score, reason = await judge_answer(
                qa["query"], answer, qa.get("reference", ""), judge_model
            )

            elapsed = time.time() - t0
            user_scores.append(score)

            log.info(f"  [{i+1}/{len(qa_pairs)}] score={score}/10 | {elapsed:.1f}s")
            log.debug(f"  --- QA Detail ---")
            log.debug(f"    Query: {qa['query']}")
            log.debug(f"    Reference: {qa.get('reference', '')}")
            log.debug(f"    Predicted: {answer}")
            log.debug(f"    Score: {score} | Reason: {reason}")
            log.debug(f"    Retrieved docs: {len(retrieved_docs)}")

            all_results.append({
                "user_id": user_short,
                "query": qa["query"],
                "predicted": answer,
                "reference": qa.get("reference", ""),
                "score": score,
                "judge_reason": reason,
                "elapsed_s": round(elapsed, 1),
                "n_retrieved_docs": len(retrieved_docs),
            })

        if user_scores:
            avg = sum(user_scores) / len(user_scores)
            per_user[user_short] = {
                "avg_score": round(avg, 3),
                "n_qa": len(user_scores),
                "index_time_s": round(t_index, 1),
            }
            log.info(f"  User {user_short} avg: {avg:.2f}/10 ({len(user_scores)} QA, index {t_index:.0f}s)")

        if dry_run:
            break

    # Overall stats
    all_scores = [r["score"] for r in all_results]
    overall_avg = float(np.mean(all_scores)) if all_scores else 0.0

    user_rewards = [v["avg_score"] for v in per_user.values()]
    user_se = float(np.std(user_rewards, ddof=1) / np.sqrt(len(user_rewards))) if len(user_rewards) > 1 else 0.0

    output = {
        "method": "hipporag2",
        "llm_model": llm_model,
        "embedding_model": embedding_model,
        "timestamp": datetime.now().isoformat(),
        "benchmark_eval_score": {
            "benchmark_overall_eval_score": round(overall_avg, 3),
            "benchmark_overall_eval_standard_deviation": round(user_se, 3),
        },
        "per_user": per_user,
        "total_qa": len(all_results),
        "examples": all_results,
    }

    # Save
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "_dryrun" if dry_run else ""
    out_path = results_dir / f"hipporag2_{embedding_short}_{ts}{suffix}.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    log.info(f"\n{'='*60}")
    log.info(f"Results saved: {out_path}")
    log.info(f"Overall: {overall_avg:.3f} ± {user_se:.3f} (user-level SE)")
    log.info(f"Per-user: {per_user}")
    log.info(f"{'='*60}")

    return output


async def main():
    parser = argparse.ArgumentParser(description="HippoRAG2 baseline eval on DynamicMem")
    parser.add_argument("--embedding", default="text-embedding-3-small",
                        help="Embedding model (default: text-embedding-3-small)")
    parser.add_argument("--llm_model", default="gpt-5-mini",
                        help="LLM model for OpenIE, reranking, and QA (default: gpt-5-mini)")
    parser.add_argument("--judge_model", default="gpt-5-mini",
                        help="Judge model (default: gpt-5-mini)")
    parser.add_argument("--embedding_batch_size", type=int, default=None,
                        help="Embedding batch size (default: 16 for API models, 4 for local GPU models)")
    parser.add_argument("--embedding_dtype", default=None,
                        help="Embedding model dtype for local models (e.g. float16, bfloat16). Ignored for API models.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Run 1 user, 1 QA for quick verification")
    args = parser.parse_args()

    await evaluate(
        embedding_model=args.embedding,
        llm_model=args.llm_model,
        judge_model=args.judge_model,
        dry_run=args.dry_run,
        embedding_batch_size=args.embedding_batch_size,
        embedding_dtype=args.embedding_dtype,
    )


if __name__ == "__main__":
    asyncio.run(main())
