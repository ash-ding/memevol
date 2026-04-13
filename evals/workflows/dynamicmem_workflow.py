"""
DynamicMem_Workflow: per-user Phase 1 + Phase 2 execution protocol.

Each user goes through two phases:

  Phase 1 (Update) — controlled by update_type:
    sequential  : call general_update once per app_log entry (len(app_logs) calls)
    chunked     : split app_logs into n_chunks blocks, call general_update once per block
    all_at_once : call general_update once with all logs (truncated to max_logs if set)

  Phase 2 (Retrieve) — one call per QA question:
    - build a retrieve recorder with recorder.init['query'] = current question
    - call general_retrieve(recorder) → retrieved_memo dict
    - agent answers the question using retrieved_memo
    - LLM judge scores the answer

IMPORTANT: each user gets a FRESH MemoStructure instance (per-user isolation).
Memory built during Phase 1 is only used for that same user's Phase 2.
"""

from __future__ import annotations

import asyncio
import os
import random
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple, Type

try:
    from logger import get_logger
    log = get_logger("main")
except Exception:
    import logging
    log = logging.getLogger("main")

import json
from collections import Counter, defaultdict
from datetime import datetime, date
from pathlib import Path

from agents.memo_structure import MemoStructure, Sub_memo_layer
from eval_envs.base_envs import Basic_Recorder


class _MemoryEncoder(json.JSONEncoder):
    """JSON encoder that handles common non-serializable types in memory databases."""

    def default(self, obj):
        if isinstance(obj, set):
            return sorted(obj) if all(isinstance(x, str) for x in obj) else list(obj)
        if isinstance(obj, (Counter, defaultdict)):
            return dict(obj)
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if hasattr(obj, '__float__'):
            return float(obj)
        if hasattr(obj, '__int__'):
            return int(obj)
        try:
            return super().default(obj)
        except TypeError:
            return repr(obj)


def _count_entries(db) -> dict:
    """Count entries in a dict/list database (one level)."""
    if isinstance(db, list):
        return {"_total": len(db)}
    if isinstance(db, dict):
        counts = {}
        for k, v in db.items():
            if isinstance(v, (list, set)):
                counts[k] = len(v)
            elif isinstance(v, dict):
                counts[k] = len(v)
            else:
                counts[k] = 1
        return counts
    return {}


def _dump_layer(attr, full: bool) -> dict:
    """Extract data from a single Sub_memo_layer.

    full=True  → complete database content (for status=test).
    full=False → statistics summary only (for status=search).
    """
    layer_data = {"layer_intro": getattr(attr, "layer_intro", "")}
    db = attr.database

    if db is None:
        layer_data["status"] = "empty"
        layer_data["n_entries"] = 0
        return layer_data

    db_type = type(db).__name__

    # --- NetworkX graphs ---
    if db_type in ("Graph", "DiGraph"):
        import networkx as nx
        n_nodes = db.number_of_nodes()
        n_edges = db.number_of_edges()
        layer_data["type"] = "networkx"
        layer_data["n_nodes"] = n_nodes
        layer_data["n_edges"] = n_edges
        layer_data["status"] = "ok" if n_nodes > 0 else "empty"
        type_counts = {}
        for _, d in db.nodes(data=True):
            t = d.get("type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1
        layer_data["node_type_counts"] = type_counts
        layer_data["sample_nodes"] = list(db.nodes())[:5]
        if full:
            layer_data["database"] = nx.node_link_data(db)
        return layer_data

    # --- Chroma vector store ---
    if "Chroma" in db_type:
        try:
            chroma_data = db.get(include=["documents", "metadatas"])
            ids = chroma_data.get("ids", [])
            docs = chroma_data.get("documents", [])
            metas = chroma_data.get("metadatas", [])
            layer_data["type"] = "chroma"
            layer_data["n_documents"] = len(ids)
            layer_data["status"] = "ok" if len(ids) > 0 else "empty"
            layer_data["sample_documents"] = docs[:3]
            if metas:
                layer_data["metadata_keys"] = sorted(set(k for m in metas[:10] for k in (m or {}).keys()))
            if full:
                layer_data["database"] = {"ids": ids, "documents": docs, "metadatas": metas}
        except Exception as e:
            layer_data["type"] = "chroma_error"
            layer_data["status"] = f"extraction failed: {e}"
        return layer_data

    # --- Plain dict/list ---
    if isinstance(db, (dict, list)):
        layer_data["type"] = "dict" if isinstance(db, dict) else "list"
        layer_data["entry_counts"] = _count_entries(db)
        layer_data["status"] = "ok" if len(db) > 0 else "empty"
        if isinstance(db, dict):
            layer_data["top_level_keys"] = list(db.keys())[:20]
        if full:
            layer_data["database"] = db
        return layer_data

    # --- Unknown type ---
    layer_data["type"] = db_type
    layer_data["status"] = "unknown_type"
    if full:
        layer_data["database"] = repr(db)
    return layer_data


def _dump_memory(memo: MemoStructure, user_id: str, memo_sha: str, full: bool = False) -> None:
    """Dump memory state after Phase 1 for post-hoc inspection.

    full=True  → complete database content (status=test).
    full=False → statistics summary only (status=search).
    """
    safe_user_id = user_id.replace("/", "_").replace("\\", "_")
    dump_dir = Path(os.environ.get("EVALS_LOG_DIR", "logs")) / "dynamicmem" / "memory_dumps" / memo_sha
    dump_dir.mkdir(parents=True, exist_ok=True)

    dump = {}
    for attr_name in dir(memo):
        if attr_name.startswith("_"):
            continue
        attr = getattr(memo, attr_name, None)
        if not isinstance(attr, Sub_memo_layer):
            continue
        dump[attr_name] = _dump_layer(attr, full=full)

    file_path = dump_dir / f"{safe_user_id}.json"
    try:
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(dump, f, indent=2, ensure_ascii=False, cls=_MemoryEncoder)
    except Exception as e:
        log.warning(f"Failed to dump memory for {user_id}: {e}")


class _QAProgressTracker:
    """Shared counter for QA progress across concurrent users."""

    def __init__(self, total: int):
        self.total = total
        self.completed = 0
        self._lock = asyncio.Lock()
        self._last_pct = -1  # track last logged percentage to avoid spam

    async def increment(self) -> None:
        async with self._lock:
            self.completed += 1
            pct = int(self.completed / self.total * 100) if self.total > 0 else 100
            # Log every 10% milestone
            if pct // 10 > self._last_pct // 10:
                self._last_pct = pct
                log.info(f"[Phase 2] QA Progress: {self.completed}/{self.total} ({pct}%)")


class DynamicMem_Workflow:
    def __init__(
        self,
        memo_class: Type[MemoStructure],
        model: str,
        update_type: str = "all_at_once",
        n_chunks: int = 5,
        max_logs: Optional[int] = None,
        eval_n_qa: Optional[int] = None,
        judge_model: str = "gpt-5-mini",
    ):
        self.memo_class = memo_class        # CLASS, not instance — fresh per user
        self.memo_sha = ""                 # set by caller for memory dump paths
        self.status = "search"             # set by caller: 'search' → summary dump, 'test' → full dump
        self.update_type = update_type      # 'sequential' | 'chunked' | 'all_at_once'
        self.n_chunks = n_chunks
        self.max_logs = max_logs
        self.eval_n_qa = eval_n_qa
        self.judge_model = judge_model

        # Parse "model/reasoning_effort" format (e.g. "gpt-4o-mini/low")
        if "/" in model:
            self.model, self.reasoning_effort = model.split("/", 1)
        else:
            self.model = model
            self.reasoning_effort = None

    # ------------------------------------------------------------------
    # Main entry point: run all users
    # ------------------------------------------------------------------

    async def run_all_users(
        self,
        task_list: List[str],
        mode: str = "eval",
        max_user_concurrent: int = 6,
        check_n_users: int = 3,
        check_n_qa: int = 10,
    ) -> Tuple[List[Any], int]:
        """Run Phase 1 + Phase 2 for every user in task_list.

        mode='check' samples check_n_users users with check_n_qa QA pairs each for a quick sanity check.

        Returns (results_list, results_len).
        results_list may contain DynamicMemRecorder objects or Exception objects.
        results_len equals len(results_list); downstream code uses it as a slice
        bound and filters out Exception items internally.
        """
        if mode == "check":
            task_list = random.sample(task_list, min(check_n_users, len(task_list)))

        # Compute total QA count for progress tracking
        qa_per_user = check_n_qa if mode == "check" else (self.eval_n_qa or 178)  # ~178 QA per user
        total_qa = len(task_list) * qa_per_user
        qa_tracker = _QAProgressTracker(total_qa)
        log.info(f"Starting evaluation: {len(task_list)} users × ~{qa_per_user} QA = ~{total_qa} total")

        semaphore = asyncio.Semaphore(max_user_concurrent)

        async def run_one(user_dir: str):
            async with semaphore:
                try:
                    return await self.run_single_user(user_dir, mode=mode, qa_tracker=qa_tracker, check_n_qa=check_n_qa)
                except Exception as exc:
                    user_tag = user_dir[-15:]
                    log.error(f"[ERROR] User {user_tag} failed: {exc}\n{traceback.format_exc()}")
                    # Attach user_id so launch.py can surface it in error_info
                    # (reflection LLM needs to know *which* user triggered the bug).
                    try:
                        exc.user_id = user_tag
                    except Exception:
                        pass  # some builtin exceptions forbid attribute assignment
                    return exc

        t0 = time.time()
        results = await asyncio.gather(*[run_one(u) for u in task_list])
        elapsed = time.time() - t0
        valid_count = sum(1 for r in results if not isinstance(r, Exception))
        log.info(f"Evaluation complete: {valid_count}/{len(task_list)} users succeeded in {elapsed:.1f}s")
        # Return total length, not valid_count: get_meta_eval_info uses this as a slice
        # index and skips Exception items internally. Returning valid_count would chop off
        # valid recorders that follow failed ones in the gather order.
        return list(results), len(results)

    # ------------------------------------------------------------------
    # Single-user execution
    # ------------------------------------------------------------------

    async def run_single_user(self, user_dir: str, mode: str = "eval", qa_tracker: _QAProgressTracker = None, check_n_qa: int = 10):
        """Run Phase 1 (update) then Phase 2 (retrieve + QA) for one user.

        Returns a DynamicMemRecorder with all QA steps recorded.
        """
        # Lazy imports to avoid circular import issues at module load time
        from envs.dynamicmem_env import DynamicMemRecorder, load_user_data, judge_answer
        from envs.prompts.dynamicmem_prompt import get_dynamicmem_prompt
        from utils.hire_agent import Agent

        user_tag = user_dir[-15:]

        # 1. Load data — test mode uses only 5 QA pairs for quick sanity check
        qa_size = check_n_qa if mode == "check" else self.eval_n_qa
        app_logs, _user_profile, qa_pairs = load_user_data(user_dir, qa_size)

        # 2. Fresh MemoStructure — isolated per user
        memo = self.memo_class()

        # 3. Phase 1: build memory from app logs
        t1 = time.time()
        await self._phase1_update(memo, app_logs, DynamicMemRecorder)
        t1_elapsed = time.time() - t1
        log.info(f"[Phase 1] User {user_tag} update complete ({len(app_logs)} logs, {t1_elapsed:.1f}s)")

        # 3.5. Dump memory database for post-hoc inspection (only in eval mode)
        if mode == "eval" and self.memo_sha:
            _dump_memory(memo, user_tag, self.memo_sha, full=(self.status == "test"))

        # 4. Phase 2: answer QA questions with retrieved memory
        t2 = time.time()
        recorder = DynamicMemRecorder()
        recorder.user_id = user_tag
        await recorder.log_init(app_logs)

        # Build lookup for fast app_log retrieval by ID
        app_log_lookup = {log["app_log_id"]: log for log in app_logs}

        agent = Agent(system_prompt="", model=self.model, timeout=300)

        for qa in qa_pairs:
            # Build a retrieve recorder with the current question injected
            retrieve_recorder = DynamicMemRecorder()
            retrieve_recorder.init = {
                "app_logs": app_logs,
                "query": qa["query"],
            }

            try:
                retrieved = await asyncio.wait_for(
                    memo.general_retrieve(retrieve_recorder), timeout=300
                )
            except asyncio.TimeoutError:
                # Break out (not raise) so the partial recorder — with all
                # previously successful QAs — is preserved. The failure reason
                # goes onto recorder.failure_info so launch.py can surface it
                # to the LLM reflection prompt.
                log.warning(
                    f"general_retrieve timed out for user {user_tag}, question: {qa['query'][:50]} "
                    f"— stopping user at {len(recorder.steps)}/{len(qa_pairs)} QAs"
                )
                recorder.failure_info = (
                    f"[Phase2_Retrieve] TimeoutError at QA {len(recorder.steps)+1}/{len(qa_pairs)}, "
                    f"query: {qa['query'][:120]}"
                )
                break
            except Exception as exc:
                log.warning(
                    f"general_retrieve failed for {user_tag}: {exc} "
                    f"— stopping user at {len(recorder.steps)}/{len(qa_pairs)} QAs"
                )
                recorder.failure_info = (
                    f"[Phase2_Retrieve] {type(exc).__name__}: {exc} "
                    f"(at QA {len(recorder.steps)+1}/{len(qa_pairs)}, query: {qa['query'][:120]})"
                )
                break

            # Look up the ground-truth app logs for this QA question
            app_log_ids = qa.get("metadata", {}).get("app_log_ids", [])
            relevant_app_logs = [
                app_log_lookup[lid] for lid in app_log_ids if lid in app_log_lookup
            ]

            # Build prompt and get agent answer
            prompt = get_dynamicmem_prompt(qa["query"], retrieved)
            system_msg = prompt[0]["content"]
            user_msg = prompt[1]["content"]

            # Reset agent messages for a fresh context per question
            agent.messages = [{"role": "system", "content": system_msg}]
            try:
                answer = await agent.ask(
                    user_msg,
                    with_history=False,
                    reasoning_effort=self.reasoning_effort,
                )
            except Exception as exc:
                log.warning(f"Agent ask failed for {user_dir}: {exc}")
                answer = ""

            # Judge the answer (1–10 scale + reason)
            score, judge_reason = await judge_answer(qa["query"], answer, qa.get("reference", ""), self.judge_model)

            await recorder.log_step(
                query=qa["query"],
                predicted=answer,
                reference=qa.get("reference", ""),
                score=score,
                judge_reason=judge_reason,
                qa_metadata={
                    "domain": qa.get("metadata", {}).get("domain", ""),
                    "belonged": qa.get("metadata", {}).get("belonged", ""),
                    "app_log_ids": app_log_ids,
                },
                retrieved_memory=retrieved,
                relevant_app_logs=relevant_app_logs,
            )

            if qa_tracker:
                await qa_tracker.increment()

        t2_elapsed = time.time() - t2
        scores = [s["score"] for s in recorder.steps]
        avg = sum(scores) / len(scores) if scores else 0.0
        await recorder.set_reward(avg)
        log.info(f"[Phase 2] User {user_tag} QA complete: reward={avg:.2f}/10 ({len(scores)} QA, {t2_elapsed:.1f}s)")
        return recorder

    # ------------------------------------------------------------------
    # Phase 1 helper
    # ------------------------------------------------------------------

    async def _phase1_update(
        self,
        memo: MemoStructure,
        app_logs: List[Dict],
        recorder_class,
    ) -> None:
        """Call general_update according to update_type.

        sequential  → one call per log entry
        chunked     → one call per chunk (n_chunks chunks)
        all_at_once → one call with all logs (truncated to max_logs if set)
        """

        async def _call_update(logs_chunk: List[Dict]) -> None:
            r = recorder_class()
            await r.log_init(logs_chunk)
            try:
                await memo.general_update(r)
            except Exception as exc:
                log.warning(f"general_update failed: {exc}")
                # Re-wrap with a phase tag so downstream error_info carries the
                # stage information. Keep the original exception type name and
                # message inside, and chain via `from exc` for traceback.
                raise RuntimeError(f"[Phase1_Update] {type(exc).__name__}: {exc}") from exc

        if self.update_type == "sequential":
            total = len(app_logs)
            for idx, log_entry in enumerate(app_logs, 1):
                if idx == 1 or idx % max(1, total // 10) == 0 or idx == total:
                    log.info(f"[Phase 1] general_update progress: {idx}/{total} ({idx*100//total}%)")
                await _call_update([log_entry])

        elif self.update_type == "chunked":
            n = max(1, self.n_chunks)
            total = len(app_logs)
            chunk_size = max(1, (total + n - 1) // n)  # ceiling division
            chunks = list(range(0, total, chunk_size))
            for chunk_idx, i in enumerate(chunks, 1):
                log.info(f"[Phase 1] general_update progress: chunk {chunk_idx}/{len(chunks)}")
                await _call_update(app_logs[i: i + chunk_size])

        else:  # all_at_once
            logs = app_logs[-self.max_logs:] if self.max_logs else app_logs
            log.info(f"[Phase 1] general_update started ({len(logs)} logs, mode=all_at_once)")
            await _call_update(logs)
