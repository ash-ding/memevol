"""
DynamicMem_Workflow: per-user Phase 1 + Phase 2 execution protocol (alma-specific).

Each user goes through two phases:

  Phase 1 (Update) — controlled by update_type:
    sequential  : call general_update once per app_log entry
    chunked     : split app_logs into n_chunks blocks, call general_update per block
    all_at_once : call general_update once with all logs (truncated to max_logs if set)

  Phase 2 (Retrieve) — one call per QA question:
    - build a retrieve recorder with recorder.init['query'] = current question
    - call general_retrieve(recorder) → retrieved_memo dict
    - agent answers the question using retrieved_memo
    - LLM judge scores the answer

IMPORTANT: each user gets a FRESH MemoStructure instance (per-user isolation).
Memory built during Phase 1 is only used for that same user's Phase 2.

All output (memory dumps, full traces) is written under ``self.output_run_dir``,
which should be set by the caller (typically ``baselines/alma/launch.py``).
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

from baselines.alma.harness_base import MemoStructure, Sub_memo_layer
from baselines.alma.logger import get_logger

log = get_logger("main")


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


def _dump_memory(memo: MemoStructure, user_id: str, dump_dir: Path, full: bool = False) -> None:
    """Dump memory state after Phase 1 to dump_dir/<user_id>.json."""
    safe_user_id = user_id.replace("/", "_").replace("\\", "_")
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
        self._last_pct = -1

    async def increment(self) -> None:
        async with self._lock:
            self.completed += 1
            pct = int(self.completed / self.total * 100) if self.total > 0 else 100
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
        self.memo_class = memo_class
        self.memo_sha = ""                  # set by caller (for logging only)
        self.status = "search"              # 'search' → summary dump, 'test' → full dump
        self.update_type = update_type
        self.n_chunks = n_chunks
        self.max_logs = max_logs
        self.eval_n_qa = eval_n_qa
        self.judge_model = judge_model
        # Output directory — set by launch.py before run_all_users. Memory dumps
        # go to {output_run_dir}/memory_dumps/, full traces to {output_run_dir}/traces/.
        self.output_run_dir: Optional[Path] = None

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
        check_n_users: int = 6,
        check_n_qa: int = 3,
    ) -> Tuple[List[Any], int]:
        """Run Phase 1 + Phase 2 for every user in task_list.

        mode='check' samples check_n_users users with check_n_qa QA pairs each.

        Returns (results_list, results_len). Failed users appear as Exception
        objects (with `.user_id` attribute) in the list. The returned length
        equals len(results_list); downstream filters out Exceptions itself.
        """
        if mode == "check":
            task_list = random.sample(task_list, min(check_n_users, len(task_list)))

        qa_per_user = check_n_qa if mode == "check" else (self.eval_n_qa or 178)
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
                    try:
                        exc.user_id = user_tag
                    except Exception:
                        pass
                    return exc

        t0 = time.time()
        results = await asyncio.gather(*[run_one(u) for u in task_list])
        elapsed = time.time() - t0
        valid_count = sum(1 for r in results if not isinstance(r, Exception))
        log.info(f"Evaluation complete: {valid_count}/{len(task_list)} users succeeded in {elapsed:.1f}s")
        return list(results), len(results)

    # ------------------------------------------------------------------
    # Single-user execution
    # ------------------------------------------------------------------

    async def run_single_user(self, user_dir: str, mode: str = "eval", qa_tracker: _QAProgressTracker = None, check_n_qa: int = 3):
        """Run Phase 1 (update) then Phase 2 (retrieve + QA) for one user.

        Returns a DynamicMemRecorder with all QA steps recorded.
        """
        # Lazy imports so module-load order stays cheap and avoids cycles
        from datasets.dynamicmem.env import DynamicMemRecorder, load_user_data, judge_answer
        from datasets.dynamicmem.prompts import get_dynamicmem_prompt
        from baselines.alma.llm import Agent

        user_tag = user_dir[-15:]

        qa_size = check_n_qa if mode == "check" else self.eval_n_qa
        app_logs, _user_profile, qa_pairs = load_user_data(user_dir, qa_size)

        memo = self.memo_class()

        t1 = time.time()
        await self._phase1_update(memo, app_logs, DynamicMemRecorder)
        t1_elapsed = time.time() - t1
        log.info(f"[Phase 1] User {user_tag} update complete ({len(app_logs)} logs, {t1_elapsed:.1f}s)")

        # Dump memory database for post-hoc inspection (only in eval mode)
        if mode == "eval" and self.output_run_dir is not None:
            _dump_memory(memo, user_tag, self.output_run_dir / "memory_dumps", full=(self.status == "test"))

        t2 = time.time()
        recorder = DynamicMemRecorder()
        recorder.user_id = user_tag
        await recorder.log_init(app_logs)

        app_log_lookup = {log["app_log_id"]: log for log in app_logs}

        agent = Agent(system_prompt="", model=self.model, timeout=300)

        for qa in qa_pairs:
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

            app_log_ids = qa.get("metadata", {}).get("app_log_ids", [])
            relevant_app_logs = [
                app_log_lookup[lid] for lid in app_log_ids if lid in app_log_lookup
            ]

            prompt = get_dynamicmem_prompt(qa["query"], retrieved)
            system_msg = prompt[0]["content"]
            user_msg = prompt[1]["content"]

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
        """Dispatch general_update calls according to update_type."""

        async def _call_update(logs_chunk: List[Dict]) -> None:
            r = recorder_class()
            await r.log_init(logs_chunk)
            try:
                await memo.general_update(r)
            except Exception as exc:
                log.warning(f"general_update failed: {exc}")
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
            chunk_size = max(1, (total + n - 1) // n)
            chunks = list(range(0, total, chunk_size))
            for chunk_idx, i in enumerate(chunks, 1):
                log.info(f"[Phase 1] general_update progress: chunk {chunk_idx}/{len(chunks)}")
                await _call_update(app_logs[i: i + chunk_size])

        else:  # all_at_once
            logs = app_logs[-self.max_logs:] if self.max_logs else app_logs
            log.info(f"[Phase 1] general_update started ({len(logs)} logs, mode=all_at_once)")
            await _call_update(logs)

    # ------------------------------------------------------------------
    # Full-trace persistence (no sampling)
    # ------------------------------------------------------------------

    def save_full_traces(self, recorder_list: List[Any]) -> None:
        """Write every user's full QA trajectory to output_run_dir/traces/<user_id>.json.

        Failed users (Exception objects) are skipped; partial recorders are
        serialized with their failure_info preserved.
        """
        if self.output_run_dir is None:
            log.warning("save_full_traces: output_run_dir not set, skipping")
            return

        traces_dir = self.output_run_dir / "traces"
        traces_dir.mkdir(parents=True, exist_ok=True)

        for rec in recorder_list:
            if isinstance(rec, Exception):
                continue
            user_id = getattr(rec, "user_id", "") or "unknown"
            steps = getattr(rec, "steps", [])
            payload = {
                "user_id": user_id,
                "reward": float(getattr(rec, "reward", 0.0)),
                "failure_info": getattr(rec, "failure_info", None),
                "n_qa": len(steps),
                "steps": steps,
            }
            safe = user_id.replace("/", "_").replace("\\", "_") or "unknown"
            file_path = traces_dir / f"{safe}.json"
            try:
                with file_path.open("w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2, ensure_ascii=False, cls=_MemoryEncoder)
            except Exception as e:
                log.warning(f"Failed to save trace for {user_id}: {e}")
