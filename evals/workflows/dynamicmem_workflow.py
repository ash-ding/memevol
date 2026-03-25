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
import random
import traceback
from typing import Any, Dict, List, Optional, Tuple, Type

try:
    from logger import get_logger
    log = get_logger("main")
except Exception:
    import logging
    log = logging.getLogger("main")

from agents.memo_structure import MemoStructure
from eval_envs.base_envs import Basic_Recorder


class DynamicMem_Workflow:
    def __init__(
        self,
        memo_class: Type[MemoStructure],
        model: str,
        update_type: str = "all_at_once",
        n_chunks: int = 5,
        max_logs: Optional[int] = None,
        qa_sample_size: Optional[int] = None,
        judge_model: str = "gpt-5-mini",
    ):
        self.memo_class = memo_class        # CLASS, not instance — fresh per user
        self.update_type = update_type      # 'sequential' | 'chunked' | 'all_at_once'
        self.n_chunks = n_chunks
        self.max_logs = max_logs
        self.qa_sample_size = qa_sample_size
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
        max_concurrent: int = 5,
    ) -> Tuple[List[Any], int]:
        """Run Phase 1 + Phase 2 for every user in task_list.

        mode='test' samples 1 user with 5 QA pairs for a quick sanity check.

        Returns (results_list, valid_count).
        results_list may contain DynamicMemRecorder objects or Exception objects.
        valid_count counts only DynamicMemRecorder objects.
        """
        if mode == "test":
            random.seed(42)
            task_list = random.sample(task_list, min(1, len(task_list)))

        semaphore = asyncio.Semaphore(max_concurrent)

        async def run_one(user_dir: str):
            async with semaphore:
                try:
                    return await self.run_single_user(user_dir, mode=mode)
                except Exception as exc:
                    log.error(f"[ERROR] User {user_dir} failed: {exc}\n{traceback.format_exc()}")
                    return exc

        results = await asyncio.gather(*[run_one(u) for u in task_list])
        valid_count = sum(1 for r in results if not isinstance(r, Exception))
        return list(results), valid_count

    # ------------------------------------------------------------------
    # Single-user execution
    # ------------------------------------------------------------------

    async def run_single_user(self, user_dir: str, mode: str = "eval"):
        """Run Phase 1 (update) then Phase 2 (retrieve + QA) for one user.

        Returns a DynamicMemRecorder with all QA steps recorded.
        """
        # Lazy imports to avoid circular import issues at module load time
        from envs.dynamicmem_env import DynamicMemRecorder, load_user_data, judge_answer
        from envs.prompts.dynamicmem_prompt import get_dynamicmem_prompt
        from utils.hire_agent import Agent

        # 1. Load data — test mode uses only 5 QA pairs for quick sanity check
        qa_size = 5 if mode == "test" else self.qa_sample_size
        app_logs, user_profile, qa_pairs = load_user_data(user_dir, qa_size)

        # 2. Fresh MemoStructure — isolated per user
        memo = self.memo_class()

        # 3. Phase 1: build user profile from app logs
        await self._phase1_update(memo, app_logs, user_profile, DynamicMemRecorder)

        # 4. Phase 2: answer QA questions with retrieved memory
        recorder = DynamicMemRecorder()
        await recorder.log_init(app_logs, user_profile)

        # Build lookup for fast app_log retrieval by ID
        app_log_lookup = {log["app_log_id"]: log for log in app_logs}

        agent = Agent(system_prompt="", model=self.model)

        for qa in qa_pairs:
            # Build a retrieve recorder with the current question injected
            retrieve_recorder = DynamicMemRecorder()
            retrieve_recorder.init = {
                "app_logs": app_logs,
                "user_profile": user_profile,
                "query": qa["query"],
            }

            try:
                retrieved = await asyncio.wait_for(
                    memo.general_retrieve(retrieve_recorder), timeout=300
                )
            except asyncio.TimeoutError:
                log.warning(f"general_retrieve timed out for user {user_dir}, question: {qa['query'][:50]}")
                retrieved = {}
            except Exception as exc:
                log.warning(f"general_retrieve failed for {user_dir}: {exc}")
                retrieved = {}

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

        scores = [s["score"] for s in recorder.steps]
        avg = sum(scores) / len(scores) if scores else 0.0
        await recorder.set_reward(avg)
        log.info(f"[User {user_dir[-15:]}] reward={avg:.2f}/10 ({len(scores)} QA)")
        return recorder

    # ------------------------------------------------------------------
    # Phase 1 helper
    # ------------------------------------------------------------------

    async def _phase1_update(
        self,
        memo: MemoStructure,
        app_logs: List[Dict],
        user_profile: Dict,
        recorder_class,
    ) -> None:
        """Call general_update according to update_type.

        sequential  → one call per log entry
        chunked     → one call per chunk (n_chunks chunks)
        all_at_once → one call with all logs (truncated to max_logs if set)
        """

        async def _call_update(logs_chunk: List[Dict]) -> None:
            r = recorder_class()
            await r.log_init(logs_chunk, user_profile)
            try:
                await asyncio.wait_for(memo.general_update(r), timeout=300)
            except asyncio.TimeoutError:
                log.warning("general_update timed out")
            except Exception as exc:
                log.warning(f"general_update failed: {exc}")

        if self.update_type == "sequential":
            for log_entry in app_logs:
                await _call_update([log_entry])

        elif self.update_type == "chunked":
            n = max(1, self.n_chunks)
            total = len(app_logs)
            chunk_size = max(1, (total + n - 1) // n)  # ceiling division
            for i in range(0, total, chunk_size):
                await _call_update(app_logs[i: i + chunk_size])

        else:  # all_at_once
            logs = app_logs[-self.max_logs:] if self.max_logs else app_logs
            await _call_update(logs)
