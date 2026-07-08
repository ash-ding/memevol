"""DynamicMem-specific workflow — official TCE v2 checkpoint protocol.

Unlike the other benchmarks (two-phase: ingest everything, then QA), the
official DynamicMem evaluation is CHECKPOINT-INTERLEAVED: each of the 5
checkpoints has an `as_of` cutoff; memory at cp_i reflects exactly
app_logs[:log_index+1], and cp_i's task items are answered against that
state before the next segment is ingested. `run_single_user` is overridden
accordingly; Phase-1 chunking (`update_type`) applies within each segment.

Two task families per checkpoint (both scored 0.0-1.0 by the official
holistic Core+Detail judge — see datasets/dynamicmem/tce_prompts.py):
  - Task A  state_completion  fill a state template from memory
  - Task C  apply_service     personalized service (message / structured)

Reward per user = mean over sampled item scores (judge_score_max = 1).

Deviation from the official evaluator (documented): the judge transport uses
common.llm's shared client + retry kernel (max_retries=5) instead of the
official 2-attempt loop — transport robustness only; on final failure the
item scores 0.0 exactly like the official `judge_error` path.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Dict, List, Optional, Tuple, Type

import httpx

from common.workflow import BaseWorkflow, _QAProgressTracker, log
from datasets.dynamicmem.env import (
    Basic_Recorder,
    DynamicMemRecorder,
    TASK_FAMILY_STATE_COMPLETION,
    load_user_checkpoints,
    observed_logs_for_checkpoint,
    sample_items,
    sample_items_staged,
)
from datasets.dynamicmem import tce_prompts as tce


class DynamicMemWorkflow(BaseWorkflow):
    """DynamicMem checkpoint-interleaved workflow (official TCE protocol)."""

    recorder_class: Type[Basic_Recorder] = DynamicMemRecorder
    _phase1_item_label: str = "logs"
    # Full item count per user is ~250-350 (Task A + Task C across 5 cps).
    _default_qa_per_user_hint: int = 300
    judge_score_max: int = 1  # official holistic scores are 0.0-1.0 floats

    def _qa_per_user_estimate(self, stage_spec: Optional[Dict]) -> int:
        spec = stage_spec or {}
        if "n_checkpoints" in spec:
            return int(spec["n_checkpoints"]) * (
                int(spec.get("n_task_a", 0)) + int(spec.get("n_task_c", 0))
            )
        return super()._qa_per_user_estimate(stage_spec)

    # ------------------------------------------------------------------
    # Main override: checkpoint-interleaved ingest + answer + judge
    # ------------------------------------------------------------------

    async def run_single_user(
        self,
        user_dir: str,
        *,
        stage: str = "stage3",
        stage_spec: Optional[Dict] = None,
        qa_tracker: Optional[_QAProgressTracker] = None,
    ) -> Basic_Recorder:
        user_tag = user_dir[-15:]
        app_logs, checkpoints = load_user_checkpoints(user_dir)
        spec = stage_spec or {}

        if "n_checkpoints" in spec:
            # Staged spec (forge path): first n_checkpoints checkpoints,
            # n_task_a / n_task_c items per checkpoint (nested sampling).
            checkpoints_used = checkpoints[: int(spec["n_checkpoints"])]
            sampled = sample_items_staged(
                checkpoints,
                n_checkpoints=int(spec["n_checkpoints"]),
                n_task_a=int(spec.get("n_task_a", 0)),
                n_task_c=int(spec.get("n_task_c", 0)),
                seed=user_dir,
            )
        else:
            # Legacy total-count spec (alma baseline path): n_qa items
            # stratified across checkpoints; sanity stage = first cp only.
            checkpoints_used = checkpoints[:1] if stage == "sanity" else checkpoints
            sampled = sample_items(
                checkpoints_used, spec.get("n_qa", self.eval_n_qa), seed=user_dir
            )

        sampled_by_cp: Dict[str, List[Dict]] = {}
        for item in sampled:
            sampled_by_cp.setdefault(item["checkpoint_id"], []).append(item)

        memo = self.memo_class()
        recorder = self.recorder_class()
        recorder.user_id = user_tag
        await self.phase1_log_init(recorder, app_logs)

        from common.memory_cache import user_key as _mc_user_key
        ukey = _mc_user_key(user_dir)

        prev_end = 0
        total_ingested = 0
        phase2_errs: List[Tuple[str, str]] = []   # (kind, first-error text)
        t0 = time.time()
        for cp_idx, cp in enumerate(checkpoints_used, 1):
            visible = observed_logs_for_checkpoint(cp, app_logs)
            cp_meta = {
                "checkpoint_id": cp["checkpoint_id"],
                "log_end": len(visible),
            }
            # Cross-stage memory cache — PER-CHECKPOINT snapshots. Checkpoint
            # isolation is a hard correctness constraint: cp_i's items must be
            # answered with memory reflecting exactly app_logs[:log_index_i+1],
            # and later stages still answer items at EARLY checkpoints, so the
            # snapshot for each cp is loaded (replacing the memo) rather than
            # reusing the final state.
            loaded = self._cache_load(f"{ukey}__cp{cp_idx}", extra_meta=cp_meta)
            if loaded is not None:
                memo = loaded
                prev_end = max(prev_end, len(visible))
            else:
                segment = app_logs[prev_end: len(visible)]
                prev_end = max(prev_end, len(visible))
                if segment:
                    t1 = time.time()
                    await self._phase1_update(memo, segment)
                    total_ingested += len(segment)
                    log.info(
                        f"[Checkpoint {cp_idx}/{len(checkpoints_used)}] User {user_tag} "
                        f"ingested {len(segment)} logs ({total_ingested} total, {time.time() - t1:.1f}s)"
                    )
                self._cache_save(memo, f"{ukey}__cp{cp_idx}", extra_meta=cp_meta)

            for item in sampled_by_cp.get(cp["checkpoint_id"], []):
                err = await self._run_item(memo, recorder, item, visible)
                if err is not None:
                    phase2_errs.append(err)
                if qa_tracker:
                    await qa_tracker.increment()

        # Phase-2 error tally → failure_info (the orchestrator's sanity gate
        # inspects it; see BaseWorkflow.run_single_user for the same pattern).
        if phase2_errs:
            kinds = {k: sum(1 for e, _ in phase2_errs if e == k) for k in ("retrieve", "answer")}
            recorder.failure_info = (
                f"phase2_errors: retrieve={kinds['retrieve']}/{len(sampled)} "
                f"answer={kinds['answer']}/{len(sampled)}; first: {phase2_errs[0][1]}"
            )

        scores = [s["score"] for s in recorder.steps]
        avg = sum(scores) / len(scores) if scores else 0.0
        await recorder.set_reward(avg)

        # Official-style per-task aggregates for reference (trace-level only;
        # the frontier axis is the plain item mean above).
        a_field_scores = [
            float(j.get("score_0_1") or 0.0)
            for s in recorder.steps
            if s["qa_metadata"].get("task_family") == TASK_FAMILY_STATE_COMPLETION
            for j in s["qa_metadata"].get("field_judgments", [])
        ]
        c_item_scores = [
            s["score"] for s in recorder.steps
            if s["qa_metadata"].get("task_family") != TASK_FAMILY_STATE_COMPLETION
        ]
        log.info(
            f"[Phase 2] User {user_tag} TCE complete: reward={avg:.3f}/1 "
            f"({len(scores)} items, {time.time() - t0:.1f}s) | "
            f"snapshot_holistic(micro-field)="
            f"{(sum(a_field_scores) / len(a_field_scores)) if a_field_scores else 0.0:.3f} | "
            f"rq3_apply_holistic(macro-item)="
            f"{(sum(c_item_scores) / len(c_item_scores)) if c_item_scores else 0.0:.3f}"
        )
        return recorder

    # ------------------------------------------------------------------
    # One task item: retrieve → answer → official holistic judge → record
    # ------------------------------------------------------------------

    async def _run_item(
        self,
        memo: Any,
        recorder: Basic_Recorder,
        item: Dict,
        visible_logs: List[Dict],
    ) -> Optional[Tuple[str, str]]:
        """Returns None on a clean run, or ("retrieve"|"answer", error text)
        when the harness/agent leg failed — the caller tallies these into
        recorder.failure_info for the sanity gate."""
        qa_metadata = self.build_qa_metadata(item)
        relevant_context = self.extract_relevant_context(item, visible_logs)

        # 1. Retrieval (the harness under evaluation). Per-item failure
        #    isolation: a bad query records score=0 and the loop continues.
        retrieve_recorder = self.recorder_class()
        retrieve_recorder.init = self.build_query_recorder_init(visible_logs, item)
        try:
            retrieved = await memo.general_retrieve(retrieve_recorder)
        except Exception as exc:
            log.warning(
                f"general_retrieve failed for {recorder.user_id} on "
                f"{item['checkpoint_id']}/{item['state_key']} "
                f"({type(exc).__name__}: {str(exc)[:120]}) — recording score=0"
            )
            await self.log_qa_step(
                recorder=recorder, query=item["query"], predicted="",
                reference=self._reference_str(item), score=0.0,
                judge_reason=f"[Phase2_Retrieve_ERROR] {type(exc).__name__}: {str(exc)[:200]}",
                qa_metadata=qa_metadata, retrieved_memory={},
                relevant_context=relevant_context,
            )
            return ("retrieve", f"{type(exc).__name__}: {str(exc)[:200]}")

        # 2. Answer generation (official prompt; official default answer_query
        #    behavior — pipeline.py:1147-1215).
        memory_blocks = tce.retrieved_to_memory_blocks(retrieved)
        prompt = self._build_answer_prompt(item, memory_blocks)

        from common.llm import Agent
        agent = Agent(system_prompt="", model=self.model)
        answer_err: Optional[Tuple[str, str]] = None
        try:
            raw_answer = await agent.ask(prompt, reasoning_effort=self.reasoning_effort)
        except Exception as exc:
            # Keep the official empty-answer semantics (parse failure → 0)
            # but surface the transport error to the failure_info tally.
            log.warning(f"Answer agent failed for {recorder.user_id}: {exc}")
            answer_err = ("answer", f"{type(exc).__name__}: {str(exc)[:200]}")
            raw_answer = ""

        # 3. Official holistic Core+Detail judging (0.0-1.0).
        score, judge_reason, predicted_str, extra_meta = await self._judge_item(item, raw_answer)
        qa_metadata.update(extra_meta)

        await self.log_qa_step(
            recorder=recorder, query=item["query"], predicted=predicted_str,
            reference=self._reference_str(item), score=score,
            judge_reason=judge_reason, qa_metadata=qa_metadata,
            retrieved_memory=retrieved, relevant_context=relevant_context,
        )
        return answer_err

    def _build_answer_prompt(self, item: Dict, memory_blocks: List[str]) -> str:
        if item["task_family"] == TASK_FAMILY_STATE_COMPLETION:
            return tce.build_state_completion_prompt(
                task_query=item["answer_query_text"],
                state_key=item["state_key"],
                answer_template=item["answer_template"],
                memory_blocks=memory_blocks,
            )
        return tce.build_task_c_prompt(
            task_body=item["query"],
            service_family=item["service_family"],
            output_template=item["output_template"],
            memory_blocks=memory_blocks,
        )

    @staticmethod
    def _reference_str(item: Dict) -> str:
        ref = item.get("reference")
        return ref if isinstance(ref, str) else json.dumps(ref, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Judging
    # ------------------------------------------------------------------

    async def _judge_item(
        self, item: Dict, raw_answer: str
    ) -> Tuple[float, str, str, Dict]:
        """Returns (score_0_1, judge_reason, predicted_str, extra_metadata)."""
        parsed = tce.parse_json_output(raw_answer) or {}

        if item["task_family"] == TASK_FAMILY_STATE_COMPLETION:
            state_key = item["state_key"]
            gen = tce.normalize_generation_output(parsed, [state_key])
            predicted_value = tce.align_prediction_to_template(
                tce.flatten_snapshot(gen["snapshot_state"]).get(state_key),
                item["answer_template"],
            ) if gen["snapshot_state"] else None
            evidence_records = tce.normalize_evidence_prediction(
                gen["evidence"], [state_key]
            ).get(state_key, [])

            fields = tce.snapshot_holistic_fields(item["reference"], predicted_value)
            judge_prompt = tce.build_snapshot_holistic_judge_prompt(
                state_key=state_key,
                golden_state_value=item["reference"],
                predicted_state_value=predicted_value,
                fields_to_judge=fields,
            )
            judge_out, judge_error = await self._tce_judge_call(judge_prompt)
            judgments = tce.ordered_holistic_judgments(
                fields, tce.normalize_holistic_judgments(judge_out, fields)
            )
            score = tce.snapshot_holistic_item_score(judgments)
            predicted_str = json.dumps(predicted_value, ensure_ascii=False)
            judge_item = None
        else:
            ans = tce.normalize_rq3_apply_answer_output(
                parsed, service_family=item["service_family"]
            )
            is_user_comm = item["service_family"] == "user_communication"
            # Mirrors tce_core/evaluation.py::_materialize_active_rq3_slots
            # (:496-514): predicted_answer for user_communication, otherwise
            # predicted_output.
            judge_item = {
                "service_family": item["service_family"],
                "scenario": item["scenario"],
                "task_instruction": item["task_instruction"],
                "reference_answer": item.get("reference_answer", ""),
                "reference_output": item.get("reference_output"),
                "predicted_answer": str(ans["answer"] or "") if is_user_comm else "",
                "predicted_output": None if is_user_comm else ans["answer"],
                "scoring_points": item["scoring_points"],
            }
            evidence_records = ans["evidence"]

            fields = tce.apply_holistic_fields(judge_item)
            judge_prompt = tce.build_apply_holistic_judge_prompt(
                item=judge_item, fields_to_judge=fields
            )
            judge_out, judge_error = await self._tce_judge_call(judge_prompt)
            judgments = tce.ordered_holistic_judgments(
                fields, tce.normalize_holistic_judgments(judge_out, fields)
            )
            score = tce.apply_holistic_score_from_judgments(judge_item, judgments)
            predicted_str = (
                judge_item["predicted_answer"] if is_user_comm
                else json.dumps(judge_item["predicted_output"], ensure_ascii=False)
            )

        predicted_ids = [r["app_log_id"] for r in evidence_records if r.get("app_log_id")]
        ev = tce.evidence_prf(item.get("evidence_ids", []), predicted_ids)

        n_core = sum(1 for j in judgments if j["core_correct"])
        judge_reason = (
            f"holistic {n_core}/{len(judgments)} fields core-correct"
            + (f"; JUDGE_ERROR: {judge_error}" if judge_error else "")
        )
        extra_meta = {
            "field_judgments": judgments,
            "evidence_prf": ev,
            "predicted_evidence": evidence_records,
            "judge_error": judge_error or "",
        }
        return float(score), judge_reason, predicted_str, extra_meta

    async def _tce_judge_call(self, prompt: str) -> Tuple[Optional[Dict], Optional[str]]:
        """One holistic-judge LLM call. Official config (eval_tce +
        llm_client): no temperature, no reasoning_effort, ~300s timeout;
        judge failure → (None, error) and the caller scores the item 0.0.
        Transport retries use the shared common.llm kernel (see module
        docstring for the documented deviation)."""
        from common import llm as _llm

        client = _llm._get_async_client()
        request_timeout = httpx.Timeout(300.0, connect=10.0)
        # Strip the repo-wide "/effort" suffix if present (same as Agent and
        # common.judge.Judge); the effort itself is deliberately NOT sent —
        # the official judge config passes no reasoning_effort.
        judge_model, _ = _llm._split_model_effort(self.judge_model)
        kwargs = {
            "model": judge_model,
            "messages": [{"role": "user", "content": prompt}],
        }

        async def _attempt():
            resp = await client.chat.completions.create(**kwargs, timeout=request_timeout)
            if not resp.choices:
                raise _llm.EmptyResponseError("judge returned no choices")

            from common.tokens import GLOBAL_TOKEN_TRACKER
            if GLOBAL_TOKEN_TRACKER is not None and getattr(resp, "usage", None) is not None:
                try:
                    await GLOBAL_TOKEN_TRACKER.update(model_name=self.judge_model, usage=resp.usage)
                except Exception:
                    pass
            return resp.choices[0].message.content or ""

        try:
            raw = await _llm._call_with_retries(_attempt, what="TCE-Judge", max_retries=5)
        except Exception as exc:
            log.warning(f"[TCE-Judge] failed: {exc!r}")
            return None, f"{type(exc).__name__}: {str(exc)[:200]}"
        out = tce.parse_json_output(raw)
        if out is None:
            return None, f"non-JSON judge output: {raw[:200]}"
        return out, None

    # ------------------------------------------------------------------
    # BaseWorkflow abstract hooks (run_single_user is overridden; these keep
    # the ABC satisfied and serve trace/metadata plumbing)
    # ------------------------------------------------------------------

    async def load_user_data(
        self, user_dir: str, eval_n_qa: Optional[int]
    ) -> Tuple[List[Dict], List[Dict]]:
        app_logs, checkpoints = load_user_checkpoints(user_dir)
        return app_logs, sample_items(checkpoints, eval_n_qa, seed=user_dir)

    async def phase1_log_init(self, recorder: Basic_Recorder, chunk: List[Dict]) -> None:
        """DynamicMemRecorder.log_init(app_logs)."""
        await recorder.log_init(chunk)

    def build_query_recorder_init(self, init_data: List[Dict], qa: Dict) -> Dict:
        # init_data = the app-log prefix visible at the item's checkpoint.
        return {"app_logs": init_data, "query": qa["query"]}

    def build_qa_prompt(
        self, query: str, retrieved: Dict, qa_metadata: Dict, reference: str = ""
    ) -> List[Dict]:
        # Not used by the overridden run_single_user (official prompts are
        # built per task family in _build_answer_prompt); kept for ABC compat.
        blocks = tce.retrieved_to_memory_blocks(retrieved)
        return [
            {"role": "system", "content": ""},
            {"role": "user", "content": tce.render_inline_memory_section(blocks) + "\n\n" + query},
        ]

    def extract_relevant_context(self, qa: Dict, init_data: List[Dict]) -> List[Dict]:
        evidence_ids = qa.get("evidence_ids") or qa.get("metadata", {}).get("app_log_ids", [])
        lookup = {log_entry["app_log_id"]: log_entry for log_entry in init_data}
        return [lookup[lid] for lid in evidence_ids if lid in lookup]

    def build_qa_metadata(self, qa: Dict) -> Dict:
        return {
            "task_family": qa.get("task_family", ""),
            "checkpoint_id": qa.get("checkpoint_id", ""),
            "state_key": qa.get("state_key", ""),
            "qa_id": qa.get("qa_id", ""),
            "service_family": qa.get("service_family", ""),
            "domain": qa.get("domain", ""),
            "app_log_ids": list(qa.get("evidence_ids") or []),
        }

    async def log_qa_step(
        self,
        recorder: Basic_Recorder,
        query: str,
        predicted: str,
        reference: str,
        score: float,
        judge_reason: str,
        qa_metadata: Dict,
        retrieved_memory: Dict,
        relevant_context: List[Dict],
    ) -> None:
        await recorder.log_step(
            query=query,
            predicted=predicted,
            reference=reference,
            score=score,
            judge_reason=judge_reason,
            qa_metadata=qa_metadata,
            retrieved_memory=retrieved_memory,
            relevant_app_logs=relevant_context,
        )
