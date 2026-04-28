"""LoCoMo-specific workflow.

Concrete implementation of `common.workflow.BaseWorkflow` for the LoCoMo
benchmark. Init data is a conversation dict; we override `_phase1_update`
because the base class's list-oriented chunker doesn't fit a dict shape.

Only `sample["conversation"]` and `sample["qa"]` from locomo10.json are used
in the pipeline; event_summary / observation / session_summary are ignored.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Type

from common.harness_base import Basic_Recorder, MemoStructure
from common.logger import get_logger
from common.workflow import BaseWorkflow
from datasets.locomo.env import (
    LoCoMoRecorder,
    extract_sessions,
    load_user_data,
    lookup_turns_by_dia_ids,
)
from datasets.locomo.prompts import LOCOMO_JUDGE_PROMPT, get_locomo_prompt

log = get_logger("main")


class LoCoMoWorkflow(BaseWorkflow):
    """LoCoMo two-phase per-user workflow.

    Phase 1 ingests the multi-session conversation.
    Phase 2 answers each QA question using the built memory.
    """

    recorder_class: Type[Basic_Recorder] = LoCoMoRecorder
    _phase1_item_label: str = "sessions"
    _default_qa_per_user_hint: int = 200
    judge_score_max: int = 1   # paper-aligned binary judge

    def _phase1_item_count(self, init_data) -> int:
        """Count actual sessions in the conversation dict (ignore the
        speaker_* and session_N_date_time keys)."""
        return len(extract_sessions(init_data))

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    async def load_user_data(
        self, user_dir: str, eval_n_qa: Optional[int]
    ) -> Tuple[Dict, List[Dict]]:
        conversation, _profile, qa_pairs = load_user_data(user_dir, eval_n_qa)
        return conversation, qa_pairs

    # ------------------------------------------------------------------
    # Phase 1
    # ------------------------------------------------------------------
    #
    # init_data is a conversation dict (not a list), so the base class's
    # default list-chunker cannot iterate it. We override _phase1_update
    # to extract a flat session list and dispatch per update_type.

    async def phase1_log_init(
        self, recorder: Basic_Recorder, chunk: Dict
    ) -> None:
        """Chunk is a minimal conversation subset (only the sessions we want to
        show in this Phase 1 call)."""
        await recorder.log_init(chunk)

    async def _phase1_update(
        self, memo: MemoStructure, init_data: Dict
    ) -> None:
        if not isinstance(init_data, dict):
            raise TypeError(
                f"LoCoMoWorkflow expects init_data to be a conversation dict; "
                f"got {type(init_data).__name__}"
            )

        speaker_a = init_data.get("speaker_a", "")
        speaker_b = init_data.get("speaker_b", "")
        sessions = extract_sessions(init_data)
        total = len(sessions)

        def _chunk_to_conversation(chunk_sessions) -> Dict:
            """Build a minimal conversation-shaped dict from a subset of sessions."""
            out: Dict[str, Any] = {"speaker_a": speaker_a, "speaker_b": speaker_b}
            for idx, date_time, turns in chunk_sessions:
                out[f"session_{idx}"] = turns
                if date_time:
                    out[f"session_{idx}_date_time"] = date_time
            return out

        async def _call_update(chunk_sessions) -> None:
            r = self.recorder_class()
            await self.phase1_log_init(r, _chunk_to_conversation(chunk_sessions))
            try:
                await memo.general_update(r)
            except Exception as exc:
                log.warning(f"general_update failed: {exc}")
                raise RuntimeError(
                    f"[Phase1_Update] {type(exc).__name__}: {exc}"
                ) from exc

        if self.update_type == "sequential":
            for idx, (sess_idx, _dt, _turns) in enumerate(sessions, 1):
                if idx == 1 or idx % max(1, total // 10) == 0 or idx == total:
                    log.info(
                        f"[Phase 1] general_update progress: {idx}/{total} ({idx*100//total}%)"
                    )
                await _call_update([sessions[idx - 1]])

        elif self.update_type == "chunked":
            n = max(1, self.n_chunks)
            chunk_size = max(1, (total + n - 1) // n)
            chunk_starts = list(range(0, total, chunk_size))
            for chunk_idx, i in enumerate(chunk_starts, 1):
                log.info(
                    f"[Phase 1] general_update progress: chunk {chunk_idx}/{len(chunk_starts)}"
                )
                await _call_update(sessions[i: i + chunk_size])

        else:  # all_at_once
            items = sessions[-self.max_logs:] if self.max_logs else sessions
            log.info(
                f"[Phase 1] general_update started "
                f"({len(items)} {self._phase1_item_label}, mode=all_at_once)"
            )
            await _call_update(items)

    # ------------------------------------------------------------------
    # Phase 2
    # ------------------------------------------------------------------

    def build_query_recorder_init(self, init_data: Dict, qa: Dict) -> Dict:
        """Phase 2 retrieve: pass the whole conversation + query."""
        return {"conversation": init_data, "query": qa["query"]}

    def build_qa_prompt(
        self, query: str, retrieved: Dict, qa_metadata: Dict, reference: str = ""
    ) -> List[Dict]:
        # LoCoMo dispatches by qa_metadata.category (see datasets/locomo/prompts.py).
        # cat 5 (adversarial) needs `reference` to construct the binary-choice
        # prompt; the other categories ignore it.
        return get_locomo_prompt(
            query=query,
            memory_retrived=retrieved,
            category=qa_metadata.get("category", 0),
            reference=reference,
        )

    def extract_relevant_context(self, qa: Dict, init_data: Dict) -> List[Dict]:
        evidence = qa.get("metadata", {}).get("evidence", [])
        return lookup_turns_by_dia_ids(init_data, evidence)

    def build_qa_metadata(self, qa: Dict) -> Dict:
        return {
            "category": qa.get("metadata", {}).get("category", 0),
            "evidence": qa.get("metadata", {}).get("evidence", []),
        }

    async def log_qa_step(
        self,
        recorder: Basic_Recorder,
        query: str,
        predicted: str,
        reference: str,
        score: int,
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
            relevant_turns=relevant_context,
        )

    # ------------------------------------------------------------------
    # Judge — LoCoMo paper-aligned prompt (binary 0/1)
    # ------------------------------------------------------------------

    def _make_judge(self):
        from common.judge import Judge
        return Judge(
            model=self.judge_model,
            prompt_template=LOCOMO_JUDGE_PROMPT,
            score_min=0, score_max=self.judge_score_max,
            timeout=180, max_retries=5,
        )

