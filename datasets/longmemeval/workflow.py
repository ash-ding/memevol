"""LongMemEval workflow (s and m share the same subclass via `_variant`).

Init data is a list of session dicts, so the BaseWorkflow's default
list-oriented `_phase1_update` works without modification.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Type

from common.recorder import Basic_Recorder
from common.workflow import BaseWorkflow
from datasets.longmemeval.env import (
    LongMemEvalRecorder,
    load_user_data,
)
from datasets.longmemeval.prompts import (
    LONGMEMEVAL_JUDGE_OTHER,
    LONGMEMEVAL_JUDGE_PROMPTS,
    get_longmemeval_prompt,
)


class LongMemEvalWorkflow(BaseWorkflow):
    """Base class for LongMemEval variants. Subclasses set `_variant`."""

    recorder_class: Type[Basic_Recorder] = LongMemEvalRecorder
    _phase1_item_label: str = "sessions"
    _default_qa_per_user_hint: int = 1
    judge_score_max: int = 1   # paper-aligned binary judge (Figure 10)

    _variant: str = ""  # override in concrete subclasses: "s" or "m"

    # LongMemEval has exactly 1 QA per sample — the stage spec carries only
    # n_samples (question count); any n_qa is ignored. This keeps the
    # progress-bar total accurate.
    def _qa_per_user_estimate(self, stage_spec: Optional[Dict]) -> int:
        return 1

    # ------------------------------------------------------------------
    # Data loading — reads from the variant-specific JSON
    # ------------------------------------------------------------------

    async def load_user_data(
        self, user_dir: str, eval_n_qa: Optional[int], sample_seed: Optional[str] = None
    ) -> Tuple[List[Dict], List[Dict]]:
        sessions, _profile, qa_pairs = load_user_data(
            user_dir, eval_n_qa, variant=self._variant, sample_seed=sample_seed
        )
        return sessions, qa_pairs

    # ------------------------------------------------------------------
    # Phase 1 — init_data is already a list of session dicts, so the base
    # class's default chunker does the right thing.
    # ------------------------------------------------------------------

    async def phase1_log_init(
        self, recorder: Basic_Recorder, chunk: List[Dict]
    ) -> None:
        await recorder.log_init(chunk)

    # ------------------------------------------------------------------
    # Phase 2
    # ------------------------------------------------------------------

    def build_query_recorder_init(self, init_data: List[Dict], qa: Dict) -> Dict:
        md = qa.get("metadata", {})
        return {
            "sessions": init_data,
            "query": qa["query"],
            "question_date": md.get("question_date", ""),
        }

    def build_qa_prompt(
        self, query: str, retrieved: Dict, qa_metadata: Dict, reference: str = ""
    ) -> List[Dict]:
        # LongMemEval doesn't use reference (no adversarial category like LoCoMo cat 5).
        return get_longmemeval_prompt(
            query=query,
            memory_retrived=retrieved,
            question_date=qa_metadata.get("question_date", ""),
        )

    def extract_relevant_context(
        self, qa: Dict, init_data: List[Dict]
    ) -> List[Dict]:
        gold_ids = set(qa.get("metadata", {}).get("answer_session_ids", []))
        return [s for s in init_data if s.get("session_id") in gold_ids]

    def build_qa_metadata(self, qa: Dict) -> Dict:
        md = qa.get("metadata", {})
        return {
            "question_type": md.get("question_type", ""),
            "question_date": md.get("question_date", ""),
            "answer_session_ids": md.get("answer_session_ids", []),
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
            relevant_sessions=relevant_context,
        )

    # ------------------------------------------------------------------
    # Judge — per-question_type prompts (binary 0/1), per LongMemEval
    # paper Figure 10. Cannot reuse `_make_judge` because we need 4 Judge
    # instances; instead override `judge()` directly to dispatch.
    # ------------------------------------------------------------------

    def _make_judge_for(self, prompt_template: str):
        from common.metric import Judge
        return Judge(
            model=self.judge_model,
            prompt_template=prompt_template,
            score_min=0, score_max=self.judge_score_max,
        )

    @property
    def _judge_by_type(self) -> Dict[str, Any]:
        # Lazy build: one Judge per known question_type + one fallback.
        if not hasattr(self, "_judge_by_type_cache") or self._judge_by_type_cache is None:
            cache: Dict[str, Any] = {
                qtype: self._make_judge_for(prompt)
                for qtype, prompt in LONGMEMEVAL_JUDGE_PROMPTS.items()
            }
            cache["__other__"] = self._make_judge_for(LONGMEMEVAL_JUDGE_OTHER)
            self._judge_by_type_cache = cache
        return self._judge_by_type_cache

    async def judge(
        self,
        query: str,
        predicted: str,
        reference: str,
        qa_metadata: Optional[Dict] = None,
    ) -> Tuple[int, str]:
        qtype = (qa_metadata or {}).get("question_type", "")
        judge = self._judge_by_type.get(qtype, self._judge_by_type["__other__"])
        return await judge.score(query, predicted, reference)


class LongMemEvalSWorkflow(LongMemEvalWorkflow):
    """Short variant — ~48 sessions/sample (~120k tokens)."""
    _variant: str = "s"


class LongMemEvalMWorkflow(LongMemEvalWorkflow):
    """Medium variant — ~475 sessions/sample (~1.3M tokens)."""
    _variant: str = "m"
