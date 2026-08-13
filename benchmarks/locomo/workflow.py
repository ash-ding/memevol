"""LoCoMo-specific workflow.

Concrete implementation of `common.workflow.BaseWorkflow` for the LoCoMo
benchmark. Init data is a conversation dict; we override `_phase1_update`
because the base class's list-oriented chunker doesn't fit a dict shape.

Only `sample["conversation"]` and `sample["qa"]` from locomo10.json are used
in the pipeline; event_summary / observation / session_summary are ignored.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Type

from common.memo_class import MemoClass
from common.logger import get_logger
from common.metric import bleu1, token_f1
from common.recorder import Basic_Recorder
from common.workflow import BaseWorkflow
from benchmarks.locomo.env import (
    CATEGORY as LOCOMO_CATEGORY,
    LoCoMoRecorder,
    extract_sessions,
    load_user_data,
    lookup_turns_by_dia_ids,
)
from benchmarks.locomo.prompts import LOCOMO_JUDGE_PROMPT, get_locomo_prompt

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
        self, user_dir: str, eval_n_qa: Optional[int], sample_seed: Optional[str] = None
    ) -> Tuple[Dict, List[Dict]]:
        conversation, _profile, qa_pairs = load_user_data(
            user_dir, eval_n_qa, sample_seed=sample_seed
        )
        return conversation, qa_pairs

    # ------------------------------------------------------------------
    # Phase 1
    # ------------------------------------------------------------------
    #
    # init_data is a conversation dict (not a list), so the base class's
    # list-shaped default doesn't fit. We override _phase1_update to hand
    # the whole conversation dict to build_memory_from_data in one call; the memo
    # chooses its own ingestion granularity internally.

    async def phase1_log_init(
        self, recorder: Basic_Recorder, chunk: Dict
    ) -> None:
        """Chunk is a minimal conversation subset (only the sessions we want to
        show in this Phase 1 call)."""
        await recorder.log_init(chunk)

    async def _phase1_update(
        self, memo: MemoClass, init_data: Dict
    ) -> None:
        if not isinstance(init_data, dict):
            raise TypeError(
                f"LoCoMoWorkflow expects init_data to be a conversation dict; "
                f"got {type(init_data).__name__}"
            )
        log.info(f"[Phase 1] build_memory_from_data (whole conversation)")
        r = self.recorder_class()
        await self.phase1_log_init(r, init_data)
        try:
            await memo.build_memory_from_data(r)
        except Exception as exc:
            log.warning(f"build_memory_from_data failed: {exc}")
            raise RuntimeError(f"[Phase1_Update] {type(exc).__name__}: {exc}") from exc

    # ------------------------------------------------------------------
    # Phase 2
    # ------------------------------------------------------------------

    def build_query_recorder_init(self, init_data: Dict, qa: Dict) -> Dict:
        """Phase 2 retrieve: pass the whole conversation + query."""
        return {"conversation": init_data, "query": qa["query"]}

    def build_qa_prompt(
        self, query: str, retrieved: Dict, qa_metadata: Dict, reference: str = ""
    ) -> List[Dict]:
        # LoCoMo dispatches by qa_metadata.category (see benchmarks/locomo/prompts.py).
        # Only categories 1-4 occur (cat-5 adversarial QAs are filtered at
        # load time); `reference` never reaches the QA prompt.
        return get_locomo_prompt(
            query=query,
            memory_retrived=retrieved,
            category=qa_metadata.get("category", 0),
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
        # Lexical metrics alongside the judge score. Computed here (not at
        # aggregation time) so every trace carries them per step and stays
        # re-aggregatable without re-running the eval. Pure stdlib, no LLM call.
        await recorder.log_step(
            query=query,
            predicted=predicted,
            reference=reference,
            score=score,
            judge_reason=judge_reason,
            qa_metadata=qa_metadata,
            retrieved_memory=retrieved_memory,
            relevant_turns=relevant_context,
            token_f1=token_f1(predicted, reference),
            bleu1=bleu1(predicted, reference),
        )

    # ------------------------------------------------------------------
    # Paper-comparable lexical aggregates (REPORTING ONLY)
    # ------------------------------------------------------------------

    def aggregate_extra_metrics(self, recorder_list: List) -> Dict:
        """Per-category token-F1 / BLEU-1, plus both means, into score.json's
        `extra_metrics`. Never feeds the promotion signal — `accuracy_locomo`
        keeps coming from the LLM judge.

        BOTH means are emitted, explicitly named, on purpose. The papers' "Avg."
        is the UNWEIGHTED mean over the four categories; LoCoMo's own mix is
        ~55% single-hop / ~6% open-domain, so the question-weighted mean is a
        materially different number. A bare `token_f1` key would invite someone
        to compare our weighted number against a paper's unweighted one and
        silently misread it, so no such key is emitted.
        """
        per_cat: Dict[str, List[Dict]] = {}
        for rec in recorder_list:
            if isinstance(rec, Exception):
                continue
            for step in getattr(rec, "steps", []) or []:
                cat = LOCOMO_CATEGORY.get(
                    (step.get("qa_metadata") or {}).get("category"), "other"
                )
                per_cat.setdefault(cat, []).append(step)

        if not per_cat:
            return {}

        def _mean(rows: List[Dict], key: str) -> float:
            # Recompute if a step predates per-step metrics (older traces).
            vals = [
                float(s[key]) if key in s
                else (token_f1 if key == "token_f1" else bleu1)(
                    s.get("predicted", ""), s.get("reference", "")
                )
                for s in rows
            ]
            return sum(vals) / len(vals)

        stats = {
            cat: {
                "token_f1": 100 * _mean(rows, "token_f1"),
                "bleu1": 100 * _mean(rows, "bleu1"),
                "judge": sum(float(s.get("score", 0.0)) for s in rows) / len(rows),
                "n": len(rows),
            }
            for cat, rows in per_cat.items()
        }

        # Unweighted = the papers' "Avg.", over the four KNOWN categories only
        # (an "other" bucket would mean unmapped category ids — never let it
        # into the number we compare against published results).
        known = [c for c in LOCOMO_CATEGORY.values() if c in stats]
        total = sum(s["n"] for s in stats.values())
        out: Dict = {
            "per_category": stats,
            "n_questions": total,
            "note": (
                "Reported alongside the LLM judge, never instead of it. "
                "bleu1 is unigram precision without brevity penalty (matches "
                "the LoCoMo-derived scripts the papers build on). Papers' "
                "'Avg.' == the *_unweighted means."
            ),
        }
        for key in ("token_f1", "bleu1"):
            if known:
                out[f"{key}_mean_unweighted"] = sum(stats[c][key] for c in known) / len(known)
            if total:
                out[f"{key}_mean_weighted"] = (
                    sum(s[key] * s["n"] for s in stats.values()) / total
                )
        return {"locomo_lexical": out}

    # ------------------------------------------------------------------
    # Judge — LoCoMo paper-aligned prompt (binary 0/1)
    # ------------------------------------------------------------------

    def _make_judge(self):
        from common.metric import Judge
        return Judge(
            model=self.judge_model,
            prompt_template=LOCOMO_JUDGE_PROMPT,
            score_min=0, score_max=self.judge_score_max,
        )

