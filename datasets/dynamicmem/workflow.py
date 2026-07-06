"""DynamicMem-specific workflow.

Concrete implementation of `common.workflow.BaseWorkflow` for the DynamicMem
benchmark. All behavior matches the prior `baselines.alma.workflow.DynamicMem_Workflow`
bit-for-bit; only code location and subclass decomposition have changed.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Type

from common.workflow import BaseWorkflow
from datasets.dynamicmem.env import (
    Basic_Recorder,
    DynamicMemRecorder,
    load_user_data,
)
from datasets.dynamicmem.prompts import DYNAMICMEM_JUDGE_PROMPT, get_dynamicmem_prompt


class DynamicMemWorkflow(BaseWorkflow):
    """DynamicMem two-phase per-user workflow.

    Phase 1 builds a personal profile from the user's app logs.
    Phase 2 answers QA questions using the profile.
    """

    recorder_class: Type[Basic_Recorder] = DynamicMemRecorder
    _phase1_item_label: str = "logs"
    _default_qa_per_user_hint: int = 178

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    async def load_user_data(
        self, user_dir: str, eval_n_qa: Optional[int]
    ) -> Tuple[List[Dict], List[Dict]]:
        app_logs, _user_profile, qa_pairs = load_user_data(user_dir, eval_n_qa)
        return app_logs, qa_pairs

    # ------------------------------------------------------------------
    # Phase 1: log_init
    # ------------------------------------------------------------------

    async def phase1_log_init(self, recorder: Basic_Recorder, chunk: List[Dict]) -> None:
        """DynamicMemRecorder.log_init(app_logs)."""
        await recorder.log_init(chunk)

    # ------------------------------------------------------------------
    # Phase 2: retrieve recorder + QA prompt + metadata + step logging
    # ------------------------------------------------------------------

    def build_query_recorder_init(self, init_data: List[Dict], qa: Dict) -> Dict:
        return {"app_logs": init_data, "query": qa["query"]}

    def build_qa_prompt(
        self, query: str, retrieved: Dict, qa_metadata: Dict, reference: str = ""
    ) -> List[Dict]:
        # DynamicMem doesn't use qa_metadata or reference in the prompt.
        return get_dynamicmem_prompt(query, retrieved)

    def extract_relevant_context(self, qa: Dict, init_data: List[Dict]) -> List[Dict]:
        app_log_ids = qa.get("metadata", {}).get("app_log_ids", [])
        lookup = {log["app_log_id"]: log for log in init_data}
        return [lookup[lid] for lid in app_log_ids if lid in lookup]

    def build_qa_metadata(self, qa: Dict) -> Dict:
        return {
            "domain": qa.get("metadata", {}).get("domain", ""),
            "belonged": qa.get("metadata", {}).get("belonged", ""),
            "app_log_ids": qa.get("metadata", {}).get("app_log_ids", []),
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
            relevant_app_logs=relevant_context,
        )

    # ------------------------------------------------------------------
    # Judge — DynamicMem-specific prompt (0-10 partial credit)
    # ------------------------------------------------------------------

    def _make_judge(self):
        from common.judge import Judge
        return Judge(
            model=self.judge_model,
            prompt_template=DYNAMICMEM_JUDGE_PROMPT,
            score_min=0, score_max=self.judge_score_max,
        )

