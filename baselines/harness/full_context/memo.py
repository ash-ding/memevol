"""full_context baseline — the calibration ceiling on the cost axis.

BUILD renders every visible unit to text and keeps it. RETRIEVE hands back as
much of it as the token budget allows, newest-first, restored to chronological
order. No selection, no compression, no model calls of its own: it answers
"how far does stuffing the window get you, and at what token cost?".

Together with `no_memory` (which stores nothing) this brackets the
accuracy/context-cost frontier every other harness is judged inside.

**It is only genuinely "full context" where the data fits.** Measured on this
repo's data, rendered per user:

    LoCoMo         30.0K tokens median,   32.5K max  -> WHOLE, never truncated
    LongMemEval   103K   tokens median,  105K   max  -> WHOLE, never truncated
    DynamicMem      1.04M tokens median,   1.21M max -> truncated, necessarily

DynamicMem's stream fits in no model's window, so its truncation is physics,
not a choice. Read a DynamicMem number from this baseline as "full text at
`max_tokens`", and a LoCoMo / LongMemEval number as full text outright.

The budget is counted in TOKENS, not characters, because characters are not
comparable across these datasets: the measured chars-per-token ratios are 3.5
(LoCoMo), 4.8 (LongMemEval) and 3.6 (DynamicMem), so one character cap would
hand each dataset a budget differing by up to 37% — and `memory_tokens_per_query`,
the cost axis this baseline anchors, is denominated in tokens.

This baseline needs no dependencies of its own — no vendored source, no
models, no API calls — so the repo-root venv runs it (there is no `--project`
to pass, unlike the vendored baselines).

    uv run python -m baselines.harness.eval_harness \\
        --config baselines/harness/config.example.yaml     # harness: full_context
"""
from __future__ import annotations

import logging
from typing import Dict, List

from baselines.harness.passages import app_log_to_passage
from common.memo_class import MemoClass
from common.tokens import count_text_tokens

log = logging.getLogger(__name__)

# Plain assignments, like every other baseline: tests/test_model_config.py
# reads these out of the source with ast.literal_eval, and an annotated
# assignment reads as "not declared".
CONFIG_DEFAULTS = {
    # Token budget for the context handed to the QA agent. 128K keeps LoCoMo
    # and LongMemEval WHOLE (see the table above) — those are true full
    # context. DynamicMem is truncated newest-first because 1.04M tokens fits
    # nowhere.
    #
    # THIS MUST LEAVE ROOM IN THE QA MODEL'S WINDOW. The memo is not told
    # which model answers (`llm_model` is frame config, handed to the
    # workflow, not here), so the budget cannot size itself — check it against
    # the model you are running:
    #
    #     gpt-5-mini        400,000   <- the shipped default; 128K is roomy
    #     gpt-4.1-mini    1,047,576   <- roomy
    #     gpt-4o / -mini    128,000   <- 128K IS the whole window: LOWER THIS,
    #                                    or every query 400s (see below)
    #
    # An over-budget request is not truncated by the API — OpenAI returns 400
    # `context_length_exceeded`, which `common.llm` does not retry (4xx other
    # than 429 fast-fail) and `common/workflow.py` records as a score=0 step
    # marked `[Phase2_Answer_ERROR]`. Those zeros ARE averaged into the score
    # (`common/evaluate.py::_build_score_json`), so an over-budget run reads as
    # "answered badly", not "did not run". Only the trace says otherwise.
    #
    # None = no cap at all. That is not a purer full context: on DynamicMem it
    # overflows for SOME users and not others (455K min / 1.04M median / 1.21M
    # max), producing exactly the silent partial failure described above. It
    # exists for the "LoCoMo and LongMemEval only, cost be damned" reading.
    "max_tokens": 128_000,
}

# `arm: unified` has nothing to switch here — this baseline calls no model.
UNIFIED_MODEL_KEYS = {}

#: Encoding the budget is priced in. `common.tokens` is also what the
#: framework counts `memory_tokens` with, so budget and cost axis agree on a
#: tokenizer; the framework prices against the QA model, so the two can differ
#: by a few percent when that model uses another encoding. The budget is a
#: policy, not an exact window fit, so that slack is fine.
_BUDGET_MODEL = "gpt-4o"


def _render_conversation(conv: Dict) -> List[str]:
    """LoCoMo: one block per turn, carrying its dia_id and its session date."""
    out: List[str] = []
    keys = [k for k in conv if k.startswith("session_") and isinstance(conv[k], list)]
    for key in sorted(keys, key=lambda k: int(k.split("_")[1])):
        date = conv.get(f"{key}_date_time", "")
        for turn in conv[key]:
            out.append(
                f"[{turn.get('dia_id', '?')}] {date} "
                f"{turn.get('speaker', '')}: {turn.get('text', '')}"
            )
    return out


def _render_session(session: Dict) -> str:
    """LongMemEval: one block per session, headed by its id and date."""
    body = "\n".join(
        f"{m.get('role', '')}: {m.get('content', '')}"
        for m in session.get("messages", [])
    )
    return f"[{session.get('session_id', '?')}] {session.get('date', '')}\n{body}"


class FullContextMemo(MemoClass):

    def __init__(self, config=None):
        super().__init__(config)
        #: Rendered blocks in arrival order, and each one's token count. The
        #: counts are computed once at BUILD: re-tokenizing the corpus on every
        #: query would cost more than the eval (DynamicMem is ~3.7 MB/user).
        self._blocks: List[str] = []
        self._token_counts: List[int] = []
        self._truncation_logged = False

    async def build_memory_from_data(self, recorder) -> None:
        """ACCUMULATE — `recorder.init` holds only what is newly visible, and
        DynamicMem delivers one checkpoint segment per call."""
        init = getattr(recorder, "init", None) or {}
        if "app_logs" in init:
            # The SHARED renderer: every baseline must see the same DynamicMem
            # text, or a score difference measures formatting, not memory.
            new = [app_log_to_passage(entry) for entry in init["app_logs"]]
        elif "conversation" in init:
            new = _render_conversation(init["conversation"])
        elif "sessions" in init:
            new = [_render_session(s) for s in init["sessions"]]
        else:
            new = []

        for block in new:
            self._blocks.append(block)
            self._token_counts.append(count_text_tokens(block, _BUDGET_MODEL))

    async def retrieve_memory_for_query(self, recorder) -> Dict:
        """Read-only. Newest-first to the budget, then chronological again.

        Newest-first because when the corpus does not fit, recency is the
        better default in a memory benchmark — and because dropping the tail
        would make the baseline weaker than the method it is meant to bound.
        """
        budget = self.config["max_tokens"]
        if budget is None:
            return {"inline_memory_blocks": list(self._blocks)}

        kept_reversed: List[str] = []
        used = 0
        for block, cost in zip(reversed(self._blocks), reversed(self._token_counts)):
            if used + cost > budget:
                break
            kept_reversed.append(block)
            used += cost

        if not kept_reversed and self._blocks:
            # One block alone exceeds the budget. Returning nothing would turn
            # this into `no_memory` silently, which is a worse failure than
            # overshooting by one block — so keep the newest and say so.
            kept_reversed = [self._blocks[-1]]
            used = self._token_counts[-1]

        if len(kept_reversed) < len(self._blocks) and not self._truncation_logged:
            total = sum(self._token_counts)
            log.info(
                "full_context: budget %d tokens kept %d/%d blocks (%d/%d tokens, "
                "%.1f%%) — this run is full text AT THE BUDGET, not full text",
                budget, len(kept_reversed), len(self._blocks), used, total,
                100.0 * used / total if total else 0.0,
            )
            self._truncation_logged = True

        return {"inline_memory_blocks": list(reversed(kept_reversed))}
