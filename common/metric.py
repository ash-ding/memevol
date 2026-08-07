"""Evaluation metrics — every way this repo turns a prediction into a number.

(Renamed from common/judge.py, 2026-08-06: LLM-as-judge is ONE metric method,
not the whole story.)

Currently implemented:
- `Judge` — LLM-as-judge: single-shot scoring with a per-benchmark prompt
  template + score range. Used by LoCoMo (binary, one prompt) and LongMemEval
  (binary, four per-question_type prompts); DynamicMem's official TCE holistic
  scoring lives in benchmarks/dynamicmem/tce_prompts.py instead (structured
  per-field verdicts — it shares only the common.llm transport).

Planned (stubs below, tracked in the "lexical metrics" GitHub issue):
- `token_f1` / `bleu1` — the lexical metrics the LoCoMo paper actually
  reports. A working offline implementation already exists in
  baselines/harness/score_paper_metrics.py (used for the mem0/memoryos
  paper-reproduction checks); consolidating it here would let LoCoMo emit
  paper-comparable numbers inline instead of via a separate offline pass.

`Judge` requests share `common.llm`'s per-event-loop client, global
concurrency gate, and unified retry kernel (429/5xx/timeout/empty-response
are all retried with jittered backoff — see common.llm._call_with_retries).
Token usage is reported to `common.tokens.GLOBAL_TOKEN_TRACKER` if it has
been initialized (e.g. by `init_global_tracker()` in the host launch
script). Same lazy-import pattern as `Agent.ask`, no separate "injection"
mechanism is needed.
"""
from __future__ import annotations

import json
import logging
from typing import Optional, Tuple

# Module reference (not from-import) so the shared client/kernel stay
# late-bound — monkeypatching common.llm internals (tests) or any future
# swap of the kernel takes effect here too.
from common import llm as _llm
from common.llm import _split_model_effort

log = logging.getLogger("main")


class Judge:
    """LLM-as-judge wrapper. Single-shot, prompt-templated, score+reason output."""

    DEFAULT_PROMPT_TEMPLATE = """You are an expert evaluator scoring an AI-generated answer against a reference answer.

Question: {query}
Reference: {reference}
Prediction: {prediction}

Score from 0 to 10 based on how well the prediction matches the reference's key information. Be lenient with phrasing differences and extra non-contradicting detail; deduct for missing key facts, factual errors, or contradictions.

Output ONLY a JSON object:
{{"reason": "<one sentence explanation>", "score": <integer 0-10>}}
"""

    def __init__(
        self,
        model: str = "gpt-5-mini",
        prompt_template: Optional[str] = None,
        score_min: int = 0,
        score_max: int = 10,
        timeout: float = 300.0,
        max_retries: int = 5,
    ):
        # Honour the repo-wide "model/effort" suffix convention (same parsing
        # as Agent) — "gpt-5-mini/low" must NOT be sent verbatim to the API
        # (deterministic 404 → every judge call would silently score 0).
        self.model, effort = _split_model_effort(model)
        # Explicit suffix wins; otherwise judges default to low effort
        # (scoring is a short structured task).
        self.reasoning_effort = effort or "low"
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT_TEMPLATE
        self.score_min = score_min
        self.score_max = score_max
        self.timeout = timeout
        self.max_retries = max_retries

    async def score(
        self, query: str, predicted: str, reference: str
    ) -> Tuple[int, str]:
        """Score (predicted vs reference) for a given query. Returns (score, reason).

        Never raises: on retry exhaustion or any non-retryable exception,
        returns (score_min, error_message). A malformed-JSON judge reply is
        retried once before giving up (a whole QA step's signal is worth one
        extra call).
        """
        try:
            user_msg = self.prompt_template.format(
                query=query, reference=reference, prediction=predicted
            )
        except (KeyError, IndexError, ValueError) as exc:
            # A template with unescaped braces must degrade like every other
            # judge failure (this method's contract is "never raises").
            log.warning(f"Judge prompt template failed to render: {exc!r}")
            return self.score_min, f"Judge error: bad prompt template ({exc!r})"

        try:
            for parse_attempt in range(2):  # 1 call + 1 retry on malformed JSON
                # Provider-agnostic call (OpenAI / Anthropic API / Anthropic
                # Vertex, routed by model name inside the kernel). json_mode
                # maps to OpenAI's response_format; Claude models rely on the
                # prompt's JSON demand + this parse-retry loop. Per-provider
                # effort gating also lives in the kernel.
                raw, usage = await _llm._chat_completion(
                    model=self.model,
                    messages=[{"role": "user", "content": user_msg}],
                    timeout=self.timeout,
                    max_retries=self.max_retries,
                    json_mode=True,
                    effort=self.reasoning_effort,
                    what="Judge",
                )
                # Lazy-import to match Agent.ask pattern; tracker may be None.
                from common.tokens import GLOBAL_TOKEN_TRACKER
                if GLOBAL_TOKEN_TRACKER is not None and usage:
                    try:
                        await GLOBAL_TOKEN_TRACKER.update(model_name=self.model, usage=usage)
                    except Exception as tracker_exc:
                        log.debug(f"token tracker update failed: {tracker_exc}")
                raw = raw or ""
                try:
                    result = json.loads(raw)
                except json.JSONDecodeError:
                    if parse_attempt == 0:
                        log.warning(f"Judge returned non-JSON, retrying once: {raw[:120]}")
                        continue
                    return self.score_min, f"Judge returned non-JSON: {raw[:200]}"

                try:
                    # int(float(...)) so a "8.5"/"8" string or 8.5 float from
                    # the judge coerces instead of falling into the broad
                    # except below (which would skip the parse retry).
                    score = int(float(result.get("score", self.score_min)))
                except (TypeError, ValueError, AttributeError):
                    if parse_attempt == 0:
                        log.warning(f"Judge returned malformed score, retrying once: {raw[:120]}")
                        continue
                    return self.score_min, f"Judge returned malformed score: {raw[:200]}"
                score = max(self.score_min, min(self.score_max, score))
                reason = str(result.get("reason", ""))
                return score, reason
        except Exception as exc:
            log.warning(f"Judge failed: {repr(exc)}")
            return self.score_min, f"Judge error: {exc}"

        return self.score_min, "Judge exhausted retries"


# ---------------------------------------------------------------------------
# Lexical metrics — NOT YET INTEGRATED (stubs).
#
# The LoCoMo paper's headline metrics are token-F1 / BLEU-1, not an LLM judge;
# this repo currently only computes them OFFLINE via
# baselines/harness/score_paper_metrics.py. These stubs reserve the canonical
# home; see the "Integrate lexical metrics (token-F1 / BLEU-1) into
# common/metric.py" GitHub issue for the consolidation plan.
# ---------------------------------------------------------------------------

def token_f1(prediction: str, reference: str) -> float:
    """Token-level F1 between prediction and reference (LoCoMo paper metric).

    NOT IMPLEMENTED YET — use baselines/harness/score_paper_metrics.py for
    offline scoring until the consolidation issue lands."""
    raise NotImplementedError(
        "token_f1 is a stub — see the lexical-metrics integration issue; "
        "an offline implementation lives in baselines/harness/score_paper_metrics.py"
    )


def bleu1(prediction: str, reference: str) -> float:
    """BLEU-1 between prediction and reference (LoCoMo paper metric).

    NOT IMPLEMENTED YET — use baselines/harness/score_paper_metrics.py for
    offline scoring until the consolidation issue lands."""
    raise NotImplementedError(
        "bleu1 is a stub — see the lexical-metrics integration issue; "
        "an offline implementation lives in baselines/harness/score_paper_metrics.py"
    )
