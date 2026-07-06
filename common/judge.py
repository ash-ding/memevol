"""LLM-as-judge: single-shot scoring with prompt template + retry.

Parallel to `common.llm.Agent`:
- `Agent` is multi-turn conversational (system + user + assistant history),
  used inside the QA loop.
- `Judge` is single-shot — given (query, predicted, reference), returns
  (score, reason). Scoring scale and prompt template are configurable
  per benchmark.

Requests share `common.llm`'s per-event-loop client, global concurrency
gate, and unified retry kernel (429/5xx/timeout/empty-response are all
retried with jittered backoff — see common.llm._call_with_retries).

Token usage is reported to `common.tokens.GLOBAL_TOKEN_TRACKER` if it has
been initialized (e.g. by `init_global_tracker()` in the host launch
script). Same lazy-import pattern as `Agent.ask`, no separate "injection"
mechanism is needed.
"""
from __future__ import annotations

import json
import logging
from typing import Optional, Tuple

import httpx

# Module reference (not from-import) so the shared client/kernel stay
# late-bound — monkeypatching common.llm._get_async_client (tests) or any
# future swap of the kernel takes effect here too.
from common import llm as _llm
from common.llm import EmptyResponseError, _supports_reasoning

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
        self.model = model
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
        user_msg = self.prompt_template.format(
            query=query, reference=reference, prediction=predicted
        )

        chat_kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": user_msg}],
            "response_format": {"type": "json_object"},
        }
        if _supports_reasoning(self.model):
            chat_kwargs["reasoning_effort"] = "low"

        client = _llm._get_async_client()
        request_timeout = httpx.Timeout(self.timeout, connect=10.0)

        async def _attempt():
            resp = await client.chat.completions.create(
                **chat_kwargs, timeout=request_timeout
            )
            if not resp.choices:
                raise EmptyResponseError("Judge API returned no choices")

            # Lazy-import to match Agent.ask pattern; tracker may be None.
            from common.tokens import GLOBAL_TOKEN_TRACKER
            if GLOBAL_TOKEN_TRACKER is not None and hasattr(resp, "usage") and resp.usage is not None:
                try:
                    await GLOBAL_TOKEN_TRACKER.update(model_name=self.model, usage=resp.usage)
                except Exception as tracker_exc:
                    log.debug(f"token tracker update failed: {tracker_exc}")

            return resp

        try:
            for parse_attempt in range(2):  # 1 call + 1 retry on malformed JSON
                resp = await _llm._call_with_retries(
                    _attempt, what="Judge", max_retries=self.max_retries
                )
                raw = resp.choices[0].message.content or ""
                try:
                    result = json.loads(raw)
                except json.JSONDecodeError:
                    if parse_attempt == 0:
                        log.warning(f"Judge returned non-JSON, retrying once: {raw[:120]}")
                        continue
                    return self.score_min, f"Judge returned non-JSON: {raw[:200]}"

                score = int(result.get("score", self.score_min))
                score = max(self.score_min, min(self.score_max, score))
                reason = str(result.get("reason", ""))
                return score, reason
        except Exception as exc:
            log.warning(f"Judge failed: {repr(exc)}")
            return self.score_min, f"Judge error: {exc}"

        return self.score_min, "Judge exhausted retries"
