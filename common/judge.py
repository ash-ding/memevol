"""LLM-as-judge: single-shot scoring with prompt template + retry.

Parallel to `common.llm.Agent`:
- `Agent` is multi-turn conversational (system + user + assistant history),
  used inside the QA loop.
- `Judge` is single-shot — given (query, predicted, reference), returns
  (score, reason). Scoring scale and prompt template are configurable
  per benchmark.

Token usage is reported to `common.tokens.GLOBAL_TOKEN_TRACKER` if it has
been initialized (e.g. by `init_global_tracker()` in the host launch
script). Same lazy-import pattern as `Agent.ask`, no separate "injection"
mechanism is needed.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Optional, Tuple

import httpx
import openai
from openai import AsyncOpenAI

log = logging.getLogger("main")


# OpenAI rejects `reasoning_effort` for non-reasoning models. Keep a narrow
# allowlist by name prefix; add new model families as they appear.
_REASONING_MODEL_PREFIXES = ("gpt-5", "o1", "o3", "o4")


def _supports_reasoning(model: str) -> bool:
    return any(model.startswith(p) for p in _REASONING_MODEL_PREFIXES)


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
        timeout: float = 150.0,
        max_retries: int = 6,
    ):
        self.model = model
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT_TEMPLATE
        self.score_min = score_min
        self.score_max = score_max
        self.max_retries = max_retries
        self.client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            timeout=httpx.Timeout(timeout, connect=10.0),
            max_retries=0,
        )

    async def score(
        self, query: str, predicted: str, reference: str
    ) -> Tuple[int, str]:
        """Score (predicted vs reference) for a given query. Returns (score, reason).

        On retry exhaustion or any non-retryable exception, returns (score_min, error_message).
        Token usage is reported to GLOBAL_TOKEN_TRACKER if it has been
        initialized; otherwise silently skipped (matches Agent.ask).
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

        for attempt in range(self.max_retries):
            try:
                resp = await self.client.chat.completions.create(**chat_kwargs)

                # Lazy-import to match Agent.ask pattern; tracker may be None.
                from common.tokens import GLOBAL_TOKEN_TRACKER
                if GLOBAL_TOKEN_TRACKER is not None and hasattr(resp, "usage") and resp.usage is not None:
                    try:
                        await GLOBAL_TOKEN_TRACKER.update(model_name=self.model, usage=resp.usage)
                    except Exception as tracker_exc:
                        log.debug(f"token tracker update failed: {tracker_exc}")

                raw = resp.choices[0].message.content or ""
                try:
                    result = json.loads(raw)
                except json.JSONDecodeError:
                    return self.score_min, f"Judge returned non-JSON: {raw[:200]}"

                score = int(result.get("score", self.score_min))
                score = max(self.score_min, min(self.score_max, score))
                reason = str(result.get("reason", ""))
                return score, reason

            except (asyncio.TimeoutError, openai.APITimeoutError, openai.APIConnectionError, openai.InternalServerError) as exc:
                if attempt < self.max_retries - 1:
                    delay = min(60, 2 ** (attempt + 1))  # 2, 4, 8, 16, 32, (cap) 60
                    log.warning(f"Judge retry {attempt+1}/{self.max_retries}: {repr(exc)}")
                    await asyncio.sleep(delay)
                    continue
                log.warning(f"Judge failed after {self.max_retries} attempts: {repr(exc)}")
                return self.score_min, f"Judge error: {exc}"
            except Exception as exc:
                log.warning(f"Judge failed: {repr(exc)}")
                return self.score_min, f"Judge error: {exc}"

        return self.score_min, "Judge exhausted retries"
