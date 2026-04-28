"""Prompt builders for the DynamicMem benchmark — both QA agent and judge."""

from typing import Dict
import json

DYNAMICMEM_BASIC_PROMPT = """\
You are a personal assistant with access to a user's app activity logs.
You will be asked questions about the user's habits, preferences, and life patterns.

### Task
Answer the question based on the user's activity data and any memory context provided.

### Guidelines
- Be specific and factual. Include concrete details (times, days, locations, frequencies) when available.
- If you are not certain, provide your best estimate based on observable patterns.
- Keep your answer concise (1–3 sentences).
- Output only your answer — no preamble, no explanation, no "Based on the logs..." prefix.
"""


# --- LLM-as-judge prompt ---------------------------------------------------
# DynamicMem answers are multi-point factual summaries about user activity
# patterns. Partial credit (0-10) makes sense because predictions can hit some
# key points and miss others. Lifted verbatim from the pre-v8 BASIC_JUDGE_PROMPT.

DYNAMICMEM_JUDGE_PROMPT = """You are an expert evaluator. Your task is to rate the quality of an AI-generated answer based on a standard reference.

[Question]: {query}
[Standard Reference]: {reference}
[AI Prediction]: {prediction}

Rate the prediction on a scale of 0 to 10 based on whether it correctly covers the KEY POINTS in the reference.

**Important Evaluation Principle:**
- Focus on whether the prediction captures the ESSENTIAL information from the reference.
- Additional details, elaborations, or supplementary information in the prediction should NOT be penalized, as long as they don't contradict the reference.
- Only deduct points for: missing key points, factual errors, or contradictions with the reference.

Scoring criteria:
- 0-2: Mostly or completely incorrect. Contains major factual errors, contradicts the reference, or misses almost all key points.
- 3-5: Partially correct but missing significant key points from the reference, or contains notable factual errors.
- 6-8: Covers most key points from the reference correctly. Minor omissions or inaccuracies may exist, but core information is accurate.
- 9-10: Fully covers all key points from the reference accurately. No factual errors or contradictions. (Extra details beyond the reference are acceptable.)

Output ONLY a JSON object with two keys:
- "reason": a brief explanation for the score (focus on which key points were hit or missed)
- "score": a number from 0 to 10 (integer)

Output format(JSON):
{{"reason": "...", "score": ...}}
"""


def get_dynamicmem_prompt(
    query: str,
    memory_retrived: Dict = {},
    **kargs,
):
    """Build the two-message prompt for the DynamicMem agent.

    Returns:
        [{'role': 'system', 'content': ...}, {'role': 'user', 'content': ...}]
    """
    memory_block = ""
    if memory_retrived:
        memory_block = (
            "\n### Retrieved Memory\n"
            "Here are relevant facts and patterns retrieved from memory "
            "that may help you answer:\n"
            + json.dumps(memory_retrived, indent=2, ensure_ascii=False)
            + "\n"
        )

    system_prompt = DYNAMICMEM_BASIC_PROMPT + memory_block

    user_message = f"Question: {query}"

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
