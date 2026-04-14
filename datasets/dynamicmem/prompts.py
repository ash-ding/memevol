"""Prompt builder for the DynamicMem benchmark."""

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
