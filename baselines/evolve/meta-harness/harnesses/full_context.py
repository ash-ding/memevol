"""Baseline harness: full context.

The calibration ceiling on the cost axis — every visible unit is rendered to
text at BUILD time and the most recent `MAX_CHARS` worth is handed back
verbatim at RETRIEVE time. No selection, no compression, no LLM calls: it
answers "how far does stuffing the window get you, and at what token cost?".

Together with `no_memory` this brackets the accuracy/context-cost frontier the
search loop optimizes over.
"""

from typing import Dict, List

from common.memo_class import MemoClass

MAX_CHARS = 30_000


def _render_app_log(log: dict) -> str:
    return (
        f"[{log.get('app_log_id', '?')}] {log.get('timestamp', '')} "
        f"{log.get('app_name', '')}/{log.get('api_name', '')}\n"
        f"request: {log.get('request', '')}\nresponse: {log.get('response', '')}"
    )


def _render_conversation(conv: dict) -> List[str]:
    out = []
    keys = [k for k in conv if k.startswith("session_") and isinstance(conv[k], list)]
    for key in sorted(keys, key=lambda k: int(k.split("_")[1])):
        date = conv.get(f"{key}_date_time", "")
        for turn in conv[key]:
            out.append(
                f"[{turn.get('dia_id', '?')}] {date} "
                f"{turn.get('speaker', '')}: {turn.get('text', '')}"
            )
    return out


def _render_session(session: dict) -> str:
    body = "\n".join(
        f"{m.get('role', '')}: {m.get('content', '')}" for m in session.get("messages", [])
    )
    return f"[{session.get('session_id', '?')}] {session.get('date', '')}\n{body}"


class FullContextHarness(MemoClass):

    def __init__(self, config=None):
        super().__init__(config)
        self._blocks: List[str] = []

    async def build_memory_from_data(self, recorder) -> None:
        init = recorder.init
        if "app_logs" in init:
            self._blocks += [_render_app_log(log) for log in init["app_logs"]]
        elif "conversation" in init:
            self._blocks += _render_conversation(init["conversation"])
        elif "sessions" in init:
            self._blocks += [_render_session(s) for s in init["sessions"]]

    async def retrieve_memory_for_query(self, recorder) -> Dict:
        # Newest-first budget fill, then restore chronological order.
        kept, used = [], 0
        for block in reversed(self._blocks):
            if used + len(block) > MAX_CHARS:
                break
            kept.append(block)
            used += len(block)
        return {"inline_memory_blocks": list(reversed(kept))}
