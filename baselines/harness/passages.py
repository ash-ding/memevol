"""The shared text rendering of one DynamicMem app-log entry.

Six of the seven baselines ingest DynamicMem logs, and they must all see the
SAME text: a score difference has to come from how a system remembers, not
from one of them getting a tidier rendering of the input. So this lives in one
place and every baseline imports it.

It used to live in hipporag2's memo.py (the first baseline written), which
meant importing hipporag2 — and its multi-GB dependencies — to reach eight
lines of string formatting. That is also what broke packaging a baseline for
forge's container, where hipporag2 is not installed.

The text is verbatim from the original `eval_hipporag2.py` (lines 99-113); any
edit here changes every baseline's input and invalidates comparisons with
previously recorded numbers.
"""
from __future__ import annotations

import json


def app_log_to_passage(log_entry: dict) -> str:
    ts = log_entry.get("timestamp", ""); app = log_entry.get("app_name", "")
    api = log_entry.get("api_name", "")
    req = json.dumps(log_entry.get("request", {}), ensure_ascii=False)
    resp = json.dumps(log_entry.get("response", {}), ensure_ascii=False)
    domain = log_entry.get("metadata", {}).get("domain", "")
    return (f"[{ts}] App: {app}, Action: {api}\nDomain: {domain}\n"
            f"Request: {req}\nResponse: {resp}")
