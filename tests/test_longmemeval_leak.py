"""LongMemEval: the memory system must not see upstream's scoring labels.

Two fields in longmemeval_s_cleaned.json mark the answer location, and both
used to reach MemoClass verbatim (Phase 1 ingest AND Phase 2 retrieve, which
is handed the full session list again):

  * `session_id` — every gold session is named "answer_<hash>", every
    distractor "sharegpt_*" / "ultrachat_*". 948/948 vs 0/22919 across the
    file: a perfect classifier. Upstream's own reader never sees it either —
    run_generation.py labels sessions positionally ("### Session 45:").

  * `has_answer` — set on evidence turns, but the key is present ONLY inside
    gold sessions (10960 messages, 0 outside), so its mere PRESENCE is a
    perfect classifier regardless of its value. Upstream's README calls it a
    label for "turn-level memory recall accuracy evaluation".

Either channel alone shrinks the haystack from ~47.7 to ~1.9 sessions per
question, which silently turns the -S variant into upstream's much easier
`oracle` variant — so both must stay scrubbed, and the held-out split does
not catch a regression here (the same labels are present there).

Tests needing the 277 MB data file skip when it is absent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.longmemeval.env import DATA_PATH, _build_sessions, load_user_data
from benchmarks.longmemeval.workflow import LongMemEvalWorkflow

needs_data = pytest.mark.skipif(
    not DATA_PATH.is_file(), reason=f"LongMemEval data not fetched ({DATA_PATH})"
)

_LEAKY_PREFIXES = ("answer_", "sharegpt_", "ultrachat_")

_SAMPLE = {
    "question_id": "deadbeef",
    "question": "How many weeks ago did I get the chandelier?",
    "answer": 4,
    "question_type": "temporal-reasoning",
    "question_date": "2023/04/01 (Sat) 08:09",
    "haystack_session_ids": ["ultrachat_84486", "answer_0b4a8adc_1", "sharegpt_1"],
    "haystack_dates": ["2023/02/02 (Thu) 04:45",
                       "2023/03/04 (Sat) 22:43",
                       "2023/03/09 (Thu) 09:00"],
    "haystack_sessions": [
        [{"role": "user", "content": "noise"}],
        [{"role": "user", "content": "gold", "has_answer": True},
         {"role": "assistant", "content": "reply", "has_answer": False}],
        [{"role": "user", "content": "more noise"}],
    ],
    "answer_session_ids": ["answer_0b4a8adc_1"],
}


def test_session_ids_are_positional_not_upstream():
    sessions = _build_sessions(_SAMPLE)
    assert [s["session_id"] for s in sessions] == [
        "session_001", "session_002", "session_003"
    ]
    for s in sessions:
        assert not any(p in s["session_id"] for p in _LEAKY_PREFIXES)


def test_has_answer_is_stripped_from_every_message():
    for s in _build_sessions(_SAMPLE):
        for m in s["messages"]:
            assert set(m) == {"role", "content"}


def test_scrubbing_preserves_content_and_dates():
    """The scrub must remove labels only — never alter what the memory
    system is supposed to learn from."""
    sessions = _build_sessions(_SAMPLE)
    assert [s["date"] for s in sessions] == _SAMPLE["haystack_dates"]
    assert sessions[1]["messages"] == [
        {"role": "user", "content": "gold"},
        {"role": "assistant", "content": "reply"},
    ]
    assert [len(s["messages"]) for s in sessions] == [1, 2, 1]


def test_gold_lookup_survives_the_rename():
    """answer_session_ids is translated onto the normalized ids, so the
    recorder-side relevant-context lookup still resolves. This metadata is
    recorder-side only — the workflow never puts it in recorder.init."""
    sessions = _build_sessions(_SAMPLE)
    qa = {"metadata": {"answer_session_ids": ["session_002"]}}
    hits = LongMemEvalWorkflow.extract_relevant_context(None, qa, sessions)
    assert [h["session_id"] for h in hits] == ["session_002"]
    assert hits[0]["messages"][0]["content"] == "gold"


def test_retrieve_phase_init_carries_the_scrubbed_sessions():
    """Phase 2 hands the FULL session list to retrieve_memory_for_query, so a
    leak there would be exploitable even if Phase 1 dropped the fields."""
    sessions = _build_sessions(_SAMPLE)
    qa = {"query": "q", "metadata": {"question_date": "2023/04/01 (Sat) 08:09"}}
    init = LongMemEvalWorkflow.build_query_recorder_init(None, sessions, qa)
    for s in init["sessions"]:
        assert not any(p in s["session_id"] for p in _LEAKY_PREFIXES)
        for m in s["messages"]:
            assert "has_answer" not in m


@needs_data
def test_no_leak_survives_anywhere_in_the_dataset():
    from benchmarks.longmemeval.env import _load_samples

    n_sessions = n_messages = 0
    for sample in _load_samples():
        for s in _build_sessions(sample):
            n_sessions += 1
            assert not any(p in s["session_id"] for p in _LEAKY_PREFIXES), s["session_id"]
            for m in s["messages"]:
                n_messages += 1
                assert "has_answer" not in m
    # sanity: we really did walk the whole haystack, not an empty list
    assert n_sessions > 20000, n_sessions
    assert n_messages > 200000, n_messages


@needs_data
def test_gold_sessions_still_resolve_across_the_dataset():
    from benchmarks.longmemeval.env import _load_samples

    checked = 0
    for sample in _load_samples()[:50]:
        if not sample["answer_session_ids"]:
            continue
        sessions, _, qa = load_user_data(sample["question_id"])
        expected = [
            f"session_{i + 1:03d}"
            for i, sid in enumerate(sample["haystack_session_ids"])
            if sid in set(sample["answer_session_ids"])
        ]
        assert qa[0]["metadata"]["answer_session_ids"] == expected
        hits = LongMemEvalWorkflow.extract_relevant_context(None, qa[0], sessions)
        assert [h["session_id"] for h in hits] == expected
        checked += 1
    assert checked >= 40, checked
