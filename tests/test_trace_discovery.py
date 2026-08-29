"""Client-discovery unit tests (Phase B), rooted at ``self.memory``.

Covers: nested + multiple clients, cycle safety, ``__dict__``-less objects via
``gc.get_referents``, never-crash on a raising ``@property``, the hard
NO-``@property``-side-effect rule, and bound degradation (no hang).
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _trace_convention_fakes import FakeChromaCollection

from tracing import discovery
from tracing.discovery import MAX_DEPTH, MAX_VISITED, collect_subtree_ids, discover


def _client():
    return FakeChromaCollection(["x"], ["doc"], [{}])


# --- nested + multiple clients ---------------------------------------------


def test_discovery_finds_nested_client():
    class Holder:
        def __init__(self, c):
            self.client = c
            self.label = "h"

    dr = discover({"vectors": Holder(_client())})
    assert [c.cap.kind for c in dr.clients] == ["chroma"]
    assert dr.clients[0].path == "memory.vectors.client"


def test_discovery_finds_multiple_clients_under_one_memory():
    dr = discover({"a": _client(), "b": _client(),
                   "notes": [{"id": "n0", "content": "hi"}]})
    assert sorted(c.cap.kind for c in dr.clients) == ["chroma", "chroma"]
    assert any(i.item_id == "n0" for i in dr.items)  # in-heap part still captured


# --- cycle safety -----------------------------------------------------------


def test_discovery_cycle_does_not_hang():
    d = {}
    d["self"] = d                       # back-reference
    d["note"] = {"id": "n0", "content": "hi"}
    dr = discover(d)                    # must terminate
    assert any(i.item_id == "n0" for i in dr.items)


# --- __dict__-less object via gc.get_referents ------------------------------


def test_discovery_reaches_client_in_dictless_object():
    class Slotted:  # no __dict__ -> only gc.get_referents can see the slot
        __slots__ = ("client",)

        def __init__(self, c):
            self.client = c

    assert not hasattr(Slotted(_client()), "__dict__")
    dr = discover({"wrap": Slotted(_client())})
    assert [c.cap.kind for c in dr.clients] == ["chroma"]


# --- never-crash on a raising @property -------------------------------------


def test_discovery_never_crashes_on_raising_property():
    class Bomb:
        @property
        def scroll(self):  # a matcher probes 'scroll'; this blows up
            raise RuntimeError("boom")

        def __init__(self):
            self.note = {"id": "n0", "content": "safe"}

    dr = discover({"x": Bomb()})       # must not raise
    assert any(i.item_id == "n0" for i in dr.items)


# --- HARD RULE: discovery must NOT trigger a @property side effect -----------


def test_discovery_does_not_trigger_property_side_effect():
    class Lazy:
        opened = False

        def __init__(self):
            self.note = {"id": "n0", "content": "safe"}

        @property
        def lazy_value(self):          # opening a connection as a side effect
            type(self).opened = True
            return 42

    discover({"x": Lazy()})
    assert Lazy.opened is False         # __dict__ read directly; property untouched


# --- bounds: degrade, never hang -------------------------------------------


def test_discovery_depth_bound_degrades_gracefully():
    # Nest deeper than MAX_DEPTH; capture must stop and note the bound, not hang.
    node = {"id": "deep", "content": "leaf"}
    for _ in range(MAX_DEPTH + 5):
        node = {"child": node}
    dr = discover(node)
    assert any("MAX_DEPTH" in n for n in dr.notes)


def test_discovery_visited_bound_degrades_gracefully():
    big = {f"k{i}": {"id": f"r{i}", "sub": {"deep": i}} for i in range(MAX_VISITED + 50)}
    dr = discover(big)                 # must terminate
    assert dr.truncated
    assert any("MAX_VISITED" in n for n in dr.notes)


# --- collect_subtree_ids (used to exclude self.memory from state/) -----------


def test_collect_subtree_ids_covers_children():
    child = {"id": "n0"}
    root = {"a": child, "b": [child]}
    ids = collect_subtree_ids(root)
    assert id(root) in ids and id(child) in ids


def test_should_skip_and_helpers_are_defensive():
    # Sanity: skip predicate handles machinery + scalars without raising.
    assert discovery._should_skip(3) is True
    assert discovery._should_skip(lambda: None) is True
    assert discovery._should_skip("text") is False
