"""Letting several users of one harness baseline run at the same time.

The workflow evaluates up to `max_sample_concurrent` users concurrently on ONE
asyncio event loop, which only switches between users at an `await`. Six of the
seven vendored memory systems are synchronous — their LLM calls, embedding,
indexing and search all run without ever awaiting — so calling them directly
inside an `async def` hook holds the loop for the whole call and the users run
one after another. Their memo.py hooks therefore hand the synchronous work to a
worker thread with `asyncio.to_thread`, which frees the loop while it runs (and
copies the caller's context, so token accounting keeps its phase).

Once vendored code runs on several threads at once, two things it shares must
be made safe here — without editing anything under `src/`:

  * stdout. The memos silence vendored debug `print()`s by swapping
    `sys.stdout`, which is process-global. Per-call `contextlib.redirect_stdout`
    blocks overlapping on different threads restore each other's streams out of
    order and can leave `sys.stdout` pointing at a closed file. `quiet_stdout`
    is one reference-counted swap shared by all threads.

  * local models. One sentence-transformers / cross-encoder model is shared by
    every user of a process (the embedder factory memoizes it). Hugging Face
    fast tokenizers raise "RuntimeError: Already borrowed" under concurrent
    use, and parallel forward passes on one model only contend for the same
    device anyway. `serialize_calls` puts a per-model lock around the method
    vendored code calls, so local inference runs one call at a time while the
    network-bound work around it overlaps freely.

  * model loading. Loading a Hugging Face model temporarily patches
    process-global torch state (weights are first created on the "meta"
    device), so two threads loading models at once corrupt each other's load
    ("Cannot copy out of meta tensor"). `model_load_lock` makes every load — the
    embedder factory and each memo's one-time system construction — take turns.
"""
from __future__ import annotations

import contextlib
import functools
import os
import sys
import threading
from typing import Any, Iterator

# Held while a memo constructs its memory system and while the embedder factory
# loads a model. Re-entrant: construction calls the factory.
model_load_lock = threading.RLock()

_quiet_lock = threading.Lock()
_quiet_depth = 0
_saved_stdout: Any = None
_sink: Any = None


@contextlib.contextmanager
def quiet_stdout() -> Iterator[None]:
    """Discard `print()` output while any thread is inside this block.

    Reference-counted: the first entry swaps `sys.stdout` for a UTF-8 devnull
    sink, the last exit restores the original. The sink is UTF-8 with
    `errors="replace"` because lightmem's extractor prints raw user text, which a
    platform-default encoding (cp1252 on Windows) cannot always encode. Output
    from other code on the same process during that window is silenced too; the
    file log (run.log) is unaffected.
    """
    global _quiet_depth, _saved_stdout, _sink
    with _quiet_lock:
        if _quiet_depth == 0:
            _sink = open(os.devnull, "w", encoding="utf-8", errors="replace")
            _saved_stdout = sys.stdout
            sys.stdout = _sink
        _quiet_depth += 1
    try:
        yield
    finally:
        with _quiet_lock:
            _quiet_depth -= 1
            if _quiet_depth == 0:
                sys.stdout = _saved_stdout
                _sink.close()
                _saved_stdout = _sink = None


def serialize_calls(obj: Any, method_name: str) -> Any:
    """Guard `obj.<method_name>` with a lock private to `obj`, in place.

    The locked wrapper is set as an INSTANCE attribute, so the object keeps its
    type (vendored `isinstance` checks still pass) and every existing reference
    to it — including ones vendored code already holds — is covered.
    Idempotent. Returns `obj`.
    """
    method = getattr(obj, method_name)
    if getattr(method, "_serialized", False):
        return obj
    lock = threading.Lock()

    @functools.wraps(method)
    def locked(*args: Any, **kwargs: Any) -> Any:
        with lock:
            return method(*args, **kwargs)

    locked._serialized = True  # type: ignore[attr-defined]
    setattr(obj, method_name, locked)
    return obj
