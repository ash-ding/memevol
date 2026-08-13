"""Mem0 — vendored subset for the memevol mem0 baseline.

INTEGRATION ADAPTATION — this file is the ONE vendored file that is NOT
byte-identical to upstream. Two reasons, both forced by vendoring:

  1. Upstream reads its version from installed package metadata
     (``importlib.metadata.version("mem0ai")``). A vendored copy has no
     ``mem0ai`` distribution installed, so that call raises
     ``PackageNotFoundError`` at import time. The version is inlined instead —
     ``mem0.memory.telemetry`` reads ``mem0.__version__``.
  2. Upstream eagerly imports ``mem0.client.main`` (``MemoryClient`` /
     ``AsyncMemoryClient``, the hosted-platform HTTP client), which this
     baseline does not use and which is not vendored. Importing it here would
     put the un-vendored ``client/`` subtree on the import path.

Everything else under ``src/mem0/`` is byte-identical to upstream @ v2.0.17
(``12c47f524935692e27ad48d829f35fa1e4417181``). See
baselines/harness/mem0/README.md for the full provenance + verification command.
"""

__version__ = "2.0.17"  # upstream reads this from mem0ai package metadata

from mem0.memory.main import AsyncMemory, Memory  # noqa
