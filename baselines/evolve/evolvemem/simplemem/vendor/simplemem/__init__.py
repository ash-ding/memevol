"""Vendored SimpleMem (aiming-lab/SimpleMem @ main, MIT) — text core only.

PATCH #1 (of 3; see the harness README for the full list): upstream's
package __init__ pulls in the productized surface (router / AutoMemory /
multimodal / MCP), none of which the baseline uses and which would drag in
heavy optional deps. This minimal __init__ exposes only the paper-relevant
text pipeline.
"""

from simplemem.text.system import SimpleMemSystem  # noqa: F401

__all__ = ["SimpleMemSystem"]
