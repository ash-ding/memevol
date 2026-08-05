"""SimpleMem — vendored subset for the memevol simplemem baseline.

INTEGRATION ADAPTATION — this file is the ONE vendored file that is NOT
byte-identical to upstream. Upstream's ``simplemem/__init__.py`` eagerly imports
``simplemem.router`` (the AutoMemory text/multimodal router) and
``simplemem.config``/``simplemem.optimize``, which in turn pull in the heavy
``multimodal/`` and ``evolver/`` subtrees (LanceDB-agnostic vision/audio models,
EvolveMem search loop, langgraph, litellm, …) that this baseline does NOT use.

Only the text pipeline is vendored: ``simplemem.text.system.SimpleMemSystem`` and
its ``simplemem.core.*`` dependencies (both byte-identical to upstream @
db80b6a7c591e0ea730a058e9f5fc4eb06572299). The baseline imports the submodule
directly (``from simplemem.text.system import SimpleMemSystem``), so this package
initializer only needs to exist — it intentionally re-exports nothing and imports
nothing, to keep the un-vendored subtrees off the import path.

See baselines/harness/simplemem/README.md for the full provenance + verification
command.
"""

__version__ = "0.3.0"  # matches upstream simplemem.__version__ at the vendored commit
