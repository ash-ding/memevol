"""Back-compat shim — the memory-system contract moved to common/memo_class.py
(2026-08-06; class renamed MemoStructure → MemoClass). Every historical import
path keeps working:

    from common.harness_base import MemoStructure   # == MemoClass
    from common.harness_base import Basic_Recorder  # legacy envelope path
"""
from common.memo_class import MemoClass, MemoStructure  # noqa: F401
from common.recorder import Basic_Recorder  # noqa: F401
