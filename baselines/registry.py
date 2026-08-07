"""Back-compat re-export — the dataset registry moved to datasets/registry.py
(2026-08: it is pure dataset-layer wiring, needed by common.evaluate too).
Every existing `from baselines.registry import ...` keeps working."""
from datasets.registry import REGISTRY, DATASETS, resolve  # noqa: F401
