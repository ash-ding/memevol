"""Back-compat shim — the registry now lives at baselines/registry.py
(shared by alma, cc, hipporag2). Import from there."""
from baselines.registry import REGISTRY, DATASETS, resolve  # noqa: F401
