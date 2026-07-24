"""Genotype archive + population checkpoint for MemEvolve.

Archive layout (mirrors alma's memo_archive discipline):

    memo_archive/<dataset>/<sha>/
        encode.py / store.py / retrieve.py / manage.py   the genotype Ω
        assembled.py                                     runnable harness file
        meta.json    {sha, parent, iteration, seed_name?, design_rationale,
                      defect_profile?}

    memo_archive/<dataset>/population_<tag>.json         resumable dual-loop
        {"iterations": [{"k", "candidates": [{"sha", "parent", "perf",
          "cost", "delay", "run_dir"}], "parents": [sha, ...]}]}
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from baselines.evolve.memevolve.design_space import (
    OPERATORS,
    assemble_genotype,
    genotype_sha,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MEMEVOLVE_ROOT = PROJECT_ROOT / "baselines" / "evolve" / "memevolve"
ARCHIVE_ROOT = MEMEVOLVE_ROOT / "memo_archive"


class GenotypeArchive:
    def __init__(self, dataset: str):
        self.dataset = dataset
        self.root = ARCHIVE_ROOT / dataset
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, operators: Dict[str, str], meta: Dict[str, Any]) -> str:
        """Assemble + persist a genotype; returns its sha. Idempotent."""
        sha = genotype_sha(operators)
        gdir = self.root / sha
        gdir.mkdir(parents=True, exist_ok=True)
        for name in OPERATORS:
            (gdir / f"{name}.py").write_text(operators[name].strip() + "\n", encoding="utf-8")
        (gdir / "assembled.py").write_text(assemble_genotype(operators), encoding="utf-8")
        (gdir / "meta.json").write_text(
            json.dumps({"sha": sha, **meta}, indent=2, ensure_ascii=False),
            encoding="utf-8")
        return sha

    def load_operators(self, sha: str) -> Dict[str, str]:
        gdir = self.root / sha
        if not gdir.is_dir():
            raise FileNotFoundError(f"genotype {sha} not in archive {self.root}")
        return {name: (gdir / f"{name}.py").read_text(encoding="utf-8")
                for name in OPERATORS}

    def assembled_path(self, sha: str) -> Path:
        path = self.root / sha / "assembled.py"
        if not path.exists():
            raise FileNotFoundError(f"assembled genotype missing: {path}")
        return path

    def read_meta(self, sha: str) -> Dict[str, Any]:
        path = self.root / sha / "meta.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


class PopulationState:
    """Resumable record of the dual-evolution loop."""

    def __init__(self, dataset: str, tag: str = "default"):
        self.path = ARCHIVE_ROOT / dataset / f"population_{tag}.json"
        self.iterations: List[Dict[str, Any]] = []
        if self.path.exists():
            with self.path.open(encoding="utf-8") as f:
                self.iterations = json.load(f).get("iterations", [])

    @property
    def completed(self) -> int:
        return len(self.iterations)

    def last_parents(self) -> List[str]:
        return self.iterations[-1]["parents"] if self.iterations else []

    def all_evaluated(self) -> Dict[str, Dict[str, Any]]:
        """sha → most recent feedback record (perf/cost/delay/run_dir)."""
        out: Dict[str, Dict[str, Any]] = {}
        for it in self.iterations:
            for cand in it["candidates"]:
                out[cand["sha"]] = cand
        return out

    def best(self) -> Optional[Dict[str, Any]]:
        cands = list(self.all_evaluated().values())
        return max(cands, key=lambda c: c.get("perf", 0.0)) if cands else None

    def record_iteration(self, k: int, candidates: List[Dict[str, Any]],
                         parents: List[str]) -> None:
        self.iterations.append({"k": k, "candidates": candidates, "parents": parents})
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"iterations": self.iterations}, indent=2, ensure_ascii=False),
            encoding="utf-8")
