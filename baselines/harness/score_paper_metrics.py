"""Score a harness baseline's traces with the metrics the memory papers report.

The shared judge emits a binary CORRECT/WRONG score, but Mem0 (arXiv 2504.19413)
and MemoryOS (arXiv 2506.06326) both report token-F1 and BLEU-1 on LoCoMo. A
reproduction check therefore cannot use the judge number at all — it has to
recompute the papers' metrics from the same predictions.

    uv run python baselines/harness/score_paper_metrics.py \
        baselines/harness/mem0/results/locomo/search

TWO AVERAGES, deliberately. The papers' "Avg. F1" is the UNWEIGHTED mean over the
four categories — MemoryOS Table 3 reports 36.23, and (35.27+41.15+20.02+48.62)/4
= 36.27 confirms it. Question-weighted means are a different number entirely,
because LoCoMo is 55% single-hop and 6% open-domain, so comparing a weighted
mean against a paper's unweighted one silently misreads the result.
"""
from __future__ import annotations

import argparse
import json
import re
import string
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

# LoCoMo category ids. cat-5 (adversarial) carries no gold answer and is excluded
# at load time by datasets/locomo/env.py — the papers exclude it too.
CATEGORY = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop"}

# Published LoCoMo numbers, GPT-4o-mini, for the reproduction check.
PAPER_F1 = {
    "mem0":     {"single-hop": 38.72, "multi-hop": 28.64, "temporal": 48.93, "open-domain": 47.65},
    "memoryos": {"single-hop": 35.27, "multi-hop": 41.15, "temporal": 20.02, "open-domain": 48.62},
}
PAPER_B1 = {
    "mem0":     {"single-hop": 27.13, "multi-hop": 21.58, "temporal": 40.51, "open-domain": 38.72},
    "memoryos": {"single-hop": 25.22, "multi-hop": 30.76, "temporal": 16.52, "open-domain": 42.99},
}


def _normalize(text: object) -> str:
    """SQuAD-style: lowercase, drop punctuation and articles, collapse spaces."""
    s = "".join(c for c in str(text).lower() if c not in set(string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", s).split())


def token_f1(pred: object, gold: object) -> float:
    p, g = _normalize(pred).split(), _normalize(gold).split()
    if not p or not g:
        return float(p == g)
    hit = sum((Counter(p) & Counter(g)).values())
    if not hit:
        return 0.0
    precision, recall = hit / len(p), hit / len(g)
    return 2 * precision * recall / (precision + recall)


def bleu1(pred: object, gold: object) -> float:
    """Unigram precision — BLEU-1 without the brevity penalty, which is what the
    LoCoMo evaluation scripts these papers build on actually compute."""
    p, g = _normalize(pred).split(), _normalize(gold).split()
    if not p or not g:
        return float(p == g)
    return sum((Counter(p) & Counter(g)).values()) / len(p)


def score_dir(run_dir: Path) -> Dict[str, Dict[str, float]]:
    per_cat: Dict[str, List[List[float]]] = {}
    traces = sorted(run_dir.rglob("traces/*.json"))
    if not traces:
        raise SystemExit(f"no traces under {run_dir}")
    for path in traces:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"  ! unreadable {path.name}: {exc}", file=sys.stderr)
            continue
        for step in payload.get("steps", []) or []:
            cat = CATEGORY.get((step.get("qa_metadata") or {}).get("category"), "other")
            row = per_cat.setdefault(cat, [])
            row.append([
                token_f1(step.get("predicted", ""), step.get("reference", "")),
                bleu1(step.get("predicted", ""), step.get("reference", "")),
                float(step.get("score", 0.0)),
            ])
    return {cat: {
        "f1": 100 * sum(r[0] for r in rows) / len(rows),
        "b1": 100 * sum(r[1] for r in rows) / len(rows),
        "judge": sum(r[2] for r in rows) / len(rows),
        "n": len(rows),
    } for cat, rows in per_cat.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description="token-F1 / BLEU-1 over a baseline's traces")
    ap.add_argument("run_dir", help="e.g. baselines/harness/mem0/results/locomo/search")
    ap.add_argument("--paper", choices=sorted(PAPER_F1), default=None,
                    help="also print the published per-category numbers side by side")
    args = ap.parse_args()

    stats = score_dir(Path(args.run_dir))
    order = ["single-hop", "multi-hop", "temporal", "open-domain"]
    cats = [c for c in order if c in stats] + [c for c in stats if c not in order]

    paper_f1 = PAPER_F1.get(args.paper or "", {})
    paper_b1 = PAPER_B1.get(args.paper or "", {})
    head = f"{'category':<13}{'F1':>7}{'BLEU-1':>8}{'judge':>7}{'n':>6}"
    if paper_f1:
        head += f"{'paper F1':>10}{'paper B1':>10}"
    print(head)
    for cat in cats:
        s = stats[cat]
        line = f"{cat:<13}{s['f1']:7.1f}{s['b1']:8.1f}{s['judge']:7.2f}{s['n']:6d}"
        if paper_f1:
            line += f"{paper_f1.get(cat, float('nan')):10.1f}{paper_b1.get(cat, float('nan')):10.1f}"
        print(line)

    known = [c for c in cats if c in order]
    total = sum(stats[c]["n"] for c in cats)
    # Unweighted = the papers' "Avg."; weighted = what the benchmark's own
    # category mix produces. They differ a lot on LoCoMo, hence both.
    print()
    if known:
        un_f1 = sum(stats[c]["f1"] for c in known) / len(known)
        un_b1 = sum(stats[c]["b1"] for c in known) / len(known)
        target = (f"   (paper: {sum(paper_f1.values()) / len(paper_f1):.1f})"
                  if paper_f1 else "")
        print(f"unweighted mean over {len(known)} categories — the papers' 'Avg.':")
        print(f"    F1 {un_f1:.1f}   BLEU-1 {un_b1:.1f}{target}")
    w_f1 = sum(stats[c]["f1"] * stats[c]["n"] for c in cats) / total
    w_b1 = sum(stats[c]["b1"] * stats[c]["n"] for c in cats) / total
    w_j = sum(stats[c]["judge"] * stats[c]["n"] for c in cats) / total
    print(f"question-weighted over all {total} questions (NOT comparable to the papers):")
    print(f"    F1 {w_f1:.1f}   BLEU-1 {w_b1:.1f}   judge {w_j:.3f}")


if __name__ == "__main__":
    main()
