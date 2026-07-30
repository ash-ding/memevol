"""EvolveMem meta-analyzer — two guard modes + persistent evolution state.

ELITIST (default — matches the OFFICIAL EvolveMem code, whose
EvolutionConfig ships elitist=True with acceptance_threshold=0.003,
max_changes_per_round=2, max_consec_noaccept=5; its comments call this
"the default going forward" over the paper's free-form update style):
    a round's candidate θ is ADOPTED as incumbent only if its score
    exceeds the incumbent's by > acceptance_threshold; otherwise it is
    rejected and the next candidate is built from the incumbent again.
    At most max_changes_per_round adjustments apply per round; evolution
    stops after max_consec_noaccept consecutive rejections.

PAPER (--guard paper — the paper's Eq. 4, which the official code moved
away from):
    θ_{r+1} = θ*                      if f_{r-1} − f_r > τ_rev   (revert)
            = perturb(θ_r)            if |f_r − f_{r-1}| < ε for 2 rounds (explore)
            = clamp(θ_r ⊕ Δθ_r)       otherwise                  (apply)
    Convergence: improvement below ε with no pending adjustments.

State lives in config_archive/<dataset>/evolution_log.json and is resumable:
each round records θ, score, the diagnosis, the guard action, and the run
dir, so re-launching with the same args continues where it stopped.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from common.logger import get_logger

from baselines.evolve.evolvemem.action_space import (
    apply_adjustments,
    clamp_config,
    perturb_config,
)

log = get_logger("main")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ARCHIVE_ROOT = PROJECT_ROOT / "baselines" / "evolve" / "evolvemem" / "config_archive"


class EvolutionState:
    """Round records + best-so-far tracking, persisted as JSON."""

    def __init__(self, dataset: str, tag: str = "default"):
        self.path = ARCHIVE_ROOT / dataset / f"evolution_log_{tag}.json"
        self.rounds: List[Dict[str, Any]] = []
        self.stagnant_rounds = 0
        if self.path.exists():
            with self.path.open(encoding="utf-8") as f:
                data = json.load(f)
            self.rounds = data.get("rounds", [])
            self.stagnant_rounds = int(data.get("stagnant_rounds", 0))

    # ------------------------------------------------------------------

    @property
    def completed(self) -> int:
        return len(self.rounds)

    def best(self) -> Optional[Dict[str, Any]]:
        scored = [r for r in self.rounds if r.get("score") is not None]
        return max(scored, key=lambda r: r["score"]) if scored else None

    def next_config(self) -> Dict[str, Any]:
        """The θ the NEXT round should evaluate (recorded by the previous
        round's guard decision), or defaults for round 0."""
        if not self.rounds:
            return clamp_config({})
        return self.rounds[-1]["next_config"]

    def record_round(
        self,
        config: Dict[str, Any],
        score: float,
        diagnosis: Dict[str, Any],
        action: str,
        next_config: Dict[str, Any],
        run_dir: Path,
    ) -> None:
        self.rounds.append({
            "round": self.completed,
            "config": config,
            "score": score,
            "diagnosis_summary": diagnosis.get("summary", ""),
            "diagnosis": diagnosis,
            "action": action,
            "next_config": next_config,
            "run_dir": str(run_dir),
        })
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        best = self.best()
        payload = {
            "rounds": self.rounds,
            "stagnant_rounds": self.stagnant_rounds,
            "best": {"round": best["round"], "score": best["score"],
                     "config": best["config"]} if best else None,
        }
        self.path.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                             encoding="utf-8")


def guarded_update(
    state: EvolutionState,
    current_config: Dict[str, Any],
    score: float,
    proposal: Dict[str, Any],
    tau_rev: float = 0.05,
    epsilon: float = 0.01,
    explore_seed: int = 0,
    space=None,
) -> Tuple[Dict[str, Any], str, bool]:
    """Return (θ_{r+1}, action, converged). Mutates state.stagnant_rounds.

    `space` — action-space module providing apply_adjustments / clamp_config /
    perturb_config (e.g. action_space_simplemem for the real-substrate θ);
    None = the native EvolveMemMemo space."""
    if space is None:
        from baselines.evolve.evolvemem import action_space as space
    prev_score = state.rounds[-1]["score"] if state.rounds else None
    best = state.best()

    # Branch 1 — revert-on-regression.
    if prev_score is not None and prev_score - score > tau_rev and best is not None:
        state.stagnant_rounds = 0
        log.info(f"guard: REVERT (f dropped {prev_score:.3f}→{score:.3f} > τ_rev={tau_rev}); "
                 f"back to best round {best['round']} ({best['score']:.3f})")
        return dict(best["config"]), "revert", False

    # Stagnation bookkeeping for branch 2.
    if prev_score is not None and abs(score - prev_score) < epsilon:
        state.stagnant_rounds += 1
    else:
        state.stagnant_rounds = 0

    # Branch 2 — explore-on-stagnation (2 consecutive flat rounds).
    if state.stagnant_rounds >= 2:
        state.stagnant_rounds = 0
        log.info("guard: EXPLORE (2 stagnant rounds) — random perturbation η_exp")
        return space.perturb_config(current_config, seed=explore_seed), "explore", False

    # Branch 3 — apply the diagnosis proposal.
    adjustments = list(proposal.get("adjustments", []))
    per_cat = proposal.get("per_category")
    next_cfg = space.apply_adjustments(current_config, adjustments)
    if isinstance(per_cat, list) and per_cat:
        next_cfg = space.clamp_config({**next_cfg, "per_category": per_cat})

    # Plateau convergence: improvement below ε with nothing left to try this
    # round (Algorithm 1 line 17) — flag converged AFTER recording the apply.
    converged = (
        prev_score is not None
        and score - prev_score < epsilon
        and state.completed >= 2
        and not adjustments
    )
    action = "apply" if adjustments or per_cat else "noop"
    log.info(f"guard: {action.upper()} ({len(adjustments)} adjustments"
             + (f", {len(per_cat)} per-category rules" if isinstance(per_cat, list) and per_cat else "")
             + ")")
    return next_cfg, action, converged


def elitist_update(
    state: EvolutionState,
    current_config: Dict[str, Any],
    score: float,
    proposal: Dict[str, Any],
    acceptance_threshold: float = 0.003,
    max_changes_per_round: int = 2,
    max_consec_noaccept: int = 5,
    space=None,
) -> Tuple[Dict[str, Any], str, bool]:
    """Official-code guard: strict hill-climbing with an incumbent.

    Mirrors the upstream EvolutionConfig(elitist=True) semantics:
      - round 0 is always accepted (it defines the first incumbent);
      - a later round is accepted iff score > incumbent_score +
        acceptance_threshold; otherwise rejected — the next candidate is
        built from the INCUMBENT config, not the rejected one;
      - at most `max_changes_per_round` adjustments are applied per round
        (the diagnosis is asked to order proposals by expected impact);
      - `max_consec_noaccept` consecutive rejections ⇒ converged.

    Returns (θ_{r+1}, action, converged); action ∈ {accept, reject}.
    Records rely on `action` to reconstruct the incumbent on resume.
    """
    if space is None:
        from baselines.evolve.evolvemem import action_space as space

    # Incumbent = the last ACCEPTED round (round 0 always counts).
    incumbent_cfg, incumbent_score = current_config, score   # round-0 case
    consec_noaccept = 0
    accepted_this_round = True
    if state.rounds:
        accepted = [r for r in state.rounds
                    if r["round"] == 0 or r.get("action") == "accept"]
        inc = accepted[-1]
        # Did THIS round's candidate beat the incumbent?
        if score > inc["score"] + acceptance_threshold:
            incumbent_cfg, incumbent_score = current_config, score
        else:
            incumbent_cfg, incumbent_score = inc["config"], inc["score"]
            accepted_this_round = False
        # Consecutive-rejection streak including this round.
        for r in reversed(state.rounds):
            if r["round"] == 0 or r.get("action") == "accept":
                break
            consec_noaccept += 1
        if not accepted_this_round:
            consec_noaccept += 1

    action = "accept" if accepted_this_round else "reject"
    if not accepted_this_round:
        log.info(f"guard(elitist): REJECT (score {score:.3f} ≤ incumbent "
                 f"{incumbent_score:.3f} + {acceptance_threshold}); "
                 f"consec_noaccept={consec_noaccept}")
    else:
        log.info(f"guard(elitist): ACCEPT (incumbent score → {incumbent_score:.3f})")

    if consec_noaccept >= max_consec_noaccept:
        log.info(f"guard(elitist): CONVERGED ({consec_noaccept} consecutive rejections)")
        return dict(incumbent_cfg), action, True

    # Next candidate = incumbent ⊕ top-N adjustments (diagnosis orders them
    # by expected impact; the cap is the official max_changes_per_round).
    adjustments = list(proposal.get("adjustments", []))[:max_changes_per_round]
    next_cfg = space.apply_adjustments(incumbent_cfg, adjustments)
    per_cat = proposal.get("per_category")
    if isinstance(per_cat, list) and per_cat:
        next_cfg = space.clamp_config({**next_cfg, "per_category": per_cat})
    if not adjustments and not per_cat:
        # Nothing left to try — treat as terminal (mirrors official
        # convergence when the diagnosis has no further proposals).
        return dict(incumbent_cfg), action, True
    return next_cfg, action, False
