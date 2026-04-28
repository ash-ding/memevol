"""
Memo manager for alma: persists generated memo code, tracks reward / visit
counts for softmax selection, and orchestrates subprocess evaluation through
baselines.alma.eval_runner.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from baselines.alma.eval_runner import get_output_run_dir, run_evaluation
from baselines.alma.sampling import build_analysis_artifact

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ALMA_ROOT = PROJECT_ROOT / "baselines" / "alma"


class Memo_Manager:
    def __init__(self, archive_root_dir: str = "memo_archive/", status: str = 'search', history_ckpt_path: Optional[str] = None):
        self.project_root = PROJECT_ROOT
        self.ARCHIVE_ROOT = ALMA_ROOT / Path(archive_root_dir) / "dynamicmem"
        self.ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)
        self.LOGS_ROOT = ALMA_ROOT / "logs"
        self.LOGS_ROOT.mkdir(parents=True, exist_ok=True)
        self.no_memo_reward = 0.0
        if history_ckpt_path is None:
            self.memo_db: Dict[str, Any] = {}
        else:
            with open(self.LOGS_ROOT / history_ckpt_path, encoding="utf-8") as f:
                self.memo_db = json.load(f)
            if 'no_mem' in self.memo_db and 'reward' in self.memo_db['no_mem']:
                self.no_memo_reward = self.memo_db['no_mem']['reward']

    def save_memo_structure(self, code_str: str, memo_SHA: str):
        """Register a MemoStructure from code string: save to file and register in memo_db."""
        code_file = self.ARCHIVE_ROOT / f"memo_structure_{memo_SHA}.py"
        code_file.write_text(code_str, encoding="utf-8")
        if memo_SHA not in self.memo_db:
            self.memo_db[memo_SHA] = {}

    async def execute_memo_structure(
        self,
        code_str: str = None,
        target_sha: str = None,
        mode: str = 'check',
        model: str = 'gpt-5-mini',
        eval_n_samples: int = 6,
        status: str = 'search',
        update_type: str = 'all_at_once',
        n_chunks: int = 5,
        max_logs: Optional[int] = None,
        eval_n_qa: Optional[int] = None,
        max_sample_concurrent: int = 6,
        n_score_bins: int = 3,
        samples_per_bin: int = 3,
        judge_model: str = "gpt-5-mini",
        check_n_samples: int = 6,
        check_n_qa: int = 3,
    ):
        """
        Extract python code from markdown-like LLM output, persist it, run the
        subprocess eval, then assemble the compressed analysis artifact via
        sampling.build_analysis_artifact.

        Returns: (all_success, artifact_dict, structure_sha, code)
            all_success — True iff score.json has no invalid users / partial failures
            artifact_dict — {benchmark_eval_score, examples, token_usage}
        """
        if code_str:
            match = re.search(r"```(?:python)?(.*?)```", code_str, re.DOTALL)
            code = match.group(1).strip() if match else code_str.strip()
            if not code:
                raise ValueError("No code found in input")
        else:
            code = self.read_source_code(target_sha)

        if target_sha:
            structure_sha = target_sha
        else:
            ts = time.time()
            rand_uuid = uuid.uuid4().hex
            raw_str = f"{ts}_{rand_uuid}"
            structure_sha = hashlib.sha1(raw_str.encode("utf-8")).hexdigest()[:8]

        self.save_memo_structure(code_str=code, memo_SHA=structure_sha)

        output_run_dir = await run_evaluation(
            memory_SHA=structure_sha,
            mode=mode,
            model=model,
            eval_n_samples=eval_n_samples,
            status=status,
            update_type=update_type,
            n_chunks=n_chunks,
            max_logs=max_logs,
            eval_n_qa=eval_n_qa,
            max_sample_concurrent=max_sample_concurrent,
            judge_model=judge_model,
            check_n_samples=check_n_samples,
            check_n_qa=check_n_qa,
        )

        score_path = output_run_dir / "score.json"
        if not score_path.exists():
            raise FileNotFoundError(
                f"can't find: {score_path}, examination failed with unknown error."
            )

        artifact = build_analysis_artifact(
            output_run_dir,
            n_score_bins=n_score_bins,
            samples_per_bin=samples_per_bin,
        )

        # all_success mirrors the historical contract: any error_info entry in
        # examples (which sampling.build_analysis_artifact appends for invalid /
        # partially failed users) flips it to False and triggers sanity-check retry.
        all_success = not any("error_info" in ex for ex in artifact.get("examples", []))

        # Aggregate subprocess token usage into the main-process global tracker.
        from common.tokens import GLOBAL_TOKEN_TRACKER
        if GLOBAL_TOKEN_TRACKER is not None:
            for model_name, usage_dict in artifact.get("token_usage", {}).items():
                await GLOBAL_TOKEN_TRACKER.update(model_name=model_name, usage=usage_dict)

        return all_success, artifact, structure_sha, code

    def read_source_code(self, memo_SHA: str) -> str:
        """Read the Python source code of a memo structure by SHA."""
        file_path = self.ARCHIVE_ROOT / f"memo_structure_{memo_SHA}.py"
        if not file_path.exists() or not file_path.is_file():
            file_path = self.ARCHIVE_ROOT.parent / 'baseline' / f"memo_structure_{memo_SHA}.py"
            if not file_path.exists():
                raise FileNotFoundError(f"The file {file_path} does not exist or is not a file.")
        return file_path.read_text(encoding="utf-8")

    def read_eval_result(
        self,
        memo_SHA: str,
        mode: str,
        status: str = 'search',
        n_score_bins: int = 3,
        samples_per_bin: int = 3,
    ) -> Dict:
        """Reconstruct the analysis artifact for a previously-run memo."""
        output_run_dir = get_output_run_dir(memo_SHA, status, mode)
        score_path = output_run_dir / "score.json"
        if not score_path.exists():
            raise FileNotFoundError(f"No score.json at {output_run_dir}")
        return build_analysis_artifact(
            output_run_dir,
            n_score_bins=n_score_bins,
            samples_per_bin=samples_per_bin,
        )

    def update_parent(self, memo_sha: str, parent: str):
        assert memo_sha in self.memo_db
        self.memo_db[memo_sha]['parent'] = parent
        if parent:
            self.memo_db[memo_sha]['improve_score'] = (
                self.memo_db[memo_sha]['reward'] - self.memo_db[parent]['reward']
            )

    def update_analysis(self, memo_sha: str, suggestion: dict):
        assert memo_sha in self.memo_db
        self.memo_db[memo_sha]['suggestion'] = suggestion

    def update_reward(self, memo_sha: str, reward: float, alpha: float = 0.5):
        """Compute normalized reward for softmax selection."""
        assert memo_sha in self.memo_db

        def sigmoid(x, lam=1.0):
            return 1 / (1 + np.exp(-lam * x))

        self.memo_db[memo_sha]['reward'] = reward
        self.memo_db[memo_sha]['normalized_reward'] = sigmoid(reward - self.no_memo_reward)
        self.memo_db[memo_sha]['visit_time'] = 0
        penalty = np.log1p(self.memo_db[memo_sha]['visit_time'])
        self.memo_db[memo_sha]['final_score'] = self.memo_db[memo_sha]['normalized_reward'] - alpha * penalty

    def update_visit_time(self, memo_sha: str, alpha: float = 0.5):
        assert memo_sha in self.memo_db
        if 'visit_time' not in self.memo_db[memo_sha]:
            self.memo_db[memo_sha]['visit_time'] = 1
        else:
            self.memo_db[memo_sha]['visit_time'] += 1
        penalty = np.log1p(self.memo_db[memo_sha]['visit_time'])
        self.memo_db[memo_sha]['final_score'] = self.memo_db[memo_sha]['normalized_reward'] - alpha * penalty

    def select_structure(self, maximum_size: int = 5, seed: int = 42, tau: float = 0.5):
        """Softmax over final_score with a local RandomState (avoids global pollution)."""
        rng = np.random.RandomState(seed)

        # Skip reserved top-level keys like `completed_steps` (int) and
        # `token_usage` (dict without final_score). The isinstance(v, dict)
        # guard prevents "argument of type 'int' is not iterable" when a
        # reserved key is an int.
        valid_items = [
            (k, v["final_score"])
            for k, v in self.memo_db.items()
            if isinstance(v, dict) and "final_score" in v
        ]
        if not valid_items:
            raise RuntimeError("No available memory structure for selection.")

        keys, scores = zip(*valid_items)
        scores = np.array(scores, dtype=float)

        logits = scores / tau
        exp_score = np.exp(logits - np.max(logits))
        probs = exp_score / np.sum(exp_score)

        k = min(maximum_size, len(scores))
        selected_indices = rng.choice(len(scores), size=k, replace=False, p=probs)
        return [keys[i] for i in selected_indices]
