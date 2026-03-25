import re
import os
import sys
import traceback
from datetime import datetime
from typing import Optional
import time
import uuid
import hashlib
from pathlib import Path
import numpy as np
from typing import Dict, List, Any
from eval_runner import run_evaluation
import json


class Memo_Manager:
    def __init__(self, archive_root_dir: str = "memo_archive/", status: str = 'train', history_ckpt_path: Optional[str] = None):
        self.project_root = Path(__file__).resolve().parent.parent
        self.ARCHIVE_ROOT = self.project_root / Path(archive_root_dir) / "dynamicmem"
        self.ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)
        self.no_memo_reward = 0.0
        if history_ckpt_path is None:
            self.memo_db = {}
        else:
            with open(self.project_root / 'logs' / history_ckpt_path, encoding="utf-8") as f:
                self.memo_db = json.load(f)

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
        mode: str = 'test',
        model: str = 'gpt-5-mini',
        train_size: int = 6,
        status: str = 'train',
        update_type: str = 'all_at_once',
        n_chunks: int = 5,
        max_logs: Optional[int] = None,
        qa_sample_size: Optional[int] = None,
        max_concurrent: int = 6,
        n_score_bins: int = 3,
        samples_per_bin: int = 3,
        judge_model: str = "gpt-5-mini",
    ):
        """
        Extract python code from markdown-like LLM output (if any), write it to disk,
        then run test/eval via eval_runner.

        Args:
            mode: 'test' — quick sanity check (3 random users); 'eval' — full train_size users.
            status: data split — 'train' or 'eval'.
            update_type: how app_logs are batched for general_update —
                'all_at_once' | 'chunked' | 'sequential'.
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

        await run_evaluation(
            memory_SHA=structure_sha,
            mode=mode,
            model=model,
            train_size=train_size,
            status=status,
            update_type=update_type,
            n_chunks=n_chunks,
            max_logs=max_logs,
            qa_sample_size=qa_sample_size,
            max_concurrent=max_concurrent,
            n_score_bins=n_score_bins,
            samples_per_bin=samples_per_bin,
            judge_model=judge_model,
        )

        json_path = Path(f"evals/logs/dynamicmem/{structure_sha}_{mode}.json")
        if not json_path.exists():
            raise FileNotFoundError(f"can't find: {json_path}, examination failed with unknown error.")

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        examples = data.get('examples', [])
        all_success = True
        if examples:
            for example in examples:
                if "error_info" in example:
                    all_success = False

        # Update global token tracker
        token_usg = data.get('token_usage', {})
        from evals.agents.base import GLOBAL_TOKEN_TRACKER
        if GLOBAL_TOKEN_TRACKER is not None:
            for model_name, usage_dict in token_usg.items():
                await GLOBAL_TOKEN_TRACKER.update(model_name=model_name, usage=usage_dict)

        return all_success, data, structure_sha, code

    def read_source_code(self, memo_SHA: str) -> str:
        """Read the Python source code of a memo structure by SHA."""
        file_path = self.ARCHIVE_ROOT / f"memo_structure_{memo_SHA}.py"
        if not file_path.exists() or not file_path.is_file():
            file_path = self.ARCHIVE_ROOT.parent / 'baseline' / f"memo_structure_{memo_SHA}.py"
            if not file_path.exists():
                raise FileNotFoundError(f"The file {file_path} does not exist or is not a file.")
        return file_path.read_text(encoding="utf-8")

    def read_eval_result(self, memo_SHA: str, mode: str) -> Dict:
        store_path = Path('evals/logs/')
        store_path.mkdir(parents=True, exist_ok=True)
        eval_path = self.project_root / store_path / 'dynamicmem' / f"{memo_SHA}_{mode}.json"
        with open(eval_path, "r", encoding="utf-8") as f:
            eval_result = json.load(f)
        return eval_result

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
        np.random.seed(seed)

        valid_items = [(k, v["final_score"]) for k, v in self.memo_db.items() if "final_score" in v]
        if not valid_items:
            raise RuntimeError("No available memory structure for selection.")

        keys, scores = zip(*valid_items)
        scores = np.array(scores, dtype=float)

        logits = scores / tau
        exp_score = np.exp(logits - np.max(logits))
        probs = exp_score / np.sum(exp_score)

        k = min(maximum_size, len(scores))
        selected_indices = np.random.choice(len(scores), size=k, replace=False, p=probs)
        return [keys[i] for i in selected_indices]
