from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
from evals.agents.memo_structure import MemoStructure
from evals.agents.base import init_global_tracker
from evals.utils.hire_agent import Agent
from core.memo_manager import Memo_Manager
from core.meta_agent_prompt import build_analysis_prompt, build_generate_new_code_prompt, build_reflection_prompt
import os
import asyncio
import json
from evals.logger import get_logger
from datetime import datetime
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
import importlib

log = get_logger("main")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "evals" / "logs"

RECORDER = {
    'dynamicmem': "DynamicMemRecorder",
}


class MetaAgent:
    def __init__(
        self,
        meta_model: str = 'gpt-5',
        execution_model: str = 'gpt-5-mini',
        status: str = 'search',
        history_ckpt_path: Optional[str] = None,
    ):
        self.memo_manager = Memo_Manager(
            status=status,
            history_ckpt_path=history_ckpt_path,
        )
        self.history_ckpt_path = history_ckpt_path
        self.examine_trial = 3
        self.meta_model = meta_model
        self.execution_model = execution_model

    def _get_recorder_class(self):
        from envs.dynamicmem_env import DynamicMemRecorder
        return DynamicMemRecorder

    def read_memo_info(self, memo_SHA: str, mode: str = 'check'):
        assert memo_SHA in self.memo_manager.memo_db

        source_code: str = self.memo_manager.read_source_code(memo_SHA)
        evals = self.memo_manager.read_eval_result(memo_SHA, mode)
        evals["source_code"] = source_code
        parent = self.memo_manager.memo_db[memo_SHA].get('parent')
        if parent:
            try:
                evals['improve_example'] = {
                    'source_code': self.memo_manager.read_source_code(parent),
                    'suggestion': self.memo_manager.memo_db[memo_SHA]['suggestion'],
                    'improve_score': self.memo_manager.memo_db[memo_SHA]['improve_score'],
                }
            except Exception as e:
                log.warning(f"Fail to form improve sample: {e}")
        return evals

    async def analyze_memo_structure(self, memo_SHA: str):
        """Analyze current MemoStructure and return suggestions."""
        memo_info = self.read_memo_info(memo_SHA, mode='eval')
        sys_msg, user_msg, output_schema = build_analysis_prompt(memo_info)
        analysis_agent = Agent(
            system_prompt=sys_msg,
            output_schema=output_schema,
            model=self.meta_model,
            timeout=600,
        )
        result = await analysis_agent.ask(user_msg, reasoning_effort='medium')
        return result, memo_info

    async def generate_new_code(
        self,
        analysis_result: Dict[str, Any] = {},
        memo_info: Dict[str, Any] = {},
    ) -> str:
        """Generate a new MemoStructure code based on analysis. Returns code_str."""
        recorder = self._get_recorder_class()
        sys_msg, user_msg = build_generate_new_code_prompt(
            memo_info=memo_info,
            analysis_result=analysis_result,
            recorder=recorder,
        )
        gen_code_agent = Agent(system_prompt=sys_msg, model=self.meta_model, timeout=600)
        code_str = await gen_code_agent.ask(user_msg, reasoning_effort='medium')
        return code_str

    async def examine_new_code(
        self,
        code_str: str,
        eval_n_users: int = 6,
        update_type: str = 'all_at_once',
        n_chunks: int = 5,
        max_logs: Optional[int] = None,
        eval_n_qa: Optional[int] = None,
        max_user_concurrent: int = 6,
        n_score_bins: int = 3,
        samples_per_bin: int = 3,
        judge_model: str = "gpt-5-mini",
        check_n_users: int = 3,
        check_n_qa: int = 10,
    ):
        """
        Sanity-check (gating step) for LLM-generated code: verify it runs without crashing.
        Retries up to self.examine_trial times, using build_reflection_prompt to fix errors.
        On success, returns (memo_SHA, code_str).
        """
        memo_SHA = None
        recorder = self._get_recorder_class()
        for attempt in range(self.examine_trial):
            log.info(f"Start examination for round {attempt}")
            try:
                success, examine_log, memo_SHA, code_str = await self.memo_manager.execute_memo_structure(
                    code_str=code_str,
                    target_sha=memo_SHA,
                    mode='check',
                    model=self.execution_model,
                    eval_n_users=eval_n_users,
                    status='search',
                    update_type=update_type,
                    n_chunks=n_chunks,
                    max_logs=max_logs,
                    eval_n_qa=eval_n_qa,
                    max_user_concurrent=max_user_concurrent,
                    n_score_bins=n_score_bins,
                    samples_per_bin=samples_per_bin,
                    judge_model=judge_model,
                    check_n_users=check_n_users,
                    check_n_qa=check_n_qa,
                )
            except Exception as e:
                success = False
                examine_log = e
            if success:
                break
            log.warning(f"Fail examination for round {attempt}")
            try:
                sys_msg, user_msg = build_reflection_prompt(
                    code_str=code_str,
                    recorder=recorder,
                    error_msg=examine_log,
                )
                reflect_fix_agent = Agent(system_prompt=sys_msg, model=self.meta_model, timeout=600)
                code_str = await reflect_fix_agent.ask(user_msg)
                log.info(f"Retry finished for round {attempt}")
            except Exception as e:
                log.warning(f"Reflection LLM failed for round {attempt}: {e}")
        if not success:
            raise RuntimeError(f"Fail to revise code in {self.examine_trial} attempts.")
        self.memo_manager.save_memo_structure(code_str, memo_SHA)
        return memo_SHA, code_str

    async def run_single_memo(
        self,
        memo_SHA: str,
        eval_n_users: int = 6,
        status: str = 'search',
        update_type: str = 'all_at_once',
        n_chunks: int = 5,
        max_logs: Optional[int] = None,
        eval_n_qa: Optional[int] = None,
        max_user_concurrent: int = 6,
        n_score_bins: int = 3,
        samples_per_bin: int = 3,
        judge_model: str = "gpt-5-mini",
        **kargs,
    ):
        if status == 'search':
            self.memo_manager.update_visit_time(memo_SHA)

            log.info("[blue]━━━━━━━━━━━━━━━ STRUCTURE REFLECTION ━━━━━━━━━━━━━━━[/blue]")
            analysis_result, memo_info = await self.analyze_memo_structure(memo_SHA=memo_SHA)
            log.info(f"[blue][FINISH STRUCTURE REFLECTION]: {analysis_result.get('suggested_changes', [{}])[0].get('what', '')[:50]}...[/blue]")

            log.info("[blue]━━━━━━━━━━━━━━━ CODE GENERATION ━━━━━━━━━━━━━━━[/blue]")
            new_code = await self.generate_new_code(
                analysis_result=analysis_result, memo_info=memo_info
            )

            new_memo_SHA, new_code = await self.examine_new_code(
                code_str=new_code,
                eval_n_users=eval_n_users,
                update_type=update_type,
                n_chunks=n_chunks,
                max_logs=max_logs,
                eval_n_qa=eval_n_qa,
                max_user_concurrent=max_user_concurrent,
                n_score_bins=n_score_bins,
                samples_per_bin=samples_per_bin,
                judge_model=judge_model,
                check_n_users=kargs.get('check_n_users', 3),
                check_n_qa=kargs.get('check_n_qa', 10),
            )

            self.memo_manager.update_analysis(memo_sha=new_memo_SHA, suggestion=analysis_result)
            log.info(f"[blue][FINISH CODE GENERATION] Code SHA: {new_memo_SHA} | Code example: {new_code[:30]}...[/blue]")
        else:
            new_memo_SHA = memo_SHA

        log.info("[blue]━━━━━━━━━━━━━━━ CODE EXECUTION ━━━━━━━━━━━━━━━[/blue]")
        _, eval_result, _, _ = await self.memo_manager.execute_memo_structure(
            target_sha=new_memo_SHA,
            mode='eval',
            model=self.execution_model,
            eval_n_users=eval_n_users,
            status=status,
            update_type=update_type,
            n_chunks=n_chunks,
            max_logs=max_logs,
            eval_n_qa=eval_n_qa,
            max_user_concurrent=max_user_concurrent,
            n_score_bins=n_score_bins,
            samples_per_bin=samples_per_bin,
            judge_model=judge_model,
        )

        if status == 'search':
            self.memo_manager.update_reward(
                new_memo_SHA,
                eval_result.get('benchmark_eval_score', {}).get('benchmark_overall_eval_score', 0.0),
            )
            self.memo_manager.update_parent(new_memo_SHA, memo_SHA)
        log.info(
            f"[blue][FINISH CODE EXECUTION] Code SHA: {new_memo_SHA} | "
            f"Reward: {eval_result.get('benchmark_eval_score', {}).get('benchmark_overall_eval_score', 0.0)}[/blue]"
        )

    async def forward(
        self,
        steps: int = 10,
        max_memo_concurrent: int = 2,
        max_user_concurrent: int = 6,
        result_dir: str = "check",
        eval_n_users: int = 6,
        update_type: str = 'all_at_once',
        n_chunks: int = 5,
        max_logs: Optional[int] = None,
        eval_n_qa: Optional[int] = None,
        n_score_bins: int = 3,
        samples_per_bin: int = 3,
        judge_model: str = "gpt-5-mini",
        check_n_users: int = 3,
        check_n_qa: int = 10,
    ):
        project_root = Path(__file__).resolve().parent.parent

        if self.history_ckpt_path is None:
            # Run no_mem baseline to calibrate normalized reward
            _, eval_result, _, _ = await self.memo_manager.execute_memo_structure(
                target_sha='no_mem',
                mode='eval',
                model=self.execution_model,
                eval_n_users=eval_n_users,
                status='search',
                update_type=update_type,
                n_chunks=n_chunks,
                max_logs=max_logs,
                eval_n_qa=eval_n_qa,
                max_user_concurrent=max_user_concurrent,
                n_score_bins=n_score_bins,
                samples_per_bin=samples_per_bin,
                judge_model=judge_model,
            )
            self.memo_manager.no_memo_reward = (
                eval_result.get('benchmark_eval_score', {}).get('benchmark_overall_eval_score', 0.0)
            )
            # Record baseline score in memo_db for visibility (no final_score → won't be selected for evolution)
            self.memo_manager.memo_db['no_mem'] = {
                'reward': self.memo_manager.no_memo_reward,
                'role': 'baseline',
            }

            tracker = init_global_tracker()

            log.info("[blue]━━━━━━━━━━━━━━━ CODE GENERATION ━━━━━━━━━━━━━━━[/blue]")
            new_code = await self.generate_new_code()

            new_memo_SHA, new_code = await self.examine_new_code(
                code_str=new_code,
                eval_n_users=eval_n_users,
                update_type=update_type,
                n_chunks=n_chunks,
                max_logs=max_logs,
                eval_n_qa=eval_n_qa,
                max_user_concurrent=max_user_concurrent,
                n_score_bins=n_score_bins,
                samples_per_bin=samples_per_bin,
                judge_model=judge_model,
                check_n_users=check_n_users,
                check_n_qa=check_n_qa,
            )
            log.info(f"[blue][FINISH CODE GENERATION] Code SHA: {new_memo_SHA} | Code example: {new_code[:30]}...[/blue]")

            log.info("[blue]━━━━━━━━━━━━━━━ CODE EXECUTION ━━━━━━━━━━━━━━━[/blue]")
            _, eval_result, _, _ = await self.memo_manager.execute_memo_structure(
                target_sha=new_memo_SHA,
                mode='eval',
                model=self.execution_model,
                eval_n_users=eval_n_users,
                status='search',
                update_type=update_type,
                n_chunks=n_chunks,
                max_logs=max_logs,
                eval_n_qa=eval_n_qa,
                max_user_concurrent=max_user_concurrent,
                n_score_bins=n_score_bins,
                samples_per_bin=samples_per_bin,
                judge_model=judge_model,
            )
            self.memo_manager.update_reward(
                new_memo_SHA,
                eval_result.get('benchmark_eval_score', {}).get('benchmark_overall_eval_score', 0.0),
            )
            self.memo_manager.update_parent(new_memo_SHA, '')
            log.info(
                f"[blue][FINISH CODE EXECUTION] Code SHA: {new_memo_SHA} | "
                f"Reward: {eval_result.get('benchmark_eval_score', {}).get('benchmark_overall_eval_score', 0.0)}[/blue]"
            )
        else:
            tracker = init_global_tracker()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if not self.history_ckpt_path:
            file_name = f"{result_dir}_dynamicmem_{update_type}_{steps}_{timestamp}.json"
        else:
            file_name = self.history_ckpt_path

        semaphore = asyncio.Semaphore(max_memo_concurrent)
        for step in range(steps):
            with open(project_root / "logs" / file_name, "w", encoding="utf-8") as f:
                json.dump(self.memo_manager.memo_db, f, ensure_ascii=False, indent=2)
            result_path = project_root / "logs" / file_name
            log.info(f"[LOG RESULT] Results saved to: {result_path}")
            log.info(f"[blue]\n━━━━━━━━━━━━━━━ STEP [{step + 1}] ━━━━━━━━━━━━━━━[/blue]")
            memo_SHA_list = self.memo_manager.select_structure()
            log.info(f"Select memos {memo_SHA_list}")

            async def sem_task(memo_SHA):
                async with semaphore:
                    max_retries = 2
                    for attempt in range(max_retries + 1):
                        try:
                            await self.run_single_memo(
                                memo_SHA,
                                eval_n_users=eval_n_users,
                                update_type=update_type,
                                n_chunks=n_chunks,
                                max_logs=max_logs,
                                eval_n_qa=eval_n_qa,
                                max_user_concurrent=max_user_concurrent,
                                n_score_bins=n_score_bins,
                                samples_per_bin=samples_per_bin,
                                judge_model=judge_model,
                                check_n_users=check_n_users,
                                check_n_qa=check_n_qa,
                            )
                            break
                        except Exception as e:
                            err_str = str(e).lower()
                            is_transient = any(kw in err_str for kw in ["timed out", "timeout", "502", "503", "connection"])
                            if is_transient and attempt < max_retries:
                                delay = 30 * (attempt + 1)
                                log.warning(f"Memo {memo_SHA} failed (attempt {attempt+1}/{max_retries+1}): {e}, retrying in {delay}s")
                                await asyncio.sleep(delay)
                            else:
                                log.warning(f"Memo {memo_SHA} failed: {e}")
                                break

            await asyncio.gather(*(sem_task(sha) for sha in memo_SHA_list))

        self.memo_manager.memo_db['token_usage'] = tracker.summary()
        with open(project_root / "logs" / file_name, "w", encoding="utf-8") as f:
            json.dump(self.memo_manager.memo_db, f, ensure_ascii=False, indent=2)
        result_path = project_root / "logs" / file_name
        log.info(f"Results saved to: {result_path}")
        tracker.print_summary()
