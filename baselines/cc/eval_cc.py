"""
Claude Code baseline evaluation on DynamicMem benchmark.

Uses Claude Code SDK to answer QA questions by searching app log files
with built-in tools (Read, Grep, Glob). Same judge as meta-learning pipeline.

Usage:
    python baselines/cc/eval_cc.py --model claude-sonnet-4-20250514
    python baselines/cc/eval_cc.py --model claude-opus-4-20250514
    python baselines/cc/eval_cc.py --model all   # run both
    python baselines/cc/eval_cc.py --model claude-sonnet-4-20250514 --dry_run  # single QA test
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import logging

import numpy as np

# Add project root to sys.path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from common.judge import Judge
from datasets.dynamicmem.env import get_task_list, load_user_data
from claude_code_sdk import query, ClaudeCodeOptions, AssistantMessage, ResultMessage, UserMessage, TextBlock, ToolUseBlock, ToolResultBlock, SystemMessage
from claude_code_sdk._errors import MessageParseError

# Monkey-patch SDK's parse_message to skip unknown message types
# (e.g. rate_limit_event not yet handled in SDK 0.0.25).
# Without this, the entire query stream aborts on first unknown event.
import claude_code_sdk._internal.message_parser as _mp
import claude_code_sdk._internal.client as _cl

_original_parse = _mp.parse_message

def _lenient_parse(data):
    try:
        return _original_parse(data)
    except MessageParseError:
        # Return a harmless SystemMessage so the stream continues
        return SystemMessage(subtype="unknown", data=data)

_cl.parse_message = _lenient_parse

# ---------------------------------------------------------------------------
# Logging — console + file
# ---------------------------------------------------------------------------
LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

log = logging.getLogger("cc_eval")
log.setLevel(logging.DEBUG)
log.propagate = False

# Console handler — INFO level, concise
_console = logging.StreamHandler(sys.stdout)
_console.setLevel(logging.INFO)
_console.setFormatter(logging.Formatter("%(message)s"))
log.addHandler(_console)

# File handler — DEBUG level, with timestamps (set per-run in main())
_file_handler: logging.FileHandler | None = None


def _init_file_logger(model_short: str, dry_run: bool):
    global _file_handler
    if _file_handler:
        log.removeHandler(_file_handler)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "_dryrun" if dry_run else ""
    log_path = LOG_DIR / f"cc_{model_short}_{ts}{suffix}.log"
    _file_handler = logging.FileHandler(log_path, encoding="utf-8")
    _file_handler.setLevel(logging.DEBUG)
    _file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    log.addHandler(_file_handler)
    log.info(f"Log file: {log_path}")


# Load .env for OPENAI_API_KEY (used by judge)
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass


SYSTEM_PROMPT = """\
You are a personal assistant with access to a user's app activity logs.
You will be asked questions about the user's habits, preferences, and life patterns.

### Data
The file `app_log_large.json` in your working directory contains the user's complete app usage logs.
Each entry has: app_log_id, timestamp, app_name, api_name, request, response, and metadata.
This is the ONLY file you should read. Do not access any other files or directories.

### Task
Use your tools to search the app logs for relevant information,
then answer the question.

### Guidelines
- Be specific and factual. Include concrete details (times, days, locations, frequencies) when available.
- If you are not certain, provide your best estimate based on observable patterns.
- Keep your answer concise (1–3 sentences).
- Output only your answer — no preamble, no explanation, no "Based on the logs..." prefix.
"""

MODEL_ALIASES = {
    "sonnet": "claude-sonnet-4-20250514",
    "opus": "claude-opus-4-20250514",
}

ALL_MODELS = ["claude-sonnet-4-20250514", "claude-opus-4-20250514"]

# Disallow all MCP tools to ensure pure log-retrieval evaluation
DISALLOWED_TOOLS = [
    "mcp__*",  # wildcard may not work; list explicit prefixes below as fallback
    "mcp__claude_ai_Gmail__gmail_create_draft",
    "mcp__claude_ai_Gmail__gmail_get_profile",
    "mcp__claude_ai_Gmail__gmail_list_drafts",
    "mcp__claude_ai_Gmail__gmail_list_labels",
    "mcp__claude_ai_Gmail__gmail_read_message",
    "mcp__claude_ai_Gmail__gmail_read_thread",
    "mcp__claude_ai_Gmail__gmail_search_messages",
    "mcp__claude_ai_Google_Calendar__gcal_create_event",
    "mcp__claude_ai_Google_Calendar__gcal_delete_event",
    "mcp__claude_ai_Google_Calendar__gcal_find_meeting_times",
    "mcp__claude_ai_Google_Calendar__gcal_find_my_free_time",
    "mcp__claude_ai_Google_Calendar__gcal_get_event",
    "mcp__claude_ai_Google_Calendar__gcal_list_calendars",
    "mcp__claude_ai_Google_Calendar__gcal_list_events",
    "mcp__claude_ai_Google_Calendar__gcal_respond_to_event",
    "mcp__claude_ai_Google_Calendar__gcal_update_event",
    "mcp__leann-server__leann_list",
    "mcp__leann-server__leann_search",
]


def resolve_model(model_str: str) -> List[str]:
    if model_str == "all":
        return ALL_MODELS
    return [MODEL_ALIASES.get(model_str, model_str)]


async def ask_cc(question: str, tmp_dir: str, model: str, max_turns: int = 30) -> tuple[str, dict, list]:
    """Ask Claude Code a question with tool access to tmp_dir.

    Returns (answer_text, usage_info, trace).
    trace is a list of dicts recording each step:
      {"role": "assistant", "type": "text", "content": "..."}
      {"role": "assistant", "type": "tool_use", "tool": "Grep", "input": {...}}
      {"role": "tool", "tool_use_id": "...", "content": "...(truncated)"}
    """
    options = ClaudeCodeOptions(
        cwd=tmp_dir,
        system_prompt=SYSTEM_PROMPT,
        permission_mode="bypassPermissions",
        max_turns=max_turns,
        model=model,
        disallowed_tools=DISALLOWED_TOOLS,
    )

    final_text = ""
    usage_info = {}
    trace = []

    async for msg in query(prompt=f"Question: {question}", options=options):
        if isinstance(msg, AssistantMessage):
            for block in msg.content:
                if isinstance(block, TextBlock):
                    final_text = block.text
                    trace.append({"role": "assistant", "type": "text", "content": block.text})
                elif isinstance(block, ToolUseBlock):
                    trace.append({
                        "role": "assistant",
                        "type": "tool_use",
                        "tool": block.name,
                        "tool_use_id": block.id,
                        "input": block.input,
                    })
                elif isinstance(block, ToolResultBlock):
                    # Tool results inside assistant message
                    content = block.content or ""
                    if isinstance(content, str) and len(content) > 500:
                        content = content[:500] + f"...(truncated, {len(block.content)} chars)"
                    trace.append({
                        "role": "tool",
                        "tool_use_id": block.tool_use_id,
                        "content": content,
                        "is_error": block.is_error,
                    })
        elif isinstance(msg, UserMessage):
            # UserMessage contains tool results
            for block in msg.content:
                if isinstance(block, ToolResultBlock):
                    content = block.content or ""
                    if isinstance(content, str) and len(content) > 500:
                        content = content[:500] + f"...(truncated, {len(block.content)} chars)"
                    trace.append({
                        "role": "tool",
                        "tool_use_id": block.tool_use_id,
                        "content": content,
                        "is_error": block.is_error,
                    })
        elif isinstance(msg, ResultMessage):
            final_text = msg.result or final_text
            usage_info = msg.usage or {}

    return final_text, usage_info, trace


def setup_temp_dir(user_id: str, app_logs: list) -> str:
    """Create isolated temp directory with only app_log_large.json."""
    tmp_dir = tempfile.mkdtemp(prefix=f"cc_eval_{user_id}_")
    log_path = os.path.join(tmp_dir, "app_log_large.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(app_logs, f, ensure_ascii=False)
    return tmp_dir


async def evaluate_model(
    model: str,
    judge_model: str,
    max_turns: int,
    max_concurrent: int = 1,
    dry_run: bool = False,
):
    """Run full evaluation for one model on test users 007-010."""
    task_list = get_task_list(status="test", eval_n_samples=4)
    model_short = model.replace("claude-", "").replace("-20250514", "")
    _init_file_logger(model_short, dry_run)

    log.info(f"\n{'='*60}")
    log.info(f"Model: {model}")
    log.info(f"Users: {[t.split('/')[-1] for t in task_list]}")
    log.info(f"Max concurrent: {max_concurrent}")
    log.info(f"{'='*60}")

    all_results = []
    per_user = {}
    total_usage = {}
    _usage_lock = asyncio.Lock()

    semaphore = asyncio.Semaphore(max_concurrent)
    completed = 0
    total_qa = 0

    judge = Judge(model=judge_model)

    async def run_single_qa(user_short, qa, idx, n_total, tmp_dir):
        nonlocal completed
        async with semaphore:
            t0 = time.time()

            try:
                answer, usage, trace = await ask_cc(
                    qa["query"], tmp_dir, model, max_turns
                )
            except Exception as e:
                log.warning(f"  [User {user_short} Q{idx+1}] CC error: {e}")
                answer = ""
                usage = {}
                trace = [{"error": str(e)}]

            score, reason = await judge.score(
                qa["query"], answer, qa.get("reference", "")
            )

            elapsed = time.time() - t0

            # Accumulate token usage (thread-safe)
            async with _usage_lock:
                for k, v in usage.items():
                    if isinstance(v, (int, float)):
                        total_usage[k] = total_usage.get(k, 0) + v
                completed += 1

            tool_calls = [t for t in trace if t.get("type") == "tool_use"]
            n_turns = len(tool_calls)

            log.info(f"  [{completed}/{total_qa}] User {user_short} Q{idx+1} | score={score}/10 | {n_turns} tools | {elapsed:.1f}s")

            # Detailed debug log for file only
            log.debug(f"--- QA Detail: User {user_short} Q{idx+1} ---")
            log.debug(f"  Query: {qa['query']}")
            log.debug(f"  Reference: {qa.get('reference', '')}")
            log.debug(f"  Predicted: {answer}")
            log.debug(f"  Score: {score}/10 | Reason: {reason}")
            log.debug(f"  Tools: {[t['tool'] for t in tool_calls]}")
            for step in trace:
                if step.get("type") == "tool_use":
                    inp = json.dumps(step["input"], ensure_ascii=False)
                    if len(inp) > 300:
                        inp = inp[:300] + "..."
                    log.debug(f"    -> {step['tool']}({inp})")
                elif step.get("role") == "tool":
                    content = step.get("content", "")
                    if isinstance(content, str) and len(content) > 200:
                        content = content[:200] + "..."
                    log.debug(f"    <- result: {content}")
            log.debug(f"---")

            return {
                "user_id": user_short,
                "query": qa["query"],
                "predicted": answer,
                "reference": qa.get("reference", ""),
                "score": score,
                "judge_reason": reason,
                "elapsed_s": round(elapsed, 1),
                "n_tool_calls": n_turns,
                "tools_used": [t["tool"] for t in tool_calls],
                "trace": trace,
            }

    # Build all tasks across all users
    all_tasks = []  # (user_short, qa, idx, n_total, tmp_dir)
    temp_dirs = []

    for user_dir in task_list:
        user_id = user_dir.split("/")[-1]
        user_short = user_id.split("_")[0]

        app_logs, user_profile, qa_pairs = load_user_data(user_dir, eval_n_qa=None)
        if dry_run:
            qa_pairs = qa_pairs[:1]

        tmp_dir = setup_temp_dir(user_short, app_logs)
        temp_dirs.append(tmp_dir)

        for idx, qa in enumerate(qa_pairs):
            all_tasks.append((user_short, qa, idx, len(qa_pairs), tmp_dir))

        if dry_run:
            break

    total_qa = len(all_tasks)
    log.info(f"\nTotal QA pairs: {total_qa}")

    try:
        # Launch all QA tasks concurrently (semaphore controls parallelism)
        results = await asyncio.gather(*[
            run_single_qa(user_short, qa, idx, n_total, tmp_dir)
            for user_short, qa, idx, n_total, tmp_dir in all_tasks
        ])

        all_results = list(results)
    finally:
        for tmp_dir in temp_dirs:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    # Compute per-user stats
    from collections import defaultdict
    user_scores_map = defaultdict(list)
    for r in all_results:
        user_scores_map[r["user_id"]].append(r["score"])

    for uid, scores in user_scores_map.items():
        avg = sum(scores) / len(scores)
        per_user[uid] = {"avg_score": round(avg, 3), "n_qa": len(scores)}
        log.info(f"  User {uid} avg: {avg:.2f}/10 ({len(scores)} QA)")

    # Compute overall stats
    all_scores = [r["score"] for r in all_results]
    overall_avg = float(np.mean(all_scores)) if all_scores else 0.0
    overall_se = float(np.std(all_scores, ddof=1) / np.sqrt(len(all_scores))) if len(all_scores) > 1 else 0.0

    # Per-user reward for SE calculation (matching meta-learning's approach)
    user_rewards = [v["avg_score"] for v in per_user.values()]
    user_se = float(np.std(user_rewards, ddof=1) / np.sqrt(len(user_rewards))) if len(user_rewards) > 1 else 0.0

    output = {
        "model": model,
        "timestamp": datetime.now().isoformat(),
        "benchmark_eval_score": {
            "benchmark_overall_eval_score": round(overall_avg, 3),
            "benchmark_overall_eval_standard_deviation": round(user_se, 3),
        },
        "per_user": per_user,
        "total_qa": len(all_results),
        "token_usage": total_usage,
        "examples": all_results,
    }

    # Save results
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "_dryrun" if dry_run else ""
    out_path = results_dir / f"cc_{model_short}_{ts}{suffix}.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    log.info(f"\n{'='*60}")
    log.info(f"Results saved: {out_path}")
    log.info(f"Overall: {overall_avg:.3f} ± {user_se:.3f} (user-level SE)")
    log.info(f"Per-user: {per_user}")
    log.info(f"{'='*60}")

    return output


async def main():
    parser = argparse.ArgumentParser(description="Claude Code baseline eval on DynamicMem")
    parser.add_argument("--model", default="sonnet",
                        help="Model: claude-sonnet-4-20250514, claude-opus-4-20250514, sonnet, opus, or all")
    parser.add_argument("--judge_model", default="gpt-5-mini",
                        help="Judge model (default: gpt-5-mini)")
    parser.add_argument("--max_turns", type=int, default=30,
                        help="Max tool-use turns per QA question")
    parser.add_argument("--max_concurrent", type=int, default=5,
                        help="Max concurrent QA evaluations (default: 5)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Run single QA for quick verification")
    args = parser.parse_args()

    models = resolve_model(args.model)

    for model in models:
        await evaluate_model(
            model=model,
            judge_model=args.judge_model,
            max_turns=args.max_turns,
            max_concurrent=args.max_concurrent,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    asyncio.run(main())
