#!/bin/bash

# ===========================================================================
# alma baseline — Evaluation examples
# Run a saved memo structure on the held-out test split (users 007-010).
# Invoke from the project root.
#
# The flat --eval_n_qa knob was removed; sizes come from the shared `stages`
# schema. --status test evaluates the memo through the gauntlet by default
# (--progressive); --no-progressive does a single terminal-size pass.
# ===========================================================================

# --- Evaluate a specific memo on held-out test users (progressive gauntlet) ---
# <SHA> must exist under baselines/evolve/alma/memo_archive/dynamicmem/memo_structure_<SHA>.py
# Results land in baselines/evolve/alma/results/dynamicmem/<SHA>_test_eval/
python baselines/evolve/alma/run_main.py \
    --meta_model gpt-5 \
    --execution_model gpt-5-mini \
    --judge_model gpt-5-mini \
    --status test \
    --memo_SHA <SHA> \
    --progressive

# --- Evaluate the no-memory baseline on held-out users ---
# Establishes the floor reward; useful for comparing any learned memo.
python baselines/evolve/alma/run_main.py \
    --meta_model gpt-5 \
    --execution_model gpt-5-mini \
    --judge_model gpt-5-mini \
    --status test \
    --memo_SHA no_mem \
    --progressive

# --- Quick smoke eval (tiny sizes via --stages) to verify plumbing ---
SMOKE_STAGES='{"sanity_check":{"n_users":1,"n_checkpoints":1,"n_task_a":1,"n_task_c":1},"stage1":{"n_users":1,"n_checkpoints":1,"n_task_a":1,"n_task_c":1,"threshold":0.0},"stage2":{"n_users":1,"n_checkpoints":1,"n_task_a":1,"n_task_c":1,"threshold":0.0},"stage3":{"n_users":1,"n_checkpoints":1,"n_task_a":1,"n_task_c":1}}'
python baselines/evolve/alma/run_main.py \
    --meta_model gpt-5 \
    --execution_model gpt-5-mini \
    --judge_model gpt-5-mini \
    --status test \
    --memo_SHA no_mem \
    --progressive \
    --stages "$SMOKE_STAGES" \
    --max_sample_concurrent 1
