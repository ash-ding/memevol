#!/bin/bash

# ===========================================================================
# alma baseline — Evaluation examples
# Run a saved memo structure on the held-out test split (users 007-010).
# Invoke from the project root.
# ===========================================================================

# --- Evaluate a specific memo on held-out test users ---
# <SHA> must exist under baselines/alma/memo_archive/dynamicmem/memo_structure_<SHA>.py
# Results land in baselines/alma/results/dynamicmem/<SHA>_test_eval/
python baselines/alma/run_main.py \
    --meta_model gpt-5 \
    --execution_model gpt-5-mini \
    --judge_model gpt-5-mini \
    --status test \
    --memo_SHA <SHA> \
    --update_type all_at_once

# --- Evaluate the no-memory baseline on held-out users ---
# Establishes the floor reward; useful for comparing any learned memo.
python baselines/alma/run_main.py \
    --meta_model gpt-5 \
    --execution_model gpt-5-mini \
    --judge_model gpt-5-mini \
    --status test \
    --memo_SHA no_mem \
    --update_type all_at_once

# --- Evaluate with chunked update (same memo, different Phase 1 batching) ---
# Handy for ablations without retraining the memo.
python baselines/alma/run_main.py \
    --meta_model gpt-5 \
    --execution_model gpt-5-mini \
    --judge_model gpt-5-mini \
    --status test \
    --memo_SHA <SHA> \
    --update_type chunked \
    --n_chunks 10

# --- Evaluate with sequential update (most expensive, most fine-grained) ---
python baselines/alma/run_main.py \
    --meta_model gpt-5 \
    --execution_model gpt-5-mini \
    --judge_model gpt-5-mini \
    --status test \
    --memo_SHA <SHA> \
    --update_type sequential

# --- Quick smoke eval (1 user, 3 QA) to verify plumbing ---
python baselines/alma/run_main.py \
    --meta_model gpt-5 \
    --execution_model gpt-5-mini \
    --judge_model gpt-5-mini \
    --status test \
    --memo_SHA no_mem \
    --eval_n_qa 3 \
    --max_sample_concurrent 1
