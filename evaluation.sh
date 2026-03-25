#!/bin/bash

# ===========================================================================
# memevol — Evaluation examples (test learned memory designs on held-out users)
# ===========================================================================

# --- Evaluate a specific memory design on held-out eval users (007-010) ---
# Uses all refined QA pairs (~94 per user), no sampling
python run_main.py \
    --meta_model gpt-5 \
    --execution_model gpt-5-mini \
    --judge_model gpt-5-mini \
    --status eval \
    --memo_SHA 53cee295 \
    --update_type all_at_once

# --- Evaluate the no_mem baseline ---
python run_main.py \
    --meta_model gpt-5 \
    --execution_model gpt-5-mini \
    --judge_model gpt-5-mini \
    --status eval \
    --memo_SHA no_mem \
    --update_type all_at_once

# --- Evaluate with chunked update mode ---
python run_main.py \
    --meta_model gpt-5 \
    --execution_model gpt-5-mini \
    --judge_model gpt-5-mini \
    --status eval \
    --memo_SHA 53cee295 \
    --update_type chunked \
    --n_chunks 10

# --- Evaluate with a stronger judge model ---
python run_main.py \
    --meta_model gpt-4.1 \
    --execution_model gpt-4o-mini \
    --judge_model gpt-5 \
    --status eval \
    --memo_SHA 53cee295 \
    --update_type all_at_once
