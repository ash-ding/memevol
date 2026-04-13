#!/bin/bash

# ===========================================================================
# memevol — Test examples (evaluate learned memory designs on held-out users)
# ===========================================================================

# --- Evaluate a specific memory design on held-out test users (007-010) ---
# Uses all QA pairs (~178 per user), no sampling
python run_main.py \
    --meta_model gpt-5 \
    --execution_model gpt-5-mini \
    --judge_model gpt-5-mini \
    --status test \
    --memo_SHA 53cee295 \
    --update_type all_at_once

# --- Evaluate the no_mem baseline ---
python run_main.py \
    --meta_model gpt-5 \
    --execution_model gpt-5-mini \
    --judge_model gpt-5-mini \
    --status test \
    --memo_SHA no_mem \
    --update_type all_at_once

# --- Evaluate with chunked update mode ---
python run_main.py \
    --meta_model gpt-5 \
    --execution_model gpt-5-mini \
    --judge_model gpt-5-mini \
    --status test \
    --memo_SHA 53cee295 \
    --update_type chunked \
    --n_chunks 10

# --- Evaluate with a stronger judge model ---
python run_main.py \
    --meta_model gpt-4.1 \
    --execution_model gpt-4o-mini \
    --judge_model gpt-5 \
    --status test \
    --memo_SHA 53cee295 \
    --update_type all_at_once
