#!/bin/bash

# ===========================================================================
# memevol — Training examples (meta-learning memory designs for DynamicMem)
# ===========================================================================

# --- Full training run (default settings) ---
# 6 train users, 10 meta-learning steps, 20 QA per user per round
python run_main.py \
    --meta_model gpt-5 \
    --execution_model gpt-5-mini \
    --judge_model gpt-5-mini \
    --status train \
    --train_size 6 \
    --qa_sample_size 20 \
    --update_type all_at_once \
    --steps 10 \
    --max_concurrent 5 \
    --n_score_bins 3 \
    --samples_per_bin 3

# --- Quick smoke test (minimal workload, verify end-to-end) ---
# 2 users, 2 steps, 10 QA per user
python run_main.py \
    --meta_model gpt-5 \
    --execution_model gpt-5-mini \
    --judge_model gpt-5-mini \
    --status train \
    --train_size 2 \
    --qa_sample_size 10 \
    --update_type all_at_once \
    --steps 2 \
    --max_concurrent 3

# --- Chunked update mode ---
# Feed app_logs in 10 chunks instead of all at once
python run_main.py \
    --meta_model gpt-5 \
    --execution_model gpt-5-mini \
    --judge_model gpt-5-mini \
    --status train \
    --train_size 4 \
    --qa_sample_size 20 \
    --update_type chunked \
    --n_chunks 10 \
    --steps 10

# --- Sequential update mode ---
# Feed app_logs one by one (most expensive, most fine-grained)
python run_main.py \
    --meta_model gpt-5 \
    --execution_model gpt-5-mini \
    --judge_model gpt-5-mini \
    --status train \
    --train_size 2 \
    --qa_sample_size 10 \
    --update_type sequential \
    --steps 5

# --- Resume from checkpoint ---
# Continue a previous training run from a saved checkpoint in logs/
python run_main.py \
    --meta_model gpt-5 \
    --execution_model gpt-5-mini \
    --judge_model gpt-5-mini \
    --status train \
    --train_size 6 \
    --qa_sample_size 20 \
    --update_type all_at_once \
    --steps 10 \
    --history_ckpt_path check_dynamicmem_all_at_once_10_20260324_120000.json
