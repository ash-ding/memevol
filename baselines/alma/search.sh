#!/bin/bash

# ===========================================================================
# alma baseline — Search examples (meta-learning memory designs for DynamicMem)
# Run from the project root.
# ===========================================================================

# --- Full search run (default settings) ---
# 6 search users, 10 meta-learning steps, 20 QA per user per round
python baselines/alma/run_main.py \
    --meta_model gpt-5 \
    --execution_model gpt-5-mini \
    --judge_model gpt-5-mini \
    --status search \
    --eval_n_samples 6 \
    --eval_n_qa 20 \
    --steps 10 \
    --max_sample_concurrent 5 \
    --n_score_bins 3 \
    --samples_per_bin 3

# --- Quick smoke test (minimal workload, verify end-to-end) ---
# 2 users, 2 steps, 10 QA per user
python baselines/alma/run_main.py \
    --meta_model gpt-5 \
    --execution_model gpt-5-mini \
    --judge_model gpt-5-mini \
    --status search \
    --eval_n_samples 2 \
    --eval_n_qa 10 \
    --steps 2 \
    --max_sample_concurrent 3

# --- Resume from checkpoint ---
# Checkpoint files live under baselines/alma/logs/ after the first step.
python baselines/alma/run_main.py \
    --meta_model gpt-5 \
    --execution_model gpt-5-mini \
    --judge_model gpt-5-mini \
    --status search \
    --eval_n_samples 6 \
    --eval_n_qa 20 \
    --steps 10 \
    --history_ckpt_path check_dynamicmem_10_20260324_120000.json
