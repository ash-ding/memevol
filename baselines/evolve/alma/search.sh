#!/bin/bash

# ===========================================================================
# alma baseline — Search examples (meta-learning memory designs)
# Run from the project root. One benchmark per run via --dataset.
#
# Evaluation SIZES no longer live on flat --eval_n_samples/--eval_n_qa flags —
# they were removed. Each candidate is now scored through the SHARED staged
# gauntlet (common.staged_eval: stage1 -> stage2 -> stage3 with promotion
# thresholds), identical to forge. Sizes come from the family DEFAULT_STAGES;
# override with --stages '<json>'.
# ===========================================================================

# --- Full progressive search (default stages) ---
# stage1 -> stage2 -> stage3 gauntlet per candidate, 10 meta-learning steps.
python baselines/evolve/alma/run_main.py \
    --meta_model gpt-5 \
    --execution_model gpt-5-mini \
    --judge_model gpt-5-mini \
    --status search \
    --dataset dynamicmem \
    --progressive \
    --steps 10 \
    --max_sample_concurrent 5 \
    --n_score_bins 3 \
    --samples_per_bin 3

# --- Progressive + per-step random sampling ---
# Each search step draws a DIFFERENT deterministic task subset (seeded by
# --sampling_seed + step) — reduces overfitting to one fixed subset.
python baselines/evolve/alma/run_main.py \
    --meta_model gpt-5 \
    --execution_model gpt-5-mini \
    --judge_model gpt-5-mini \
    --status search \
    --dataset locomo \
    --progressive \
    --random_sample \
    --sampling_seed 42 \
    --steps 10 \
    --max_sample_concurrent 5

# --- Quick smoke test (tiny sizes via --stages, verify end-to-end) ---
# A minimal 1-user/1-checkpoint gauntlet, 2 steps. thresholds 0.0 so nothing is
# eliminated early. (--stages must keep sizes non-decreasing across stage1..3.)
SMOKE_STAGES='{"sanity_check":{"n_users":1,"n_checkpoints":1,"n_task_a":1,"n_task_c":1},"stage1":{"n_users":1,"n_checkpoints":1,"n_task_a":1,"n_task_c":1,"threshold":0.0},"stage2":{"n_users":1,"n_checkpoints":1,"n_task_a":1,"n_task_c":1,"threshold":0.0},"stage3":{"n_users":1,"n_checkpoints":1,"n_task_a":1,"n_task_c":1}}'
python baselines/evolve/alma/run_main.py \
    --meta_model gpt-5 \
    --execution_model gpt-5-mini \
    --judge_model gpt-5-mini \
    --status search \
    --dataset dynamicmem \
    --progressive \
    --stages "$SMOKE_STAGES" \
    --steps 2 \
    --max_sample_concurrent 3

# --- Non-progressive search (single terminal-size pass per candidate) ---
# --no-progressive skips the gauntlet: one stage3-size pass, no thresholds.
python baselines/evolve/alma/run_main.py \
    --meta_model gpt-5 \
    --execution_model gpt-5-mini \
    --judge_model gpt-5-mini \
    --status search \
    --dataset dynamicmem \
    --no-progressive \
    --steps 10 \
    --max_sample_concurrent 5

# --- Resume from checkpoint ---
# Checkpoint files live under baselines/evolve/alma/logs/ after the first step.
python baselines/evolve/alma/run_main.py \
    --meta_model gpt-5 \
    --execution_model gpt-5-mini \
    --judge_model gpt-5-mini \
    --status search \
    --dataset dynamicmem \
    --progressive \
    --steps 10 \
    --history_ckpt_path check_dynamicmem_10_20260324_120000.json
