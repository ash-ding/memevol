#!/bin/bash

# ===========================================================================
# alma baseline — Search examples (meta-learning memory designs)
# Run from the project root. One benchmark per run via --dataset.
#
# Evaluation SIZES live in the --config YAML only — there is NO sizing CLI flag
# (the old --stages was removed). progressive: true scores each candidate
# through the SHARED staged gauntlet (common.evaluate: stage1 -> stage2 ->
# stage3 with promotion thresholds), sized by the config's `stages` block (or
# the family DEFAULT_STAGES). progressive: false does ONE pass sized by the
# config's REQUIRED `single_stage` block. Edit
# baselines/evolve/alma/config.example.yaml (or a copy) to change sizes.
# ===========================================================================

# --- Full progressive search (gauntlet; stages from the config) ---
# stage1 -> stage2 -> stage3 gauntlet per candidate, 10 meta-learning steps.
uv run --project baselines/evolve/alma python baselines/evolve/alma/run.py \
    --config baselines/evolve/alma/config.example.yaml \
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
uv run --project baselines/evolve/alma python baselines/evolve/alma/run.py \
    --config baselines/evolve/alma/config.example.yaml \
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

# --- Non-progressive search (single pass per candidate, sized by single_stage) ---
# --no-progressive skips the gauntlet: ONE pass, no thresholds. The pass is
# sized by the config's REQUIRED `single_stage` block (a null field = whole
# split for that dimension). For a quick smoke run, point --config at a config
# whose `single_stage` uses tiny sizes.
uv run --project baselines/evolve/alma python baselines/evolve/alma/run.py \
    --config baselines/evolve/alma/config.example.yaml \
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
uv run --project baselines/evolve/alma python baselines/evolve/alma/run.py \
    --config baselines/evolve/alma/config.example.yaml \
    --meta_model gpt-5 \
    --execution_model gpt-5-mini \
    --judge_model gpt-5-mini \
    --status search \
    --dataset dynamicmem \
    --progressive \
    --steps 10 \
    --history_ckpt_path check_dynamicmem_10_20260324_120000.json
