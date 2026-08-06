#!/bin/bash

# ===========================================================================
# alma baseline — Evaluation examples
# Run a saved memo structure on the held-out test split (users 007-010).
# Invoke from the project root.
#
# Evaluation SIZES live in the --config YAML only (the flat --eval_n_qa and the
# --stages CLI flags were removed). --status test evaluates the memo through the
# gauntlet by default (--progressive), sized by the config's `stages` block;
# --no-progressive does ONE pass sized by the config's REQUIRED `single_stage`
# block. Edit baselines/evolve/alma/config.example.yaml (or a copy) to change
# sizes.
# ===========================================================================

# --- Evaluate a specific memo on held-out test users (progressive gauntlet) ---
# <SHA> must exist under baselines/evolve/alma/memo_archive/dynamicmem/memo_structure_<SHA>.py
# Results land in baselines/evolve/alma/results/dynamicmem/<SHA>_test_eval/
uv run --project baselines/evolve/alma python baselines/evolve/alma/run.py \
    --config baselines/evolve/alma/config.example.yaml \
    --meta_model gpt-5 \
    --execution_model gpt-5-mini \
    --judge_model gpt-5-mini \
    --status test \
    --memo_SHA <SHA> \
    --progressive

# --- Evaluate the no-memory baseline on held-out users ---
# Establishes the floor reward; useful for comparing any learned memo.
uv run --project baselines/evolve/alma python baselines/evolve/alma/run.py \
    --config baselines/evolve/alma/config.example.yaml \
    --meta_model gpt-5 \
    --execution_model gpt-5-mini \
    --judge_model gpt-5-mini \
    --status test \
    --memo_SHA no_mem \
    --progressive

# --- Single-pass eval (no gauntlet), sized by the config's single_stage ---
# --no-progressive runs ONE pass; sizes come from the config's REQUIRED
# `single_stage` block. For a quick plumbing check, point --config at a config
# whose `single_stage` uses tiny sizes.
uv run --project baselines/evolve/alma python baselines/evolve/alma/run.py \
    --config baselines/evolve/alma/config.example.yaml \
    --meta_model gpt-5 \
    --execution_model gpt-5-mini \
    --judge_model gpt-5-mini \
    --status test \
    --memo_SHA no_mem \
    --no-progressive \
    --max_sample_concurrent 1
