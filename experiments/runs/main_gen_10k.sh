#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL="${BASE_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"

python3 -m data.build_generative_datasets \
  --max-train-samples 10000 \
  --validation-size 1000 \
  --test-size 1000

python3 -m experiments.train.train_lora_domain_gen \
  --config experiments/configs/lora_config_a.yaml \
  --data-root data/processed_gen \
  --max-train-samples 10000 \
  --max-eval-samples 1000

python3 -m experiments.train.train_lora_mixed_gen \
  --config experiments/configs/lora_config_a.yaml \
  --data-root data/processed_gen \
  --max-train-samples-per-dataset 10000 \
  --max-eval-samples-per-dataset 1000

python3 -m experiments.train.train_straw \
  --config experiments/configs/lora_config_b_straw.yaml \
  --data-root data/processed_gen \
  --max-train-samples-per-dataset 10000 \
  --max-eval-samples-per-dataset 1000 \
  --output-dir models/checkpoints/straw/gen_main_10k

python3 -m experiments.eval.run_eval_gen \
  --base-model "$BASE_MODEL" \
  --data-root data/processed_gen \
  --split test \
  --max-samples 1000 \
  --max-new-tokens 128 \
  --output experiments/results/eval_results_gen_main_base.json

python3 -m experiments.eval.run_eval_gen \
  --base-model "$BASE_MODEL" \
  --adapter models/checkpoints/lora_mixed_gen/adapter \
  --data-root data/processed_gen \
  --split test \
  --max-samples 1000 \
  --max-new-tokens 128 \
  --output experiments/results/eval_results_gen_main_mixed.json

python3 -m experiments.eval.run_eval_straw_gen \
  --base-model "$BASE_MODEL" \
  --straw-config experiments/configs/lora_config_b_straw.yaml \
  --hypernet-ckpt models/checkpoints/straw/gen_main_10k/best_hypernet.pt \
  --data-root data/processed_gen \
  --split test \
  --limit 1000 \
  --max-new-tokens 128 \
  --output-path experiments/results/straw_eval_gen_main.json
