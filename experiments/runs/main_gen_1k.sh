#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL="${BASE_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"
WANDB_PROJECT="${WANDB_PROJECT:-straw}"
RUN_TAG="${RUN_TAG:-deadline_1k100}"

TRAIN_SAMPLES="${TRAIN_SAMPLES:-1000}"
EVAL_SAMPLES="${EVAL_SAMPLES:-100}"
TEST_SAMPLES="${TEST_SAMPLES:-100}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
CODE_MAX_NEW_TOKENS="${CODE_MAX_NEW_TOKENS:-256}"
RESULTS_DIR="${RESULTS_DIR:-experiments/results/${RUN_TAG}}"
BA_PATH="${BA_PATH:-${RESULTS_DIR}/ba_straw_eval_1k.pt}"
STRAW_OUTPUT_DIR="${STRAW_OUTPUT_DIR:-models/checkpoints/straw/${RUN_TAG}}"

mkdir -p "${RESULTS_DIR}"

echo "[1/11] Building processed datasets (train=${TRAIN_SAMPLES}, eval=${EVAL_SAMPLES}, test=${TEST_SAMPLES})"

python3 -m data.build_generative_datasets \
  --max-train-samples "${TRAIN_SAMPLES}" \
  --validation-size "${EVAL_SAMPLES}" \
  --test-size "${TEST_SAMPLES}"

echo "[2/11] Training domain-specific LoRA adapters (one adapter per dataset)"
python3 -m experiments.train.train_lora_domain_gen \
  --config experiments/configs/lora_config_a.yaml \
  --data-root data/processed_gen \
  --max-train-samples "${TRAIN_SAMPLES}" \
  --max-eval-samples "${EVAL_SAMPLES}" \
  --use-wandb \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-tags "${RUN_TAG}" "deadline" "train" "domain_lora" "1k100"

echo "[3/11] Training one mixed LoRA adapter (shared across datasets)"
python3 -m experiments.train.train_lora_mixed_gen \
  --config experiments/configs/lora_config_a.yaml \
  --data-root data/processed_gen \
  --max-train-samples-per-dataset "${TRAIN_SAMPLES}" \
  --max-eval-samples-per-dataset "${EVAL_SAMPLES}" \
  --use-wandb \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-tags "${RUN_TAG}" "deadline" "train" "mixed_lora" "1k100"

echo "[4/11] Training STRAW hypernetwork (dynamic adapters)"
python3 -m experiments.train.train_straw \
  --config experiments/configs/lora_config_b_straw.yaml \
  --data-root data/processed_gen \
  --max-train-samples-per-dataset "${TRAIN_SAMPLES}" \
  --max-eval-samples-per-dataset "${EVAL_SAMPLES}" \
  --output-dir "${STRAW_OUTPUT_DIR}" \
  --use-wandb \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-run-name "straw_gen_1k100" \
  --wandb-tags "${RUN_TAG}" "deadline" "train" "straw" "1k100"

echo "[5/11] Evaluating base model"
python3 -m experiments.eval.run_eval_gen \
  --base-model "${BASE_MODEL}" \
  --data-root data/processed_gen \
  --split test \
  --max-samples "${TEST_SAMPLES}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --output "${RESULTS_DIR}/eval_results_gen_1k_base.json" \
  --use-wandb \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-run-name "eval_base_1k100" \
  --wandb-tags "${RUN_TAG}" "deadline" "eval" "base" "1k100"

echo "[6/11] Evaluating mixed LoRA adapter"
python3 -m experiments.eval.run_eval_gen \
  --base-model "${BASE_MODEL}" \
  --adapter models/checkpoints/lora_mixed_gen/adapter \
  --data-root data/processed_gen \
  --split test \
  --max-samples "${TEST_SAMPLES}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --output "${RESULTS_DIR}/eval_results_gen_1k_mixed.json" \
  --use-wandb \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-run-name "eval_mixed_1k100" \
  --wandb-tags "${RUN_TAG}" "deadline" "eval" "mixed_lora" "1k100"

echo "[7/11] Evaluating domain LoRA adapter on SAMSum"
python3 -m experiments.eval.run_eval_gen \
  --base-model "${BASE_MODEL}" \
  --adapter models/checkpoints/lora_domain_gen/samsum_gen/adapter \
  --datasets samsum_gen \
  --data-root data/processed_gen \
  --split test \
  --max-samples "${TEST_SAMPLES}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --output "${RESULTS_DIR}/eval_results_gen_1k_domain_samsum.json" \
  --use-wandb \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-run-name "eval_domain_samsum_1k100" \
  --wandb-tags "${RUN_TAG}" "deadline" "eval" "domain_lora" "samsum_gen" "1k100"

echo "[8/11] Evaluating domain LoRA adapter on Dolly"
python3 -m experiments.eval.run_eval_gen \
  --base-model "${BASE_MODEL}" \
  --adapter models/checkpoints/lora_domain_gen/dolly_gen/adapter \
  --datasets dolly_gen \
  --data-root data/processed_gen \
  --split test \
  --max-samples "${TEST_SAMPLES}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --output "${RESULTS_DIR}/eval_results_gen_1k_domain_dolly.json" \
  --use-wandb \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-run-name "eval_domain_dolly_1k100" \
  --wandb-tags "${RUN_TAG}" "deadline" "eval" "domain_lora" "dolly_gen" "1k100"

echo "[9/11] Evaluating domain LoRA adapter on CodeAlpaca (longer generation limit)"
python3 -m experiments.eval.run_eval_gen \
  --base-model "${BASE_MODEL}" \
  --adapter models/checkpoints/lora_domain_gen/codealpaca_gen/adapter \
  --datasets codealpaca_gen \
  --data-root data/processed_gen \
  --split test \
  --max-samples "${TEST_SAMPLES}" \
  --max-new-tokens "${CODE_MAX_NEW_TOKENS}" \
  --output "${RESULTS_DIR}/eval_results_gen_1k_domain_codealpaca.json" \
  --use-wandb \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-run-name "eval_domain_codealpaca_1k100" \
  --wandb-tags "${RUN_TAG}" "deadline" "eval" "domain_lora" "codealpaca_gen" "1k100"

echo "[10/11] Evaluating STRAW + saving BA payload"
python3 -m experiments.eval.run_eval_straw_gen \
  --base-model "${BASE_MODEL}" \
  --straw-config experiments/configs/lora_config_b_straw.yaml \
  --hypernet-ckpt "${STRAW_OUTPUT_DIR}/best_hypernet.pt" \
  --data-root data/processed_gen \
  --split test \
  --limit "${TEST_SAMPLES}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --output-path "${RESULTS_DIR}/straw_eval_gen_1k.json" \
  --save-ba-path "${BA_PATH}" \
  --use-wandb \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-run-name "eval_straw_1k100" \
  --wandb-tags "${RUN_TAG}" "deadline" "eval" "straw" "1k100"

echo "[11/11] Generating BA visualizations (GIF heatmaps + comparison + fancy 3D)"
python3 -m experiments.report.visualize_ba_heatmaps \
  --inputs "${BA_PATH}" \
  --layers 0 8 16 24 31 \
  --output-dir "${RESULTS_DIR}/ba_viz_1k"

python3 -m experiments.report.compare_ba_domains \
  --input "${BA_PATH}" \
  --layers 0 8 16 24 30 \
  --output-dir "${RESULTS_DIR}/ba_compare_1k"

python3 -m experiments.report.ba_fancy_3d \
  --input "${BA_PATH}" \
  --layers 0 16 30 \
  --output-dir "${RESULTS_DIR}/ba_fancy_3d_1k"

echo "Done. All result files and visualizations are under: ${RESULTS_DIR}"
