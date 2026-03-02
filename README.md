# STRAW_MHACK

Dynamic hypernetwork-generated LoRA adapters ("STRAW") for multi-domain generative adaptation on:
- `samsum_gen` (dialogue summarization)
- `dolly_gen` (instruction following)
- `codealpaca_gen` (code generation)

This repo compares:
- Base model (no adaptation)
- Domain LoRA (one adapter per dataset)
- Mixed LoRA (single shared adapter)
- STRAW (dynamic per-input adapter generation)

## Quick Start (1k setup)

Run the end-to-end 1k pipeline:

```bash
cd /home/shadeform/STRAW_MHACK
bash experiments/runs/main_gen_1k.sh
```

Environment overrides:

```bash
RUN_TAG=run_a TRAIN_SAMPLES=1000 EVAL_SAMPLES=100 TEST_SAMPLES=100 MAX_NEW_TOKENS=256 CODE_MAX_NEW_TOKENS=256 bash experiments/runs/main_gen_1k.sh
```

All run outputs are grouped under:

- `experiments/results/${RUN_TAG}`
- STRAW checkpoint: `models/checkpoints/straw/${RUN_TAG}`

## What `main_gen_1k.sh` runs

1. Build datasets (`data.build_generative_datasets`)
2. Train domain LoRA (`experiments.train.train_lora_domain_gen`)
3. Train mixed LoRA (`experiments.train.train_lora_mixed_gen`)
4. Train STRAW (`experiments.train.train_straw`)
5. Eval base (`experiments.eval.run_eval_gen`)
6. Eval mixed LoRA (`experiments.eval.run_eval_gen`)
7. Eval domain LoRA on `samsum_gen`
8. Eval domain LoRA on `dolly_gen`
9. Eval domain LoRA on `codealpaca_gen`
10. Eval STRAW + save BA payload (`experiments.eval.run_eval_straw_gen`)
11. Generate BA visualizations:
   - heatmap PNG/GIF (`experiments.report.visualize_ba_heatmaps`)
   - domain comparison maps (`experiments.report.compare_ba_domains`)
   - minimalist 3D surfaces (`experiments.report.ba_fancy_3d`)

## W&B Logging

The scripts support:
- `--use-wandb`
- `--wandb-project`
- `--wandb-run-name`
- `--wandb-tags`

Training runs primarily log loss/scheduler stats.
Evaluation runs log per-dataset scores and macro average:
- `eval/<dataset>/score`
- `eval/macro_avg_score`

## Current STRAW Defaults

From `experiments/configs/lora_config_b_straw.yaml`:
- `model_type: cnn`
- `straw_rank: 16`
- `lora_alpha: 16`
- `layer_stride: 1`
- `learning_rate: 3.0e-5`
- `num_train_epochs: 5`
- `warmup_ratio: 0.05`

From `experiments/configs/lora_config_a.yaml` (LoRA baselines):
- `num_train_epochs: 5`
- `warmup_ratio: 0.05`
- `lr_scheduler_type: cosine`

## Latest Run Snapshot (`straw_1k_v2`)

Results are saved under `experiments/results/straw_1k_v2`.

| Model | SAMSum | Dolly | CodeAlpaca | Macro Avg |
|---|---:|---:|---:|---:|
| Base | 0.1974 | 0.2446 | 0.0000 | 0.1474 |
| Mixed LoRA | 0.3931 | 0.3409 | 0.0500 | 0.2613 |
| STRAW | 0.4007 | 0.3412 | 0.0500 | 0.2640 |

Domain LoRA in-domain references:
- SAMSum adapter: `0.4274`
- Dolly adapter: `0.3619`
- CodeAlpaca adapter: `0.1000`

BA visualization outputs:
- `experiments/results/straw_1k_v2/ba_viz_1k`
- `experiments/results/straw_1k_v2/ba_compare_1k`
- `experiments/results/straw_1k_v2/ba_fancy_3d_1k`

Presentation-ready report:
- `experiments/results/straw_1k_v2/NOTION_REPORT.md`

## BA Visualization Commands

Generate BA payload (50 samples per dataset):

```bash
python3 -m experiments.eval.run_eval_straw_gen \
  --base-model mistralai/Mistral-7B-Instruct-v0.3 \
  --straw-config experiments/configs/lora_config_b_straw.yaml \
  --hypernet-ckpt models/checkpoints/straw/gen_main_1k/best_hypernet.pt \
  --data-root data/processed_gen \
  --datasets samsum_gen dolly_gen codealpaca_gen \
  --split test \
  --limit 50 \
  --max-new-tokens 128 \
  --output-path experiments/results/straw_eval_gen_50_each.json \
  --save-ba-path experiments/results/ba_straw_eval_50_each.pt
```

Heatmap/GIF visualization:

```bash
python3 -m experiments.report.visualize_ba_heatmaps \
  --inputs experiments/results/ba_straw_eval_50_each.pt \
  --layers 0 8 16 24 30 \
  --output-dir experiments/results/ba_viz_50_each \
  --fps 1
```

Domain comparison maps:

```bash
python3 -m experiments.report.compare_ba_domains \
  --input experiments/results/ba_straw_eval_50_each.pt \
  --layers 0 8 16 24 30 \
  --output-dir experiments/results/ba_compare_50_each
```

Fancy 3D surfaces:

```bash
python3 -m experiments.report.ba_fancy_3d \
  --input experiments/results/ba_straw_eval_50_each.pt \
  --layers 0 16 30 \
  --output-dir experiments/results/ba_fancy_3d_50_each
```

## Notes

- Use `python3 -m ...` from repo root to avoid import path issues.
