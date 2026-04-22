# Model Ablation Study

Systematic ablation study for the MACCP (Model-Agreement-Conditioned Conformal
Prediction) component triage system.  Tests whether replacing the DeBERTa-base
agree-model with stronger or structurally different models improves prediction
set efficiency.

## Data

Pre-computed predictions and parquet splits are in `data/{eclipse,mozilla}/`:

| File | Description | Source |
|------|-------------|--------|
| `deberta_{cal,test}_{probs,preds}.npy` | DeBERTa-v3-base softmax probs and argmax preds | `finetune_deberta.py` run on TamIA |
| `xgb_{cal,test}_{probs,preds}.npy` | XGBoost (TF-IDF + metadata) probs and preds | `run_maccp.py` |
| `{cal,test}_labels.npy` | Ground-truth integer labels | Shared across models |
| `label_mapping.json` | Component name -> integer index | Eclipse: 30 classes, Mozilla: 20 classes |
| `{train,cal,test}.parquet` | Text + metadata splits | Temporal split from full dataset |

Parquet columns:
- **Eclipse**: text (summary [SEP] description), label, component_mapped, creation_time, component, product, op_sys, severity, priority
- **Mozilla**: text (summary only), label, component_mapped, creation_ts, product, component

## Experiments

### Experiment 1: Hybrid MACCP (`experiment1_hybrid.py`)
CPU-only, runs in seconds.  Tests 4 agree/disagree model routing configurations
(DeBERTa/DeBERTa, XGBoost/XGBoost, DeBERTa/XGBoost, XGBoost/DeBERTa) at
alpha in {0.05, 0.10, 0.20} on both datasets.

### Experiment 2: LLM Zero-Shot (`experiment2_llm_zeroshot.py`)
Calls Fireworks API with DeepSeek-v3 and GLM-5 for zero-shot component
prediction.  8 concurrent workers, checkpoints every 200 examples.  Builds
approximate probability vectors (0.80 for predicted class, uniform rest) for
MACCP integration.  Requires `FIREWORKS_API_KEY` env var or `../../.fireworks_key`.

### Experiment 3: DeBERTa-v3-large (`experiment3_finetune_deberta_large.py`)
Fine-tunes microsoft/deberta-v3-large (304M params) using HuggingFace Trainer.
lr=1e-5, batch=16, grad_accum=2, 3 epochs, bf16 training, float32 model loading.
Saves probabilities as .npy for MACCP and runs full 4-config comparison.

### Experiment 4: Qwen2.5-32B-Instruct (`experiment4_finetune_qwen.py`)
LoRA fine-tuning (r=16, alpha=32) of a 32B parameter model with DeepSpeed ZeRO
Stage 2 and CPU offloading.  Chat-format training.  Extracts probabilities via
first-token logits (with multi-token fallback).  2 epochs, lr=2e-4, batch=2,
grad_accum=8.

### Experiment 5: Standalone Inference (`experiment5_inference_finetuned.py`)
Loads fine-tuned models from `models/{model}/{dataset}/` and runs inference
without re-training.  Useful for re-evaluation or when models were trained on
a different node.

### Experiment 6: Final Comparison (`experiment6_comparison.py`)
Aggregates results from all experiments into master comparison tables: accuracy,
agreement rates, and MACCP metrics (coverage, set size, singleton rate).

## Running

### Local (CPU experiments only)
```bash
cd model_ablation
bash scripts/run_all.sh
```

### TamIA HPC (all experiments)
```bash
cd ~/links/projects/aip-rnishat/shared/perf-regression-ci
bash model_ablation/scripts/run_all_tamia.sh
```

### Individual SLURM jobs
```bash
sbatch model_ablation/scripts/run_deberta_large.slurm  # 4h, gpubase_bynode_b1
sbatch model_ablation/scripts/run_qwen.slurm           # 12h, gpubase_bynode_b2
```

## Directory Structure

```
model_ablation/
  data/
    eclipse/          # 30-class Eclipse component assignment
    mozilla/          # 20-class Mozilla Firefox component assignment
  scripts/
    utils.py                            # Shared RAPS/MACCP functions
    cost_tracker.py                     # LLM API cost tracking
    experiment1_hybrid.py               # Hybrid MACCP (4 configs)
    experiment2_llm_zeroshot.py         # LLM zero-shot via Fireworks
    experiment3_finetune_deberta_large.py  # DeBERTa-v3-large fine-tuning
    experiment4_finetune_qwen.py        # Qwen2.5-32B LoRA fine-tuning
    experiment5_inference_finetuned.py   # Standalone inference
    experiment6_comparison.py           # Final comparison tables
    ds_config_zero2.json                # DeepSpeed ZeRO-2 config
    run_deberta_large.slurm             # SLURM for experiment 3
    run_qwen.slurm                      # SLURM for experiment 4
    run_all.sh                          # Local master script
    run_all_tamia.sh                    # TamIA HPC master script
  models/             # Saved fine-tuned models (gitignored)
  results/            # Experiment output JSON files (gitignored)
  checkpoints/        # Training checkpoints (gitignored)
  logs/               # SLURM and training logs (gitignored)
```

## Cost Budget

LLM API calls (experiment 2) are budgeted at $34.00 USD total:
- DeepSeek-v3: $0.56/$1.68 per 1M input/output tokens
- GLM-5: $1.00/$3.20 per 1M input/output tokens

The cost tracker prints estimates before making calls and checkpoints usage
throughout the run.
