# Agreement Matters: Reliable Bug Triage with Conformal Prediction

**QRS 2026 Submission** - Model-Agreement Conditioned Conformal Prediction (MACCP) for multi-class bug component assignment.

We introduce MACCP, which partitions bugs by whether two diverse classifiers (DeBERTa transformer + XGBoost tree ensemble) agree on the predicted component, then applies Mondrian conformal prediction with per-group coverage guarantees. On Eclipse (30 components, 158K bugs), the agree partition achieves 61.2% single-component prediction sets with 90.0% empirical singleton accuracy, enabling automated triage for 29% of incoming bugs.

## Key Results

### Eclipse (30 classes, alpha = 0.10)

| Method | Coverage | Mean Set Size | Singletons | Sing. Acc |
|--------|----------|---------------|------------|-----------|
| DeBERTa RAPS | 87.3% | 5.30 | 23.5% | 84.7% |
| XGBoost RAPS | 93.5% | 3.12 | 20.1% | 93.6% |
| **MACCP agree (46.7%)** | **91.5%** | **2.30** | **61.2%** | **90.0%** |
| MACCP disagree (53.3%) | 88.4% | 8.52 | 0.1% | N/A |
| Hybrid MACCP overall | 92.9% | 3.41 | 32.2% | 90.3% |

### Cross-Model Agreement Gaps (Within Identical Confidence Bins)

| Confidence Bin | n | Agree Acc. | Disagree Acc. | Gap |
|----------------|---|-----------|---------------|-----|
| [0.00, 0.30) | 1,341 | 78.0% | 8.7% | +69.3pp |
| [0.30, 0.50) | 3,910 | 77.0% | 13.4% | +63.6pp |
| [0.50, 0.70) | 4,172 | 75.5% | 19.6% | +55.9pp |
| [0.70, 0.85) | 3,594 | 79.3% | 25.2% | +54.2pp |
| [0.85, 1.00) | 12,482 | 89.7% | 46.6% | +43.0pp |

### Coverage Under Temporal Shift

| alpha | Nominal | Eclipse-30 | Eclipse-31 | Mozilla | ServiceNow |
|-------|---------|-----------|-----------|---------|------------|
| 0.01 | 99% | 97.0% | 97.4% | 99.6% | 98.9% |
| 0.05 | 95% | 92.6% | 93.6% | 95.6% | 95.3% |
| 0.10 | 90% | 87.3% | 89.5% | 90.4% | 91.6% |
| 0.20 | 80% | 79.0% | 80.1% | 83.9% | 84.4% |

## Datasets

| Dataset | Classes | Train | Calibration | Test | Temporal Span | Text |
|---------|---------|-------|-------------|------|---------------|------|
| Eclipse (Zenodo 2024) | 30 | 102,725 | 30,017 | 25,499 | 2001-2024 | Summary + Description |
| Mozilla Core (BugsRepo) | 20 | 28,964 | 8,289 | 8,480 | 1998-2016 | Summary only |
| ServiceNow (UCI 498) | 11 | 13,635 | 4,545 | 4,546 | 2016-2017 | None (metadata only) |

All datasets use strictly chronological 60/20/20 splits (train/calibration/test). Data files are not included in this repository due to size; see the paper for download instructions.

## Repository Structure

```
paper/                              # QRS 2026 submission draft PDF

src/
    conformal/                      # MACCP framework (primary codebase)
        data/                       # Dataset loaders
            eclipse_zenodo_loader.py    # Eclipse Zenodo 2024 (301K bugs)
            servicenow_loader.py        # ServiceNow UCI 498
            jm1_loader.py              # JM1 defect data (side study)
            eclipse_loader.py           # Eclipse MSR 2013 (legacy)
        stages/                     # Per-dataset configuration
            eclipse_config.py           # 30-class component assignment
            servicenow_config.py        # 11-class incident routing
            jm1_config.py              # Binary defect prediction
        pipeline/                   # Entry points and orchestration
            prepare_data.py             # 60/20/20 temporal split + label mapping
            finetune_deberta.py         # DeBERTa-v3-base fine-tuning (DDP)
            run_conformal.py            # RAPS conformal prediction + AUGRC
            run_maccp.py                # MACCP: agreement-conditioned CP
            run_xgb_agreement_analysis.py   # Cross-model agreement analysis
            run_eclipse.py              # Eclipse end-to-end pipeline
            run_mozilla_core_maccp.py   # Mozilla Core end-to-end pipeline
            run_servicenow_conformal.py # ServiceNow XGBoost RAPS
            run_baselines.py            # Baseline gauntlet (majority, LR, RF, XGB)
            generate_explanations.py    # Generate explanation examples
        evaluation/                 # Metrics and comparison utilities
            comparison.py               # Cross-method comparison
        explanation_generator.py    # Rule-based explanation module
        llm/                        # LLM classifier (requires API key)
            fireworks_client.py         # Fireworks AI client
            llm_classifier.py           # LLM-based classification

    common/                         # Shared utilities across all modules
        data_paths.py               # Path constants, column names, LEAKAGE_COLUMNS
        evaluation_utils.py         # Binary/multiclass metrics, threshold tuning
        model_utils.py              # Model save/load, cross-validation, seeds
        visualization_utils.py      # Plotting with consistent styling

    cascade_legacy/                 # Legacy cascade framework (not used in paper)
        framework/                  # GeneralCascade, ConfidenceStage
        stages/                     # Mozilla Perfherder stage implementations
        pipeline/                   # Entry points (run_cascade, cross_repo)
        data/                       # Data loader for Perfherder alerts
        evaluation/                 # Metrics, calibration, bootstrap CIs
        bug_prediction/             # Retrieval + LLM triage baselines

    legacy_phases/                  # Legacy 8-phase comparison pipeline (not used in paper)
        phase_1/ - phase_7/         # Binary/multi-class classification, CPD, forecasting
        phase_8_deep_learning/      # CNN, LSTM, GRU, Transformer experiments

model_ablation/                     # Side study: model and baseline comparison
    scripts/                        # Experiment scripts
        experiment1_hybrid.py           # Hybrid MACCP configurations (A/B/C/D)
        experiment2_llm_zeroshot.py     # LLM zero-shot (GLM-5)
        experiment3_finetune_deberta_large.py  # DeBERTa-v3-large
        experiment4_finetune_qwen.py    # Qwen 32B LoRA fine-tuning
        experiment5_inference_finetuned.py     # Inference on fine-tuned LLMs
        experiment6_comparison.py       # Master comparison table
        ds_config_zero2.json            # DeepSpeed ZeRO Stage 2 config
        run_deberta_large.slurm         # SLURM job for DeBERTa-large
        run_qwen.slurm                  # SLURM job for Qwen (4x H100)
        run_deepseek.slurm              # SLURM job for DeepSeek-R1
    xgboost_ablation/               # XGBoost feature-channel ablation
        run_ablation.py                 # Metadata-only vs TF-IDF-only vs full
        results.json                    # Ablation results
    mozilla_fix/                    # Mozilla XGBoost feature engineering
        run_mozilla_fix.py              # Reporter component history features
    data/                           # Cached predictions (Eclipse + Mozilla)
        eclipse/                        # DeBERTa/XGBoost preds + parquet splits
        mozilla/                        # DeBERTa/XGBoost preds + parquet splits

jobs/                               # TamIA HPC SLURM scripts
    run_deberta_finetune.sh             # DeBERTa fine-tuning (4x H100)
    run_eclipse_deberta.sh              # Eclipse evaluation
    run_mozilla_core_deberta.sh         # Mozilla Core evaluation
    setup_env.sh                        # Environment setup on TamIA

scripts/                            # Local helper scripts
    tamia_push_and_run.sh               # Push code + submit SLURM job
    tamia_pull_results.sh               # Pull results from TamIA
    tamia_watch_jobs.sh                 # Job monitoring with crash detection
```

## Setup

### Requirements

```bash
python -m venv venv
source venv/bin/activate
pip install pandas numpy scikit-learn xgboost scipy matplotlib seaborn
pip install torch transformers datasets accelerate   # For DeBERTa fine-tuning
pip install pyarrow                                   # For parquet data files
```

### Data Preparation

Prepare Eclipse (or any dataset) with chronological 60/20/20 split:

```bash
python src/conformal/pipeline/prepare_data.py \
    --data_dir data/external/eclipse_zenodo \
    --output_dir data/eclipse/ \
    --top_k 30 --dataset_name eclipse --no_other \
    --date_col creation_time --component_col component \
    --summary_col summary --description_col description
```

### DeBERTa Fine-tuning

Single GPU:
```bash
python src/conformal/pipeline/finetune_deberta.py \
    --data_dir data/eclipse/ --output_dir models/deberta_eclipse/ \
    --max_length 512 --batch_size 16 --epochs 3 --lr 2e-5
```

Multi-GPU (4x H100 via DDP):
```bash
torchrun --nproc_per_node=4 src/conformal/pipeline/finetune_deberta.py \
    --data_dir data/eclipse/ --output_dir models/deberta_eclipse/ \
    --max_length 512 --batch_size 16 --epochs 3 --lr 2e-5 --distributed
```

### XGBoost Training + Agreement Analysis

```bash
python src/conformal/pipeline/run_xgb_agreement_analysis.py
```

### Conformal Prediction (RAPS)

```bash
python src/conformal/pipeline/run_conformal.py \
    --model_dir models/deberta_eclipse/ --data_dir data/eclipse/ \
    --output_dir results/eclipse/ --method raps \
    --alpha_levels 0.01 0.05 0.10 0.20
```

### MACCP (Agreement-Conditioned CP)

```bash
python src/conformal/pipeline/run_maccp.py
```

### XGBoost Feature-Channel Ablation

```bash
python model_ablation/xgboost_ablation/run_ablation.py
```

## Method Overview

1. **Two diverse classifiers** process each bug report in parallel:
   - DeBERTa-v3-base (text: summary + description, 184M parameters)
   - XGBoost (metadata: product, reporter, severity, priority + 500-dim TF-IDF)

2. **Agreement check**: if both models predict the same component, the bug enters the **agree** partition; otherwise the **disagree** partition.

3. **Mondrian conformal prediction**: separate RAPS thresholds per group.
   - Agree group gets a tight threshold (small, precise prediction sets)
   - Disagree group gets a wide threshold (large, conservative sets)
   - Each group receives an independent coverage guarantee: P(true class in set) >= 1 - alpha

4. **Operational tiers**:
   - **AUTO-TRIAGE** (29%): agree-partition singletons, 90.0% empirical accuracy
   - **REVIEW SHORTLIST** (10%): agree-partition non-singletons, mean 2.3 options
   - **NEEDS CAREFUL REVIEW** (56%): disagree partition with uncertainty context

## Paper

The submission draft PDF is located in the `paper/` directory.

## Technical Notes

- **Temporal split**: strictly chronological 60/20/20 to prevent data leakage
- **RAPS parameters**: lambda = 0.01, k_reg = 5 (consistent across all experiments)
- **Coverage guarantee**: applies to prediction sets (set membership), not to singleton accuracy. Singleton accuracy is an empirical observation.
- **Exchangeability**: temporal splits violate strict exchangeability; coverage is empirically validated (within 1-4pp of nominal across 15-23 year spans)
- **MACCP applicability**: requires two competent classifiers with diverse feature representations. On ServiceNow (no text data), only single-model XGBoost RAPS is evaluated.
