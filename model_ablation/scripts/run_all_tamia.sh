#!/bin/bash
# run_all_tamia.sh -- Submit all model ablation experiments on TamIA HPC.
#
# Usage:
#   cd ~/links/projects/aip-rnishat/shared/perf-regression-ci
#   bash model_ablation/scripts/run_all_tamia.sh
#
# This script:
#   1. Runs experiment 1 (hybrid MACCP) interactively (fast, CPU-only)
#   2. Runs experiment 2 (LLM zero-shot) interactively (CPU, requires API key)
#   3. Submits experiment 3 (DeBERTa-large) as SLURM job
#   4. Submits experiment 4 (Qwen32B) as SLURM job
#   5. Prints instructions for experiment 6 (comparison) after jobs complete

set -e

PROJ_DIR=~/links/projects/aip-rnishat/shared/perf-regression-ci
ABLATION_DIR=$PROJ_DIR/model_ablation
SCRIPT_DIR=$ABLATION_DIR/scripts

cd $PROJ_DIR

module load python/3.11.5 cuda/12.6 scipy-stack/2026a arrow/17.0.0
source $SCRATCH/venv_cascade/bin/activate

echo "============================================"
echo "MODEL ABLATION: TamIA HPC SUBMISSION"
echo "============================================"
echo "Project dir: $PROJ_DIR"
echo "Start: $(date)"
echo ""

# Create output directories
mkdir -p $ABLATION_DIR/results/hybrid
mkdir -p $ABLATION_DIR/results/llm_zeroshot
mkdir -p $ABLATION_DIR/results/deberta_large
mkdir -p $ABLATION_DIR/results/qwen32b
mkdir -p $ABLATION_DIR/models/deberta_large/eclipse
mkdir -p $ABLATION_DIR/models/deberta_large/mozilla
mkdir -p $ABLATION_DIR/models/qwen32b/eclipse
mkdir -p $ABLATION_DIR/models/qwen32b/mozilla
mkdir -p $ABLATION_DIR/checkpoints
mkdir -p $ABLATION_DIR/logs

# ---- Experiment 1: Hybrid MACCP (interactive, fast) ----
echo "===== EXPERIMENT 1: HYBRID MACCP (interactive) ====="
python -u $SCRIPT_DIR/experiment1_hybrid.py
echo ""

# ---- Experiment 2: LLM zero-shot (interactive) ----
echo "===== EXPERIMENT 2: LLM ZERO-SHOT (interactive) ====="
python -u $SCRIPT_DIR/experiment2_llm_zeroshot.py
echo ""

# ---- Experiment 3: DeBERTa-large (SLURM) ----
echo "===== EXPERIMENT 3: SUBMITTING DeBERTa-large SLURM JOB ====="
JOB3=$(sbatch --parsable $SCRIPT_DIR/run_deberta_large.slurm)
echo "  Submitted job: $JOB3"

# ---- Experiment 4: Qwen32B (SLURM) ----
echo "===== EXPERIMENT 4: SUBMITTING Qwen32B SLURM JOB ====="
JOB4=$(sbatch --parsable $SCRIPT_DIR/run_qwen.slurm)
echo "  Submitted job: $JOB4"

echo ""
echo "============================================"
echo "SUBMITTED JOBS"
echo "============================================"
echo "  DeBERTa-large: $JOB3 (4h, gpubase_bynode_b1)"
echo "  Qwen32B:       $JOB4 (12h, gpubase_bynode_b2)"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f $ABLATION_DIR/logs/deberta_large_${JOB3}.out"
echo "  tail -f $ABLATION_DIR/logs/qwen32b_${JOB4}.out"
echo ""
echo "After BOTH jobs complete, run the final comparison:"
echo "  python -u $SCRIPT_DIR/experiment6_comparison.py"
echo ""
echo "Or submit experiment 5 (standalone inference) if needed:"
echo "  python -u $SCRIPT_DIR/experiment5_inference_finetuned.py --model deberta_large --dataset eclipse"
echo "  python -u $SCRIPT_DIR/experiment5_inference_finetuned.py --model qwen32b --dataset eclipse"
echo "============================================"
