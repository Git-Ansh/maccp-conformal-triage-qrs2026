#!/bin/bash
#SBATCH --job-name=xgb_agreement
#SBATCH --account=aip-rnishat
#SBATCH --partition=gpubase_bynode_b1
#SBATCH --gpus-per-node=h100:4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=0:30:00
#SBATCH --output=jobs/xgb_agreement_%j.out
#SBATCH --error=jobs/xgb_agreement_%j.err

echo "=== XGBoost + DeBERTa Agreement Analysis ==="
echo "Date: $(date)"

module load python/3.11.5 cuda/12.6 scipy-stack/2026a arrow/17.0.0
source $SCRATCH/venv_cascade/bin/activate

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

PROJ_DIR=~/links/projects/aip-rnishat/shared/perf-regression-ci
cd $PROJ_DIR

# Point paths to TamIA scratch
export ECLIPSE_DATA_DIR=$SCRATCH/perf-regression-ci-outputs/eclipse_processed
export DEBERTA_DIR=$SCRATCH/perf-regression-ci-outputs/deberta_eclipse
export OUTPUT_DIR=$SCRATCH/perf-regression-ci-outputs/agreement_analysis

python -u src/conformal/pipeline/run_xgb_agreement_analysis.py

echo "=== Done ==="
echo "Date: $(date)"
