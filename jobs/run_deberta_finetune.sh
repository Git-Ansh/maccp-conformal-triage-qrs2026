#!/bin/bash
#SBATCH --job-name=deberta_ft
#SBATCH --account=aip-rnishat
#SBATCH --partition=gpubase_bynode_b1
#SBATCH --gpus-per-node=h100:4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=3:00:00
#SBATCH --output=jobs/deberta_ft_%j.out
#SBATCH --error=jobs/deberta_ft_%j.err

echo "=== DeBERTa Fine-Tuning for Eclipse Component Assignment ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

module load python/3.11.5 cuda/12.6 scipy-stack/2026a
source $SCRATCH/venv_cascade/bin/activate

# Force offline mode for HuggingFace (compute nodes have no internet)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

PROJ_DIR=~/links/projects/aip-rnishat/shared/perf-regression-ci
cd $PROJ_DIR

# Verify GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}, Device: {torch.cuda.get_device_name(0)}')"

# Run DeBERTa fine-tuning
python -u src/conformal/pipeline/run_deberta_finetune.py

echo "=== Done ==="
echo "Date: $(date)"
