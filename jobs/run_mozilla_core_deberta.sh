#!/bin/bash
#SBATCH --job-name=deberta_core
#SBATCH --account=aip-rnishat
#SBATCH --partition=gpubase_bynode_b1
#SBATCH --gpus-per-node=h100:4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=2:00:00
#SBATCH --output=jobs/deberta_core_%j.out
#SBATCH --error=jobs/deberta_core_%j.err

# ─── DeBERTa Fine-Tuning + Conformal Prediction for Mozilla Core Component Assignment ───
# 29K train, 8.3K cal, 8.5K test, 20 classes (no Other)
# Expected runtime: ~15-30 min fine-tuning + ~5 min conformal on 4x H100

echo "=== Mozilla Core DeBERTa Pipeline ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

module load python/3.11.5 cuda/12.6 scipy-stack/2026a arrow/17.0.0
source $SCRATCH/venv_cascade/bin/activate

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

PROJ_DIR=~/links/projects/aip-rnishat/shared/perf-regression-ci
cd $PROJ_DIR

# Verify GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}, Device: {torch.cuda.get_device_name(0)}')"

DATA_DIR=$PROJ_DIR/data/mozilla_core
MODEL_DIR=$SCRATCH/perf-regression-ci-outputs/deberta_mozilla_core
RESULTS_DIR=$SCRATCH/perf-regression-ci-outputs/conformal_mozilla_core

# ─── Step 1: Fine-tune DeBERTa (4x H100 via torchrun) ───
echo ""
echo "=== Step 1: Fine-tuning DeBERTa ==="
torchrun --nproc_per_node=4 src/conformal/pipeline/finetune_deberta.py \
    --data_dir $DATA_DIR \
    --output_dir $MODEL_DIR \
    --model_name microsoft/deberta-v3-base \
    --max_length 256 \
    --batch_size 16 \
    --epochs 5 \
    --lr 2e-5 \
    --patience 2 \
    --distributed

if [ $? -ne 0 ]; then
    echo "FAILED: DeBERTa fine-tuning"
    exit 1
fi

# ─── Step 2: Conformal prediction on DeBERTa outputs ───
echo ""
echo "=== Step 2: Conformal Prediction ==="
python -u src/conformal/pipeline/run_conformal.py \
    --model_dir $MODEL_DIR \
    --data_dir $DATA_DIR \
    --output_dir $RESULTS_DIR \
    --method raps \
    --lam 0.01 \
    --k_reg 5 \
    --alpha_levels 0.01 0.05 0.10 0.20 \
    --n_bootstrap 10000

echo ""
echo "=== Pipeline Complete ==="
echo "Data: $DATA_DIR"
echo "Model: $MODEL_DIR/best_model/"
echo "Conformal results: $RESULTS_DIR/"
echo "Date: $(date)"
