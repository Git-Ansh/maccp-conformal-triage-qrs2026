#!/bin/bash
#SBATCH --job-name=deberta_mozilla
#SBATCH --account=aip-rnishat
#SBATCH --partition=gpubase_bynode_b1
#SBATCH --gpus-per-node=h100:4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=1:00:00
#SBATCH --output=jobs/deberta_mozilla_%j.out
#SBATCH --error=jobs/deberta_mozilla_%j.err

echo "=== DeBERTa Fine-Tuning on Mozilla Firefox ==="
echo "Date: $(date)"

module load python/3.11.5 cuda/12.6 scipy-stack/2026a arrow/17.0.0
source $SCRATCH/venv_cascade/bin/activate

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

PROJ_DIR=~/links/projects/aip-rnishat/shared/perf-regression-ci
cd $PROJ_DIR

DATA_DIR=$PROJ_DIR/data/bugsrepo_firefox
MODEL_DIR=$SCRATCH/perf-regression-ci-outputs/deberta_bugsrepo_firefox
RESULTS_DIR=$SCRATCH/perf-regression-ci-outputs/conformal_bugsrepo_firefox

# DeBERTa fine-tuning (7K train, small - will be fast)
torchrun --nproc_per_node=4 src/conformal/pipeline/finetune_deberta.py \
    --data_dir $DATA_DIR \
    --output_dir $MODEL_DIR \
    --model_name microsoft/deberta-v3-base \
    --max_length 128 \
    --batch_size 32 \
    --epochs 5 \
    --lr 2e-5 \
    --patience 2 \
    --distributed

if [ $? -ne 0 ]; then
    echo "FAILED: DeBERTa fine-tuning"
    exit 1
fi

# Conformal prediction
python -u src/conformal/pipeline/run_conformal.py \
    --model_dir $MODEL_DIR \
    --data_dir $DATA_DIR \
    --output_dir $RESULTS_DIR \
    --method raps \
    --lam 0.01 \
    --k_reg 3 \
    --alpha_levels 0.01 0.05 0.10 0.20 \
    --n_bootstrap 10000

echo "=== Done ==="
echo "Date: $(date)"
