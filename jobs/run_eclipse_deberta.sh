#!/bin/bash
#SBATCH --job-name=deberta_eclipse
#SBATCH --account=aip-rnishat
#SBATCH --partition=gpubase_bynode_b1
#SBATCH --gpus-per-node=h100:4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=3:00:00
#SBATCH --output=jobs/deberta_eclipse_%j.out
#SBATCH --error=jobs/deberta_eclipse_%j.err

# ─── DeBERTa Fine-Tuning + Conformal Prediction for Eclipse Component Assignment ───
# Expected runtime: ~30-60 min fine-tuning + ~5 min conformal on 4x H100

echo "=== Eclipse DeBERTa Pipeline ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

module load python/3.11.5 cuda/12.6 scipy-stack/2026a arrow/17.0.0
source $SCRATCH/venv_cascade/bin/activate

# Force offline mode for HuggingFace (compute nodes have no internet)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

PROJ_DIR=~/links/projects/aip-rnishat/shared/perf-regression-ci
cd $PROJ_DIR

# Verify GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}, Device: {torch.cuda.get_device_name(0)}')"

DATA_DIR=$PROJ_DIR/data/external/eclipse_zenodo_lite
PROCESSED_DIR=$SCRATCH/perf-regression-ci-outputs/eclipse_processed
MODEL_DIR=$SCRATCH/perf-regression-ci-outputs/deberta_eclipse
RESULTS_DIR=$SCRATCH/perf-regression-ci-outputs/conformal_eclipse

# ─── Step 1: Prepare data (60/20/20 temporal split + label mapping) ───
echo ""
echo "=== Step 1: Preparing Data ==="
python -u src/conformal/pipeline/prepare_data.py \
    --data_dir $DATA_DIR \
    --output_dir $PROCESSED_DIR \
    --top_k 30 \
    --dataset_name eclipse \
    --date_col creation_time \
    --component_col component \
    --summary_col summary \
    --description_col description \
    --no_other

if [ $? -ne 0 ]; then
    echo "FAILED: Data preparation"
    exit 1
fi

# ─── Step 2: Fine-tune DeBERTa (4x H100 via torchrun) ───
echo ""
echo "=== Step 2: Fine-tuning DeBERTa ==="
torchrun --nproc_per_node=4 src/conformal/pipeline/finetune_deberta.py \
    --data_dir $PROCESSED_DIR \
    --output_dir $MODEL_DIR \
    --model_name microsoft/deberta-v3-base \
    --max_length 512 \
    --batch_size 16 \
    --epochs 5 \
    --lr 2e-5 \
    --patience 2 \
    --distributed

if [ $? -ne 0 ]; then
    echo "FAILED: DeBERTa fine-tuning"
    exit 1
fi

# ─── Step 3: Conformal prediction on DeBERTa outputs ───
echo ""
echo "=== Step 3: Conformal Prediction ==="
python -u src/conformal/pipeline/run_conformal.py \
    --model_dir $MODEL_DIR \
    --data_dir $PROCESSED_DIR \
    --output_dir $RESULTS_DIR \
    --method raps \
    --lam 0.01 \
    --k_reg 5 \
    --alpha_levels 0.01 0.05 0.10 0.20 \
    --n_bootstrap 10000

echo ""
echo "=== Pipeline Complete ==="
echo "Processed data: $PROCESSED_DIR"
echo "Model: $MODEL_DIR/best_model/"
echo "Conformal results: $RESULTS_DIR/"
echo "Date: $(date)"
