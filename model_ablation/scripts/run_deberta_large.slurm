#!/bin/bash
#SBATCH --job-name=deb_large_ablation
#SBATCH --account=aip-rnishat
#SBATCH --partition=gpubase_bynode_b1
#SBATCH --gpus-per-node=h100:4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=3:00:00
#SBATCH --output=model_ablation/logs/deberta_large_%j.out
#SBATCH --error=model_ablation/logs/deberta_large_%j.err

echo "=== DeBERTa-v3-large Ablation ==="
echo "Date: $(date)"

module load python/3.11.5 cuda/12.6 scipy-stack/2026a arrow/17.0.0
source $SCRATCH/venv_cascade/bin/activate

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

PROJ_DIR=~/links/projects/aip-rnishat/shared/perf-regression-ci
cd $PROJ_DIR

python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"

echo "=== Eclipse ==="
python -u model_ablation/scripts/experiment3_finetune_deberta_large.py --dataset eclipse

echo "=== Mozilla ==="
python -u model_ablation/scripts/experiment3_finetune_deberta_large.py --dataset mozilla

echo "=== Done: $(date) ==="
