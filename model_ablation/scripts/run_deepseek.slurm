#!/bin/bash
#SBATCH --job-name=deepseek_lora
#SBATCH --account=aip-rnishat
#SBATCH --partition=gpubase_bynode_b2
#SBATCH --gpus-per-node=h100:4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=model_ablation/logs/deepseek_%j.out
#SBATCH --error=model_ablation/logs/deepseek_%j.err

echo "=== DeepSeek-R1-Distill-Qwen-32B LoRA Fine-tuning ==="
echo "Date: $(date)"

module load python/3.11.5 cuda/12.6 scipy-stack/2026a arrow/17.0.0
source $SCRATCH/venv_cascade/bin/activate

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export QWEN_MODEL_PATH=/scratch/a/anshshah/deepseek_manual

PROJ_DIR=~/links/projects/aip-rnishat/shared/perf-regression-ci
cd $PROJ_DIR

python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"

echo "=== Eclipse ==="
deepspeed --num_gpus=4 model_ablation/scripts/experiment4_finetune_qwen.py --dataset eclipse

echo "=== Mozilla ==="
deepspeed --num_gpus=4 model_ablation/scripts/experiment4_finetune_qwen.py --dataset mozilla

echo "=== Done: $(date) ==="
