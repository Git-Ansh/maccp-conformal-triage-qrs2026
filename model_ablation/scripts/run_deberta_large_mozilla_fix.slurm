#!/bin/bash
#SBATCH --job-name=deb_large_moz_fix
#SBATCH --account=aip-rnishat
#SBATCH --partition=gpubase_bynode_b1
#SBATCH --gpus-per-node=h100:4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=2:00:00
#SBATCH --output=model_ablation/logs/deberta_large_mozilla_fix_%j.out
#SBATCH --error=model_ablation/logs/deberta_large_mozilla_fix_%j.err

echo "=== DeBERTa-large Mozilla Retrain (lr=5e-6, maxlen=128, epochs=5) ==="
echo "Date: $(date)"

module load python/3.11.5 cuda/12.6 scipy-stack/2026a arrow/17.0.0
source $SCRATCH/venv_cascade/bin/activate

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

PROJ_DIR=~/links/projects/aip-rnishat/shared/perf-regression-ci
cd $PROJ_DIR

python -u -c "
import sys
sys.path.insert(0, 'model_ablation/scripts')
import experiment3_finetune_deberta_large as exp3
# Override hyperparameters
exp3.LEARNING_RATE = 5e-6
exp3.MAX_LENGTH = 128
exp3.NUM_EPOCHS = 5
exp3.WARMUP_RATIO = 0.15
print('Overrides: lr=5e-6, max_length=128, epochs=5, warmup=0.15')
exp3.train_and_evaluate('mozilla')
"

echo "=== Done: $(date) ==="
