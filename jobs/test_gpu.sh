#!/bin/bash
#SBATCH --job-name=test_gpu
#SBATCH --account=aip-rnishat
#SBATCH --partition=gpubase_bynode_b1
#SBATCH --gpus-per-node=h100:4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=0:15:00
#SBATCH --output=jobs/test_gpu_%j.out
#SBATCH --error=jobs/test_gpu_%j.err

echo "=== GPU Test Job ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo ""

module load python/3.11.5 cuda/12.6 scipy-stack/2026a
source $SCRATCH/venv_cascade/bin/activate

# Force offline mode for HuggingFace (compute nodes have no internet)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "=== Python ==="
python --version
echo ""

echo "=== CUDA ==="
nvidia-smi
echo ""

echo "=== PyTorch ==="
python -c "
import torch
print(f'torch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device count: {torch.cuda.device_count()}')
    print(f'Device name: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    # Quick compute test
    x = torch.randn(1000, 1000, device='cuda')
    y = torch.mm(x, x)
    print(f'Matrix multiply test: OK ({y.shape})')
else:
    print('ERROR: CUDA not available!')
"
echo ""

echo "=== Packages ==="
python -c "
import xgboost; print(f'xgboost: {xgboost.__version__}')
import sklearn; print(f'sklearn: {sklearn.__version__}')
import pandas; print(f'pandas: {pandas.__version__}')
" 2>&1

# Check sentence-transformers
python -c "
from sentence_transformers import SentenceTransformer
print('sentence-transformers: OK')
model = SentenceTransformer('all-MiniLM-L6-v2')
emb = model.encode(['test sentence'])
print(f'Embedding shape: {emb.shape}')
print(f'Device: {model.device}')
" 2>&1

echo ""
echo "=== Test Complete ==="
echo "All systems operational."
