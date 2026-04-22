#!/bin/bash
#SBATCH --job-name=setup_env
#SBATCH --account=aip-rnishat
#SBATCH --partition=cpubase_bycore_b2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=jobs/setup_env_%j.out
#SBATCH --error=jobs/setup_env_%j.err

# One-time environment setup for perf-regression-ci on TamIA
# Run this ONCE after cloning the repo

echo "=== Environment Setup ==="
echo "Date: $(date)"
echo "Host: $(hostname)"

module load python/3.11.5 cuda/12.6 scipy-stack/2026a

# Create virtual environment
echo "Creating virtual environment at $SCRATCH/venv_cascade..."
virtualenv --no-download $SCRATCH/venv_cascade
source $SCRATCH/venv_cascade/bin/activate

# Alliance pre-built wheels (fast, no internet needed)
echo "Installing Alliance pre-built packages..."
pip install --no-index torch torchvision 2>&1 | tail -3
pip install --no-index xgboost scikit-learn pandas matplotlib seaborn 2>&1 | tail -3

# Packages needing internet (available on login nodes)
echo "Installing additional packages..."
pip install sentence-transformers 2>&1 | tail -3
pip install tabpfn 2>&1 | tail -3
pip install mapie imbalanced-learn 2>&1 | tail -3
pip install betacal venn-abers 2>&1 | tail -3
pip install ruptures joblib scipy 2>&1 | tail -3

# Verify
echo ""
echo "=== Verification ==="
python -c "
import torch; print(f'torch {torch.__version__}, CUDA={torch.cuda.is_available()}')
import xgboost; print(f'xgboost {xgboost.__version__}')
import sklearn; print(f'sklearn {sklearn.__version__}')
import pandas; print(f'pandas {pandas.__version__}')
try:
    from sentence_transformers import SentenceTransformer
    print('sentence-transformers: OK')
except: print('sentence-transformers: FAILED')
try:
    import mapie; print(f'mapie {mapie.__version__}')
except: print('mapie: FAILED')
try:
    import tabpfn; print('tabpfn: OK')
except: print('tabpfn: FAILED (may need GPU node)')
try:
    import imblearn; print(f'imblearn {imblearn.__version__}')
except: print('imblearn: FAILED')
"

# Create output symlinks
PROJ_DIR=~/links/projects/aip-rnishat/shared/perf-regression-ci
cd $PROJ_DIR
mkdir -p $SCRATCH/perf-regression-ci-outputs/eclipse
mkdir -p $SCRATCH/perf-regression-ci-outputs/jm1
mkdir -p $SCRATCH/perf-regression-ci-outputs/servicenow
mkdir -p $SCRATCH/perf-regression-ci-outputs/mozilla

# Only create symlinks if they don't exist
[ ! -L conformal_outputs ] && ln -s $SCRATCH/perf-regression-ci-outputs conformal_outputs 2>/dev/null
[ ! -L cascade_outputs ] && ln -s $SCRATCH/perf-regression-ci-outputs/mozilla cascade_outputs 2>/dev/null

echo ""
echo "=== Setup Complete ==="
echo "Venv: $SCRATCH/venv_cascade"
echo "Outputs: $SCRATCH/perf-regression-ci-outputs"
