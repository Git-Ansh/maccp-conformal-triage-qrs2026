#!/bin/bash
# run_all.sh -- Master script to run all model ablation experiments locally.
#
# Usage:
#   cd model_ablation
#   bash scripts/run_all.sh
#
# Prerequisites:
#   - Python venv with torch, transformers, peft, xgboost, pandas, numpy
#   - Data in data/{eclipse,mozilla}/
#   - For experiment 2: FIREWORKS_API_KEY env var or ../../.fireworks_key file
#
# Note: Experiments 3-4 require GPU(s). On CPU-only machines, only
#       experiments 1, 2, and 6 will run successfully.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
cd "$BASE_DIR"

# Use project venv if available
if [ -f "../venv/Scripts/python.exe" ]; then
    PYTHON="../venv/Scripts/python.exe"
elif [ -f "../venv/bin/python" ]; then
    PYTHON="../venv/bin/python"
else
    PYTHON="python"
fi

echo "============================================"
echo "MODEL ABLATION: RUNNING ALL EXPERIMENTS"
echo "============================================"
echo "Python: $PYTHON"
echo "Working dir: $(pwd)"
echo "Start: $(date)"
echo ""

# Create output directories
mkdir -p results/hybrid results/llm_zeroshot results/deberta_large results/qwen32b
mkdir -p models/deberta_large models/qwen32b
mkdir -p checkpoints logs

# ---- Experiment 1: Hybrid MACCP (CPU, fast) ----
echo ""
echo "===== EXPERIMENT 1: HYBRID MACCP ====="
$PYTHON scripts/experiment1_hybrid.py
E1_EXIT=$?
echo "Experiment 1 exit: $E1_EXIT"

# ---- Experiment 2: LLM zero-shot (CPU, requires API key) ----
echo ""
echo "===== EXPERIMENT 2: LLM ZERO-SHOT ====="
$PYTHON scripts/experiment2_llm_zeroshot.py
E2_EXIT=$?
echo "Experiment 2 exit: $E2_EXIT"

# ---- Experiment 3: DeBERTa-large fine-tuning (GPU required) ----
echo ""
echo "===== EXPERIMENT 3: DeBERTa-LARGE FINE-TUNING ====="
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    $PYTHON scripts/experiment3_finetune_deberta_large.py --dataset eclipse
    E3_ECLIPSE=$?
    $PYTHON scripts/experiment3_finetune_deberta_large.py --dataset mozilla
    E3_MOZILLA=$?
    echo "Experiment 3 exit: eclipse=$E3_ECLIPSE, mozilla=$E3_MOZILLA"
else
    echo "SKIPPED: No GPU available. Use SLURM script for HPC."
    E3_ECLIPSE=0
    E3_MOZILLA=0
fi

# ---- Experiment 4: Qwen32B LoRA fine-tuning (GPU required) ----
echo ""
echo "===== EXPERIMENT 4: Qwen32B LoRA FINE-TUNING ====="
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    $PYTHON scripts/experiment4_finetune_qwen.py --dataset eclipse
    E4_ECLIPSE=$?
    $PYTHON scripts/experiment4_finetune_qwen.py --dataset mozilla
    E4_MOZILLA=$?
    echo "Experiment 4 exit: eclipse=$E4_ECLIPSE, mozilla=$E4_MOZILLA"
else
    echo "SKIPPED: No GPU available. Use SLURM script for HPC."
    E4_ECLIPSE=0
    E4_MOZILLA=0
fi

# ---- Experiment 6: Final comparison ----
echo ""
echo "===== EXPERIMENT 6: FINAL COMPARISON ====="
$PYTHON scripts/experiment6_comparison.py
E6_EXIT=$?
echo "Experiment 6 exit: $E6_EXIT"

echo ""
echo "============================================"
echo "ALL EXPERIMENTS COMPLETE"
echo "End: $(date)"
echo "  Exp 1 (Hybrid):        $E1_EXIT"
echo "  Exp 2 (LLM):           $E2_EXIT"
echo "  Exp 3 (DeBERTa-L):     eclipse=$E3_ECLIPSE, mozilla=$E3_MOZILLA"
echo "  Exp 4 (Qwen32B):       eclipse=$E4_ECLIPSE, mozilla=$E4_MOZILLA"
echo "  Exp 6 (Comparison):    $E6_EXIT"
echo "============================================"
