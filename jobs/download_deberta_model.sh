#!/bin/bash
# Run this on TamIA LOGIN NODE (has internet) BEFORE submitting the GPU job.
# Downloads the DeBERTa model to HuggingFace cache so compute nodes can use it offline.

echo "=== Pre-downloading DeBERTa-v3-base ==="
echo "This must run on the LOGIN node (internet required)."

module load python/3.11.5 cuda/12.6 scipy-stack/2026a
source $SCRATCH/venv_cascade/bin/activate

python -c "
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

model_name = 'microsoft/deberta-v3-base'
cache_dir = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
print(f'Cache dir: {cache_dir}')

print('Downloading tokenizer...')
tok = AutoTokenizer.from_pretrained(model_name)
print(f'  Tokenizer vocab size: {tok.vocab_size}')

print('Downloading model (184M parameters)...')
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=31)
n_params = sum(p.numel() for p in model.parameters())
print(f'  Parameters: {n_params/1e6:.1f}M')

print('Verifying offline load...')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
tok2 = AutoTokenizer.from_pretrained(model_name)
model2 = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=31)
print('  Offline load: OK')

print()
print('[DONE] DeBERTa model cached. Safe to submit GPU job with HF_HUB_OFFLINE=1.')
"
