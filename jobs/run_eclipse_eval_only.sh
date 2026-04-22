#!/bin/bash
#SBATCH --job-name=deberta_eval
#SBATCH --account=aip-rnishat
#SBATCH --partition=gpubase_bynode_b1
#SBATCH --gpus-per-node=h100:4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=0:30:00
#SBATCH --output=jobs/deberta_eval_%j.out
#SBATCH --error=jobs/deberta_eval_%j.err

# Quick eval + conformal on already-trained model (no training needed)

echo "=== DeBERTa Eval + Conformal (no training) ==="
echo "Date: $(date)"
echo "Host: $(hostname)"

module load python/3.11.5 cuda/12.6 scipy-stack/2026a arrow/17.0.0
source $SCRATCH/venv_cascade/bin/activate

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

PROJ_DIR=~/links/projects/aip-rnishat/shared/perf-regression-ci
cd $PROJ_DIR

MODEL_DIR=$SCRATCH/perf-regression-ci-outputs/deberta_eclipse
PROCESSED_DIR=$SCRATCH/perf-regression-ci-outputs/eclipse_processed
RESULTS_DIR=$SCRATCH/perf-regression-ci-outputs/conformal_eclipse

# Verify model exists
ls -la $MODEL_DIR/best_model/config.json

# Run final eval (saves cal/test probs + preds)
python -u -c "
import json, numpy as np, torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, top_k_accuracy_score

model_dir = Path('$MODEL_DIR')
data_dir = Path('$PROCESSED_DIR')

label_map = json.load(open(data_dir / 'label_mapping.json'))
num_classes = len(label_map)
inv_map = {v: k for k, v in label_map.items()}

cal_df = pd.read_parquet(data_dir / 'cal.parquet')
test_df = pd.read_parquet(data_dir / 'test.parquet')
print(f'Cal: {len(cal_df):,} | Test: {len(test_df):,} | Classes: {num_classes}')

tokenizer = AutoTokenizer.from_pretrained(str(model_dir / 'best_model'))
model = AutoModelForSequenceClassification.from_pretrained(
    str(model_dir / 'best_model'), torch_dtype=torch.float32
).to('cuda').eval()

class DS(Dataset):
    def __init__(self, texts, labels, tok, ml=512):
        self.texts, self.labels, self.tok, self.ml = texts, labels, tok, ml
    def __len__(self): return len(self.texts)
    def __getitem__(self, i):
        e = self.tok(self.texts[i], max_length=self.ml, padding='max_length', truncation=True, return_tensors='pt')
        return {'input_ids': e['input_ids'].squeeze(0), 'attention_mask': e['attention_mask'].squeeze(0),
                'label': torch.tensor(self.labels[i], dtype=torch.long)}

for split_name, df in [('cal', cal_df), ('test', test_df)]:
    ds = DS(df['text'].tolist(), df['label'].tolist(), tokenizer)
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    all_probs, all_labels, all_preds = [], [], []
    with torch.no_grad():
        for batch in dl:
            out = model(input_ids=batch['input_ids'].to('cuda'), attention_mask=batch['attention_mask'].to('cuda'))
            probs = torch.softmax(out.logits.float(), dim=-1)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(batch['label'].numpy())
            all_preds.extend(probs.argmax(dim=-1).cpu().numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    acc = accuracy_score(all_labels, all_preds)
    f1m = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    f1w = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    top3 = top_k_accuracy_score(all_labels, all_probs, k=3, labels=range(num_classes))

    print(f'\n{split_name.upper()} SET:')
    print(f'  Accuracy: {acc:.4f}')
    print(f'  F1 (macro): {f1m:.4f}')
    print(f'  F1 (weighted): {f1w:.4f}')
    print(f'  Top-3 accuracy: {top3:.4f}')

    np.save(model_dir / f'{split_name}_probs.npy', all_probs)
    np.save(model_dir / f'{split_name}_labels.npy', all_labels)
    np.save(model_dir / f'{split_name}_preds.npy', all_preds)
    print(f'  Saved {split_name}_probs.npy ({all_probs.shape})')

# Print per-class report for test
print('\nPer-class report (test):')
target_names = [inv_map[i] for i in range(num_classes)]
print(classification_report(all_labels, all_preds, target_names=target_names, digits=3, zero_division=0))

# GO/NO-GO
acc = accuracy_score(all_labels, all_preds)
if acc >= 0.75: print(f'[GO] Test accuracy {acc:.1%}. Conformal sets will be practical.')
elif acc >= 0.70: print(f'[VIABLE] Test accuracy {acc:.1%}. Sets will be moderate.')
elif acc >= 0.65: print(f'[BORDERLINE] Test accuracy {acc:.1%}. Consider fewer components.')
else: print(f'[PROBLEM] Test accuracy {acc:.1%}. Need to regroup.')
"

if [ $? -ne 0 ]; then
    echo "FAILED: Eval"
    exit 1
fi

# Run conformal prediction
echo ""
echo "=== Conformal Prediction ==="
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
echo "=== Done ==="
echo "Date: $(date)"
