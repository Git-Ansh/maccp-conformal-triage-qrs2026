#!/usr/bin/env python3
"""
Phase 8: Deep Learning Experiments for Performance Regression Detection

This script implements CNN, LSTM/GRU, and Transformer baselines for:
1. Time series classification (has_bug prediction from raw time series)
2. Hybrid models (time series + alert metadata features)

Expected outcomes based on literature:
- DL typically loses to XGBoost on tabular data <50K samples
- But time series structure may give DL an advantage

Models implemented:
1. 1D-CNN: Convolutional approach for local pattern detection
2. LSTM: Sequential modeling with long-term dependencies
3. GRU: Lighter alternative to LSTM
4. Transformer: Attention-based architecture
5. Hybrid: DL time series encoder + tabular features

Evaluation:
- Same temporal split as Phase 1 (80/20)
- Same target: has_bug
- Metrics: F1, Precision, Recall, MCC, AUC
"""

import sys
import warnings
from pathlib import Path
from datetime import datetime
import gc
import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    matthews_corrcoef, classification_report, confusion_matrix
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# Try importing deep learning libraries
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("WARNING: PyTorch not available. Install with: pip install torch")

# Paths
SRC_DIR = Path(__file__).parent.parent
PROJECT_ROOT = SRC_DIR.parent
DATA_PATH = PROJECT_ROOT / "data" / "alerts_data.csv"
TIMESERIES_DIR = PROJECT_ROOT / "data" / "timeseries-data"
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / 'reports').mkdir(exist_ok=True)
(OUTPUT_DIR / 'models').mkdir(exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
if TORCH_AVAILABLE:
    torch.manual_seed(RANDOM_SEED)

# Hyperparameters
SEQUENCE_LENGTH = 50  # Number of time points to use
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-3
HIDDEN_DIM = 64
NUM_LAYERS = 2
DROPOUT = 0.3
PATIENCE = 10  # Early stopping patience


# ============================================
# DATA LOADING AND PREPROCESSING
# ============================================

def find_timeseries_file(alert_id, timeseries_dir):
    """Find the time series file for a given alert ID."""
    # Search in all repository subdirectories
    for repo_dir in timeseries_dir.iterdir():
        if repo_dir.is_dir() and not repo_dir.name.endswith('.zip'):
            # Handle nested structure (repo/repo/)
            inner_dir = repo_dir / repo_dir.name
            if inner_dir.exists():
                ts_file = inner_dir / f"{alert_id}_timeseries_data.csv"
                if ts_file.exists():
                    return ts_file
            # Also check direct structure
            ts_file = repo_dir / f"{alert_id}_timeseries_data.csv"
            if ts_file.exists():
                return ts_file
    return None


def load_timeseries(file_path, seq_length=SEQUENCE_LENGTH):
    """Load and preprocess a single time series file."""
    try:
        df = pd.read_csv(file_path)
        if 'value' not in df.columns:
            return None
        
        # Sort by timestamp
        if 'push_timestamp' in df.columns:
            df = df.sort_values('push_timestamp')
        
        values = df['value'].values
        
        # Handle variable length - pad or truncate
        if len(values) < seq_length:
            # Pad with mean
            padded = np.full(seq_length, np.mean(values))
            padded[-len(values):] = values
            values = padded
        else:
            # Take last seq_length values (most recent)
            values = values[-seq_length:]
        
        # Normalize
        mean, std = np.mean(values), np.std(values) + 1e-8
        values = (values - mean) / std
        
        return values.astype(np.float32)
    except Exception as e:
        return None


def load_data_with_timeseries(max_samples=None):
    """Load alerts data and match with time series."""
    print("Loading alerts data...")
    df = pd.read_csv(DATA_PATH)
    print(f"  Total alerts: {len(df)}")
    
    # Create target
    df['has_bug'] = df['alert_summary_bug_number'].notna().astype(int)
    print(f"  Alerts with bugs: {df['has_bug'].sum()} ({df['has_bug'].mean()*100:.1f}%)")
    
    # The file names are {signature_id}_timeseries_data.csv
    # signature_id is the time series identifier
    df['ts_file_id'] = df['signature_id'].astype(int)
    
    # Build a cache of available time series files for faster lookup
    print("\nBuilding time series file index...")
    ts_file_cache = {}
    for repo_dir in TIMESERIES_DIR.iterdir():
        if repo_dir.is_dir() and not repo_dir.name.endswith('.zip'):
            inner_dir = repo_dir / repo_dir.name
            if inner_dir.exists():
                for f in inner_dir.glob("*_timeseries_data.csv"):
                    try:
                        sig_id = int(f.stem.split('_')[0])
                        ts_file_cache[sig_id] = f
                    except:
                        pass
    print(f"  Found {len(ts_file_cache)} time series files")
    
    # Load time series for each alert
    print("\nLoading time series data...")
    timeseries_data = []
    valid_indices = []
    
    sample_df = df if max_samples is None else df.head(max_samples)
    
    for idx, row in sample_df.iterrows():
        sig_id = row['ts_file_id']
        ts_file = ts_file_cache.get(sig_id)
        
        if ts_file is not None:
            ts_values = load_timeseries(ts_file)
            if ts_values is not None:
                timeseries_data.append(ts_values)
                valid_indices.append(idx)
        
        if len(valid_indices) % 1000 == 0 and len(valid_indices) > 0:
            print(f"  Loaded {len(valid_indices)} time series...")
    
    print(f"  Successfully loaded {len(valid_indices)} time series")
    
    # Filter dataframe to only include alerts with time series
    df_filtered = df.loc[valid_indices].reset_index(drop=True)
    timeseries_array = np.array(timeseries_data)
    
    return df_filtered, timeseries_array


def prepare_tabular_features(df):
    """Prepare tabular features (same as Phase 1)."""
    features = {}
    
    # Magnitude features (direction-agnostic)
    features['magnitude_abs'] = np.abs(df['single_alert_amount_abs'].fillna(0))
    features['magnitude_pct_abs'] = np.abs(df['single_alert_amount_pct'].fillna(0))
    features['t_value_abs'] = np.abs(df['single_alert_t_value'].fillna(0))
    
    # Value features
    features['value_mean'] = (df['single_alert_new_value'].fillna(0) + 
                              df['single_alert_prev_value'].fillna(0)) / 2
    
    # Context features (encoded)
    context_cols = [
        'alert_summary_repository',
        'single_alert_series_signature_framework_id',
        'single_alert_series_signature_machine_platform',
        'single_alert_series_signature_suite',
    ]
    
    for col in context_cols:
        if col in df.columns:
            le = LabelEncoder()
            features[f'{col}_enc'] = le.fit_transform(df[col].fillna('unknown').astype(str))
    
    # Workflow feature
    if 'single_alert_manually_created' in df.columns:
        features['manually_created'] = df['single_alert_manually_created'].fillna(0).astype(int)
    
    feature_df = pd.DataFrame(features)
    return feature_df.values.astype(np.float32)


# ============================================
# PYTORCH DATASETS
# ============================================

if TORCH_AVAILABLE:
    class TimeSeriesDataset(Dataset):
        """Dataset for time series only."""
        def __init__(self, timeseries, labels):
            self.timeseries = torch.tensor(timeseries, dtype=torch.float32)
            self.labels = torch.tensor(labels, dtype=torch.float32)
        
        def __len__(self):
            return len(self.labels)
        
        def __getitem__(self, idx):
            return self.timeseries[idx], self.labels[idx]
    
    
    class HybridDataset(Dataset):
        """Dataset for time series + tabular features."""
        def __init__(self, timeseries, tabular, labels):
            self.timeseries = torch.tensor(timeseries, dtype=torch.float32)
            self.tabular = torch.tensor(tabular, dtype=torch.float32)
            self.labels = torch.tensor(labels, dtype=torch.float32)
        
        def __len__(self):
            return len(self.labels)
        
        def __getitem__(self, idx):
            return self.timeseries[idx], self.tabular[idx], self.labels[idx]


# ============================================
# DEEP LEARNING MODELS
# ============================================

if TORCH_AVAILABLE:
    class CNN1D(nn.Module):
        """1D Convolutional Neural Network for time series."""
        def __init__(self, seq_length, hidden_dim=64, dropout=0.3):
            super(CNN1D, self).__init__()
            
            self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
            self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
            
            self.pool = nn.MaxPool1d(2)
            self.dropout = nn.Dropout(dropout)
            self.relu = nn.ReLU()
            
            # Calculate output size after convolutions
            conv_out_size = seq_length // 8  # After 3 pooling layers
            
            self.fc1 = nn.Linear(128 * conv_out_size, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, 1)
        
        def forward(self, x):
            # x: (batch, seq_length) -> (batch, 1, seq_length)
            x = x.unsqueeze(1)
            
            x = self.pool(self.relu(self.conv1(x)))
            x = self.pool(self.relu(self.conv2(x)))
            x = self.pool(self.relu(self.conv3(x)))
            
            x = x.flatten(1)
            x = self.dropout(self.relu(self.fc1(x)))
            x = self.fc2(x)
            
            return x.squeeze()
    
    
    class LSTMClassifier(nn.Module):
        """LSTM for time series classification."""
        def __init__(self, input_dim=1, hidden_dim=64, num_layers=2, dropout=0.3):
            super(LSTMClassifier, self).__init__()
            
            self.lstm = nn.LSTM(
                input_dim, hidden_dim, num_layers,
                batch_first=True, dropout=dropout if num_layers > 1 else 0,
                bidirectional=True
            )
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_dim * 2, 1)  # *2 for bidirectional
        
        def forward(self, x):
            # x: (batch, seq_length) -> (batch, seq_length, 1)
            x = x.unsqueeze(-1)
            
            lstm_out, (h_n, c_n) = self.lstm(x)
            
            # Use last hidden state from both directions
            hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
            
            x = self.dropout(hidden)
            x = self.fc(x)
            
            return x.squeeze()
    
    
    class GRUClassifier(nn.Module):
        """GRU for time series classification."""
        def __init__(self, input_dim=1, hidden_dim=64, num_layers=2, dropout=0.3):
            super(GRUClassifier, self).__init__()
            
            self.gru = nn.GRU(
                input_dim, hidden_dim, num_layers,
                batch_first=True, dropout=dropout if num_layers > 1 else 0,
                bidirectional=True
            )
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_dim * 2, 1)
        
        def forward(self, x):
            x = x.unsqueeze(-1)
            
            gru_out, h_n = self.gru(x)
            hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
            
            x = self.dropout(hidden)
            x = self.fc(x)
            
            return x.squeeze()
    
    
    class TransformerClassifier(nn.Module):
        """Transformer encoder for time series classification."""
        def __init__(self, seq_length, d_model=64, nhead=4, num_layers=2, dropout=0.3):
            super(TransformerClassifier, self).__init__()
            
            self.embedding = nn.Linear(1, d_model)
            self.pos_encoding = nn.Parameter(torch.randn(1, seq_length, d_model))
            
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            
            self.fc = nn.Linear(d_model, 1)
            self.dropout = nn.Dropout(dropout)
        
        def forward(self, x):
            # x: (batch, seq_length) -> (batch, seq_length, 1)
            x = x.unsqueeze(-1)
            
            x = self.embedding(x) + self.pos_encoding
            x = self.transformer(x)
            
            # Global average pooling
            x = x.mean(dim=1)
            x = self.dropout(x)
            x = self.fc(x)
            
            return x.squeeze()
    
    
    class HybridModel(nn.Module):
        """Hybrid model combining time series encoder with tabular features."""
        def __init__(self, seq_length, tabular_dim, hidden_dim=64, dropout=0.3):
            super(HybridModel, self).__init__()
            
            # Time series encoder (CNN)
            self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
            self.pool = nn.MaxPool1d(2)
            self.relu = nn.ReLU()
            
            conv_out_size = seq_length // 4
            self.ts_fc = nn.Linear(64 * conv_out_size, hidden_dim)
            
            # Tabular encoder
            self.tab_fc1 = nn.Linear(tabular_dim, hidden_dim)
            self.tab_fc2 = nn.Linear(hidden_dim, hidden_dim)
            
            # Combined classifier
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(hidden_dim * 2, 1)
        
        def forward(self, ts, tab):
            # Time series branch
            ts = ts.unsqueeze(1)
            ts = self.pool(self.relu(self.conv1(ts)))
            ts = self.pool(self.relu(self.conv2(ts)))
            ts = ts.flatten(1)
            ts = self.relu(self.ts_fc(ts))
            
            # Tabular branch
            tab = self.relu(self.tab_fc1(tab))
            tab = self.relu(self.tab_fc2(tab))
            
            # Combine
            combined = torch.cat([ts, tab], dim=1)
            combined = self.dropout(combined)
            output = self.classifier(combined)
            
            return output.squeeze()


# ============================================
# TRAINING AND EVALUATION
# ============================================

def train_model(model, train_loader, val_loader, criterion, optimizer, 
                device, epochs=EPOCHS, patience=PATIENCE, model_name="model"):
    """Train a model with early stopping."""
    best_val_f1 = 0
    best_model_state = None
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_f1': []}
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch in train_loader:
            if len(batch) == 2:
                x, y = batch
                x, y = x.to(device), y.to(device)
                output = model(x)
            else:
                ts, tab, y = batch
                ts, tab, y = ts.to(device), tab.to(device), y.to(device)
                output = model(ts, tab)
            
            loss = criterion(output, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 2:
                    x, y = batch
                    x, y = x.to(device), y.to(device)
                    output = model(x)
                else:
                    ts, tab, y = batch
                    ts, tab, y = ts.to(device), tab.to(device), y.to(device)
                    output = model(ts, tab)
                
                loss = criterion(output, y)
                val_loss += loss.item()
                
                preds = (torch.sigmoid(output) > 0.5).float()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_f1 = f1_score(all_labels, all_preds, zero_division=0)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_f1)
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: Train Loss={train_loss:.4f}, "
                  f"Val Loss={val_loss:.4f}, Val F1={val_f1:.4f}")
        
        # Early stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history, best_val_f1


def evaluate_model(model, test_loader, device, model_name="model"):
    """Evaluate model and return metrics."""
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 2:
                x, y = batch
                x, y = x.to(device), y.to(device)
                output = model(x)
            else:
                ts, tab, y = batch
                ts, tab, y = ts.to(device), tab.to(device), y.to(device)
                output = model(ts, tab)
            
            probs = torch.sigmoid(output)
            preds = (probs > 0.5).float()
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    metrics = {
        'model': model_name,
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'f1': f1_score(all_labels, all_preds, zero_division=0),
        'mcc': matthews_corrcoef(all_labels, all_preds),
        'auc': roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.5
    }
    
    return metrics


# ============================================
# MAIN EXPERIMENT
# ============================================

def main():
    print("="*60)
    print("PHASE 8: DEEP LEARNING EXPERIMENTS")
    print("="*60)
    print(f"Started at: {datetime.now().isoformat()}")
    
    if not TORCH_AVAILABLE:
        print("\nERROR: PyTorch not available. Please install with:")
        print("  pip install torch")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load data
    print("\n[1/5] Loading data with time series...")
    df, timeseries = load_data_with_timeseries(max_samples=None)  # Set to None for full dataset
    
    if len(df) == 0:
        print("ERROR: No data loaded. Check time series directory.")
        return
    
    # Prepare features
    print("\n[2/5] Preparing features...")
    tabular_features = prepare_tabular_features(df)
    labels = df['has_bug'].values
    
    # Scale tabular features
    scaler = StandardScaler()
    tabular_features = scaler.fit_transform(tabular_features)
    
    print(f"  Time series shape: {timeseries.shape}")
    print(f"  Tabular features shape: {tabular_features.shape}")
    print(f"  Labels: {labels.sum()} positive ({labels.mean()*100:.1f}%)")
    
    # Temporal split (80/20)
    print("\n[3/5] Creating temporal train/val/test split...")
    n = len(labels)
    train_idx = int(n * 0.7)
    val_idx = int(n * 0.85)
    
    ts_train, ts_val, ts_test = timeseries[:train_idx], timeseries[train_idx:val_idx], timeseries[val_idx:]
    tab_train, tab_val, tab_test = tabular_features[:train_idx], tabular_features[train_idx:val_idx], tabular_features[val_idx:]
    y_train, y_val, y_test = labels[:train_idx], labels[train_idx:val_idx], labels[val_idx:]
    
    print(f"  Train: {len(y_train)} samples, {y_train.sum()} positive ({y_train.mean()*100:.1f}%)")
    print(f"  Val: {len(y_val)} samples, {y_val.sum()} positive ({y_val.mean()*100:.1f}%)")
    print(f"  Test: {len(y_test)} samples, {y_test.sum()} positive ({y_test.mean()*100:.1f}%)")
    
    # Create data loaders with class balancing
    pos_weight = (1 - y_train.mean()) / y_train.mean()
    print(f"  Class imbalance weight: {pos_weight:.2f}")
    
    # Weighted sampler for training
    weights = np.where(y_train == 1, pos_weight, 1.0)
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    
    # Time series only datasets
    train_ts_dataset = TimeSeriesDataset(ts_train, y_train)
    val_ts_dataset = TimeSeriesDataset(ts_val, y_val)
    test_ts_dataset = TimeSeriesDataset(ts_test, y_test)
    
    train_ts_loader = DataLoader(train_ts_dataset, batch_size=BATCH_SIZE, sampler=sampler)
    val_ts_loader = DataLoader(val_ts_dataset, batch_size=BATCH_SIZE)
    test_ts_loader = DataLoader(test_ts_dataset, batch_size=BATCH_SIZE)
    
    # Hybrid datasets
    train_hybrid_dataset = HybridDataset(ts_train, tab_train, y_train)
    val_hybrid_dataset = HybridDataset(ts_val, tab_val, y_val)
    test_hybrid_dataset = HybridDataset(ts_test, tab_test, y_test)
    
    train_hybrid_loader = DataLoader(train_hybrid_dataset, batch_size=BATCH_SIZE, sampler=sampler)
    val_hybrid_loader = DataLoader(val_hybrid_dataset, batch_size=BATCH_SIZE)
    test_hybrid_loader = DataLoader(test_hybrid_dataset, batch_size=BATCH_SIZE)
    
    # Loss function with class weighting
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
    
    # Train and evaluate models
    print("\n[4/5] Training deep learning models...")
    results = []
    
    # Model configurations
    models_config = [
        ('CNN-1D', CNN1D(SEQUENCE_LENGTH, HIDDEN_DIM, DROPOUT), train_ts_loader, val_ts_loader, test_ts_loader),
        ('LSTM', LSTMClassifier(1, HIDDEN_DIM, NUM_LAYERS, DROPOUT), train_ts_loader, val_ts_loader, test_ts_loader),
        ('GRU', GRUClassifier(1, HIDDEN_DIM, NUM_LAYERS, DROPOUT), train_ts_loader, val_ts_loader, test_ts_loader),
        ('Transformer', TransformerClassifier(SEQUENCE_LENGTH, HIDDEN_DIM, 4, NUM_LAYERS, DROPOUT), train_ts_loader, val_ts_loader, test_ts_loader),
        ('Hybrid-CNN', HybridModel(SEQUENCE_LENGTH, tabular_features.shape[1], HIDDEN_DIM, DROPOUT), train_hybrid_loader, val_hybrid_loader, test_hybrid_loader),
    ]
    
    for model_name, model, train_loader, val_loader, test_loader in models_config:
        print(f"\n  Training {model_name}...")
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
        
        model, history, best_val_f1 = train_model(
            model, train_loader, val_loader, criterion, optimizer, device,
            epochs=EPOCHS, patience=PATIENCE, model_name=model_name
        )
        
        metrics = evaluate_model(model, test_loader, device, model_name)
        results.append(metrics)
        
        print(f"    Test Results - F1: {metrics['f1']:.4f}, Precision: {metrics['precision']:.4f}, "
              f"Recall: {metrics['recall']:.4f}, MCC: {metrics['mcc']:.4f}, AUC: {metrics['auc']:.4f}")
        
        # Save model
        torch.save(model.state_dict(), OUTPUT_DIR / 'models' / f'{model_name.lower()}.pt')
        
        # Clear memory
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Save results
    print("\n[5/5] Saving results...")
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / 'reports' / 'dl_experiment_results.csv', index=False)
    
    # Display summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(results_df.to_string(index=False))
    
    # Compare to Phase 1 ML baseline
    print("\n" + "="*60)
    print("COMPARISON TO PHASE 1 ML BASELINES")
    print("="*60)
    print("\nPhase 1 Results (from run_fixed.py):")
    print("  XGBoost:    F1=0.394, Precision=0.509, Recall=0.321")
    print("  Random Forest: F1=0.338, Precision=0.469, Recall=0.264")
    
    best_dl = results_df.loc[results_df['f1'].idxmax()]
    print(f"\nBest DL Model: {best_dl['model']}")
    print(f"  F1: {best_dl['f1']:.4f}, Precision: {best_dl['precision']:.4f}, Recall: {best_dl['recall']:.4f}")
    
    if best_dl['f1'] > 0.394:
        print("\n[*] DEEP LEARNING OUTPERFORMS ML BASELINES!")
    else:
        print("\n[*] ML baselines remain competitive (expected for small tabular data)")
    
    print(f"\nFinished at: {datetime.now().isoformat()}")
    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
