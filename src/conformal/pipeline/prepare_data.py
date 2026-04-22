"""
prepare_data.py — Shared data preprocessing for dual-gate selective prediction
Creates: 60/20/20 temporal split + canonical label_mapping.json
Run this ONCE before any model training.

Usage:
    python prepare_data.py --data_path /path/to/eclipse_zenodo_2024.csv \
                           --output_dir /path/to/processed/ \
                           --top_k 30 \
                           --dataset_name eclipse
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


# ─── Noise filtering (must match your existing XGBoost pipeline EXACTLY) ───
ECLIPSE_NOISE_RESOLUTIONS = {
    "INVALID", "DUPLICATE", "WONTFIX", "WORKSFORME", "NOT_ECLIPSE"
}

MOZILLA_NOISE_RESOLUTIONS = {
    "INVALID", "DUPLICATE", "WONTFIX", "WORKSFORME", "INCOMPLETE"
}


def filter_noise_eclipse(df: pd.DataFrame) -> pd.DataFrame:
    """Remove noise bugs -- MUST match your cascade S0 filter exactly."""
    before = len(df)
    # Keep only FIXED and VERIFIED resolutions (genuine bugs)
    # Adjust this to match your existing pipeline's filter
    if "resolution" in df.columns:
        df = df[~df["resolution"].fillna("").str.upper().isin(ECLIPSE_NOISE_RESOLUTIONS)]
    print(f"  Noise filter: {before} -> {len(df)} ({before - len(df)} removed)")
    return df


def filter_noise_mozilla(df: pd.DataFrame) -> pd.DataFrame:
    """Remove noise bugs for Mozilla Bugzilla."""
    before = len(df)
    if "resolution" in df.columns:
        df = df[~df["resolution"].fillna("").str.upper().isin(MOZILLA_NOISE_RESOLUTIONS)]
    print(f"  Noise filter: {before} -> {len(df)} ({before - len(df)} removed)")
    return df


def create_temporal_splits(
    df: pd.DataFrame,
    date_col: str,
    train_frac: float = 0.60,
    cal_frac: float = 0.20,
    # test_frac = 1 - train_frac - cal_frac = 0.20
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    3-way temporal split: train (oldest 60%) / calibration (middle 20%) / test (newest 20%).
    Conformal prediction REQUIRES calibration data the model has never seen.
    """
    df = df.sort_values(date_col).reset_index(drop=True)
    n = len(df)
    
    train_end = int(n * train_frac)
    cal_end = int(n * (train_frac + cal_frac))
    
    train_df = df.iloc[:train_end].copy()
    cal_df = df.iloc[train_end:cal_end].copy()
    test_df = df.iloc[cal_end:].copy()
    
    print(f"  Temporal split:")
    print(f"    Train: {len(train_df):,} ({train_df[date_col].min()} -> {train_df[date_col].max()})")
    print(f"    Cal:   {len(cal_df):,} ({cal_df[date_col].min()} -> {cal_df[date_col].max()})")
    print(f"    Test:  {len(test_df):,} ({test_df[date_col].min()} -> {test_df[date_col].max()})")
    
    return train_df, cal_df, test_df


def create_label_mapping(
    train_df: pd.DataFrame,
    component_col: str,
    top_k: int = 30,
) -> dict[str, int]:
    """
    Create canonical label mapping from TRAINING data only.
    Components outside top-K become "Other".
    Sorted alphabetically for determinism.
    
    CRITICAL: Both XGBoost and DeBERTa MUST use this same mapping.
    """
    top_components = (
        train_df[component_col]
        .value_counts()
        .head(top_k)
        .index
        .tolist()
    )
    top_components_sorted = sorted(top_components)
    
    label_map = {comp: i for i, comp in enumerate(top_components_sorted)}
    label_map["Other"] = top_k  # Last class
    
    print(f"  Label mapping: {top_k} components + Other = {top_k + 1} classes")
    print(f"  Components: {top_components_sorted[:5]}... + Other")
    
    return label_map


def apply_label_mapping(
    df: pd.DataFrame,
    component_col: str,
    label_map: dict[str, int],
) -> pd.DataFrame:
    """Map component strings to integer labels. Unknown components -> 'Other'."""
    other_id = max(label_map.values())
    df = df.copy()
    df["label"] = df[component_col].map(label_map).fillna(other_id).astype(int)
    df["component_mapped"] = df[component_col].where(
        df[component_col].isin(label_map), other="Other"
    )
    return df


def build_text_field(
    df: pd.DataFrame,
    summary_col: str = "summary",
    description_col: str = "description",
    max_desc_chars: int = 1500,
) -> pd.DataFrame:
    """
    Combine Summary + Description into a single text field for DeBERTa.
    
    Truncate description to max_desc_chars. DeBERTa tokenizer will further
    truncate to MAX_LENGTH tokens, but pre-truncating saves memory during
    tokenization of 100K+ samples.
    """
    df = df.copy()
    
    summary = df[summary_col].fillna("").astype(str)
    
    if description_col in df.columns:
        desc = df[description_col].fillna("").astype(str).str[:max_desc_chars]
        df["text"] = summary + " [SEP] " + desc
    else:
        # Mozilla may only have summary
        df["text"] = summary
    
    # Report text length stats
    text_lens = df["text"].str.len()
    print(f"  Text field: mean={text_lens.mean():.0f} chars, "
          f"median={text_lens.median():.0f}, max={text_lens.max()}")
    
    return df


def print_class_distribution(df: pd.DataFrame, split_name: str, label_map: dict):
    """Print class distribution for sanity checking."""
    inv_map = {v: k for k, v in label_map.items()}
    counts = df["label"].value_counts().sort_index()
    print(f"\n  {split_name} class distribution (top 5 + Other):")
    for label_id in counts.head(5).index:
        name = inv_map.get(label_id, f"Unknown_{label_id}")
        print(f"    {name}: {counts[label_id]:,} ({counts[label_id]/len(df)*100:.1f}%)")
    other_id = max(label_map.values())
    if other_id in counts.index:
        print(f"    Other: {counts[other_id]:,} ({counts[other_id]/len(df)*100:.1f}%)")
    print(f"    Total: {len(df):,}")


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--data_path", help="Path to raw CSV")
    group.add_argument("--data_dir", help="Directory of CSVs (e.g., eclipse_zenodo_lite/)")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--top_k", type=int, default=30, help="Number of top components")
    parser.add_argument("--dataset_name", default="eclipse",
                        choices=["eclipse", "mozilla_firefox"])
    parser.add_argument("--date_col", default="creation_ts",
                        help="Timestamp column name")
    parser.add_argument("--component_col", default="component",
                        help="Component/category column name")
    parser.add_argument("--summary_col", default="summary")
    parser.add_argument("--description_col", default="description")
    parser.add_argument("--no_other", action="store_true",
                        help="Drop samples outside top-K components instead of mapping to Other")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ─── Load ───
    if args.data_dir:
        data_dir = Path(args.data_dir)
        csv_files = sorted(data_dir.glob("*.csv"))
        print(f"Loading {len(csv_files)} CSVs from {data_dir}/...")
        dfs = []
        for f in csv_files:
            part = pd.read_csv(f, low_memory=False)
            print(f"  {f.stem}: {len(part):,} rows")
            dfs.append(part)
        df = pd.concat(dfs, ignore_index=True)
    else:
        print(f"Loading {args.data_path}...")
        df = pd.read_csv(args.data_path, low_memory=False)

    # Normalize column names to lowercase for consistency
    df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")
    print(f"  Raw: {len(df):,} rows, {len(df.columns)} columns")
    print(f"  Columns: {list(df.columns)}")
    
    # ─── Parse dates ───
    df[args.date_col] = pd.to_datetime(df[args.date_col], errors="coerce")
    df = df.dropna(subset=[args.date_col, args.component_col])
    
    # ─── Filter noise ───
    print("\nFiltering noise...")
    if args.dataset_name == "eclipse":
        df = filter_noise_eclipse(df)
    elif args.dataset_name == "mozilla_firefox":
        df = filter_noise_mozilla(df)
    
    # ─── Temporal split ───
    print("\nCreating temporal splits...")
    train_df, cal_df, test_df = create_temporal_splits(df, args.date_col)
    
    # ─── Label mapping (from TRAINING data only) ───
    print("\nCreating label mapping from training data...")
    label_map = create_label_mapping(train_df, args.component_col, args.top_k)

    if args.no_other:
        # Drop "Other" from mapping and remove unmapped samples
        label_map.pop("Other", None)
        print(f"  --no_other: keeping only top-{args.top_k} components ({len(label_map)} classes)")

    # ─── Apply mapping to all splits ───
    train_df = apply_label_mapping(train_df, args.component_col, label_map)
    cal_df = apply_label_mapping(cal_df, args.component_col, label_map)
    test_df = apply_label_mapping(test_df, args.component_col, label_map)

    if args.no_other:
        for name in ["train", "cal", "test"]:
            df_cur = {"train": train_df, "cal": cal_df, "test": test_df}[name]
            before = len(df_cur)
            df_cur = df_cur[df_cur["component_mapped"] != "Other"].copy()
            if name == "train": train_df = df_cur
            elif name == "cal": cal_df = df_cur
            else: test_df = df_cur
            print(f"    {name}: {before} -> {len(df_cur)} (dropped {before - len(df_cur)} Other)")
    
    # ─── Build text field ───
    print("\nBuilding text fields...")
    train_df = build_text_field(train_df, args.summary_col, args.description_col)
    cal_df = build_text_field(cal_df, args.summary_col, args.description_col)
    test_df = build_text_field(test_df, args.summary_col, args.description_col)
    
    # ─── Print distributions ───
    print_class_distribution(train_df, "Train", label_map)
    print_class_distribution(cal_df, "Calibration", label_map)
    print_class_distribution(test_df, "Test", label_map)
    
    # ─── Check for rare classes ───
    train_counts = train_df["label"].value_counts()
    rare_classes = train_counts[train_counts < 50]
    if len(rare_classes) > 0:
        inv_map = {v: k for k, v in label_map.items()}
        print(f"\n  WARNING: {len(rare_classes)} classes have <50 training samples:")
        for label_id, count in rare_classes.items():
            print(f"    {inv_map.get(label_id, label_id)}: {count}")
        print("  Conformal calibration may be unreliable for these classes.")
    
    cal_counts = cal_df["label"].value_counts()
    rare_cal = cal_counts[cal_counts < 20]
    if len(rare_cal) > 0:
        inv_map = {v: k for k, v in label_map.items()}
        print(f"\n  WARNING: {len(rare_cal)} classes have <20 calibration samples:")
        for label_id, count in rare_cal.items():
            print(f"    {inv_map.get(label_id, label_id)}: {count}")
        print("  Mondrian CP is unreliable for these classes.")
    
    # ─── Save ───
    print(f"\nSaving to {output_dir}/...")
    
    # Label mapping — THE canonical source of truth
    mapping_path = output_dir / "label_mapping.json"
    json.dump(label_map, open(mapping_path, "w"), indent=2)
    print(f"  [OK] label_mapping.json ({len(label_map)} classes)")
    
    # Splits — only the columns both models need
    cols_to_save = ["text", "label", "component_mapped", args.date_col]
    # Add any metadata columns XGBoost might need
    meta_cols = [c for c in df.columns if c in [
        "severity", "priority", "op_sys", "rep_platform",
        "reporter", "product", args.component_col
    ]]
    cols_to_save.extend(meta_cols)
    cols_to_save = [c for c in cols_to_save if c in train_df.columns]
    
    train_df[cols_to_save].to_parquet(output_dir / "train.parquet", index=False)
    cal_df[cols_to_save].to_parquet(output_dir / "cal.parquet", index=False)
    test_df[cols_to_save].to_parquet(output_dir / "test.parquet", index=False)
    
    print(f"  [OK] train.parquet ({len(train_df):,} rows)")
    print(f"  [OK] cal.parquet ({len(cal_df):,} rows)")
    print(f"  [OK] test.parquet ({len(test_df):,} rows)")
    
    # Save dataset config for reproducibility
    config = {
        "dataset_name": args.dataset_name,
        "top_k": args.top_k,
        "num_classes": len(label_map),
        "train_size": len(train_df),
        "cal_size": len(cal_df),
        "test_size": len(test_df),
        "date_range": {
            "train": [str(train_df[args.date_col].min()), str(train_df[args.date_col].max())],
            "cal": [str(cal_df[args.date_col].min()), str(cal_df[args.date_col].max())],
            "test": [str(test_df[args.date_col].min()), str(test_df[args.date_col].max())],
        },
        "class_distribution": {
            "train_other_pct": float((train_df["label"] == args.top_k).mean() * 100),
            "test_other_pct": float((test_df["label"] == args.top_k).mean() * 100),
        }
    }
    json.dump(config, open(output_dir / "dataset_config.json", "w"), indent=2)
    print(f"  [OK] dataset_config.json")
    
    print("\n[DONE] Done. Both XGBoost and DeBERTa must load label_mapping.json from this directory.")


if __name__ == "__main__":
    main()
