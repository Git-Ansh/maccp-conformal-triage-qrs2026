"""
Phase 6: Downstream Alert Analysis

Analyze relationships between primary and downstream alerts.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.data_paths import ALERT_ID_COL


def build_alert_graph(
    alerts_df: pd.DataFrame,
    summary_col: str = 'alert_summary_id',
    related_col: str = 'single_alert_related_summary_id'
) -> Dict[str, List[str]]:
    """
    Build a graph of alert relationships.

    Args:
        alerts_df: DataFrame with alert data
        summary_col: Column containing summary ID
        related_col: Column containing related summary ID

    Returns:
        Dictionary mapping summary_id to list of related alert_ids
    """
    graph = defaultdict(list)

    for _, row in alerts_df.iterrows():
        alert_id = str(row.get(ALERT_ID_COL, ''))
        summary_id = str(row.get(summary_col, ''))
        related_id = str(row.get(related_col, ''))

        if summary_id and summary_id != 'nan':
            graph[summary_id].append(alert_id)

        # Link to related summary if exists
        if related_id and related_id != 'nan' and related_id != summary_id:
            graph[f'related_{related_id}'].append(alert_id)

    # Filter to summaries with multiple alerts
    graph = {k: v for k, v in graph.items() if len(v) > 1}

    print(f"Built alert graph: {len(graph)} alert groups")

    return dict(graph)


def identify_downstream_alerts(
    alerts_df: pd.DataFrame,
    summary_col: str = 'alert_summary_id',
    timestamp_col: str = 'push_timestamp'
) -> pd.DataFrame:
    """
    Identify downstream alerts within each summary group.

    The first alert in a summary (by timestamp) is the primary,
    others are downstream.

    Args:
        alerts_df: DataFrame with alert data
        summary_col: Column containing summary ID
        timestamp_col: Column containing timestamp

    Returns:
        DataFrame with is_primary and is_downstream columns added
    """
    df = alerts_df.copy()

    # Convert timestamp
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')

    # Initialize columns
    df['is_primary'] = False
    df['is_downstream'] = False

    # Group by summary
    for summary_id, group in df.groupby(summary_col):
        if pd.isna(summary_id):
            continue

        # Sort by timestamp
        group = group.sort_values(timestamp_col)
        alert_ids = group.index.tolist()

        if len(alert_ids) > 0:
            # First alert is primary
            df.loc[alert_ids[0], 'is_primary'] = True
            # Rest are downstream
            for aid in alert_ids[1:]:
                df.loc[aid, 'is_downstream'] = True

    n_primary = df['is_primary'].sum()
    n_downstream = df['is_downstream'].sum()
    print(f"Identified {n_primary} primary alerts, {n_downstream} downstream alerts")

    return df


def analyze_downstream_clusters(
    alerts_df: pd.DataFrame,
    cluster_labels: np.ndarray,
    summary_col: str = 'alert_summary_id'
) -> pd.DataFrame:
    """
    Analyze whether downstream alerts cluster with their primary.

    Args:
        alerts_df: DataFrame with alerts
        cluster_labels: Cluster labels for each alert
        summary_col: Column containing summary ID

    Returns:
        DataFrame with analysis results per summary
    """
    df = alerts_df.copy()
    df['cluster'] = cluster_labels

    results = []

    for summary_id, group in df.groupby(summary_col):
        if pd.isna(summary_id) or len(group) < 2:
            continue

        clusters_in_group = group['cluster'].values
        unique_clusters = set(clusters_in_group)

        # Calculate cluster coherence
        most_common_cluster = pd.Series(clusters_in_group).mode()
        if len(most_common_cluster) > 0:
            coherence = (clusters_in_group == most_common_cluster.iloc[0]).mean()
        else:
            coherence = 0

        results.append({
            'summary_id': summary_id,
            'n_alerts': len(group),
            'n_unique_clusters': len(unique_clusters),
            'cluster_coherence': coherence,
            'all_same_cluster': len(unique_clusters) == 1
        })

    results_df = pd.DataFrame(results)

    if len(results_df) > 0:
        print(f"Analyzed {len(results_df)} summaries with multiple alerts")
        print(f"  Average coherence: {results_df['cluster_coherence'].mean():.3f}")
        print(f"  Summaries with all same cluster: {results_df['all_same_cluster'].sum()}")

    return results_df


def compute_downstream_metrics(
    alerts_df: pd.DataFrame,
    feature_cols: List[str]
) -> pd.DataFrame:
    """
    Compute feature similarity between primary and downstream alerts.

    Args:
        alerts_df: DataFrame with is_primary and is_downstream columns
        feature_cols: Feature columns to compare

    Returns:
        DataFrame with similarity metrics per summary
    """
    valid_cols = [c for c in feature_cols if c in alerts_df.columns]

    if not valid_cols:
        print("No valid feature columns found")
        return pd.DataFrame()

    results = []

    for summary_id, group in alerts_df.groupby('alert_summary_id'):
        if pd.isna(summary_id):
            continue

        primary = group[group['is_primary']]
        downstream = group[group['is_downstream']]

        if len(primary) == 0 or len(downstream) == 0:
            continue

        primary_features = primary[valid_cols].values[0]
        downstream_features = downstream[valid_cols].values

        # Compute average distance
        distances = []
        for df_row in downstream_features:
            # Euclidean distance
            dist = np.sqrt(np.nansum((primary_features - df_row) ** 2))
            distances.append(dist)

        results.append({
            'summary_id': summary_id,
            'n_downstream': len(downstream),
            'avg_distance': np.mean(distances),
            'min_distance': np.min(distances),
            'max_distance': np.max(distances)
        })

    return pd.DataFrame(results)


def get_summary_statistics(alerts_df: pd.DataFrame) -> Dict:
    """
    Get summary statistics about alert relationships.

    Args:
        alerts_df: DataFrame with alert data

    Returns:
        Dictionary with statistics
    """
    stats = {
        'total_alerts': len(alerts_df),
        'total_summaries': alerts_df['alert_summary_id'].nunique()
    }

    # Alerts per summary
    alerts_per_summary = alerts_df.groupby('alert_summary_id').size()
    stats['avg_alerts_per_summary'] = alerts_per_summary.mean()
    stats['max_alerts_per_summary'] = alerts_per_summary.max()
    stats['single_alert_summaries'] = (alerts_per_summary == 1).sum()
    stats['multi_alert_summaries'] = (alerts_per_summary > 1).sum()

    # Related summaries
    if 'single_alert_related_summary_id' in alerts_df.columns:
        related = alerts_df['single_alert_related_summary_id'].notna()
        stats['alerts_with_related'] = related.sum()
        stats['related_ratio'] = related.mean()

    return stats


def find_alert_chains(
    alerts_df: pd.DataFrame,
    max_depth: int = 5
) -> List[List[str]]:
    """
    Find chains of related alerts.

    Args:
        alerts_df: DataFrame with alert data
        max_depth: Maximum chain depth

    Returns:
        List of alert chains (each chain is a list of alert_ids)
    """
    # Build adjacency based on related_summary_id
    related_col = 'single_alert_related_summary_id'
    summary_col = 'alert_summary_id'

    if related_col not in alerts_df.columns:
        return []

    # Map summary_id to alerts
    summary_to_alerts = defaultdict(list)
    for _, row in alerts_df.iterrows():
        summary_id = row.get(summary_col)
        alert_id = row.get(ALERT_ID_COL)
        if pd.notna(summary_id):
            summary_to_alerts[str(summary_id)].append(str(alert_id))

    # Build chain starting from each alert
    chains = []
    visited = set()

    for _, row in alerts_df.iterrows():
        alert_id = str(row.get(ALERT_ID_COL))
        if alert_id in visited:
            continue

        chain = [alert_id]
        current_summary = str(row.get(summary_col, ''))
        visited.add(alert_id)

        for _ in range(max_depth):
            # Find related summary
            related_summary = str(row.get(related_col, ''))
            if related_summary == 'nan' or related_summary == current_summary:
                break

            # Add alerts from related summary
            related_alerts = summary_to_alerts.get(related_summary, [])
            new_alerts = [a for a in related_alerts if a not in visited]

            if not new_alerts:
                break

            chain.extend(new_alerts)
            visited.update(new_alerts)
            current_summary = related_summary

        if len(chain) > 1:
            chains.append(chain)

    print(f"Found {len(chains)} alert chains")
    if chains:
        chain_lengths = [len(c) for c in chains]
        print(f"  Avg chain length: {np.mean(chain_lengths):.2f}")
        print(f"  Max chain length: {max(chain_lengths)}")

    return chains
