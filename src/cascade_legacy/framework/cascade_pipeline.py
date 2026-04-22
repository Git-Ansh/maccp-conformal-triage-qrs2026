"""
Generalizable Confidence-Gated Cascade Pipeline.

Chains multiple ConfidenceStages with configurable routing.
Each stage processes a subset of inputs, and its confident outputs
are routed to the next stage or to a terminal decision.

To apply to a new CI system:
1. Define stages (classes, features, target accuracy)
2. Define routing rules (which output goes where)
3. Provide labeled training data
4. Call pipeline.fit() then pipeline.predict()

The framework handles calibration, threshold tuning, and routing automatically.
No code changes needed -- only configuration.

Bug fixes applied:
  H1: Explicit final_stage_name/final_decision tracking per row
  H3: Vectorized routing (boolean masks instead of per-row get_loc)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable, Tuple, Any
from dataclasses import dataclass, field

from cascade.framework.confidence_stage import ConfidenceStage


@dataclass
class StageConfig:
    """Configuration for a single cascade stage."""
    name: str
    classes: Dict[int, str]
    target_accuracy: float = 0.85
    # Columns to use as features (extracted from DataFrame)
    feature_columns: Optional[List[str]] = None
    # Column containing the target label
    target_column: str = 'status'
    # Label merge map (e.g., {8: 4} to merge Backedout into Ack)
    label_merge: Optional[Dict[int, int]] = None
    # Filter function: given DataFrame, return filtered DataFrame
    # Used to select which rows this stage trains/predicts on
    input_filter: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None
    # Routing: class_code -> action
    # 'terminal' = final decision, 'defer' = human review, 'next' = pass to next stage
    routing: Dict[int, str] = field(default_factory=dict)
    # Custom model (optional, default: XGBoost/RF)
    model: Optional[Any] = None
    # Whether to scale features
    scale: bool = True
    # Output column name prefix
    output_prefix: Optional[str] = None
    # Optional text column for TF-IDF features (A2)
    text_column: Optional[str] = None
    # Calibration method override (default: 'auto')
    calibration_method: str = 'auto'
    # Min threshold samples override
    min_threshold_samples: int = 20


class GeneralCascade:
    """
    A configurable cascade of confidence-gated classification stages.

    Architecture:
        Input -> Stage 0 -> {confident: route, uncertain: defer}
                             |
                Stage 1 -> {confident: route, uncertain: defer}
                             |
                Stage N -> {confident: terminal, uncertain: defer}

    Each stage:
    - Trains a calibrated classifier
    - Tunes per-class confidence thresholds via OOF
    - At inference: routes confident predictions, defers uncertain ones

    H1 fix: Tracks final_stage_name and final_decision per row explicitly.
    H3 fix: Vectorized routing with boolean masks (O(C*K) instead of O(N^2)).
    """

    def __init__(
        self,
        stages: List[StageConfig],
        random_state: int = 42,
        defer_label: int = -1,
    ):
        """
        Args:
            stages: Ordered list of stage configurations
            random_state: Random seed for reproducibility
            defer_label: Label used for deferred predictions
        """
        self.stage_configs = stages
        self.random_state = random_state
        self.defer_label = defer_label
        self._stages: Dict[str, ConfidenceStage] = {}
        self._is_fitted = False

    def fit(
        self,
        df: pd.DataFrame,
        n_cv_folds: int = 5,
    ) -> 'GeneralCascade':
        """
        Train all cascade stages sequentially.

        Args:
            df: Training DataFrame with features and labels
            n_cv_folds: Number of CV folds

        Returns:
            self
        """
        print("=" * 70)
        print("TRAINING GENERAL CASCADE PIPELINE")
        print("=" * 70)

        for i, config in enumerate(self.stage_configs):
            print(f"\n[{i+1}/{len(self.stage_configs)}] Training: {config.name}")
            print("-" * 50)

            # Filter training data for this stage
            stage_df = df.copy()
            if config.input_filter is not None:
                stage_df = config.input_filter(stage_df)

            if len(stage_df) == 0:
                print(f"  WARNING: No training data for stage '{config.name}'")
                continue

            # Apply label merge
            target_col = config.target_column
            if config.label_merge:
                stage_df = stage_df.copy()
                stage_df[target_col] = stage_df[target_col].replace(config.label_merge)

            # Filter to valid classes
            valid = stage_df[target_col].isin(config.classes.keys())
            stage_df = stage_df[valid].copy()

            if len(stage_df) == 0:
                print(f"  WARNING: No valid samples for stage '{config.name}'")
                continue

            # Extract features and target
            feature_cols = config.feature_columns
            if feature_cols is None:
                # Auto-detect: all numeric columns except target
                feature_cols = [
                    c for c in stage_df.select_dtypes(include=[np.number]).columns
                    if c != target_col
                ]

            available_cols = [c for c in feature_cols if c in stage_df.columns]
            X = stage_df[available_cols].fillna(0).values
            y = stage_df[target_col].values

            # A2: Extract text data if configured
            text_data = None
            if config.text_column and config.text_column in stage_df.columns:
                text_data = stage_df[config.text_column].fillna('').values

            # Create and fit stage
            stage = ConfidenceStage(
                name=config.name,
                classes=config.classes,
                target_accuracy=config.target_accuracy,
                calibration_method=config.calibration_method,
                n_cv_folds=n_cv_folds,
                random_state=self.random_state,
                model=config.model,
                defer_label=self.defer_label,
                min_threshold_samples=config.min_threshold_samples,
            )
            stage.fit(X, y, feature_names=available_cols, scale=config.scale,
                      text_data=text_data)
            self._stages[config.name] = stage

        self._is_fitted = True
        print("\n" + "=" * 70)
        print("CASCADE TRAINING COMPLETE")
        print("=" * 70)
        return self

    def predict(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Run the full cascade on input data.

        For each row:
        1. Stage 0 processes it
        2. If confident -> route per config (terminal or next stage)
        3. If uncertain -> defer (label = defer_label)
        4. Repeat for subsequent stages on routed rows

        H1: Tracks final_stage_name and final_decision per row.
        H3: Vectorized routing with boolean masks.

        Args:
            df: Input DataFrame with feature columns

        Returns:
            DataFrame with added prediction columns per stage,
            plus cascade_final_stage, cascade_final_pred, cascade_is_automated.
        """
        if not self._is_fitted:
            raise RuntimeError("Pipeline not fitted. Call fit() first.")

        result_df = df.copy()
        n_rows = len(df)

        # H1: Track final decision per row explicitly
        final_stage = np.full(n_rows, '', dtype=object)
        final_pred = np.full(n_rows, self.defer_label)
        final_conf = np.zeros(n_rows)

        # Track which rows are still active (not yet terminally decided)
        active_mask = np.ones(n_rows, dtype=bool)

        # H3: Build position mapper once for index->position conversion
        idx_to_pos = pd.Series(np.arange(n_rows), index=result_df.index)

        for i, config in enumerate(self.stage_configs):
            stage = self._stages.get(config.name)
            if stage is None:
                continue

            prefix = config.output_prefix or config.name

            # Initialize columns
            result_df[f'{prefix}_pred'] = self.defer_label
            result_df[f'{prefix}_confidence'] = 0.0
            result_df[f'{prefix}_is_confident'] = False

            # Apply input filter to determine which active rows this stage sees
            if config.input_filter is not None:
                active_df = result_df[active_mask].copy()
                try:
                    filtered = config.input_filter(active_df)
                    stage_indices = filtered.index
                except Exception:
                    stage_indices = active_df.index
            else:
                stage_indices = result_df[active_mask].index

            if len(stage_indices) == 0:
                continue

            # Extract features
            feature_cols = config.feature_columns
            if feature_cols is None:
                feature_cols = stage.feature_names or []

            available_cols = [c for c in feature_cols if c in result_df.columns]
            X = result_df.loc[stage_indices, available_cols].fillna(0).values

            # A2: Extract text data if configured
            text_data = None
            if config.text_column and config.text_column in result_df.columns:
                text_data = result_df.loc[stage_indices, config.text_column].fillna('').values

            # Predict
            preds = stage.predict(X, text_data=text_data)

            # Store results
            result_df.loc[stage_indices, f'{prefix}_pred'] = preds['class']
            result_df.loc[stage_indices, f'{prefix}_confidence'] = preds['confidence']
            result_df.loc[stage_indices, f'{prefix}_is_confident'] = preds['is_confident']

            # H3: Vectorized routing (replaces per-row loop with get_loc)
            stage_positions = idx_to_pos[stage_indices].values
            confident_mask = preds['is_confident']
            uncertain_mask = ~confident_mask
            pred_classes = preds['class']

            # Handle uncertain items
            defer_route = config.routing.get(self.defer_label, 'defer')
            if defer_route == 'defer' and uncertain_mask.any():
                uncertain_pos = stage_positions[uncertain_mask]
                active_mask[uncertain_pos] = False

            # Handle confident items per class routing
            for class_code in config.classes:
                route = config.routing.get(class_code, 'terminal')
                class_match = confident_mask & (pred_classes == class_code)
                if not class_match.any():
                    continue

                matched_pos = stage_positions[class_match]

                if route == 'terminal':
                    active_mask[matched_pos] = False
                    # H1: Record terminal decision
                    final_stage[matched_pos] = config.name
                    final_pred[matched_pos] = class_code
                    final_conf[matched_pos] = preds['confidence'][class_match]
                elif route == 'next':
                    # H1: Record intermediate decision (may be overwritten by later stage)
                    final_stage[matched_pos] = config.name
                    final_pred[matched_pos] = class_code
                    final_conf[matched_pos] = preds['confidence'][class_match]
                    # Stays active for next stage

        # Items still active after all stages -> deferred
        # (No stage made a terminal decision)

        # H1: Add explicit tracking columns
        result_df['cascade_final_stage'] = final_stage
        result_df['cascade_final_pred'] = final_pred
        result_df['cascade_final_confidence'] = final_conf

        # H1: cascade_is_automated = has a terminal confident prediction
        # (not just OR of all stage is_confident flags)
        result_df['cascade_is_automated'] = (final_pred != self.defer_label)

        return result_df

    def evaluate(
        self,
        df: pd.DataFrame,
        predictions: pd.DataFrame,
        true_label_column: str = 'status',
        label_merge: Optional[Dict[int, int]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate cascade performance against ground truth.

        Args:
            df: Original data with true labels
            predictions: Output from predict()
            true_label_column: Column with true labels
            label_merge: Optional merge map for true labels

        Returns:
            Dict of evaluation metrics including majority baseline.
        """
        true = df[true_label_column].values.copy()
        if label_merge:
            true = pd.Series(true).replace(label_merge).values

        results = {}

        # Majority baseline (D1: always report)
        majority_class = pd.Series(true).mode().iloc[0]
        majority_acc = (true == majority_class).mean()
        results['majority_baseline'] = {
            'accuracy': float(majority_acc),
            'majority_class': int(majority_class),
        }

        # Per-stage evaluation
        for config in self.stage_configs:
            prefix = config.output_prefix or config.name
            pred_col = f'{prefix}_pred'
            conf_col = f'{prefix}_is_confident'

            if pred_col not in predictions.columns:
                continue

            pred = predictions[pred_col].values
            is_conf = predictions[conf_col].values.astype(bool)

            # Apply stage's label merge to true labels for comparison
            stage_true = true.copy()
            if config.label_merge:
                stage_true = pd.Series(stage_true).replace(config.label_merge).values

            n_conf = is_conf.sum()
            stage_result = {
                'n_total': len(true),
                'n_confident': int(n_conf),
                'coverage': n_conf / len(true) if len(true) > 0 else 0,
            }

            if n_conf > 0:
                stage_result['accuracy'] = (
                    stage_true[is_conf] == pred[is_conf]
                ).mean()

            results[config.name] = stage_result

        # End-to-end evaluation using final decisions (H1)
        final_pred = predictions['cascade_final_pred'].values
        final_conf = predictions['cascade_final_confidence'].values
        is_automated = predictions['cascade_is_automated'].values.astype(bool)

        # Apply the most inclusive merge map for e2e comparison
        e2e_true = true.copy()
        for config in self.stage_configs:
            if config.label_merge:
                e2e_true = pd.Series(e2e_true).replace(config.label_merge).values

        # Coverage-accuracy curve at various thresholds
        results['end_to_end'] = {}
        for t in [0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]:
            mask = is_automated & (final_conf >= t)
            n = mask.sum()
            if n > 0:
                acc = (e2e_true[mask] == final_pred[mask]).mean()
                results['end_to_end'][f't_{t:.2f}'] = {
                    'accuracy': float(acc),
                    'coverage': float(n / len(true)),
                    'n_automated': int(n),
                    'n_deferred': int(len(true) - n),
                    'accuracy_lift': float(acc - majority_acc),
                }

        # Default operating point (all automated items)
        if is_automated.any():
            results['end_to_end']['operating_point'] = {
                'accuracy': float((e2e_true[is_automated] == final_pred[is_automated]).mean()),
                'coverage': float(is_automated.mean()),
                'n_automated': int(is_automated.sum()),
                'n_deferred': int((~is_automated).sum()),
            }

        return results

    def apply_llm_rescue(
        self,
        predictions: pd.DataFrame,
        llm_classify_fn: Callable,
        text_column: str,
        confidence_threshold: float = 0.6,
    ) -> pd.DataFrame:
        """
        Apply LLM rescue stage to deferred items from XGBoost cascade.

        Takes items where cascade_is_automated=False (deferred) and runs them
        through an LLM classifier with consistency sampling. If the LLM is
        confident, the item is rescued (automated). Otherwise, it stays deferred
        with rich context for human review.

        Args:
            predictions: Output from predict() with cascade columns
            llm_classify_fn: Function(texts: List[str]) -> List[Dict]
                Each dict has 'prediction', 'confidence', 'all_predictions',
                'reasoning'.
            text_column: Column name containing text to send to LLM
            confidence_threshold: Min LLM agreement rate to automate

        Returns:
            Updated predictions DataFrame with LLM rescue columns:
            - llm_pred: LLM prediction for deferred items
            - llm_confidence: LLM consistency-based confidence
            - llm_rescued: True if LLM confidently classified a deferred item
            - cascade_final_pred_with_llm: Updated final prediction
            - cascade_is_automated_with_llm: Updated automation flag
        """
        result = predictions.copy()

        # Initialize LLM columns
        result['llm_pred'] = self.defer_label
        result['llm_confidence'] = 0.0
        result['llm_rescued'] = False
        result['llm_reasoning'] = ''

        # Find deferred items
        deferred_mask = ~result['cascade_is_automated'].astype(bool)
        n_deferred = deferred_mask.sum()

        if n_deferred == 0:
            print("  No deferred items for LLM rescue")
            result['cascade_final_pred_with_llm'] = result['cascade_final_pred']
            result['cascade_is_automated_with_llm'] = result['cascade_is_automated']
            return result

        print(f"  LLM rescue: processing {n_deferred} deferred items...")

        # Get text for deferred items
        if text_column not in result.columns:
            print(f"  WARNING: text column '{text_column}' not found, skipping LLM rescue")
            result['cascade_final_pred_with_llm'] = result['cascade_final_pred']
            result['cascade_is_automated_with_llm'] = result['cascade_is_automated']
            return result

        deferred_texts = result.loc[deferred_mask, text_column].fillna('').tolist()

        # Run LLM classifier on deferred items
        llm_results = llm_classify_fn(deferred_texts)

        # Apply results
        deferred_indices = result[deferred_mask].index
        for idx, llm_result in zip(deferred_indices, llm_results):
            pred = llm_result.get('prediction', self.defer_label)
            conf = llm_result.get('confidence', 0.0)
            reasoning = llm_result.get('reasoning', '')

            result.loc[idx, 'llm_pred'] = pred
            result.loc[idx, 'llm_confidence'] = conf
            result.loc[idx, 'llm_reasoning'] = reasoning

            if conf >= confidence_threshold:
                result.loc[idx, 'llm_rescued'] = True

        n_rescued = result['llm_rescued'].sum()
        rescue_rate = n_rescued / n_deferred if n_deferred > 0 else 0
        print(f"  LLM rescued: {n_rescued}/{n_deferred} ({rescue_rate:.1%})")

        # Build updated final predictions
        result['cascade_final_pred_with_llm'] = result['cascade_final_pred'].copy()
        result['cascade_is_automated_with_llm'] = result['cascade_is_automated'].copy()

        rescued_mask = result['llm_rescued'].astype(bool)
        if rescued_mask.any():
            result.loc[rescued_mask, 'cascade_final_pred_with_llm'] = result.loc[rescued_mask, 'llm_pred']
            result.loc[rescued_mask, 'cascade_is_automated_with_llm'] = True

        total_automated = result['cascade_is_automated_with_llm'].astype(bool).sum()
        print(f"  Total automated: {total_automated}/{len(result)} "
              f"({total_automated/len(result):.1%}) "
              f"[was {(~deferred_mask).sum()}/{len(result)}]")

        return result

    def get_stage(self, name: str) -> Optional[ConfidenceStage]:
        """Get a fitted stage by name."""
        return self._stages.get(name)

    def print_evaluation(self, eval_results: Dict[str, Any]):
        """Pretty-print evaluation results."""
        print("\n" + "=" * 70)
        print("CASCADE EVALUATION")
        print("=" * 70)

        # Majority baseline
        if 'majority_baseline' in eval_results:
            mb = eval_results['majority_baseline']
            print(f"\n  Majority baseline: {mb['accuracy']:.1%} "
                  f"(class {mb['majority_class']})")

        for config in self.stage_configs:
            if config.name in eval_results:
                r = eval_results[config.name]
                print(f"\n  {config.name}:")
                print(f"    Confident: {r['n_confident']}/{r['n_total']} "
                      f"({r['coverage']:.1%})")
                if 'accuracy' in r:
                    print(f"    Accuracy:  {r['accuracy']:.1%}")

        if 'end_to_end' in eval_results:
            print(f"\n  End-to-end coverage-accuracy curve:")
            print(f"  {'Threshold':>10} {'Coverage':>10} {'Accuracy':>10} "
                  f"{'Lift':>8} {'Automated':>10}")
            print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*10}")
            for key, val in sorted(eval_results['end_to_end'].items()):
                if key == 'operating_point':
                    continue
                t = key.replace('t_', '')
                lift = val.get('accuracy_lift', 0)
                print(f"  {t:>10} {val['coverage']:>10.1%} "
                      f"{val['accuracy']:>10.1%} {lift:>+8.1%} "
                      f"{val['n_automated']:>10}")

            if 'operating_point' in eval_results['end_to_end']:
                op = eval_results['end_to_end']['operating_point']
                print(f"\n  Operating point: {op['accuracy']:.1%} accuracy, "
                      f"{op['coverage']:.1%} coverage "
                      f"({op['n_automated']} automated, "
                      f"{op['n_deferred']} deferred)")
