"""
End-to-end Cascading Confidence-Gated Triage Pipeline.
Orchestrates Stage 0 -> Stage 1 -> Stage 2 -> Stage 3.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent.resolve()))

from cascade.stages.stage_0_invalid_filter import train_stage_0, predict_stage_0
from cascade.stages.stage_1_disposition import (
    train_stage_1, predict_stage_1, get_cross_validated_predictions,
    STATUS_MERGE, DISPOSITION_CLASSES
)
from cascade.stages.stage_2a_alert_roles import train_stage_2a, predict_stage_2a
from cascade.stages.stage_2b_noise_filter import train_stage_2b, predict_stage_2b
from cascade.stages.stage_3_bug_linkage import train_stage_3, predict_stage_3
from cascade.evaluation.metrics import stage_metrics, end_to_end_metrics, print_stage_report
from cascade.evaluation.selective import coverage_accuracy_curve, workload_reduction


class CascadePipeline:
    """
    Full cascade pipeline: Invalid Filter -> Disposition -> Alert Roles -> Bug Linkage.
    """

    def __init__(self):
        self.stage_0_artifacts = None
        self.stage_1_artifacts = None
        self.stage_2a_artifacts = None
        self.stage_2b_artifacts = None
        self.stage_3a_artifacts = None  # Mode A (with status)
        self.stage_3b_artifacts = None  # Mode B (without status)
        self.cv_predictions = None

    def train(
        self,
        train_summaries: pd.DataFrame,
        train_alerts: pd.DataFrame,
        calibration_method: str = 'isotonic'
    ):
        """
        Train all cascade stages sequentially.
        """
        print("=" * 70)
        print("TRAINING CASCADE PIPELINE")
        print("=" * 70)

        # Stage 0: Invalid Filter
        print("\n[1/6] Training Stage 0: Group Invalid Filter")
        print("-" * 50)
        self.stage_0_artifacts = train_stage_0(train_summaries, calibration_method)

        # Stage 1: Group Disposition (on non-Invalid)
        print("\n[2/6] Training Stage 1: Group Disposition")
        print("-" * 50)
        self.stage_1_artifacts = train_stage_1(train_summaries, calibration_method)

        # Cross-validated predictions for Stage 3
        print("\n[3/6] Getting cross-validated Stage 1 predictions")
        print("-" * 50)
        self.cv_predictions = get_cross_validated_predictions(train_summaries)
        print(f"  Generated CV predictions for {len(self.cv_predictions)} summaries")

        # Stage 2a: Individual Alert Roles
        print("\n[4/6] Training Stage 2a: Individual Alert Roles")
        print("-" * 50)
        self.stage_2a_artifacts = train_stage_2a(
            train_alerts, train_summaries, calibration_method
        )

        # Stage 2b: Noise Filter
        print("\n[5/6] Training Stage 2b: Noise Filter")
        print("-" * 50)
        self.stage_2b_artifacts = train_stage_2b(train_alerts, calibration_method)

        # Stage 3: Bug Linkage (both modes)
        print("\n[6/6] Training Stage 3: Bug Linkage")
        print("-" * 50)
        self.stage_3a_artifacts = train_stage_3(
            train_summaries, self.cv_predictions, mode='A', calibration_method=calibration_method
        )
        self.stage_3b_artifacts = train_stage_3(
            train_summaries, mode='B', calibration_method=calibration_method
        )

        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)

    def predict(
        self,
        test_summaries: pd.DataFrame,
        test_alerts: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run the full cascade on test data.

        Returns:
            (summary_results, alert_results) DataFrames with all predictions
        """
        print("\n" + "=" * 70)
        print("RUNNING CASCADE INFERENCE")
        print("=" * 70)

        # === STAGE 0: Invalid Filter ===
        print("\n[Stage 0] Group Invalid Filter")
        summaries = predict_stage_0(self.stage_0_artifacts, test_summaries)

        s0_invalid = summaries['s0_pred'] == 3
        s0_valid = summaries['s0_pred'] == 0
        s0_uncertain = summaries['s0_pred'] == -1
        print(f"  Invalid:      {s0_invalid.sum()}")
        print(f"  Valid -> S1:   {s0_valid.sum()}")
        print(f"  Uncertain:    {s0_uncertain.sum()}")

        # === STAGE 1: Group Disposition (for valid groups) ===
        print("\n[Stage 1] Group Disposition")
        valid_for_s1 = summaries[s0_valid].copy()
        if len(valid_for_s1) > 0:
            valid_with_s1 = predict_stage_1(self.stage_1_artifacts, valid_for_s1)
            # Merge Stage 1 predictions back
            for col in ['s1_pred', 's1_confidence', 's1_is_confident'] + \
                       [c for c in valid_with_s1.columns if c.startswith('s1_proba_')]:
                summaries.loc[s0_valid, col] = valid_with_s1[col].values

        summaries['s1_pred'] = summaries.get('s1_pred', pd.Series(-1, index=summaries.index)).fillna(-1).astype(int)
        summaries['s1_is_confident'] = summaries.get('s1_is_confident', pd.Series(False, index=summaries.index)).fillna(False)

        s1_confident = summaries['s1_is_confident'] & s0_valid
        s1_uncertain = ~summaries['s1_is_confident'] & s0_valid
        print(f"  Confident:    {s1_confident.sum()}")
        print(f"  Uncertain:    {s1_uncertain.sum()}")

        # === Determine group final status ===
        summaries['group_auto_status'] = -1  # default: Investigating
        summaries.loc[s0_invalid, 'group_auto_status'] = 3  # Invalid
        summaries.loc[s1_confident, 'group_auto_status'] = summaries.loc[s1_confident, 's1_pred']
        summaries['group_is_automated'] = (summaries['group_auto_status'] != -1)

        print(f"\n  Auto-labeled summaries: {summaries['group_is_automated'].sum()}")
        print(f"  Deferred to sheriff:   {(~summaries['group_is_automated']).sum()}")

        # === STAGE 2: Individual Alert Roles ===
        print("\n[Stage 2] Individual Alert Roles")
        alerts = test_alerts.copy()
        alerts['alert_pred_status'] = -1  # default: Investigating
        alerts['alert_is_automated'] = False

        # Stage 2a: For confident groups
        confident_summary_ids = set(summaries[s1_confident]['alert_summary_id'])
        conf_alert_mask = alerts['alert_summary_id'].isin(confident_summary_ids)
        confident_alerts = alerts[conf_alert_mask].copy()
        if len(confident_alerts) > 0:
            confident_alerts_pred = predict_stage_2a(
                self.stage_2a_artifacts, confident_alerts, summaries
            )
            alerts.loc[conf_alert_mask, 'alert_pred_status'] = confident_alerts_pred['s2a_pred'].values
            alerts.loc[conf_alert_mask, 'alert_is_automated'] = confident_alerts_pred['s2a_is_confident'].values
            print(f"  Stage 2a ({len(confident_alerts)} alerts): {confident_alerts_pred['s2a_is_confident'].sum()} confident")

        # Stage 2b: For uncertain/investigating groups
        uncertain_summary_ids = set(
            summaries[s0_uncertain | s1_uncertain]['alert_summary_id']
        )
        unc_alert_mask = alerts['alert_summary_id'].isin(uncertain_summary_ids)
        uncertain_alerts = alerts[unc_alert_mask].copy()
        if len(uncertain_alerts) > 0:
            uncertain_alerts_pred = predict_stage_2b(self.stage_2b_artifacts, uncertain_alerts)
            alerts.loc[unc_alert_mask, 'alert_pred_status'] = uncertain_alerts_pred['s2b_pred'].values
            # Only noise-flagged alerts are "automated" in Stage 2b
            alerts.loc[unc_alert_mask, 'alert_is_automated'] = uncertain_alerts_pred['s2b_is_noise'].values
            noise_count = uncertain_alerts_pred['s2b_is_noise'].sum()
            print(f"  Stage 2b ({len(uncertain_alerts)} alerts): {noise_count} flagged as noise")

        # Invalid group alerts: all auto-labeled as Invalid
        invalid_summary_ids = set(summaries[s0_invalid]['alert_summary_id'])
        invalid_alert_mask = alerts['alert_summary_id'].isin(invalid_summary_ids)
        alerts.loc[invalid_alert_mask, 'alert_pred_status'] = 3
        alerts.loc[invalid_alert_mask, 'alert_is_automated'] = True

        n_auto_alerts = alerts['alert_is_automated'].sum()
        print(f"\n  Total auto-labeled alerts: {n_auto_alerts}/{len(alerts)}")

        # === STAGE 3: Bug Linkage ===
        print("\n[Stage 3] Bug Linkage (has_bug)")

        # Build Stage 1 probabilities for Mode A
        s1_proba_cols = [c for c in summaries.columns if c.startswith('s1_proba_')]
        if s1_proba_cols:
            stage_1_proba = summaries[['alert_summary_id'] + s1_proba_cols].copy()
            # Rename to match expected cv_proba_class_ format
            rename_map = {}
            for col in s1_proba_cols:
                cls_num = col.replace('s1_proba_', '')
                rename_map[col] = f'cv_proba_class_{cls_num}'
            stage_1_proba = stage_1_proba.rename(columns=rename_map)
            if 's1_confidence' in summaries.columns:
                stage_1_proba['cv_pred_confidence'] = summaries['s1_confidence'].values
            if 's1_pred' in summaries.columns:
                stage_1_proba['cv_pred_disposition'] = summaries['s1_pred'].values
        else:
            stage_1_proba = None

        # Mode A for confident groups, Mode B for uncertain
        auto_summaries = summaries[summaries['group_is_automated']].copy()
        deferred_summaries = summaries[~summaries['group_is_automated']].copy()

        if len(auto_summaries) > 0:
            auto_summaries = predict_stage_3(
                self.stage_3a_artifacts, auto_summaries, stage_1_proba
            )
            for col in ['s3_pred', 's3_confidence', 's3_is_confident', 's3_source']:
                summaries.loc[auto_summaries.index, col] = auto_summaries[col].values

        if len(deferred_summaries) > 0:
            deferred_summaries = predict_stage_3(
                self.stage_3b_artifacts, deferred_summaries
            )
            for col in ['s3_pred', 's3_confidence', 's3_is_confident', 's3_source']:
                summaries.loc[deferred_summaries.index, col] = deferred_summaries[col].values

        # Fill defaults
        summaries['s3_pred'] = summaries.get('s3_pred', 0).fillna(0).astype(int)
        summaries['s3_is_confident'] = summaries.get('s3_is_confident', False).fillna(False)

        s3_conf = summaries['s3_is_confident'].sum()
        print(f"  has_bug confident: {s3_conf}/{len(summaries)}")

        return summaries, alerts

    def evaluate(
        self,
        summaries: pd.DataFrame,
        alerts: pd.DataFrame
    ) -> Dict:
        """
        Evaluate the full cascade results.
        """
        print("\n" + "=" * 70)
        print("CASCADE EVALUATION")
        print("=" * 70)

        results = {}

        # Stage 0 evaluation
        true_invalid = (summaries['alert_summary_status'] == 3).astype(int).values
        s0_pred_invalid = (summaries['s0_pred'] == 3).astype(int).values
        s0_confident = np.asarray(summaries['s0_is_confident'].values, dtype=bool) if 's0_is_confident' in summaries else np.ones(len(summaries), dtype=bool)

        s0_metrics = stage_metrics(true_invalid, s0_pred_invalid, s0_confident, "Stage 0: Invalid Filter")
        results['stage_0'] = s0_metrics
        print(print_stage_report(s0_metrics))

        # Stage 1 evaluation (on non-Invalid, non-uncertain summaries)
        s1_mask = summaries['s0_pred'] == 0  # Valid groups passed to Stage 1
        if s1_mask.sum() > 0:
            s1_df = summaries[s1_mask]
            true_status = s1_df['alert_summary_status'].replace(STATUS_MERGE).values
            pred_status = s1_df['s1_pred'].values.astype(int)
            s1_confident = np.asarray(s1_df['s1_is_confident'].values, dtype=bool)

            s1_metrics = stage_metrics(true_status, pred_status, s1_confident, "Stage 1: Group Disposition")
            results['stage_1'] = s1_metrics
            print(print_stage_report(s1_metrics))

        # End-to-end summary metrics
        true_status_all = summaries['alert_summary_status'].replace(STATUS_MERGE).values
        pred_status_all = summaries['group_auto_status'].values.astype(int) if 'group_auto_status' in summaries else np.full(len(summaries), -1)
        is_automated = np.asarray(summaries['group_is_automated'].values, dtype=bool) if 'group_is_automated' in summaries else np.zeros(len(summaries), dtype=bool)

        # has_bug pred uses encoding: 0=uncertain, 1=no bug, 2=has bug
        # Map to binary for comparison: 2->1 (has bug), 1->0 (no bug), 0->keep as uncertain
        has_bug_pred_raw = summaries['s3_pred'].values.astype(int) if 's3_pred' in summaries else None
        has_bug_pred_binary = None
        if has_bug_pred_raw is not None:
            has_bug_pred_binary = np.where(has_bug_pred_raw == 2, 1, 0)
        has_bug_conf = np.asarray(summaries['s3_is_confident'].values, dtype=bool) if 's3_is_confident' in summaries else None

        e2e = end_to_end_metrics(
            true_status_all, pred_status_all, is_automated,
            has_bug_true=summaries['has_bug'].values,
            has_bug_pred=has_bug_pred_binary,
            has_bug_is_confident=has_bug_conf,
        )
        results['end_to_end'] = e2e

        print(f"\n{'='*60}")
        print("  END-TO-END RESULTS")
        print(f"{'='*60}")
        print(f"  Summaries automated:  {e2e['n_summaries_automated']}/{e2e['n_summaries_total']} "
              f"({e2e['summary_automation_rate']:.1%})")
        if 'summary_accuracy_auto' in e2e:
            print(f"  Summary accuracy:     {e2e['summary_accuracy_auto']:.4f}")
        if 'has_bug_accuracy' in e2e:
            print(f"  has_bug accuracy:     {e2e['has_bug_accuracy']:.4f} "
                  f"(coverage: {e2e['has_bug_coverage']:.1%})")

        # Workload reduction
        wl = workload_reduction(e2e['n_summaries_total'], e2e['n_summaries_deferred'])
        results['workload'] = wl
        print(f"\n  Workload reduction:   {wl['reduction_pct']:.1f}%")
        print(f"  Groups saved:         {wl['n_automated']} of {wl['n_total']}")

        return results
