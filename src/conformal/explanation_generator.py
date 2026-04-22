"""
explanation_generator.py -- Human-readable explanations for MACCP prediction sets.

Deterministic rule-based text generation (NOT an LLM).
Designed for the Model-Agreement-Conditioned Conformal Prediction (MACCP) system.

Usage:
    from src.conformal.explanation_generator import TriageExplanationGenerator

    gen = TriageExplanationGenerator(label_map)
    explanation = gen.explain(
        conformal_set=[2, 7, 21],
        deberta_probs=deb_probs[i],
        xgboost_probs=xgb_probs[i],
        deberta_pred=deb_preds[i],
        xgboost_pred=xgb_preds[i],
        agreement=bool(deb_preds[i] == xgb_preds[i]),
        alpha=0.10,
    )
"""

from __future__ import annotations

import numpy as np


class TriageExplanationGenerator:
    """Generate structured human-readable explanations for MACCP conformal prediction sets.

    All output is deterministic rule-based text -- no LLM calls are made.

    Parameters
    ----------
    label_map : dict
        Maps component name (str) to class index (int).
        e.g. {"BIRT": 0, "Core": 1, ...}
    agree_accuracy : float
        Historical accuracy on the calibration set when DeBERTa and XGBoost agree.
        Defaults to 0.857 (from Eclipse MACCP results).
    disagree_accuracy : float
        Historical accuracy when models disagree.
        Defaults to 0.271 (from Eclipse MACCP results).
    enabled : bool
        On/off switch.  When False, explain() returns None immediately (zero overhead).
    """

    ACTION_AUTO_TRIAGE = "AUTO-TRIAGE"
    ACTION_AUTO_CAUTION = "AUTO-TRIAGE WITH CAUTION"
    ACTION_REVIEW_SHORTLIST = "REVIEW SHORTLIST"
    ACTION_REVIEW_NEEDED = "REVIEW NEEDED"
    ACTION_CAREFUL = "NEEDS CAREFUL REVIEW"

    def __init__(
        self,
        label_map: dict,
        agree_accuracy: float = 0.857,
        disagree_accuracy: float = 0.271,
        enabled: bool = True,
    ) -> None:
        self.label_map = label_map
        self.inv_map = {v: k for k, v in label_map.items()}
        self.agree_accuracy = agree_accuracy
        self.disagree_accuracy = disagree_accuracy
        self.enabled = enabled

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _name(self, class_idx: int) -> str:
        """Return class name for a class index."""
        return self.inv_map.get(int(class_idx), f"Class_{class_idx}")

    def _ranked_classes(self, probs: np.ndarray) -> list[tuple[int, int, float]]:
        """Return list of (rank_1based, class_idx, prob) sorted by probability descending."""
        order = np.argsort(-probs)
        return [(rank + 1, int(cls), float(probs[cls])) for rank, cls in enumerate(order)]

    def _confidence_tier(self, deberta_probs: np.ndarray, agreement: bool) -> str:
        max_prob = float(np.max(deberta_probs))
        if max_prob > 0.85 or (agreement and max_prob > 0.70):
            return "HIGH"
        if max_prob >= 0.50:
            return "MEDIUM"
        return "LOW"

    def _reliability_estimate(self, agreement: bool, confidence_tier: str) -> str:
        if agreement:
            return (
                f"High -- both models agree. "
                f"In similar cases, accuracy is ~{self.agree_accuracy:.0%}."
            )
        if confidence_tier == "HIGH":
            return (
                "Moderate -- models disagree but DeBERTa is very confident. "
                "Review recommended."
            )
        return (
            f"Low -- models disagree. "
            f"In similar cases, accuracy is ~{self.disagree_accuracy:.0%}. "
            f"Careful review recommended."
        )

    def _top_candidate(
        self,
        deberta_ranks: list[tuple[int, int, float]],
        agreement: bool,
    ) -> str:
        rank, cls_idx, prob = deberta_ranks[0]
        name = self._name(cls_idx)
        if agreement:
            reason = "Both models agree on this component."
        else:
            reason = "Highest probability from text analysis."
        return f"{name} -- ranked #{rank} by DeBERTa ({prob:.0%} confidence). {reason}"

    def _alternative_candidates(
        self,
        conformal_set: list[int],
        deberta_probs: np.ndarray,
        xgboost_probs: np.ndarray,
        deberta_pred: int,
        xgboost_pred: int,
        agreement: bool,
        alpha: float,
    ) -> list[str]:
        """Build alternative candidate strings for all set members except DeBERTa's top pick."""
        deb_top = int(deberta_pred)
        xgb_top = int(xgboost_pred)
        coverage_pct = int(round((1.0 - alpha) * 100))

        # Build a lookup: class_idx -> (rank, prob) from DeBERTa
        deb_rank_lookup: dict[int, tuple[int, float]] = {}
        for rank, cls_idx, prob in self._ranked_classes(deberta_probs):
            deb_rank_lookup[cls_idx] = (rank, prob)

        # Sort remaining set members by DeBERTa probability descending
        others = [c for c in conformal_set if c != deb_top]
        others_sorted = sorted(others, key=lambda c: deberta_probs[c], reverse=True)

        lines: list[str] = []
        for cls_idx in others_sorted:
            name = self._name(cls_idx)
            deb_rank, deb_prob = deb_rank_lookup.get(cls_idx, (0, float(deberta_probs[cls_idx])))
            xgb_prob = float(xgboost_probs[cls_idx])

            # If this is XGBoost's top pick and models disagree, use XGBoost framing
            if (not agreement) and cls_idx == xgb_top:
                line = (
                    f"{name} -- ranked #1 by XGBoost ({xgb_prob:.0%} confidence). "
                    f"Highest probability from metadata patterns."
                )
            else:
                line = (
                    f"{name} -- ranked #{deb_rank} by DeBERTa ({deb_prob:.0%} confidence). "
                    f"Included to meet {coverage_pct}% coverage guarantee."
                )
            lines.append(line)
        return lines

    def _action_recommendation(
        self, set_size: int, agreement: bool
    ) -> str:
        if set_size == 1:
            if agreement:
                return (
                    "AUTO-TRIAGE -- High confidence, models agree, singleton prediction set."
                )
            return (
                "AUTO-TRIAGE WITH CAUTION -- "
                "Singleton set but models disagree on other candidates."
            )
        if set_size <= 3:
            if agreement:
                return (
                    f"REVIEW SHORTLIST -- Models agree. "
                    f"Choose from {set_size} candidates."
                )
            return (
                f"REVIEW SHORTLIST -- Models disagree. "
                f"Examine the bug report and choose from {set_size} candidates."
            )
        if set_size <= 5:
            return f"REVIEW NEEDED -- {set_size} candidates in prediction set."
        return (
            f"NEEDS CAREFUL REVIEW -- "
            f"Large prediction set ({set_size} candidates). Model is uncertain."
        )

    def _model_reasoning(
        self,
        deberta_probs: np.ndarray,
        xgboost_probs: np.ndarray,
    ) -> dict[str, str]:
        """Summarise top-3 predictions for each model."""
        deb_top3 = self._ranked_classes(deberta_probs)[:3]
        xgb_top3 = self._ranked_classes(xgboost_probs)[:3]

        def _fmt(ranks: list[tuple[int, int, float]], prefix: str) -> str:
            # prefix already includes "points to" / "point to" as appropriate
            if len(ranks) < 3:
                # Fewer than 3 classes -- edge case
                parts = [f"{self._name(c)} ({p:.0%})" for _, c, p in ranks]
                connector = prefix + " " + parts[0]
                if len(parts) > 1:
                    connector += " over " + " and ".join(parts[1:])
                return connector + "."
            _, c1, p1 = ranks[0]
            _, c2, p2 = ranks[1]
            _, c3, p3 = ranks[2]
            return (
                f"{prefix} {self._name(c1)} ({p1:.0%}) "
                f"over {self._name(c2)} ({p2:.0%}) "
                f"and {self._name(c3)} ({p3:.0%})."
            )

        return {
            "deberta": _fmt(deb_top3, "Text analysis points to"),
            "xgboost": _fmt(xgb_top3, "Metadata patterns point to"),
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def explain(
        self,
        conformal_set: list[int],
        deberta_probs: np.ndarray,
        xgboost_probs: np.ndarray,
        deberta_pred: int,
        xgboost_pred: int,
        agreement: bool,
        alpha: float,
    ) -> dict | None:
        """Generate a structured explanation for one MACCP prediction.

        Parameters
        ----------
        conformal_set : list[int]
            Class indices included in the conformal prediction set.
        deberta_probs : np.ndarray
            Probability vector over all classes from DeBERTa.
        xgboost_probs : np.ndarray
            Probability vector over all classes from XGBoost.
        deberta_pred : int
            DeBERTa's argmax prediction (class index).
        xgboost_pred : int
            XGBoost's argmax prediction (class index).
        agreement : bool
            True when deberta_pred == xgboost_pred.
        alpha : float
            Conformal alpha level used (e.g. 0.10 for 90% coverage).

        Returns
        -------
        dict or None
            Explanation dict, or None if generator is disabled.
        """
        if not self.enabled:
            return None

        deberta_probs = np.asarray(deberta_probs, dtype=float)
        xgboost_probs = np.asarray(xgboost_probs, dtype=float)
        deberta_pred = int(deberta_pred)
        xgboost_pred = int(xgboost_pred)
        agreement = bool(agreement)

        set_size = len(conformal_set)
        confidence_tier = self._confidence_tier(deberta_probs, agreement)
        deb_ranks = self._ranked_classes(deberta_probs)

        prediction_set_names = [self._name(c) for c in conformal_set]

        return {
            "prediction_set": prediction_set_names,
            "set_size": set_size,
            "agreement_status": "AGREE" if agreement else "DISAGREE",
            "confidence_tier": confidence_tier,
            "reliability_estimate": self._reliability_estimate(agreement, confidence_tier),
            "top_candidate": self._top_candidate(deb_ranks, agreement),
            "alternative_candidates": self._alternative_candidates(
                conformal_set,
                deberta_probs,
                xgboost_probs,
                deberta_pred,
                xgboost_pred,
                agreement,
                alpha,
            ),
            "action_recommendation": self._action_recommendation(set_size, agreement),
            "model_reasoning": self._model_reasoning(deberta_probs, xgboost_probs),
        }

    def generate_summary_stats(self, explanations: list[dict]) -> dict:
        """Compute summary statistics across a list of explanation dicts.

        Parameters
        ----------
        explanations : list[dict]
            List of dicts returned by explain().  None values are ignored.

        Returns
        -------
        dict
            Aggregated statistics.
        """
        valid = [e for e in explanations if e is not None]
        total = len(valid)
        if total == 0:
            return {
                "total": 0,
                "action_distribution": {},
                "agreement_distribution": {},
                "confidence_distribution": {},
                "avg_set_size": 0.0,
                "different_top_candidates": 0.0,
            }

        # Action distribution.
        # Matching order: ACTION_AUTO_CAUTION must precede ACTION_AUTO_TRIAGE because
        # "AUTO-TRIAGE WITH CAUTION".startswith("AUTO-TRIAGE") is True.
        # Output order follows the spec (AUTO-TRIAGE first).
        _match_order = [
            self.ACTION_AUTO_CAUTION,
            self.ACTION_AUTO_TRIAGE,
            self.ACTION_REVIEW_SHORTLIST,
            self.ACTION_REVIEW_NEEDED,
            self.ACTION_CAREFUL,
        ]
        _spec_order = [
            self.ACTION_AUTO_TRIAGE,
            self.ACTION_AUTO_CAUTION,
            self.ACTION_REVIEW_SHORTLIST,
            self.ACTION_REVIEW_NEEDED,
            self.ACTION_CAREFUL,
        ]
        _counts: dict[str, int] = {k: 0 for k in _match_order}
        for e in valid:
            rec = e["action_recommendation"]
            for k in _match_order:
                if rec.startswith(k):
                    _counts[k] += 1
                    break
        action_dist: dict[str, int] = {k: _counts[k] for k in _spec_order}

        agree_dist = {"AGREE": 0, "DISAGREE": 0}
        conf_dist = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
        set_sizes: list[int] = []
        n_different_top: int = 0

        for e in valid:
            agree_dist[e["agreement_status"]] = agree_dist.get(e["agreement_status"], 0) + 1
            conf_dist[e["confidence_tier"]] = conf_dist.get(e["confidence_tier"], 0) + 1
            set_sizes.append(e["set_size"])

            # "different top candidates" = DeBERTa and XGBoost disagree on top pick
            # We detect this by checking for "ranked #1 by XGBoost" in alternative_candidates
            if any("ranked #1 by XGBoost" in c for c in e["alternative_candidates"]):
                n_different_top += 1
            # Also count when agreement_status is DISAGREE (XGBoost top may equal DeBERTa set pick)
            elif e["agreement_status"] == "DISAGREE":
                # XGBoost top pick differs from DeBERTa top pick by definition when disagree
                n_different_top += 1

        return {
            "total": total,
            "action_distribution": action_dist,
            "agreement_distribution": agree_dist,
            "confidence_distribution": conf_dist,
            "avg_set_size": float(np.mean(set_sizes)),
            "different_top_candidates": float(n_different_top / total),
        }
