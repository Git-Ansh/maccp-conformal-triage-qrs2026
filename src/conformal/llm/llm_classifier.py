"""
LLM-based classifier with confidence estimation via consistency sampling.

Replaces or complements XGBoost ConfidenceStage for text-heavy tasks.
Uses few-shot prompting with DeepSeek V3 via Fireworks AI.

Confidence estimation:
  - Run the LLM N times with temperature > 0 (consistency sampling)
  - Agreement rate = confidence (e.g., 4/5 agree -> 0.80 confidence)
  - Same confidence gating as ConfidenceStage: defer when uncertain

This is NOT a drop-in replacement for ConfidenceStage (no fit/predict sklearn
interface). Instead it's a standalone classifier evaluated alongside XGBoost.
"""

import json
import numpy as np
import pandas as pd
from collections import Counter
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from conformal.llm.fireworks_client import FireworksClient


class LLMClassifier:
    """
    Few-shot LLM classifier with consistency-based confidence.

    Usage:
        clf = LLMClassifier(
            task_description="Predict the Eclipse bug component",
            class_names=["UI", "Core", "Platform", ...],
        )
        clf.set_examples(train_texts, train_labels, n_examples=5)
        results = clf.classify_batch(test_texts, n_samples=5)
        # results = list of {prediction, confidence, all_predictions, ...}
    """

    def __init__(
        self,
        task_description: str,
        class_names: List[str],
        n_consistency: int = 5,
        temperature: float = 0.7,
        client: Optional[FireworksClient] = None,
    ):
        """
        Args:
            task_description: What the classifier does (for system prompt)
            class_names: Valid class labels
            n_consistency: Number of consistency samples for confidence
            temperature: Sampling temperature for consistency
            client: FireworksClient instance (created if not provided)
        """
        self.task_description = task_description
        self.class_names = class_names
        self.n_consistency = n_consistency
        self.temperature = temperature
        self.client = client or FireworksClient()

        self._examples = []
        self._system_prompt = None
        self._build_system_prompt()

    def _build_system_prompt(self):
        """Build the system prompt with task description and valid classes."""
        class_list = ", ".join(self.class_names)
        self._system_prompt = (
            f"{self.task_description}\n\n"
            f"Valid classes: [{class_list}]\n\n"
            f"You MUST respond with ONLY a JSON object, nothing else. "
            f"No explanation, no reasoning, no preamble. Just the JSON:\n"
            f'{{"class": "<one of the valid classes>"}}'
        )

    def set_examples(
        self,
        texts: List[str],
        labels: List[str],
        n_examples: int = 5,
        strategy: str = "diverse",
    ):
        """
        Set few-shot examples from training data.

        Args:
            texts: Training text samples
            labels: Training labels
            n_examples: How many examples to include in prompt
            strategy: 'diverse' (one per class) or 'random'
        """
        if strategy == "diverse":
            # Pick one example per class (up to n_examples)
            seen = set()
            examples = []
            for text, label in zip(texts, labels):
                if label not in seen and len(examples) < n_examples:
                    examples.append({"text": text, "label": label})
                    seen.add(label)
            # Fill remaining with random if needed
            if len(examples) < n_examples:
                remaining = [
                    {"text": t, "label": l}
                    for t, l in zip(texts, labels)
                    if l not in seen
                ]
                np.random.shuffle(remaining)
                examples.extend(remaining[: n_examples - len(examples)])
        else:
            idx = np.random.choice(len(texts), size=min(n_examples, len(texts)),
                                   replace=False)
            examples = [{"text": texts[i], "label": labels[i]} for i in idx]

        self._examples = examples

    def _build_prompt(self, text: str) -> str:
        """Build the few-shot prompt for a single sample."""
        parts = []

        # Few-shot examples
        if self._examples:
            parts.append("Here are examples:\n")
            for i, ex in enumerate(self._examples, 1):
                parts.append(f"Example {i}:")
                parts.append(f"  Text: {ex['text'][:200]}")
                parts.append(f"  Class: {ex['label']}")
                parts.append("")

        # Current sample
        parts.append("Now classify this:\n")
        parts.append(f"Text: {text[:500]}")

        return "\n".join(parts)

    def classify_single(
        self,
        text: str,
        n_samples: Optional[int] = None,
    ) -> Dict:
        """
        Classify a single text with consistency-based confidence.

        Args:
            text: Input text to classify
            n_samples: Override n_consistency for this call

        Returns:
            Dict with:
                - prediction: most common predicted class
                - confidence: agreement rate (0.0 to 1.0)
                - all_predictions: list of all N predictions
                - reasoning: reasoning from the majority prediction
        """
        n = n_samples or self.n_consistency
        prompt = self._build_prompt(text)

        if n == 1:
            # Single deterministic call
            responses = [self.client.chat(
                prompt=prompt,
                system=self._system_prompt,
                temperature=0.0,
                max_tokens=1000,
            )]
        else:
            # Consistency sampling
            responses = self.client.chat_n(
                prompt=prompt,
                system=self._system_prompt,
                n=n,
                temperature=self.temperature,
                max_tokens=1000,
            )

        # Parse all responses
        predictions = []
        reasonings = []
        for resp in responses:
            parsed = self._parse_response(resp)
            predictions.append(parsed["class"])
            reasonings.append(parsed.get("reasoning", ""))

        # Majority vote
        counter = Counter(predictions)
        majority_class, majority_count = counter.most_common(1)[0]
        confidence = majority_count / len(predictions)

        # Get reasoning from a majority prediction
        majority_reasoning = ""
        for pred, reason in zip(predictions, reasonings):
            if pred == majority_class and reason:
                majority_reasoning = reason
                break

        return {
            "prediction": majority_class,
            "confidence": confidence,
            "all_predictions": predictions,
            "reasoning": majority_reasoning,
            "n_samples": len(predictions),
            "n_unique": len(counter),
        }

    def classify_batch(
        self,
        texts: List[str],
        n_samples: Optional[int] = None,
        progress_interval: int = 50,
    ) -> List[Dict]:
        """
        Classify a batch of texts.

        Args:
            texts: List of input texts
            n_samples: Override n_consistency
            progress_interval: Print progress every N items

        Returns:
            List of classification result dicts
        """
        results = []
        for i, text in enumerate(texts):
            result = self.classify_single(text, n_samples=n_samples)
            results.append(result)

            if (i + 1) % progress_interval == 0:
                stats = self.client.stats()
                print(f"  Classified {i+1}/{len(texts)} "
                      f"(API calls: {stats['api_calls']}, "
                      f"cache: {stats['cache_hits']}, "
                      f"tokens: {stats['total_tokens']:,})")

        return results

    def _parse_response(self, response: str) -> Dict:
        """Parse LLM response into structured prediction."""
        # Try JSON parsing
        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                parsed = json.loads(response[start:end])
                predicted = parsed.get("class", "").strip()
                # Validate against known classes
                if predicted in self.class_names:
                    return parsed
                # Try case-insensitive match
                lower_map = {c.lower(): c for c in self.class_names}
                if predicted.lower() in lower_map:
                    parsed["class"] = lower_map[predicted.lower()]
                    return parsed
        except json.JSONDecodeError:
            pass

        # Fallback: scan response for class names
        response_lower = response.lower()
        for cls in self.class_names:
            if cls.lower() in response_lower:
                return {"class": cls, "reasoning": "parsed from text"}

        # Last resort: return first class
        return {"class": self.class_names[0], "reasoning": "fallback (parse failed)"}

    def evaluate(
        self,
        texts: List[str],
        true_labels: List[str],
        n_samples: Optional[int] = None,
        confidence_threshold: float = 0.6,
    ) -> Dict:
        """
        Full evaluation with accuracy, coverage, and confidence analysis.

        Args:
            texts: Test texts
            true_labels: Ground truth labels
            n_samples: Override n_consistency
            confidence_threshold: Min confidence to count as "confident"

        Returns:
            Dict with evaluation metrics
        """
        results = self.classify_batch(texts, n_samples=n_samples)

        predictions = [r["prediction"] for r in results]
        confidences = [r["confidence"] for r in results]

        # Overall accuracy
        correct = sum(p == t for p, t in zip(predictions, true_labels))
        overall_acc = correct / len(true_labels)

        # Confident subset
        confident_mask = np.array(confidences) >= confidence_threshold
        n_confident = confident_mask.sum()

        if n_confident > 0:
            confident_preds = [p for p, m in zip(predictions, confident_mask) if m]
            confident_true = [t for t, m in zip(true_labels, confident_mask) if m]
            confident_acc = sum(
                p == t for p, t in zip(confident_preds, confident_true)
            ) / n_confident
            coverage = n_confident / len(true_labels)
        else:
            confident_acc = 0.0
            coverage = 0.0

        # Deferred subset analysis
        deferred_mask = ~confident_mask
        n_deferred = deferred_mask.sum()
        if n_deferred > 0:
            deferred_preds = [p for p, m in zip(predictions, deferred_mask) if m]
            deferred_true = [t for t, m in zip(true_labels, deferred_mask) if m]
            deferred_acc = sum(
                p == t for p, t in zip(deferred_preds, deferred_true)
            ) / n_deferred
        else:
            deferred_acc = 0.0

        # Per-class accuracy
        per_class = {}
        for cls in set(true_labels):
            cls_mask = [t == cls for t in true_labels]
            cls_preds = [p for p, m in zip(predictions, cls_mask) if m]
            cls_true = [t for t, m in zip(true_labels, cls_mask) if m]
            if cls_true:
                per_class[cls] = {
                    "accuracy": sum(p == t for p, t in zip(cls_preds, cls_true)) / len(cls_true),
                    "n": len(cls_true),
                    "avg_confidence": float(np.mean([
                        c for c, m in zip(confidences, cls_mask) if m
                    ])),
                }

        # Coverage-accuracy curve
        curve = []
        for t in np.arange(0.2, 1.01, 0.1):
            t_mask = np.array(confidences) >= t
            t_n = t_mask.sum()
            if t_n > 0:
                t_preds = [p for p, m in zip(predictions, t_mask) if m]
                t_true = [l for l, m in zip(true_labels, t_mask) if m]
                t_acc = sum(p == t_ for p, t_ in zip(t_preds, t_true)) / t_n
                curve.append({
                    "threshold": float(t),
                    "coverage": t_n / len(true_labels),
                    "accuracy": t_acc,
                    "n": t_n,
                })

        return {
            "overall_accuracy": overall_acc,
            "confident_accuracy": confident_acc,
            "coverage": coverage,
            "deferred_accuracy": deferred_acc,
            "n_total": len(true_labels),
            "n_confident": int(n_confident),
            "n_deferred": int(n_deferred),
            "confidence_threshold": confidence_threshold,
            "per_class": per_class,
            "curve": curve,
            "client_stats": self.client.stats(),
        }
