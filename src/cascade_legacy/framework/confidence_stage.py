"""
Generalizable Confidence-Gated Classification Stage.

A single reusable building block for the cascade. Each stage:
1. Trains a classifier on labeled data
2. Calibrates probabilities (isotonic / Platt scaling)
3. Tunes confidence thresholds via out-of-fold predictions
4. At inference: predicts class + confidence, gates uncertain cases

This is dataset-agnostic. To adapt to a new CI system:
- Provide features, labels, and class names
- The stage handles calibration, threshold tuning, and routing automatically

Bug fixes applied:
  G1: Calibrated OOF predictions (thresholds tuned on calibrated probs)
  G2: Threshold search finds highest-coverage threshold meeting accuracy target
  G3: min_threshold_samples raised to 20 (configurable)
  I1: Auto-select calibration method based on class sizes
  A1: Default model changed to XGBoost (RF fallback)
  A2: Optional text feature support (TF-IDF)
  A5: explain_deferral() for rich context on deferred cases
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from sklearn.base import BaseEstimator, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

# GPU detection for XGBoost and matrix operations
HAS_CUDA = False
try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
    if HAS_CUDA:
        _GPU_DEVICE = torch.device('cuda')
except ImportError:
    pass

# Also check XGBoost GPU support
HAS_XGBOOST_GPU = False
if HAS_XGBOOST:
    try:
        _test = XGBClassifier(tree_method='hist', device='cuda', n_estimators=1)
        _test.fit(np.random.rand(10, 2), np.random.randint(0, 2, 10))
        HAS_XGBOOST_GPU = True
        del _test
    except Exception:
        pass


class ConfidenceStage:
    """
    A single confidence-gated classification stage.

    Encapsulates: model training, probability calibration, OOF threshold tuning,
    per-class confidence gates, and prediction with abstention.

    Usage:
        stage = ConfidenceStage(
            name='disposition',
            classes={4: 'Actionable', 6: 'Wontfix', 7: 'Fixed', 1: 'Downstream'},
            target_accuracy=0.85,
        )
        stage.fit(X_train, y_train, feature_names=feature_cols)
        predictions = stage.predict(X_test)
        # predictions['class'] = predicted class or -1 (deferred)
        # predictions['confidence'] = calibrated probability
        # predictions['is_confident'] = True/False
    """

    def __init__(
        self,
        name: str,
        classes: Dict[int, str],
        target_accuracy: float = 0.85,
        calibration_method: str = 'auto',
        n_cv_folds: int = 5,
        random_state: int = 42,
        model: Optional[BaseEstimator] = None,
        defer_label: int = -1,
        min_threshold_samples: int = 20,
        text_max_features: int = 500,
        margin_threshold: Optional[float] = None,
        decision_thresholds: Optional[Dict[int, float]] = None,
    ):
        """
        Args:
            name: Stage name (for logging)
            classes: Mapping of class_code -> class_name
            target_accuracy: Target accuracy per class for threshold tuning
            calibration_method: 'isotonic', 'sigmoid', or 'auto' (selects
                based on minority class size: sigmoid if <500, isotonic otherwise)
            n_cv_folds: Number of CV folds for calibration and OOF
            random_state: Random seed
            model: Custom model (default: XGBoost if available, else RF)
            defer_label: Label to use for deferred/uncertain predictions
            min_threshold_samples: Minimum samples per class for threshold tuning
            text_max_features: Max TF-IDF features when text preprocessing is used
            margin_threshold: Optional minimum gap between top-1 and top-2
                probabilities for a prediction to be confident. Catches items
                where the model is torn between classes (e.g., noise vs valid
                with 0.55/0.45 probs). None disables margin check (default).
            decision_thresholds: Optional per-class decision thresholds that
                override argmax. Maps class_code -> probability threshold.
                If P(class) > threshold, predict that class (checked in order
                of decreasing probability). Useful for minority classes where
                calibrated P(minority) rarely exceeds 0.50 but detecting the
                minority class is important (e.g., noise gates).
        """
        self.name = name
        self.classes = classes
        self.target_accuracy = target_accuracy
        self.calibration_method = calibration_method
        self.n_cv_folds = n_cv_folds
        self.random_state = random_state
        self.defer_label = defer_label
        self.min_threshold_samples = min_threshold_samples
        self.text_max_features = text_max_features
        self.margin_threshold = margin_threshold
        self.decision_thresholds = decision_thresholds

        self._model = model
        self._calibrated_model = None
        self._label_encoder = None
        self._scaler = None
        self._per_class_thresholds = None
        self._feature_names = None
        self._oof_proba = None
        self._tfidf = None
        self._cal_method = None
        self._is_fitted = False

    def _default_model(self) -> BaseEstimator:
        """Default model: XGBoost with GPU preferred, RF fallback (A1)."""
        if HAS_XGBOOST:
            params = dict(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state,
                eval_metric='logloss',
                n_jobs=-1,
            )
            if HAS_XGBOOST_GPU:
                params['tree_method'] = 'hist'
                params['device'] = 'cuda'
            return XGBClassifier(**params)
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=self.random_state,
            n_jobs=-1,
        )

    def _select_calibration_method(self, y: np.ndarray) -> str:
        """Auto-select calibration method based on class sizes (I1).

        Isotonic calibration requires many samples per class to avoid
        overfitting. Falls back to sigmoid (Platt scaling) for small classes.
        """
        if self.calibration_method != 'auto':
            return self.calibration_method
        min_class_count = min(np.bincount(y))
        if min_class_count < 500:
            return 'sigmoid'
        return 'isotonic'

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        scale: bool = True,
        text_data: Optional[np.ndarray] = None,
    ) -> 'ConfidenceStage':
        """
        Train the stage: fit model, calibrate, tune thresholds.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target labels (original class codes, not encoded)
            feature_names: Optional feature column names
            scale: Whether to apply StandardScaler
            text_data: Optional text data for TF-IDF preprocessing (A2).
                Array of strings, one per sample.

        Returns:
            self
        """
        self._feature_names = feature_names

        # Encode labels to 0..n-1
        self._label_encoder = LabelEncoder()
        y_enc = self._label_encoder.fit_transform(y)

        # Scale features
        if scale:
            self._scaler = StandardScaler()
            X_scaled = self._scaler.fit_transform(X)
        else:
            self._scaler = None
            X_scaled = np.array(X, dtype=float)

        # A2: Text preprocessing - fit TF-IDF and append features
        if text_data is not None:
            X_scaled = self._fit_text_features(X_scaled, text_data)

        n_classes = len(self._label_encoder.classes_)
        class_counts = np.bincount(y_enc, minlength=n_classes)
        print(f"[{self.name}] Training: {len(y)} samples, {n_classes} classes")
        for i, code in enumerate(self._label_encoder.classes_):
            label = self.classes.get(int(code), str(code))
            print(f"  {label} ({code}): {class_counts[i]}")

        # Get base model
        base_model = self._model if self._model is not None else self._default_model()

        # I1: Select calibration method based on class sizes
        cal_method = self._select_calibration_method(y_enc)
        self._cal_method = cal_method
        print(f"[{self.name}] Calibration method: {cal_method}")

        # G1: Generate CALIBRATED OOF predictions for threshold tuning
        # Previously used raw model probs, but thresholds are applied to
        # calibrated model output. Now OOF uses CalibratedClassifierCV
        # within each fold so threshold distribution matches inference.
        print(f"[{self.name}] Generating calibrated OOF predictions "
              f"({self.n_cv_folds}-fold)...")
        self._oof_proba = self._get_calibrated_oof_predictions(
            base_model, X_scaled, y_enc, cal_method
        )

        # G2, G3: Find per-class thresholds on calibrated OOF probs
        self._per_class_thresholds = self._find_per_class_thresholds(
            y_enc, self._oof_proba
        )
        print(f"[{self.name}] Per-class thresholds:")
        for i, code in enumerate(self._label_encoder.classes_):
            label = self.classes.get(int(code), str(code))
            print(f"  {label}: {self._per_class_thresholds[i]:.2f}")
        if self.margin_threshold is not None:
            print(f"[{self.name}] Margin threshold: {self.margin_threshold:.2f}")
        if self.decision_thresholds is not None:
            for code, dt in self.decision_thresholds.items():
                label = self.classes.get(code, str(code))
                print(f"[{self.name}] Decision threshold for {label}: {dt:.2f}")

        # Step 3: Train calibrated model on full data
        print(f"[{self.name}] Training calibrated model...")
        self._calibrated_model = CalibratedClassifierCV(
            clone(base_model),
            method=cal_method,
            cv=self.n_cv_folds,
        )
        self._calibrated_model.fit(X_scaled, y_enc)

        # Report OOF performance
        self._report_oof_performance(y_enc)

        self._is_fitted = True
        return self

    def predict(
        self,
        X: np.ndarray,
        return_proba: bool = False,
        text_data: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Predict with confidence gating.

        Args:
            X: Feature matrix
            return_proba: Whether to include full probability matrix
            text_data: Optional text data for TF-IDF (must match training)

        Returns:
            Dict with keys:
                'class': predicted class codes (-1 for deferred)
                'confidence': calibrated max probability
                'is_confident': boolean mask
                'predicted_raw': predicted class codes (ignoring threshold)
                'proba': full probability matrix (if return_proba=True)
        """
        if not self._is_fitted:
            raise RuntimeError(f"Stage '{self.name}' is not fitted. Call fit() first.")

        if self._scaler is not None:
            X_scaled = self._scaler.transform(X)
        else:
            X_scaled = np.array(X, dtype=float)

        # A2: Transform text features if TF-IDF was fitted during training
        if text_data is not None and self._tfidf is not None:
            X_scaled = self._transform_text_features(X_scaled, text_data)

        proba = self._calibrated_model.predict_proba(X_scaled)

        # Decision thresholds: override argmax for specific classes
        # This allows detecting minority classes where P(class) < 0.50
        # but above a custom decision threshold
        if self.decision_thresholds is not None:
            classes = self._label_encoder.classes_
            predicted_idx = np.argmax(proba, axis=1)
            # Check each class with a custom threshold (priority: higher prob first)
            for class_code, dt in self.decision_thresholds.items():
                # Find the encoded index for this class code
                idx_matches = np.where(classes == class_code)[0]
                if len(idx_matches) == 0:
                    continue
                class_idx = idx_matches[0]
                # Override: if P(class) > threshold and it's not already predicted
                override_mask = (proba[:, class_idx] >= dt) & (predicted_idx != class_idx)
                predicted_idx[override_mask] = class_idx
            confidence = proba[np.arange(len(proba)), predicted_idx]
            predicted_codes = self._label_encoder.inverse_transform(predicted_idx)
        else:
            # GPU-accelerated max/argmax for large batches
            confidence, predicted_idx = self._gpu_max_and_argmax(proba)
            predicted_codes = self._label_encoder.inverse_transform(predicted_idx)

        # Apply per-class thresholds
        per_sample_threshold = self._per_class_thresholds[predicted_idx]
        is_confident = confidence >= per_sample_threshold

        # Margin-based uncertainty: require sufficient gap between top-1 and top-2
        if self.margin_threshold is not None and proba.shape[1] >= 2:
            top2 = np.partition(proba, -2, axis=1)[:, -2:]
            margin = top2[:, 1] - top2[:, 0]
            is_confident = is_confident & (margin >= self.margin_threshold)

        gated_codes = np.where(is_confident, predicted_codes, self.defer_label)

        result = {
            'class': gated_codes,
            'confidence': confidence,
            'is_confident': is_confident,
            'predicted_raw': predicted_codes,
        }

        if return_proba:
            result['proba'] = proba
            result['class_codes'] = self._label_encoder.classes_

        return result

    def get_oof_predictions(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return OOF predictions for downstream stage training.

        Returns:
            (oof_proba, label_encoder_classes)
        """
        if self._oof_proba is None:
            raise RuntimeError("No OOF predictions available. Call fit() first.")
        return self._oof_proba, self._label_encoder.classes_

    def coverage_accuracy_curve(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        thresholds: Optional[np.ndarray] = None,
        text_data: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        Compute coverage-accuracy tradeoff on a test set.

        Args:
            X: Test features
            y_true: True class codes
            thresholds: Confidence thresholds to evaluate
            text_data: Optional text data

        Returns:
            DataFrame with threshold, coverage, accuracy columns.
            Includes 'is_operating_point' column marking the per-class
            threshold operating point (L1).
        """
        if thresholds is None:
            thresholds = np.arange(0.40, 0.96, 0.05)

        preds = self.predict(X, text_data=text_data)
        confidence = preds['confidence']
        predicted_raw = preds['predicted_raw']

        # Find the effective operating point threshold (average of per-class)
        op_threshold = np.mean(self._per_class_thresholds)

        results = []
        for t in thresholds:
            mask = confidence >= t
            n = mask.sum()
            if n > 0:
                acc = (y_true[mask] == predicted_raw[mask]).mean()
                results.append({
                    'threshold': t,
                    'coverage': n / len(y_true),
                    'accuracy': acc,
                    'n_predicted': int(n),
                    'n_deferred': int(len(y_true) - n),
                    'is_operating_point': abs(t - op_threshold) < 0.025,
                })

        return pd.DataFrame(results)

    def explain_deferral(
        self,
        X: np.ndarray,
        top_k: int = 3,
        text_data: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        """
        Rich context for deferred cases (A5).

        For each sample, returns candidate classes, confidence scores,
        and feature importance to help human reviewers.

        Args:
            X: Feature matrix for deferred samples
            top_k: Number of top candidate classes to return
            text_data: Optional text data

        Returns:
            List of dicts, one per sample, with:
                - candidates: list of (class_code, class_name, probability)
                - confidence: max probability
                - predicted_class: raw prediction (before gating)
                - threshold: per-class threshold that wasn't met
                - feature_importance: top features (if model supports it)
        """
        if not self._is_fitted:
            raise RuntimeError(f"Stage '{self.name}' is not fitted.")

        preds = self.predict(X, return_proba=True, text_data=text_data)
        proba = preds['proba']
        classes = self._label_encoder.classes_

        # Try to get feature importance
        feature_imp = self._get_feature_importance()

        explanations = []
        for i in range(len(X)):
            # Sort classes by probability (descending)
            sorted_idx = np.argsort(proba[i])[::-1][:top_k]
            candidates = []
            for idx in sorted_idx:
                code = int(classes[idx])
                candidates.append({
                    'class_code': code,
                    'class_name': self.classes.get(code, str(code)),
                    'probability': float(proba[i, idx]),
                })

            pred_idx = np.argmax(proba[i])
            pred_code = int(classes[pred_idx])

            explanation = {
                'candidates': candidates,
                'confidence': float(preds['confidence'][i]),
                'predicted_class': pred_code,
                'predicted_name': self.classes.get(pred_code, str(pred_code)),
                'threshold': float(self._per_class_thresholds[pred_idx]),
                'gap': float(self._per_class_thresholds[pred_idx]
                             - preds['confidence'][i]),
            }

            if feature_imp is not None and self._feature_names is not None:
                top_feat_idx = np.argsort(feature_imp)[::-1][:5]
                explanation['top_features'] = [
                    (self._feature_names[j], float(feature_imp[j]))
                    for j in top_feat_idx
                    if j < len(self._feature_names)
                ]

            explanations.append(explanation)

        return explanations

    def _get_feature_importance(self) -> Optional[np.ndarray]:
        """Extract feature importance from the underlying model if possible."""
        try:
            # CalibratedClassifierCV wraps calibrated_classifiers_
            cal = self._calibrated_model
            if hasattr(cal, 'calibrated_classifiers_') and cal.calibrated_classifiers_:
                base = cal.calibrated_classifiers_[0].estimator
                if hasattr(base, 'feature_importances_'):
                    return base.feature_importances_
        except Exception:
            pass
        return None

    def _fit_text_features(
        self, X: np.ndarray, text_data: np.ndarray
    ) -> np.ndarray:
        """Fit TF-IDF on text data and append to feature matrix (A2)."""
        from sklearn.feature_extraction.text import TfidfVectorizer

        text_data = np.asarray(text_data, dtype=str)
        # Replace NaN/None with empty string
        text_data = np.where(
            (text_data == 'nan') | (text_data == 'None') | (text_data == ''),
            '',
            text_data,
        )

        self._tfidf = TfidfVectorizer(
            max_features=self.text_max_features,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
        )
        text_features = self._tfidf.fit_transform(text_data).toarray()
        print(f"[{self.name}] TF-IDF: {text_features.shape[1]} text features added")
        return np.hstack([X, text_features])

    def _transform_text_features(
        self, X: np.ndarray, text_data: np.ndarray
    ) -> np.ndarray:
        """Transform text data using fitted TF-IDF and append (A2)."""
        text_data = np.asarray(text_data, dtype=str)
        text_data = np.where(
            (text_data == 'nan') | (text_data == 'None') | (text_data == ''),
            '',
            text_data,
        )
        text_features = self._tfidf.transform(text_data).toarray()
        return np.hstack([X, text_features])

    def _get_calibrated_oof_predictions(
        self,
        base_model: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray,
        cal_method: str,
    ) -> np.ndarray:
        """Generate calibrated out-of-fold probability predictions (G1).

        Each OOF fold trains a CalibratedClassifierCV so that the OOF
        probability distribution matches what the final calibrated model
        produces. This ensures thresholds tuned on OOF probs are valid
        at inference time.
        """
        n_classes = len(np.unique(y))
        oof_proba = np.zeros((len(y), n_classes))

        skf = StratifiedKFold(
            n_splits=self.n_cv_folds,
            shuffle=True,
            random_state=self.random_state,
        )

        # Inner CV for calibration within each OOF fold
        inner_cv = min(3, self.n_cv_folds)

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            train_y = y[train_idx]
            min_class_in_fold = min(
                np.bincount(train_y, minlength=n_classes)
            )

            if min_class_in_fold >= inner_cv + 1:
                # Enough samples for inner calibration CV
                fold_cal = CalibratedClassifierCV(
                    clone(base_model),
                    method=cal_method,
                    cv=inner_cv,
                )
                fold_cal.fit(X[train_idx], y[train_idx])
                oof_proba[val_idx] = fold_cal.predict_proba(X[val_idx])
            else:
                # Fallback: raw model (insufficient samples for inner CV)
                fold_model = clone(base_model)
                fold_model.fit(X[train_idx], y[train_idx])
                oof_proba[val_idx] = fold_model.predict_proba(X[val_idx])

        return oof_proba

    def _find_per_class_thresholds(
        self,
        y_true: np.ndarray,
        proba: np.ndarray,
    ) -> np.ndarray:
        """Find per-class confidence thresholds targeting self.target_accuracy.

        G2: Collects ALL thresholds meeting accuracy target and returns the
            one with highest coverage (lowest threshold).
        G3: Uses self.min_threshold_samples (default 20, was 5).

        Uses GPU-accelerated threshold sweep when available.
        """
        n_classes = proba.shape[1]
        thresholds = np.full(n_classes, 0.50)
        _, predicted = self._gpu_max_and_argmax(proba)
        min_samples = self.min_threshold_samples

        for c in range(n_classes):
            class_mask = predicted == c
            if class_mask.sum() < min_samples:
                continue

            class_true = y_true[class_mask]
            class_conf = np.max(proba[class_mask], axis=1)

            best_t, _ = self._gpu_threshold_sweep(
                class_true, class_conf, c,
                self.target_accuracy, min_samples
            )
            thresholds[c] = best_t

        return thresholds

    def _report_oof_performance(self, y_enc: np.ndarray):
        """Print OOF performance summary."""
        oof_conf = np.max(self._oof_proba, axis=1)
        oof_pred = np.argmax(self._oof_proba, axis=1)

        # At per-class thresholds
        per_sample_t = self._per_class_thresholds[oof_pred]
        is_conf = oof_conf >= per_sample_t
        n_conf = is_conf.sum()

        if n_conf > 0:
            acc = (y_enc[is_conf] == oof_pred[is_conf]).mean()
            cov = n_conf / len(y_enc)
            print(f"[{self.name}] OOF: {acc:.1%} accuracy, {cov:.1%} coverage "
                  f"({n_conf}/{len(y_enc)})")

        # Report majority baseline
        majority_class = np.argmax(np.bincount(y_enc))
        majority_acc = (y_enc == majority_class).mean()
        print(f"[{self.name}] Majority baseline: {majority_acc:.1%} "
              f"(class {int(self._label_encoder.classes_[majority_class])})")

    # --- GPU-accelerated matrix operations ---

    @staticmethod
    def _gpu_max_and_argmax(proba: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """GPU-accelerated max and argmax over probability matrix.

        Uses torch CUDA matmul-compatible ops for large arrays.
        Falls back to numpy for small arrays or if no GPU.
        """
        if HAS_CUDA and proba.shape[0] > 1000:
            t = torch.from_numpy(proba).to(_GPU_DEVICE)
            confidence, predicted_idx = t.max(dim=1)
            return confidence.cpu().numpy(), predicted_idx.cpu().numpy()
        return np.max(proba, axis=1), np.argmax(proba, axis=1)

    @staticmethod
    def _gpu_threshold_sweep(
        class_true: np.ndarray,
        class_conf: np.ndarray,
        class_id: int,
        target_accuracy: float,
        min_samples: int,
    ) -> Tuple[float, float]:
        """GPU-accelerated threshold sweep using vectorized matrix ops.

        Instead of looping over thresholds, computes accuracy at all
        thresholds simultaneously via matrix multiplication.
        """
        thresholds = np.arange(0.40, 0.96, 0.01)

        if HAS_CUDA and len(class_conf) > 500:
            # GPU vectorized: build mask matrix (n_thresholds x n_samples)
            t_conf = torch.from_numpy(class_conf.astype(np.float32)).to(_GPU_DEVICE)
            t_thresholds = torch.from_numpy(thresholds.astype(np.float32)).to(_GPU_DEVICE)
            t_true = torch.from_numpy((class_true == class_id).astype(np.float32)).to(_GPU_DEVICE)

            # mask_matrix[i, j] = 1 if conf[j] >= threshold[i]
            mask_matrix = (t_conf.unsqueeze(0) >= t_thresholds.unsqueeze(1)).float()

            # counts per threshold
            counts = mask_matrix.sum(dim=1)

            # correct predictions per threshold: sum of (mask * correct)
            correct = (mask_matrix * t_true.unsqueeze(0)).sum(dim=1)

            # accuracy and coverage
            acc = torch.where(counts >= min_samples, correct / counts, torch.zeros_like(counts))
            cov = mask_matrix.mean(dim=1)

            # Find best: highest coverage meeting accuracy target
            valid = (acc >= target_accuracy) & (counts >= min_samples)
            if valid.any():
                valid_covs = torch.where(valid, cov, torch.tensor(-1.0, device=_GPU_DEVICE))
                best_idx = valid_covs.argmax().item()
                return float(thresholds[best_idx]), float(cov[best_idx].item())

            return 0.50, 0.0
        else:
            # CPU fallback
            best_t = 0.50
            best_cov = 0.0
            for t in thresholds:
                t_mask = class_conf >= t
                if t_mask.sum() < min_samples:
                    continue
                acc = (class_true[t_mask] == class_id).mean()
                cov = t_mask.mean()
                if acc >= target_accuracy and cov > best_cov:
                    best_t = t
                    best_cov = cov
            return best_t, best_cov

    @property
    def class_names(self) -> Dict[int, str]:
        """Return the class code -> name mapping."""
        return self.classes.copy()

    @property
    def feature_names(self) -> Optional[List[str]]:
        """Return feature names if provided during fit."""
        return self._feature_names
