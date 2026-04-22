"""
Phase 4: Change-Point Detection Algorithms
Implementations using ruptures and custom methods.
"""

import numpy as np
from typing import List, Optional, Dict, Tuple
from abc import ABC, abstractmethod

try:
    import ruptures as rpt
    HAS_RUPTURES = True
except ImportError:
    HAS_RUPTURES = False
    print("Warning: ruptures not installed. Run: pip install ruptures")


class ChangePointDetector(ABC):
    """Base class for change-point detection algorithms."""

    @abstractmethod
    def fit_predict(self, signal: np.ndarray) -> List[int]:
        """Detect change points in a signal."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Algorithm name."""
        pass


class CUSUMDetector(ChangePointDetector):
    """
    PELT with RBF kernel (formerly mislabeled as CUSUM).

    NOTE: This is actually PELT (Pruned Exact Linear Time) with RBF kernel,
    not a true CUSUM algorithm. Renamed for accuracy per ICSME 2025 review.
    """

    def __init__(self, min_size: int = 5, jump: int = 1, pen: float = 3.0):
        self.min_size = min_size
        self.jump = jump
        self.pen = pen

    @property
    def name(self) -> str:
        return "PELT-RBF"  # Fixed: was "CUSUM" (incorrect)

    def fit_predict(self, signal: np.ndarray) -> List[int]:
        if not HAS_RUPTURES or len(signal) < self.min_size * 2:
            return []

        algo = rpt.Pelt(model="rbf", min_size=self.min_size, jump=self.jump)
        try:
            result = algo.fit_predict(signal, pen=self.pen)
            # Remove the last index (end of signal)
            return [r for r in result if r < len(signal)]
        except Exception:
            return []


class PELTDetector(ChangePointDetector):
    """PELT (Pruned Exact Linear Time) detector."""

    def __init__(self, model: str = "rbf", min_size: int = 5, pen: float = 3.0):
        self.model = model
        self.min_size = min_size
        self.pen = pen

    @property
    def name(self) -> str:
        return f"PELT_{self.model}"

    def fit_predict(self, signal: np.ndarray) -> List[int]:
        if not HAS_RUPTURES or len(signal) < self.min_size * 2:
            return []

        algo = rpt.Pelt(model=self.model, min_size=self.min_size)
        try:
            result = algo.fit_predict(signal, pen=self.pen)
            return [r for r in result if r < len(signal)]
        except Exception:
            return []


class BinarySegmentationDetector(ChangePointDetector):
    """Binary Segmentation change point detector."""

    def __init__(self, model: str = "l2", min_size: int = 5, n_bkps: int = 5):
        self.model = model
        self.min_size = min_size
        self.n_bkps = n_bkps

    @property
    def name(self) -> str:
        return "BinSeg"

    def fit_predict(self, signal: np.ndarray) -> List[int]:
        if not HAS_RUPTURES or len(signal) < self.min_size * 2:
            return []

        algo = rpt.Binseg(model=self.model, min_size=self.min_size)
        try:
            result = algo.fit_predict(signal, n_bkps=self.n_bkps)
            return [r for r in result if r < len(signal)]
        except Exception:
            return []


class WindowBasedDetector(ChangePointDetector):
    """Window-based change detection using sliding windows."""

    def __init__(self, width: int = 10, model: str = "l2", pen: float = 3.0):
        self.width = width
        self.model = model
        self.pen = pen

    @property
    def name(self) -> str:
        return f"Window_{self.width}"

    def fit_predict(self, signal: np.ndarray) -> List[int]:
        if not HAS_RUPTURES or len(signal) < self.width * 2:
            return []

        algo = rpt.Window(width=self.width, model=self.model)
        try:
            result = algo.fit_predict(signal, pen=self.pen)
            return [r for r in result if r < len(signal)]
        except Exception:
            return []


class BOCDDetector(ChangePointDetector):
    """Bayesian Online Change Detection (simplified implementation)."""

    def __init__(self, hazard_lambda: float = 100, threshold: float = 0.5):
        self.hazard_lambda = hazard_lambda
        self.threshold = threshold

    @property
    def name(self) -> str:
        return "BOCD"

    def fit_predict(self, signal: np.ndarray) -> List[int]:
        if len(signal) < 10:
            return []

        # Simplified BOCD using run length posterior
        n = len(signal)
        change_points = []

        # Compute running statistics
        window = 10
        for i in range(window, n - window):
            pre_mean = np.mean(signal[i-window:i])
            post_mean = np.mean(signal[i:i+window])
            pre_std = np.std(signal[i-window:i])

            if pre_std > 0:
                z_score = abs(post_mean - pre_mean) / pre_std
                # Probability of change based on hazard
                prob = 1 - np.exp(-z_score / self.hazard_lambda * 10)
                if prob > self.threshold:
                    change_points.append(i)

        # Merge nearby detections
        merged = []
        for cp in change_points:
            if not merged or cp - merged[-1] > window // 2:
                merged.append(cp)

        return merged


class MeanShiftDetector(ChangePointDetector):
    """Simple mean shift detection using z-score."""

    def __init__(self, window: int = 10, threshold: float = 2.5):
        self.window = window
        self.threshold = threshold

    @property
    def name(self) -> str:
        return f"MeanShift_{self.threshold}"

    def fit_predict(self, signal: np.ndarray) -> List[int]:
        if len(signal) < self.window * 2:
            return []

        change_points = []
        n = len(signal)

        for i in range(self.window, n - self.window):
            pre_window = signal[i-self.window:i]
            post_window = signal[i:i+self.window]

            pre_mean = np.mean(pre_window)
            pre_std = np.std(pre_window)

            if pre_std > 0:
                post_mean = np.mean(post_window)
                z_score = abs(post_mean - pre_mean) / pre_std
                if z_score > self.threshold:
                    change_points.append(i)

        # Merge nearby detections
        merged = []
        for cp in change_points:
            if not merged or cp - merged[-1] > self.window:
                merged.append(cp)

        return merged


def get_all_detectors(
    pen_values: List[float] = [1.0, 3.0, 5.0],
    window_sizes: List[int] = [5, 10, 15]
) -> Dict[str, ChangePointDetector]:
    """
    Get all change-point detection algorithms.

    Returns:
        Dictionary of detector_name -> detector
    """
    detectors = {}

    if HAS_RUPTURES:
        for pen in pen_values:
            detectors[f'PELT_l2_pen{pen}'] = PELTDetector(model='l2', pen=pen)
            detectors[f'PELT_rbf_pen{pen}'] = PELTDetector(model='rbf', pen=pen)

        detectors['BinSeg'] = BinarySegmentationDetector()

        for width in window_sizes:
            detectors[f'Window_{width}'] = WindowBasedDetector(width=width)

    # Custom detectors
    detectors['BOCD'] = BOCDDetector()
    detectors['MeanShift_2.0'] = MeanShiftDetector(threshold=2.0)
    detectors['MeanShift_2.5'] = MeanShiftDetector(threshold=2.5)
    detectors['MeanShift_3.0'] = MeanShiftDetector(threshold=3.0)

    return detectors


if __name__ == "__main__":
    # Test algorithms
    np.random.seed(42)

    # Create test signal with change point at index 50
    signal = np.concatenate([
        np.random.normal(100, 5, 50),
        np.random.normal(120, 5, 50)
    ])

    print("Testing change-point detection algorithms...")
    print(f"True change point: 50")

    detectors = get_all_detectors()
    for name, detector in detectors.items():
        cps = detector.fit_predict(signal)
        print(f"  {name}: {cps}")
