"""
Anomaly & Outlier Detection Module using Isolation Forest.

Detects rare, suspicious data points without ground-truth anomaly labels
by measuring average isolation path lengths in randomized trees.
"""

from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


class AnomalyDetector:
    """Isolation Forest anomaly scorer."""

    def __init__(self, contamination: float = 0.05, random_state: int = 42) -> None:
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100,
        )

    def fit_predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit model and return binary anomaly flags (-1 = anomaly, 1 = normal)
        and continuous anomaly decision scores.
        """
        labels = self.model.fit_predict(X)
        scores = self.model.decision_function(X)  # Lower values indicate higher anomaly likelihood
        return labels, scores

    def summarize_anomalies(self, df: pd.DataFrame, labels: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
        """Summarize detected anomalies count, ratio, and threshold score."""
        n_anomalies = int(np.sum(labels == -1))
        total = len(labels)
        return {
            "total_samples": total,
            "anomalies_detected": n_anomalies,
            "anomaly_ratio": float(n_anomalies / total),
            "min_anomaly_score": float(np.min(scores[labels == -1])) if n_anomalies > 0 else 0.0,
        }
