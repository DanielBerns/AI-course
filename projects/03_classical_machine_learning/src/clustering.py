"""
Unsupervised Clustering & Silhouette Analysis Module.

Evaluates K-Means clustering quality across candidate cluster sizes K
using Silhouette Coefficients and the Elbow inertia metric.
"""

from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples


class ClusteringAnalyzer:
    """Analyze cluster separation and optimal K selection using K-Means."""

    def __init__(self, k_range: Tuple[int, int] = (2, 8), random_state: int = 42) -> None:
        self.k_range = k_range
        self.random_state = random_state
        self.results_: Dict[int, Dict[str, float]] = {}

    def fit_evaluate(self, X: np.ndarray) -> Dict[int, Dict[str, float]]:
        """Evaluate inertia and silhouette score across K range."""
        for k in range(self.k_range[0], self.k_range[1] + 1):
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(X)

            sil_score = float(silhouette_score(X, labels))
            inertia = float(kmeans.inertia_)

            self.results_[k] = {
                "inertia": inertia,
                "silhouette_score": sil_score,
            }

        return self.results_

    def get_optimal_k(self) -> int:
        """Return K that maximizes the Silhouette Coefficient."""
        if not self.results_:
            raise RuntimeError("Must call fit_evaluate before retrieving optimal K.")
        return max(self.results_.keys(), key=lambda k: self.results_[k]["silhouette_score"])

    def plot_evaluation(self, output_path: str = None) -> plt.Figure:
        """Plot Inertia (Elbow) and Silhouette Score curves."""
        ks = list(self.results_.keys())
        inertias = [self.results_[k]["inertia"] for k in ks]
        silhouettes = [self.results_[k]["silhouette_score"] for k in ks]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

        # Elbow Plot
        ax1.plot(ks, inertias, "o-", color="#263c72", linewidth=2)
        ax1.set_title("Elbow Method (Inertia vs. K)", fontsize=11, fontweight="bold", color="#263c72")
        ax1.set_xlabel("Number of Clusters K")
        ax1.set_ylabel("Inertia (Within-Cluster Sum of Squares)")

        # Silhouette Score Plot
        ax2.plot(ks, silhouettes, "s-", color="#009688", linewidth=2)
        ax2.set_title("Silhouette Coefficient vs. K", fontsize=11, fontweight="bold", color="#263c72")
        ax2.set_xlabel("Number of Clusters K")
        ax2.set_ylabel("Mean Silhouette Score")
        ax2.axvline(x=self.get_optimal_k(), color="#e67e22", linestyle="--", label=f"Optimal K = {self.get_optimal_k()}")
        ax2.legend()

        plt.tight_layout()
        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches="tight")

        return fig
