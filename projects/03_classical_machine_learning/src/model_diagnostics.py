"""
Model Performance Evaluation & Diagnostics Module.

Generates Learning Curves and Validation Curves to diagnose overfitting (high variance)
versus underfitting (high bias), and outputs ROC-AUC and Precision-Recall evaluation curves.
"""

from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score


def plot_learning_curve(
    estimator: object,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    scoring: str = "roc_auc",
    output_path: Optional[str] = None,
) -> Tuple[plt.Figure, Dict[str, float]]:
    """
    Plot learning curve (training vs. validation score across training sizes)
    and diagnose underfitting vs. overfitting.
    """
    train_sizes, train_scores, val_scores = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        scoring=scoring,
        train_sizes=np.linspace(0.1, 1.0, 10),
        n_jobs=-1,
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    # Compute diagnostic gap (Overfitting = train_score - val_score at max size)
    final_gap = float(train_mean[-1] - val_mean[-1])
    final_val_score = float(val_mean[-1])

    is_overfitting = final_gap > 0.08
    is_underfitting = final_val_score < 0.70 and final_gap < 0.05

    diagnosis = "Balanced Generalization"
    if is_overfitting:
        diagnosis = f"OVERFITTING Detected (High Variance, Train-Val Gap: {final_gap:.3f})"
    elif is_underfitting:
        diagnosis = f"UNDERFITTING Detected (High Bias, Val Score: {final_val_score:.3f})"

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_sizes, train_mean, "o-", color="#e67e22", label="Training Score", linewidth=2)
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color="#e67e22")

    ax.plot(train_sizes, val_mean, "s-", color="#263c72", label="Validation Score", linewidth=2)
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color="#263c72")

    ax.set_title(f"Learning Curve Analysis\n[{diagnosis}]", fontsize=12, fontweight="bold", color="#263c72")
    ax.set_xlabel("Training Set Size (Samples)", fontsize=10)
    ax.set_ylabel(f"Score ({scoring})", fontsize=10)
    ax.legend(frameon=True, loc="lower right")
    ax.set_ylim(0.4, 1.02)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    diagnostics_summary = {
        "final_train_score": train_mean[-1],
        "final_val_score": val_mean[-1],
        "generalization_gap": final_gap,
        "diagnosis": diagnosis,
        "is_overfitting": is_overfitting,
        "is_underfitting": is_underfitting,
    }

    return fig, diagnostics_summary


def plot_roc_pr_curves(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Plot ROC-AUC and Precision-Recall curves side-by-side."""
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc_val = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    pr_auc_val = average_precision_score(y_true, y_probs)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ROC Curve
    ax1.plot(fpr, tpr, color="#263c72", linewidth=2.5, label=f"ROC Curve (AUC = {roc_auc_val:.3f})")
    ax1.plot([0, 1], [0, 1], color="#999999", linestyle="--")
    ax1.set_title("Receiver Operating Characteristic (ROC)", fontsize=11, fontweight="bold", color="#263c72")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.legend(loc="lower right")

    # Precision-Recall Curve
    ax2.plot(recall, precision, color="#009688", linewidth=2.5, label=f"PR Curve (AP = {pr_auc_val:.3f})")
    ax2.set_title("Precision-Recall Curve", fontsize=11, fontweight="bold", color="#263c72")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.legend(loc="lower left")

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig
