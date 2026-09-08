"""
Exploratory Data Analysis (EDA) & Data Visualization Module.

Generates high-aesthetic, publication-grade figures for distribution analysis,
correlation heatmaps, missing value profiles, and demographic comparisons.
"""

import os
from typing import List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Configure global aesthetic style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


def plot_missingness(df: pd.DataFrame, output_path: Optional[str] = None) -> plt.Figure:
    """Plot missing data proportions per column."""
    missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(4, len(df.columns) * 0.3)))

    bars = ax.barh(missing_pct.index, missing_pct.values, color="#263c72", alpha=0.85)
    ax.set_xlabel("Missing Data (%)", fontsize=11, fontweight="bold", color="#263c72")
    ax.set_title("Column Missingness Profile", fontsize=13, fontweight="bold", pad=12, color="#263c72")
    ax.set_xlim(0, max(100, missing_pct.max() + 10))

    for bar in bars:
        width = bar.get_width()
        if width > 0:
            ax.text(
                width + 1,
                bar.get_y() + bar.get_height() / 2,
                f"{width:.1f}%",
                va="center",
                ha="left",
                fontsize=9,
                color="#4b505a",
            )

    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig


def plot_distributions(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Plot feature distributions with KDE and histogram overlays."""
    if columns is None:
        columns = list(df.select_dtypes(include=[np.number]).columns[:6])

    n_cols = 3
    n_rows = (len(columns) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    axes = np.array(axes).flatten()

    for idx, col in enumerate(columns):
        ax = axes[idx]
        sns.histplot(
            df[col].dropna(),
            kde=True,
            ax=ax,
            color="#009688",
            edgecolor="white",
            alpha=0.6,
            line_kws={"linewidth": 2, "color": "#263c72"},
        )
        ax.set_title(f"Distribution of {col}", fontsize=11, fontweight="bold", color="#263c72")
        ax.set_xlabel(col, fontsize=10)
        ax.set_ylabel("Frequency", fontsize=10)

    # Hide unused axes
    for idx in range(len(columns), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig


def plot_correlation_matrix(df: pd.DataFrame, output_path: Optional[str] = None) -> plt.Figure:
    """Plot lower-triangular correlation heatmap for numeric features."""
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(9, 7))
    cmap = sns.diverging_palette(220, 20, as_cmap=True)

    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmax=1.0,
        vmin=-1.0,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation"},
        annot=True,
        fmt=".2f",
        annot_kws={"size": 8},
        ax=ax,
    )
    ax.set_title("Feature Correlation Matrix", fontsize=13, fontweight="bold", pad=12, color="#263c72")
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig
