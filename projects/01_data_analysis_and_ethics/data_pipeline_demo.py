"""
Project 1: Data Pipeline, Exploratory Analysis & Bias Audit.

Demonstrates end-to-end data cleaning, EDA visualization, representation profiling,
and algorithmic bias auditing on a real-world domain dataset.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, fetch_california_housing

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from cleaning import DataCleaner
from visualization import plot_missingness, plot_distributions, plot_correlation_matrix
from bias_audit import BiasAuditor


def main() -> None:
    print("=" * 70)
    print("PROJECT 1: DATA PIPELINE, EXPLORATORY ANALYSIS & BIAS AUDIT")
    print("=" * 70)

    # 1. Load Real-World Domain Dataset (Breast Cancer Clinical Dataset)
    print("\n[Step 1] Loading Real-World Medical Dataset (UCI Breast Cancer Diagnostic)...")
    raw_data = load_breast_cancer(as_frame=True)
    df = raw_data.frame

    # Rename target column and add synthetic demographic attribute for ethical bias audit demo
    df = df.rename(columns={"target": "diagnosis"})  # 1 = Benign, 0 = Malignant

    # Inject realistic demographic proxy group (age_group & patient_tier) to demonstrate bias audit
    np.random.seed(42)
    df["patient_group"] = np.random.choice(["Group_A", "Group_B"], size=len(df), p=[0.65, 0.35])

    # Inject deliberate missingness and outliers for cleaning demonstration
    df_corrupted = df.copy()
    corrupt_idx = np.random.choice(df.index, size=40, replace=False)
    df_corrupted.loc[corrupt_idx, "mean radius"] = np.nan

    outlier_idx = np.random.choice(df.index, size=15, replace=False)
    df_corrupted.loc[outlier_idx, "mean texture"] = df["mean texture"].max() * 4.5

    print(f"Dataset shape: {df_corrupted.shape[0]} rows, {df_corrupted.shape[1]} columns")

    # 2. Data Quality Reporting & Cleaning
    print("\n[Step 2] Executing Data Quality Report & Cleaning Pipeline...")
    cleaner = DataCleaner(numeric_impute_strategy="median", outlier_method="iqr")
    report_before = cleaner.generate_quality_report(df_corrupted)
    print(f" - Initial Duplicate Rows: {report_before['duplicate_rows']}")
    print(f" - Missing Values ('mean radius'): {report_before['missing_counts'].get('mean radius', 0)}")
    print(f" - Outliers Detected ('mean texture'): {report_before['outlier_counts'].get('mean texture', 0)}")

    cleaned_df = cleaner.fit_transform(df_corrupted, clip_outliers=True)
    report_after = cleaner.generate_quality_report(cleaned_df)
    print(f" - Post-Cleaning Missing Values ('mean radius'): {report_after['missing_counts'].get('mean radius', 0)}")
    print(" -> Data Cleaning completed successfully.")

    # 3. Exploratory Data Analysis (EDA) Visualization
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n[Step 3] Generating EDA Visualizations in '{output_dir}'...")

    plot_missingness(df_corrupted, output_path=os.path.join(output_dir, "missingness.png"))
    plot_distributions(cleaned_df, columns=["mean radius", "mean texture", "mean perimeter"], output_path=os.path.join(output_dir, "distributions.png"))
    plot_correlation_matrix(cleaned_df.iloc[:, :8], output_path=os.path.join(output_dir, "correlations.png"))
    print(" -> Generated missingness, distribution, and correlation plots.")

    # 4. Algorithmic Bias & Representation Audit
    print("\n[Step 4] Executing Algorithmic Bias & Representation Audit...")
    auditor = BiasAuditor(
        protected_attribute="patient_group",
        privileged_group_value="Group_A",
        unprivileged_group_value="Group_B",
    )

    rep_audit = auditor.audit_dataset_representation(cleaned_df)
    print(f" - Group_A (Privileged) Ratio: {rep_audit['privileged_ratio']:.2%}")
    print(f" - Group_B (Unprivileged) Ratio: {rep_audit['unprivileged_ratio']:.2%}")
    print(f" - Representation Disparity: {rep_audit['representation_disparity']:.4f}")

    # Simulated diagnostic model predictions
    y_true = cleaned_df["diagnosis"]
    # Simulated model with slight group disparity
    y_pred = y_true.copy()
    bias_mask = (cleaned_df["patient_group"] == "Group_B") & (np.random.rand(len(cleaned_df)) < 0.15)
    y_pred.loc[bias_mask] = 1 - y_pred.loc[bias_mask]

    fairness_metrics = auditor.evaluate_fairness_metrics(cleaned_df, y_true, y_pred)
    print("\n Fairness & Disparate Impact Results:")
    print(f" - Disparate Impact Ratio: {fairness_metrics['disparate_impact_ratio']:.4f}")
    print(f" - Demographic Parity Difference: {fairness_metrics['demographic_parity_difference']:.4f}")
    print(f" - Equalized Odds Difference: {fairness_metrics['equalized_odds_difference']:.4f}")
    print(f" - Meets 4/5ths (80%) Rule: {fairness_metrics['meets_four_fifths_rule']}")

    print("\n" + "=" * 70)
    print("PROJECT 1 COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
