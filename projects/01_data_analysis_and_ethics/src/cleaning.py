"""
Data Cleaning & Quality Pipeline Module.

Provides robust cleaning, missing value imputation, outlier handling,
and data quality reporting routines for tabular data.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


class DataCleaner:
    """Production-grade data cleaner for structured tabular data."""

    def __init__(
        self,
        numeric_impute_strategy: str = "median",
        categorical_impute_strategy: str = "most_frequent",
        outlier_method: str = "iqr",
        iqr_multiplier: float = 1.5,
        z_score_threshold: float = 3.0,
    ) -> None:
        self.numeric_impute_strategy = numeric_impute_strategy
        self.categorical_impute_strategy = categorical_impute_strategy
        self.outlier_method = outlier_method
        self.iqr_multiplier = iqr_multiplier
        self.z_score_threshold = z_score_threshold
        self.imputation_values_: Dict[str, Union[float, str]] = {}
        self.bounds_: Dict[str, Tuple[float, float]] = {}

    def fit(self, df: pd.DataFrame) -> "DataCleaner":
        """Compute imputation statistics and outlier thresholds from training data."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns

        # Compute imputation values
        for col in numeric_cols:
            if self.numeric_impute_strategy == "median":
                self.imputation_values_[col] = float(df[col].median())
            elif self.numeric_impute_strategy == "mean":
                self.imputation_values_[col] = float(df[col].mean())
            else:
                self.imputation_values_[col] = 0.0

            # Compute outlier bounds
            if self.outlier_method == "iqr":
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - self.iqr_multiplier * iqr
                upper = q3 + self.iqr_multiplier * iqr
                self.bounds_[col] = (float(lower), float(upper))
            elif self.outlier_method == "z_score":
                mean = df[col].mean()
                std = df[col].std()
                if std > 0:
                    lower = mean - self.z_score_threshold * std
                    upper = mean + self.z_score_threshold * std
                else:
                    lower, upper = df[col].min(), df[col].max()
                self.bounds_[col] = (float(lower), float(upper))

        for col in categorical_cols:
            if self.categorical_impute_strategy == "most_frequent":
                mode_val = df[col].mode()
                self.imputation_values_[col] = str(mode_val.iloc[0]) if not mode_val.empty else "Missing"
            else:
                self.imputation_values_[col] = "Missing"

        return self

    def transform(self, df: pd.DataFrame, clip_outliers: bool = True) -> pd.DataFrame:
        """Apply fitted cleaning, imputation, and outlier treatment to a DataFrame."""
        cleaned_df = df.copy()

        # Fill missing values using fitted statistics
        for col, val in self.imputation_values_.items():
            if col in cleaned_df.columns:
                cleaned_df[col] = cleaned_df[col].fillna(val)

        # Clip outliers for numeric columns
        if clip_outliers:
            for col, (lower, upper) in self.bounds_.items():
                if col in cleaned_df.columns:
                    cleaned_df[col] = cleaned_df[col].clip(lower=lower, upper=upper)

        return cleaned_df

    def fit_transform(self, df: pd.DataFrame, clip_outliers: bool = True) -> pd.DataFrame:
        """Fit cleaner on df and return cleaned DataFrame."""
        return self.fit(df).transform(df, clip_outliers=clip_outliers)

    def generate_quality_report(self, df: pd.DataFrame) -> Dict[str, Union[int, float, Dict]]:
        """Generate summary report detailing missing values, data types, and potential outliers."""
        total_rows = len(df)
        total_cols = len(df.columns)
        missing_counts = df.isnull().sum().to_dict()
        missing_percentages = (df.isnull().sum() / total_rows * 100).round(2).to_dict()

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_counts = {}
        for col in numeric_cols:
            if col in self.bounds_:
                lower, upper = self.bounds_[col]
                outliers = df[(df[col] < lower) | (df[col] > upper)]
                outlier_counts[col] = len(outliers)

        return {
            "total_rows": total_rows,
            "total_columns": total_cols,
            "duplicate_rows": int(df.duplicated().sum()),
            "missing_counts": missing_counts,
            "missing_percentages": missing_percentages,
            "outlier_counts": outlier_counts,
        }
