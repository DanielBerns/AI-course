"""
Algorithmic Bias & Fairness Audit Module.

Evaluates datasets and model predictions for demographic representation gaps,
disparate impact, statistical parity, and equalized odds metrics.
"""

from typing import Dict, Any, Union
import numpy as np
import pandas as pd


class BiasAuditor:
    """Audit tabular dataset demographic representation and classifier fairness metrics."""

    def __init__(
        self,
        protected_attribute: str,
        privileged_group_value: Any,
        unprivileged_group_value: Any,
    ) -> None:
        self.protected_attribute = protected_attribute
        self.privileged_group_value = privileged_group_value
        self.unprivileged_group_value = unprivileged_group_value

    def audit_dataset_representation(self, df: pd.DataFrame) -> Dict[str, Union[int, float]]:
        """Evaluate demographic breakdown and representation proportions."""
        if self.protected_attribute not in df.columns:
            raise KeyError(f"Protected attribute column '{self.protected_attribute}' not found in DataFrame.")

        counts = df[self.protected_attribute].value_counts().to_dict()
        total = len(df)

        priv_count = counts.get(self.privileged_group_value, 0)
        unread_count = counts.get(self.unprivileged_group_value, 0)

        return {
            "total_samples": total,
            "privileged_count": priv_count,
            "privileged_ratio": round(priv_count / total, 4) if total > 0 else 0.0,
            "unprivileged_count": unread_count,
            "unprivileged_ratio": round(unread_count / total, 4) if total > 0 else 0.0,
            "representation_disparity": round(abs(priv_count - unread_count) / total, 4) if total > 0 else 0.0,
        }

    def evaluate_fairness_metrics(
        self,
        df: pd.DataFrame,
        y_true: Union[pd.Series, np.ndarray],
        y_pred: Union[pd.Series, np.ndarray],
    ) -> Dict[str, float]:
        """
        Compute Disparate Impact Ratio, Demographic Parity Difference,
        and Equalized Odds Difference across demographic groups.
        """
        df_audit = df.copy()
        df_audit["_y_true"] = np.array(y_true)
        df_audit["_y_pred"] = np.array(y_pred)

        priv_df = df_audit[df_audit[self.protected_attribute] == self.privileged_group_value]
        unpriv_df = df_audit[df_audit[self.protected_attribute] == self.unprivileged_group_value]

        # Selection Rates (P(y_pred = 1))
        priv_selection_rate = float(priv_df["_y_pred"].mean()) if len(priv_df) > 0 else 0.0
        unpriv_selection_rate = float(unpriv_df["_y_pred"].mean()) if len(unpriv_df) > 0 else 0.0

        # Disparate Impact Ratio (80% rule: threshold typically 0.80)
        disparate_impact = unpriv_selection_rate / (priv_selection_rate + 1e-8)
        demographic_parity_diff = unpriv_selection_rate - priv_selection_rate

        # True Positive Rates (Sensitivity)
        priv_tpr = float(
            (priv_df["_y_pred"] == 1)[priv_df["_y_true"] == 1].mean()
        ) if (priv_df["_y_true"] == 1).sum() > 0 else 0.0

        unpriv_tpr = float(
            (unpriv_df["_y_pred"] == 1)[unpriv_df["_y_true"] == 1].mean()
        ) if (unpriv_df["_y_true"] == 1).sum() > 0 else 0.0

        # False Positive Rates
        priv_fpr = float(
            (priv_df["_y_pred"] == 1)[priv_df["_y_true"] == 0].mean()
        ) if (priv_df["_y_true"] == 0).sum() > 0 else 0.0

        unpriv_fpr = float(
            (unpriv_df["_y_pred"] == 1)[unpriv_df["_y_true"] == 0].mean()
        ) if (unpriv_df["_y_true"] == 0).sum() > 0 else 0.0

        equalized_odds_diff = abs(unpriv_tpr - priv_tpr) + abs(unpriv_fpr - priv_fpr)

        return {
            "privileged_selection_rate": round(priv_selection_rate, 4),
            "unprivileged_selection_rate": round(unpriv_selection_rate, 4),
            "disparate_impact_ratio": round(disparate_impact, 4),
            "demographic_parity_difference": round(demographic_parity_diff, 4),
            "privileged_tpr": round(priv_tpr, 4),
            "unprivileged_tpr": round(unpriv_tpr, 4),
            "equalized_odds_difference": round(equalized_odds_diff, 4),
            "meets_four_fifths_rule": bool(0.80 <= disparate_impact <= 1.25),
        }
