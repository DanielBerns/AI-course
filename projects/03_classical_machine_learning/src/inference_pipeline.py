"""
Production Inference Pipeline Service.

Loads serialized pipeline artifacts (`joblib`) and exposes low-latency
batch and single-sample prediction interfaces.
"""

import os
from typing import Dict, List, Union
import joblib
import numpy as np
import pandas as pd


class ClassicalInferenceService:
    """Production inference engine for serialized ML pipelines."""

    def __init__(self, model_path: str) -> None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Pipeline artifact model not found at path '{model_path}'.")
        self.model_path = model_path
        self.pipeline = joblib.load(model_path)

    def predict(self, input_data: Union[pd.DataFrame, Dict, List[Dict]]) -> np.ndarray:
        """Make class predictions for tabular inputs."""
        df = self._format_input(input_data)
        return self.pipeline.predict(df)

    def predict_proba(self, input_data: Union[pd.DataFrame, Dict, List[Dict]]) -> np.ndarray:
        """Return class probability distributions."""
        df = self._format_input(input_data)
        if hasattr(self.pipeline, "predict_proba"):
            return self.pipeline.predict_proba(df)
        elif hasattr(self.pipeline, "decision_function"):
            return self.pipeline.decision_function(df)
        else:
            raise AttributeError("Loaded pipeline estimator does not support probability estimation.")

    def _format_input(self, input_data: Union[pd.DataFrame, Dict, List[Dict]]) -> pd.DataFrame:
        if isinstance(input_data, pd.DataFrame):
            return input_data
        elif isinstance(input_data, dict):
            return pd.DataFrame([input_data])
        elif isinstance(input_data, list):
            return pd.DataFrame(input_data)
        else:
            raise TypeError(f"Unsupported input type '{type(input_data)}'. Pass DataFrame, dict, or list of dicts.")
