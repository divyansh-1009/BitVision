"""Model loading, prediction, and evaluation utilities."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)


def list_models(directory: str | Path) -> list[str]:
    """Return sorted list of ``.pkl`` filenames in *directory*."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    return sorted(
        f.name
        for f in directory.iterdir()
        if f.suffix.lower() == ".pkl"
    )


def load_model(filepath: str | Path):
    """Load a scikit-learn compatible model via joblib."""
    filepath = Path(filepath)
    if not filepath.is_file():
        raise FileNotFoundError(f"Model file not found: {filepath}")
    return joblib.load(filepath)


def predict(model, features_df: pd.DataFrame) -> np.ndarray:
    """Run inference and return predictions as a 1-D array."""
    return np.asarray(model.predict(features_df)).ravel()


def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Compute regression evaluation metrics."""
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "MAPE": float(mean_absolute_percentage_error(actual, predicted)) * 100,
        "R2": float(r2_score(actual, predicted)),
    }
