"""Model discovery, loading, inference, and metric helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PRIMARY_MODEL_NAME = "bitcoin_ridge_model.pkl"


def list_models(models_dir: Path) -> list[str]:
    if not models_dir.exists():
        return []
    names = sorted([p.name for p in models_dir.glob("*.pkl") if p.is_file()])
    if PRIMARY_MODEL_NAME in names:
        names.remove(PRIMARY_MODEL_NAME)
        names.insert(0, PRIMARY_MODEL_NAME)
    return names


def choose_default_model(available_models: list[str]) -> str | None:
    if not available_models:
        return None
    if PRIMARY_MODEL_NAME in available_models:
        return PRIMARY_MODEL_NAME
    return available_models[0]


def load_model(path: Path) -> Any:
    return joblib.load(path)


def align_features_for_model(model: Any, features: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    expected = list(getattr(model, "feature_names_in_", []))
    if not expected:
        numeric = features.select_dtypes(include="number")
        return numeric, []

    aligned_source = features.copy()
    lower_lookup = {col.lower(): col for col in aligned_source.columns}
    aliases = {
        "Return_1d": ["return_1d"],
        "Return_3d": ["return_3d"],
        "SMA_7_diff": ["sma_7_diff"],
        "Volatility_7": ["volatility_7d", "volatility_7"],
    }
    for target, candidates in aliases.items():
        if target in aligned_source.columns:
            continue
        for candidate in candidates:
            source_col = lower_lookup.get(candidate.lower())
            if source_col and source_col in aligned_source.columns:
                aligned_source[target] = pd.to_numeric(aligned_source[source_col], errors="coerce")
                break

    missing = [col for col in expected if col not in aligned_source.columns]
    if not missing:
        for col in expected:
            aligned_source[col] = pd.to_numeric(aligned_source[col], errors="coerce")
    aligned = aligned_source.reindex(columns=expected)
    return aligned, missing


def predict(model: Any, features: pd.DataFrame) -> np.ndarray:
    preds = model.predict(features)
    return np.asarray(preds)


def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    mae = mean_absolute_error(actual, predicted)
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    mape = float(np.nanmean(np.abs((actual - predicted) / np.where(actual == 0, np.nan, actual))) * 100)
    r2 = r2_score(actual, predicted)
    return {"MAE": float(mae), "RMSE": rmse, "MAPE": mape, "R2": float(r2)}
