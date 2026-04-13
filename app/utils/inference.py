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


def prediction_comparison_table(
    df: pd.DataFrame,
    model: Any,
) -> tuple[pd.DataFrame, list[str], pd.DataFrame, str | None, pd.Series]:
    """Align numeric features, predict next-period return, map to next close.

    Expects ``df`` with ``Date``, ``Close``, and numeric feature columns (raw paths
    should call :func:`app.utils.feature_engineering.build_features` first).

    Returns ``(comparison_df, missing_feature_names, df_used, empty_reason, pred_next_close_series)``.
    ``comparison_df`` has columns ``Date``, ``ActualNextClose``, ``PredictedNextClose``
    (after ``dropna`` on rows with no actual next close). ``df_used`` is the frame
    after any internal ``build_features`` retry for column alignment.

    ``pred_next_close_series`` is **PredictedNextClose** for every ``valid_idx`` row
    (including the last bar, where ``ActualNextClose`` may be unknown). It is empty
    when there are no clean feature rows.

    ``empty_reason`` is ``None`` on success. If ``comparison_df`` is empty:
    ``"no_clean_feature_rows"`` when no rows survive feature ``dropna``;
    ``"no_comparison_rows"`` when aligned predictions drop out (e.g. one row);
    otherwise ``None`` when failure is expressed via ``missing_feature_names``.
    """
    from app.utils.feature_engineering import build_features

    empty_series = pd.Series(dtype=float)

    work = df
    if "Date" not in work.columns or "Close" not in work.columns:
        return pd.DataFrame(), ["Date_or_Close"], work, None, empty_series

    non_feature_cols = {"Date", "Close"}
    feature_cols = [c for c in work.columns if c not in non_feature_cols]
    numeric_features = work[feature_cols].select_dtypes(include="number")

    if numeric_features.empty:
        return pd.DataFrame(), ["no_numeric_feature_columns"], work, None, empty_series

    aligned_features, missing_features = align_features_for_model(model, numeric_features)
    if missing_features:
        has_ohlc = {"Open", "High", "Low", "Close"}.issubset(set(work.columns))
        if has_ohlc:
            work = build_features(work)
            feature_cols = [c for c in work.columns if c not in non_feature_cols]
            numeric_features = work[feature_cols].select_dtypes(include="number")
            aligned_features, missing_features = align_features_for_model(model, numeric_features)

    if missing_features:
        return pd.DataFrame(), missing_features, work, None, empty_series

    features_clean = aligned_features.dropna()
    if features_clean.empty:
        return pd.DataFrame(), [], work, "no_clean_feature_rows", empty_series

    valid_idx = features_clean.index
    dates = work.loc[valid_idx, "Date"]
    current_close = pd.to_numeric(work.loc[valid_idx, "Close"], errors="coerce")

    try:
        pred_next_return = predict(model, features_clean)
    except Exception as e:
        raise RuntimeError(f"Prediction failed: {e}") from e

    pred_next_close = current_close * (1 + pd.Series(pred_next_return, index=valid_idx))
    actual_next_close = pd.to_numeric(work["Close"], errors="coerce").shift(-1).loc[valid_idx]

    comparison_df = pd.DataFrame(
        {
            "Date": dates,
            "ActualNextClose": actual_next_close,
            "PredictedNextClose": pred_next_close,
        },
        index=valid_idx,
    ).dropna()

    if comparison_df.empty:
        return pd.DataFrame(), [], work, "no_comparison_rows", pred_next_close

    return comparison_df, [], work, None, pred_next_close


def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    mae = mean_absolute_error(actual, predicted)
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    mape = float(np.nanmean(np.abs((actual - predicted) / np.where(actual == 0, np.nan, actual))) * 100)
    r2 = r2_score(actual, predicted)
    return {"MAE": float(mae), "RMSE": rmse, "MAPE": mape, "R2": float(r2)}
