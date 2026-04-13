"""Model predictions, forecasts, and accuracy metrics."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.components.charts import actual_vs_predicted
from app.components.metrics import empty_state, render_metric_row
from app.utils.config import (
    LAYOUT,
    MODELS_DIR,
    PAGE_ICON,
    PAGE_TITLE,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)
from app.utils.data_loader import list_data_files, load_processed_data, load_raw_data, resolve_data_file
from app.utils.feature_engineering import build_features
from app.utils.inference import (
    choose_default_model,
    compute_metrics,
    list_models,
    load_model,
    prediction_comparison_table,
)

st.set_page_config(page_title=f"{PAGE_TITLE} — Predictions", page_icon=PAGE_ICON, layout=LAYOUT)

st.title("Predictions")
st.caption("Model forecasts, residual analysis, and accuracy metrics")

# ── Check prerequisites ─────────────────────────────────────────────────────

available_models = list_models(MODELS_DIR)

if not available_models:
    empty_state(
        "No models found",
        "Place trained `.pkl` model files (scikit-learn compatible, saved via `joblib`) "
        "in the `models/` directory.\n\n"
        "```python\nimport joblib\njoblib.dump(model, 'models/my_model.pkl')\n```",
    )
    st.stop()

processed_files = list_data_files(PROCESSED_DATA_DIR)
raw_files = list_data_files(RAW_DATA_DIR)

if not processed_files and not raw_files:
    empty_state(
        "No data available",
        "Add CSV files to `data/raw/` or `data/processed/` so the model has data to predict on.",
    )
    st.stop()

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Prediction Settings")
    default_model = choose_default_model(available_models)
    selected_model_name = default_model if default_model else available_models[0]

    if processed_files:
        data_file = st.selectbox("Data file", processed_files, key="pred_data")
        data_path = resolve_data_file(PROCESSED_DATA_DIR, data_file)
    else:
        data_file = st.selectbox("Data file (raw)", raw_files, key="pred_data_raw")
        data_path = resolve_data_file(RAW_DATA_DIR, data_file)

# ── Load model & data ───────────────────────────────────────────────────────

model = load_model(MODELS_DIR / selected_model_name)

if str(data_path).startswith(str(PROCESSED_DATA_DIR)):
    df = load_processed_data(data_path)
else:
    df = load_raw_data(data_path)
    df = build_features(df)

if "Date" not in df.columns or "Close" not in df.columns:
    st.error("Data must contain `Date` and `Close` columns.")
    st.stop()

# ── Prepare features & predict ───────────────────────────────────────────────

try:
    comparison_df, missing_features, df, empty_reason, _pred_series = prediction_comparison_table(df, model)
except RuntimeError as e:
    st.error(str(e))
    st.stop()

if missing_features:
    if missing_features == ["no_numeric_feature_columns"]:
        st.error("No numeric feature columns found for prediction.")
    else:
        st.error(
            "The selected model expects missing feature columns: "
            + ", ".join(missing_features[:15])
            + ("..." if len(missing_features) > 15 else "")
        )
    st.stop()

if comparison_df.empty:
    # Short testing files can produce all-NaN rolling features (e.g., 7-day windows).
    # Retry by prepending training-history rows, then keep only the selected file rows.
    has_ohlc = {"Date", "Open", "High", "Low", "Close"}.issubset(set(df.columns))
    testing_like = "test" in data_file.lower()
    training_candidates = [f for f in raw_files if "train" in f.lower() and f != data_file]

    if has_ohlc and testing_like and training_candidates:
        history_path = resolve_data_file(RAW_DATA_DIR, training_candidates[0])
        history_df = load_raw_data(history_path)

        if not history_df.empty:
            selected_raw = load_raw_data(data_path)
            selected_raw = selected_raw.copy()
            selected_raw["__is_selected__"] = 1
            history_df = history_df.copy()
            history_df["__is_selected__"] = 0

            combined = pd.concat([history_df, selected_raw], ignore_index=True)
            combined = combined.sort_values("Date").drop_duplicates(subset=["Date"], keep="last")
            enriched = build_features(combined)

            selected_enriched = enriched[enriched["__is_selected__"] == 1].copy()
            if "__is_selected__" in selected_enriched.columns:
                selected_enriched = selected_enriched.drop(columns=["__is_selected__"])

            try:
                comparison_df, missing_features, df, empty_reason, _pred_series = prediction_comparison_table(
                    selected_enriched, model
                )
            except RuntimeError as e:
                st.error(str(e))
                st.stop()

            if missing_features:
                st.error(
                    "The selected model expects missing feature columns: "
                    + ", ".join(missing_features[:15])
                    + ("..." if len(missing_features) > 15 else "")
                )
                st.stop()

if comparison_df.empty:
    if empty_reason == "no_comparison_rows":
        st.error("No valid rows remain after aligning next-close targets.")
    else:
        st.error(
            "All rows contain NaN values after dropping missing data. "
            "For short testing files, include enough prior history (or use combined train+test data) "
            "so rolling features can be computed."
        )
    st.stop()

# ── Timeframe filter ────────────────────────────────────────────────────────

st.subheader("Timeframe")
tf_col1, tf_col2 = st.columns(2)
min_date = comparison_df["Date"].min().date()
max_date = comparison_df["Date"].max().date()
start_date = tf_col1.date_input(
    "Start date",
    value=min_date,
    min_value=min_date,
    max_value=max_date,
    key="pred_start_date",
)
end_date = tf_col2.date_input(
    "End date",
    value=max_date,
    min_value=min_date,
    max_value=max_date,
    key="pred_end_date",
)

if start_date > end_date:
    st.error("Start date must be before or equal to end date.")
    st.stop()

mask = (
    (comparison_df["Date"] >= pd.Timestamp(start_date))
    & (comparison_df["Date"] <= pd.Timestamp(end_date))
)
comparison_df = comparison_df.loc[mask]

if comparison_df.empty:
    st.warning("No prediction rows available in the selected timeframe.")
    st.stop()

plot_idx = comparison_df.index
dates = comparison_df["Date"]
actual = comparison_df["ActualNextClose"].to_numpy(dtype=float)
predicted = comparison_df["PredictedNextClose"].to_numpy(dtype=float)

# ── Actual vs Predicted ──────────────────────────────────────────────────────

st.subheader("Actual Next Close vs. Predicted Next Close")
st.plotly_chart(
    actual_vs_predicted(dates, pd.Series(actual, index=plot_idx), pd.Series(predicted, index=plot_idx)),
    width="stretch",
)

# ── Performance metrics ──────────────────────────────────────────────────────

st.subheader("Performance Metrics")
metrics = compute_metrics(actual, predicted)
render_metric_row(
    [
        {"label": "MAE", "value": f"${metrics['MAE']:,.2f}"},
        {"label": "RMSE", "value": f"${metrics['RMSE']:,.2f}"},
        {"label": "MAPE", "value": f"{metrics['MAPE']:.2f}%"},
        {"label": "R-squared", "value": f"{metrics['R2']:.4f}"},
    ]
)

# ── Error summary ────────────────────────────────────────────────────────────

st.subheader("Error Summary")
residuals = actual - predicted
err_df = pd.DataFrame(
    {
        "Statistic": ["Mean Error", "Median Error", "Std Error", "Min Error", "Max Error"],
        "Value": [
            f"${np.mean(residuals):,.2f}",
            f"${np.median(residuals):,.2f}",
            f"${np.std(residuals):,.2f}",
            f"${np.min(residuals):,.2f}",
            f"${np.max(residuals):,.2f}",
        ],
    }
)
st.dataframe(err_df, width="stretch", hide_index=True)
