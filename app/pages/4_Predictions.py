"""Model predictions, forecasts, and accuracy metrics."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.components.charts import actual_vs_predicted, residual_chart
from app.components.metrics import empty_state, render_metric_row
from app.utils.config import (
    LAYOUT,
    MODELS_DIR,
    PAGE_ICON,
    PAGE_TITLE,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)
from src.data_loader import list_data_files, load_processed_data, load_raw_data
from src.feature_engineering import build_features
from src.inference import compute_metrics, list_models, load_model, predict

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
    selected_model_name = st.selectbox("Model", available_models)

    if processed_files:
        data_file = st.selectbox("Data file", processed_files, key="pred_data")
        data_path = PROCESSED_DATA_DIR / data_file
    else:
        data_file = st.selectbox("Data file (raw)", raw_files, key="pred_data_raw")
        data_path = RAW_DATA_DIR / data_file

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

# ── Prepare features ────────────────────────────────────────────────────────

non_feature_cols = {"Date", "Close"}
feature_cols = [c for c in df.columns if c not in non_feature_cols]
numeric_features = df[feature_cols].select_dtypes(include="number")

if numeric_features.empty:
    st.error("No numeric feature columns found for prediction.")
    st.stop()

features_clean = numeric_features.dropna()
if features_clean.empty:
    st.error("All rows contain NaN values after dropping missing data.")
    st.stop()

valid_idx = features_clean.index
dates = df.loc[valid_idx, "Date"]
actual = df.loc[valid_idx, "Close"].values

try:
    predicted = predict(model, features_clean)
except Exception as e:
    st.error(f"Prediction failed: {e}")
    st.stop()

# ── Actual vs Predicted ──────────────────────────────────────────────────────

st.subheader("Actual vs. Predicted")
st.plotly_chart(
    actual_vs_predicted(dates, pd.Series(actual, index=valid_idx), pd.Series(predicted, index=valid_idx)),
    use_container_width=True,
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

# ── Residual analysis ────────────────────────────────────────────────────────

st.subheader("Residual Analysis")

residuals = actual - predicted

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("**Residuals Over Time**")
    st.plotly_chart(
        residual_chart(dates, pd.Series(residuals, index=valid_idx)),
        use_container_width=True,
    )

with col_b:
    st.markdown("**Residual Distribution**")
    fig_hist = px.histogram(
        x=residuals,
        nbins=50,
        template="plotly_dark",
        color_discrete_sequence=["#f7931a"],
        labels={"x": "Residual"},
    )
    fig_hist.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=300)
    st.plotly_chart(fig_hist, use_container_width=True)

# ── Error summary ────────────────────────────────────────────────────────────

st.subheader("Error Summary")
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
st.dataframe(err_df, use_container_width=True, hide_index=True)
