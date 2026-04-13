"""Minimal OHLCV inference for testing the price model."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.components.metrics import empty_state
from app.utils.config import LAYOUT, MODELS_DIR, PAGE_ICON, PAGE_TITLE
from app.utils.data_loader import prepare_user_ohlc_df
from app.utils.feature_engineering import build_features
from app.utils.inference import (
    choose_default_model,
    list_models,
    load_model,
    prediction_comparison_table,
)

# Synthetic history: ramp prior rows toward the user's candle so rolling features (and model output)
# respond to OHLCV changes. The last row is exactly the user's inputs.
DUMMY_REPEAT_ROWS = 55


def _dataframe_from_single_ohlcv(open_: float, high: float, low: float, close: float, volume: float) -> pd.DataFrame:
    n = DUMMY_REPEAT_ROWS
    price_scale = np.linspace(0.97, 1.0, n)
    vol_scale = np.linspace(0.65, 1.0, n)
    df = pd.DataFrame(
        {
            "Open": float(open_) * price_scale,
            "High": float(high) * price_scale,
            "Low": float(low) * price_scale,
            "Close": float(close) * price_scale,
            "Volume": float(volume) * vol_scale,
        }
    )
    df["Date"] = pd.date_range("2020-01-01", periods=n, freq="D")
    return df


def _run_and_display(df: pd.DataFrame, model, *, batch: bool) -> None:
    try:
        df_f = build_features(df)
    except Exception as e:
        st.error(f"Feature build failed: {e}")
        return
    try:
        _comp, missing, _u, _er, pred_all = prediction_comparison_table(df_f, model)
    except RuntimeError as e:
        st.error(str(e))
        return
    if missing:
        st.error("Missing features: " + ", ".join(missing[:20]))
        return
    if pred_all.empty:
        st.error("No prediction (need enough rows for rolling features).")
        return
    if batch:
        out = df_f.loc[pred_all.index, ["Open", "High", "Low", "Close", "Volume"]].copy()
        out["PredictedNextClose"] = pred_all.values
        out = out.reset_index(drop=True)
        if len(out) > 500:
            st.caption(f"Showing first 500 of {len(out)} rows.")
            out = out.head(500)
        st.dataframe(out, use_container_width=True, hide_index=True)
    else:
        li = pred_all.index[-1]
        lp = float(pred_all.loc[li])
        st.metric("Predicted next close", f"${lp:,.2f}")


st.set_page_config(page_title=f"{PAGE_TITLE} — Inference", page_icon=PAGE_ICON, layout=LAYOUT)

st.title("Inference")
st.markdown(
    "Predict **Bitcoin** closes from the trained model using OHLCV data."
)

available_models = list_models(MODELS_DIR)
if not available_models:
    empty_state(
        "No models found",
        "Place trained `.pkl` files in the `models/` directory.",
    )
    st.stop()

selected_model_name = choose_default_model(available_models) or available_models[0]
model = load_model(MODELS_DIR / selected_model_name)

st.subheader("Test values")

with st.form("inf_single"):
    r1, r2, r3, r4, r5 = st.columns(5)
    with r1:
        v_open = st.number_input("Open", value=95000.0, step=10.0, format="%.2f")
    with r2:
        v_high = st.number_input("High", value=95100.0, step=10.0, format="%.2f")
    with r3:
        v_low = st.number_input("Low", value=94900.0, step=10.0, format="%.2f")
    with r4:
        v_close = st.number_input("Close", value=95050.0, step=10.0, format="%.2f")
    with r5:
        v_vol = st.number_input("Volume", value=1_000_000.0, step=10_000.0, format="%.0f")

    single_go = st.form_submit_button("Predict", type="primary")

if single_go:
    d0 = _dataframe_from_single_ohlcv(v_open, v_high, v_low, v_close, v_vol)
    _run_and_display(d0, model, batch=False)

st.divider()

st.subheader("Upload CSV")

csv_up = st.file_uploader(
    " ",
    type=["csv"],
    key="inf_csv_multi",
    label_visibility="collapsed",
)
csv_go = st.button("Predict on uploaded CSV", key="inf_csv_btn")

if csv_go:
    if csv_up is None:
        st.warning("Choose a CSV file first.")
    else:
        try:
            raw = pd.read_csv(csv_up)
            d1 = prepare_user_ohlc_df(raw, synthesize_date_if_missing=True, strict_ohlc_drop=True)
        except ValueError as e:
            st.error(str(e))
            d1 = None
        if d1 is not None and len(d1) < 40:
            st.error("CSV needs at least about **40 rows** so rolling features are valid.")
            d1 = None
        if d1 is not None:
            _run_and_display(d1, model, batch=True)
