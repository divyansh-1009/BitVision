"""BitVision Dashboard -- main entry point for the Streamlit app."""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.components.charts import sparkline_chart
from app.components.metrics import empty_state, render_metric_row
from app.utils.config import (
    LAYOUT,
    MODELS_DIR,
    PAGE_ICON,
    PAGE_TITLE,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)
from src.data_loader import list_data_files, load_raw_data
from src.inference import list_models

st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout=LAYOUT)

st.title(f"{PAGE_ICON} BitVision")
st.caption("Bitcoin Price Prediction & Analysis Dashboard")

st.divider()

# ── Load data ────────────────────────────────────────────────────────────────

raw_files = list_data_files(RAW_DATA_DIR)
processed_files = list_data_files(PROCESSED_DATA_DIR)
data_available = bool(raw_files or processed_files)

if not data_available:
    empty_state(
        "No data found",
        "Place your raw OHLC CSV files in `data/raw/` and/or processed CSVs in "
        "`data/processed/`.\n\n"
        "**Expected columns (raw):** `Date, Open, High, Low, Close, Volume`\n\n"
        "```\ndata/\n  raw/\n    btc_ohlc.csv\n  processed/\n    btc_features.csv\n```",
    )
    st.stop()

# Prefer processed data; fall back to raw
if processed_files:
    data_path = PROCESSED_DATA_DIR / processed_files[0]
else:
    data_path = RAW_DATA_DIR / raw_files[0]

df = load_raw_data(data_path)

# ── Metric cards ─────────────────────────────────────────────────────────────

latest = df.iloc[-1]
prev = df.iloc[-2] if len(df) > 1 else latest

price_change_24h = (latest["Close"] - prev["Close"]) / prev["Close"] * 100 if prev["Close"] else 0

week_ago = df.iloc[-7] if len(df) >= 7 else df.iloc[0]
price_change_7d = (latest["Close"] - week_ago["Close"]) / week_ago["Close"] * 100 if week_ago["Close"] else 0

month_ago = df.iloc[-30] if len(df) >= 30 else df.iloc[0]
price_change_30d = (latest["Close"] - month_ago["Close"]) / month_ago["Close"] * 100 if month_ago["Close"] else 0

render_metric_row(
    [
        {
            "label": "Latest Close",
            "value": f"${latest['Close']:,.2f}",
            "delta": f"{price_change_24h:+.2f}%",
            "delta_color": "normal",
        },
        {
            "label": "7-Day Change",
            "value": f"{price_change_7d:+.2f}%",
            "delta": f"${latest['Close'] - week_ago['Close']:+,.2f}",
        },
        {
            "label": "30-Day Change",
            "value": f"{price_change_30d:+.2f}%",
            "delta": f"${latest['Close'] - month_ago['Close']:+,.2f}",
        },
        {
            "label": "Volume",
            "value": f"{latest['Volume']:,.0f}",
        },
    ]
)

# ── Sparkline ────────────────────────────────────────────────────────────────

st.subheader("Price — Last 30 Days")
recent = df.tail(30)
st.plotly_chart(sparkline_chart(recent["Date"], recent["Close"]), use_container_width=True)

# ── Quick stats ──────────────────────────────────────────────────────────────

st.subheader("Quick Stats")

year_data = df.tail(365)

col1, col2, col3 = st.columns(3)
col1.metric("52-Week High", f"${year_data['High'].max():,.2f}")
col2.metric("52-Week Low", f"${year_data['Low'].min():,.2f}")
col3.metric("All-Time High", f"${df['High'].max():,.2f}")

col4, col5, col6 = st.columns(3)
col4.metric("Avg Daily Volume (30d)", f"{recent['Volume'].mean():,.0f}")
col5.metric("Latest Date", str(latest["Date"].date()) if hasattr(latest["Date"], "date") else str(latest["Date"]))
col6.metric("Total Records", f"{len(df):,}")

# ── Prediction summary ──────────────────────────────────────────────────────

st.divider()
st.subheader("Prediction Summary")

available_models = list_models(MODELS_DIR)
if not available_models:
    st.info(
        "No trained models found. Place `.pkl` model files in `models/` "
        "to see prediction summaries here.",
        icon="🤖",
    )
else:
    st.success(
        f"**{len(available_models)} model(s) available:** {', '.join(available_models)}. "
        "Visit the **Predictions** page for detailed forecasts.",
        icon="✅",
    )
