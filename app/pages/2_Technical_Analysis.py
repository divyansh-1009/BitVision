"""Technical indicator visualisations with tunable parameters."""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.components.charts import technical_subplot
from app.components.metrics import empty_state
from app.utils.config import (
    LAYOUT,
    PAGE_ICON,
    PAGE_TITLE,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)
from src.data_loader import list_data_files, load_raw_data

st.set_page_config(page_title=f"{PAGE_TITLE} — Technical Analysis", page_icon=PAGE_ICON, layout=LAYOUT)

st.title("Technical Analysis")
st.caption("RSI, MACD, Bollinger Bands, Stochastic Oscillator, ATR, and OBV")

# ── Data loading ─────────────────────────────────────────────────────────────

raw_files = list_data_files(RAW_DATA_DIR)
processed_files = list_data_files(PROCESSED_DATA_DIR)

if not raw_files and not processed_files:
    empty_state(
        "No data available",
        "Add CSV files to `data/raw/` or `data/processed/` to compute technical indicators.\n\n"
        "**Required columns:** `Date, Open, High, Low, Close, Volume`\n\n"
        "Indicators will be computed on-the-fly from raw OHLC data.",
    )
    st.stop()

all_files: list[tuple[str, Path]] = []
for f in processed_files:
    all_files.append((f"[processed] {f}", PROCESSED_DATA_DIR / f))
for f in raw_files:
    all_files.append((f"[raw] {f}", RAW_DATA_DIR / f))

# ── Sidebar ──────────────────────────────────────────────────────────────────

INDICATOR_OPTIONS = ["RSI", "MACD", "Bollinger Bands", "Stochastic", "ATR", "OBV"]

with st.sidebar:
    st.header("Indicator Settings")

    selected_label = st.selectbox("Data file", [l for l, _ in all_files], key="ta_file")
    selected_path = dict(all_files)[selected_label]

    indicators = st.multiselect(
        "Indicators",
        INDICATOR_OPTIONS,
        default=["RSI", "MACD"],
    )

    st.subheader("Parameters")

    rsi_period = st.slider("RSI period", 5, 30, 14) if "RSI" in indicators else 14

    if "MACD" in indicators:
        macd_fast = st.slider("MACD fast", 5, 20, 12)
        macd_slow = st.slider("MACD slow", 15, 40, 26)
        macd_signal = st.slider("MACD signal", 3, 15, 9)
    else:
        macd_fast, macd_slow, macd_signal = 12, 26, 9

    if "Bollinger Bands" in indicators:
        bb_window = st.slider("BB window", 5, 50, 20, key="ta_bb_w")
        bb_std = st.slider("BB std dev", 1.0, 3.0, 2.0, 0.5, key="ta_bb_s")
    else:
        bb_window, bb_std = 20, 2.0

    if "Stochastic" in indicators:
        stoch_k = st.slider("%K period", 5, 21, 14)
        stoch_d = st.slider("%D period", 2, 7, 3)
    else:
        stoch_k, stoch_d = 14, 3

    atr_period = st.slider("ATR period", 5, 30, 14) if "ATR" in indicators else 14

# ── Load & display ───────────────────────────────────────────────────────────

df = load_raw_data(selected_path)

import pandas as pd

st.subheader("Date Range")
col1, col2 = st.columns(2)
min_date = df["Date"].min().date()
max_date = df["Date"].max().date()
start = col1.date_input("Start", value=min_date, min_value=min_date, max_value=max_date, key="ta_start")
end = col2.date_input("End", value=max_date, min_value=min_date, max_value=max_date, key="ta_end")

mask = (df["Date"] >= pd.Timestamp(start)) & (df["Date"] <= pd.Timestamp(end))
filtered = df.loc[mask].copy()

if filtered.empty:
    st.warning("No data in the selected date range.")
    st.stop()

if not indicators:
    st.info("Select at least one indicator from the sidebar.")
    st.stop()

fig = technical_subplot(
    filtered,
    indicators,
    rsi_period=rsi_period,
    macd_fast=macd_fast,
    macd_slow=macd_slow,
    macd_signal=macd_signal,
    bb_window=bb_window,
    bb_std=bb_std,
    stoch_k=stoch_k,
    stoch_d=stoch_d,
    atr_period=atr_period,
)
st.plotly_chart(fig, use_container_width=True)
