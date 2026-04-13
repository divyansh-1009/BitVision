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
from app.utils.data_loader import list_data_files, load_raw_data, resolve_data_file

st.set_page_config(page_title=f"{PAGE_TITLE} — Technical Analysis", page_icon=PAGE_ICON, layout=LAYOUT)

st.title("Technical Analysis")
st.caption("Price panel overlays (SMA, EMA, Bollinger) plus RSI and MACD subplots")

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
    all_files.append((f"[processed] {f}", resolve_data_file(PROCESSED_DATA_DIR, f)))
for f in raw_files:
    all_files.append((f"[raw] {f}", resolve_data_file(RAW_DATA_DIR, f)))

# ── Sidebar ──────────────────────────────────────────────────────────────────

INDICATOR_OPTIONS = ["RSI", "MACD"]

with st.sidebar:
    st.header("Indicator Settings")

    selected_label = st.selectbox("Data file", [l for l, _ in all_files], key="ta_file")
    selected_path = dict(all_files)[selected_label]

    st.subheader("Moving Averages")
    sma_options = st.multiselect("SMA windows", [20, 50, 100, 200], default=[], key="ta_sma")
    ema_options = st.multiselect("EMA spans", [12, 26, 50], default=[], key="ta_ema")

    st.subheader("Bollinger Bands")
    show_bb = st.checkbox(
        "Show Bollinger Bands",
        value=False,
        key="ta_show_bb",
        help="Bands are middle ± 2× the rolling standard deviation of closes over the window (σ comes from your data).",
    )
    bb_window = st.slider("BB window", 5, 50, 20, key="ta_bb_w") if show_bb else 20

    indicators = st.multiselect(
        "Indicators",
        INDICATOR_OPTIONS,
        default=["RSI", "MACD"],
    )

    st.subheader("Parameters")

    rsi_period = st.slider("RSI period", 5, 30, 14) if "RSI" in indicators else 14

    if "MACD" in indicators:
        macd_fast = st.slider(
            "MACD fast",
            5,
            20,
            12,
            help="Span of the faster EMA of close. Not drawn alone — it enters the MACD line as the first term.",
        )
        macd_slow = st.slider(
            "MACD slow",
            15,
            40,
            26,
            help="Span of the slower EMA of close. There is no separate line for it; it is subtracted inside the yellow MACD line.",
        )
        macd_signal = st.slider(
            "MACD signal",
            3,
            15,
            9,
            help="Span of the EMA applied to the MACD line to produce the blue signal line.",
        )
    else:
        macd_fast, macd_slow, macd_signal = 12, 26, 9

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
    sma_windows=sma_options or None,
    ema_spans=ema_options or None,
    show_bollinger=show_bb,
    bb_window=bb_window,
)
st.plotly_chart(fig, width="stretch")
