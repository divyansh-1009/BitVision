"""Interactive price charts with overlays and volume."""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.components.charts import candlestick_chart
from app.components.metrics import empty_state
from app.utils.config import (
    LAYOUT,
    PAGE_ICON,
    PAGE_TITLE,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)
from app.utils.data_loader import list_data_files, load_raw_data, resolve_data_file

st.set_page_config(page_title=f"{PAGE_TITLE} — Price Charts", page_icon=PAGE_ICON, layout=LAYOUT)

st.title("Price Charts")
st.caption("Interactive OHLC, candlestick, and line charts with moving-average overlays")

# ── Data loading ─────────────────────────────────────────────────────────────

raw_files = list_data_files(RAW_DATA_DIR)
processed_files = list_data_files(PROCESSED_DATA_DIR)

if not raw_files and not processed_files:
    empty_state(
        "No data available",
        "Add CSV files to `data/raw/` or `data/processed/` to view price charts.\n\n"
        "**Required columns:** `Date, Open, High, Low, Close, Volume`",
    )
    st.stop()

all_files: list[tuple[str, Path]] = []
for f in processed_files:
    all_files.append((f"[processed] {f}", resolve_data_file(PROCESSED_DATA_DIR, f)))
for f in raw_files:
    all_files.append((f"[raw] {f}", resolve_data_file(RAW_DATA_DIR, f)))

# ── Sidebar controls ────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Chart Settings")

    selected_label = st.selectbox("Data file", [label for label, _ in all_files])
    selected_path = dict(all_files)[selected_label]

    chart_type = st.radio("Chart type", ["Candlestick", "Line"], horizontal=True)

    st.subheader("Moving Averages")
    sma_options = st.multiselect("SMA windows", [20, 50, 100, 200], default=[])
    ema_options = st.multiselect("EMA spans", [12, 26, 50], default=[])

    st.subheader("Bollinger Bands")
    show_bb = st.checkbox("Show Bollinger Bands", value=False)
    bb_window = st.slider("BB window", 5, 50, 20) if show_bb else 20
    bb_std = st.slider("BB std dev", 1.0, 3.0, 2.0, 0.5) if show_bb else 2.0

    show_volume = st.checkbox("Show volume", value=True)

# ── Load & filter ────────────────────────────────────────────────────────────

df = load_raw_data(selected_path)

st.subheader("Date Range")
col1, col2 = st.columns(2)
min_date = df["Date"].min().date()
max_date = df["Date"].max().date()
start = col1.date_input("Start", value=min_date, min_value=min_date, max_value=max_date)
end = col2.date_input("End", value=max_date, min_value=min_date, max_value=max_date)

import pandas as pd

mask = (df["Date"] >= pd.Timestamp(start)) & (df["Date"] <= pd.Timestamp(end))
filtered = df.loc[mask].copy()

if filtered.empty:
    st.warning("No data in the selected date range.")
    st.stop()

# ── Chart ────────────────────────────────────────────────────────────────────

fig = candlestick_chart(
    filtered,
    show_volume=show_volume,
    sma_windows=sma_options or None,
    ema_spans=ema_options or None,
    show_bollinger=show_bb,
    bollinger_window=bb_window,
    bollinger_std=bb_std,
    chart_type=chart_type,
    height=650,
)
st.plotly_chart(fig, use_container_width=True)

# ── Period statistics ────────────────────────────────────────────────────────

st.subheader("Period Statistics")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Mean Close", f"${filtered['Close'].mean():,.2f}")
c2.metric("Std Dev", f"${filtered['Close'].std():,.2f}")
c3.metric("Min Close", f"${filtered['Close'].min():,.2f}")
c4.metric("Max Close", f"${filtered['Close'].max():,.2f}")
period_return = (filtered["Close"].iloc[-1] - filtered["Close"].iloc[0]) / filtered["Close"].iloc[0] * 100
c5.metric("Period Return", f"{period_return:+.2f}%")
