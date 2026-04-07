"""Data exploration: tables, statistics, correlations, and distributions."""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.components.charts import correlation_heatmap
from app.components.metrics import empty_state
from app.utils.config import (
    LAYOUT,
    PAGE_ICON,
    PAGE_TITLE,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)
from src.data_loader import list_data_files, load_processed_data, load_raw_data

st.set_page_config(page_title=f"{PAGE_TITLE} — Data Explorer", page_icon=PAGE_ICON, layout=LAYOUT)

st.title("Data Explorer")
st.caption("Browse raw and processed datasets, view statistics, and explore correlations")

# ── Data availability ────────────────────────────────────────────────────────

raw_files = list_data_files(RAW_DATA_DIR)
processed_files = list_data_files(PROCESSED_DATA_DIR)

if not raw_files and not processed_files:
    empty_state(
        "No data files found",
        "Place CSV files in `data/raw/` (OHLC data) or `data/processed/` "
        "(feature-enriched data) to explore them here.\n\n"
        "```\ndata/\n  raw/\n    btc_ohlc.csv\n  processed/\n    btc_features.csv\n```",
    )
    st.stop()

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Data Source")

    source_options = []
    if raw_files:
        source_options.append("Raw")
    if processed_files:
        source_options.append("Processed")

    source = st.radio("Data type", source_options, horizontal=True)

    if source == "Raw":
        selected_file = st.selectbox("File", raw_files, key="de_raw")
        data_path = RAW_DATA_DIR / selected_file
    else:
        selected_file = st.selectbox("File", processed_files, key="de_proc")
        data_path = PROCESSED_DATA_DIR / selected_file

# ── Load ─────────────────────────────────────────────────────────────────────

if source == "Processed":
    df = load_processed_data(data_path)
else:
    df = load_raw_data(data_path)

# ── Interactive data table ───────────────────────────────────────────────────

st.subheader("Data Table")
st.dataframe(df, use_container_width=True, height=400)

# ── Descriptive statistics ───────────────────────────────────────────────────

st.subheader("Descriptive Statistics")
st.dataframe(df.describe(), use_container_width=True)

# ── Missing values ───────────────────────────────────────────────────────────

st.subheader("Missing Values")
missing = df.isnull().sum()
missing = missing[missing > 0]
if missing.empty:
    st.success("No missing values detected.")
else:
    missing_df = pd.DataFrame({"Column": missing.index, "Missing Count": missing.values, "% Missing": (missing.values / len(df) * 100).round(2)})
    st.dataframe(missing_df, use_container_width=True, hide_index=True)

# ── Correlation heatmap ──────────────────────────────────────────────────────

numeric_cols = df.select_dtypes(include="number").columns.tolist()

if len(numeric_cols) >= 2:
    st.subheader("Correlation Heatmap")
    with st.expander("Select columns for correlation", expanded=False):
        corr_cols = st.multiselect(
            "Columns",
            numeric_cols,
            default=numeric_cols[:10],
            key="de_corr_cols",
        )
    if len(corr_cols) >= 2:
        st.plotly_chart(correlation_heatmap(df[corr_cols]), use_container_width=True)
    else:
        st.info("Select at least 2 columns.")

# ── Feature distributions ───────────────────────────────────────────────────

st.subheader("Feature Distributions")

dist_col = st.selectbox("Column", numeric_cols, key="de_dist_col")
chart_kind = st.radio("Chart type", ["Histogram", "Box Plot"], horizontal=True, key="de_dist_kind")

if chart_kind == "Histogram":
    fig = px.histogram(
        df,
        x=dist_col,
        nbins=60,
        template="plotly_dark",
        color_discrete_sequence=["#f7931a"],
    )
else:
    fig = px.box(
        df,
        y=dist_col,
        template="plotly_dark",
        color_discrete_sequence=["#f7931a"],
    )

fig.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=350)
st.plotly_chart(fig, use_container_width=True)

# ── Download ─────────────────────────────────────────────────────────────────

st.divider()
st.download_button(
    label="Download data as CSV",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name=f"bitvision_{selected_file}",
    mime="text/csv",
)
