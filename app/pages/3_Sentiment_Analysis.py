"""Sentiment and news analysis visualisations."""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.components.charts import fear_greed_gauge, sentiment_timeline, sentiment_vs_price
from app.components.metrics import empty_state, render_metric_row
from app.utils.config import (
    LAYOUT,
    PAGE_ICON,
    PAGE_TITLE,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)
from src.data_loader import list_data_files, load_processed_data, load_raw_data

st.set_page_config(page_title=f"{PAGE_TITLE} — Sentiment", page_icon=PAGE_ICON, layout=LAYOUT)

st.title("Sentiment Analysis")
st.caption("News sentiment, Fear & Greed index, and sentiment-price correlation")

# ── Check for processed data with sentiment columns ──────────────────────────

processed_files = list_data_files(PROCESSED_DATA_DIR)
raw_files = list_data_files(RAW_DATA_DIR)

if not processed_files:
    empty_state(
        "No processed data with sentiment features",
        "This page requires processed data containing sentiment columns.\n\n"
        "Place a CSV in `data/processed/` with columns like:\n"
        "- `Sentiment_score` (daily aggregate sentiment, e.g. -1 to 1)\n"
        "- `News_sentiment` (news-based score)\n"
        "- `Fear_greed_index` (0-100 scale)\n\n"
        "```\ndata/\n  processed/\n    btc_features.csv   # must include sentiment columns\n```",
    )
    st.stop()

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Settings")
    selected_file = st.selectbox("Processed file", processed_files, key="sent_file")

df = load_processed_data(PROCESSED_DATA_DIR / selected_file)

SENTIMENT_COLS = [c for c in df.columns if "sentiment" in c.lower() or "fear" in c.lower() or "greed" in c.lower()]

if not SENTIMENT_COLS:
    empty_state(
        "No sentiment columns detected",
        f"The file `{selected_file}` does not contain sentiment-related columns.\n\n"
        "Expected column names containing `sentiment`, `fear`, or `greed` (case-insensitive).\n\n"
        f"**Available columns:** {', '.join(df.columns.tolist())}",
    )
    st.stop()

# ── Determine best sentiment column ──────────────────────────────────────────

with st.sidebar:
    sent_col = st.selectbox("Sentiment column", SENTIMENT_COLS, key="sent_col")

has_price = "Close" in df.columns
has_date = "Date" in df.columns

if not has_date:
    st.error("Processed data must contain a `Date` column.")
    st.stop()

# ── Sentiment metrics ────────────────────────────────────────────────────────

st.subheader("Overview")

latest_sent = df[sent_col].dropna().iloc[-1] if not df[sent_col].dropna().empty else None
avg_sent = df[sent_col].mean()

metrics = [
    {"label": "Latest Score", "value": f"{latest_sent:.3f}" if latest_sent is not None else "N/A"},
    {"label": "Average Score", "value": f"{avg_sent:.3f}"},
    {"label": "Std Dev", "value": f"{df[sent_col].std():.3f}"},
]

if has_price:
    corr = df[sent_col].corr(df["Close"])
    metrics.append({"label": "Correlation w/ Price", "value": f"{corr:.3f}"})

render_metric_row(metrics)

# ── Timeline ─────────────────────────────────────────────────────────────────

st.subheader("Sentiment Over Time")
st.plotly_chart(sentiment_timeline(df["Date"], df[sent_col]), use_container_width=True)

# ── Fear & Greed gauge ───────────────────────────────────────────────────────

fear_cols = [c for c in df.columns if "fear" in c.lower() or "greed" in c.lower()]
if fear_cols:
    fg_col = fear_cols[0]
    latest_fg = df[fg_col].dropna().iloc[-1] if not df[fg_col].dropna().empty else 50
    st.subheader("Fear & Greed Index")
    st.plotly_chart(fear_greed_gauge(float(latest_fg)), use_container_width=True)

# ── Distribution ─────────────────────────────────────────────────────────────

st.subheader("Sentiment Distribution")
fig_hist = px.histogram(
    df,
    x=sent_col,
    nbins=50,
    template="plotly_dark",
    color_discrete_sequence=["#f7931a"],
)
fig_hist.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=300)
st.plotly_chart(fig_hist, use_container_width=True)

# ── Sentiment vs Price ───────────────────────────────────────────────────────

if has_price:
    st.subheader("Sentiment vs. Price")
    st.plotly_chart(
        sentiment_vs_price(df["Date"], df["Close"], df[sent_col]),
        use_container_width=True,
    )
