"""Centralised paths, constants, and page configuration for the Streamlit app."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

PAGE_ICON = "₿"
PAGE_TITLE = "BitVision"
LAYOUT = "wide"

OHLC_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]

SENTIMENT_COLUMNS = [
    "Sentiment_score",
    "News_sentiment",
    "Fear_greed_index",
]

PLOTLY_TEMPLATE = "plotly_dark"

COLOR_UP = "#26a69a"
COLOR_DOWN = "#ef5350"
COLOR_PRIMARY = "#f7931a"
COLOR_SECONDARY = "#4fc3f7"
COLOR_ACCENT = "#ab47bc"
