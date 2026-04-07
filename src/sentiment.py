"""Load and merge sentiment / news analysis data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


EXPECTED_SENTIMENT_COLUMNS = {"Date", "Sentiment_score"}


def load_sentiment_data(filepath: str | Path) -> pd.DataFrame:
    """Load a sentiment CSV and normalise its columns."""
    filepath = Path(filepath)
    if not filepath.is_file():
        raise FileNotFoundError(f"Sentiment file not found: {filepath}")

    df = pd.read_csv(filepath)
    df.columns = [c.strip().capitalize() for c in df.columns]

    if "Date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["Date"]):
        df["Date"] = pd.to_datetime(df["Date"], infer_datetime_format=True)

    return df


def merge_sentiment_with_price(
    price_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join sentiment data onto the price DataFrame on Date."""
    if "Date" not in price_df.columns or "Date" not in sentiment_df.columns:
        raise ValueError("Both DataFrames must contain a 'Date' column.")

    price_df = price_df.copy()
    sentiment_df = sentiment_df.copy()

    price_df["Date"] = pd.to_datetime(price_df["Date"])
    sentiment_df["Date"] = pd.to_datetime(sentiment_df["Date"])

    merged = price_df.merge(sentiment_df, on="Date", how="left")
    return merged
