"""Data loading, validation, and file listing utilities for BitVision."""

import os
from pathlib import Path

import pandas as pd


REQUIRED_OHLC_COLUMNS = {"Date", "Open", "High", "Low", "Close", "Volume"}

DATE_PARSE_FORMATS = [
    "%Y-%m-%d",
    "%Y-%m-%d %H:%M:%S",
    "%m/%d/%Y",
    "%d-%m-%Y",
]


def list_data_files(directory: str | Path) -> list[str]:
    """Return sorted list of CSV filenames in *directory*."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    return sorted(f.name for f in directory.iterdir() if f.suffix.lower() == ".csv")


def validate_ohlc(df: pd.DataFrame) -> bool:
    """Check that *df* contains the required OHLC columns (case-insensitive)."""
    cols_upper = {c.strip().capitalize() for c in df.columns}
    return REQUIRED_OHLC_COLUMNS.issubset(cols_upper)


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Capitalise column names so downstream code can rely on a single convention."""
    df.columns = [c.strip().capitalize() for c in df.columns]
    return df


def _parse_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the *Date* column to ``datetime64`` and set it as the index."""
    if "Date" not in df.columns:
        return df

    if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
        df["Date"] = pd.to_datetime(df["Date"], infer_datetime_format=True)

    df = df.sort_values("Date").reset_index(drop=True)
    return df


def load_raw_data(filepath: str | Path) -> pd.DataFrame:
    """Load a raw OHLC CSV, normalise columns, parse dates and sort."""
    filepath = Path(filepath)
    if not filepath.is_file():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    df = pd.read_csv(filepath)
    df = _normalise_columns(df)

    if not validate_ohlc(df):
        raise ValueError(
            f"CSV is missing required columns. "
            f"Expected {REQUIRED_OHLC_COLUMNS}, got {set(df.columns)}"
        )

    df = _parse_date_column(df)
    return df


def load_processed_data(filepath: str | Path) -> pd.DataFrame:
    """Load a processed CSV that already contains features."""
    filepath = Path(filepath)
    if not filepath.is_file():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    df = pd.read_csv(filepath)
    df = _normalise_columns(df)
    df = _parse_date_column(df)
    return df
