"""Data loading helpers for Streamlit pages."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _normalize_market_headers(df: pd.DataFrame) -> pd.DataFrame:
    canonical = {
        "date": "Date",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    }
    rename_map: dict[str, str] = {}
    for col in df.columns:
        normalized = str(col).strip().lower().replace(" ", "_")
        if normalized in canonical:
            rename_map[col] = canonical[normalized]
    return df.rename(columns=rename_map)


def _candidate_dirs(base_dir: Path) -> list[Path]:
    """Return existing directories to scan, including flat data dir fallback."""
    dirs: list[Path] = []
    if base_dir.exists() and base_dir.is_dir():
        dirs.append(base_dir)
    parent = base_dir.parent
    if parent.exists() and parent.is_dir() and parent not in dirs:
        dirs.append(parent)
    return dirs


def list_data_files(base_dir: Path) -> list[str]:
    """List CSV filenames available for a logical data bucket."""
    if base_dir.exists() and base_dir.is_dir():
        return sorted([p.name for p in base_dir.glob("*.csv") if p.is_file()])

    seen: set[str] = set()
    files: list[str] = []
    parent = base_dir.parent
    if not parent.exists() or not parent.is_dir():
        return files

    fallback_candidates = sorted([p for p in parent.glob("*.csv") if p.is_file()])
    if base_dir.name == "processed":
        filtered = [
            p for p in fallback_candidates if any(k in p.name.lower() for k in ("processed", "feature", "features"))
        ]
        fallback_candidates = filtered

    for path in fallback_candidates:
        if path.name in seen:
            continue
        seen.add(path.name)
        files.append(path.name)
    return files


def resolve_data_file(base_dir: Path, filename: str) -> Path:
    """Resolve filename against logical dir and fallback parent."""
    for directory in _candidate_dirs(base_dir):
        candidate = directory / filename
        if candidate.exists():
            return candidate
    return base_dir / filename


def _clean_numeric_strings(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    for col in cleaned.columns:
        if cleaned[col].dtype == object:
            series = cleaned[col].astype(str).str.replace(",", "", regex=False)
            numeric = pd.to_numeric(series, errors="coerce")
            if numeric.notna().sum() > 0:
                cleaned[col] = numeric
    return cleaned


def load_raw_data(path: Path) -> pd.DataFrame:
    """Load and normalize OHLC-like CSV data."""
    df = pd.read_csv(path)
    df = _normalize_market_headers(df)
    df = _clean_numeric_strings(df)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if "Date" in df.columns:
        df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    return df


def load_processed_data(path: Path) -> pd.DataFrame:
    """Load feature-enriched CSV data."""
    df = pd.read_csv(path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    return _clean_numeric_strings(df)
