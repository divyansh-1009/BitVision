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


def prepare_user_ohlc_df(
    df: pd.DataFrame,
    *,
    synthesize_date_if_missing: bool = True,
    strict_ohlc_drop: bool = True,
) -> pd.DataFrame:
    """Normalize user-supplied OHLCV: headers, types, optional Date, default Volume.

    Required columns after normalization: Open, High, Low, Close.
    Volume is optional; missing or NaN values become 0.

    If ``synthesize_date_if_missing`` is True and there is no usable Date column,
    assigns ``pd.date_range`` (daily) for ordering only.

    If ``strict_ohlc_drop`` is True, drops rows with null OHLC after coercion (manual input).
    If False, matches legacy ``load_raw_data`` behaviour (coerce only, no OHLC dropna).

    Raises:
        ValueError: empty frame or missing required OHLC columns.
    """
    if df.empty:
        raise ValueError("Input data is empty.")

    out = _normalize_market_headers(df.copy())
    out = _clean_numeric_strings(out)

    required = {"Open", "High", "Low", "Close"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}.")

    for col in ["Open", "High", "Low", "Close"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if strict_ohlc_drop:
        out = out.dropna(subset=list(required)).copy()
        if out.empty:
            raise ValueError("All rows dropped: Open, High, Low, and Close must be numeric and non-null.")

    if "Volume" not in out.columns:
        out["Volume"] = 0.0
    else:
        out["Volume"] = pd.to_numeric(out["Volume"], errors="coerce").fillna(0.0)

    if "Date" in out.columns:
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
        if out["Date"].notna().any():
            out = out.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
        elif synthesize_date_if_missing:
            out = out.drop(columns=["Date"], errors="ignore")
            out["Date"] = pd.date_range(start="2000-01-01", periods=len(out), freq="D")
        else:
            out = out.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    elif synthesize_date_if_missing:
        out["Date"] = pd.date_range(start="2000-01-01", periods=len(out), freq="D")

    return out.reset_index(drop=True)


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
    return prepare_user_ohlc_df(df, synthesize_date_if_missing=False, strict_ohlc_drop=False)


def load_processed_data(path: Path) -> pd.DataFrame:
    """Load feature-enriched CSV data."""
    df = pd.read_csv(path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    return _clean_numeric_strings(df)
