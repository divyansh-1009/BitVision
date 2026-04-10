"""Feature engineering from OHLCV market data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.utils.technical_indicators import (
    compute_atr,
    compute_bollinger_bands,
    compute_macd,
    compute_obv,
    compute_rsi,
    compute_sma,
)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build model features directly from raw OHLCV data."""
    required_cols = {"Open", "High", "Low", "Close"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required OHLC columns for features: {', '.join(sorted(missing))}")

    out = df.copy()
    # Some CSV parsers can keep numeric columns as string/Arrow dtypes.
    # Force numeric conversion so rolling ops and pct_change are reliable.
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["Open", "High", "Low", "Close"]).copy()
    close = out["Close"]

    out["return_1d"] = close.pct_change()
    out["Return_1d"] = out["return_1d"]
    out["Return_3d"] = close.pct_change(3)
    out["return_7d"] = close.pct_change(7)
    out["return_30d"] = close.pct_change(30)
    out["volatility_7d"] = out["return_1d"].rolling(7, min_periods=7).std()
    out["Volatility_7"] = out["volatility_7d"]
    out["volatility_30d"] = out["return_1d"].rolling(30, min_periods=30).std()

    out["sma_7"] = compute_sma(close, window=7)
    out["SMA_7_diff"] = (close - out["sma_7"]) / out["sma_7"].replace(0, np.nan)
    out["sma_20"] = compute_sma(close, window=20)
    out["sma_50"] = compute_sma(close, window=50)
    out["ema_12"] = close.ewm(span=12, adjust=False).mean()
    out["ema_26"] = close.ewm(span=26, adjust=False).mean()

    upper, middle, lower = compute_bollinger_bands(close, window=20, num_std=2.0)
    out["bb_upper"] = upper
    out["bb_middle"] = middle
    out["bb_lower"] = lower
    out["bb_width"] = (upper - lower) / middle.replace(0, np.nan)

    out["rsi_14"] = compute_rsi(close, period=14)
    macd_line, signal_line, hist = compute_macd(close, fast=12, slow=26, signal=9)
    out["macd"] = macd_line
    out["macd_signal"] = signal_line
    out["macd_hist"] = hist
    out["atr_14"] = compute_atr(out["High"], out["Low"], close, period=14)

    if "Volume" in out.columns:
        out["obv"] = compute_obv(close, out["Volume"])
        out["volume_change_1d"] = out["Volume"].pct_change()
        out["volume_sma_7"] = compute_sma(out["Volume"], window=7)

    return out
