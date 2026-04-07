"""Orchestrates raw OHLC data into a feature-rich DataFrame."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.technical_indicators import (
    compute_atr,
    compute_bollinger_bands,
    compute_ema,
    compute_macd,
    compute_obv,
    compute_rsi,
    compute_sma,
    compute_stochastic,
)


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive price-based features from OHLC data."""
    df = df.copy()
    df["Daily_return"] = df["Close"].pct_change()
    df["Log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["Price_change"] = df["Close"].diff()
    df["High_low_range"] = df["High"] - df["Low"]
    df["Rolling_volatility_7"] = df["Daily_return"].rolling(window=7, min_periods=1).std()
    df["Rolling_volatility_30"] = df["Daily_return"].rolling(window=30, min_periods=1).std()
    return df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute and attach all technical indicators."""
    df = df.copy()

    df["Sma_20"] = compute_sma(df["Close"], window=20)
    df["Sma_50"] = compute_sma(df["Close"], window=50)
    df["Sma_200"] = compute_sma(df["Close"], window=200)
    df["Ema_12"] = compute_ema(df["Close"], span=12)
    df["Ema_26"] = compute_ema(df["Close"], span=26)

    df["Rsi"] = compute_rsi(df["Close"], period=14)

    macd_line, signal_line, histogram = compute_macd(df["Close"])
    df["Macd"] = macd_line
    df["Macd_signal"] = signal_line
    df["Macd_hist"] = histogram

    upper, middle, lower = compute_bollinger_bands(df["Close"])
    df["Bollinger_upper"] = upper
    df["Bollinger_middle"] = middle
    df["Bollinger_lower"] = lower

    k, d = compute_stochastic(df["High"], df["Low"], df["Close"])
    df["Stochastic_k"] = k
    df["Stochastic_d"] = d

    df["Atr"] = compute_atr(df["High"], df["Low"], df["Close"])
    df["Obv"] = compute_obv(df["Close"], df["Volume"])

    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full pipeline: raw OHLC -> enriched DataFrame with all features."""
    df = add_price_features(df)
    df = add_technical_indicators(df)
    return df
