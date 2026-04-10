"""Technical indicator computations used by charts and features."""

from __future__ import annotations

import pandas as pd


def compute_sma(series: pd.Series, window: int = 20) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def compute_ema(series: pd.Series, span: int = 20) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def compute_bollinger_bands(
    series: pd.Series, *, window: int = 20, num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    middle = compute_sma(series, window=window)
    std = series.rolling(window=window, min_periods=window).std()
    upper = middle + (num_std * std)
    lower = middle - (num_std * std)
    return upper, middle, lower


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    return 100 - (100 / (1 + rs))


def compute_macd(
    series: pd.Series, *, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    fast_ema = compute_ema(series, span=fast)
    slow_ema = compute_ema(series, span=slow)
    macd_line = fast_ema - slow_ema
    signal_line = compute_ema(macd_line, span=signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_stochastic(
    high: pd.Series, low: pd.Series, close: pd.Series, *, k_period: int = 14, d_period: int = 3
) -> tuple[pd.Series, pd.Series]:
    lowest_low = low.rolling(window=k_period, min_periods=k_period).min()
    highest_high = high.rolling(window=k_period, min_periods=k_period).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, pd.NA)
    d = k.rolling(window=d_period, min_periods=d_period).mean()
    return k, d


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, *, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = close.diff().fillna(0).apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    return (direction * volume.fillna(0)).cumsum()
