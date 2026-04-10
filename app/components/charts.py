"""Reusable Plotly chart builders for the BitVision app."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app.utils.config import (
    COLOR_ACCENT,
    COLOR_DOWN,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    COLOR_UP,
    PLOTLY_TEMPLATE,
)


def candlestick_chart(
    df: pd.DataFrame,
    *,
    show_volume: bool = True,
    sma_windows: list[int] | None = None,
    ema_spans: list[int] | None = None,
    show_bollinger: bool = False,
    bollinger_window: int = 20,
    bollinger_std: float = 2.0,
    chart_type: str = "Candlestick",
    height: int = 600,
) -> go.Figure:
    """Build an interactive candlestick (or line/OHLC) chart with optional overlays."""
    from app.utils.technical_indicators import compute_bollinger_bands, compute_ema, compute_sma

    row_heights = [0.7, 0.3] if show_volume else [1.0]
    rows = 2 if show_volume else 1

    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
    )

    if chart_type == "Candlestick":
        fig.add_trace(
            go.Candlestick(
                x=df["Date"],
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                increasing_line_color=COLOR_UP,
                decreasing_line_color=COLOR_DOWN,
                name="OHLC",
            ),
            row=1,
            col=1,
        )
    elif chart_type == "OHLC":
        fig.add_trace(
            go.Ohlc(
                x=df["Date"],
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                increasing_line_color=COLOR_UP,
                decreasing_line_color=COLOR_DOWN,
                name="OHLC",
            ),
            row=1,
            col=1,
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=df["Date"],
                y=df["Close"],
                mode="lines",
                name="Close",
                line=dict(color=COLOR_PRIMARY, width=2),
            ),
            row=1,
            col=1,
        )

    overlay_colors = ["#ffab40", "#42a5f5", "#ab47bc", "#66bb6a", "#ef5350"]

    if sma_windows:
        for i, w in enumerate(sma_windows):
            sma = compute_sma(df["Close"], window=w)
            fig.add_trace(
                go.Scatter(
                    x=df["Date"],
                    y=sma,
                    mode="lines",
                    name=f"SMA {w}",
                    line=dict(width=1.2, color=overlay_colors[i % len(overlay_colors)]),
                ),
                row=1,
                col=1,
            )

    if ema_spans:
        for i, s in enumerate(ema_spans):
            ema = compute_ema(df["Close"], span=s)
            fig.add_trace(
                go.Scatter(
                    x=df["Date"],
                    y=ema,
                    mode="lines",
                    name=f"EMA {s}",
                    line=dict(width=1.2, dash="dot", color=overlay_colors[(i + 2) % len(overlay_colors)]),
                ),
                row=1,
                col=1,
            )

    if show_bollinger:
        upper, middle, lower = compute_bollinger_bands(
            df["Close"], window=bollinger_window, num_std=bollinger_std,
        )
        fig.add_trace(
            go.Scatter(x=df["Date"], y=upper, mode="lines", name="BB Upper",
                       line=dict(width=1, color=COLOR_SECONDARY, dash="dash")),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=df["Date"], y=lower, mode="lines", name="BB Lower",
                       line=dict(width=1, color=COLOR_SECONDARY, dash="dash"),
                       fill="tonexty", fillcolor="rgba(79,195,247,0.08)"),
            row=1, col=1,
        )

    if show_volume:
        volume_up = "#80deea"
        volume_down = "#ffab91"
        colors = [
            volume_up if c >= o else volume_down
            for c, o in zip(df["Close"], df["Open"])
        ]
        fig.add_trace(
            go.Bar(
                x=df["Date"],
                y=df["Volume"],
                marker_color=colors,
                name="Volume",
                opacity=0.95,
            ),
            row=2, col=1,
        )
        fig.update_yaxes(title_text="Volume", row=2, col=1)

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=height,
        xaxis_rangeslider_visible=False,
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h", y=1.02, x=0),
        yaxis_title="Price (USD)",
    )

    return fig



def sparkline_chart(dates: pd.Series, values: pd.Series, height: int = 150) -> go.Figure:
    """Tiny area chart used on the dashboard."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=values,
            mode="lines",
            fill="tozeroy",
            line=dict(color=COLOR_PRIMARY, width=2),
            fillcolor="rgba(247,147,26,0.15)",
        )
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=height,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=False,
    )
    return fig


def technical_subplot(
    df: pd.DataFrame,
    indicators: list[str],
    *,
    rsi_period: int = 14,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    bb_window: int = 20,
    bb_std: float = 2.0,
    stoch_k: int = 14,
    stoch_d: int = 3,
    atr_period: int = 14,
    height_per_panel: int = 200,
) -> go.Figure:
    """Stacked subplots of selected technical indicators, synced on x-axis."""
    from app.utils.technical_indicators import (
        compute_atr,
        compute_bollinger_bands,
        compute_macd,
        compute_obv,
        compute_rsi,
        compute_sma,
        compute_stochastic,
    )

    n_panels = 1 + len(indicators)
    if indicators:
        # Keep the price panel compact and allocate more space to indicator panes.
        price_height = 0.28
        indicator_height = (1.0 - price_height) / len(indicators)
        panel_heights = [price_height] + [indicator_height] * len(indicators)
    else:
        panel_heights = [1.0]

    total_height = max(700, 420 + (300 * len(indicators)))

    titles = ["Price"] + indicators
    fig = make_subplots(
        rows=n_panels,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=panel_heights,
        subplot_titles=titles,
    )

    upper, mid, lower = compute_bollinger_bands(df["Close"], window=bb_window, num_std=bb_std)
    fig.add_trace(
        go.Candlestick(
            x=df["Date"], open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
            increasing_line_color=COLOR_UP, decreasing_line_color=COLOR_DOWN, name="OHLC",
        ),
        row=1, col=1,
    )

    if "Bollinger Bands" in indicators:
        fig.add_trace(go.Scatter(x=df["Date"], y=upper, mode="lines", name="BB Upper",
                                 line=dict(width=1, dash="dash", color=COLOR_SECONDARY)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df["Date"], y=lower, mode="lines", name="BB Lower",
                                 line=dict(width=1, dash="dash", color=COLOR_SECONDARY),
                                 fill="tonexty", fillcolor="rgba(79,195,247,0.08)"), row=1, col=1)

    sma20 = compute_sma(df["Close"], window=20)
    fig.add_trace(go.Scatter(x=df["Date"], y=sma20, mode="lines", name="SMA 20",
                             line=dict(width=1, color="#ffab40")), row=1, col=1)

    for idx, name in enumerate(indicators, start=2):
        if name == "RSI":
            rsi = compute_rsi(df["Close"], period=rsi_period)
            fig.add_trace(go.Scatter(x=df["Date"], y=rsi, mode="lines", name="RSI",
                                     line=dict(color=COLOR_ACCENT)), row=idx, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=idx, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=idx, col=1)
            fig.update_yaxes(range=[0, 100], row=idx, col=1)

        elif name == "MACD":
            ml, sl, hist = compute_macd(df["Close"], fast=macd_fast, slow=macd_slow, signal=macd_signal)
            colors = [COLOR_UP if v >= 0 else COLOR_DOWN for v in hist]
            fig.add_trace(go.Bar(x=df["Date"], y=hist, marker_color=colors, name="MACD Hist", opacity=0.5),
                          row=idx, col=1)
            fig.add_trace(go.Scatter(x=df["Date"], y=ml, mode="lines", name="MACD",
                                     line=dict(color=COLOR_PRIMARY)), row=idx, col=1)
            fig.add_trace(go.Scatter(x=df["Date"], y=sl, mode="lines", name="Signal",
                                     line=dict(color=COLOR_SECONDARY)), row=idx, col=1)

        elif name == "Stochastic":
            k, d = compute_stochastic(df["High"], df["Low"], df["Close"],
                                       k_period=stoch_k, d_period=stoch_d)
            fig.add_trace(go.Scatter(x=df["Date"], y=k, mode="lines", name="%K",
                                     line=dict(color=COLOR_PRIMARY)), row=idx, col=1)
            fig.add_trace(go.Scatter(x=df["Date"], y=d, mode="lines", name="%D",
                                     line=dict(color=COLOR_SECONDARY)), row=idx, col=1)
            fig.add_hline(y=80, line_dash="dash", line_color="red", opacity=0.5, row=idx, col=1)
            fig.add_hline(y=20, line_dash="dash", line_color="green", opacity=0.5, row=idx, col=1)
            fig.update_yaxes(range=[0, 100], row=idx, col=1)

        elif name == "ATR":
            atr = compute_atr(df["High"], df["Low"], df["Close"], period=atr_period)
            fig.add_trace(go.Scatter(x=df["Date"], y=atr, mode="lines", name="ATR",
                                     line=dict(color=COLOR_ACCENT)), row=idx, col=1)

        elif name == "OBV":
            obv = compute_obv(df["Close"], df["Volume"])
            fig.add_trace(go.Scatter(x=df["Date"], y=obv, mode="lines", name="OBV",
                                     line=dict(color=COLOR_SECONDARY)), row=idx, col=1)

        elif name == "Bollinger Bands":
            pass

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=total_height,
        margin=dict(l=0, r=0, t=60, b=20),
        xaxis_rangeslider_visible=False,
        # With many indicators selected, legends become unreadable and obscure plots.
        # Subplot titles and known colors are clearer in this dense layout.
        showlegend=False,
    )
    return fig



def sentiment_timeline(dates: pd.Series, scores: pd.Series) -> go.Figure:
    """Line chart of sentiment scores over time."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dates, y=scores, mode="lines+markers",
            name="Sentiment", line=dict(color=COLOR_PRIMARY, width=2),
            marker=dict(size=3),
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=350,
        margin=dict(l=0, r=0, t=30, b=0),
        yaxis_title="Sentiment Score",
        xaxis_title="Date",
    )
    return fig


def fear_greed_gauge(value: float) -> go.Figure:
    """Plotly gauge chart for a Fear & Greed index (0-100)."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            title={"text": "Fear & Greed Index"},
            gauge=dict(
                axis=dict(range=[0, 100]),
                bar=dict(color=COLOR_PRIMARY),
                steps=[
                    dict(range=[0, 25], color="#d32f2f"),
                    dict(range=[25, 45], color="#f57c00"),
                    dict(range=[45, 55], color="#fdd835"),
                    dict(range=[55, 75], color="#66bb6a"),
                    dict(range=[75, 100], color="#2e7d32"),
                ],
                threshold=dict(line=dict(color="white", width=2), value=value),
            ),
        )
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=300,
        margin=dict(l=40, r=40, t=60, b=20),
    )
    return fig


def sentiment_vs_price(
    dates: pd.Series,
    price: pd.Series,
    sentiment: pd.Series,
) -> go.Figure:
    """Dual-axis chart overlaying sentiment and BTC price."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=dates, y=price, mode="lines", name="Close Price",
                   line=dict(color=COLOR_PRIMARY, width=2)),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=dates, y=sentiment, mode="lines", name="Sentiment",
                   line=dict(color=COLOR_SECONDARY, width=2)),
        secondary_y=True,
    )
    fig.update_yaxes(title_text="Price (USD)", secondary_y=False)
    fig.update_yaxes(title_text="Sentiment Score", secondary_y=True)
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=400,
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h", y=1.05, x=0),
    )
    return fig



def actual_vs_predicted(
    dates: pd.Series,
    actual: pd.Series,
    predicted: pd.Series,
) -> go.Figure:
    """Overlay actual and predicted close prices."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=dates, y=actual, mode="lines", name="Actual",
                   line=dict(color=COLOR_PRIMARY, width=2))
    )
    fig.add_trace(
        go.Scatter(x=dates, y=predicted, mode="lines", name="Predicted",
                   line=dict(color=COLOR_SECONDARY, width=2, dash="dot"))
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=450,
        margin=dict(l=0, r=0, t=30, b=0),
        yaxis_title="Price (USD)",
        legend=dict(orientation="h", y=1.05, x=0),
    )
    return fig


def residual_chart(dates: pd.Series, residuals: pd.Series) -> go.Figure:
    """Residuals over time + zero line."""
    fig = go.Figure()
    colors = [COLOR_UP if r >= 0 else COLOR_DOWN for r in residuals]
    fig.add_trace(
        go.Bar(x=dates, y=residuals, marker_color=colors, name="Residual", opacity=0.7)
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=300,
        margin=dict(l=0, r=0, t=30, b=0),
        yaxis_title="Residual (Actual - Predicted)",
    )
    return fig


def correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Plotly heatmap of feature correlations."""
    numeric = df.select_dtypes(include="number")
    corr = numeric.corr()
    fig = go.Figure(
        go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.columns.tolist(),
            colorscale="RdBu_r",
            zmin=-1,
            zmax=1,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            textfont=dict(size=9),
        )
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=max(400, 30 * len(corr.columns)),
        margin=dict(l=0, r=0, t=30, b=0),
    )
    return fig
