import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from utils.helper import drop_holiday_nans, get_plotly_rangebreaks

def create_volume_chart(df: pd.DataFrame, volume_indicators: dict = None) -> go.Figure:
    """
    Renders Volume bars and volume indicator subplots with average volume overlay.
    """
    if df is not None and not df.empty:
        df = drop_holiday_nans(df)

    if df is None or df.empty or "Volume" not in df.columns:
        return None

    num_subs = 1 + (len(volume_indicators) if volume_indicators else 0)
    fig = make_subplots(
        rows=num_subs,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=["<b>Volume · 20-Day SMA</b>"] + [f"<b>{name}</b>" for name in (volume_indicators.keys() if volume_indicators else [])]
    )

    # 1. Volume Bars
    colors = ["#00E676" if close >= open_p else "#FF5252" for close, open_p in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(
        x=df.index,
        y=df["Volume"],
        name="Volume",
        marker_color=colors
    ), row=1, col=1)
    
    # Volume 20-day SMA overlay
    vol_sma = df["Volume"].rolling(window=20).mean()
    fig.add_trace(go.Scatter(
        x=df.index,
        y=vol_sma,
        mode="lines",
        name="Vol SMA (20)",
        line=dict(width=1.5, color="#38BDF8")
    ), row=1, col=1)

    # 2. Volume Indicators
    if volume_indicators:
        row_idx = 2
        for name, series_data in volume_indicators.items():
            if isinstance(series_data, pd.Series):
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=series_data,
                    mode="lines",
                    name=name,
                    line=dict(width=1.5, color="#29B6F6")
                ), row=row_idx, col=1)
                
                if name == "RVOL":
                    fig.add_hline(y=1.0, line_dash="dash", line_color="#8B949E", row=row_idx, col=1, annotation_text="Average (1.0x)")
                    fig.add_hline(y=1.5, line_dash="dash", line_color="#FFA726", row=row_idx, col=1, annotation_text="Elevated (1.5x)")
                elif name == "CMF":
                    fig.add_hline(y=0.0, line_dash="dash", line_color="#8B949E", row=row_idx, col=1)
                    
            row_idx += 1

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0B0F19",
        plot_bgcolor="#0F172A",
        height=180 * num_subs,
        margin=dict(l=20, r=20, t=30, b=20),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0, font=dict(size=11, color="#E2E8F0"), bgcolor="rgba(0,0,0,0)")
    )
    
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)", rangebreaks=get_plotly_rangebreaks(df))
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)", side="right")

    return fig
