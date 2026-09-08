import plotly.graph_objects as go
import pandas as pd
import numpy as np
from utils.helper import drop_holiday_nans, get_plotly_rangebreaks

def create_price_chart(
    df: pd.DataFrame,
    ticker: str,
    overlays: dict = None,
    show_events: bool = False,
    event_markers: list = None,
    support_resistance: dict = None
) -> go.Figure:
    """
    Renders interactive Plotly Candlestick chart with clean price-axis overlays,
    technical event markers, and support/resistance key levels.
    """
    fig = go.Figure()
    
    if df is not None and not df.empty:
        df = drop_holiday_nans(df)
        
    if df is None or df.empty:
        fig.update_layout(title="No data available")
        return fig

    # 1. Candlestick Trace
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="OHLC",
        increasing_line_color="#00E676",
        decreasing_line_color="#FF5252",
        increasing_fillcolor="rgba(0, 230, 118, 0.3)",
        decreasing_fillcolor="rgba(255, 82, 82, 0.3)"
    ))
    
    # 2. Overlays on Price Axis
    if overlays:
        colors = ["#29B6F6", "#AB47BC", "#FFA726", "#EC407A", "#26A69A", "#7E57C2", "#D4E157"]
        color_idx = 0
        
        for name, series_data in overlays.items():
            color = colors[color_idx % len(colors)]
            
            if isinstance(series_data, pd.Series):
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=series_data,
                    mode="lines",
                    name=name,
                    line=dict(width=1.5, color=color)
                ))
                color_idx += 1
            elif isinstance(series_data, pd.DataFrame):
                # E.g., Bollinger Bands, Donchian, Keltner, Ichimoku
                # Filter out Width and PctB if present in DataFrame (belong to Panels)
                cols_to_plot = [c for c in series_data.columns if "Width" not in c and "PctB" not in c]
                
                for col in cols_to_plot:
                    col_color = colors[color_idx % len(colors)]
                    dash_style = "dash" if "Upper" in col or "Lower" in col or "Senkou" in col else "solid"
                    
                    clean_col_name = col.replace("BB_", "").replace("DC_", "").replace("KC_", "")
                    
                    fig.add_trace(go.Scatter(
                        x=df.index,
                        y=series_data[col],
                        mode="lines",
                        name=f"{name} ({clean_col_name})",
                        line=dict(width=1.2, color=col_color, dash=dash_style)
                    ))
                    color_idx += 1

    # 3. Support & Resistance Horizontal Lines
    if support_resistance:
        sr_colors = {
            "Resistance 2": "#FF1744",
            "Resistance 1": "#FF5252",
            "Support 1": "#00E676",
            "Support 2": "#00B0FF"
        }
        for label, val in support_resistance.items():
            if label in sr_colors and not pd.isna(val):
                fig.add_hline(
                    y=val,
                    line_dash="dot",
                    line_color=sr_colors[label],
                    annotation_text=f"{label}: {val:,.2f}",
                    annotation_position="top right",
                    annotation_font=dict(color=sr_colors[label], size=10)
                )

    # 4. Event Markers Annotations
    if show_events and event_markers:
        for marker in event_markers:
            # marker = {"date": timestamp, "price": y_val, "text": "Golden Cross", "symbol": "triangle-up", "color": "#00E676"}
            fig.add_trace(go.Scatter(
                x=[marker["date"]],
                y=[marker["price"]],
                mode="markers+text",
                name=marker["text"],
                marker=dict(
                    symbol=marker.get("symbol", "circle"),
                    size=11,
                    color=marker.get("color", "#00E676"),
                    line=dict(width=1, color="#FFFFFF")
                ),
                text=[marker["text"]],
                textposition=marker.get("position", "top center"),
                textfont=dict(size=10, color=marker.get("color", "#FFFFFF")),
                showlegend=False
            ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0B0F19",
        plot_bgcolor="#0F172A",
        height=520,
        margin=dict(l=20, r=20, t=55, b=20),
        xaxis=dict(
            rangeslider=dict(visible=False),
            gridcolor="rgba(255,255,255,0.06)",
            rangebreaks=get_plotly_rangebreaks(df),
            rangeselector=dict(
                buttons=[
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=3, label="3M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(count=3, label="3Y", step="year", stepmode="backward"),
                    dict(count=5, label="5Y", step="year", stepmode="backward"),
                    dict(step="all", label="MAX")
                ],
                bgcolor="#1E293B",
                activecolor="#EC407A",
                font=dict(color="#F8FAFC", size=10),
                bordercolor="rgba(255,255,255,0.15)",
                borderwidth=1,
                x=0,
                y=1.04,
                xanchor="left",
                yanchor="bottom"
            )
        ),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)", side="right", title="Price"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.04,
            xanchor="right",
            x=1.0,
            font=dict(size=11, color="#E2E8F0"),
            bgcolor="rgba(0,0,0,0)"
        )
    )
    
    return fig
