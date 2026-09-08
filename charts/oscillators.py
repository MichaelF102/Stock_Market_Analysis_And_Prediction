import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from utils.helper import drop_holiday_nans, get_plotly_rangebreaks

def create_oscillator_charts(df: pd.DataFrame, oscillators: dict) -> go.Figure:
    """
    Renders stacked subplots for oscillator & panel indicators with clean legends,
    threshold reference lines, and latest-value badges.
    """
    if df is not None and not df.empty:
        df = drop_holiday_nans(df)

    if not oscillators or df is None or df.empty:
        return None

    num_oscillators = len(oscillators)
    subplot_titles = []
    
    for name, series_data in oscillators.items():
        if isinstance(series_data, pd.Series):
            last_val = series_data.dropna().iloc[-1] if not series_data.dropna().empty else 0.0
            subplot_titles.append(f"<b>{name}</b> &nbsp;&nbsp;(Latest: <b>{last_val:.2f}</b>)")
        elif isinstance(series_data, pd.DataFrame):
            last_row = series_data.dropna().iloc[-1] if not series_data.dropna().empty else None
            if last_row is not None and not last_row.empty:
                val_str = ", ".join([f"{c}: {last_row[c]:.2f}" for c in last_row.index[:2]])
                subplot_titles.append(f"<b>{name}</b> &nbsp;&nbsp;({val_str})")
            else:
                subplot_titles.append(f"<b>{name}</b>")

    fig = make_subplots(
        rows=num_oscillators,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=subplot_titles
    )

    row_idx = 1
    colors = ["#29B6F6", "#AB47BC", "#FFA726", "#EC407A", "#00E676", "#FF5252", "#D4E157"]

    for name, series_data in oscillators.items():
        if isinstance(series_data, pd.Series):
            fig.add_trace(go.Scatter(
                x=df.index,
                y=series_data,
                mode="lines",
                name=name,
                line=dict(width=1.5, color=colors[row_idx % len(colors)])
            ), row=row_idx, col=1)
            
            # Threshold lines & shading
            if "RSI" in name or "MFI" in name:
                fig.add_hline(y=70, line_dash="dash", line_color="#FF5252", row=row_idx, col=1, annotation_text="Overbought 70", annotation_position="top right")
                fig.add_hline(y=30, line_dash="dash", line_color="#00E676", row=row_idx, col=1, annotation_text="Oversold 30", annotation_position="bottom right")
                fig.add_hline(y=50, line_dash="dot", line_color="#8B949E", row=row_idx, col=1)
            elif "Stochastic" in name or "Williams" in name:
                fig.add_hline(y=80 if "Stoch" in name else -20, line_dash="dash", line_color="#FF5252", row=row_idx, col=1)
                fig.add_hline(y=20 if "Stoch" in name else -80, line_dash="dash", line_color="#00E676", row=row_idx, col=1)
            elif name in ["CMF", "ROC", "CMO", "TRIX", "Chaikin_Osc"]:
                fig.add_hline(y=0, line_dash="dash", line_color="#8B949E", row=row_idx, col=1)

        elif isinstance(series_data, pd.DataFrame):
            col_idx = 0
            for col in series_data.columns:
                if col == "Hist": # MACD Histogram bar chart
                    colors_hist = ["#00E676" if val >= 0 else "#FF5252" for val in series_data[col]]
                    fig.add_trace(go.Bar(
                        x=df.index,
                        y=series_data[col],
                        name=f"{name} (Histogram)",
                        marker_color=colors_hist
                    ), row=row_idx, col=1)
                else:
                    fig.add_trace(go.Scatter(
                        x=df.index,
                        y=series_data[col],
                        mode="lines",
                        name=f"{name} ({col})",
                        line=dict(width=1.5, color=colors[col_idx % len(colors)])
                    ), row=row_idx, col=1)
                col_idx += 1
                
            if "ADX" in name:
                fig.add_hline(y=25, line_dash="dash", line_color="#FFA726", row=row_idx, col=1, annotation_text="Strong Trend (25)", annotation_position="top right")
                
        row_idx += 1

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0B0F19",
        plot_bgcolor="#0F172A",
        height=210 * num_oscillators,
        margin=dict(l=20, r=20, t=30, b=20),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0, font=dict(size=11, color="#E2E8F0"), bgcolor="rgba(0,0,0,0)")
    )
    
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)", rangebreaks=get_plotly_rangebreaks(df))
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)", side="right")

    return fig
