import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from utils.helper import drop_holiday_nans, get_plotly_rangebreaks

def create_backtest_charts(backtest_results: dict) -> tuple[go.Figure, go.Figure]:
    """
    Renders Strategy Equity Curve vs Buy & Hold and Underwater Drawdown chart.
    """
    if not backtest_results:
        return None, None
        
    df = backtest_results.get("data", backtest_results.get("df", pd.DataFrame()))
    if df is not None and not df.empty:
        df = drop_holiday_nans(df)
    
    if df is None or df.empty:
        return None, None
        
    strat_eq_col = "StrategyEquity" if "StrategyEquity" in df.columns else ("Strategy_Equity" if "Strategy_Equity" in df.columns else None)
    bh_eq_col = "BuyHoldEquity" if "BuyHoldEquity" in df.columns else ("Benchmark_Equity" if "Benchmark_Equity" in df.columns else None)
    
    if not strat_eq_col or not bh_eq_col:
        return None, None

    # 1. Equity Curve
    fig_equity = go.Figure()
    fig_equity.add_trace(go.Scatter(
        x=df.index,
        y=df[strat_eq_col],
        mode="lines",
        name="Strategy Equity",
        line=dict(width=2, color="#00E676")
    ))
    fig_equity.add_trace(go.Scatter(
        x=df.index,
        y=df[bh_eq_col],
        mode="lines",
        name="Buy & Hold Benchmark",
        line=dict(width=1.5, color="#38BDF8", dash="dash")
    ))
    
    fig_equity.update_layout(
        title=dict(text="<b>Portfolio Equity Curve (Initial ₹100,000 / $100,000)</b>", font=dict(size=16, color="#FFFFFF")),
        template="plotly_dark",
        paper_bgcolor="#0B0F19",
        plot_bgcolor="#0F172A",
        height=400,
        margin=dict(l=30, r=30, t=50, b=30),
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)", rangebreaks=get_plotly_rangebreaks(df)),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)", side="right", title="Equity Value"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0, font=dict(size=11, color="#E2E8F0"), bgcolor="rgba(0,0,0,0)")
    )

    # 2. Drawdown Chart
    fig_dd = go.Figure()
    strat_dd_col = "StrategyDrawdown" if "StrategyDrawdown" in df.columns else ("Strategy_Drawdown" if "Strategy_Drawdown" in df.columns else None)
    bh_dd_col = "BuyHoldDrawdown" if "BuyHoldDrawdown" in df.columns else ("Benchmark_Drawdown" if "Benchmark_Drawdown" in df.columns else None)
    
    if strat_dd_col:
        fig_dd.add_trace(go.Scatter(
            x=df.index,
            y=df[strat_dd_col] * 100.0,
            mode="lines",
            name="Strategy Drawdown",
            fill="tozeroy",
            line=dict(width=1, color="#FF5252"),
            fillcolor="rgba(255, 82, 82, 0.2)"
        ))
    if bh_dd_col:
        fig_dd.add_trace(go.Scatter(
            x=df.index,
            y=df[bh_dd_col] * 100.0,
            mode="lines",
            name="Buy & Hold Drawdown",
            line=dict(width=1, color="#FFA726", dash="dash")
        ))
    
    fig_dd.update_layout(
        title=dict(text="<b>Underwater Drawdown (%)</b>", font=dict(size=16, color="#FFFFFF")),
        template="plotly_dark",
        paper_bgcolor="#0B0F19",
        plot_bgcolor="#0F172A",
        height=250,
        margin=dict(l=30, r=30, t=50, b=30),
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)", rangebreaks=get_plotly_rangebreaks(df)),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)", side="right", title="Drawdown %"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0, font=dict(size=11, color="#E2E8F0"), bgcolor="rgba(0,0,0,0)")
    )

    return fig_equity, fig_dd
