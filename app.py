import re
import os
import glob
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# ─────────────────────────────
# Streamlit Page configuration
# ─────────────────────────────
st.set_page_config(
    page_title="Slope Trading Analyzer",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────
# Global CSS
# ─────────────────────────────
st.markdown(
    """
<style>
    .main {
        padding-top: 2rem;
    }
    
    .stSelectbox > div > div > div {
        background-color: #f8f9fa;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .positive-return {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    
    .negative-return {
        background: linear-gradient(135deg, #fc4a1a 0%, #f7b733 100%);
    }
    
    .neutral-metric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    h1 {
        color: #2c3e50;
        font-weight: 700;
        margin-bottom: 2rem;
    }
    
    h2, h3 {
        color: #34495e;
        font-weight: 600;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #f8f9fa;
        border-radius: 10px 10px 0px 0px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #667eea;
        color: white;
    }

    /* YEARLY MINI CARDS */
    .yearly-container {
        display: flex;
        flex-direction: row;
        flex-wrap: nowrap;
        justify-content: flex-start;
        align-items: flex-start;
        gap: 16px;
        overflow-x: auto;
        padding-bottom: 6px;
    }

    .yearly-card {
        background: #f8f9fa;
        padding: 0.7rem 1rem;
        border-radius: 8px;
        min-width: 160px;
        max-width: 160px;
        border-left: 6px solid #888;
        flex-shrink: 0;
    }

    .yearly-card.positive {
        border-color: #2ecc71;
    }

    .yearly-card.negative {
        border-color: #e74c3c;
    }

    .yearly-title {
        font-weight: 700;
        font-size: 0.9rem;
        margin-bottom: 0.3rem;
    }

    .yearly-line {
        font-size: 0.75rem;
        margin: 0.15rem 0;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────
# Cached I/O helpers (speed-up)
# ─────────────────────────────


@st.cache_data
def cached_list_branches(trade_logs_path: str):
    csv_files = glob.glob(os.path.join(trade_logs_path, "*.csv"))
    branches = [os.path.splitext(os.path.basename(f))[0] for f in csv_files]
    branches.sort()
    return branches


@st.cache_data
def cached_load_csv(path: str, date_col: str):
    """
    Generic cached CSV loader: parses date column and sorts.
    IMPORTANT: Caller should .copy() before mutating.
    """
    df = pd.read_csv(path, parse_dates=[date_col])
    df.sort_values(date_col, inplace=True)
    return df


# ─────────────────────────────
# Core Analyzer
# ─────────────────────────────
class SlopeTradingAnalyzer:
    def __init__(self):
        self.trade_logs_path = "./trade_logs"
        self.tickers_path = "./tickers"

    # ───────────── I/O ─────────────
    def load_available_branches(self):
        """Load all available trading branches from the trade_logs directory (cached)."""
        try:
            return cached_list_branches(self.trade_logs_path)
        except Exception as e:
            st.error(f"Error loading branches: {e}")
            return []

    def load_branch_data(self, branch_name):
        """Load trading data for a specific branch (cached CSV)."""
        try:
            file_path = os.path.join(self.trade_logs_path, f"{branch_name}.csv")
            df = cached_load_csv(file_path, "Date").copy()
            return df
        except Exception as e:
            st.error(f"Error loading branch {branch_name}: {e}")
            return None

    def load_ticker_data(self, ticker):
        """Load price data for a specific ticker (cached CSV)."""
        try:
            file_path = os.path.join(self.tickers_path, f"{ticker}.csv")
            df = cached_load_csv(file_path, "Date").copy()
            return df
        except Exception as e:
            st.error(f"Error loading ticker {ticker}: {e}")
            return None

    def extract_ticker_from_branch(self, branch_name):
        """
        Extract ticker symbol from branch name.
        Pattern: 15D_RSI_AAPL_LT33 or 14D_RSI_ADEA_LT37_and_200D_RSI_ADEA_LT50
        Ticker is typically the 3rd token.
        """
        parts = branch_name.split("_")
        if len(parts) >= 3:
            return parts[2]
        return None

    # ───────────── Core calculations ─────────────
    def calculate_slope(self, prices: pd.Series, window: int) -> pd.Series:
        """Calculate slope over a rolling window (kept same logic but slightly cleaned)."""
        values = prices.values.astype(float)
        n = len(values)
        slopes = np.full(n, np.nan, dtype=float)

        if n < window:
            return pd.Series(slopes, index=prices.index)

        x = np.arange(window, dtype=float)

        for i in range(window - 1, n):
            y = values[i - window + 1 : i + 1]
            if len(y) > 1:
                # linear regression slope on window
                slope = np.polyfit(x, y, 1)[0]
                slope_pct = (slope * (window - 1) / y[0]) * 100 if y[0] != 0 else 0.0
                slopes[i] = slope_pct

        return pd.Series(slopes, index=prices.index)

    def apply_slope_filter(self, branch_data, ticker_data, slope_window, pos_threshold, neg_threshold):
        """
        Apply slope filtering with Flag-based trading logic.
        Logic kept identical to your original version for correctness.
        """
        merged = pd.merge(
            branch_data,
            ticker_data[["Date", "Close", "Volume"]],
            on="Date",
            how="left",
        ).sort_values("Date")

        merged["Slope"] = self.calculate_slope(merged["Close"], slope_window)

        merged["Flag"] = 0
        merged["Entry_Signal"] = 0
        merged["Exit_Signal"] = 0
        merged["InTrade"] = 0

        flag = 0
        in_position = False
        entry_price = None
        entry_date = None

        signals = []

        for i in range(len(merged)):
            row = merged.iloc[i]

            if pd.isna(row["Slope"]):
                merged.at[merged.index[i], "Flag"] = flag
                continue

            # Flag logic: Flag turns to 1 when RSI gets activated
            if row["Active"] == 1 and flag == 0:
                flag = 1

            # Entry condition: Slope is positive (green) AND Flag is 1
            if not in_position and flag == 1 and row["Slope"] > pos_threshold:
                in_position = True
                entry_price = row["Close"]
                entry_date = row["Date"]
                merged.at[merged.index[i], "Entry_Signal"] = 1
                merged.at[merged.index[i], "InTrade"] = 1

            elif in_position:
                merged.at[merged.index[i], "InTrade"] = 1

                # Exit condition: Slope is no longer green (falls below positive threshold)
                if row["Slope"] <= pos_threshold:
                    in_position = False
                    exit_price = row["Close"]
                    exit_date = row["Date"]
                    merged.at[merged.index[i], "Exit_Signal"] = 1
                    merged.at[merged.index[i], "InTrade"] = 0

                    flag = 0

                    if entry_price and entry_price != 0:
                        trade_return = (exit_price - entry_price) / entry_price * 100
                        days_held = (exit_date - entry_date).days

                        signals.append(
                            {
                                "Entry_Date": entry_date,
                                "Exit_Date": exit_date,
                                "Entry_Price": entry_price,
                                "Exit_Price": exit_price,
                                "Return_Pct": trade_return,
                                "Days_Held": days_held,
                            }
                        )

            merged.at[merged.index[i], "Flag"] = flag

        signals_df = pd.DataFrame(signals)
        if not signals_df.empty:
            signals_df["Entry_Date"] = pd.to_datetime(signals_df["Entry_Date"])
            signals_df["Exit_Date"] = pd.to_datetime(signals_df["Exit_Date"])

        return merged, signals_df

    # ───────────── Metrics ─────────────
    def calculate_performance_metrics(self, signals_df):
        """Calculate comprehensive performance metrics."""
        if signals_df.empty:
            return {}

        returns = signals_df["Return_Pct"].values
        num_trades = len(returns)

        total_return = float(np.sum(returns))
        win_rate = float(len(returns[returns > 0]) / num_trades * 100) if num_trades > 0 else 0.0
        avg_return = float(np.mean(returns)) if num_trades > 0 else 0.0
        avg_days_held = float(signals_df["Days_Held"].mean()) if num_trades > 0 else 0.0

        max_drawdown = self.calculate_max_drawdown(returns)
        sharpe_ratio = self.calculate_sharpe_ratio(returns)
        volatility = float(np.std(returns)) if num_trades > 0 else 0.0

        if not signals_df.empty:
            total_days_in_market = float(signals_df["Days_Held"].sum())
            date_range = (signals_df["Exit_Date"].max() - signals_df["Entry_Date"].min()).days
            time_in_market = float(total_days_in_market / date_range * 100) if date_range > 0 else 0.0
        else:
            time_in_market = 0.0

        return {
            "Total_Return_Pct": total_return,
            "Win_Rate_Pct": win_rate,
            "Max_Drawdown_Pct": max_drawdown,
            "Num_Trades": num_trades,
            "Time_In_Market_Pct": time_in_market,
            "Avg_Days_Held": avg_days_held,
            "Avg_Return_Pct": avg_return,
            "Sharpe_Ratio": sharpe_ratio,
            "Volatility_Pct": volatility,
        }

    def calculate_max_drawdown(self, returns):
        if len(returns) == 0:
            return 0.0
        equity = np.cumprod(1 + np.array(returns) / 100.0)
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity / running_max - 1.0) * 100
        return float(np.min(drawdown))

    def calculate_sharpe_ratio(self, returns, risk_free_rate=0.02):
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        excess_returns = np.mean(returns) / 100.0 - risk_free_rate / 252.0
        return float(
            (excess_returns * np.sqrt(252))
            / (np.std(returns) / 100.0)
        )

    def compute_yearly_stats(self, signals_df):
        """Compute per-year stats: Return, Max DD, Trades, Avg Hold."""
        if signals_df.empty:
            return {}

        df = signals_df.copy()
        df["Year"] = df["Exit_Date"].dt.year

        yearly = {}
        for year, group in df.groupby("Year"):
            returns = group["Return_Pct"].values
            if len(returns) == 0:
                continue

            total_return = float(np.sum(returns))
            equity = np.cumprod(1 + returns / 100.0)
            running_max = np.maximum.accumulate(equity)
            dd = (equity / running_max - 1.0) * 100
            max_dd = float(np.min(dd))

            yearly[year] = {
                "Return": total_return,
                "MaxDD": max_dd,
                "Trades": int(len(group)),
                "AvgHold": float(group["Days_Held"].mean()),
            }

        return yearly

    # ───────────── Charts ─────────────
    def create_price_chart(self, merged_data, signals_df, branch_name, slope_window, pos_threshold, neg_threshold):
        """
        Single-panel price chart:
        - Price line
        - Slope segments (green/gray)
        - Entry/Exit markers & vertical lines
        - RSI activation markers
        - Flag markers
        """
        colors = {
            "price_base": "#e5e7eb",
            "slope_segment_green": "#10b981",
            "slope_segment_gray": "#6b7280",
            "rsi_entry": "#3b82f6",
            "rsi_exit_green": "#16a34a",
            "rsi_exit_gray": "#6b7280",
            "rsi_activation": "#8b5cf6",
            "flag_activation": "#8b5cf6",
        }

        df = merged_data.copy().set_index("Date")

        if len(df) > 0:
            last_date = df.index.max()
            start_date = last_date - pd.DateOffset(years=5)
            df = df[df.index >= start_date]

        if df.empty:
            fig = go.Figure()
            fig.update_layout(
                template="plotly_white", title="No data available for the last 5 years"
            )
            return fig

        fig = go.Figure()

        # Base price line
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Close"],
                mode="lines",
                line=dict(color=colors["price_base"], width=2),
                name="Price",
                hoverinfo="skip",
            )
        )

        # Slope-based segments
        slope_series = df["Slope"]
        slope_entry_idx = (slope_series > pos_threshold) & (slope_series.shift(1) <= pos_threshold)
        slope_exit_idx = (slope_series < neg_threshold) & (slope_series.shift(1) >= neg_threshold)

        slope_entry_dates = slope_series.index[slope_entry_idx].tolist()
        slope_exit_dates = slope_series.index[slope_exit_idx].tolist()

        slope_trades = []
        i = j = 0
        while i < len(slope_entry_dates) and j < len(slope_exit_dates):
            entry_date = slope_entry_dates[i]
            while j < len(slope_exit_dates) and slope_exit_dates[j] <= entry_date:
                j += 1
            if j < len(slope_exit_dates):
                exit_date = slope_exit_dates[j]
                if entry_date in df.index and exit_date in df.index:
                    entry_price = df.loc[entry_date, "Close"]
                    exit_price = df.loc[exit_date, "Close"]
                    slope_return = (exit_price - entry_price) / entry_price * 100
                    slope_trades.append(
                        {
                            "entry_date": entry_date,
                            "exit_date": exit_date,
                            "return": slope_return,
                        }
                    )
                j += 1
            i += 1

        for seg in slope_trades:
            mask = (df.index >= seg["entry_date"]) & (df.index <= seg["exit_date"])
            segment_df = df[mask]
            if len(segment_df) > 1:
                color = (
                    colors["slope_segment_green"]
                    if seg["return"] > 0
                    else colors["slope_segment_gray"]
                )
                fig.add_trace(
                    go.Scatter(
                        x=segment_df.index,
                        y=segment_df["Close"],
                        mode="lines",
                        line=dict(color=color, width=4),
                        name="Slope Segment",
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )

        shapes = []
        if not signals_df.empty:
            mask_signals = signals_df["Entry_Date"] >= df.index.min()
            s_df = signals_df[mask_signals].copy()

            if not s_df.empty:
                for d in s_df["Entry_Date"]:
                    shapes.append(
                        dict(
                            type="line",
                            x0=d,
                            x1=d,
                            y0=0,
                            y1=1,
                            xref="x",
                            yref="paper",
                            line=dict(
                                color="rgba(59,130,246,0.40)", width=1.2, dash="dot"
                            ),
                        )
                    )
                for d in s_df["Exit_Date"]:
                    shapes.append(
                        dict(
                            type="line",
                            x0=d,
                            x1=d,
                            y0=0,
                            y1=1,
                            xref="x",
                            yref="paper",
                            line=dict(
                                color="rgba(22,163,74,0.40)", width=1.2, dash="dot"
                            ),
                        )
                    )

                fig.add_trace(
                    go.Scatter(
                        x=s_df["Entry_Date"],
                        y=s_df["Entry_Price"],
                        mode="markers+text",
                        marker=dict(
                            color=colors["rsi_entry"],
                            size=14,
                            symbol="triangle-up",
                            line=dict(color="white", width=1),
                        ),
                        text=["ENTRY"] * len(s_df),
                        textposition="top center",
                        name="Entry",
                        hoverinfo="skip",
                    )
                )

                exit_colors = [
                    colors["rsi_exit_green"] if r > 0 else colors["rsi_exit_gray"]
                    for r in s_df["Return_Pct"]
                ]
                fig.add_trace(
                    go.Scatter(
                        x=s_df["Exit_Date"],
                        y=s_df["Exit_Price"],
                        mode="markers+text",
                        marker=dict(
                            color=exit_colors,
                            size=14,
                            symbol="triangle-down",
                            line=dict(color="white", width=1),
                        ),
                        text=[f"{r:+.1f}%" for r in s_df["Return_Pct"]],
                        textposition="bottom center",
                        name="Exit",
                        hoverinfo="skip",
                    )
                )

        # RSI activation markers
        if "Active" in df.columns:
            rsi_points = df[(df["Active"] == 1) & (df["Active"].shift(1) == 0)]
            if not rsi_points.empty:
                fig.add_trace(
                    go.Scatter(
                        x=rsi_points.index,
                        y=rsi_points["Close"],
                        mode="markers+text",
                        marker=dict(
                            color=colors["rsi_activation"],
                            size=12,
                            symbol="triangle-up",
                        ),
                        text=["RSI"] * len(rsi_points),
                        textposition="top center",
                        name="RSI Activation",
                        hoverinfo="skip",
                    )
                )

        # Flag markers
        if "Flag" in df.columns:
            flag_points = df[(df["Flag"] == 1) & (df["Flag"].shift(1) == 0)]
            if not flag_points.empty:
                fig.add_trace(
                    go.Scatter(
                        x=flag_points.index,
                        y=flag_points["Close"],
                        mode="markers+text",
                        marker=dict(
                            color=colors["flag_activation"],
                            size=14,
                            symbol="diamond",
                        ),
                        text=["FLAG"] * len(flag_points),
                        textposition="middle right",
                        name="Flag",
                        hoverinfo="skip",
                    )
                )

        last_date = df.index.max()
        start_date = df.index.min()

        fig.update_layout(
            shapes=shapes,
            template="plotly_white",
            hovermode=False,
            height=750,
            margin=dict(l=60, r=40, t=100, b=60),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.06,
                xanchor="right",
                x=1,
                bgcolor="rgba(255,255,255,0.7)",
                bordercolor="rgba(0,0,0,0.2)",
                borderwidth=1,
            ),
            title=dict(
                text=f"<b>{branch_name}</b> — Price + Slope Segments + Signals (Last 5 Years Max)",
                x=0.5,
                y=0.98,
                font=dict(size=18),
            ),
            xaxis=dict(
                range=[start_date, last_date],
                showgrid=True,
                gridcolor="rgba(0,0,0,0.15)",
                gridwidth=1,
                zeroline=False,
                showline=True,
                linecolor="rgba(0,0,0,0.40)",
                linewidth=1.2,
                rangeselector=dict(
                    buttons=[
                        dict(count=7, label="7D", step="day", stepmode="backward"),
                        dict(count=1, label="1M", step="month", stepmode="backward"),
                        dict(count=3, label="3M", step="month", stepmode="backward"),
                        dict(count=6, label="6M", step="month", stepmode="backward"),
                        dict(count=1, label="1Y", step="year", stepmode="backward"),
                        dict(count=2, label="2Y", step="year", stepmode="backward"),
                        dict(count=3, label="3Y", step="year", stepmode="backward"),
                        dict(count=4, label="4Y", step="year", stepmode="backward"),
                        dict(count=5, label="5Y", step="year", stepmode="backward"),
                        dict(step="all", label="ALL"),
                    ],
                    bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="rgba(0,0,0,0.3)",
                    borderwidth=1,
                    font=dict(size=12),
                ),
                rangeslider=dict(
                    visible=True,
                    thickness=0.05,
                    bgcolor="rgba(230,230,230,0.4)",
                ),
                type="date",
            ),
            yaxis=dict(
                title="Price",
                tickformat="$,.2f",
                showgrid=True,
                gridcolor="rgba(0,0,0,0.12)",
                gridwidth=1.2,
                zeroline=False,
                showline=True,
                linecolor="rgba(0,0,0,0.40)",
                linewidth=1.2,
            ),
        )

        return fig

    # ───────────── RSI trigger price chart (dynamic) ─────────────
    def create_rsi_trigger_price_chart(self, ticker_df, branch_df, branch_name):
        """
        Fully TradingView-like price chart for RSI triggers.
        Clean candles, volume bars, hover crosshair, TV-style period selectors.
        """
        rsi_period, rsi_threshold = self.parse_rsi_from_branch(branch_name)

        df = ticker_df.copy()
        df = df.merge(branch_df[["Date", "Active"]], on="Date", how="left")
        df["Active"] = df["Active"].fillna(0)
        df["RSI"] = self.compute_rsi(df["Close"], rsi_period)

        # Time filter (last 1 year)
        max_date = df["Date"].max()
        df = df[df["Date"] >= max_date - pd.DateOffset(years=1)]

        # TradingView candle colors
        bull = "#26a69a"
        bear = "#ef5350"

        df["Volume_Color"] = [
            bull if df["Close"].iloc[i] >= df["Open"].iloc[i] else bear
            for i in range(len(df))
        ]

        oversold = df[df["RSI"] < rsi_threshold]

        # Figure
        fig = go.Figure()

        # Candles
        fig.add_trace(
            go.Candlestick(
                x=df["Date"],
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                increasing_line_color=bull,
                decreasing_line_color=bear,
                increasing_fillcolor=bull,
                decreasing_fillcolor=bear,
                name="Price",
            )
        )

        # Volume bars
        fig.add_trace(
            go.Bar(
                x=df["Date"],
                y=df["Volume"],
                marker_color=df["Volume_Color"],
                opacity=0.35,
                name="Volume",
                yaxis="y2",
            )
        )

        # RSI trigger markers
        fig.add_trace(
            go.Scatter(
                x=oversold["Date"],
                y=oversold["Close"],
                mode="markers",
                name=f"RSI < {rsi_threshold}",
                marker=dict(
                    color="#1e88e5",
                    size=12,
                    symbol="triangle-up",
                    line=dict(color="white", width=1),
                ),
            )
        )

        # Layout (TradingView style)
        fig.update_layout(
            title=dict(
                text=f"<b>{branch_name}</b> — RSI Trigger Chart",
                x=0.5,
                font=dict(size=20),
            ),
            height=750,
            margin=dict(l=60, r=40, t=60, b=40),
            plot_bgcolor="white",
            paper_bgcolor="white",
            hovermode="x unified",
            showlegend=False,
            xaxis=dict(
                type="date",
                rangeslider=dict(visible=False),
                showgrid=False,
                rangeselector=dict(
                    buttons=[
                        dict(count=7, label="7D", step="day", stepmode="backward"),
                        dict(count=1, label="1M", step="month", stepmode="backward"),
                        dict(count=3, label="3M", step="month", stepmode="backward"),
                        dict(count=6, label="6M", step="month", stepmode="backward"),
                        dict(step="year", stepmode="todate", label="YTD"),   # TradingView-style YTD
                        dict(count=1, label="1Y", step="year", stepmode="backward"),
                        dict(step="all", label="ALL"),
                    ],
                    bgcolor="rgba(255,255,255,0.8)",
                    borderwidth=1,
                    bordercolor="rgba(200,200,200,0.4)",
                ),
            ),
            yaxis=dict(
                title="Price",
                showgrid=True,
                gridcolor="rgba(200,200,200,0.25)",
                zeroline=False,
            ),
            yaxis2=dict(
                overlaying="y",
                side="right",
                showgrid=False,
                visible=False,
                range=[0, df["Volume"].max() * 5],
            ),
        )

        # Crosshair
        fig.update_xaxes(
            showspikes=True,
            spikethickness=1,
            spikecolor="#888",
            spikedash="solid",
            spikesnap="cursor",
            showline=True,
            linecolor="#ccc",
        )
        fig.update_yaxes(
            showspikes=True,
            spikethickness=1,
            spikecolor="#888",
            spikedash="solid",
            spikesnap="cursor",
            showline=True,
            linecolor="#ccc",
        )

        return fig



    # ───────────── RSI helpers ─────────────
    def parse_rsi_from_branch(self, branch_name):
        """
        Extract RSI period and threshold from ANY branch name.
        Examples:
          - 15D_RSI_A_LT33
          - 5d_RSI_AAON_LT26
          - 14D_RSI_ADEA_LT37_and_200D_RSI_ADEA_LT50
          - 15D_RSI_A_LT33_daily_trade_log
        Returns: (period, threshold)
        """
        base = branch_name.split("_daily_trade_log")[0]

        m = re.search(r"(\d+)[dD]_RSI_[^_]+_(LT|GT)(\d+)", base)
        if m:
            period = int(m.group(1))
            threshold = int(m.group(3))
            return period, threshold

        parts = base.split("_")
        rsi_period = None
        rsi_threshold = None

        for part in parts:
            if (part.endswith("D") or part.endswith("d")) and part[:-1].isdigit():
                rsi_period = int(part[:-1])

            if part.startswith("LT") and part[2:].isdigit():
                rsi_threshold = int(part[2:])
            if part.startswith("GT") and part[2:].isdigit():
                rsi_threshold = int(part[2:])

            if rsi_period is not None and rsi_threshold is not None:
                break

        if rsi_period is None:
            rsi_period = 14
        if rsi_threshold is None:
            rsi_threshold = 30

        return rsi_period, rsi_threshold

    def compute_rsi(self, close_series: pd.Series, period: int):
        delta = close_series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


# ─────────────────────────────
# Streamlit helpers
# ─────────────────────────────
@st.cache_resource
def init_analyzer():
    return SlopeTradingAnalyzer()


def clean_branch_name(branch_raw: str) -> str:
    """
    Convert '15D_RSI_A_LT33_daily_trade_log' → '15D RSI A LT33'
    """
    branch_raw = branch_raw.split("_daily_trade_log")[0]
    pretty = branch_raw.replace("_", " ")
    return pretty


def display_performance_metrics(metrics):
    if not metrics:
        st.warning("No metrics available.")
        return

    metric_items = {
        "Total Return": f"{metrics.get('Total_Return_Pct', 0):.2f}%",
        "Win Rate": f"{metrics.get('Win_Rate_Pct', 0):.2f}%",
        "Max Drawdown": f"{metrics.get('Max_Drawdown_Pct', 0):.2f}%",
        "Trades": str(metrics.get("Num_Trades", 0)),
        "Avg Hold Days": f"{metrics.get('Avg_Days_Held', 0):.1f}",
    }

    cols = st.columns(len(metric_items))
    for col, (title, value) in zip(cols, metric_items.items()):
        with col:
            st.markdown(
                f"""
            <div style="
                background: #f8f9fa;
                padding: 0.8rem 1rem;
                border-radius: 8px;
                border-left: 6px solid #667eea;
                min-width: 140px;
                text-align: center;
            ">
                <div style="font-size: 0.75rem; font-weight: 600;">{title}</div>
                <div style="font-size: 1rem; font-weight: 700; margin-top: 4px;">{value}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )


def display_yearly_cards(yearly_dict):
    if not yearly_dict:
        st.info("No yearly results available.")
        return

    years = sorted(yearly_dict.keys())
    last_5_years = years[-5:]

    cols = st.columns(len(last_5_years))
    for col, year in zip(cols, last_5_years):
        stats = yearly_dict[year]
        border_color = "#2ecc71" if stats["Return"] > 0 else "#e74c3c"

        with col:
            st.markdown(
                f"""
            <div style="
                background:#f8f9fa;
                padding:0.7rem 1rem;
                border-left:6px solid {border_color};
                border-radius:8px;
                min-width:140px;
            ">
                <div style="font-weight:700; font-size:0.9rem;">{year}</div>
                <div style="font-size:0.75rem;">Return: {stats['Return']:.2f}%</div>
                <div style="font-size:0.75rem;">Max DD: {stats['MaxDD']:.2f}%</div>
                <div style="font-size:0.75rem;">Trades: {stats['Trades']}</div>
                <div style="font-size:0.75rem;">Avg Hold: {stats['AvgHold']:.1f} days</div>
            </div>
            """,
                unsafe_allow_html=True,
            )


# ─────────────────────────────
# Main App
# ─────────────────────────────
def main():
    st.title("Slope Trading Analyzer")
    st.markdown("### Advanced RSI + Slope Filter Backtesting System")

    analyzer = init_analyzer()

    # ───────── Sidebar ─────────
    with st.sidebar:
        st.header("🔧 Configuration")

        st.subheader("Slope Parameters")
        slope_window = st.slider("Slope Length (days)", 5, 30, 15)
        pos_threshold = st.slider("Positive Threshold (%)", 0.0, 20.0, 5.0, 0.5)
        neg_threshold = st.slider("Negative Threshold (%)", -10.0, 10.0, 0.0, 0.5)

        st.divider()

        st.subheader("Branch Selection")
        available_branches = analyzer.load_available_branches()
        branch_display_map = {clean_branch_name(b): b for b in available_branches}

        if not available_branches:
            st.error("No trading branches found in ./trade_logs/")
            st.stop()

        branch_pretty = st.selectbox(
            "Select Branch for Detailed Analysis",
            list(branch_display_map.keys()),
        )
        branch_to_analyze = branch_display_map[branch_pretty]

        st.divider()
        st.markdown("### Analysis Options")
        show_yearly = st.checkbox("Show Yearly Breakdown", value=True)
        show_individual_charts = st.checkbox("Show Individual Charts", value=True)

    all_branches = available_branches

    # Params snapshot for overall caching
    current_params = (slope_window, pos_threshold, neg_threshold)

    tab_individual, tab_overall, tab_reports = st.tabs(
        ["Individual Analysis", "Overall Results", "Detailed Reports"]
    )

    # ───────── Individual Analysis ─────────
    with tab_individual:
        if branch_to_analyze:
            branch_data = analyzer.load_branch_data(branch_to_analyze)
            ticker = analyzer.extract_ticker_from_branch(branch_to_analyze)

            if branch_data is not None and ticker:
                ticker_data = analyzer.load_ticker_data(ticker)

                if ticker_data is not None:
                    merged_data, signals_df = analyzer.apply_slope_filter(
                        branch_data,
                        ticker_data,
                        slope_window,
                        pos_threshold,
                        neg_threshold,
                    )

                    metrics = analyzer.calculate_performance_metrics(signals_df)

                    st.subheader(f"Performance Metrics - {branch_pretty}")
                    display_performance_metrics(metrics)

                    yearly = analyzer.compute_yearly_stats(signals_df)
                    if show_yearly:
                        st.subheader("Yearly Performance Breakdown")
                        display_yearly_cards(yearly)

                    st.divider()

                    if show_individual_charts:
                        st.subheader(
                            "Price Chart with Slope Segments & Signals (Last 5 Years Max)"
                        )
                        chart = analyzer.create_price_chart(
                            merged_data,
                            signals_df,
                            branch_to_analyze,
                            slope_window,
                            pos_threshold,
                            neg_threshold,
                        )
                        st.plotly_chart(chart, use_container_width=True)

                        st.subheader("RSI Trigger Price Chart (Gray Candles)")
                        rsi_period, rsi_threshold = analyzer.parse_rsi_from_branch(
                            branch_to_analyze
                        )
                        st.caption(
                            f"Debug RSI: period={rsi_period}, threshold={rsi_threshold}, branch={branch_to_analyze}"
                        )
                        rsi_chart = analyzer.create_rsi_trigger_price_chart(
                            ticker_data, branch_data, branch_to_analyze
                        )
                        st.plotly_chart(rsi_chart, use_container_width=True)

                    if not signals_df.empty:
                        st.subheader("Trade Details")
                        trade_display = signals_df.copy()
                        trade_display["Entry_Date"] = trade_display[
                            "Entry_Date"
                        ].dt.strftime("%Y-%m-%d")
                        trade_display["Exit_Date"] = trade_display[
                            "Exit_Date"
                        ].dt.strftime("%Y-%m-%d")
                        trade_display = trade_display.round(2)
                        st.dataframe(trade_display, use_container_width=True)

    # ───────── Overall Results ─────────
    with tab_overall:
        st.header("Overall Performance Summary (All Branches)")

        # Button-based recompute to avoid heavy loop on every interaction
        run_overall = st.button("Run / Recompute Overall Analysis")

        params_changed = (
            st.session_state.get("overall_params") != current_params
        )

        if run_overall or st.session_state.get("overall_results") is None or params_changed:
            all_results = []
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, branch in enumerate(all_branches):
                status_text.text(f"Processing {branch}... ({i+1}/{len(all_branches)})")

                branch_data = analyzer.load_branch_data(branch)
                if branch_data is None:
                    continue

                ticker = analyzer.extract_ticker_from_branch(branch)
                if not ticker:
                    continue

                ticker_data = analyzer.load_ticker_data(ticker)
                if ticker_data is None:
                    continue

                merged_data, signals_df = analyzer.apply_slope_filter(
                    branch_data,
                    ticker_data,
                    slope_window,
                    pos_threshold,
                    neg_threshold,
                )

                metrics = analyzer.calculate_performance_metrics(signals_df)
                metrics["Branch"] = branch
                metrics["Ticker"] = ticker
                all_results.append(metrics)

                progress_bar.progress((i + 1) / len(all_branches))

            status_text.text("Analysis complete!")
            progress_bar.empty()
            status_text.empty()

            if all_results:
                st.session_state["overall_results"] = pd.DataFrame(all_results)
                st.session_state["overall_params"] = current_params
            else:
                st.session_state["overall_results"] = None

        results_df = st.session_state.get("overall_results")

        if results_df is not None and not results_df.empty:
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                avg_return = results_df["Total_Return_Pct"].mean()
                st.metric("Average Total Return", f"{avg_return:.2f}%")
            with col2:
                avg_win_rate = results_df["Win_Rate_Pct"].mean()
                st.metric("Average Win Rate", f"{avg_win_rate:.1f}%")
            with col3:
                avg_trades = results_df["Num_Trades"].mean()
                st.metric("Average Trades", f"{avg_trades:.1f}")
            with col4:
                avg_drawdown = results_df["Max_Drawdown_Pct"].mean()
                st.metric("Average Max DD", f"{avg_drawdown:.2f}%")

            st.divider()

            st.subheader("Branch Performance Comparison")
            display_df = results_df.copy().round(2)
            st.dataframe(display_df, use_container_width=True)

            st.subheader("Performance Visualization")
            col1, col2 = st.columns(2)

            with col1:
                fig_scatter = px.scatter(
                    results_df,
                    x="Win_Rate_Pct",
                    y="Total_Return_Pct",
                    hover_data=["Branch", "Num_Trades"],
                    title="Return vs Win Rate",
                    labels={
                        "Win_Rate_Pct": "Win Rate (%)",
                        "Total_Return_Pct": "Total Return (%)",
                    },
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

            with col2:
                fig_hist = px.histogram(
                    results_df,
                    x="Total_Return_Pct",
                    title="Return Distribution",
                    labels={"Total_Return_Pct": "Total Return (%)"},
                )
                st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.info("Click the button above to run the overall analysis.")

    # ───────── Detailed Reports ─────────
    with tab_reports:
        st.header("Detailed Reports")

        results_df = st.session_state.get("overall_results")
        if results_df is not None and not results_df.empty:
            st.subheader("Export Results")

            if st.button("Generate CSV Report"):
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV Report",
                    data=csv,
                    file_name=f"slope_trading_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )

            if show_yearly:
                st.subheader(
                    "Yearly Performance Breakdown (Note: per-branch yearly stats shown in Individual Analysis)"
                )
                st.info(
                    "Global yearly aggregation across all branches can be added here if needed."
                )

            st.subheader("Current Parameters Summary")
            param_df = pd.DataFrame(
                {
                    "Parameter": [
                        "Slope Window",
                        "Positive Threshold",
                        "Negative Threshold",
                    ],
                    "Value": [
                        f"{slope_window} days",
                        f"{pos_threshold}%",
                        f"{neg_threshold}%",
                    ],
                }
            )
            st.table(param_df)
        else:
            st.info("Run the Overall Results tab at least once to populate the reports.")


if __name__ == "__main__":
    main()
