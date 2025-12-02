import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import os
import glob
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Slope Trading Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
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
</style>
""", unsafe_allow_html=True)

class SlopeTradingAnalyzer:
    def __init__(self):
        self.trade_logs_path = "./trade_logs"
        self.tickers_path = "./tickers"
        
    def load_available_branches(self):
        """Load all available trading branches from the trade_logs directory"""
        try:
            csv_files = glob.glob(os.path.join(self.trade_logs_path, "*.csv"))
            branches = [os.path.splitext(os.path.basename(f))[0] for f in csv_files]
            return sorted(branches)
        except Exception as e:
            st.error(f"Error loading branches: {e}")
            return []
    
    def load_branch_data(self, branch_name):
        """Load trading data for a specific branch"""
        try:
            file_path = os.path.join(self.trade_logs_path, f"{branch_name}.csv")
            df = pd.read_csv(file_path)
            df['Date'] = pd.to_datetime(df['Date'])
            return df
        except Exception as e:
            st.error(f"Error loading branch {branch_name}: {e}")
            return None
    
    def load_ticker_data(self, ticker):
        """Load price data for a specific ticker"""
        try:
            file_path = os.path.join(self.tickers_path, f"{ticker}.csv")
            df = pd.read_csv(file_path)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
            return df
        except Exception as e:
            st.error(f"Error loading ticker {ticker}: {e}")
            return None
    
    def extract_ticker_from_branch(self, branch_name):
        """Extract ticker symbol from branch name"""
        # Pattern: XdD_RSI_TICKER_LTxx or XdD_RSI_TICKER_LTxx_and_...
        parts = branch_name.split('_')
        if len(parts) >= 3:
            return parts[2]  # TICKER is usually the 3rd part
        return None
    
    def calculate_slope(self, prices, window):
        """Calculate slope over a rolling window"""
        slopes = []
        for i in range(len(prices)):
            if i < window - 1:
                slopes.append(np.nan)
            else:
                y = prices[i-window+1:i+1].values
                x = np.arange(len(y))
                if len(y) > 1:
                    slope = np.polyfit(x, y, 1)[0]
                    # Convert to percentage
                    slope_pct = (slope * (window-1) / y[0]) * 100 if y[0] != 0 else 0
                    slopes.append(slope_pct)
                else:
                    slopes.append(np.nan)
        return pd.Series(slopes, index=prices.index)
    
    def apply_slope_filter(self, branch_data, ticker_data, slope_window, pos_threshold, neg_threshold):
        """Apply slope filtering with Flag-based trading logic"""
        # Merge branch data with ticker data
        merged = pd.merge(branch_data, ticker_data[['Date', 'Close', 'Volume']], on='Date', how='left')
        merged = merged.sort_values('Date')
        
        # Calculate slope
        merged['Slope'] = self.calculate_slope(merged['Close'], slope_window)
        
        # Initialize new columns
        merged['Flag'] = 0  # Flag for RSI activation
        merged['Entry_Signal'] = 0
        merged['Exit_Signal'] = 0
        merged['InTrade'] = 0
        
        # Flag and trading logic
        flag = 0
        in_position = False
        entry_price = None
        entry_date = None
        
        signals = []
        
        for i in range(len(merged)):
            row = merged.iloc[i]
            
            if pd.isna(row['Slope']):
                merged.at[merged.index[i], 'Flag'] = flag
                continue
            
            # Flag logic: Flag turns to 1 when RSI gets activated
            if row['Active'] == 1 and flag == 0:
                flag = 1
            
            # Entry condition: Slope is positive (green) AND Flag is 1
            if not in_position and flag == 1 and row['Slope'] > pos_threshold:
                in_position = True
                entry_price = row['Close']
                entry_date = row['Date']
                merged.at[merged.index[i], 'Entry_Signal'] = 1
                merged.at[merged.index[i], 'InTrade'] = 1
                
            # Continue holding position
            elif in_position:
                merged.at[merged.index[i], 'InTrade'] = 1
                
                # Exit condition: Slope is no longer green (falls below positive threshold)
                if row['Slope'] <= pos_threshold:
                    in_position = False
                    exit_price = row['Close']
                    exit_date = row['Date']
                    merged.at[merged.index[i], 'Exit_Signal'] = 1
                    merged.at[merged.index[i], 'InTrade'] = 0
                    
                    # After we sell, turn flag to 0
                    flag = 0
                    
                    # Calculate trade return
                    if entry_price and entry_price != 0:
                        trade_return = (exit_price - entry_price) / entry_price * 100
                        days_held = (exit_date - entry_date).days
                        
                        signals.append({
                            'Entry_Date': entry_date,
                            'Exit_Date': exit_date,
                            'Entry_Price': entry_price,
                            'Exit_Price': exit_price,
                            'Return_Pct': trade_return,
                            'Days_Held': days_held
                        })
            
            # Store current flag value
            merged.at[merged.index[i], 'Flag'] = flag
        
        return merged, pd.DataFrame(signals)
    
    def calculate_performance_metrics(self, signals_df):
        """Calculate comprehensive performance metrics"""
        if signals_df.empty:
            return {}
            
        returns = signals_df['Return_Pct'].values
        
        # Basic metrics
        total_return = np.sum(returns)
        num_trades = len(returns)
        win_rate = len(returns[returns > 0]) / num_trades * 100 if num_trades > 0 else 0
        avg_return = np.mean(returns) if num_trades > 0 else 0
        avg_days_held = np.mean(signals_df['Days_Held']) if num_trades > 0 else 0
        
        # Risk metrics
        max_drawdown = self.calculate_max_drawdown(returns)
        sharpe_ratio = self.calculate_sharpe_ratio(returns)
        volatility = np.std(returns) if num_trades > 0 else 0
        
        # Time in market
        if not signals_df.empty:
            total_days_in_market = signals_df['Days_Held'].sum()
            date_range = (signals_df['Exit_Date'].max() - signals_df['Entry_Date'].min()).days
            time_in_market = (total_days_in_market / date_range * 100) if date_range > 0 else 0
        else:
            time_in_market = 0
            
        return {
            'Total_Return_Pct': total_return,
            'Win_Rate_Pct': win_rate,
            'Max_Drawdown_Pct': max_drawdown,
            'Num_Trades': num_trades,
            'Time_In_Market_Pct': time_in_market,
            'Avg_Days_Held': avg_days_held,
            'Avg_Return_Pct': avg_return,
            'Sharpe_Ratio': sharpe_ratio,
            'Volatility_Pct': volatility
        }
    
    def calculate_max_drawdown(self, returns):
        """Calculate maximum drawdown"""
        if len(returns) == 0:
            return 0
        cumulative = np.cumprod(1 + np.array(returns) / 100)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative / running_max - 1) * 100
        return np.min(drawdown)
    
    def calculate_sharpe_ratio(self, returns, risk_free_rate=0.02):
        """Calculate Sharpe ratio (annualized)"""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0
        excess_returns = np.mean(returns) / 100 - risk_free_rate / 252  # Daily risk-free rate
        return (excess_returns * np.sqrt(252)) / (np.std(returns) / 100) if np.std(returns) != 0 else 0
    
    def create_price_chart(self, merged_data, signals_df, branch_name, slope_window, pos_threshold, neg_threshold):
        """Create enhanced 4-panel chart with slope analysis"""
        
        colors = {
            "price_base": "#e5e7eb",
            "ma_20": "#f59e0b",
            "ma_50": "#8b5cf6", 
            "volume": "#94a3b8",
            "slope": "#3b82f6",
            "slope_entry": "#16a34a",
            "slope_exit": "#6b7280",  # Changed from red to gray
            "slope_segment_green": "#10b981",
            "slope_segment_gray": "#6b7280",  # Changed from red to gray
            "rsi_entry": "#3b82f6",
            "rsi_exit_green": "#16a34a",
            "rsi_exit_gray": "#6b7280",  # Changed from red to gray
            "rsi_activation": "#8b5cf6",  # Purple for RSI activation
            "position": "#059669",
        }

        df_view = merged_data.copy().set_index("Date")
        
        # Ensure Volume column exists
        if "Volume" not in df_view.columns:
            df_view["Volume"] = np.nan
            
        # InTrade column is now created in the apply_slope_filter method

        fig = make_subplots(
            rows=5,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.4, 0.18, 0.18, 0.12, 0.12],
            specs=[
                [{}],  # Price
                [{}],  # Volume
                [{}],  # Slope
                [{}],  # Flag
                [{}],  # Position
            ],
            subplot_titles=(
                f"{branch_name} - Price with Flag-Based Slope Signals",
                "Volume",
                f"Slope Indicator ({slope_window}d)",
                "Flag Status",
                "Position Status"
            )
        )

        # ======================================================
        # 1) PRICE PANEL WITH SLOPE SIGNALS
        # ======================================================

        # Base price line (light gray)
        fig.add_trace(
            go.Scatter(
                x=df_view.index,
                y=df_view["Close"],
                mode="lines",
                line=dict(color=colors["price_base"], width=2),
                name="Price Base",
                hoverinfo='skip',
                showlegend=False
            ),
            row=1,
            col=1,
        )
        
        # Enhanced Moving Averages
        if len(df_view) > 20:
            ma_20 = df_view['Close'].rolling(20).mean()
            fig.add_trace(
                go.Scatter(
                    x=df_view.index,
                    y=ma_20,
                    mode='lines',
                    name='MA(20)',
                    line=dict(color=colors["ma_20"], width=2.5),
                    hoverinfo='skip'
                ),
                row=1, col=1
            )
        
        if len(df_view) > 50:
            ma_50 = df_view['Close'].rolling(50).mean()
            fig.add_trace(
                go.Scatter(
                    x=df_view.index,
                    y=ma_50,
                    mode='lines',
                    name='MA(50)',
                    line=dict(color=colors["ma_50"], width=2),
                    hoverinfo='skip'
                ),
                row=1, col=1
            )

        # -------------------------
        # SLOPE-BASED COLORED SEGMENTS ON PRICE
        # -------------------------
        slope_series = df_view["Slope"]

        # Create slope-only entry/exit logic for visualization
        slope_entry_idx = (slope_series > pos_threshold) & (slope_series.shift(1) <= pos_threshold)
        slope_exit_idx = (slope_series < neg_threshold) & (slope_series.shift(1) >= neg_threshold)

        slope_entry_dates = slope_series.index[slope_entry_idx].tolist()
        slope_exit_dates = slope_series.index[slope_exit_idx].tolist()

        # Match entries with exits for slope visualization
        slope_trades = []
        i = j = 0

        while i < len(slope_entry_dates) and j < len(slope_exit_dates):
            entry_date = slope_entry_dates[i]

            # Find first exit after this entry
            while j < len(slope_exit_dates) and slope_exit_dates[j] <= entry_date:
                j += 1

            if j < len(slope_exit_dates):
                exit_date = slope_exit_dates[j]

                entry_price = df_view.loc[entry_date, "Close"]
                exit_price = df_view.loc[exit_date, "Close"]

                slope_return = ((exit_price - entry_price) / entry_price) * 100.0

                slope_trades.append({
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "return": slope_return,
                })

                j += 1
            i += 1

        # Draw colored segments on price chart for each slope period
        for trade in slope_trades:
            mask = (df_view.index >= trade["entry_date"]) & (df_view.index <= trade["exit_date"])
            segment_data = df_view[mask]

            if len(segment_data) > 1:
                segment_color = (
                    colors["slope_segment_green"]
                    if trade["return"] > 0
                    else colors["slope_segment_gray"]  # Changed from red to gray
                )

                fig.add_trace(
                    go.Scatter(
                        x=segment_data.index,
                        y=segment_data["Close"],
                        mode="lines",
                        line=dict(color=segment_color, width=4),
                        name="Slope Period",
                        hoverinfo='skip',
                        showlegend=False,
                    ),
                    row=1,
                    col=1,
                )

        # -------------------------
        # RSI+SLOPE TRADE MARKERS (actual trades from signals_df)
        # -------------------------
        if not signals_df.empty:
            # Entry markers
            fig.add_trace(
                go.Scatter(
                    x=signals_df['Entry_Date'],
                    y=signals_df['Entry_Price'],
                    mode="markers+text",
                    marker=dict(
                        color=colors["rsi_entry"],
                        size=14,
                        symbol="triangle-up",
                        line=dict(color="white", width=2),
                    ),
                    text=["ENTRY"] * len(signals_df),
                    textposition="top center",
                    name="Trade Entry",
                    hoverinfo='skip',
                ),
                row=1,
                col=1,
            )

            # Exit markers with return percentages
            exit_colors = [
                colors["rsi_exit_green"] if r > 0 else colors["rsi_exit_gray"]  # Changed from red to gray
                for r in signals_df['Return_Pct']
            ]
            exit_texts = [f"{r:+.1f}%" for r in signals_df['Return_Pct']]

            fig.add_trace(
                go.Scatter(
                    x=signals_df['Exit_Date'],
                    y=signals_df['Exit_Price'],
                    mode="markers+text",
                    marker=dict(
                        color=exit_colors,
                        size=14,
                        symbol="triangle-down",
                        line=dict(color="white", width=2),
                    ),
                    text=exit_texts,
                    textposition="bottom center",
                    name="Trade Exit",
                    hoverinfo='skip',
                ),
                row=1,
                col=1,
            )

        # -------------------------
        # RSI ACTIVATION MARKERS (Purple triangles when RSI condition first activates)
        # -------------------------
        # Find RSI activation points (when Active goes from 0 to 1)
        rsi_activation_points = df_view[(df_view['Active'] == 1) & (df_view['Active'].shift(1) == 0)]
        
        if not rsi_activation_points.empty:
            fig.add_trace(
                go.Scatter(
                    x=rsi_activation_points.index,
                    y=rsi_activation_points['Close'],
                    mode="markers+text",
                    marker=dict(
                        color=colors["rsi_activation"],
                        size=10,
                        symbol="triangle-up",
                        line=dict(color="white", width=1.5),
                    ),
                    text=["RSI"] * len(rsi_activation_points),
                    textposition="top center",
                    name="RSI Activation",
                    hoverinfo='skip',
                ),
                row=1,
                col=1,
            )

        # -------------------------
        # FLAG ACTIVATION MARKERS (Purple diamonds when Flag turns to 1)
        # -------------------------
        # Find Flag activation points (when Flag goes from 0 to 1)
        flag_activation_points = df_view[(df_view['Flag'] == 1) & (df_view['Flag'].shift(1) == 0)]
        
        if not flag_activation_points.empty:
            fig.add_trace(
                go.Scatter(
                    x=flag_activation_points.index,
                    y=flag_activation_points['Close'],
                    mode="markers+text",
                    marker=dict(
                        color=colors["rsi_activation"],
                        size=12,
                        symbol="diamond",
                        line=dict(color="white", width=1.5),
                    ),
                    text=["FLAG"] * len(flag_activation_points),
                    textposition="middle right",
                    name="Flag Activation",
                    hoverinfo='skip',
                ),
                row=1,
                col=1,
            )

        # ======================================================
        # 2) VOLUME PANEL
        # ======================================================
        fig.add_trace(
            go.Bar(
                x=df_view.index,
                y=df_view["Volume"],
                name="Volume",
                marker_color=colors["volume"],
                opacity=0.7,
                hoverinfo='skip',
            ),
            row=2,
            col=1
        )

        # ======================================================
        # 3) SLOPE PANEL
        # ======================================================
        fig.add_trace(
            go.Scatter(
                x=df_view.index,
                y=df_view["Slope"],
                mode="lines",
                line=dict(color=colors["slope"], width=3),
                fill="tozeroy",
                fillcolor="rgba(59, 130, 246, 0.2)",
                name=f"Slope ({slope_window}d)",
                hoverinfo='skip',
            ),
            row=3,
            col=1,
        )

        # Entry threshold line
        fig.add_trace(
            go.Scatter(
                x=df_view.index,
                y=[pos_threshold] * len(df_view),
                mode="lines",
                line=dict(color=colors["slope_entry"], width=2, dash="dash"),
                name=f"Entry Threshold ({pos_threshold}%)",
                showlegend=False,
                hoverinfo='skip'
            ),
            row=3,
            col=1,
        )

        # Exit threshold line
        fig.add_trace(
            go.Scatter(
                x=df_view.index,
                y=[neg_threshold] * len(df_view),
                mode="lines",
                line=dict(color=colors["slope_exit"], width=2, dash="dash"),
                name=f"Exit Threshold ({neg_threshold}%)",
                showlegend=False,
                hoverinfo='skip'
            ),
            row=3,
            col=1,
        )

        # ======================================================
        # 4) FLAG PANEL
        # ======================================================
        fig.add_trace(
            go.Scatter(
                x=df_view.index,
                y=df_view["Flag"],
                mode="lines",
                line=dict(color="#8b5cf6", width=4, shape="hv"),
                fill="tozeroy",
                fillcolor="rgba(139, 92, 246, 0.3)",
                name="Flag Status",
                hoverinfo='skip',
            ),
            row=4,
            col=1,
        )

        # ======================================================
        # 5) POSITION PANEL
        # ======================================================
        fig.add_trace(
            go.Scatter(
                x=df_view.index,
                y=df_view["InTrade"],
                mode="lines",
                line=dict(color=colors["position"], width=4, shape="hv"),
                fill="tozeroy",
                fillcolor="rgba(5, 150, 105, 0.3)",
                name="Position Status",
                hoverinfo='skip',
            ),
            row=5,
            col=1,
        )

        # ======================================================
        # ENHANCED LAYOUT
        # ======================================================
        x_min, x_max = df_view.index.min(), df_view.index.max()

        fig.update_layout(
            template="plotly_white",
            hovermode=False,  # Disable hover information
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(family="Inter", size=11),
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="rgba(0, 0, 0, 0.2)",
                borderwidth=1
            ),
            height=1100,
            margin=dict(l=80, r=60, t=80, b=80),
            font=dict(family="Inter", size=12),
            title=dict(
                text=f"<b>{branch_name}</b> - Comprehensive Slope Trading Analysis",
                font=dict(size=16, color="#1f2937"),
                x=0.5
            ),
            # Add crosshair lines for all panels (both X and Y spikes)
            xaxis=dict(
                showspikes=True, 
                spikemode='across', 
                spikesnap='cursor', 
                spikedash='solid', 
                spikecolor='rgba(0,0,0,0.4)', 
                spikethickness=1
            ),
            yaxis=dict(
                showspikes=True, 
                spikemode='across', 
                spikesnap='cursor', 
                spikedash='solid', 
                spikecolor='rgba(0,0,0,0.4)', 
                spikethickness=1
            ),
            xaxis2=dict(
                showspikes=True, 
                spikemode='across', 
                spikesnap='cursor', 
                spikedash='solid', 
                spikecolor='rgba(0,0,0,0.4)', 
                spikethickness=1
            ),
            yaxis2=dict(
                showspikes=True, 
                spikemode='across', 
                spikesnap='cursor', 
                spikedash='solid', 
                spikecolor='rgba(0,0,0,0.4)', 
                spikethickness=1
            ),
            xaxis3=dict(
                showspikes=True, 
                spikemode='across', 
                spikesnap='cursor', 
                spikedash='solid', 
                spikecolor='rgba(0,0,0,0.4)', 
                spikethickness=1
            ),
            yaxis3=dict(
                showspikes=True, 
                spikemode='across', 
                spikesnap='cursor', 
                spikedash='solid', 
                spikecolor='rgba(0,0,0,0.4)', 
                spikethickness=1
            ),
            xaxis4=dict(
                showspikes=True, 
                spikemode='across', 
                spikesnap='cursor', 
                spikedash='solid', 
                spikecolor='rgba(0,0,0,0.4)', 
                spikethickness=1
            ),
            yaxis4=dict(
                showspikes=True, 
                spikemode='across', 
                spikesnap='cursor', 
                spikedash='solid', 
                spikecolor='rgba(0,0,0,0.4)', 
                spikethickness=1
            ),
            xaxis5=dict(
                showspikes=True, 
                spikemode='across', 
                spikesnap='cursor', 
                spikedash='solid', 
                spikecolor='rgba(0,0,0,0.4)', 
                spikethickness=1
            ),
            yaxis5=dict(
                showspikes=True, 
                spikemode='across', 
                spikesnap='cursor', 
                spikedash='solid', 
                spikecolor='rgba(0,0,0,0.4)', 
                spikethickness=1
            )
        )

        # Apply crosshairs globally to ALL axes (this ensures both horizontal and vertical lines)
        fig.update_xaxes(showspikes=True, spikemode='across', spikesnap='cursor', spikedash='solid', spikecolor='rgba(0,0,0,0.6)', spikethickness=1)
        fig.update_yaxes(showspikes=True, spikemode='across', spikesnap='cursor', spikedash='solid', spikecolor='rgba(0,0,0,0.6)', spikethickness=1)

        # Enhanced X-axis with range selector and slider
        fig.update_xaxes(
            range=[x_min, x_max],
            rangeselector=dict(
                buttons=[
                    dict(count=7, label="7D", step="day", stepmode="backward"),
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=3, label="3M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(count=2, label="2Y", step="year", stepmode="backward"),
                    dict(step="all", label="All"),
                ],
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="rgba(0, 0, 0, 0.2)",
                borderwidth=1
            ),
            rangeslider=dict(
                visible=True,
                thickness=0.05,
                bgcolor="rgba(248, 249, 250, 0.8)"
            ),
            type="date",
            row=5,
            col=1,
        )

        # Y-axis formatting with crosshair spikes
        fig.update_yaxes(
            title_text="Price (USD)", 
            tickformat="$,.2f", 
            row=1, 
            col=1,
            gridcolor="rgba(0, 0, 0, 0.1)",
            title_font=dict(size=12, color="#374151"),
            showspikes=True,
            spikecolor="rgba(0, 0, 0, 0.6)",
            spikethickness=1,
            spikedash="solid"
        )
        fig.update_yaxes(
            title_text="Volume", 
            row=2, 
            col=1, 
            showgrid=False,
            title_font=dict(size=12, color="#374151"),
            showspikes=True,
            spikecolor="rgba(0, 0, 0, 0.6)",
            spikethickness=1,
            spikedash="solid"
        )
        fig.update_yaxes(
            title_text="Slope (%)", 
            row=3, 
            col=1,
            gridcolor="rgba(0, 0, 0, 0.1)",
            title_font=dict(size=12, color="#374151"),
            showspikes=True,
            spikecolor="rgba(0, 0, 0, 0.6)",
            spikethickness=1,
            spikedash="solid"
        )
        fig.update_yaxes(
            title_text="Flag", 
            row=4, 
            col=1, 
            range=[-0.1, 1.1],
            tickvals=[0, 1],
            ticktext=["Off", "On"],
            gridcolor="rgba(0, 0, 0, 0.1)",
            title_font=dict(size=12, color="#374151"),
            showspikes=True,
            spikecolor="rgba(0, 0, 0, 0.6)",
            spikethickness=1,
            spikedash="solid"
        )
        fig.update_yaxes(
            title_text="Position", 
            row=5, 
            col=1, 
            range=[-0.1, 1.1],
            tickvals=[0, 1],
            ticktext=["Out", "In"],
            gridcolor="rgba(0, 0, 0, 0.1)",
            title_font=dict(size=12, color="#374151"),
            showspikes=True,
            spikecolor="rgba(0, 0, 0, 0.6)",
            spikethickness=1,
            spikedash="solid"
        )

        return fig

# Initialize the analyzer
@st.cache_resource
def init_analyzer():
    return SlopeTradingAnalyzer()

def display_metrics_cards(metrics, col1, col2, col3):
    """Display metrics in styled cards"""
    with col1:
        return_class = "positive-return" if metrics.get('Total_Return_Pct', 0) > 0 else "negative-return"
        st.markdown(f"""
        <div class="metric-card {return_class}">
            <h3>Total Return</h3>
            <h2>{metrics.get('Total_Return_Pct', 0):.2f}%</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card neutral-metric">
            <h3>Number of Trades</h3>
            <h2>{int(metrics.get('Num_Trades', 0))}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        win_class = "positive-return" if metrics.get('Win_Rate_Pct', 0) > 50 else "negative-return"
        st.markdown(f"""
        <div class="metric-card {win_class}">
            <h3>Win Rate</h3>
            <h2>{metrics.get('Win_Rate_Pct', 0):.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card neutral-metric">
            <h3>Avg Days Held</h3>
            <h2>{metrics.get('Avg_Days_Held', 0):.1f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        dd_class = "positive-return" if metrics.get('Max_Drawdown_Pct', 0) > -10 else "negative-return"
        st.markdown(f"""
        <div class="metric-card {dd_class}">
            <h3>Max Drawdown</h3>
            <h2>{metrics.get('Max_Drawdown_Pct', 0):.2f}%</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card neutral-metric">
            <h3>Time in Market</h3>
            <h2>{metrics.get('Time_In_Market_Pct', 0):.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)

def main():
    st.title("Slope Trading Analyzer")
    st.markdown("### Advanced RSI + Slope Filter Backtesting System")
    
    analyzer = init_analyzer()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Slope parameters
        st.subheader("Slope Parameters")
        slope_window = st.slider("Slope Window (days)", 5, 30, 15)
        pos_threshold = st.slider("Positive Threshold (%)", 0.0, 20.0, 5.0, 0.5)
        neg_threshold = st.slider("Negative Threshold (%)", -10.0, 10.0, 0.0, 0.5)
        
        st.divider()
        
        # Branch selection
        st.subheader("Branch Selection")
        available_branches = analyzer.load_available_branches()
        
        if not available_branches:
            st.error("No trading branches found in ./trade_logs/")
            st.stop()
        
        # Preset branches from user's example
        default_branches = [
            "15d_RSI_A_LT33",
            "5d_RSI_AAON_LT26", 
            "7d_RSI_AAT_LT29",
            "21d_RSI_ABT_LT38",
            "17d_RSI_ACAD_LT31",
            "5d_RSI_ACGL_LT22",
            "10d_RSI_ACLS_LT25",
            "8d_RSI_ADBE_LT30",
            "14d_RSI_ADEA_LT37_and_200d_RSI_ADEA_LT50",
            "5d_RSI_ADI_LT34"
        ]
        
        # Filter available branches to match defaults
        available_defaults = [b for b in default_branches if b in available_branches]
        
        selected_branches = st.multiselect(
            "Select Branches to Analyze",
            available_branches,
            default=available_defaults[:5] if available_defaults else available_branches[:5]
        )
        
        if not selected_branches:
            st.warning("Please select at least one branch to analyze.")
            st.stop()
        
        st.divider()
        st.markdown("### Analysis Options")
        show_yearly = st.checkbox("Show Yearly Breakdown", value=True)
        show_individual_charts = st.checkbox("Show Individual Charts", value=True)
    
    # Main analysis
    if selected_branches:
        # Tabs for different views
        tab1, tab2, tab3 = st.tabs(["Overall Results", "Individual Analysis", "Detailed Reports"])
        
        with tab1:
            st.header("Overall Performance Summary")
            
            # Process all selected branches
            all_results = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, branch in enumerate(selected_branches):
                status_text.text(f'Processing {branch}... ({i+1}/{len(selected_branches)})')
                
                # Load branch data
                branch_data = analyzer.load_branch_data(branch)
                if branch_data is None:
                    continue
                
                # Get ticker
                ticker = analyzer.extract_ticker_from_branch(branch)
                if not ticker:
                    continue
                
                # Load ticker data
                ticker_data = analyzer.load_ticker_data(ticker)
                if ticker_data is None:
                    continue
                
                # Apply slope filter
                merged_data, signals_df = analyzer.apply_slope_filter(
                    branch_data, ticker_data, slope_window, pos_threshold, neg_threshold
                )
                
                # Calculate metrics
                metrics = analyzer.calculate_performance_metrics(signals_df)
                metrics['Branch'] = branch
                metrics['Ticker'] = ticker
                all_results.append(metrics)
                
                progress_bar.progress((i + 1) / len(selected_branches))
            
            status_text.text('Analysis complete!')
            progress_bar.empty()
            status_text.empty()
            
            if all_results:
                results_df = pd.DataFrame(all_results)
                
                # Summary statistics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    avg_return = results_df['Total_Return_Pct'].mean()
                    st.metric("Average Total Return", f"{avg_return:.2f}%")
                
                with col2:
                    avg_win_rate = results_df['Win_Rate_Pct'].mean()
                    st.metric("Average Win Rate", f"{avg_win_rate:.1f}%")
                
                with col3:
                    avg_trades = results_df['Num_Trades'].mean()
                    st.metric("Average Trades", f"{avg_trades:.1f}")
                
                with col4:
                    avg_drawdown = results_df['Max_Drawdown_Pct'].mean()
                    st.metric("Average Max DD", f"{avg_drawdown:.2f}%")
                
                st.divider()
                
                # Results table
                st.subheader("Branch Performance Comparison")
                
                # Format the dataframe for display
                display_df = results_df.copy()
                display_df = display_df.round(2)
                
                # Color code the dataframe
                def color_returns(val):
                    if pd.isna(val):
                        return ''
                    color = 'green' if val > 0 else 'red'
                    return f'background-color: {color}; color: white'
                
                styled_df = display_df.style.applymap(
                    color_returns, 
                    subset=['Total_Return_Pct', 'Max_Drawdown_Pct']
                )
                
                st.dataframe(styled_df, use_container_width=True)
                
                # Performance visualization
                st.subheader("Performance Visualization")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Return vs Win Rate scatter
                    fig_scatter = px.scatter(
                        results_df, 
                        x='Win_Rate_Pct', 
                        y='Total_Return_Pct',
                        hover_data=['Branch', 'Num_Trades'],
                        title='Return vs Win Rate',
                        labels={'Win_Rate_Pct': 'Win Rate (%)', 'Total_Return_Pct': 'Total Return (%)'}
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)
                
                with col2:
                    # Return distribution
                    fig_hist = px.histogram(
                        results_df, 
                        x='Total_Return_Pct',
                        title='Return Distribution',
                        labels={'Total_Return_Pct': 'Total Return (%)'}
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
        
        with tab2:
            st.header("Individual Branch Analysis")
            
            if selected_branches:
                branch_to_analyze = st.selectbox(
                    "Select Branch for Detailed Analysis",
                    selected_branches
                )
                
                if branch_to_analyze:
                    # Load and process the selected branch
                    branch_data = analyzer.load_branch_data(branch_to_analyze)
                    ticker = analyzer.extract_ticker_from_branch(branch_to_analyze)
                    
                    if branch_data is not None and ticker:
                        ticker_data = analyzer.load_ticker_data(ticker)
                        
                        if ticker_data is not None:
                            # Apply slope filter
                            merged_data, signals_df = analyzer.apply_slope_filter(
                                branch_data, ticker_data, slope_window, pos_threshold, neg_threshold
                            )
                            
                            # Calculate metrics
                            metrics = analyzer.calculate_performance_metrics(signals_df)
                            
                            # Display metrics
                            st.subheader(f"Performance Metrics - {branch_to_analyze}")
                            col1, col2, col3 = st.columns(3)
                            display_metrics_cards(metrics, col1, col2, col3)
                            
                            # Additional metrics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Average Return per Trade", f"{metrics.get('Avg_Return_Pct', 0):.2f}%")
                            with col2:
                                st.metric("Sharpe Ratio", f"{metrics.get('Sharpe_Ratio', 0):.2f}")
                            with col3:
                                st.metric("Volatility", f"{metrics.get('Volatility_Pct', 0):.2f}%")
                            
                            st.divider()
                            
                            # Enhanced Price chart
                            if show_individual_charts:
                                st.subheader("Enhanced 4-Panel Analysis Chart")
                                chart = analyzer.create_price_chart(
                                    merged_data, signals_df, branch_to_analyze, 
                                    slope_window, pos_threshold, neg_threshold
                                )
                                st.plotly_chart(chart, use_container_width=True)
                            
                            # Trade details
                            if not signals_df.empty:
                                st.subheader("Trade Details")
                                
                                trade_display = signals_df.copy()
                                trade_display['Entry_Date'] = trade_display['Entry_Date'].dt.strftime('%Y-%m-%d')
                                trade_display['Exit_Date'] = trade_display['Exit_Date'].dt.strftime('%Y-%m-%d')
                                trade_display = trade_display.round(2)
                                
                                st.dataframe(trade_display, use_container_width=True)
        
        with tab3:
            st.header("Detailed Reports")
            
            if all_results:
                # Export functionality
                st.subheader("Export Results")
                
                if st.button("Generate CSV Report"):
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV Report",
                        data=csv,
                        file_name=f"slope_trading_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
                # Yearly breakdown if requested
                if show_yearly:
                    st.subheader("Yearly Performance Breakdown")
                    st.info("Yearly breakdown functionality can be added based on your specific requirements.")
                
                # Parameter sensitivity analysis
                st.subheader("Current Parameters Summary")
                
                param_df = pd.DataFrame({
                    'Parameter': ['Slope Window', 'Positive Threshold', 'Negative Threshold'],
                    'Value': [f"{slope_window} days", f"{pos_threshold}%", f"{neg_threshold}%"]
                })
                
                st.table(param_df)

if __name__ == "__main__":
    main()