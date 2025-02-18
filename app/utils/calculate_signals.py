import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta


def calculate_signals(df):
    """Calculate trading signals based on technical indicators"""
    signals = pd.DataFrame(index=df.index)
    
    # MA Crossover Signals
    signals['MA_Cross'] = np.where(
        ((df['MA_Fast'].shift(1) < df['MA_Slow'].shift(1)) & 
         (df['MA_Fast'] > df['MA_Slow'])),
        'BUY',
        np.where(
            ((df['MA_Fast'].shift(1) > df['MA_Slow'].shift(1)) & 
             (df['MA_Fast'] < df['MA_Slow'])),
            'SELL',
            'HOLD'
        )
    )
    
    # RSI Signals
    if 'RSI' in df.columns:
        signals['RSI_Signal'] = np.where(
            df['RSI'] < 30, 'BUY',
            np.where(df['RSI'] > 70, 'SELL', 'HOLD')
        )
    
    # MACD Signals
    if all(col in df.columns for col in ['MACD_line', 'MACD_signal']):
        signals['MACD_Signal'] = np.where(
            ((df['MACD_line'] > df['MACD_signal']) & 
             (df['MACD_line'].shift(1) <= df['MACD_signal'].shift(1))),
            'BUY',
            np.where(
                ((df['MACD_line'] < df['MACD_signal']) & 
                 (df['MACD_line'].shift(1) >= df['MACD_signal'].shift(1))),
                'SELL',
                'HOLD'
            )
        )
    
    # Stochastic Signals
    if all(col in df.columns for col in ['%K', '%D']):
        signals['Stoch_Signal'] = np.where(
            ((df['%K'] < 20) & (df['%D'] < 20) & (df['%K'] > df['%D'])),
            'BUY',
            np.where(
                ((df['%K'] > 80) & (df['%D'] > 80) & (df['%K'] < df['%D'])),
                'SELL',
                'HOLD'
            )
        )
    
    return signals

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from typing import Dict, Tuple

class TradingSignals:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.generate_signals()
    
    def generate_signals(self) -> None:
        """Generate all trading signals at once."""
        self._generate_sma_signals()
        self._generate_rsi_signals()
        self._generate_stochastic_signals()
    
    def _generate_sma_signals(self) -> None:
        """Generate SMA crossover signals."""
        self.df['MA_Signal'] = 0
        self.df.loc[self.df['MA_Fast'] > self.df['MA_Slow'], 'MA_Signal'] = 1
        self.df.loc[self.df['MA_Fast'] < self.df['MA_Slow'], 'MA_Signal'] = -1
    
    def _generate_rsi_signals(self) -> None:
        """Generate RSI signals."""
        self.df['RSI_Signal'] = 0
        self.df.loc[self.df['RSI'] > 70, 'RSI_Signal'] = -1
        self.df.loc[self.df['RSI'] < 30, 'RSI_Signal'] = 1
    
    def _generate_stochastic_signals(self) -> None:
        """Generate Stochastic signals."""
        self.df['Stochastic_Signal'] = 0
        self.df.loc[self.df['%K'] > self.df['%D'], 'Stochastic_Signal'] = 1
        self.df.loc[self.df['%K'] < self.df['%D'], 'Stochastic_Signal'] = -1
        
    def _generate_macd_signals(self) -> None:
        """Generate MACD signals."""
        self.df['MACD_Signal'] = 0
        self.df.loc[self.df['MACD_hist'] > 0, 'MACD_Signal'] = 1
        self.df.loc[self.df['MACD_hist'] < 0, 'MACD_Signal'] = -1

    def create_price_sma_plot(self) -> go.Figure:
        """Create price and SMA crossover signals plot."""
        fig = go.Figure()
        
        # Price
        fig.add_trace(go.Scatter(
            x=self.df.index, 
            y=self.df['Close'], 
            name='Price',
            line=dict(color='black')
        ))
        
        # SMAs
        fig.add_trace(go.Scatter(
            x=self.df.index,
            y=self.df['MA_Fast'],
            name='MA Fast',
            line=dict(color='blue')
        ))
        
        fig.add_trace(go.Scatter(
            x=self.df.index,
            y=self.df['MA_Slow'],
            name='MA Slow',
            line=dict(color='red')
        ))
        
        # Buy/Sell signals
        buy_signals = self.df[self.df['MA_Signal'] == 1]
        sell_signals = self.df[self.df['MA_Signal'] == -1]
        
        fig.add_trace(go.Scatter(
            x=buy_signals.index,
            y=buy_signals['Close'],
            mode='markers',
            name='Buy Signal',
            marker=dict(symbol='triangle-up', size=10, color='green')
        ))
        
        fig.add_trace(go.Scatter(
            x=sell_signals.index,
            y=sell_signals['Close'],
            mode='markers',
            name='Sell Signal',
            marker=dict(symbol='triangle-down', size=10, color='red')
        ))
        
        fig.update_layout(
            title='Price and SMA Crossover Signals',
            height=400,
            showlegend=True
        )
        
        return fig

    def create_rsi_plot(self) -> go.Figure:
        """Create RSI signals plot."""
        fig = go.Figure()
        
        # RSI line
        fig.add_trace(go.Scatter(
            x=self.df.index,
            y=self.df['RSI'],
            name='RSI',
            line=dict(color='purple')
        ))
        
        # Overbought/Oversold lines
        fig.add_hline(y=70, line_dash="dash", line_color="red", name="Overbought")
        fig.add_hline(y=30, line_dash="dash", line_color="green", name="Oversold")
        
        # Buy/Sell signals
        buy_signals = self.df[self.df['RSI_Signal'] == 1]
        sell_signals = self.df[self.df['RSI_Signal'] == -1]
        
        fig.add_trace(go.Scatter(
            x=buy_signals.index,
            y=buy_signals['RSI'],
            mode='markers',
            name='Buy Signal',
            marker=dict(symbol='triangle-up', size=10, color='green')
        ))
        
        fig.add_trace(go.Scatter(
            x=sell_signals.index,
            y=sell_signals['RSI'],
            mode='markers',
            name='Sell Signal',
            marker=dict(symbol='triangle-down', size=10, color='red')
        ))
        
        fig.update_layout(
            title='RSI Signals',
            height=300,
            showlegend=True,
            yaxis=dict(range=[0, 100])
        )
        
        return fig

    def create_stochastic_plot(self) -> go.Figure:
        """Create Stochastic signals plot."""
        fig = go.Figure()
        
        # %K and %D lines
        fig.add_trace(go.Scatter(
            x=self.df.index,
            y=self.df['%K'],
            name='%K',
            line=dict(color='blue')
        ))
        
        fig.add_trace(go.Scatter(
            x=self.df.index,
            y=self.df['%D'],
            name='%D',
            line=dict(color='orange')
        ))
        
        # Overbought/Oversold lines
        fig.add_hline(y=80, line_dash="dash", line_color="red", name="Overbought")
        fig.add_hline(y=20, line_dash="dash", line_color="green", name="Oversold")
        
        # Crossover signals
        signals = self.df[self.df['Stochastic_Signal'] != 0]
        
        fig.add_trace(go.Scatter(
            x=signals.index,
            y=signals['%K'],
            mode='markers',
            name='Crossover Signals',
            marker=dict(
                symbol=np.where(signals['Stochastic_Signal'] == 1, 'triangle-up', 'triangle-down'),
                size=10,
                color=np.where(signals['Stochastic_Signal'] == 1, 'green', 'red')
            )
        ))
        
        fig.update_layout(
            title='Stochastic Signals',
            height=300,
            showlegend=True,
            yaxis=dict(range=[0, 100])
        )
        
        return fig

    def create_macd_plot(self) -> go.Figure:
        """Create MACD signals plot."""
        fig = go.Figure()
        
        # MACD line
        fig.add_trace(go.Scatter(
            x=self.df.index,
            y=self.df['MACD_line'],
            name='MACD',
            line=dict(color='blue')
        ))
        
        # Signal line
        fig.add_trace(go.Scatter(
            x=self.df.index,
            y=self.df['MACD_signal'],
            name='Signal',
            line=dict(color='orange')
        ))
        
        # Histogram
        colors = ['red' if val < 0 else 'green' for val in self.df['MACD_hist']]
        fig.add_trace(go.Bar(
            x=self.df.index,
            y=self.df['MACD_hist'],
            name='Histogram',
            marker_color=colors
        ))
        
        fig.update_layout(
            title='MACD',
            height=300,
            showlegend=True
        )
        
        return fig

    def get_signal_metrics(self) -> Dict[str, Tuple[str, str, str]]:
        """Get the current metrics for the trading signals dashboard."""
        last_row = self.df.iloc[-1]
        
        return {
            "MA Crossover": (
                "MA_Cross",
                str(last_row['MA_Signal']),
                "Bullish" if last_row['MA_Signal'] == 1 else "Bearish" if last_row['MA_Signal'] == -1 else "Neutral"
            ),
            "RSI": (
                "Current RSI",
                f"{last_row['RSI']:.2f}",
                "Oversold" if last_row['RSI'] < 30 else "Overbought" if last_row['RSI'] > 70 else "Neutral"
            ),
            "MACD": (
                "MACD Histogram",
                f"{last_row['MACD_hist']:.3f}",
                "Bullish" if last_row['MACD_hist'] > 0 else "Bearish"
            ),
            "Stochastic": (
                "%K",
                f"{last_row['%K']:.2f}",
                f"%D: {last_row['%D']:.2f}"
            ),
            "Trend": (
                "Trend Strength",
                f"{abs(last_row['MA_Fast'] - last_row['MA_Slow']) / last_row['Close'] * 100:.2f}%",
                None
            ),
            "Volatility": (
                "Bollinger Band Width",
                f"{((last_row['BB_upper'] - last_row['BB_lower']) / last_row['BB_middle'] * 100):.2f}%" if 'BB_upper' in self.df else "N/A",
                None
            )
        }

    def generate_consensus_signals(self) -> None:
        """Generate consensus signals where all indicators agree."""
        # Convert all signals to same scale (-1, 0, 1)
        self.df['MACD_Signal'] = np.where(self.df['MACD_hist'] > 0, 1, -1)
        
        # Calculate consensus
        signals_to_compare = ['MA_Signal', 'RSI_Signal', 'Stochastic_Signal', 'MACD_Signal']
        
        # Initialize consensus column
        self.df['Consensus'] = 0
        
        # Check for bullish consensus (all 1)
        bullish_consensus = (self.df[signals_to_compare] == 1).all(axis=1)
        self.df.loc[bullish_consensus, 'Consensus'] = 1
        
        # Check for bearish consensus (all -1)
        bearish_consensus = (self.df[signals_to_compare] == -1).all(axis=1)
        self.df.loc[bearish_consensus, 'Consensus'] = -1

    def create_consensus_plot(self) -> go.Figure:
        """Create consensus signals plot."""
        fig = go.Figure()
        
        # Price line
        fig.add_trace(go.Scatter(
            x=self.df.index,
            y=self.df['Close'],
            name='Price',
            line=dict(color='black', width=1)
        ))
        
        # Find consensus points
        bullish_consensus = self.df[self.df['Consensus'] == 1]
        bearish_consensus = self.df[self.df['Consensus'] == -1]
        
        # Add bullish consensus points
        if not bullish_consensus.empty:
            fig.add_trace(go.Scatter(
                x=bullish_consensus.index,
                y=bullish_consensus['Close'],
                mode='markers',
                name='Bullish Consensus',
                marker=dict(
                    symbol='star',
                    size=15,
                    color='green',
                    line=dict(width=2, color='darkgreen')
                )
            ))
        
        # Add bearish consensus points
        if not bearish_consensus.empty:
            fig.add_trace(go.Scatter(
                x=bearish_consensus.index,
                y=bearish_consensus['Close'],
                mode='markers',
                name='Bearish Consensus',
                marker=dict(
                    symbol='star',
                    size=15,
                    color='red',
                    line=dict(width=2, color='darkred')
                )
            ))
        
        fig.update_layout(
            title='Signal Consensus Points',
            height=300,
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig

    def get_consensus_metrics(self) -> Dict[str, str]:
        """Get metrics about the current consensus state."""
        last_row = self.df.iloc[-1]
        
        # Count how many signals are bullish/bearish
        signals_to_check = ['MA_Signal', 'RSI_Signal', 'Stochastic_Signal', 'MACD_Signal']
        bullish_count = sum(1 for signal in signals_to_check if last_row[signal] == 1)
        bearish_count = sum(1 for signal in signals_to_check if last_row[signal] == -1)
        neutral_count = len(signals_to_check) - bullish_count - bearish_count
        
        # Get the latest consensus if it exists
        last_consensus_idx = self.df[self.df['Consensus'] != 0].index[-1] if any(self.df['Consensus'] != 0) else None
        if last_consensus_idx:
            last_consensus = self.df.loc[last_consensus_idx, 'Consensus']
            days_since_consensus = (self.df.index[-1] - last_consensus_idx).days
        else:
            last_consensus = None
            days_since_consensus = None
        
        return {
            "current_signals": f"Bullish: {bullish_count}, Bearish: {bearish_count}, Neutral: {neutral_count}",
            "last_consensus": "Bullish" if last_consensus == 1 else "Bearish" if last_consensus == -1 else "None",
            "days_since_consensus": str(days_since_consensus) if days_since_consensus else "N/A"
        }

def render_trading_signals_tab(df: pd.DataFrame) -> None:
    """Main function to render the trading signals tab."""
    st.subheader("Trading Signals")
    
    # Initialize trading signals
    signals = TradingSignals(df)
    signals.generate_consensus_signals()
    
    # Display metrics
    st.subheader("Trading Signals Summary")
    metrics = signals.get_signal_metrics()
    
    cols = st.columns(len(metrics))
    for col, (title, (metric_name, value, delta)) in zip(cols, metrics.items()):
        with col:
            st.markdown(f"### {title}")
            st.metric(metric_name, value, delta=delta)
    
    # Create and display individual plots
    st.plotly_chart(signals.create_price_sma_plot(), use_container_width=True)
    st.plotly_chart(signals.create_rsi_plot(), use_container_width=True)
    st.plotly_chart(signals.create_stochastic_plot(), use_container_width=True)
    st.plotly_chart(signals.create_macd_plot(), use_container_width=True)
    
    # Display consensus analysis
    st.subheader("Signal Consensus Analysis")
    
    # Display consensus metrics
    consensus_metrics = signals.get_consensus_metrics()
    cols = st.columns(3)
    
    with cols[0]:
        st.metric("Current Signal Distribution", consensus_metrics["current_signals"])
    with cols[1]:
        st.metric("Last Consensus Signal", consensus_metrics["last_consensus"])
    with cols[2]:
        st.metric("Days Since Last Consensus", consensus_metrics["days_since_consensus"])
    
    # Display consensus plot
    st.plotly_chart(signals.create_consensus_plot(), use_container_width=True)
