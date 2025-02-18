import yfinance as yf
import pandas as pd
import numpy as np
import ta
import streamlit as st

# Move all strategy functions here
def momentum_strategy(df, lookback, threshold):
    """Calculate momentum strategy signals with improved risk management"""
    df['momentum'] = df['Close'].pct_change(lookback)
    df['SMA20'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['SMA50'] = ta.trend.sma_indicator(df['Close'], window=50)
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
    df['ATR_Pct'] = df['ATR'] / df['Close']
    
    df['signal'] = 0
    df['position'] = 0
    df['entry_price'] = 0.0
    
    for i in range(lookback, len(df)):
        if df.loc[df.index[i-1], 'position'] != 0:
            entry_price = df.loc[df.index[i-1], 'entry_price']
            current_pos = df.loc[df.index[i-1], 'position']
            pnl_pct = (df['Close'].iloc[i] - entry_price) / entry_price * current_pos
            
            if (pnl_pct >= threshold * 2 or 
                pnl_pct <= -threshold or
                (current_pos == 1 and df['RSI'].iloc[i] > 75) or
                (current_pos == -1 and df['RSI'].iloc[i] < 25) or
                (current_pos == 1 and df['Close'].iloc[i] < df['SMA20'].iloc[i]) or
                (current_pos == -1 and df['Close'].iloc[i] > df['SMA20'].iloc[i])):
                
                df.loc[df.index[i], 'signal'] = -current_pos
                df.loc[df.index[i], 'position'] = 0
                df.loc[df.index[i], 'entry_price'] = 0.0
                continue
            
            df.loc[df.index[i], 'position'] = current_pos
            df.loc[df.index[i], 'entry_price'] = entry_price
            continue
        
        if (df['momentum'].iloc[i] > threshold and 
            df['Close'].iloc[i] > df['SMA20'].iloc[i] and
            df['RSI'].iloc[i] < 70 and df['RSI'].iloc[i] > 30 and
            df['ATR_Pct'].iloc[i] < 0.05):
            
            df.loc[df.index[i], 'signal'] = 1
            df.loc[df.index[i], 'position'] = 1
            df.loc[df.index[i], 'entry_price'] = df['Close'].iloc[i]
            
        elif (df['momentum'].iloc[i] < -threshold and 
              df['Close'].iloc[i] < df['SMA20'].iloc[i] and
              df['RSI'].iloc[i] < 70 and df['RSI'].iloc[i] > 30 and
              df['ATR_Pct'].iloc[i] < 0.05):
            
            df.loc[df.index[i], 'signal'] = -1
            df.loc[df.index[i], 'position'] = -1
            df.loc[df.index[i], 'entry_price'] = df['Close'].iloc[i]
    
    return df

def box_breakout_strategy(df, box_period, volume_threshold):
    """Calculate box breakout strategy signals"""
    df['high_box'] = df['High'].rolling(box_period).max()
    df['low_box'] = df['Low'].rolling(box_period).min()
    df['avg_volume'] = df['Volume'].rolling(box_period).mean()
    
    df['signal'] = 0
    
    breakout_up = (df['Close'] > df['high_box'].shift(1)) & \
                  (df['Volume'] > df['avg_volume'] * volume_threshold) & \
                  (df['Close'] > df['Open'])
                  
    breakout_down = (df['Close'] < df['low_box'].shift(1)) & \
                    (df['Volume'] > df['avg_volume'] * volume_threshold) & \
                    (df['Close'] < df['Open'])
    
    df.loc[breakout_up, 'signal'] = 1
    df.loc[breakout_down, 'signal'] = -1
    
    df['signal'] = df['signal'] * (df['signal'].shift(1) != df['signal']).astype(int)
    
    return df

def canslim_strategy(df, volume_threshold=1.5, eps_growth=0.25, breakout_atr=2):
    """CANSLIM trading strategy implementation"""
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
    df['ATR_Pct'] = df['ATR'] / df['Close']
    df['SMA20'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['SMA50'] = ta.trend.sma_indicator(df['Close'], window=50)
    df['SMA200'] = ta.trend.sma_indicator(df['Close'], window=200)
    df['Volume_SMA20'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA20']
    df['RS_Change'] = df['Close'].pct_change(20)
    df['Upper_Channel'] = df['High'].rolling(window=20).max()
    df['Lower_Channel'] = df['Low'].rolling(window=20).min()
    
    df['signal'] = 0
    df['position'] = 0
    df['entry_price'] = 0.0
    
    df['Market_Trend'] = np.where(
        (df['SMA20'] > df['SMA50']) & (df['SMA50'] > df['SMA200']),
        1,
        np.where(
            (df['SMA20'] < df['SMA50']) & (df['SMA50'] < df['SMA200']),
            -1,
            0
        )
    )
    
    for i in range(20, len(df)):
        if df.loc[df.index[i-1], 'position'] != 0:
            entry_price = df.loc[df.index[i-1], 'entry_price']
            current_pos = df.loc[df.index[i-1], 'position']
            pnl_pct = (df['Close'].iloc[i] - entry_price) / entry_price * current_pos
            
            exit_conditions = (
                (pnl_pct <= -0.07) or
                (pnl_pct >= 0.20) or
                (current_pos == 1 and df['Close'].iloc[i] < df['SMA20'].iloc[i]) or
                (current_pos == -1 and df['Close'].iloc[i] > df['SMA20'].iloc[i]) or
                (df['Volume_Ratio'].iloc[i] < 0.5)
            )
            
            if exit_conditions:
                df.loc[df.index[i], 'signal'] = -current_pos
                df.loc[df.index[i], 'position'] = 0
                df.loc[df.index[i], 'entry_price'] = 0.0
                continue
            
            df.loc[df.index[i], 'position'] = current_pos
            df.loc[df.index[i], 'entry_price'] = entry_price
            continue
        
        long_conditions = (
            df['Market_Trend'].iloc[i] == 1 and
            df['Close'].iloc[i] > df['Upper_Channel'].iloc[i-1] and
            df['Volume_Ratio'].iloc[i] > volume_threshold and
            df['RS_Change'].iloc[i] > eps_growth and
            df['Close'].iloc[i] > df['SMA50'].iloc[i] and
            df['ATR_Pct'].iloc[i] < 0.05
        )
        
        short_conditions = (
            df['Market_Trend'].iloc[i] == -1 and
            df['Close'].iloc[i] < df['Lower_Channel'].iloc[i-1] and
            df['Volume_Ratio'].iloc[i] > volume_threshold and
            df['RS_Change'].iloc[i] < -eps_growth and
            df['Close'].iloc[i] < df['SMA50'].iloc[i] and
            df['ATR_Pct'].iloc[i] < 0.05
        )
        
        if long_conditions:
            df.loc[df.index[i], 'signal'] = 1
            df.loc[df.index[i], 'position'] = 1
            df.loc[df.index[i], 'entry_price'] = df['Close'].iloc[i]
        elif short_conditions:
            df.loc[df.index[i], 'signal'] = -1
            df.loc[df.index[i], 'position'] = -1
            df.loc[df.index[i], 'entry_price'] = df['Close'].iloc[i]
    
    return df

def value_investing_strategy(df, pe_threshold=20, div_yield_min=0.02):
    """Value investing strategy based on P/E and dividend yield"""
    df['signal'] = 0
    df['position'] = 0
    df['entry_price'] = 0.0
    
    stock = yf.Ticker(st.session_state.symbol)
    
    try:
        pe_ratio = stock.info.get('forwardPE', float('inf'))
        div_yield = stock.info.get('dividendYield', 0)
        
        # Calculate additional metrics
        df['SMA50'] = ta.trend.sma_indicator(df['Close'], window=50)
        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
        df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
        df['ATR_Pct'] = df['ATR'] / df['Close']
        
        for i in range(1, len(df)):
            if df.loc[df.index[i-1], 'position'] != 0:
                entry_price = df.loc[df.index[i-1], 'entry_price']
                current_pos = df.loc[df.index[i-1], 'position']
                pnl_pct = (df['Close'].iloc[i] - entry_price) / entry_price * current_pos
                
                # Exit conditions
                if (pnl_pct >= 0.20 or  # Take profit at 20%
                    pnl_pct <= -0.10 or  # Stop loss at 10%
                    (current_pos == 1 and df['Close'].iloc[i] < df['SMA50'].iloc[i])):  # Trend reversal
                    
                    df.loc[df.index[i], 'signal'] = -current_pos
                    df.loc[df.index[i], 'position'] = 0
                    df.loc[df.index[i], 'entry_price'] = 0.0
                    continue
                
                df.loc[df.index[i], 'position'] = current_pos
                df.loc[df.index[i], 'entry_price'] = entry_price
                continue
            
            # Entry conditions
            if (pe_ratio < pe_threshold and 
                div_yield > div_yield_min and 
                df['RSI'].iloc[i] < 60 and  # Not overbought
                df['ATR_Pct'].iloc[i] < 0.03):  # Not too volatile
                
                df.loc[df.index[i], 'signal'] = 1
                df.loc[df.index[i], 'position'] = 1
                df.loc[df.index[i], 'entry_price'] = df['Close'].iloc[i]
            
            elif pe_ratio > pe_threshold * 1.5:  # Overvalued
                df.loc[df.index[i], 'signal'] = -1
                df.loc[df.index[i], 'position'] = -1
                df.loc[df.index[i], 'entry_price'] = df['Close'].iloc[i]
    
    except:
        st.error("Unable to fetch fundamental data")
    
    return df

def mean_reversion_strategy(df, bb_period=20, bb_std=2):
    """Mean reversion strategy using Bollinger Bands"""
    # Calculate indicators
    df['BB_middle'] = ta.volatility.bollinger_mavg(df['Close'], bb_period)
    df['BB_upper'] = ta.volatility.bollinger_hband(df['Close'], bb_period, bb_std)
    df['BB_lower'] = ta.volatility.bollinger_lband(df['Close'], bb_period, bb_std)
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
    df['ATR_Pct'] = df['ATR'] / df['Close']
    
    # Initialize position tracking
    df['signal'] = 0
    df['position'] = 0
    df['entry_price'] = 0.0
    
    for i in range(1, len(df)):
        if df.loc[df.index[i-1], 'position'] != 0:
            entry_price = df.loc[df.index[i-1], 'entry_price']
            current_pos = df.loc[df.index[i-1], 'position']
            pnl_pct = (df['Close'].iloc[i] - entry_price) / entry_price * current_pos
            
            # Exit conditions
            if (pnl_pct >= 0.15 or  # Take profit at 15%
                pnl_pct <= -0.05 or  # Stop loss at 5%
                (current_pos == 1 and df['Close'].iloc[i] > df['BB_middle'].iloc[i]) or  # Return to mean
                (current_pos == -1 and df['Close'].iloc[i] < df['BB_middle'].iloc[i])):  # Return to mean
                
                df.loc[df.index[i], 'signal'] = -current_pos
                df.loc[df.index[i], 'position'] = 0
                df.loc[df.index[i], 'entry_price'] = 0.0
                continue
            
            df.loc[df.index[i], 'position'] = current_pos
            df.loc[df.index[i], 'entry_price'] = entry_price
            continue
        
        # Entry conditions with confirmation
        if (df['Close'].iloc[i] < df['BB_lower'].iloc[i] and  # Price below lower band
            df['RSI'].iloc[i] < 30 and  # Oversold
            df['ATR_Pct'].iloc[i] < 0.03):  # Not too volatile
            
            df.loc[df.index[i], 'signal'] = 1
            df.loc[df.index[i], 'position'] = 1
            df.loc[df.index[i], 'entry_price'] = df['Close'].iloc[i]
            
        elif (df['Close'].iloc[i] > df['BB_upper'].iloc[i] and  # Price above upper band
              df['RSI'].iloc[i] > 70 and  # Overbought
              df['ATR_Pct'].iloc[i] < 0.03):  # Not too volatile
            
            df.loc[df.index[i], 'signal'] = -1
            df.loc[df.index[i], 'position'] = -1
            df.loc[df.index[i], 'entry_price'] = df['Close'].iloc[i]
    
    return df

def position_sizing(df, risk_per_trade, account_size):
    """Calculate position sizes based on risk parameters"""
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
    
    atr_multiplier = 2
    df['ATR_Pct'] = df['ATR'] / df['Close']
    stop_multiplier = np.where(df['ATR_Pct'] > 0.02, 1.5, atr_multiplier)
    
    df['StopLoss'] = df['Close'] - (df['ATR'] * stop_multiplier * df['signal'])
    min_stop_distance = df['Close'] * 0.01
    
    risk_amount = account_size * risk_per_trade
    stop_distance = (df['Close'] - df['StopLoss']).abs()
    stop_distance = stop_distance.clip(lower=min_stop_distance)
    
    df['PositionSize'] = risk_amount / stop_distance
    
    max_position_pct = 0.20
    max_position = (account_size * max_position_pct) / df['Close']
    df['PositionSize'] = df['PositionSize'].clip(upper=max_position)
    
    df['Strategy_Returns'] = np.where(
        df['signal'] != 0,
        df['PositionSize'] * df['Close'].pct_change() * df['signal'],
        0
    )
    
    df['Strategy_Returns'] = df['Strategy_Returns'].fillna(0)
    
    return df

def calculate_performance(df):
    """Calculate strategy performance metrics with improved risk measures"""
    returns = df[df['Strategy_Returns'] != 0]['Strategy_Returns']
    
    if len(returns) == 0:
        return {
            'Total Return': 0,
            'Sharpe Ratio': 0,
            'Max Drawdown': 0
        }
    
    cumulative_returns = (1 + returns).cumprod()
    total_return = cumulative_returns.iloc[-1] - 1 if len(cumulative_returns) > 0 else 0
    
    risk_free_rate = 0.02
    excess_returns = returns - risk_free_rate/252
    sharpe_ratio = np.sqrt(252) * (excess_returns.mean() / excess_returns.std()) if excess_returns.std() != 0 else 0
    
    rolling_max = cumulative_returns.expanding().max()
    drawdowns = cumulative_returns / rolling_max - 1
    max_drawdown = drawdowns.min()
    
    performance = {
        'Total Return': total_return,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown
    }
    
    return performance


def canslim_strategy(df, volume_threshold=1.5, eps_growth=0.25, breakout_atr=2):
    """
    CANSLIM trading strategy implementation
    C - Current quarterly earnings
    A - Annual earnings growth
    N - New products/management/highs
    S - Supply and demand (volume)
    L - Leader or laggard
    I - Institutional sponsorship
    M - Market direction
    """
    # Calculate technical indicators
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
    df['ATR_Pct'] = df['ATR'] / df['Close']
    
    # Moving averages for trend following
    df['SMA20'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['SMA50'] = ta.trend.sma_indicator(df['Close'], window=50)
    df['SMA200'] = ta.trend.sma_indicator(df['Close'], window=200)
    
    # Volume indicators
    df['Volume_SMA20'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA20']
    
    # Relative Strength
    df['RS_Change'] = df['Close'].pct_change(20)  # 20-day price change
    
    # Price channels
    df['Upper_Channel'] = df['High'].rolling(window=20).max()
    df['Lower_Channel'] = df['Low'].rolling(window=20).min()
    
    # Initialize signals
    df['signal'] = 0
    df['position'] = 0
    df['entry_price'] = 0.0
    
    # Market trend
    df['Market_Trend'] = np.where(
        (df['SMA20'] > df['SMA50']) & (df['SMA50'] > df['SMA200']),
        1,  # Uptrend
        np.where(
            (df['SMA20'] < df['SMA50']) & (df['SMA50'] < df['SMA200']),
            -1,  # Downtrend
            0  # Sideways
        )
    )
    
    # Generate signals
    for i in range(20, len(df)):
        # Skip if already in position
        if df.loc[df.index[i-1], 'position'] != 0:
            # Check exit conditions
            entry_price = df.loc[df.index[i-1], 'entry_price']
            current_pos = df.loc[df.index[i-1], 'position']
            
            # Calculate profit/loss percentage
            pnl_pct = (df['Close'].iloc[i] - entry_price) / entry_price * current_pos
            
            # Exit conditions
            exit_conditions = (
                (pnl_pct <= -0.07) or  # Stop loss at 7%
                (pnl_pct >= 0.20) or   # Take profit at 20%
                (current_pos == 1 and df['Close'].iloc[i] < df['SMA20'].iloc[i]) or  # Trend reversal
                (current_pos == -1 and df['Close'].iloc[i] > df['SMA20'].iloc[i]) or
                (df['Volume_Ratio'].iloc[i] < 0.5)  # Volume dry up
            )
            
            if exit_conditions:
                df.loc[df.index[i], 'signal'] = -current_pos
                df.loc[df.index[i], 'position'] = 0
                df.loc[df.index[i], 'entry_price'] = 0.0
                continue
            
            # Maintain position
            df.loc[df.index[i], 'position'] = current_pos
            df.loc[df.index[i], 'entry_price'] = entry_price
            continue
        
        # Entry conditions
        # Long entry conditions (CANSLIM criteria)
        long_conditions = (
            df['Market_Trend'].iloc[i] == 1 and  # Market in uptrend
            df['Close'].iloc[i] > df['Upper_Channel'].iloc[i-1] and  # Breakout
            df['Volume_Ratio'].iloc[i] > volume_threshold and  # Strong volume
            df['RS_Change'].iloc[i] > eps_growth and  # Strong relative strength
            df['Close'].iloc[i] > df['SMA50'].iloc[i] and  # Above major MA
            df['ATR_Pct'].iloc[i] < 0.05  # Not too volatile
        )
        
        # Short entry conditions
        short_conditions = (
            df['Market_Trend'].iloc[i] == -1 and  # Market in downtrend
            df['Close'].iloc[i] < df['Lower_Channel'].iloc[i-1] and  # Breakdown
            df['Volume_Ratio'].iloc[i] > volume_threshold and  # Strong volume
            df['RS_Change'].iloc[i] < -eps_growth and  # Weak relative strength
            df['Close'].iloc[i] < df['SMA50'].iloc[i] and  # Below major MA
            df['ATR_Pct'].iloc[i] < 0.05  # Not too volatile
        )
        
        # Set signals
        if long_conditions:
            df.loc[df.index[i], 'signal'] = 1
            df.loc[df.index[i], 'position'] = 1
            df.loc[df.index[i], 'entry_price'] = df['Close'].iloc[i]
        elif short_conditions:
            df.loc[df.index[i], 'signal'] = -1
            df.loc[df.index[i], 'position'] = -1
            df.loc[df.index[i], 'entry_price'] = df['Close'].iloc[i]
    
    return df