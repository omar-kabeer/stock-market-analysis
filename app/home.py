import streamlit as st

# Page config must be the first Streamlit command
st.set_page_config(
    page_title="Stock Analysis Dashboard",
    page_icon="📈",
    layout="wide"
)
import pandas as pd
import yfinance as yf

from utils.trading_strategies import (
    momentum_strategy, box_breakout_strategy, canslim_strategy,
    value_investing_strategy, mean_reversion_strategy,
    position_sizing, calculate_performance
)



# Custom CSS
st.markdown("""
    <style>
    .big-font {
        font-size:3rem !important;
        font-weight: 600;
    }
    .medium-font {
        font-size:1.5rem !important;
    }
    .card {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from newsapi import NewsApiClient
from textblob import TextBlob
import ta
from sklearn.preprocessing import StandardScaler
from scipy import stats

import ta
import os
import sys
import time
import random
from newsapi import NewsApiClient
from textblob import TextBlob
from datetime import datetime, timedelta
import os
import dotenv

dotenv.load_dotenv()
api_key = os.getenv('NEWS_API_KEY')
# Get the absolute path of the app directory (parent of pages)
app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add the app directory to Python path
if app_dir not in sys.path:
    sys.path.append(app_dir)

from utils.pattern_recognition import PatternRecognition, PatternType
from utils.backtesting import Backtester
from utils.sentiment_analysis import NewsAnalyzer
from utils.calculate_signals import calculate_signals
from utils.calculate_signals import TradingSignals
from utils.calculate_signals import render_trading_signals_tab

# Dictionary of popular stocks
STOCK_LIST = {
        'Technology': {
            'Apple Inc.': 'AAPL',
            'Microsoft': 'MSFT',
            'Alphabet (Google) Class A': 'GOOGL',
            'Amazon': 'AMZN',
            'NVIDIA': 'NVDA',
            'Meta Platforms': 'META',
            'Tesla': 'TSLA',
            'Intel': 'INTC',
            'Advanced Micro Devices': 'AMD',
            'Cisco Systems': 'CSCO'
        },
        'Finance': {
            'JPMorgan Chase': 'JPM',
            'Bank of America': 'BAC',
            'Goldman Sachs': 'GS',
            'Visa Inc.': 'V',
            'Mastercard': 'MA',
            'Morgan Stanley': 'MS',
            'Wells Fargo': 'WFC',
            'Citigroup': 'C',
            'American Express': 'AXP',
            'BlackRock': 'BLK'
        },
        'Healthcare': {
            'Johnson & Johnson': 'JNJ',
            'UnitedHealth': 'UNH',
            'Pfizer': 'PFE',
            'Abbott Labs': 'ABT',
            'Merck & Co.': 'MRK',
            'Eli Lilly': 'LLY',
            'Moderna': 'MRNA',
            'Regeneron': 'REGN',
            'Gilead Sciences': 'GILD',
            'Bristol-Myers Squibb': 'BMY'
        },
        'Energy': {
            'ExxonMobil': 'XOM',
            'Chevron': 'CVX',
            'ConocoPhillips': 'COP',
            'Schlumberger': 'SLB',
            'Halliburton': 'HAL'
        },
        'Consumer Discretionary': {
            'Nike': 'NKE',
            'Starbucks': 'SBUX',
            "McDonald's": 'MCD',
            'Walmart': 'WMT',
            'The Home Depot': 'HD'
        },
        'Industrials': {
            'General Electric': 'GE',
            'Boeing': 'BA',
            'Lockheed Martin': 'LMT',
            'Honeywell': 'HON',
            'Caterpillar': 'CAT'
        },
        'Forex': {
            'EUR/USD': 'EURUSD=X',
            'GBP/USD': 'GBPUSD=X',
            'USD/JPY': 'USDJPY=X',
            'USD/CHF': 'USDCHF=X',
            'AUD/USD': 'AUDUSD=X',
            'USD/CAD': 'USDCAD=X',
            'NZD/USD': 'NZDUSD=X',
            'USD/CNY': 'USDCNY=X',
            'USD/INR': 'USDINR=X',
            'USD/BRL': 'USDBRL=X'
        },
        'Cryptocurrencies': {
            'Bitcoin': 'BTC-USD',
            'Ethereum': 'ETH-USD',
            'Binance Coin': 'BNB-USD',
            'Solana': 'SOL-USD',
            'Ripple (XRP)': 'XRP-USD',
            'Cardano': 'ADA-USD',
            'Dogecoin': 'DOGE-USD',
            'Polkadot': 'DOT-USD',
            'Shiba Inu': 'SHIB-USD',
            'Litecoin': 'LTC-USD'
        },
        'Precious Metals': {
            'Gold': 'GC=F',
            'Silver': 'SI=F'
        },
        'Commodities': {
            'Crude Oil (WTI)': 'CL=F',
            'Brent Oil': 'BZ=F',
            'Natural Gas': 'NG=F',
            'Corn': 'ZC=F',
            'Soybeans': 'ZS=F'
        },
        'ETFs': {
            'S&P 500 ETF (SPY)': 'SPY',
            'Nasdaq 100 ETF (QQQ)': 'QQQ',
            'Russell 2000 ETF (IWM)': 'IWM',
            'Gold ETF (GLD)': 'GLD',
            'Real Estate ETF (VNQ)': 'VNQ'
        },
        'Bonds': {
            'US 10-Year Treasury': '^TNX',
            'US 30-Year Treasury': '^TYX',
            'US 2-Year Treasury': '^IRX'
        },
        'Halal': {
            'SP Funds S&P 500 Sharia Industry Exclusions ETF (SPUS)': 'SPUS',
            'SP Funds S&P Global REIT Sharia ETF (SPRE)': 'SPRE',
            'Wahed FTSE USA Shariah ETF (HLAL)': 'HLAL',
            'Wahed Dow Jones Islamic World ETF (UMMA)': 'UMMA',
            'iShares MSCI Islamic US ETF (ISUS)': 'ISUS',
            'iShares MSCI Islamic World ETF (ISWD)': 'ISWD',
            'iShares MSCI Emerging Markets Islamic ETF (ISDE)': 'ISDE',
            'SP Funds Dow Jones Global Sukuk ETF (SPSK)': 'SPSK',
            'Franklin FTSE Saudi Arabia ETF (FLSA)': 'FLSA',
            'Franklin FTSE UAE ETF (FLAU)': 'FLAU',
            'Amana Growth Fund (AMAGX)': 'AMAGX',
            'Amana Income Fund (AMANX)': 'AMANX',
            'Franklin FTSE All World Islamic ETF (FLIW)': 'FLIW',
            'iShares USD Sukuk ETF (SUKU)': 'SUKU',
            'Franklin Global Sukuk Fund (FGSF)': 'FGSF',
            'SP Funds Dow Jones Sukuk ETF (SPSK)': 'SPSK',
            'Emirates Islamic Sukuk Fund (EISF)': 'EISF',
            'iShares USD Sukuk ETF (SUKU)': 'SUKU',        
        }
    }

# Custom CSS
st.markdown("""
    <style>
    .big-font {
        font-size:3rem !important;
        font-weight: 600;
    }
    .medium-font {
        font-size:1.5rem !important;
    }
    .card {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .about-section {
        padding: 2rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
    [data-testid="stSidebarNav"] {display: none;}
    </style>
""", unsafe_allow_html=True)


# Sidebar Configuration
st.sidebar.title('Analysis Parameters')

# Initialize session state for symbol if not already present
if 'symbol' not in st.session_state:
    st.session_state.symbol = None

# Sidebar Configuration
st.sidebar.title('Analysis Parameters')

# Stock Selection
selected_sector = st.sidebar.selectbox('Select Sector', options=list(STOCK_LIST.keys()), key='sector_select')
selected_company = st.sidebar.selectbox(
    'Select Company',
    options=list(STOCK_LIST[selected_sector].keys()),
    key='company_select'
)
symbol = STOCK_LIST[selected_sector][selected_company]

# Custom Stock Input
use_custom = st.sidebar.checkbox('Enter Custom Stock Symbol')
if use_custom:
    symbol = st.sidebar.text_input('Enter Stock Symbol', '')

# Update session state with selected symbol
st.session_state.symbol = symbol

# Timeframe Selection
timeframe_option = st.sidebar.radio(
    'Select Timeframe Option',
    ['Period', 'Interval']
)

if timeframe_option == 'Period':
    period = st.sidebar.selectbox(
        'Select Period',
        ['1y', '1d', '5d', '1mo', '3mo', '6mo', '2y', '5y', 'ytd', 'max']
    )
    interval = '1d'  # Default interval for period
else:
    period = 'max'  # Default to max when using interval
    interval = st.sidebar.selectbox(
        'Select Interval',
        ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']
    )
    
    # Add interval-specific period limits
    if interval in ['1m', '2m', '5m', '15m', '30m']:
        period = st.sidebar.selectbox(
            'Select Period (Limited by Interval)',
            ['1d', '5d', '7d'],
            help='Short intervals are limited to recent data'
        )
    elif interval in ['60m', '90m', '1h']:
        period = st.sidebar.selectbox(
            'Select Period (Limited by Interval)',
            ['1d', '5d', '7d', '1mo', '3mo'],
            help='Hourly data is limited to 3 months'
        )

# Technical Indicators Parameters
st.sidebar.subheader('Technical Indicators')

# Moving Averages
ma_fast = st.sidebar.selectbox('Fast MA Period', [3, 5, 7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84, 91, 98, 100], index=1)
ma_slow = st.sidebar.selectbox('Slow MA Period', [7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84, 91, 98, 100], index=1)


# Bollinger Bands
show_bbands = st.sidebar.checkbox('Show Bollinger Bands', value=True)
bb_period = st.sidebar.slider('BB Period', 5, 50, 20)
bb_std = st.sidebar.slider('BB Std Dev', 1.0, 3.0, 2.0)

# MACD
show_macd = st.sidebar.checkbox('Show MACD', value=True)
macd_fast = st.sidebar.slider('MACD Fast', 5, 20, 12)
macd_slow = st.sidebar.slider('MACD Slow', 20, 40, 26)
macd_signal = st.sidebar.slider('MACD Signal', 5, 15, 9)

# RSI
show_rsi = st.sidebar.checkbox('Show RSI', value=True)
rsi_period = st.sidebar.slider('RSI Period', 5, 30, 14)

# Stochastic
show_stoch = st.sidebar.checkbox('Show Stochastic', value=True)
stoch_period = st.sidebar.slider('Stochastic Period', 5, 30, 14)
stoch_smooth = st.sidebar.slider('Stochastic Smooth', 1, 10, 3)

if st.session_state.symbol:
    try:
        # Load data
        stock = yf.Ticker(st.session_state.symbol)
        df = stock.history(period=period, interval=interval)
        
        if df.empty:
            st.error(f"No data available for {st.session_state.symbol} with selected timeframe parameters")
            st.stop()
            
        # Calculate ALL technical indicators upfront
        # Moving Averages
        df['MA_Fast'] = ta.trend.sma_indicator(df['Close'], window=ma_fast)
        df['MA_Slow'] = ta.trend.sma_indicator(df['Close'], window=ma_slow)
        
        # Bollinger Bands
        df['BB_middle'] = ta.volatility.bollinger_mavg(df['Close'], bb_period)
        df['BB_upper'] = ta.volatility.bollinger_hband(df['Close'], bb_period, bb_std)
        df['BB_lower'] = ta.volatility.bollinger_lband(df['Close'], bb_period, bb_std)
        
        # MACD
        df['MACD_line'] = ta.trend.macd(df['Close'], macd_fast, macd_slow)
        df['MACD_signal'] = ta.trend.macd_signal(df['Close'], macd_fast, macd_slow, macd_signal)
        df['MACD_hist'] = ta.trend.macd_diff(df['Close'], macd_fast, macd_slow, macd_signal)
        
        # RSI
        df['RSI'] = ta.momentum.rsi(df['Close'], window=rsi_period)
        
        # Stochastic
        df['%K'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'], window=stoch_period)
        df['%D'] = df['%K'].rolling(window=stoch_smooth).mean()
        
        timeframe_info = f"Period: {period}" if timeframe_option == 'Period' else f"Interval: {interval}"
        
    except Exception as e:
        st.error(f"Error loading data for {st.session_state.symbol}: {str(e)}")
        st.stop()





# Top Navigation Tabs
page = st.tabs(["Home", "Stock Analysis", "Trading Strategies", "About"])

with page[0]:
    # Header
    col1, col2 = st.columns([2,1])
    with col1:
        st.markdown('<p class="big-font">Stock Market Analytics Hub</p>', unsafe_allow_html=True)
        st.markdown('Your comprehensive platform for stock market analysis and insights')

    # Market Overview Section
    st.markdown('## Market Overview')
    market_indices = {
        'S&P 500': '^GSPC',
        'Dow Jones': '^DJI',
        'NASDAQ': '^IXIC'
    }

    # Create three columns for market indices
    cols = st.columns(len(market_indices))
    for col, (index_name, symbol) in zip(cols, market_indices.items()):
        with col:
            try:
                index_data = yf.Ticker(symbol).history(period='1d')
                current_price = index_data['Close'].iloc[-1]
                price_change = index_data['Close'].iloc[-1] - index_data['Open'].iloc[0]
                pct_change = (price_change / index_data['Open'].iloc[0]) * 100
                
                st.metric(
                    label=index_name,
                    value=f"{current_price:,.2f}",
                    delta=f"{pct_change:.2f}%"
                )
            except:
                st.error(f"Unable to fetch data for {index_name}")

    # Quick Links Section
    st.markdown('## 🔗 Quick Access')
    col1, col2 = st.columns(2)

    with col1:
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 📊 Stock Analysis")
            st.markdown("Access comprehensive stock analysis tools:")
            st.markdown("- Technical Analysis\n- Fundamental Analysis\n- Price Predictions")
            st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### ℹ️ About")
            st.markdown("Learn more about our platform:")
            st.markdown("- Platform Features\n- Data Sources\n- Contact Information")
            st.markdown('</div>', unsafe_allow_html=True)
    # Flatten the dictionary to get a list of all stock symbols
    all_stocks = [symbol for sector in STOCK_LIST.values() for symbol in sector.values()]

    # Select 4 random stocks
    featured_stocks = random.sample(all_stocks, 4)
    # Featured Stocks Section
    st.markdown('## 🌟 Featured Stocks')

    # Only show random stocks if no specific symbol is selected
    if not st.session_state.symbol:
        # Flatten the dictionary to get a list of all stock symbols
        all_stocks = [symbol for sector in STOCK_LIST.values() for symbol in sector.values()]
        # Select 4 random stocks
        featured_stocks = random.sample(all_stocks, 4)
    else:
        # Use the selected symbol and 3 random ones
        all_stocks = [symbol for sector in STOCK_LIST.values() for symbol in sector.values()]
        remaining_stocks = [s for s in all_stocks if s != st.session_state.symbol]
        featured_stocks = [st.session_state.symbol] + random.sample(remaining_stocks, 3)

    cols = st.columns(len(featured_stocks))

    for col, symbol in zip(cols, featured_stocks):
        with col:
            try:
                stock = yf.Ticker(symbol)
                stock_info = stock.history(period="1d")
                if stock_info.empty:
                    raise ValueError("No recent data available")
                
                stock_name = stock.ticker
                stock_price = round(stock_info['Close'].iloc[-1], 2)
                
                st.markdown(f"### {stock_name}")
                st.markdown(f"**💰 Price:** `${stock_price}`")
            except Exception as e:
                st.error(f"⚠️ Unable to fetch data for {symbol}")
                st.write(f"Error: {e}")

with page[3]:
    # Custom CSS
    st.markdown("""
        <style>
        .about-section {
            padding: 2rem;
            border-radius: 0.5rem;
            background-color: #f8f9fa;
            margin: 1rem 0;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title('About Stock Market Analytics Hub')

    # Platform Overview
    st.markdown('<div class="about-section">', unsafe_allow_html=True)
    st.header('🎯 Platform Overview')
    st.write("""
    Our Stock Market Analytics Hub provides comprehensive tools and insights for stock market analysis. 
    Whether you're a beginner or an experienced trader, our platform offers valuable features to support 
    your investment decisions.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    # Features
    st.markdown('<div class="about-section">', unsafe_allow_html=True)
    st.header('✨ Key Features')
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        #### Technical Analysis
        - Real-time stock price data
        - Multiple technical indicators
        - Interactive charts
        - Custom timeframe selection
        """)

    with col2:
        st.markdown("""
        #### Fundamental Analysis
        - Company financials
        - Key metrics and ratios
        - Industry comparison
        - Historical performance
        """)
    st.markdown('</div>', unsafe_allow_html=True)

    # Data Sources
    st.markdown('<div class="about-section">', unsafe_allow_html=True)
    st.header('📊 Data Sources')
    st.write("""
    Our platform utilizes data from various reliable sources:
    - Yahoo Finance API for real-time market data
    - Company financial reports
    - Market indices and economic indicators
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    # Contact Information
    st.markdown("""
        <div class="contact-section" style="padding: 20px 0;">
            <h1 style="font-size: 2rem; margin-bottom: 30px;">📬 Contact Me</h1>
        </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Connect With Me")
        st.markdown("""
            <div style="display: flex; gap: 20px; margin: 20px 0;">
                <a href="https://www.linkedin.com/in/umar-kabir-mba-9b8a6a88/" target="_blank" style="text-decoration: none;">
                    <img src="https://cdn.jsdelivr.net/npm/simple-icons@v9/icons/linkedin.svg" alt="LinkedIn" width="35" height="35" style="filter: invert(1); transition: transform 0.3s;">
                </a>
                <a href="https://github.com/omar-kabeer" target="_blank" style="text-decoration: none;">
                    <img src="https://cdn.jsdelivr.net/npm/simple-icons@v9/icons/github.svg" alt="GitHub" width="35" height="35" style="filter: invert(1); transition: transform 0.3s;">
                </a>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("### Contact Details")
        st.markdown("""
            - 📧 Email: [uksaid12@gmail.com](mailto:uksaid12@gmail.com)
            - 🕒 Available: Monday-Friday, 9 AM - 5 PM WAT
        """)
        
    
    
with page[1]:
    # Title and stock selection
    st.title("📊 Stock Analysis Dashboard")
    if st.session_state.symbol:
        try:
            # Load data
            stock = yf.Ticker(st.session_state.symbol)
            df = stock.history(period=period, interval=interval)
            
            if df.empty:
                st.error(f"No data available for {st.session_state.symbol} with selected timeframe parameters")
                st.stop()
                
            # Calculate ALL technical indicators upfront
            # Moving Averages
            df['MA_Fast'] = ta.trend.sma_indicator(df['Close'], window=ma_fast)
            df['MA_Slow'] = ta.trend.sma_indicator(df['Close'], window=ma_slow)
            
            # Bollinger Bands
            df['BB_middle'] = ta.volatility.bollinger_mavg(df['Close'], bb_period)
            df['BB_upper'] = ta.volatility.bollinger_hband(df['Close'], bb_period, bb_std)
            df['BB_lower'] = ta.volatility.bollinger_lband(df['Close'], bb_period, bb_std)
            
            # MACD
            df['MACD_line'] = ta.trend.macd(df['Close'], macd_fast, macd_slow)
            df['MACD_signal'] = ta.trend.macd_signal(df['Close'], macd_fast, macd_slow, macd_signal)
            df['MACD_hist'] = ta.trend.macd_diff(df['Close'], macd_fast, macd_slow, macd_signal)
            
            # RSI
            df['RSI'] = ta.momentum.rsi(df['Close'], window=rsi_period)
            
            # Stochastic
            df['%K'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'], window=stoch_period)
            df['%D'] = df['%K'].rolling(window=stoch_smooth).mean()
            
            timeframe_info = f"Period: {period}" if timeframe_option == 'Period' else f"Interval: {interval}"
            
            # Create tabs
            tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
                "📊 Technical Analysis",
                "🔍 Advanced Analysis",
                "🎯 Trading Signals",
                "💭 Sentiment Analysis",
                "📑 Fundamental Analysis",
                "⏮️ Backtesting Analysis",
                "📈 Pattern Recognition"
            ])

            # Technical Analysis Tab
            with tab1:
                
                    # Calculate trading signals
                    signals = calculate_signals(df)
                    
                    # Create visualization
                    fig = make_subplots(rows=6, cols=1, shared_xaxes=True,
                                    vertical_spacing=0.03,
                                    row_heights=[0.4, 0.15, 0.15, 0.1, 0.1, 0.1])

                    # Price Chart with MAs and BBands
                    fig.add_trace(go.Candlestick(
                        x=df.index, open=df['Open'], high=df['High'],
                        low=df['Low'], close=df['Close'], name='Price'
                    ), row=1, col=1)
                    
                    fig.add_trace(go.Scatter(x=df.index, y=df['MA_Fast'],
                                            name=f'MA{ma_fast}', line=dict(color='blue')), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df.index, y=df['MA_Slow'],
                                            name=f'MA{ma_slow}', line=dict(color='red')), row=1, col=1)
                    
                    if show_bbands:
                        fig.add_trace(go.Scatter(x=df.index, y=df['BB_upper'], name='BB Upper',
                                                line=dict(color='gray', dash='dash')), row=1, col=1)
                        fig.add_trace(go.Scatter(x=df.index, y=df['BB_lower'], name='BB Lower',
                                                line=dict(color='gray', dash='dash'),
                                                fill='tonexty'), row=1, col=1)
                    
                    # Volume
                    colors = ['red' if row['Open'] > row['Close'] else 'green' for index, row in df.iterrows()]
                    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume',
                                        marker_color=colors), row=2, col=1)
                    
                    # MACD
                    if show_macd:
                        fig.add_trace(go.Scatter(x=df.index, y=df['MACD_line'],
                                                name='MACD', line=dict(color='blue')), row=3, col=1)
                        fig.add_trace(go.Scatter(x=df.index, y=df['MACD_signal'],
                                                name='Signal', line=dict(color='orange')), row=3, col=1)
                        fig.add_trace(go.Bar(x=df.index, y=df['MACD_hist'],
                                            name='MACD Hist'), row=3, col=1)
                    
                    # RSI
                    if show_rsi:
                        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'],
                                                name='RSI', line=dict(color='purple')), row=4, col=1)
                        fig.add_hline(y=70, line_dash="dash", line_color="red", row=4, col=1)
                        fig.add_hline(y=30, line_dash="dash", line_color="green", row=4, col=1)
                    
                    # Stochastic
                    if show_stoch:
                        fig.add_trace(go.Scatter(x=df.index, y=df['%K'],
                                                name='%K', line=dict(color='blue')), row=5, col=1)
                        fig.add_trace(go.Scatter(x=df.index, y=df['%D'],
                                                name='%D', line=dict(color='orange')), row=5, col=1)
                        fig.add_hline(y=80, line_dash="dash", line_color="red", row=5, col=1)
                        fig.add_hline(y=20, line_dash="dash", line_color="green", row=5, col=1)
                    
                    # Signals
                    signal_colors = {'BUY': 'green', 'SELL': 'red', 'HOLD': 'gray'}
                    fig.add_trace(go.Scatter(x=df.index, y=[0]*len(df),
                                    mode='markers',
                                    marker=dict(
                                        size=10,
                                        color=[signal_colors[s] for s in signals['MA_Cross']],
                                        symbol='triangle-up'
                                    ),
                                    name='Signals'), row=6, col=1)
                    
                    # Update layout
                    fig.update_layout(
                        title=f'{selected_company} ({st.session_state.symbol}) Technical Analysis - {timeframe_info}',
                        height=1200,
                        showlegend=True,
                        xaxis_rangeslider_visible=False
                    )
                    
                    # Add timestamp of last update
                    st.markdown(f"*Last updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*")
                    
                    # Add data range information
                    st.markdown(f"**Data Range:** {df.index[0].strftime('%Y-%m-%d %H:%M:%S')} to {df.index[-1].strftime('%Y-%m-%d %H:%M:%S')}")
                    
                    # Add warning for real-time data if using intraday intervals
                    if interval in ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h']:
                        st.warning("⚠️ Using intraday data. Some indicators may be more volatile than usual.")
                    
                    # Add axis titles
                    fig.update_yaxes(title_text="Price", row=1, col=1)
                    fig.update_yaxes(title_text="Volume", row=2, col=1)
                    fig.update_yaxes(title_text="MACD", row=3, col=1)
                    fig.update_yaxes(title_text="RSI", row=4, col=1)
                    fig.update_yaxes(title_text="Stoch", row=5, col=1)
                    fig.update_yaxes(title_text="Signals", row=6, col=1)
                    
                    st.plotly_chart(fig, use_container_width=True)



            # Advanced Analysis Tab
            with tab2:
                st.subheader("Advanced Analysis")
                
                # Volatility Analysis
                df['Returns'] = df['Close'].pct_change()
                volatility = df['Returns'].std() * np.sqrt(252)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Annual Volatility", f"{volatility:.2%}")
                    
                with col2:
                    sharpe_ratio = (df['Returns'].mean() * 252) / volatility
                    st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
                
                # Return Distribution
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=df['Returns'].dropna(),
                    nbinsx=50,
                    name='Returns Distribution'
                ))
                fig.update_layout(title='Returns Distribution')
                st.plotly_chart(fig, use_container_width=True)
                
                # Trend Analysis
                trend_strength = abs(df['MA_Fast'].iloc[-1] - df['MA_Slow'].iloc[-1]) / df['Close'].iloc[-1] * 100
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Trend Analysis")
                    st.write(f"Trend Strength: {trend_strength:.2f}%")
                    st.write("Current Trend:", "Bullish" if df['MA_Fast'].iloc[-1] > df['MA_Slow'].iloc[-1] else "Bearish")
                
                with col2:
                    st.markdown("### Volatility Analysis")
                    if show_bbands:
                        bb_width = (df['BB_upper'].iloc[-1] - df['BB_lower'].iloc[-1]) / df['BB_middle'].iloc[-1] * 100
                        st.write(f"Bollinger Band Width: {bb_width:.2f}%")
                        st.write("Volatility:", "High" if bb_width > 5 else "Low" if bb_width < 2 else "Medium")

            # Trading Signals Tab
            with tab3:
                # Generate trading signals
                df['MA_Signal'] = 0
                df.loc[df['MA_Fast'] > df['MA_Slow'], 'MA_Signal'] = 1
                df.loc[df['MA_Fast'] < df['MA_Slow'], 'MA_Signal'] = -1
                
                df['RSI_Signal'] = 0
                df.loc[df['RSI'] > 70, 'RSI_Signal'] = -1
                df.loc[df['RSI'] < 30, 'RSI_Signal'] = 1    
                
                df['Stochastic_Signal'] = 0
                df.loc[df['%K'] > df['%D'], 'Stochastic_Signal'] = 1
                df.loc[df['%K'] < df['%D'], 'Stochastic_Signal'] = -1
                
                df['MACD_Signal'] = 0
                df.loc[df['MACD_hist'] > 0, 'MACD_Signal'] = 1
                df.loc[df['MACD_hist'] < 0, 'MACD_Signal'] = -1
                
                render_trading_signals_tab(df)
                
            # Sentiment Analysis Tab
            with tab4:
                st.write(st.session_state.symbol)
                st.subheader("Sentiment Analysis")
                
                
                news_analyzer = NewsAnalyzer()
                articles = news_analyzer.fetch_news(st.session_state.symbol, selected_company)
                sentiment_results = news_analyzer.get_sentiment_summary(articles)

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### Overall Sentiment")
                    sentiment_score = sentiment_results['overall_sentiment']
                    st.metric(
                        "Sentiment Score",
                        f"{sentiment_score:.2f}",
                        delta="Positive" if sentiment_score > 0 else "Negative" if sentiment_score < 0 else "Neutral"
                    )

                with col2:
                    st.markdown("### News Distribution")
                    st.write(f"✅ Positive News: {sentiment_results['positive_news']}")
                    st.write(f"❌ Negative News: {sentiment_results['negative_news']}")
                    st.write(f"⚪ Neutral News: {sentiment_results['neutral_news']}")

                # Plot sentiment trends
                sentiment_data = sentiment_results['sentiment_data']
                if not sentiment_data.empty:
                    fig_sentiment = go.Figure()
                    fig_sentiment.add_trace(go.Scatter(
                        x=sentiment_data['date'],
                        y=sentiment_data['sentiment'],
                        name='News Sentiment',
                        mode='lines+markers',
                        line=dict(color='royalblue', width=2)
                    ))
                    fig_sentiment.update_layout(
                        title='📊 News Sentiment Trend',
                        xaxis_title="Date",
                        yaxis_title="Sentiment Score",
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig_sentiment, use_container_width=True)

                    # Display recent news with sentiment
                    st.subheader("Recent News Headlines")
                    for _, row in sentiment_data.sort_values('date', ascending=False).head(10).iterrows():
                        sentiment = row['sentiment']
                        emoji = "✅" if sentiment > 0.1 else "❌" if sentiment < -0.1 else "⚪"
                        st.write(f"{emoji} **{row['headline']}**")
                        st.write(f"Sentiment: {sentiment:.2f} | Date: {row['date'].strftime('%Y-%m-%d %H:%M')}")
                        st.markdown("---")

            # Fundamental Analysis Tab
            with tab5:
                st.subheader("Fundamental Analysis")
                
                st.write(st.session_state.symbol)
                company_info = yf.Ticker(st.session_state.symbol).info
                st.write(f"**Company Name:** {company_info.get('longName', 'N/A')}")
                st.write(f"**Sector:** {company_info.get('sector', 'N/A')}")
                st.write(f"**Industry:** {company_info.get('industry', 'N/A')}")
                st.write(f"**Country:** {company_info.get('country', 'N/A')}")
                st.write(f"**Website:** {company_info.get('website', 'N/A')}")
                st.write(f"**Description:** {company_info.get('longBusinessSummary', 'N/A')}")
                
                
                # Display key financial metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Market Cap", f"${company_info.get('marketCap', 0):,.0f}")
                    st.metric("P/E Ratio", f"{company_info.get('trailingPE', 0):.2f}")
                    st.metric("Dividend Yield", f"{company_info.get('dividendYield', 0):.2%}")
                    
                    
                with col2:
                    st.metric("Revenue", f"${company_info.get('totalRevenue', 0):,.0f}")
                    st.metric("EPS", f"${company_info.get('trailingEps', 0):.2f}")
                    st.metric("Beta", f"{company_info.get('beta', 0):.2f}")
                    
                    
                with col3:
                    st.metric("52 Week High", f"${company_info.get('fiftyTwoWeekHigh', 0):.2f}")
                    st.metric("52 Week Low", f"${company_info.get('fiftyTwoWeekLow', 0):.2f}")
                    st.metric("52 Week Range", f"${company_info.get('fiftyDayRange', 0):.2f}")
            # Backtesting Analysis Tab
            with tab6:
                st.subheader("Backtesting Analysis")
                
                try:
                    # Calculate indicators if they don't exist
                    if 'MA_Fast' not in df.columns:
                        df['MA_Fast'] = ta.trend.sma_indicator(df['Close'], window=ma_fast)
                    if 'MA_Slow' not in df.columns:
                        df['MA_Slow'] = ta.trend.sma_indicator(df['Close'], window=ma_slow)
                    if 'RSI' not in df.columns:
                        df['RSI'] = ta.momentum.rsi(df['Close'], window=rsi_period)
                    
                    # Generate signals
                    df['MA_Signal'] = 0
                    df.loc[df['MA_Fast'] > df['MA_Slow'], 'MA_Signal'] = 1
                    df.loc[df['MA_Fast'] < df['MA_Slow'], 'MA_Signal'] = -1
                    
                    df['RSI_Signal'] = 0
                    df.loc[df['RSI'] > 70, 'RSI_Signal'] = -1
                    df.loc[df['RSI'] < 30, 'RSI_Signal'] = 1
                    
                    df['Stochastic_Signal'] = 0
                    if '%K' in df.columns and '%D' in df.columns:
                        df.loc[df['%K'] > df['%D'], 'Stochastic_Signal'] = 1
                        df.loc[df['%K'] < df['%D'], 'Stochastic_Signal'] = -1
                    
                    strategies = {
                        "MA": df['MA_Signal'].shift(1),
                        "RSI": df['RSI_Signal'].shift(1),
                        "Stochastic": df['Stochastic_Signal'].shift(1)
                    }
                    
                    results = {}
                    for name, signals in strategies.items():
                        backtester = Backtester(df)
                        results[name] = backtester.test_strategy(signals)
                    
                    # Plot strategy performance
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Market'))
                    
                    for name, _ in strategies.items():
                        returns = (1 + df[f'{name}_Signal'].shift(1) * df['Close'].pct_change()).fillna(1)
                    df[f'Cumulative_{name}_Returns'] = returns.cumprod()
                    fig.add_trace(go.Scatter(x=df.index, y=df[f'Cumulative_{name}_Returns'], name=name))
                    
                    fig.update_layout(
                        title='Strategy Performance',
                        xaxis_title="Date",
                        yaxis_title="Cumulative Returns",
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Calculate market returns
                    df['Cumulative_Market_Returns'] = (1 + df['Close'].pct_change()).fillna(1).cumprod()
                    
                    # Display metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        best_strategy = max(results, key=lambda k: results[k]['total_return'])
                        best_return = results[best_strategy]['total_return']
                        st.metric("Best Strategy Return", f"{best_strategy}: {best_return:.2f}%")
                    
                    with col2:
                        market_return = (df['Cumulative_Market_Returns'].iloc[-1] - 1) * 100
                        st.metric("Market Return", f"{market_return:.2f}%")
                    
                    with col3:
                        best_sharpe = max(results, key=lambda k: results[k]['sharpe_ratio'])
                        st.metric("Best Sharpe Ratio", f"{best_sharpe}: {results[best_sharpe]['sharpe_ratio']:.2f}")
                        
                except Exception as e:
                    st.error(f"Error in backtesting analysis: {str(e)}")
                    st.info("Please ensure you have selected a valid stock symbol and timeframe.")
                    
            # Pattern Recognition Tab
            with tab7:
                st.subheader("Pattern Recognition")
                
                # Calculate indicators
                df['MA_Fast'] = ta.trend.sma_indicator(df['Close'], window=20)
                df['MA_Slow'] = ta.trend.sma_indicator(df['Close'], window=50)
                df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
                
                # Initialize pattern recognition
                pattern_recognition = PatternRecognition(df)
                patterns = pattern_recognition.find_all_patterns()
                
                # Plot price and moving averages
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Price'))
                fig.add_trace(go.Scatter(x=df.index, y=df['MA_Fast'], name='MA Fast'))
                fig.add_trace(go.Scatter(x=df.index, y=df['MA_Slow'], name='MA Slow'))
                fig.update_layout(title='Price and Moving Averages')
                st.plotly_chart(fig, use_container_width=True)
                
                # RSI Chart
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI'))
                fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
                fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
                fig_rsi.update_layout(title='Relative Strength Index (RSI)')
                st.plotly_chart(fig_rsi, use_container_width=True)
                
                # Plot detected patterns
                fig_patterns = pattern_recognition.plot_patterns()
                st.plotly_chart(fig_patterns, use_container_width=True)
                
                # Display detected patterns
                for pattern_type, detected_patterns in patterns.items():
                    if detected_patterns:
                        st.write(f"### {pattern_type.value.replace('_', ' ').title()} Patterns")
                        pattern_data = [{
                            'Start Date': pattern.start_idx,
                            'End Date': pattern.end_idx,
                            'Confidence': pattern.confidence,
                            'Support Price': pattern.support_price,
                            'Resistance Price': pattern.resistance_price,
                            'Target Price': pattern.target_price
                        } for pattern in detected_patterns]
                        st.dataframe(pd.DataFrame(pattern_data))
                
        except Exception as e:
            st.error(f"Error loading data for {st.session_state.symbol}: {str(e)}")
    else:
        st.warning("Please select a stock symbol to view analysis.")



with page[2]:
    st.subheader("Trading Strategies")
    st.write("Explore recommended trading strategies based on asset class.")
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📱 Technology & Growth Stocks",
        "💰 Financial & Value Stocks",
        "💊 Healthcare & Defensive Stocks",
        "⚡ Energy & Commodities",
        "🌐 Forex Trading",
        "🕌 Halal Investing",
        "₿ Cryptocurrencies"
    ])
    
    with tab1:
        st.markdown("## 📱 Technology & Growth Stocks Trading Strategies")
        
        # Strategy Selection
        strategy_type = st.selectbox(
            "Select Strategy",
            ["Momentum Strategy", "Box Breakout Strategy", "CANSLIM Strategy"]
        )
        
        # Strategy Parameters Section
        st.subheader("Strategy Parameters")
        col1, col2 = st.columns(2)
        
        with col1:
            # Common parameters
            risk_per_trade = st.slider("Risk Per Trade (%)", 0.5, 5.0, 2.0, 0.1)
            account_size = st.number_input("Account Size ($)", 1000, 1000000, 100000, 1000)
            
        with col2:
            # Strategy-specific parameters
            if strategy_type == "Momentum Strategy":
                lookback = st.slider("Lookback Period", 1, 100, 20)
                momentum_threshold = st.slider("Momentum Threshold (%)", 0.0, 10.0, 2.0, 0.1)
            elif strategy_type == "Box Breakout Strategy":
                box_period = st.slider("Box Period", 3, 20, 5)
                volume_threshold = st.slider("Volume Threshold", 1.0, 3.0, 1.5, 0.1)
            elif strategy_type == "CANSLIM Strategy":
                volume_threshold = st.slider("Volume Threshold", 0.1, 2.0, 1.5, 0.1)
                eps_growth = st.slider("EPS Growth Threshold (%)", 1.0, 25.0, 10.0, 0.5)
                breakout_atr = st.slider("Breakout ATR Multiple", 0.1, 3.0, 1.5, 0.1)

        # Strategy Implementation
        if st.session_state.symbol and st.button("Run Strategy Analysis"):
            try:
                with st.spinner("Analyzing strategy performance..."):
                    # Fetch data with longer timeframe
                    stock = yf.Ticker(st.session_state.symbol)
                    df = stock.history(period=period)  # Changed from 1y to 2y
                    
                    if df.empty:
                        st.error("No data available for the selected symbol")
                        st.stop()
                    
                    if len(df) < 100:  # Ensure enough data for analysis
                        st.error("Not enough historical data for analysis")
                        st.stop()
                    
                    # Initialize strategy results
                    if strategy_type == "Momentum Strategy":
                        df = momentum_strategy(df, lookback, momentum_threshold/100)
                    elif strategy_type == "Box Breakout Strategy":
                        df = box_breakout_strategy(df, box_period, volume_threshold)
                    elif strategy_type == "CANSLIM Strategy":
                        df = canslim_strategy(df, volume_threshold, eps_growth/100, breakout_atr)
                    
                    # Ensure signal column exists before position sizing
                    if 'signal' not in df.columns:
                        st.error("Strategy did not generate any signals")
                        st.stop()
                        
                    # Apply position sizing and risk management
                    df = position_sizing(df, risk_per_trade/100, account_size)
                    
                    # Calculate performance metrics
                    performance = calculate_performance(df)
                    
                    # Display results
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Return", f"{performance['Total Return']:.2%}")
                    with col2:
                        st.metric("Sharpe Ratio", f"{performance['Sharpe Ratio']:.2f}")
                    with col3:
                        st.metric("Max Drawdown", f"{performance['Max Drawdown']:.2%}")
                    
                    # Plot strategy performance
                    fig = go.Figure()
                    
                    # Price and signals
                    fig.add_trace(go.Candlestick(
                        x=df.index,
                        open=df['Open'],
                        high=df['High'],
                        low=df['Low'],
                        close=df['Close'],
                        name='Price'
                    ))
                    
                    # Add buy signals
                    buy_signals = df[df['signal'] == 1]
                    fig.add_trace(go.Scatter(
                        x=buy_signals.index,
                        y=buy_signals['Close'],
                        mode='markers',
                        marker=dict(symbol='triangle-up', size=15, color='green'),
                        name='Buy Signal'
                    ))
                    
                    # Add sell signals
                    sell_signals = df[df['signal'] == -1]
                    fig.add_trace(go.Scatter(
                        x=sell_signals.index,
                        y=sell_signals['Close'],
                        mode='markers',
                        marker=dict(symbol='triangle-down', size=15, color='red'),
                        name='Sell Signal'
                    ))
                    
                    # Update layout
                    fig.update_layout(
                        title=f'{strategy_type} Performance - {st.session_state.symbol}',
                        yaxis_title='Price',
                        xaxis_title='Date',
                        height=600,
                        template='plotly_white'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display position details
                    st.subheader("Recent Trading Signals")
                    signals_df = df[df['signal'] != 0].tail(10)[['Close', 'signal', 'PositionSize', 'StopLoss']]
                    signals_df['Type'] = signals_df['signal'].map({1: 'Buy', -1: 'Sell'})
                    signals_df['Date'] = signals_df.index
                    st.dataframe(signals_df[['Date', 'Type', 'Close', 'PositionSize', 'StopLoss']])
                    
                    # Risk Analysis
                    st.subheader("Risk Analysis")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("Position Sizing Stats:")
                        st.write(f"Average Position Size: {df['PositionSize'].mean():.0f} shares")
                        st.write(f"Max Position Size: {df['PositionSize'].max():.0f} shares")
                    
                    with col2:
                        st.write("Stop Loss Analysis:")
                        avg_stop_distance = ((df['Close'] - df['StopLoss']).abs() / df['Close']).mean()
                        st.write(f"Average Stop Loss Distance: {avg_stop_distance:.2%}")
                    
                    # Update the Trading Journal section
                    st.subheader("Trading Journal")
                    journal_df = df[df['signal'] != 0].copy()
                    journal_df['Entry Price'] = journal_df['Close']
                    journal_df['Exit Price'] = journal_df['Close'].shift(-1)
                    journal_df['P&L'] = (journal_df['Exit Price'] - journal_df['Entry Price']) * journal_df['signal']
                    journal_df['Return'] = journal_df['P&L'] / journal_df['Entry Price']  # Calculate percentage return

                    # Remove rows with NaN exit prices (last trade)
                    journal_df = journal_df.dropna(subset=['Exit Price'])

                    # Format the journal dataframe for display
                    display_df = journal_df[['Entry Price', 'Exit Price', 'PositionSize', 'StopLoss', 'P&L', 'Return']].head(10)
                    display_df['P&L'] = display_df['P&L'].round(2)
                    display_df['Return'] = display_df['Return'].map('{:.2%}'.format)
                    st.dataframe(display_df)

                    # Update Strategy Statistics calculation
                    st.subheader("Strategy Statistics")
                    # Calculate statistics with safety checks
                    completed_trades = journal_df[journal_df['P&L'].notna()]
                    total_trades = len(completed_trades)

                    if total_trades > 0:
                        winning_trades = len(completed_trades[completed_trades['P&L'] > 0])
                        losing_trades = len(completed_trades[completed_trades['P&L'] < 0])
                        win_rate = winning_trades / total_trades if total_trades > 0 else 0
                        
                        # Calculate average win/loss with safety checks
                        avg_win = completed_trades[completed_trades['P&L'] > 0]['Return'].mean() if winning_trades > 0 else 0
                        avg_loss = completed_trades[completed_trades['P&L'] < 0]['Return'].mean() if losing_trades > 0 else 0
                        
                        # Display metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Win Rate", f"{win_rate:.2%}")
                        with col2:
                            st.metric("Average Win", f"{avg_win:.2%}" if winning_trades > 0 else "No winning trades")
                        with col3:
                            st.metric("Average Loss", f"{avg_loss:.2%}" if losing_trades > 0 else "No losing trades")
                        
                        # Add trade count information
                        st.write(f"Total Completed Trades: {total_trades}")
                        st.write(f"Winning Trades: {winning_trades}")
                        st.write(f"Losing Trades: {losing_trades}")
                        
                        # Add profitability metrics
                        total_profit = completed_trades[completed_trades['P&L'] > 0]['P&L'].sum()
                        total_loss = completed_trades[completed_trades['P&L'] < 0]['P&L'].sum()
                        net_profit = total_profit + total_loss
                        
                        st.write("---")
                        st.write(f"Total Profit: ${total_profit:.2f}")
                        st.write(f"Total Loss: ${total_loss:.2f}")
                        st.write(f"Net Profit: ${net_profit:.2f}")
                    else:
                        st.write("No completed trades to analyze")
                    
            except Exception as e:
                st.error(f"Error in strategy analysis: {str(e)}")
                st.info("Please ensure you have selected a valid stock symbol and timeframe.")
        
        # Strategy Documentation
        with st.expander("Strategy Documentation"):
            if strategy_type == "Momentum Strategy":
                st.markdown("""
                ### Momentum Strategy
                This strategy combines multiple momentum indicators:
                - Price momentum over specified lookback period
                - Moving average crossovers
                - RSI for overbought/oversold conditions
                - MACD for trend confirmation
                
                #### Parameters:
                - Lookback Period: Window for momentum calculation
                - Momentum Threshold: Minimum momentum required for signal
                - Risk Per Trade: Maximum risk per position
                """)
            elif strategy_type == "Box Breakout Strategy":
                st.markdown("""
                ### Box Breakout Strategy
                This strategy identifies price breakouts from consolidation:
                - Defines price range over specified period
                - Confirms breakouts with volume
                - Uses ATR for stop loss placement
                
                #### Parameters:
                - Box Period: Number of periods to define the box
                - Volume Threshold: Required volume for breakout confirmation
                - Risk Per Trade: Maximum risk per position
                """)
            elif strategy_type == "CANSLIM Strategy":
                st.markdown("""
                ### CANSLIM Strategy
                This strategy implements the CANSLIM trading strategy:
                - C - Current quarterly earnings
                - A - Annual earnings growth
                - N - New products/management/highs
                - S - Supply and demand (volume)
                - L - Leader or laggard
                - I - Institutional sponsorship
                - M - Market direction
                """)
        
        # Risk Warning
        st.warning("""
        ⚠️ **Risk Warning**: All trading strategies involve risk of loss. 
        Past performance does not guarantee future results. 
        Always use proper risk management and consider your investment objectives.
        """)

with tab2:
    st.markdown("""
    ## 💰 Financial & Value Stocks
    ### Recommended Strategies
    1. **Value Investing**
       - Focus on P/E ratio and book value
       - Dividend yield analysis
       - Strong balance sheet metrics
    
    2. **Mean Reversion**
       - Use Bollinger Bands for overbought/oversold conditions
       - Monitor historical price ranges
       - Consider sector-wide movements
    """)

    # Strategy Selection
    strategy_type = st.selectbox(
        "Select Strategy",
        ["Value Investing", "Mean Reversion"]
    )
    
    if strategy_type == "Value Investing":
        pe_threshold = st.slider("P/E Ratio Threshold", 5, 30, 20)
        div_yield_min = st.slider("Minimum Dividend Yield (%)", 0.0, 10.0, 2.0) / 100
        risk_per_trade = st.slider("Risk Per Trade (%)", 0.1, 5.0, 1.0)
        account_size = st.number_input("Account Size ($)", 10000, 1000000, 100000, 10000)
        
        if st.button("Run Value Strategy"):
            try:
                with st.spinner("Analyzing strategy performance..."):
                    # Fetch data
                    stock = yf.Ticker(st.session_state.symbol)
                    df = stock.history(period='1y')
                    
                    if df.empty:
                        st.error("No data available for the selected symbol")
                        st.stop()
                    
                    # Run strategy
                    df = value_investing_strategy(df, pe_threshold, div_yield_min)
                    df = position_sizing(df, risk_per_trade/100, account_size)
                    
                    # Calculate performance
                    performance = calculate_performance(df)
                    
                    # Display metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Return", f"{performance['Total Return']:.2%}")
                    with col2:
                        st.metric("Sharpe Ratio", f"{performance['Sharpe Ratio']:.2f}")
                    with col3:
                        st.metric("Max Drawdown", f"{performance['Max Drawdown']:.2%}")
                    
                    # Strategy Performance Chart
                    st.subheader("Strategy Performance")
                    cumulative_returns = (1 + df['Strategy_Returns']).cumprod()
                    st.line_chart(cumulative_returns)
                    
                    # Recent Trading Signals
                    st.subheader("Recent Trading Signals")
                    signals = df[df['signal'] != 0].tail(5)
                    st.dataframe(signals[['Close', 'signal', 'position', 'entry_price']])
                    
                    # Risk Analysis
                    st.subheader("Risk Analysis")
                    st.write("Position Sizing Stats:")
                    st.write(f"Average Position Size: {df['PositionSize'].mean():.0f} shares")
                    st.write(f"Max Position Size: {df['PositionSize'].max():.0f} shares")
                    
                    # Trading Journal
                    st.subheader("Trading Journal")
                    journal_df = df[df['signal'] != 0].copy()
                    journal_df['Entry Price'] = journal_df['Close']
                    journal_df['Exit Price'] = journal_df['Close'].shift(-1)
                    journal_df['P&L'] = (journal_df['Exit Price'] - journal_df['Entry Price']) * journal_df['signal']
                    journal_df['Return'] = journal_df['P&L'] / journal_df['Entry Price']
                    st.dataframe(journal_df[['Entry Price', 'Exit Price', 'PositionSize', 'StopLoss', 'P&L', 'Return']].head(10))
                    
                    # Strategy Statistics
                    st.subheader("Strategy Statistics")
                    completed_trades = journal_df[journal_df['P&L'].notna()]
                    total_trades = len(completed_trades)
                    if total_trades > 0:
                        winning_trades = len(completed_trades[completed_trades['P&L'] > 0])
                        win_rate = winning_trades / total_trades
                        st.write(f"Win Rate: {win_rate:.2%}")
                        st.write(f"Total Trades: {total_trades}")
                        st.write(f"Winning Trades: {winning_trades}")
                        st.write(f"Losing Trades: {total_trades - winning_trades}")
            
            except Exception as e:
                st.error(f"Error in strategy analysis: {str(e)}")
                st.info("Please ensure you have selected a valid stock symbol.")
            
    elif strategy_type == "Mean Reversion":
        bb_period = st.slider("Bollinger Band Period", 10, 50, 20)
        bb_std = st.slider("Standard Deviation", 1.0, 3.0, 2.0)
        
        if st.button("Run Mean Reversion Strategy"):
            df = mean_reversion_strategy(df, bb_period, bb_std)

with tab3:
    st.markdown("""
    ## 💊 Healthcare & Defensive Stocks
    ### Recommended Strategies
    1. **Conservative Growth**
       - Focus on stable earnings growth
       - Monitor regulatory news
       - Use longer-term moving averages
    
    2. **Dividend Growth**
       - Track dividend history and growth
       - Monitor payout ratio
       - Focus on cash flow stability
    """)

with tab4:
    st.markdown("""
    ## ⚡ Energy & Commodities
    ### Recommended Strategies
    1. **Trend Following**
       - Use longer timeframes
       - Monitor global supply/demand
       - Consider seasonal patterns
    
    2. **Pairs Trading**
       - Trade correlated commodities
       - Monitor spread relationships
       - Consider geopolitical factors
    """)

with tab5:
    st.markdown("""
    ## 🌐 Forex Trading
    ### Recommended Strategies
    1. **Carry Trade**
       - Focus on interest rate differentials
       - Monitor central bank policies
       - Consider economic indicators
    
    2. **Range Trading**
       - Identify support and resistance levels
       - Use oscillators (RSI, Stochastic)
       - Monitor economic calendars
    
    3. **Trend Following**
       - Use multiple timeframe analysis
       - Monitor economic fundamentals
       - Consider political events
    """)

with tab6:
    st.markdown("""
    ## 🕌 Halal Investing
    ### Recommended Strategies
    1. **Shariah-Compliant Value Investing**
       - Screen for halal businesses
       - Avoid companies with high debt ratios
       - Focus on ethical revenue sources
    
    2. **Dividend-Focus Strategy**
       - Screen for halal dividend stocks
       - Monitor business activities
       - Avoid interest-based income
    
    3. **ETF-Based Strategy**
       - Use Shariah-compliant ETFs
       - Regular portfolio rebalancing
       - Monitor index compliance
    """)

with tab7:
    st.markdown("""
    ## ₿ Cryptocurrencies
    ### Recommended Strategies
    1. **Momentum Trading**
       - Focus on volume and price action
       - Use volatility indicators
       - Monitor market sentiment
    
    2. **HODLing (Long-term)**
       - Dollar-cost averaging
       - Focus on fundamental analysis
       - Monitor network metrics
    
    3. **Grid Trading**
       - Set up price grids in volatile markets
       - Automated buy/sell orders
       - Regular rebalancing
    """)

# Add general risk warning
st.warning("""
⚠️ **Risk Warning**: All trading strategies involve risk of loss. Past performance is not indicative of future results. 
Always conduct thorough research and consider seeking professional financial advice before implementing any trading strategy.
""")

# Add educational resources
st.markdown("""
### 📚 Additional Resources
- Research and backtest strategies using the Technical Analysis tab
- Monitor market sentiment in the Sentiment Analysis tab
- Use Pattern Recognition to identify trading opportunities
- Combine multiple strategies for diversification
""")




