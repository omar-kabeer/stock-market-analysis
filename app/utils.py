import yfinance as yf
import pandas as pd

def load_stock_data(ticker, start_date, end_date):
    """Fetch stock data using yfinance"""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        return df
    except:
        return None

def calculate_metrics(df):
    """Calculate key financial metrics"""
    metrics = {}
    
    # Price metrics
    metrics['current_price'] = df['Close'].iloc[-1]
    metrics['price_change'] = df['Close'].iloc[-1] - df['Close'].iloc[0]
    metrics['price_change_pct'] = (metrics['price_change'] / df['Close'].iloc[0]) * 100
    
    # Volume metrics
    metrics['avg_volume'] = df['Volume'].mean()
    metrics['volume_change_pct'] = ((df['Volume'].iloc[-1] / df['Volume'].iloc[0]) - 1) * 100
    
    # Volatility metrics
    metrics['volatility'] = df['Close'].pct_change().std() * (252 ** 0.5)  # Annualized
    
    return metrics