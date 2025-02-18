import pandas as pd
import matplotlib.pyplot as plt

def load_data(stock_path, index_path):
    """Loads stock and benchmark index data."""
    stock_df = pd.read_csv(stock_path, parse_dates=['Date'], index_col='Date')
    index_df = pd.read_csv(index_path, parse_dates=['Date'], index_col='Date')
    return stock_df, index_df

def normalize_prices(df):
    """Normalizes prices to a common starting point (e.g., 100)."""
    return (df['Close'] / df['Close'].iloc[0]) * 100

def compare_performance(stock_df, index_df, title="Stock vs. Benchmark Performance"):
    """
    Compares stock and benchmark index performance using normalized prices.
    """
    # Normalize prices
    stock_normalized = normalize_prices(stock_df)
    index_normalized = normalize_prices(index_df)

    # Plot normalized performance
    plt.figure(figsize=(12, 6))
    plt.plot(stock_normalized.index, stock_normalized, label="Stock", color='blue')
    plt.plot(index_normalized.index, index_normalized, label="Benchmark Index", color='green')
    plt.xlabel("Date")
    plt.ylabel("Normalized Price (Base=100)")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

def plot_cumulative_returns(stock_df, index_df, title="Cumulative Returns"):
    """
    Plots cumulative returns for stock and benchmark index.
    """
    # Calculate daily returns
    stock_returns = stock_df['Close'].pct_change().dropna()
    index_returns = index_df['Close'].pct_change().dropna()

    # Calculate cumulative returns
    stock_cumulative = (1 + stock_returns).cumprod() - 1
    index_cumulative = (1 + index_returns).cumprod() - 1

    # Plot cumulative returns
    plt.figure(figsize=(12, 6))
    plt.plot(stock_cumulative.index, stock_cumulative, label="Stock", color='blue')
    plt.plot(index_cumulative.index, index_cumulative, label="Benchmark Index", color='green')
    plt.xlabel("Date")
    plt.ylabel("Cumulative Returns")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

def plot_rolling_correlation(stock_df, index_df, window=30, title="Rolling Correlation"):
    """
    Plots rolling correlation between stock and benchmark index returns.
    """
    # Calculate daily returns
    stock_returns = stock_df['Close'].pct_change().dropna()
    index_returns = index_df['Close'].pct_change().dropna()

    # Calculate rolling correlation
    rolling_corr = stock_returns.rolling(window=window).corr(index_returns)

    # Plot rolling correlation
    plt.figure(figsize=(12, 6))
    plt.plot(rolling_corr.index, rolling_corr, label=f"{window}-Day Rolling Correlation", color='purple')
    plt.xlabel("Date")
    plt.ylabel("Correlation")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

if __name__ == "__main__":
    # Load data
    stock_file = "../data/processed_stock_data.csv"  # Processed stock data
    index_file = "../data/benchmark_index.csv"  # Benchmark index data
    stock_df, index_df = load_data(stock_file, index_file)

    # Compare normalized performance
    compare_performance(stock_df, index_df, title="Stock vs. Benchmark Performance (Normalized)")

    # Plot cumulative returns
    plot_cumulative_returns(stock_df, index_df, title="Cumulative Returns: Stock vs. Benchmark Index")

    # Plot rolling correlation
    plot_rolling_correlation(stock_df, index_df, window=30, title="30-Day Rolling Correlation")