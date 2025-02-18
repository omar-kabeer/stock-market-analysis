import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import mplfinance as mpf
import matplotlib.dates as mdates

def plot_candlestick(df, title="Candlestick Chart", figsize=(12, 8), 
                      bollinger=True, moving_averages=None, price_line=False, 
                      macd=False, rsi=False, days=None):
    """
    Plots a candlestick chart with optional Bollinger Bands, Moving Averages, 
    Price Line, MACD, and RSI indicators.

    Parameters:
    - df: DataFrame with 'Date', 'Open', 'High', 'Low', 'Close', 'Volume' columns.
    - title: Chart title.
    - figsize: Tuple defining figure size.
    - bollinger: Boolean to enable Bollinger Bands.
    - moving_averages: List of integers representing MA periods (e.g., [10, 20, 50]).
    - price_line: Boolean to enable a line plot of closing prices.
    - macd: Boolean to enable MACD indicator.
    - rsi: Boolean to enable RSI indicator.
    - days: Number of most recent days to display (if None, show all data).
    """

    # Ensure 'Date' is in datetime format and set as index
    #df['Date'] = pd.to_datetime(df['Date'], errors='coerce')  # Convert to datetime
    
    #df = df.sort_values('Date').reset_index(drop=True)
    #df.index = df['Date']
    #df = df.drop(columns=['Date'], axis=1)

    # Ensure index is sorted (some plotting functions require this)
    df = df.sort_index()

    # Limit to the most recent `days` if specified
    if days is not None:
        df = df.iloc[-days:]

    # List to store additional plots
    add_plots = []

    # Bollinger Bands
    if bollinger:
        df['Middle_BB'] = df['Close'].rolling(window=20).mean()
        df['Upper_BB'] = df['Middle_BB'] + 2 * df['Close'].rolling(window=20).std()
        df['Lower_BB'] = df['Middle_BB'] - 2 * df['Close'].rolling(window=20).std()
        
        add_plots.extend([
            mpf.make_addplot(df['Middle_BB'], color='blue', linestyle='dashed'),
            mpf.make_addplot(df['Upper_BB'], color='red', linestyle='dotted'),
            mpf.make_addplot(df['Lower_BB'], color='red', linestyle='dotted')
        ])

    # Moving Averages
    if moving_averages:
        for ma in moving_averages:
            df[f"MA_{ma}"] = df['Close'].rolling(window=ma).mean()
            add_plots.append(mpf.make_addplot(df[f"MA_{ma}"], label=f"MA {ma}", linestyle='solid'))
    
    # Price Line
    if price_line:
        add_plots.append(mpf.make_addplot(df['Close'], color='black', linestyle='solid', secondary_y=False))

    # MACD Calculation
    if macd:
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        add_plots.extend([
            mpf.make_addplot(df['MACD'], panel=1, color='blue', secondary_y=False),
            mpf.make_addplot(df['Signal'], panel=1, color='red', linestyle='dashed', secondary_y=False)
        ])

    # RSI Calculation
    if rsi:
        delta = df['Close'].diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)

        avg_gain = pd.Series(gain).rolling(window=14, min_periods=1).mean()
        avg_loss = pd.Series(loss).rolling(window=14, min_periods=1).mean()
        rs = avg_gain / (avg_loss + 1e-10)  # Avoid division by zero
        df['RSI'] = 100 - (100 / (1 + rs))

        add_plots.append(mpf.make_addplot(df['RSI'], panel=2, color='purple', secondary_y=False))

    # Define the number of panels (MACD and RSI add extra panels)
    num_panels = 1 + (1 if macd else 0) + (1 if rsi else 0)

    # Set x-axis format to show date as DD/MM/YYYY
    fig, axes = mpf.plot(df, type='candle', style='charles', title=title,
                         volume=False, show_nontrading=False, figsize=figsize, addplot=add_plots, 
                         panel_ratios=(6, 2, 2)[:num_panels], ylabel="Price", returnfig=True, xlabel="Date",mav=(10,20))
    
    # Format x-axis
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())

    # Show the plot
    mpf.show()


# Example usage:
# df = pd.read_csv('your_data.csv', index_col='Date', parse_dates=True)
# plot_candlestick(df, moving_averages=[10, 20, 50], bollinger=True, price_line=True, macd=True, rsi=True, days=100)


def plot_price_and_moving_averages(df, title="Price and Moving Averages"):
    """
    Plots closing prices and moving averages.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Close'], label='Close Price', color='blue', alpha=0.8)
    plt.plot(df.index, df['MA_20'], label='20-Day MA', linestyle='dashed', color='red', alpha=0.8)
    plt.plot(df.index, df['MA_50'], label='50-Day MA', linestyle='dashed', color='green', alpha=0.8)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

def plot_bollinger_bands(df, title="Bollinger Bands"):
    """
    Plots Bollinger Bands.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Close'], label='Close Price', color='blue', alpha=0.8)
    plt.plot(df.index, df['Bollinger_Upper'], label='Bollinger Upper', linestyle='dotted', color='purple', alpha=0.6)
    plt.plot(df.index, df['Bollinger_Lower'], label='Bollinger Lower', linestyle='dotted', color='orange', alpha=0.6)
    plt.fill_between(df.index, df['Bollinger_Upper'], df['Bollinger_Lower'], color='gray', alpha=0.1)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

def plot_macd(df, title="MACD"):
    """
    Plots MACD and its signal line.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['MACD'], label='MACD', color='blue', alpha=0.8)
    plt.plot(df.index, df['MACD_Signal'], label='MACD Signal', linestyle='dashed', color='red', alpha=0.8)
    plt.bar(df.index, df['MACD_Hist'], label='MACD Histogram', color='gray', alpha=0.5)
    plt.xlabel("Date")
    plt.ylabel("MACD")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

def plot_rsi(df, title="RSI"):
    """
    Plots Relative Strength Index (RSI).
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['RSI'], label='RSI', color='blue', alpha=0.8)
    plt.axhline(70, linestyle='dashed', color='red', alpha=0.6, label='Overbought (70)')
    plt.axhline(30, linestyle='dashed', color='green', alpha=0.6, label='Oversold (30)')
    plt.xlabel("Date")
    plt.ylabel("RSI")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

if __name__ == "__main__":
    # Load processed data
    import pandas as pd
    file_path = "data/processed_stock_data.csv"  # Update with your file path
    df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')

    # Plot candlestick chart
    plot_candlestick(df, title="Candlestick Chart")

    # Plot price and moving averages
    plot_price_and_moving_averages(df, title="Price and Moving Averages")

    # Plot Bollinger Bands
    plot_bollinger_bands(df, title="Bollinger Bands")

    # Plot MACD
    plot_macd(df, title="MACD")

    # Plot RSI
    plot_rsi(df, title="Relative Strength Index (RSI)")