
import pandas as pd
import os
import numpy as np
def load_data(file_path):
    """Loads stock data from a CSV file."""
    return pd.read_csv(file_path)

def clean_data(df):
    """Handles missing values and ensures proper data types."""
    df = df.drop([0, 1], axis=0).reset_index(drop=True)  # Drop unnecessary rows and reset index
    df.rename(columns={'Price': 'Date'}, inplace=True)  # Rename 'Price' column to 'Date'
    df = df.dropna()  # Drop rows with missing values
    df = df.sort_index()  # Ensure data is sorted by index
    for col in df.columns[1:]:  # Skip the 'Date' column
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def add_features(df):
    """
    Adds relevant technical analysis features to the dataset.
    """
    # Ensure 'Date' is a datetime column and sort by date
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    # 1. Daily Returns
    df['Returns'] = df['Close'].pct_change()

    # 2. Moving Averages
    df['MA_20'] = df['Close'].rolling(window=20).mean()  # 20-day moving average
    df['MA_50'] = df['Close'].rolling(window=50).mean()  # 50-day moving average
    df['MA_200'] = df['Close'].rolling(window=200).mean()  # 200-day moving average

    # 3. Exponential Moving Averages (EMA)
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()  # 12-day EMA
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()  # 26-day EMA

    # 4. Moving Average Convergence Divergence (MACD)
    df['MACD'] = df['EMA_12'] - df['EMA_26']  # MACD line
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()  # Signal line
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']  # MACD histogram

    # 5. Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # 6. Bollinger Bands
    df['Bollinger_Mid'] = df['MA_20']  # Middle band (20-day MA)
    df['Bollinger_Std'] = df['Close'].rolling(window=20).std()  # Standard deviation
    df['Bollinger_Upper'] = df['Bollinger_Mid'] + (2 * df['Bollinger_Std'])  # Upper band
    df['Bollinger_Lower'] = df['Bollinger_Mid'] - (2 * df['Bollinger_Std'])  # Lower band

    # 7. Average True Range (ATR)
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = true_range.rolling(window=14).mean()

    # 8. On-Balance Volume (OBV)
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

    # 9. Stochastic Oscillator
    low_min = df['Low'].rolling(window=14).min()
    high_max = df['High'].rolling(window=14).max()
    df['Stochastic_%K'] = 100 * ((df['Close'] - low_min) / (high_max - low_min))
    df['Stochastic_%D'] = df['Stochastic_%K'].rolling(window=3).mean()  # Signal line

    # 10. Price Rate of Change (ROC)
    df['ROC'] = df['Close'].pct_change(periods=10)  # 10-day rate of change

    # 11. Volatility (Standard Deviation of Returns)
    df['Volatility'] = df['Returns'].rolling(window=20).std()

    # 12. Price Momentum
    df['Momentum'] = df['Close'] - df['Close'].shift(4)  # 4-day momentum

    return df

def process_ticker_file(file_path, output_folder):
    """Processes a single ticker's data and saves the result."""
    # Load data
    df = load_data(file_path)
    
    # Clean data
    df = clean_data(df)
    
    # Add features
    df = add_features(df)
    
    # Save processed data
    ticker_name = os.path.basename(file_path).replace('_prices.csv', '')  # Extract ticker name from file name
    output_file = os.path.join(output_folder, f"{ticker_name}_processed.csv")
    df.to_csv(output_file, index=False)
    print(f"Processed data saved to {output_file}.")

if __name__ == "__main__":
    # Define input and output folders
    input_folder = "data"  # Folder containing downloaded ticker files
    output_folder = "processed_data"  # Folder to save processed files
    
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Loop through all CSV files in the input folder
    for file_name in os.listdir(input_folder):
        if file_name.endswith("_prices.csv"):  # Process only ticker files
            file_path = os.path.join(input_folder, file_name)
            print(f"Processing {file_name}...")
            try:
                process_ticker_file(file_path, output_folder)
            except Exception as e:
                print(f"Failed to process {file_name}. Error: {e}")
    
    print("Data processing complete.")