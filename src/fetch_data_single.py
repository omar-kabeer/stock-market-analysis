# Stock Market Performance Analysis

## Project Structure
# - data/: Contains raw and processed stock data
# - notebooks/: Jupyter notebooks for EDA and modeling
# - src/: Python scripts for data processing and analysis
# - reports/: Summary of findings and insights
# - README.md: Project overview and usage instructions

import pandas as pd
import yfinance as yf

# Define a list of available tickers
available_tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA',
        'META', 'AMZN', 'TSM', 'NFLX', 'GOOG',
        'WMT', 'ORCL', 'QCOM', 'IBM', 'HPQ',
        'JNJ', 'VZ', 'BA', 'CAT', 'CSCO',
        'MMM', 'PFE', 'GE', 'TM', 'V',
        'IBM', 'MMM', 'WMT', 'VZ', 'BA',
        'CAT', 'CSCO', 'GE', 'TM', 'V',
        'IBM', 'MMM', 'WMT', 'VZ', 'BA',
        'CAT', 'CSCO', 'GE', 'TM', 
        ] 

# Remove duplicates from the ticker list (if any)
available_tickers = list(set(available_tickers))

# Display the list of available tickers
print("Available tickers:")
for i, ticker in enumerate(available_tickers, start=1):
    print(f"{i}. {ticker}")

# Define date range
start_date = '2012-06-01'
end_date = (pd.Timestamp.today() - pd.Timedelta(days=1)).strftime('%Y-%m-%d')  # Set end_date to yesterday

# Fetch stock data for a single ticker
def fetch_stock_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end, interval='1d')
    return data

# Loop through each ticker, fetch data, and save to a separate CSV file
for ticker in available_tickers:
    print(f"Downloading data for {ticker}...")
    try:
        # Fetch data for the current ticker
        stock_data = fetch_stock_data(ticker, start_date, end_date)
        
        # Save the data to a CSV file
        output_file = f'data/{ticker}_prices.csv'
        stock_data.to_csv(output_file)
        
        print(f"Data for {ticker} saved to {output_file}.")
    except Exception as e:
        print(f"Failed to download data for {ticker}. Error: {e}")

print("Data collection complete.")