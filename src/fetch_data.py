# Stock Market Performance Analysis

## Project Structure
# - data/: Contains raw and processed stock data
# - notebooks/: Jupyter notebooks for EDA and modeling
# - src/: Python scripts for data processing and analysis
# - reports/: Summary of findings and insights
# - README.md: Project overview and usage instructions

# Step 1: Data Collection & Preprocessing
import pandas as pd
import yfinance as yf

# Define stocks and date range
stocks = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA',
        'META', 'AMZN', 'TSM', 'NFLX', 'GOOG',
        'WMT', 'ORCL', 'QCOM', 'IBM', 'HPQ',
        'QCOM', 'IBM', 'HPQ', 'QCOM', 'IBM',
        'HPQ', 'QCOM', 'IBM', 'HPQ', 'QCOM',
        'IBM', 'HPQ', 'QCOM', 'IBM', 'HPQ',
        'QCOM', 'IBM', 'HPQ', 'QCOM', 'IBM',
        'HPQ', 'QCOM', 'IBM', 'HPQ', 'QCOM',
        ]  # Includes S&P 500 as benchmark
# Define date range
start_date = '2012-06-01'
end_date = (pd.Timestamp.today() - pd.Timedelta(days=1)).strftime('%Y-%m-%d')  # Set end_date to yesterday

# Fetch stock data
def fetch_stock_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end, interval='1d')['Close']
    return data

stock_data = fetch_stock_data(stocks, start_date, end_date)
stock_data.to_csv('data/stock_prices.csv')

print("Data collection complete.")
