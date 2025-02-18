# Stock Market Analytics Dashboard

## Overview

A comprehensive stock market analysis platform built with Streamlit, offering real-time technical analysis, sentiment analysis, pattern recognition, and backtesting capabilities.

## ⚠️ Trading Disclaimer

This application is for educational and research purposes only. The information provided should not be construed as financial advice. Trading stocks carries significant risks, and past performance is not indicative of future results. Always conduct your own research and consider consulting with a financial advisor before making investment decisions.

## Features

- Real-time stock data visualization and analysis
- Technical indicators (MA, RSI, MACD, Bollinger Bands)
- Pattern recognition for trading signals
- News sentiment analysis
- Backtesting strategies
- Support for multiple asset classes (Stocks, Crypto, Forex, ETFs)

## Installation

### Option 1: Manual Installation

```shell
# Clone repository
git clone <repository-url>

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Add your NEWS_API_KEY to .env
```

### Option 2: Using Setup Scripts

Windows:

```shell
setup.bat
```

Linux/Mac:

```shell
chmod +x setup.sh
./setup.sh
```

## Usage

```shell
streamlit run app/home.py
```

## Project Structure

```text
├── app/
│   ├── home.py              # Main application
│   ├── pages/              # Streamlit pages
│   └── utils/              # Utility functions
├── data/                   # Data storage
├── notebooks/             # Analysis notebooks
├── requirements.txt       # Dependencies
└── .env                   # Environment variables
```

## Configuration

Required environment variables:

- `NEWS_API_KEY`: For sentiment analysis (get from [NewsAPI](https://newsapi.org/))

## Dependencies

Core requirements:

- Python 3.8+
- streamlit
- yfinance
- pandas
- plotly
- ta (Technical Analysis)
- newsapi-python
- textblob

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

MIT License - See [LICENSE](LICENSE) file for details

Copyright (c) 2024 - Present

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for full details.
You are free to:

- Commercial use
- Modify
- Distribute
- Private use
