import yfinance as yf
import pandas as pd

def fetch_stock_data(stock_symbol):
    """Fetches stock price data from Yahoo Finance."""
    stock = yf.Ticker(stock_symbol)
    hist = stock.history(period="1d")  # Get the last day's stock price
    
    if hist.empty:
        print(f"No data found for {stock_symbol}")
        return None
    
    data = {
        "Stock": stock_symbol,
        "Date": hist.index[-1].strftime("%Y-%m-%d"),
        "Open": hist["Open"].iloc[-1],
        "High": hist["High"].iloc[-1],
        "Low": hist["Low"].iloc[-1],
        "Close": hist["Close"].iloc[-1],
        "Volume": hist["Volume"].iloc[-1]
    }
    return data

# List of stocks to fetch data for
stocks = ["AAPL", "TSLA", "AMZN", "GOOGL"]

# Fetch stock data and store in a DataFrame
data_list = [fetch_stock_data(stock) for stock in stocks if fetch_stock_data(stock)]
df = pd.DataFrame(data_list)

# Save to CSV
df.to_csv("stock_prices.csv", index=False)
print("✅ Stock data saved to stock_prices.csv")
