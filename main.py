import time
import io
import csv
import json
import os
import datetime
import pytz
import pandas as pd
import yfinance as yf
from datetime import datetime as dt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from google.cloud import storage
import tempfile

# Global variables for simulation
# These file names represent the blob names in your Cloud Storage bucket.
DATA_FILE = "data/yfinance_data.csv"
TRADE_LOG_FILE = "data/trade_log.csv"
BUCKET_NAME = 'gabe-jay-stock'

# Tickers list (100 tickers expected in your CSV header row)
TICKERS = [
    "AAPL", "NVDA", "MSFT", "AMZN", "META", "GOOGL", "AVGO", "TSLA",
    "BRK-B", "GOOG", "JPM", "LLY", "V", "XOM", "COST", "MA", "UNH",
    "NFLX", "WMT", "PG", "JNJ", "HD", "ABBV", "BAC", "CRM", "KO",
    "ORCL", "CVX", "WFC", "CSCO", "IBM", "PM", "ABT", "ACN", "MRK",
    "MCD", "LIN", "GE", "ISRG", "PEP", "PLTR", "TMO", "DIS", "GS",
    "ADBE", "NOW", "T", "TXN", "QCOM", "VZ", "AMD", "SPGI", "UBER",
    "BKNG", "AXP", "CAT", "RTX", "MS", "AMGN", "INTU", "PGR", "BSX",
    "C", "PFE", "UNP", "NEE", "BLK", "AMAT", "CMCSA", "HON", "SCHW",
    "GILD", "TJX", "LOW", "DHR", "BA", "FI", "SYK", "TMUS", "COP",
    "SBUX", "ADP", "PANW", "VRTX", "DE", "ADI", "ETN", "MDT", "BX",
    "BMY", "MMC", "PLD", "LRCX", "MU", "INTC", "ANET", "KLAC", "CB",
    "SO", "ICE"
]

# Set up Google Cloud credentials if necessary
if "GCLOUD_CREDENTIALS" in os.environ and "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
    credentials_json = os.environ["GCLOUD_CREDENTIALS"]
    with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.json') as temp_file:
        temp_file.write(credentials_json)
        temp_file_path = temp_file.name
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_file_path

# Portfolio data structure (only used for logging)
portfolio = {
    "trade_log": []  # list of trade records
}

############################################
# Cloud Storage helper functions
############################################

def download_blob_as_string(blob_name):
    """Download a blob's content as a string from the bucket."""
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(blob_name)
    return blob.download_as_text()

def upload_blob_from_string(blob_name, data, content_type):
    """Upload a string to a blob in the bucket."""
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(blob_name)
    blob.upload_from_string(data, content_type=content_type)

############################################
# Data Download and CSV Format Conversion
############################################

def get_data():
    """
    Downloads historical data for all tickers over the past 5 days at a 15-minute interval
    using yfinance, then uploads the CSV to Cloud Storage.
    The CSV will have the following format:
      - Row 1: first cell "Price", then 100 cells of "Close", then 100 cells of "High",
               then 100 of "Low", then 100 of "Open", then 100 of "Volume".
      - Row 2: first cell "Ticker", then 100 tickers.
      - Row 3: first cell "datetime" (and nothing else).
      - Rows 4+: each row: first column is datetime, then the corresponding values.
    """
    print("Downloading data from yfinance...")
    data = yf.download(TICKERS, period='5d', interval='15m')
    csv_data = data.to_csv(index=True)
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(DATA_FILE)
    blob.upload_from_string(csv_data, content_type='text/csv')
    print(f"Data uploaded to gs://{BUCKET_NAME}/{DATA_FILE}")

def load_initial_data():
    """
    Loads the CSV file (stored in Cloud Storage) and converts it to long format.
    Interprets the CSV according to the provided format.
    Returns a DataFrame with columns: Datetime, Close, High, Low, Open, Volume, Ticker.
    """
    # Refresh data from Cloud Storage
    get_data()
    csv_text = download_blob_as_string(DATA_FILE)
    lines = csv_text.splitlines()
    if len(lines) < 4:
        raise ValueError("CSV does not contain enough header rows and data.")

    # Row 0: header row with "Price", then 100 "Close", 100 "High", 100 "Low", 100 "Open", 100 "Volume"
    # Row 1: tickers (first cell "Ticker", then 100 tickers)
    header1 = lines[0].split(',')
    header2 = lines[1].split(',')
    n_tickers = 100
    tickers = header2[1:1+n_tickers]

    metrics = ["Close", "High", "Low", "Open", "Volume"]
    # Construct column names: first column is "Datetime", then each metric for each ticker in order.
    column_names = ["Datetime"]
    for metric in metrics:
        for t in tickers:
            column_names.append(f"{metric}_{t}")

    # Load the remaining data (skip first 3 rows)
    df = pd.read_csv(io.StringIO("\n".join(lines[3:])),
                     header=None,
                     names=column_names,
                     parse_dates=["Datetime"])
    # Convert from wide to long format: one row per (Datetime, Ticker) with its 5 metrics.
    dfs = []
    for t in tickers:
        sub = df[["Datetime", f"Close_{t}", f"High_{t}", f"Low_{t}", f"Open_{t}", f"Volume_{t}"]].copy()
        sub["Ticker"] = t
        sub.rename(columns={
            f"Close_{t}": "Close",
            f"High_{t}": "High",
            f"Low_{t}": "Low",
            f"Open_{t}": "Open",
            f"Volume_{t}": "Volume"
        }, inplace=True)
        dfs.append(sub)
    df_long = pd.concat(dfs, ignore_index=True)
    df_long.sort_values(by=["Ticker", "Datetime"], inplace=True)
    return df_long

def load_all_data():
    """Loads all available data from the CSV in Cloud Storage."""
    return load_initial_data()

############################################
# Model Training and Trade Selection
############################################

def train_model(df_hist):
    """
    Trains a RandomForestRegressor model on the historical data.
    Computes features (Return, FutureReturn, SMA_20, Momentum, Bolinger Width) and returns the trained model.
    """
    print("Training model using historical data...")
    df_hist['Return'] = df_hist.groupby('Ticker')['Close'].pct_change()
    df_hist['FutureReturn'] = df_hist.groupby('Ticker')['Return'].shift(-1)
    df_hist['SMA_20'] = df_hist.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=20).mean())
    df_hist['Momentum'] = df_hist.groupby('Ticker')['Close'].transform(lambda x: x.diff(periods=5))
    df_hist['Bolinger Width'] = df_hist.groupby('Ticker')['Close'].transform(lambda x: (x.rolling(window=30).mean() + (2*x.rolling(window=30).std())) - (x.rolling(window=30).mean() - (2*x.rolling(window=30).std())))
    df_hist.dropna(inplace=True)
    
    features = ['Close', 'Return', 'SMA_20', 'Momentum', 'Bolinger Width']
    target = 'FutureReturn'
    X = df_hist[features]
    y = df_hist[target]
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    mse = mean_squared_error(y, model.predict(X))
    print(f"Model trained. MSE: {mse:.6f}")
    return model

def select_trades(model, df_hist, num_trades=3):
    """
    Uses the trained model to predict the next period's return for each ticker
    at the latest available datetime, and selects the top `num_trades` tickers.
    Prints the selected trades and returns a DataFrame of them.
    """
    print("Selecting trades...")
    latest_time = df_hist["Datetime"].max()
    latest_data = df_hist[df_hist["Datetime"] == latest_time].copy()
    
    # Recompute features on the latest data if necessary
    latest_data['Return'] = latest_data.groupby('Ticker')['Close'].pct_change().fillna(0)
    latest_data['SMA_20'] = latest_data.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=20).mean()).fillna(latest_data['Close'])
    latest_data['Momentum'] = latest_data.groupby('Ticker')['Close'].transform(lambda x: x.diff(periods=5)).fillna(0)
    latest_data['Bolinger Width'] = latest_data.groupby('Ticker')['Close'].transform(lambda x: (x.rolling(window=30).mean() + (2*x.rolling(window=30).std())) - (x.rolling(window=30).mean() - (2*x.rolling(window=30).std())))
    
    features = ['Close', 'Return', 'SMA_20', 'Momentum', 'Bolinger Width']
    latest_data['PredictedReturn'] = model.predict(latest_data[features])
    
    top_trades = latest_data.nlargest(num_trades, 'PredictedReturn')
    print("Selected trades for this cycle:")
    print(top_trades[['Ticker', 'Close', 'PredictedReturn']])
    return top_trades

############################################
# Real-Time Price Fetching
############################################

def get_current_price(ticker):
    """
    Retrieves the current price for the given ticker using yfinance.
    First tries fast_info/regularMarketPrice; if unavailable, falls back to historical data.
    """
    stock = yf.Ticker(ticker)
    try:
        price = stock.fast_info.last_price
        if price is not None:
            return float(price)
    except Exception as e:
        print(f"Error using fast_info for {ticker}: {e}")
    try:
        price = stock.info.get("regularMarketPrice")
        if price is not None:
            return float(price)
    except Exception as e:
        print(f"Error using regularMarketPrice for {ticker}: {e}")
    df = stock.history(period='1d', interval='1m')
    return float(df["Close"].iloc[-1])

############################################
# Trade Execution and Logging
############################################

def append_trade_log(trades):
    """
    Appends completed trades to the TRADE_LOG_FILE in Cloud Storage.
    If the file does not exist, writes a header first.
    """
    header = ["Ticker", "BuyTime", "BuyPrice", "SellTime", "SellPrice", "Profit", "ExpectedReturn"]
    try:
        existing_csv = download_blob_as_string(TRADE_LOG_FILE)
    except Exception:
        existing_csv = ""

    output = io.StringIO()
    if not existing_csv.strip():
        writer = csv.writer(output)
        writer.writerow(header)
    else:
        output.write(existing_csv)
        if not existing_csv.endswith("\n"):
            output.write("\n")
        writer = csv.writer(output)
    
    for trade in trades:
        ticker = trade.get("Ticker")
        buy_time = trade.get("BuyTime")
        sell_time = trade.get("SellTime")
        # Use .item() to extract a Python scalar if the value is a single-element array
        buy_price = trade.get("BuyPrice")
        sell_price = trade.get("SellPrice")
        profit = trade.get("Profit")
        
        writer.writerow([
            ticker,
            buy_time,
            sell_time,
            buy_price,
            sell_price,
            profit,
        ])
    
    upload_blob_from_string(TRADE_LOG_FILE, output.getvalue(), content_type='text/csv')
    print(f"Appended {len(trades)} trades to {TRADE_LOG_FILE}")

def execute_trades():
    """
    Executes one full trading cycle:
      1. Downloads new data and uploads the CSV.
      2. Loads the CSV (with the new format) and trains the model.
      3. Selects 3 trades based on the model's predictions.
      4. Immediately records buy price and time (printing details).
      5. Waits exactly (15 * 60 - 28) seconds.
      6. Records sell price and time (printing details).
      7. Appends all trade details to the trade log.
    """
    print("\n--- New Trading Cycle Initiated ---")
    # Step 1: Refresh data
    get_data()
    print("Data refreshed.")
    
    # Step 2: Load data and train model
    df_hist = load_all_data()
    model = train_model(df_hist)
    
    # Step 3: Select trades
    top_trades = select_trades(model, df_hist, num_trades=3)
    
    # Step 4: Record buy data immediately in UTC
    buy_time = dt.now(pytz.utc).strftime("%Y-%m-%d %H:%M:%S")
    buy_trades = []
    print("\n-- Buy Phase --")
    for _, row in top_trades.iterrows():
        ticker = row['Ticker']
        buy_price = get_current_price(ticker)
        buy_trades.append({
            "Ticker": ticker,
            "BuyTime": buy_time,
            "BuyPrice": buy_price,
            "ExpectedReturn": row['PredictedReturn']
        })
        print(f"Bought {ticker} at {buy_price} on {buy_time}")
    
    wait_seconds = 15 * 60 - 17
    print(f"\nWaiting {wait_seconds} seconds until Sell Phase...")
    time.sleep(wait_seconds)
    
    # Step 6: Record sell data in UTC
    sell_time = dt.now(pytz.utc).strftime("%Y-%m-%d %H:%M:%S")
    completed_trades = []
    print("\n-- Sell Phase --")
    for trade in buy_trades:
        ticker = trade["Ticker"]
        sell_price = get_current_price(ticker)
        profit = sell_price - trade["BuyPrice"]
        completed_trade = {
            "Ticker": ticker,
            "BuyTime": trade["BuyTime"],
            "BuyPrice": trade["BuyPrice"],
            "SellTime": sell_time,
            "SellPrice": sell_price,
            "Profit": profit,
            "ExpectedReturn": trade["ExpectedReturn"]
        }
        completed_trades.append(completed_trade)
        print(f"Sold {ticker} at {sell_price} on {sell_time} | Profit: {profit}")
    
    # Step 7: Append trade details to log
    append_trade_log(completed_trades)
    print("--- Trading Cycle Completed ---\n")

############################################
# Market Check and Main Loop
############################################

def is_market_open():
    """
    Checks if the U.S. stock market is open.
    Market hours: Monday-Friday, 9:30 AM to 4:00 PM Eastern Time.
    """
    now = dt.now(pytz.timezone("US/Eastern"))
    if now.weekday() >= 5:
        return False
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return market_open <= now <= market_close

def main():
    """
    Main loop: if the market is open, execute a new trading cycle.
    Prints status messages to the console.
    """
    while True:
        if is_market_open():
            print(f"\nMarket is open at {dt.now(pytz.timezone('US/Eastern')).isoformat()}")
            execute_trades()

        else:
            print(f"Market is closed at {dt.now(pytz.timezone('US/Eastern')).isoformat()}. Waiting for market to open...")
            time.sleep(15)

if __name__ == "__main__":
    main()