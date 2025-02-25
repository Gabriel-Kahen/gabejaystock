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
from get_data import get_data  # Ensure this function is updated similarly if needed
from google.cloud import storage

# Global variables for simulation
# These file names now represent the blob names in your Cloud Storage bucket.
DATA_FILE = "data/yfinance_data.csv"
TRADE_LOG_FILE = "data/trade_log.csv"
OPEN_POSITIONS_FILE = "data/open_positions.json"
BUCKET_NAME = 'gabe-jay-stock'

os.environ["GOOGLE_CLOUD_PROJECT"] = "gabe-jay-stock"

portfolio = {
    "current_positions": [],  # list of (ticker, buy_time, buy_price)
    "trade_log": []           # list of trade records
}

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

############################################
# Helper functions for Cloud Storage I/O
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
# Updated file I/O functions using Cloud Storage
############################################

def load_initial_data():
    """
    Loads the CSV file (stored in Cloud Storage) and converts it to long format.
    Expected CSV structure remains the same.
    """
    csv_text = download_blob_as_string(DATA_FILE)
    # Read header info (first 3 rows)
    header_info = pd.read_csv(io.StringIO(csv_text), nrows=3, header=None)
    n_tickers = 100
    metrics = ["Close", "High", "Low", "Open", "Volume"]

    # Extract tickers from row 1 (columns 1 to 1+n_tickers)
    tickers = header_info.iloc[1, 1:1+n_tickers].tolist()

    # Construct column names: first column is "Datetime", then one for each metric_ticker combination
    column_names = ["Datetime"]
    for metric in metrics:
        for t in tickers:
            column_names.append(f"{metric}_{t}")

    # Load the rest of the CSV using the constructed column names
    df = pd.read_csv(io.StringIO(csv_text), skiprows=3, header=None, names=column_names, parse_dates=["Datetime"])

    # Convert from wide to long format: each row becomes one (Datetime, Ticker) with columns for the 5 metrics.
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

def save_open_positions(positions):
    """Save current open positions to a JSON blob in Cloud Storage."""
    data = json.dumps(positions, default=str)
    upload_blob_from_string(OPEN_POSITIONS_FILE, data, content_type='application/json')
    print(f"Open positions saved to {OPEN_POSITIONS_FILE} in bucket {BUCKET_NAME}")

def load_open_positions():
    """Load open positions from a JSON blob in Cloud Storage, or return an empty list if it doesn't exist."""
    try:
        data = download_blob_as_string(OPEN_POSITIONS_FILE)
        return json.loads(data)
    except Exception as e:
        # If file doesn't exist or any error occurs, return an empty list.
        return []

def write_trade_log():
    """
    Writes the trade log from the portfolio to the trade_log.csv blob.
    This function downloads the current trade log if it exists, appends new records,
    and then uploads the updated CSV.
    """
    header = ["Stock", "Entry Time", "Exit Time", "Profit/Loss"]

    # Try to download the existing trade log; if not found, start fresh.
    try:
        existing_csv = download_blob_as_string(TRADE_LOG_FILE)
        file_lines = existing_csv.splitlines()
        file_empty = len(file_lines) == 0
        # Convert existing CSV lines into a list (if needed later)
    except Exception:
        file_empty = True

    output = io.StringIO()
    writer = csv.writer(output)
    # Write header if file is empty
    if file_empty:
        writer.writerow(header)

    # Write each trade record from portfolio.trade_log
    for trade in portfolio["trade_log"]:
        stock = trade.get("Ticker")
        entry_time = trade.get("BuyTime")
        exit_time = trade.get("SellTime")
        profit_loss = trade.get("Profit")
        # Format times if they are datetime objects or ISO strings
        if entry_time and isinstance(entry_time, dt):
            entry_time = entry_time.strftime("%Y-%m-%d %H:%M:%S")
        if exit_time and isinstance(exit_time, dt):
            exit_time = exit_time.strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([stock, entry_time, exit_time, profit_loss])
    
    upload_blob_from_string(TRADE_LOG_FILE, output.getvalue(), content_type='text/csv')
    print(f"Trade log updated in {TRADE_LOG_FILE} in bucket {BUCKET_NAME}")

############################################
# Existing functions (mostly unchanged)
############################################

def get_current_price(ticker):
    """
    Retrieves the latest close price for a given ticker using yfinance.
    """
    df = yf.download(ticker, period='1d', interval='15m')
    return df["Close"].iloc[-1]

def train_model(df_hist):
    """
    Trains a RandomForestRegressor model on all available historical data.
    Computes features such as Return, FutureReturn, SMA_3, and Momentum.
    """
    df_hist['Return'] = df_hist.groupby('Ticker')['Close'].pct_change()
    df_hist['FutureReturn'] = df_hist.groupby('Ticker')['Return'].shift(-1)
    df_hist['SMA_3'] = df_hist.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=3).mean())
    df_hist['Momentum'] = df_hist.groupby('Ticker')['Close'].transform(lambda x: x.diff(periods=3))
    df_hist.dropna(inplace=True)

    features = ['Close', 'Return', 'SMA_3', 'Momentum']
    target = 'FutureReturn'
    X = df_hist[features]
    y = df_hist[target]

    split_index = int(0.8 * len(df_hist))
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Model trained on all data. Test MSE: {mse:.6f}")
    return model

def select_trades(model, df_hist, num_trades=3):
    """
    Uses the trained model to predict the next period's return for each ticker at the latest datetime,
    and then selects the top `num_trades` tickers.
    """
    latest_time = df_hist["Datetime"].max()
    latest_data = df_hist[df_hist["Datetime"] == latest_time].copy()
    features = ['Close', 'Return', 'SMA_3', 'Momentum']
    latest_data['PredictedReturn'] = model.predict(latest_data[features])
    top_trades = latest_data.nlargest(num_trades, 'PredictedReturn')
    print("Selected trades for this cycle:")
    print(top_trades[['Ticker', 'Close', 'PredictedReturn']])
    return top_trades

def sell_positions(current_time):
    """
    Simulates selling all currently held positions.
    For each held position, fetches the latest price and logs the profit.
    """
    global portfolio
    for position in portfolio["current_positions"]:
        ticker, buy_time, buy_price = position
        current_price = get_current_price(ticker)
        profit = current_price - buy_price
        portfolio["trade_log"].append({
            "Ticker": ticker,
            "BuyTime": buy_time,
            "SellTime": current_time,
            "BuyPrice": buy_price,
            "SellPrice": current_price,
            "Profit": profit
        })
        print(f"Sold {ticker}: Bought at {buy_price}, Sold at {current_price}, Profit: {profit}")
    portfolio["current_positions"] = []
    write_trade_log()  # Write trade log to Cloud Storage after selling

def is_market_open():
    """
    U.S. stock market is typically open Monday-Friday, 9:30 AM to 4:00 PM Eastern Time.
    """
    now = dt.now(pytz.timezone('US/Eastern'))
    if now.weekday() >= 5:  # Saturday or Sunday
        return False
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return market_open <= now <= market_close

def simulate_cycle(tickers):
    current_time = dt.now()
    print(f"\n--- Cycle at {current_time} ---")
    
    # Step 1: Load previously open positions and sell them
    previous_positions = load_open_positions()
    if previous_positions:
        for position in previous_positions:
            ticker = position["Ticker"]
            buy_price = position["BuyPrice"]
            buy_time = dt.fromisoformat(position["BuyTime"])
            current_price = get_current_price(ticker)
            profit = current_price - buy_price
            trade = {
                "Ticker": ticker,
                "BuyTime": buy_time,
                "SellTime": current_time,
                "BuyPrice": buy_price,
                "SellPrice": current_price,
                "Profit": profit
            }
            portfolio["trade_log"].append(trade)
            print(f"Sold {ticker}: Bought at {buy_price}, Sold at {current_price}, Profit: {profit}")
        # After selling, clear open positions in Cloud Storage
        save_open_positions([])
        write_trade_log()  # Update your trade log in Cloud Storage

    # Step 2: Refresh data and train model
    get_data()  
    print("Data refreshed using get_data().")
    df_hist = load_all_data()
    model = train_model(df_hist)
    
    # Step 3: Select trades for this cycle
    top_trades = select_trades(model, df_hist, num_trades=3)
    
    # Step 4: "Buy" the selected stocks and store these as open positions
    new_positions = []
    for _, row in top_trades.iterrows():
        ticker = row['Ticker']
        buy_price = row['Close']
        new_position = {
            "Ticker": ticker,
            "BuyTime": current_time.isoformat(),
            "BuyPrice": buy_price
        }
        new_positions.append(new_position)
        print(f"Bought {ticker} at {buy_price}")
    
    # Save the new open positions to Cloud Storage for the next cycle
    save_open_positions(new_positions)

def run_simulation():
    tickers = TICKERS  # Using the static ticker list defined above
    if is_market_open():
        simulate_cycle(tickers)
    else:
        print("Market ain't open")

def run_simulation_handler(request):
    run_simulation()
    return 'TEST', 200

if __name__ == "__main__":
    run_simulation()    

    #add perms for cloud provider
