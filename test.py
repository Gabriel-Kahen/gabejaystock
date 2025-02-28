import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import pytz

def get_stock_price(ticker, query_time):
    """
    Fetches the stock price of a given ticker at the closest available time to the query_time.

    Parameters:
        ticker (str): The stock ticker symbol (e.g., "AAPL").
        query_time (datetime): A datetime object representing the desired query time.
                               If naive, it will be assumed to be in US Eastern Time.
    
    Returns:
        str: A message with the closest available price.
    """
    # Define US Eastern timezone
    eastern = pytz.timezone("America/New_York")
    
    # If query_time is naive, localize it to Eastern Time.
    if query_time.tzinfo is None:
        query_time = eastern.localize(query_time)
    
    # Define a window around the query_time to fetch data.
    start_date = (query_time - timedelta(days=1)).strftime('%Y-%m-%d')
    end_date = (query_time + timedelta(days=1)).strftime('%Y-%m-%d')
    
    # Download historical data with an hourly interval.
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date, interval="1m")    
    if data.empty:
        return f"No data available for {ticker} around {query_time}."
    
    # Ensure the index is in datetime format (and timezone-aware)
    data.index = pd.to_datetime(data.index)
    
    # If the index is not timezone-aware, localize it; otherwise, convert to Eastern Time.
    if data.index.tz is None:
        data.index = data.index.tz_localize(eastern)
    else:
        data.index = data.index.tz_convert(eastern)
    
    # Find the index of the closest time using get_indexer.
    closest_position = data.index.get_indexer([query_time], method='nearest')[0]
    closest_time = data.index[closest_position]
    closest_price = data.loc[closest_time, 'Close']
    
    return f"Price of {ticker} at {closest_time} is ${closest_price:.2f}"

if __name__ == "__main__":
    ticker_symbol = "SCHW"
    # Specify the query time in 24-hour format; e.g., 15:00 (3:00 PM)
    query_time = datetime(2025, 2, 26, 15, 0)
    
    print(get_stock_price(ticker_symbol, query_time))