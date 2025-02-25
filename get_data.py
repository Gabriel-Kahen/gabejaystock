import yfinance as yf
import pandas as pd
import csv

def get_data():
    tickers = [
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

    data = yf.download(tickers, period='5d', interval='15m')
    data.to_csv("yfinance_data.csv")

get_data()