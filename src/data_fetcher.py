import yfinance as yf
import pandas as pd
from typing import Optional

class StockDataFetcher:
    def __init__(self, symbol: str):
        self.symbol = symbol
        
    def fetch_data(self, period: str = "1y") -> Optional[pd.DataFrame]:
        """
        Fetch stock data for the specified symbol and period.
        
        Args:
            period (str): Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
            
        Returns:
            pd.DataFrame: DataFrame with stock data or None if fetch fails
        """
        try:
            stock = yf.Ticker(self.symbol)
            df = stock.history(period=period)
            return df
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None
