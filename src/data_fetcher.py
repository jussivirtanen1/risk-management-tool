import yfinance as yf
import pandas as pd
from typing import Optional, Dict
from datetime import datetime

class StockDataFetcher:
    def __init__(self, db_connector):
        self.db = db_connector
        self._fx_rates_cache = {}  # Cache for FX rates

    def _get_fx_rate(self, fx_ticker: str, date: str) -> float:
        """
        Get FX rate for a specific date. Caches results to avoid redundant API calls.
        
        Args:
            fx_ticker: Yahoo Finance FX ticker (e.g., 'EURUSD=X')
            date: Date for FX rate
            
        Returns:
            float: Exchange rate for the date
        """
        cache_key = f"{fx_ticker}_{date}"
        if cache_key in self._fx_rates_cache:
            return self._fx_rates_cache[cache_key]

        try:
            fx_data = yf.download(fx_ticker, start=date, end=date)
            if fx_data.empty:
                raise ValueError(f"No FX data found for {fx_ticker} on {date}")
            
            rate = fx_data['Close'].iloc[0]
            # For EUR/USD we need to take inverse if converting USD to EUR
            if fx_ticker == "EURUSD=X":
                rate = 1 / rate
                
            self._fx_rates_cache[cache_key] = rate
            return rate
        except Exception as e:
            print(f"Error fetching FX rate for {fx_ticker} on {date}: {e}")
            return None

    def fetch_monthly_prices(self, owner_id: int, start_date: str) -> pd.DataFrame:
        """
        Fetch end-of-month prices for all assets and convert to EUR if needed.
        """
        # Get active assets with their currency information
        assets = self.db.get_active_assets(owner_id)
        if assets is None or assets.empty:
            return pd.DataFrame()

        all_prices = []
        
        for _, asset in assets.iterrows():
            ticker = asset['yahoo_ticker']
            fx_ticker = asset.get('yahoo_fx_ticker')  # Use get() in case column doesn't exist
            
            try:
                # Fetch price data
                price_data = yf.download(ticker, start=start_date)
                if price_data.empty:
                    print(f"No data found for {ticker}")
                    continue

                # Resample to end of month
                monthly_data = price_data['Close'].resample('ME').last()
                
                # Convert to EUR if needed
                if fx_ticker:
                    for date, price in monthly_data.items():
                        fx_rate = self._get_fx_rate(fx_ticker, date.strftime('%Y-%m-%d'))
                        if fx_rate is not None:
                            monthly_data[date] = price * fx_rate
                
                # Create price records
                for date, price in monthly_data.items():
                    all_prices.append({
                        'asset_id': asset['asset_id'],
                        'date': date,
                        'price': price,
                        'currency': 'EUR'  # Now all prices are in EUR
                    })
                    
            except Exception as e:
                print(f"Error processing {ticker}: {e}")
                continue

        return pd.DataFrame(all_prices)
