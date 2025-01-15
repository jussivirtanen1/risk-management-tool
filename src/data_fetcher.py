import yfinance as yf
import pandas as pd
import polars as pl
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
            fx_data = pl.from_pandas(yf.download(fx_ticker, start=date, end=date))
            if fx_data.is_empty():
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

    def fetch_prices_from_yahoo(self, owner_id: int, start_date: str) -> pl.DataFrame:
        """
        Fetch end-of-month prices for all assets and convert to EUR if needed.
        """
        print(f"Starting price fetch for owner {owner_id}")
        
        # Get active assets with their currency information
        assets = self.db.get_active_assets(owner_id)
        if assets is None or assets.height == 0:
            print(f" No active assets found for owner {owner_id}")
            return pl.DataFrame()

        print(f"Found {len(assets)} active assets for owner {owner_id}:")

        all_prices = []
        for asset in assets.iter_rows(named=True):
            ticker = asset['yahoo_ticker']
            fx_ticker = asset['yahoo_fx_ticker']
            try:
                end_date = datetime.now().strftime('%Y-%m-%d')  # Set end date to today
                price_data = yf.download(ticker, start=start_date, end=end_date)
                price_data['date'] = price_data.index
                price_data = pl.from_pandas(price_data).select('Close', pl.col('date').cast(pl.Date)).rename({"Close": ticker})
            except Exception as e:
                raise e

            if price_data.is_empty():
                continue
            
            # Convert to EUR if needed
            if fx_ticker:
                if fx_ticker == "EUREUR=X":
                    price_data_with_fx = price_data.with_columns(pl.lit(1).alias(fx_ticker).cast(pl.Float64))
                    all_prices.append(price_data_with_fx)
                else:
                    # print("fx_ticker", fx_ticker)
                    end_date = datetime.now().strftime('%Y-%m-%d')  # Set end date to today
                    fx_price_data = yf.download(fx_ticker, start=start_date, end=end_date, period='1d')
                    # ticker_info = yf.Ticker(ticker)
                    # print("ticker_info", ticker_info.info)
                    fx_price_data['date'] = fx_price_data.index
                    fx_price_data = pl.from_pandas(fx_price_data)
                    fx_price_data = fx_price_data.select('Close', pl.col('date').cast(pl.Date)).rename({"Close": fx_ticker})
                    # print("fx_price_data after yf.download", fx_price_data)
                    price_data_with_fx = price_data.join(fx_price_data, on='date', how='inner')
                    price_data_with_fx = price_data_with_fx.with_columns((pl.col(ticker) / pl.col(fx_ticker)).alias(ticker))
                    price_data_with_fx = pl.concat([price_data_with_fx])
                    all_prices.append(price_data_with_fx)
        yahoo_tickers = pl.Series(assets.select('yahoo_ticker')).to_list()
        all_prices_combined = pl.concat(all_prices, how="align").select('date', pl.col(yahoo_tickers)).drop_nulls()

        print(f" Completed yahoo finance data fetch for owner {owner_id}. Total records combined: {len(all_prices_combined)}")
        return all_prices_combined
