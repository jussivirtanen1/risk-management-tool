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
        print(f"[fetch_monthly_prices] Starting price fetch for owner {owner_id}")
        
        # Get active assets with their currency information
        assets = self.db.get_active_assets(owner_id)
        if assets is None or assets.empty:
            print(f"[fetch_monthly_prices] No active assets found for owner {owner_id}")
            return pd.DataFrame()

        print(f"[fetch_monthly_prices] Found {len(assets)} active assets for owner {owner_id}:")
        for _, asset in assets.iterrows():
            print(f"[fetch_monthly_prices]   - {asset['name']} ({asset['yahoo_ticker']}) - Quantity: {asset['total_quantity']}")

        all_prices = []
        
        for _, asset in assets.iterrows():
            ticker = asset['yahoo_ticker']
            fx_ticker = asset.get('yahoo_fx_ticker')  # Use get() in case column doesn't exist
            
            try:
                # Fetch price data
                print(f"\n[fetch_monthly_prices] Fetching data for {asset['name']} ({ticker})")
                try:
                    print(f"[fetch_monthly_prices] Attempting to fetch from {start_date}")
                    price_data = yf.download(ticker, start=start_date)
                    print(f"[fetch_monthly_prices] Received data shape: {price_data.shape}")
                except Exception as e:
                    if "YFInvalidPeriodError" in str(e):
                        print(f"[fetch_monthly_prices] YFInvalidPeriodError for {ticker}, trying with more recent date")
                        recent_start = pd.Timestamp.now() - pd.DateOffset(months=6)
                        recent_start_str = recent_start.strftime('%Y-%m-%d')
                        print(f"[fetch_monthly_prices] Retrying from {recent_start_str}")
                        price_data = yf.download(ticker, start=recent_start_str)
                        print(f"[fetch_monthly_prices] Received data shape after retry: {price_data.shape}")
                    else:
                        print(f"[fetch_monthly_prices] Unexpected error for {ticker}: {str(e)}")
                        raise e

                if price_data.empty:
                    print(f"[fetch_monthly_prices] No data found for {ticker}")
                    continue

                print(f"\n[fetch_monthly_prices] Price data for {asset['name']} ({ticker}):")
                print(price_data.tail())

                # Resample to end of month
                print(f"[fetch_monthly_prices] Resampling data to monthly for {ticker}")
                monthly_data = price_data['Close'].resample('ME').last()
                print(f"[fetch_monthly_prices] Got {len(monthly_data)} monthly data points")
                
                # Convert to EUR if needed
                if fx_ticker:
                    print(f"\n[fetch_monthly_prices] Converting {ticker} prices from {fx_ticker} to EUR")
                    for date, price in monthly_data.items():
                        print(f"[fetch_monthly_prices] Getting FX rate for {date.strftime('%Y-%m-%d')}")
                        fx_rate = self._get_fx_rate(fx_ticker, date.strftime('%Y-%m-%d'))
                        if fx_rate is not None:
                            monthly_data[date] = price * fx_rate
                            print(f"[fetch_monthly_prices]   {date.strftime('%Y-%m-%d')}: {price:.2f} -> {monthly_data[date]:.2f} EUR (rate: {fx_rate:.4f})")
                        else:
                            print(f"[fetch_monthly_prices] Failed to get FX rate for {date.strftime('%Y-%m-%d')}")
                
                # Create price records
                print(f"[fetch_monthly_prices] Creating price records for {ticker}")
                records_added = 0
                for date, price in monthly_data.items():
                    all_prices.append({
                        'asset_id': asset['asset_id'],
                        'date': date,
                        'price': price,
                        'currency': 'EUR'
                    })
                    records_added += 1
                print(f"[fetch_monthly_prices] Added {records_added} price records for {ticker}")
                    
            except Exception as e:
                print(f"[fetch_monthly_prices] Error processing {ticker}: {str(e)}")
                print(f"[fetch_monthly_prices] Full error details: {repr(e)}")
                continue

        result_df = pd.DataFrame(all_prices)
        print(f"[fetch_monthly_prices] Completed processing for owner {owner_id}. Total records: {len(result_df)}")
        return result_df
