"""
Main module for running portfolio analysis and plotting.
"""

import yfinance as yf
import polars as pl
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import os
from typing import List, Optional
from src.portfolio_analyzer import PortfolioAnalyzer
from src.plotter import MovingAveragePlotter
from src.db_connector import PostgresConnector

def get_output_path(folder: str = "analysis", owner_id: int = None) -> str:
    """
    Get the path for output files.
    
    Args:
        folder: Name of the output folder
        owner_id: ID of the portfolio owner
    """
    if os.path.exists('/.dockerenv'):
        base_path = f'/app/{folder}'
    else:
        base_path = str(Path.home() / "Desktop" / folder)
    
    if owner_id is not None:
        base_path = os.path.join(base_path, f"owner_{owner_id}")
        
    return base_path

def fetch_stock_data(ticker: str, start_date: str) -> Optional[pl.DataFrame]:
    """
    Fetch stock data from Yahoo Finance.
    
    Args:
        ticker: Yahoo Finance ticker symbol
        start_date: Start date for data fetching
        
    Returns:
        Optional[pl.DataFrame]: Stock price data or None if fetch fails
    """
    try:
        # print(f"\nFetching data for {ticker}")
        try:
            # First try with original start_date
            # Set end date to today using datetime
            end_date = datetime.now().strftime('%Y-%m-%d')
            data = pl.from_pandas(yf.download(ticker, start=start_date, end=end_date))
        except Exception as e:
            # if "YFInvalidPeriodError" in str(e):
            #     print(f"Retrying {ticker} with more recent start date...")
            #     # Try with a more recent start date (6 months ago)
            #     recent_start = pd.Timestamp.now() - pd.DateOffset(months=6)
            #     data = yf.download(ticker, start=recent_start.strftime('%Y-%m-%d'), end=end_date)
            # else:
            raise e

        if data.is_empty():
            print(f"No data found for ticker {ticker}")
            return None
            
        # print(f"\nPrice data for {ticker}:")
        # print(data.tail())  # Print last 5 rows of data
        return data
        
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

def create_moving_average_plots(owner_id: int, start_date: str, ma_periods: List[int]) -> None:
    """
    Create moving average plots for all active assets.
    
    Args:
        owner_id: ID of the portfolio owner
        start_date: Start date for analysis
        ma_periods: List of periods for moving averages
    """
    output_path = get_output_path("stock_plots", owner_id)
    os.makedirs(output_path, exist_ok=True)
    
    # Get active assets
    with PostgresConnector() as db:
        active_assets = db.get_active_assets(owner_id)
        assert isinstance(active_assets, pl.DataFrame), f"Expected Polars DataFrame but got {type(active_assets)}"
        print("active_assets type")
        print(type(active_assets))
    
    if active_assets is None or active_assets.is_empty():
        print("No active assets found")
        return
    
    # Create plots for each active asset
    for asset in active_assets.iter_rows(named=True):
        ticker = asset['yahoo_ticker']
        name = asset['name']
        # print(f"\nProcessing {name} ({ticker})")
        
        # Fetch stock data
        stock_data = fetch_stock_data(ticker, start_date)
        if stock_data is None:
            continue
        
        # Create and save plot
        plotter = MovingAveragePlotter(stock_data, name)
        plot_path = plotter.plot(ma_periods, owner_id)
        
        if plot_path:
            print(f"Created plot for {name}: {plot_path}")
        else:
            print(f"Failed to create plot for {name}")

def main(start_date: str = "2023-01-01", ma_periods: List[int] = [20, 50, 200]) -> None:
    """
    Main function to run portfolio analysis and create plots for multiple owners.
    """
    owner_ids = [10, 20, 30]
    
    for owner_id in owner_ids:
        print(f"\n[main] === Processing Owner ID: {owner_id} ===")
        try:
            # Run portfolio analysis
            print("\n[main] === Running Portfolio Analysis ===")
            analyzer = PortfolioAnalyzer(owner_id, start_date)
            print(f"[main] Created analyzer for owner {owner_id}")
            
            portfolio_data = analyzer.analyze()
            
            if portfolio_data is not None:
                print(f"[main] Portfolio analysis completed successfully for owner {owner_id}")
                # print(f"[main] Portfolio data shape: {portfolio_data.shape}")
            else:
                print(f"[main] Portfolio analysis failed for owner {owner_id}")
            
            # Create moving average plots
            print(f"\n[main] === Creating Moving Average Plots for owner {owner_id} ===")
            create_moving_average_plots(owner_id, start_date, ma_periods)
            
        except Exception as e:
            print(f"[main] Error processing owner {owner_id}: {str(e)}")
            print(f"[main] Full error details: {repr(e)}")
            continue

if __name__ == "__main__":
    main() 