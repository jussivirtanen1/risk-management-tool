"""
Main module for running portfolio analysis and plotting.
"""

import yfinance as yf
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

def fetch_stock_data(ticker: str, start_date: str) -> Optional[pd.DataFrame]:
    """
    Fetch stock data from Yahoo Finance.
    
    Args:
        ticker: Yahoo Finance ticker symbol
        start_date: Start date for data fetching
        
    Returns:
        Optional[pd.DataFrame]: Stock price data or None if fetch fails
    """
    try:
        data = yf.download(ticker, start=start_date)
        if data.empty:
            print(f"No data found for ticker {ticker}")
            return None
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
    
    if active_assets is None or active_assets.empty:
        print("No active assets found")
        return
    
    # Create plots for each active asset
    for _, asset in active_assets.iterrows():
        ticker = asset['yahoo_ticker']
        name = asset['name']
        print(f"\nProcessing {name} ({ticker})")
        
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
    
    Args:
        start_date: Start date for analysis
        ma_periods: List of periods for moving averages
    """
    owner_ids = [10, 20, 30]
    
    for owner_id in owner_ids:
        print(f"\n=== Processing Owner ID: {owner_id} ===")
        try:
            # Run portfolio analysis
            print("\n=== Running Portfolio Analysis ===")
            analyzer = PortfolioAnalyzer(owner_id, start_date)
            portfolio_data = analyzer.analyze()
            
            if portfolio_data is not None:
                print("Portfolio analysis completed successfully")
            else:
                print("Portfolio analysis failed")
            
            # Create moving average plots
            print("\n=== Creating Moving Average Plots ===")
            create_moving_average_plots(owner_id, start_date, ma_periods)
            
        except Exception as e:
            print(f"Error processing owner {owner_id}: {e}")
            continue

if __name__ == "__main__":
    main() 