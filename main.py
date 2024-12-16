"""
Main application script that runs both plotting and portfolio analysis
"""

from src.plotter import MovingAveragePlotter
from src.data_fetcher import StockDataFetcher
from src.portfolio_analyzer import main as run_portfolio_analysis
import os
from datetime import datetime

def run_plotting():
    """Run the original plotting functionality"""
    # Your existing main.py plotting code here
    symbol = os.getenv('STOCK_SYMBOL', 'AAPL')
    period = os.getenv('TIME_PERIOD', '1y')
    ma_periods = [20, 50, 200]
    
    # Fetch data
    fetcher = StockDataFetcher(symbol)
    data = fetcher.fetch_data(period)
    
    if data is not None:
        # Create and save plot
        plotter = MovingAveragePlotter(data, symbol)
        save_path = '/app/stock_plots'  # Your existing output path
        plot_file = plotter.plot(ma_periods, save_path)
        if plot_file:
            print(f"Plot saved to: {plot_file}")
    else:
        print("Failed to fetch data")

def main():
    """Run all analyses"""
    # Run original plotting functionality
    print("Running plotting analysis...")
    run_plotting()
    
    # Run portfolio analysis
    print("\nRunning portfolio analysis...")
    owner_id = int(os.getenv('OWNER_ID', '10'))
    # Convert start_date string to datetime
    start_date = datetime.strptime(
        os.getenv('START_DATE', '2020-01-01'),
        '%Y-%m-%d'
    )
    run_portfolio_analysis(owner_id=owner_id, start_date=start_date)

if __name__ == "__main__":
    main()