import os
from src.data_fetcher import StockDataFetcher
from src.plotter import MovingAveragePlotter
from typing import List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_stock(symbol: str, periods: List[int], save_path: str) -> None:
    """Process a single stock: fetch data and create plot."""
    try:
        fetcher = StockDataFetcher(symbol)
        data = fetcher.fetch_data(period="1y")
        
        if data is not None and not data.empty:
            plotter = MovingAveragePlotter(data, symbol)
            plot_path = plotter.plot(periods, save_path)
            
            if plot_path:
                logger.info(f"Plot saved successfully for {symbol} at: {plot_path}")
            else:
                logger.error(f"Failed to create plot for {symbol}")
        else:
            logger.error(f"Failed to fetch data for {symbol}")
    except Exception as e:
        logger.error(f"Error processing {symbol}: {e}")

def main():
    # Get desktop path
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    output_dir = os.path.join(desktop, "stock_plots")
    
    # List of stock symbols to analyze
    stocks = [
        "AAPL",    # Apple
        "MSFT",    # Microsoft
        "GOOGL",   # Alphabet (Google)
        "AMZN",    # Amazon
        "META"     # Meta (Facebook)
    ]
    
    # Moving average periods
    periods = [20, 50, 200]
    
    logger.info(f"Starting analysis for {len(stocks)} stocks...")
    
    # Process each stock
    for symbol in stocks:
        process_stock(symbol, periods, output_dir)
    
    logger.info("Analysis complete. Check the stock_plots directory for PDF files.")

if __name__ == "__main__":
    main()