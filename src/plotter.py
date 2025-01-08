import matplotlib.pyplot as plt
import pandas as pd
import os
from typing import List, Optional
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
from datetime import datetime

class MovingAveragePlotter:
    def __init__(self, data: pd.DataFrame, asset_name: str):
        """
        Initialize the Moving Average Plotter.
        
        Args:
            data: DataFrame with stock price data
            asset_name: Name of the asset
        """
        self.data = data
        self.asset_name = asset_name
        
    def calculate_ma(self, periods: List[int]) -> None:
        """Calculate moving averages for specified periods."""
        for period in periods:
            self.data[f'MA{period}'] = self.data['Close'].rolling(window=period).mean()

    @staticmethod
    def get_plots_path(owner_id: int) -> str:
        """Get the path to plots directory for specific owner."""
        if os.path.exists('/.dockerenv'):
            base_path = '/app/stock_plots'
        else:
            base_path = str(Path.home() / "Desktop" / "stock_plots")
        
        owner_path = os.path.join(base_path, f"owner_{owner_id}")
        os.makedirs(owner_path, exist_ok=True)
        return owner_path

    def plot(self, ma_periods: List[int], owner_id: int) -> Optional[str]:
        """
        Create and save the plot.
        
        Args:
            ma_periods: List of periods for moving averages
            owner_id: ID of the portfolio owner
            
        Returns:
            Optional[str]: Path to saved plot or None if plotting fails
        """
        try:
            # Calculate moving averages
            self.calculate_ma(ma_periods)
            
            # Create plot
            plt.figure(figsize=(12, 6))
            plt.plot(self.data.index, self.data['Close'], label='Close Price')
            
            for period in ma_periods:
                plt.plot(self.data.index, self.data[f'MA{period}'], 
                        label=f'{period}-day MA')
            
            plt.title(f'{self.asset_name} Stock Price with Moving Averages')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True)
            
            # Save plot to owner-specific directory
            output_path = self.get_plots_path(owner_id)
            filename = f"{self.asset_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.png"
            plot_path = os.path.join(output_path, filename)
            plt.savefig(plot_path)
            plt.close()
            
            return plot_path
            
        except Exception as e:
            print(f"Error creating plot for {self.asset_name}: {e}")
            return None