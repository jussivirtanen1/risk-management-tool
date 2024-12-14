import matplotlib.pyplot as plt
import pandas as pd
import os
from typing import List, Optional
from matplotlib.backends.backend_pdf import PdfPages

class MovingAveragePlotter:
    def __init__(self, data: pd.DataFrame, symbol: str):
        self.data = data
        self.symbol = symbol
        
    def calculate_ma(self, periods: List[int]) -> None:
        """Calculate moving averages for specified periods."""
        for period in periods:
            self.data[f'MA{period}'] = self.data['Close'].rolling(window=period).mean()
    
    def plot(self, periods: List[int], save_path: str) -> Optional[str]:
        """
        Create and save moving average plot as PDF.
        
        Args:
            periods (List[int]): List of periods for moving averages
            save_path (str): Directory to save the plot
            
        Returns:
            str: Path to saved plot or None if failed
        """
        try:
            self.calculate_ma(periods)
            
            plt.figure(figsize=(12, 6))
            plt.plot(self.data.index, self.data['Close'], label='Close Price')
            
            for period in periods:
                plt.plot(self.data.index, self.data[f'MA{period}'], 
                        label=f'{period}-day MA')
            
            plt.title(f'{self.symbol} Stock Price with Moving Averages')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True)
            
            os.makedirs(save_path, exist_ok=True)
            file_path = os.path.join(save_path, f'{self.symbol}_moving_averages.pdf')
            plt.savefig(file_path, format='pdf', bbox_inches='tight')
            plt.close()
            
            return file_path
        except Exception as e:
            print(f"Error creating plot for {self.symbol}: {e}")
            return None