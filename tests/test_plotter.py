"""
Tests for the Plotter module and moving average plotting functionality.
"""

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import polars as pl
import numpy as np
from pathlib import Path
import os
import shutil
from src.plotter import MovingAveragePlotter
from main import create_moving_average_plots, get_output_path
import pytest
from datetime import datetime, timedelta

@pytest.fixture
def sample_stock_data():
    """Create sample stock data for testing."""
    dates = pl.date_range(start=datetime(2023, 1, 1), end=datetime(2023, 3, 1), interval="1d")
    data = pl.DataFrame({
        "Close": pl.Series(range(100)),
        "Date": dates
    })
    return data

    # dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    # data = pd.DataFrame({
    #     'Close': range(100),
    #     'Date': dates
    # }).set_index('Date')
    return data

@pytest.fixture
def plotter(sample_stock_data):
    """Create a MovingAveragePlotter instance for testing."""
    return MovingAveragePlotter(sample_stock_data, "Test Stock")

class TestMovingAveragePlotter(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create sample price data
        dates = pd.date_range('2023-01-01', '2023-03-01')
        self.sample_data = pd.DataFrame({
            'Open': np.random.randn(len(dates)) + 100,
            'High': np.random.randn(len(dates)) + 101,
            'Low': np.random.randn(len(dates)) + 99,
            'Close': np.random.randn(len(dates)) + 100,
            'Volume': np.random.randint(1000000, 10000000, len(dates))
        })
        
        self.test_symbol = "TEST"
        self.plotter = MovingAveragePlotter(self.sample_data, self.test_symbol)
        
        # Create temporary directory for test outputs
        self.test_output_dir = "test_outputs"
        os.makedirs(self.test_output_dir, exist_ok=True)

    def tearDown(self):
        """Clean up test fixtures after each test method."""
        # Remove temporary test directory
        if os.path.exists(self.test_output_dir):
            shutil.rmtree(self.test_output_dir)

    def test_calculate_ma(self):
        """Test calculation of moving averages."""
        periods = [20, 50]
        self.plotter.calculate_ma(periods)
        
        # Verify moving averages were calculated
        for period in periods:
            ma_col = f'MA{period}'
            self.assertIn(ma_col, self.plotter.data.columns)
            self.assertEqual(len(self.plotter.data[ma_col].dropna()),
                           len(self.plotter.data) - period + 1)

    def test_plot_creation(self):
        """Test creation of moving average plot."""
        with patch('matplotlib.pyplot.savefig') as mock_savefig:
            periods = [20, 50]
            output_path = self.plotter.plot(periods, 10)  # Use owner_id 10
            
            # Verify plot was attempted to be saved
            assert mock_savefig.called
            assert output_path is not None

    def test_invalid_periods(self):
        """Test handling of invalid moving average periods."""
        # Test with negative period
        with self.assertRaises(Exception):
            self.plotter.calculate_ma([-20])
        
        # Test with period longer than data
        long_period = len(self.sample_data) + 100
        self.plotter.calculate_ma([long_period])
        self.assertTrue(self.plotter.data[f'MA{long_period}'].isna().all())

    def test_get_plots_path(self):
        """Test plot path creation for multiple owners."""
        test_owner_ids = [10, 20, 30]  # Define test_owner_ids here
        for owner_id in test_owner_ids:
            path = self.plotter.get_plots_path(owner_id)
            self.assertTrue(f"owner_{owner_id}" in path)
            self.assertTrue(os.path.exists(path))

    def test_plot_error_handling(self):
        """Test error handling in plot creation."""
        with patch('matplotlib.pyplot.savefig', side_effect=Exception("Test error")):
            result = self.plotter.plot([20, 50], 10)
            assert result is None

class TestPlottingFunctionality(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.owner_id = 10
        self.start_date = "2023-01-01"
        self.ma_periods = [20, 50, 200]
        
        # Sample active assets data
        self.sample_active_assets = pl.DataFrame({
            'name': ['Stock A', 'Stock B'],
            'asset_id': [1, 2],
            'yahoo_ticker': ['AAPL', 'MSFT'],
            'total_quantity': [100, 50]
        })
        
        # Sample stock data
        dates = pd.date_range('2023-01-01', '2023-03-01')
        self.sample_stock_data = pl.DataFrame({
            'Open': np.random.randn(len(dates)) + 100,
            'High': np.random.randn(len(dates)) + 101,
            'Low': np.random.randn(len(dates)) + 99,
            'Close': np.random.randn(len(dates)) + 100,
            'Volume': np.random.randint(1000000, 10000000, len(dates))
        })

    @patch('main.PostgresConnector')
    @patch('main.fetch_stock_data')
    @patch('main.MovingAveragePlotter')
    def test_create_moving_average_plots(self, mock_plotter, mock_fetch_data, mock_db):
        """Test creation of moving average plots for all active assets."""
        # Setup mocks
        mock_instance = MagicMock()
        mock_instance.get_active_assets.return_value = self.sample_active_assets
        mock_db.return_value.__enter__.return_value = mock_instance
        
        mock_fetch_data.return_value = self.sample_stock_data
        
        mock_plotter_instance = MagicMock()
        mock_plotter_instance.plot.return_value = "test_plot.pdf"
        mock_plotter.return_value = mock_plotter_instance
        
        # Execute test
        create_moving_average_plots(self.owner_id, self.start_date, self.ma_periods)
        
        # Verify results
        self.assertEqual(mock_fetch_data.call_count, 2)  # Called for each asset
        self.assertEqual(mock_plotter.call_count, 2)  # Created for each asset
        self.assertEqual(mock_plotter_instance.plot.call_count, 2)  # Plot created for each asset

    def test_output_path(self):
        """Test output path generation."""
        # Test default path
        default_path = get_output_path()
        self.assertTrue("analysis" in default_path)
        
        # Test custom folder
        custom_path = get_output_path("stock_plots")
        self.assertTrue("stock_plots" in custom_path)
        
        # Test Docker environment
        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = True  # Simulate Docker environment
            docker_path = get_output_path("test")
            self.assertTrue(docker_path.startswith('/app/'))

if __name__ == '__main__':
    unittest.main()