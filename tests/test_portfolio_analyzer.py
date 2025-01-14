"""
Tests for the Portfolio Analyzer module.
"""

import pytest
from unittest.mock import patch, MagicMock
import polars as pl
import numpy as np
from datetime import datetime
from src.portfolio_analyzer import PortfolioAnalyzer

@pytest.fixture
def analyzer():
    """Create a portfolio analyzer instance."""
    return PortfolioAnalyzer(owner_id=10, start_date="2023-01-01")

@pytest.fixture
def sample_assets():
    """Create sample assets data."""
    return pl.DataFrame({
        'name': ['Stock A', 'Stock B'],
        'asset_id': [1, 2],
        'yahoo_ticker': ['AAPL', 'MSFT'],
        'total_quantity': [100, 50]
    })

@pytest.fixture
def sample_transactions():
    """Create sample transactions data."""
    return pl.DataFrame({
        'event_type': ['buy', 'buy', 'buy', 'buy'],
        'asset_id': [1, 2, 1, 2],
        'owner_id': [10, 10, 10, 10],
        'name': ['Stock A', 'Stock B', 'Stock A', 'Stock B'],
        'date': ['2023-02-01', '2023-02-01', '2023-02-02', '2023-02-02'],
        'quantity': [100, 50, 100, 70],
        'price_fx': [150.0, 200.0, 150.0, 175.0],
        'price_eur': [140.0, 190.0, 140.0, 175.0],
        'amount': [14000.0, 9500.0, 10000.0, 12500.0]
    })

@pytest.fixture
def sample_price_data():
    """Create sample price data."""
    dates = pl.date_range(
        start=datetime(2023, 2, 1),
        end=datetime(2023, 2, 3),
        interval="1d"
    )
    return pl.DataFrame({
        'Stock A': [150.0, 155.0, 160.0],
        'Stock B': [200.0, 205.0, 210.0],
        'date': dates
    })

@patch('src.portfolio_analyzer.PostgresConnector')
def test_fetch_portfolio_data(mock_db, analyzer, sample_assets, sample_transactions):
    """Test fetching portfolio data from database."""
    # Setup mock database connector
    mock_instance = MagicMock()
    mock_instance.get_active_assets.return_value = sample_assets
    mock_instance.get_portfolio_transactions.return_value = sample_transactions
    mock_db.return_value.__enter__.return_value = mock_instance
    
    # Execute test
    analyzer.fetch_portfolio_data()
    
    # Verify results
    assert analyzer.assets_df is not None
    assert analyzer.transactions_df is not None
    assert len(analyzer.assets_df) == 2
    assert len(analyzer.transactions_df) == 2

def test_calculate_monthly_positions(analyzer, sample_transactions, sample_assets):
    """Test calculation of monthly positions."""
    # Setup test data
    analyzer.assets_df = sample_assets
    analyzer.transactions_df = sample_transactions.with_columns(
        pl.col('date').str.strptime(pl.Date, format='%Y-%m-%d')
    )
    
    analyzer.price_data = pl.DataFrame({
        'Stock A': [150.0, 155.0, 160.0],
        'Stock B': [200.0, 205.0, 210.0]
    }).with_columns(date=pl.date_range(start=datetime(2023, 2, 1), end=datetime(2023, 2, 3), interval="1d"))
    
    # Execute test
    dates, positions = analyzer.calculate_monthly_positions()
    
    # Verify results
    assert positions is not None
    assert isinstance(positions, pl.DataFrame)
    assert len(positions) > 0, "Positions DataFrame should not be empty"
    assert all(col in positions.columns for col in ['Stock A', 'Stock B']), "All stock columns should be present"
    
    # Get the last day of February 2023
    feb_end = pl.lit("2023-02-28").str.strptime(pl.Date, format='%Y-%m-%d')
    if feb_end in positions['Date']:
        assert positions.filter(pl.col('Date') == feb_end)['Stock A'][0] == 100
        assert positions.filter(pl.col('Date') == feb_end)['Stock B'][0] == 50
    else:
        # If February end is not in the index, check the first available date
        first_date = positions['Date'][0]
        assert positions.filter(pl.col('Date') == first_date)['Stock A'][0] == 100
        assert positions.filter(pl.col('Date') == first_date)['Stock B'][0] == 50

def test_calculate_portfolio_proportions(analyzer):
    """Test calculation of portfolio proportions."""
    # Setup test data
    test_date = pl.lit("2023-02-01").str.strptime(pl.Date, format='%Y-%m-%d')
    
    # Setup assets data
    analyzer.assets_df = pl.DataFrame({
        'name': ['Stock A', 'Stock B'],
        'asset_id': [1, 2],
        'yahoo_ticker': ['AAPL', 'MSFT']
    })
    
    positions = pl.DataFrame({
        'Date': [test_date],
        'Stock A': [100],
        'Stock B': [50]
    })
    
    # Setup price data with matching date
    analyzer.price_data = pl.DataFrame({
        'date': [test_date],
        'Stock A': [150.0],
        'Stock B': [200.0]
    })
    
    # Execute test
    proportions = analyzer.calculate_portfolio_proportions(positions)
    
    # Calculate expected proportions
    total_value = 100 * 150.0 + 50 * 200.0  # Stock A value + Stock B value
    expected_stock_a = (100 * 150.0 / total_value) * 100  # Convert to percentage
    expected_stock_b = (50 * 200.0 / total_value) * 100   # Convert to percentage
    
    # Verify results
    assert proportions is not None
    assert isinstance(proportions, pl.DataFrame)
    assert abs(proportions.select(pl.exclude('Date')).row(0).sum() - 100.0) < 0.01
    assert abs(proportions['Stock A'][0] - expected_stock_a) < 0.01
    assert abs(proportions['Stock B'][0] - expected_stock_b) < 0.01

def test_invalid_start_date():
    """Test handling of invalid start date."""
    with pytest.raises(Exception):
        PortfolioAnalyzer(owner_id=10, start_date="invalid-date")

@patch('src.portfolio_analyzer.PostgresConnector')
def test_empty_portfolio(mock_db):
    """Test handling of empty portfolio data."""
    # Setup mock database connector with empty data
    mock_instance = MagicMock()
    mock_instance.get_active_assets.return_value = pl.DataFrame(
        schema={
            'name': pl.Utf8,
            'asset_id': pl.Int64,
            'yahoo_ticker': pl.Utf8,
            'total_quantity': pl.Float64
        }
    )
    mock_instance.get_portfolio_transactions.return_value = pl.DataFrame(
        schema={
            'event_type': pl.Utf8,
            'asset_id': pl.Int64,
            'owner_id': pl.Int64,
            'name': pl.Utf8,
            'date': pl.Utf8,
            'quantity': pl.Float64,
            'price_fx': pl.Float64,
            'price_eur': pl.Float64,
            'amount': pl.Float64
        }
    )
    mock_db.return_value.__enter__.return_value = mock_instance

    # Execute test
    analyzer = PortfolioAnalyzer(owner_id=10, start_date="2023-01-01")
    result = analyzer.analyze()

    # Verify result
    assert result is None  # Should return None for empty portfolio