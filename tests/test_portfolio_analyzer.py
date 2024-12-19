"""
Tests for the Portfolio Analyzer module.
"""

import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
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
    return pd.DataFrame({
        'name': ['Stock A', 'Stock B'],
        'asset_id': [1, 2],
        'yahoo_ticker': ['AAPL', 'MSFT'],
        'total_quantity': [100, 50]
    })

@pytest.fixture
def sample_transactions():
    """Create sample transactions data."""
    return pd.DataFrame({
        'event_type': ['buy', 'buy'],
        'asset_id': [1, 2],
        'owner_id': [10, 10],
        'name': ['Stock A', 'Stock B'],
        'date': ['2023-02-01', '2023-02-01'],
        'quantity': [100, 50],
        'price_fx': [150.0, 200.0],
        'price_eur': [140.0, 190.0],
        'amount': [14000.0, 9500.0]
    })

@pytest.fixture
def sample_price_data():
    """Create sample price data."""
    dates = pd.date_range('2023-02-01', '2023-02-03')
    return pd.DataFrame({
        'Stock A': [150.0, 155.0, 160.0],
        'Stock B': [200.0, 205.0, 210.0]
    }, index=dates)

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
    assert 'AAPL' in analyzer.name_ticker_map.values()

@patch('yfinance.download')
def test_fetch_market_data(mock_yf, analyzer, sample_assets):
    """Test fetching market data from Yahoo Finance."""
    # Setup test data
    analyzer.assets_df = sample_assets
    analyzer.name_ticker_map = {'Stock A': 'AAPL', 'Stock B': 'MSFT'}
    
    # Create mock data that matches yfinance output format
    dates = pd.date_range('2023-02-01', '2023-02-03')
    
    # Create mock data with proper structure and no NaN values
    mock_data = pd.DataFrame(
        {
            ('Close', 'AAPL'): [150.0, 150.0, 150.0],
            ('Close', 'MSFT'): [200.0, 200.0, 200.0],
            ('Open', 'AAPL'): [100.0, 100.0, 100.0],
            ('Open', 'MSFT'): [200.0, 200.0, 200.0],
            ('High', 'AAPL'): [101.0, 101.0, 101.0],
            ('High', 'MSFT'): [201.0, 201.0, 201.0],
            ('Low', 'AAPL'): [99.0, 99.0, 99.0],
            ('Low', 'MSFT'): [199.0, 199.0, 199.0],
            ('Adj Close', 'AAPL'): [150.0, 150.0, 150.0],
            ('Adj Close', 'MSFT'): [200.0, 200.0, 200.0],
            ('Volume', 'AAPL'): [1000000, 1000000, 1000000],
            ('Volume', 'MSFT'): [2000000, 2000000, 2000000]
        },
        index=dates
    )
    
    # Set the column index to MultiIndex and ensure data is pre-filled
    mock_data.columns = pd.MultiIndex.from_tuples(mock_data.columns)
    mock_data = mock_data.ffill()  # Pre-fill any NaN values
    mock_yf.return_value = mock_data
    
    # Execute test
    analyzer.fetch_market_data()
    
    # Verify results
    assert analyzer.price_data is not None
    assert len(analyzer.price_data.columns) == 2
    assert 'Stock A' in analyzer.price_data.columns
    assert analyzer.price_data.loc[dates[0], 'Stock A'] == 150.0
    assert analyzer.price_data.loc[dates[0], 'Stock B'] == 200.0
    
    # Verify no NaN values in the data
    assert not analyzer.price_data.isna().any().any(), "Price data should not contain NaN values"

def test_calculate_monthly_positions(analyzer, sample_transactions, sample_assets):
    """Test calculation of monthly positions."""
    # Setup test data
    analyzer.assets_df = sample_assets
    analyzer.transactions_df = sample_transactions.copy()
    analyzer.transactions_df['date'] = pd.to_datetime(analyzer.transactions_df['date'])
    
    # Create price data with proper structure
    dates = pd.date_range('2023-02-01', '2023-02-03', freq='D')
    analyzer.price_data = pd.DataFrame({
        'Stock A': [150.0, 155.0, 160.0],
        'Stock B': [200.0, 205.0, 210.0]
    }, index=dates)
    
    # Execute test
    dates, positions = analyzer.calculate_monthly_positions()
    
    # Verify results
    assert positions is not None
    assert isinstance(positions, pd.DataFrame)
    assert len(positions) > 0, "Positions DataFrame should not be empty"
    assert all(col in positions.columns for col in ['Stock A', 'Stock B']), "All stock columns should be present"
    
    # Get the last day of February 2023
    feb_end = pd.Timestamp('2023-02-28')
    if feb_end in positions.index:
        assert positions.loc[feb_end, 'Stock A'] == 100
        assert positions.loc[feb_end, 'Stock B'] == 50
    else:
        # If February end is not in the index, check the first available date
        first_date = positions.index[0]
        assert positions.loc[first_date, 'Stock A'] == 100
        assert positions.loc[first_date, 'Stock B'] == 50

def test_calculate_portfolio_proportions(analyzer):
    """Test calculation of portfolio proportions."""
    # Setup test data
    positions = pd.DataFrame({
        'Stock A': [100],
        'Stock B': [50]
    }, index=[pd.Timestamp('2023-02-01')])
    
    # Setup price data with matching date
    analyzer.price_data = pd.DataFrame({
        'Stock A': [150.0],
        'Stock B': [200.0]
    }, index=[pd.Timestamp('2023-02-01')])
    
    # Execute test
    proportions = analyzer.calculate_portfolio_proportions(positions)
    
    # Calculate expected proportions
    total_value = 100 * 150.0 + 50 * 200.0  # Stock A value + Stock B value
    expected_stock_a = (100 * 150.0 / total_value) * 100  # Convert to percentage
    expected_stock_b = (50 * 200.0 / total_value) * 100   # Convert to percentage
    
    # Verify results
    assert proportions is not None
    assert isinstance(proportions, pd.DataFrame)
    assert abs(proportions.iloc[0].sum() - 100.0) < 0.01
    assert abs(proportions.iloc[0]['Stock A'] - expected_stock_a) < 0.01
    assert abs(proportions.iloc[0]['Stock B'] - expected_stock_b) < 0.01

def test_invalid_start_date():
    """Test handling of invalid start date."""
    with pytest.raises(Exception):
        PortfolioAnalyzer(owner_id=10, start_date="invalid-date")

@patch('src.portfolio_analyzer.PostgresConnector')
def test_empty_portfolio(mock_db):
    """Test handling of empty portfolio data."""
    # Setup mock database connector with empty data
    mock_instance = MagicMock()
    mock_instance.get_active_assets.return_value = pd.DataFrame(
        columns=['name', 'asset_id', 'yahoo_ticker', 'total_quantity'])
    mock_instance.get_portfolio_transactions.return_value = pd.DataFrame(
        columns=['event_type', 'asset_id', 'owner_id', 'name', 'date', 
                'quantity', 'price_fx', 'price_eur', 'amount'])
    mock_db.return_value.__enter__.return_value = mock_instance

    # Execute test
    analyzer = PortfolioAnalyzer(owner_id=10, start_date="2023-01-01")
    result = analyzer.analyze()

    # Verify result
    assert result is None  # Should return None for empty portfolio