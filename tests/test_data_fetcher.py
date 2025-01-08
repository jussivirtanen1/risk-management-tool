import pytest
from src.data_fetcher import StockDataFetcher
import pandas as pd
from unittest.mock import Mock, patch

def test_fetch_monthly_prices_multiple_owners(test_owner_ids, mock_portfolio_data):
    """Test fetching monthly prices for multiple owners."""
    for owner_id in test_owner_ids:
        with patch('yfinance.download') as mock_yf:
            # Mock the database connector
            mock_db = Mock()
            mock_db.get_active_assets.return_value = pd.DataFrame(mock_portfolio_data[owner_id]['assets'])
            
            # Mock yfinance data
            mock_yf.return_value = pd.DataFrame({
                'Close': [100, 200, 300],
                'Date': pd.date_range(start='2023-01-01', periods=3, freq='ME')
            }).set_index('Date')
            
            # Create DataFetcher instance
            fetcher = StockDataFetcher(mock_db)
            
            # Test fetch_monthly_prices
            result = fetcher.fetch_monthly_prices(owner_id, '2023-01-01')
            
            assert result is not None
            assert not result.empty
            assert 'asset_id' in result.columns
            assert 'price' in result.columns
            assert 'date' in result.columns

def test_get_fx_rate_caching():
    """Test FX rate caching functionality."""
    with patch('yfinance.download') as mock_yf:
        mock_db = Mock()
        fetcher = StockDataFetcher(mock_db)
        
        # Mock FX data
        mock_yf.return_value = pd.DataFrame({
            'Close': [1.2],
            'Date': ['2023-01-01']
        }).set_index('Date')
        
        # First call should fetch from yfinance
        rate1 = fetcher._get_fx_rate('EURUSD=X', '2023-01-01')
        assert mock_yf.call_count == 1
        
        # Second call should use cached value
        rate2 = fetcher._get_fx_rate('EURUSD=X', '2023-01-01')
        assert mock_yf.call_count == 1  # Should not increase
        assert rate1 == rate2
