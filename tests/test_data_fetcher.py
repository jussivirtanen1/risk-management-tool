import pytest
from src.data_fetcher import StockDataFetcher
import pandas as pd

def test_fetch_data_success(mocker):
    # Mock yfinance Ticker
    mock_ticker = mocker.patch('yfinance.Ticker')
    mock_history = mocker.MagicMock(return_value=pd.DataFrame({
        'Close': [100, 101, 102]
    }))
    mock_ticker.return_value.history = mock_history
    
    fetcher = StockDataFetcher("AAPL")
    data = fetcher.fetch_data()
    
    assert data is not None
    assert len(data) == 3
    mock_history.assert_called_once_with(period="1y")

def test_fetch_data_failure(mocker):
    # Mock yfinance Ticker to raise exception
    mock_ticker = mocker.patch('yfinance.Ticker')
    mock_ticker.return_value.history.side_effect = Exception("API Error")
    
    fetcher = StockDataFetcher("AAPL")
    data = fetcher.fetch_data()
    
    assert data is None
