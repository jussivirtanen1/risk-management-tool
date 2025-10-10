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
def asset_ids():
    """Create sample assets data."""
    return pl.DataFrame({
        'name': ["Nordea Bank Oyj", "Sampo Oyj", "USA Indeksirahasto", "Tanska Indeksirahasto", 
                "Salesforce", "iShares MSCI India UCITS ETF USD (Acc)", "iShares Core S&P 500 UCITS ETF USD (Acc)",
                "Kemira Oyj", "Microsoft", "Nordea Korko A", "BAE Systems PLC", "NVIDIA"],
        'asset_id': [2008, 1012, 1007, 1008, 1013, 2000, 2001, 2010, 2009, 2015, 3000, 3001],
        'yahoo_ticker': ["NDA-FI.HE", "SAMPO.HE", "0P0001K6NM.F", "0P000134KA.CO", "CRM", "QDV5.BE", 
                        "SXR8.DE", "KEMIRA.HE", "MSFT", "0P0000JXE8.F", "BSP.DE", "NVDA"],
        'fx_rate': ["EUREUR=X", "EUREUR=X", "EUREUR=X", "EURDKK=X", "EURUSD=X", "EUREUR=X",
                    "EUREUR=X", "EUREUR=X", "EURUSD=X", "EUREUR=X", "EUREUR=X", "EURUSD=X"],
        'isin': ["FI4000297767", "FI0009003305", "IE00BMTD2W97", "SE0005993078", "US79466L3024",
                 "IE00BZCQB185", "IE00B5BMR087", "FI0009004824", "US5949181045", "FI0008814603",
                 "GB0002634946", "US67066G1040"]
    })

@pytest.fixture
def asset_transactions():
    """Create sample transactions data from real transactions."""
    return pl.DataFrame({
        'event_type': ['Osto', 'Myynti', 'Myynti', 'Myynti', 'Myynti', 'Osto', 'Osto', 'Myynti', 'Split', 'Osto',
                       'Osto', 'Osto', 'Osto', 'Osto', 'Osto', 'Osto', 'Merkintä', 'Osto', 'Osto', 'Osto',
                       'Myynti', 'Merkintä', 'Merkintä', 'Osto', 'Merkintä', 'Osto', 'Osto', 'Osto', 'Osto',
                       'Osto', 'Osto', 'Osto'],
        'asset_id': [1012, 2009, 2000, 2009, 2001, 2010, 2000, 3001, 3001, 3001,
                     3000, 2000, 2001, 2000, 2000, 2001, 2015, 1013, 2001, 2000,
                     2010, 1007, 1008, 2008, 1008, 2000, 2001, 2000, 1012, 1013,
                     2000, 2001],
        'owner_id': [20, 20, 20, 20, 20, 30, 20, 30, 30, 30,
                     30, 20, 30, 30, 20, 20, 20, 10, 20, 20,
                     20, 10, 10, 10, 10, 10, 20, 20, 10, 10,
                     20, 20],
        'name': ['Sampo Oyj', 'Microsoft', 'iShares MSCI India UCITS ETF USD (Acc)', 'Microsoft', 'iShares Core S&P 500 UCITS ETF USD (Acc)',
                 'Kemira Oyj', 'iShares MSCI India UCITS ETF USD (Acc)', 'NVIDIA', 'NVIDIA', 'NVIDIA',
                 'BAE Systems PLC', 'iShares MSCI India UCITS ETF USD (Acc)', 'iShares Core S&P 500 UCITS ETF USD (Acc)', 'iShares MSCI India UCITS ETF USD (Acc)', 'iShares MSCI India UCITS ETF USD (Acc)',
                 'iShares Core S&P 500 UCITS ETF USD (Acc)', 'Nordea Korko A', 'Salesforce', 'iShares Core S&P 500 UCITS ETF USD (Acc)', 'iShares MSCI India UCITS ETF USD (Acc)',
                 'Kemira Oyj', 'USA Indeksirahasto', 'Tanska Indeksirahasto', 'Nordea Bank Oyj', 'Tanska Indeksirahasto',
                 'iShares MSCI India UCITS ETF USD (Acc)', 'iShares Core S&P 500 UCITS ETF USD (Acc)', 'iShares MSCI India UCITS ETF USD (Acc)', 'Sampo Oyj', 'Salesforce',
                 'iShares MSCI India UCITS ETF USD (Acc)', 'iShares Core S&P 500 UCITS ETF USD (Acc)'],
        'date': ['2024-09-26', '2024-08-05', '2024-07-22', '2024-07-22', '2024-07-22',
                 '2024-07-18', '2024-07-05', '2024-06-25', '2024-06-10', '2024-03-04',
                 '2024-02-23', '2024-02-16', '2024-02-16', '2024-02-16', '2023-11-15',
                 '2023-11-15', '2023-10-06', '2023-09-29', '2023-08-15', '2023-08-15',
                 '2023-08-04', '2023-07-03', '2023-07-01', '2023-06-30', '2023-06-22',
                 '2023-06-15', '2023-06-01', '2023-06-01', '2023-05-31', '2023-04-14',
                 '2023-02-14', '2023-02-14'],
        'quantity': [250, -49, -1119, -64, -108, 240, 330, -60, 54, 6,
                     350, 309, 8, 358, 418, 9, 2355.7881, 2, 10, 423,
                     -859, 3.77, 3.5, 1, 7.27, 20, 10, 444, 3, 3,
                     472, 10],
        'price_fx': [41.81, 389.33, 9.301, 441.77, 535.42, 20.8, 9.381, 121, 0, 841.5,
                     14.66, 8.399, 490, 8.399, 7.184, 435.24, 10.6122, 158.55, 428.5, 6.906,
                     14.02, 159.24, 335, 9.86, 344.4, 55.65, 407.51, 6.756, 43.4, 172.43,
                     6.364, 398.7],
        'price_eur': [41.81, 353.29, 9.301, 405.02, 535.42, 20.8, 9.381, 112.71, 0, 777.56,
                      14.66, 8.399, 490, 8.399, 7.184, 435.24, 10.6122, 150.2, 428.5, 6.906,
                      14.02, 159.24, 51.54, 9.86, 49.2, 55.65, 407.51, 6.756, 43.4, 165,
                      6.364, 398.7],
        'amount': [10460.86, -17285.03, -10376.6, -25843.49, -57651.88, 4999.49, 3110.73, -6742.6, 0, 4680.38,
                   5146, 2610.29, 3935, 3021.84, 3017.91, 3932.16, 25000, 300.4, 4300, 2936.24,
                   -12025.12, 600.33, 180.39, 9.86, 357.68, 1124.13, 4090.1, 3014.66, 130.2, 474,
                   3018.81, 4002]
    })

@pytest.fixture
def asset_info():
    """Create sample asset information data."""
    return pl.DataFrame({
        'name': ['Nordea Bank Oyj', 'USA Indeksirahasto', 'Tanska Indeksirahasto', 'Salesforce', 'Kemira Oyj',
                'Microsoft', 'iShares Core S&P 500 UCITS ETF USD (Acc)', 'Nordea Korko A', 'BAE Systems PLC',
                'NVIDIA', 'iShares MSCI India UCITS ETF USD (Acc)', 'Sampo Oyj'],
        'asset_id': [2008, 1007, 1008, 1013, 2010, 2009, 2001, 2015, 3000, 3001, 2000, 1012],
        'currency': ['EUR', 'EUR', 'DKK', 'USD', 'EUR', 'USD', 'EUR', 'EUR', 'EUR', 'USD', 'EUR', 'EUR'],
        'instrument': ['Stock', 'Mutual fund', 'Mutual fund', 'Stock', 'Stock', 'Stock', 'ETF', 'Mutual fund', 'Stock', 'Stock', 'ETF', 'Stock'],
        'geographical_area': ['Finland', 'North America', 'Denmark', 'North America', 'Finland', 'USA', 'USA', 'Europe', 'World', 'USA', 'India', 'Finland'],
        'industry': ['Finance', 'General index', 'General index', 'IT and consulting', 'Chemicals industry', 'IT', 'General index', 'General index', 'Defense industry', 'Technology', 'General index', 'Insurance']
    })

@pytest.fixture
def asset_owner():
    """Create sample asset owner data."""
    return pl.DataFrame({
        'name': ['Nordea Bank Oyj', 'Sampo Oyj', 'USA Indeksirahasto', 'Tanska Indeksirahasto', 'Salesforce',
                'iShares MSCI India UCITS ETF USD (Acc)', 'iShares Core S&P 500 UCITS ETF USD (Acc)', 'Kemira Oyj',
                'Microsoft', 'Nordea Korko A', 'BAE Systems PLC', 'NVIDIA', 'iShares Core S&P 500 UCITS ETF USD (Acc)',
                'iShares MSCI India UCITS ETF USD (Acc)', 'iShares MSCI India UCITS ETF USD (Acc)', 'Sampo Oyj'],
        'asset_id': [2008, 1012, 1007, 1008, 1013, 2000, 2001, 2010, 2009, 2015, 3000, 3001, 2001, 2000, 2000, 1012],
        'owner_id': [10, 10, 10, 10, 10, 10, 20, 20, 20, 20, 30, 30, 30, 20, 30, 20],
        'bank': ['Nordea', 'Nordea', 'Nordnet', 'Nordnet', 'Nordea', 'Nordnet', 'Nordnet', 'Nordnet', 'Nordnet',
                'Nordea', 'Nordnet', 'Nordnet', 'Nordnet', 'Nordnet', 'Nordnet', 'Nordnet'],
        'account': ['Osakesäästötili', 'Osakesäästötili', 'Arvo-osuustili', 'Arvo-osuustili', 'Osakesäästötili',
                   'Arvo-osuustili', 'Arvo-osuustili', 'Arvo-osuustili', 'Arvo-osuustili', 'Arvo-osuustili',
                   'Arvo-osuustili', 'Arvo-osuustili', 'Arvo-osuustili', 'Arvo-osuustili', 'Arvo-osuustili',
                   'Osakesäästötili']
    })

@pytest.fixture
def price_data():
    """Create sample price data."""
    return pl.DataFrame({
        'Nordea Bank Oyj': [41.81, 42.50, 41.95],
        'Sampo Oyj': [398.7, 400.2, 399.5],
        'USA Indeksirahasto': [353.29, 355.0, 354.2],
        'Tanska Indeksirahasto': [9.301, 9.45, 9.38],
        'Salesforce': [405.02, 408.5, 407.3],
        'iShares MSCI India UCITS ETF USD (Acc)': [10.61, 10.75, 10.68],
        'iShares Core S&P 500 UCITS ETF USD (Acc)': [9.381, 9.45, 9.42],
        'Kemira Oyj': [535.42, 538.0, 536.5],
        'Microsoft': [407.51, 410.2, 409.3],
        'Nordea Korko A': [6.756, 6.78, 6.77],
        'BAE Systems PLC': [43.4, 43.8, 43.6],
        'NVIDIA': [398.7, 402.5, 400.8],
        'date': [datetime(2023, 2, 1), datetime(2023, 2, 2), datetime(2023, 2, 3)]
    })

# def test_calculate_monthly_positions(analyzer, asset_info, asset_owner, asset_transactions, asset_ids):
#     """Test to print out the sample data fixtures."""
#     print("\nAsset Info DataFrame:")
#     print(asset_info)
    
#     print("\nAsset Owner DataFrame:")  
#     print(asset_owner)

#     print("\nAsset Transactions DataFrame:")
#     print(sample_transactions)

#     print("\nAsset IDs DataFrame:")
#     print(asset_ids)

#     assert 1 == 2


# def test_calculate_monthly_positions(analyzer, sample_transactions, sample_assets):
#     """Test calculation of monthly positions."""
#     # Setup test data
#     analyzer.assets_df = sample_assets
#     analyzer.transactions_df = sample_transactions.with_columns(
#         pl.col('date').str.strptime(pl.Date, format='%Y-%m-%d')
#     )
    
#     analyzer.price_data = pl.DataFrame({
#         'Stock A': [150.0, 155.0, 160.0],
#         'Stock B': [200.0, 205.0, 210.0]
#     }).with_columns(date=pl.date_range(start=datetime(2023, 2, 1), end=datetime(2023, 2, 3), interval="1d"))
    
#     # Execute test
#     positions = analyzer.calculate_monthly_positions()
    
#     # Verify results
#     assert positions is not None
#     assert isinstance(positions, pl.DataFrame)
#     assert len(positions) > 0, "Positions DataFrame should not be empty"
#     assert all(col in positions.columns for col in ['Stock A', 'Stock B']), "All stock columns should be present"
    
    # # Get the last day of February 2023
    # feb_end = pl.lit("2023-02-28").str.strptime(pl.Date, format='%Y-%m-%d')
    # if feb_end in positions['Date']:
    #     assert positions.filter(pl.col('Date') == feb_end)['Stock A'][0] == 100
    #     assert positions.filter(pl.col('Date') == feb_end)['Stock B'][0] == 50
    # else:
    #     # If February end is not in the index, check the first available date
    #     first_date = positions['Date'][0]
    #     assert positions.filter(pl.col('Date') == first_date)['Stock A'][0] == 100
    #     assert positions.filter(pl.col('Date') == first_date)['Stock B'][0] == 50


# @patch('src.portfolio_analyzer.PostgresConnector')
# def test_fetch_portfolio_data(mock_db, analyzer, sample_assets, sample_transactions):
#     """Test fetching portfolio data from database."""
#     # Setup mock database connector
#     mock_instance = MagicMock()
#     mock_instance.get_active_assets.return_value = sample_assets
#     mock_instance.get_portfolio_transactions.return_value = sample_transactions
#     mock_db.return_value.__enter__.return_value = mock_instance
    
#     # Execute test
#     analyzer.fetch_portfolio_data()
    
#     # # Verify results
#     # assert analyzer.assets_df is not None
#     # assert analyzer.transactions_df is not None
#     # assert len(analyzer.assets_df) == 2
#     # assert len(analyzer.transactions_df) == 2



# def test_calculate_portfolio_proportions(analyzer):
#     """Test calculation of portfolio proportions."""
#     # Setup test data
#     test_date = pl.lit("2023-02-01").str.strptime(pl.Date, format='%Y-%m-%d')
    
#     # Setup assets data
#     analyzer.assets_df = pl.DataFrame({
#         'name': ['Stock A', 'Stock B'],
#         'asset_id': [1, 2],
#         'yahoo_ticker': ['AAPL', 'MSFT']
#     })
    
#     positions = pl.DataFrame({
#         'Date': [test_date],
#         'Stock A': [100],
#         'Stock B': [50]
#     })
    
#     # Setup price data with matching date
#     analyzer.price_data = pl.DataFrame({
#         'date': [test_date],
#         'Stock A': [150.0],
#         'Stock B': [200.0]
#     })
    
#     # Execute test
#     proportions = analyzer.calculate_portfolio_proportions(positions)
    
#     # Calculate expected proportions
#     total_value = 100 * 150.0 + 50 * 200.0  # Stock A value + Stock B value
#     expected_stock_a = (100 * 150.0 / total_value) * 100  # Convert to percentage
#     expected_stock_b = (50 * 200.0 / total_value) * 100   # Convert to percentage
    
    # # Verify results
    # assert proportions is not None
    # assert isinstance(proportions, pl.DataFrame)
    # assert abs(proportions.select(pl.exclude('Date')).row(0).sum() - 100.0) < 0.01
    # assert abs(proportions['Stock A'][0] - expected_stock_a) < 0.01
    # assert abs(proportions['Stock B'][0] - expected_stock_b) < 0.01

# def test_invalid_start_date():
#     """Test handling of invalid start date."""
#     with pytest.raises(Exception):
#         PortfolioAnalyzer(owner_id=10, start_date="invalid-date")

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