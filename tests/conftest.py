import pytest
import os
from typing import List

@pytest.fixture(autouse=True)
def set_test_environment():
    """Automatically set test environment for all tests"""
    os.environ['ENVIRONMENT'] = 'test'

@pytest.fixture
def test_owner_ids() -> List[int]:
    """Fixture providing test owner IDs."""
    return [10, 20, 30]

@pytest.fixture
def mock_portfolio_data(test_owner_ids):
    """Fixture providing mock portfolio data for multiple owners."""
    return {
        owner_id: {
            'assets': [
                {'asset_id': 1, 'name': f'Test Asset 1 Owner {owner_id}', 'yahoo_ticker': 'TEST1'},
                {'asset_id': 2, 'name': f'Test Asset 2 Owner {owner_id}', 'yahoo_ticker': 'TEST2'}
            ],
            'transactions': [
                {'date': '2023-01-01', 'quantity': 100, 'asset_id': 1},
                {'date': '2023-02-01', 'quantity': 50, 'asset_id': 2}
            ]
        }
        for owner_id in test_owner_ids
    }