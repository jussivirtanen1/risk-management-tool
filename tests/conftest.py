import pytest
import os

@pytest.fixture(autouse=True)
def set_test_environment():
    """Automatically set test environment for all tests"""
    os.environ['ENVIRONMENT'] = 'test'