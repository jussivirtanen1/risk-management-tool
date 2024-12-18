# tests/test_db_connector.py

import pytest
from src.db_connector import PostgresConnector
import pandas as pd
from datetime import date
from unittest.mock import patch, MagicMock
import os

@pytest.fixture
def db_connector():
    """Create a database connector instance for testing"""
    return PostgresConnector()

@pytest.fixture
def test_asset_data():
    """Sample test data for asset_info table"""
    return {
        'name': 'Test Stock',
        'asset_id': 1001,
        'currency': 'EUR',
        'instrument': 'STOCK',
        'geographical_area': 'EUROPE',
        'industry': 'TECHNOLOGY'
    }

@pytest.fixture
def test_asset_id_data():
    """Sample test data for asset_ids table"""
    return {
        'name': 'Test Stock',
        'asset_id': 1001,
        'yahoo_ticker': 'TEST.DE',
        'yahoo_fx_ticker': 'EURUSD=X',
        'isin': 'US0000000001'
    }

@pytest.fixture
def test_asset_owner_data():
    """Sample test data for asset_owner table"""
    return {
        'name': 'Test Stock',
        'asset_id': 1001,
        'owner_id': 1,
        'bank': 'Test Bank',
        'account': 'Test Account'
    }

@pytest.fixture
def test_transaction_data():
    """Sample test data for asset_transactions table"""
    return {
        'event_type': 'BUY',
        'asset_id': 1001,
        'owner_id': 1,
        'name': 'Test Stock',
        'date': date(2024, 1, 1),
        'quantity': 100.0,
        'price_fx': 150.0,
        'price_eur': 140.0,
        'amount': 14000.0
    }

def test_connection_success(db_connector):
    """Test successful database connection"""
    assert db_connector.connect() is True
    assert db_connector.conn is not None
    assert db_connector.cur is not None

def test_fetch_asset_info(db_connector):
    """Test fetching data from asset_info table"""
    query = f"""
    SELECT * FROM {db_connector.schema}.asset_info 
    LIMIT 5
    """
    df = db_connector.fetch_data(query)
    
    assert df is not None
    assert isinstance(df, pd.DataFrame)
    assert all(col in df.columns for col in [
        'name', 'asset_id', 'currency', 'instrument', 
        'geographical_area', 'industry'
    ])

def test_fetch_asset_ids(db_connector):
    """Test fetching data from asset_ids table"""
    query = f"""
    SELECT * FROM {db_connector.schema}.asset_ids 
    LIMIT 5
    """
    df = db_connector.fetch_data(query)
    
    assert df is not None
    assert isinstance(df, pd.DataFrame)
    assert all(col in df.columns for col in [
        'name', 'asset_id', 'yahoo_ticker', 
        'yahoo_fx_ticker', 'isin'
    ])

def test_fetch_asset_owner(db_connector):
    """Test fetching data from asset_owner table"""
    query = f"""
    SELECT * FROM {db_connector.schema}.asset_owner 
    LIMIT 5
    """
    df = db_connector.fetch_data(query)
    
    assert df is not None
    assert isinstance(df, pd.DataFrame)
    assert all(col in df.columns for col in [
        'name', 'asset_id', 'owner_id', 'bank', 'account'
    ])

def test_fetch_asset_transactions(db_connector):
    """Test fetching data from asset_transactions table"""
    query = f"""
    SELECT * FROM {db_connector.schema}.asset_transactions 
    LIMIT 5
    """
    df = db_connector.fetch_data(query)
    
    assert df is not None
    assert isinstance(df, pd.DataFrame)
    assert all(col in df.columns for col in [
        'event_type', 'asset_id', 'owner_id', 'name', 'date',
        'quantity', 'price_fx', 'price_eur', 'amount'
    ])

def test_insert_asset_info(db_connector, test_asset_data):
    """Test inserting data into asset_info table"""
    query = f"""
    INSERT INTO {db_connector.schema}.asset_info 
    (name, asset_id, currency, instrument, geographical_area, industry)
    VALUES (%(name)s, %(asset_id)s, %(currency)s, %(instrument)s, 
            %(geographical_area)s, %(industry)s)
    """
    success = db_connector.execute_query(query, test_asset_data)
    assert success is True

    # Verify insertion
    verify_query = f"""
    SELECT * FROM {db_connector.schema}.asset_info 
    WHERE asset_id = {test_asset_data['asset_id']}
    """
    df = db_connector.fetch_data(verify_query)
    assert len(df) > 0
    assert df.iloc[0]['name'] == test_asset_data['name']

def test_join_asset_tables(db_connector):
    """Test joining multiple asset tables"""
    query = f"""
    SELECT ai.name, ai.asset_id, ai.currency, ao.owner_id, ao.bank
    FROM {db_connector.schema}.asset_info ai
    JOIN {db_connector.schema}.asset_owner ao 
    ON ai.asset_id = ao.asset_id
    LIMIT 5
    """
    df = db_connector.fetch_data(query)
    
    assert df is not None
    assert isinstance(df, pd.DataFrame)
    assert all(col in df.columns for col in [
        'name', 'asset_id', 'currency', 'owner_id', 'bank'
    ])

def test_transaction_summary(db_connector):
    """Test transaction aggregation query"""
    query = f"""
    SELECT asset_id, 
           COUNT(*) as transaction_count,
           SUM(amount) as total_amount
    FROM {db_connector.schema}.asset_transactions
    GROUP BY asset_id
    LIMIT 5
    """
    df = db_connector.fetch_data(query)
    
    assert df is not None
    assert isinstance(df, pd.DataFrame)
    assert all(col in df.columns for col in [
        'asset_id', 'transaction_count', 'total_amount'
    ])

def test_cleanup(db_connector):
    """Test cleanup of test data"""
    cleanup_query = f"""
    DELETE FROM {db_connector.schema}.asset_info
    WHERE asset_id = 1001;
    """
    success = db_connector.execute_query(cleanup_query)
    assert success is True

def test_env_file_loading():
    """Test that the correct environment file is loaded for test environment."""
    with patch('src.db_connector.load_dotenv') as mock_load:
        db = PostgresConnector()
        mock_load.assert_called_once_with(".env")
        assert db.config["dbname"] == "am_db_test"
        assert db.schema == "asset_management_test"

def test_schema_selection():
    """Test that the test schema is selected."""
    db = PostgresConnector()
    assert db.schema == "asset_management_test"
    assert db.config["dbname"] == "am_db_test"

def test_connection_parameters():
    """Test that connection parameters are correctly set from environment variables."""
    test_params = {
        'DB_NAME': 'test_db',
        'DB_USER': 'test_user',
        'DB_PASSWORD': 'test_pass',
        'DB_HOST': 'test_host',
        'DB_PORT': '5433'
    }
    
    with patch.dict(os.environ, test_params):
        db = PostgresConnector()
        assert db.config['dbname'] == 'test_db'
        assert db.config['user'] == 'test_user'
        assert db.config['password'] == 'test_pass'
        assert db.config['host'] == 'test_host'
        assert db.config['port'] == '5433'

def test_connection_parameter_override():
    """Test that connection parameters can be overridden through kwargs."""
    db = PostgresConnector(dbname='override_db', user='override_user')
    assert db.config['dbname'] == 'override_db'
    assert db.config['user'] == 'override_user'