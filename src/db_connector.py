import psycopg
import polars as pl
from typing import Optional, Dict, Any
import os
from dotenv import load_dotenv

class PostgresConnector:
    def __init__(self, **kwargs):
        """
        Initialize PostgreSQL database connection parameters.
        Loads from environment variables by default, but allows override through kwargs
        """
        # Get DB_PARAM and remove any leading underscore
        self.db_param = os.getenv('DB_PARAM', 'test').lstrip('_')
        
        # Load the appropriate environment file
        env_file = f".env.{self.db_param}"
        load_dotenv(env_file)
        
        # Set database name and schema based on environment
        default_db_name = f"am_db_{self.db_param}"
        self.schema = f"asset_management_{self.db_param}"
        
        # Default configuration from environment variables
        self.conn_string = os.getenv("DATABASE_URL")


    def _check_test_protection(self):
        """Check if we're trying to run tests in production environment."""
        if self.db_param == "_prod" and os.getenv('PYTEST_CURRENT_TEST'):
            raise RuntimeError(
                "ERROR: Attempting to run tests in production environment. "
                "This is not allowed to protect production data."
            )

    def connect(self) -> bool:
        """
        Establish connection to the PostgreSQL database.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            # Check for test protection
            self._check_test_protection()
            
            self.conn = psycopg.connect(**self.conn_string)
            self.cur = self.conn.cursor()
            
            # Set the schema for this connection
            self.cur.execute(f"SET search_path TO {self.schema}")
            
            return True
        except Exception as e:
            print(f"Error connecting to database: {e}")
            return False

    def fetch_data(self, query: str, params: Optional[Dict[str, Any]] = None) -> Optional[pl.DataFrame]:
        """
        Execute a SELECT query and return results as a pandas DataFrame.
        
        Args:
            query (str): SQL query to execute
            params (dict, optional): Parameters for the SQL query
            
        Returns:
            Optional[pl.DataFrame]: Query results as DataFrame or None if query fails
        """
        try:
            # Check for test protection #TODO: check is test protection works correctly
            self._check_test_protection()            
            self.cur.execute(query, params)
            columns = [desc[0] for desc in self.cur.description]
            data = self.cur.fetchall()
            return pl.DataFrame(data, schema=columns, orient="row")
            
        except Exception as e:
            print(f"Database error: {e}")
            return None

    def get_asset_info(self, asset_id: Optional[int] = None) -> Optional[pl.DataFrame]:
        """
        Fetch asset information from asset_info table.
        
        Args:
            asset_id (int, optional): Specific asset ID to fetch
            
        Returns:
            Optional[pl.DataFrame]: Asset information
        """
        query = f"SELECT * FROM {self.schema}.asset_info"
        if asset_id is not None:
            query += f" WHERE asset_id = {asset_id}"
        return self.fetch_data(query)

    def get_asset_transactions(self, 
                             asset_id: Optional[int] = None,
                             start_date: Optional[str] = None,
                             end_date: Optional[str] = None) -> Optional[pl.DataFrame]:
        """
        Fetch asset transactions with optional filters.
        
        Args:
            asset_id (int, optional): Specific asset ID to fetch
            start_date (str, optional): Start date for transaction filter (YYYY-MM-DD)
            end_date (str, optional): End date for transaction filter (YYYY-MM-DD)
            
        Returns:
            Optional[pl.DataFrame]: Transaction data
        """
        query = f"SELECT * FROM {self.schema}.asset_transactions"
        conditions = []
        
        if asset_id is not None:
            conditions.append(f"asset_id = {asset_id}")
        if start_date is not None:
            conditions.append(f"date >= '{start_date}'")
        if end_date is not None:
            conditions.append(f"date <= '{end_date}'")
            
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
            
        return self.fetch_data(query)

    def get_asset_owners(self, asset_id: Optional[int] = None) -> Optional[pl.DataFrame]:
        """
        Fetch asset ownership information.
        
        Args:
            asset_id (int, optional): Specific asset ID to fetch
            
        Returns:
            Optional[pl.DataFrame]: Asset ownership data
        """
        query = f"SELECT * FROM {self.schema}.asset_owner"
        if asset_id is not None:
            query += f" WHERE asset_id = {asset_id}"
        return self.fetch_data(query)

    def get_asset_ids(self, asset_id: Optional[int] = None) -> Optional[pl.DataFrame]:
        """
        Fetch asset identification information.
        
        Args:
            asset_id (int, optional): Specific asset ID to fetch
            
        Returns:
            Optional[pl.DataFrame]: Asset identification data
        """
        query = f"SELECT * FROM {self.schema}.asset_ids"
        if asset_id is not None:
            query += f" WHERE asset_id = {asset_id}"
        return self.fetch_data(query)

    def get_portfolio_assets(self, owner_id: int) -> Optional[pl.DataFrame]:
        """Get SQL query for fetching asset information for a specific owner."""
        query = f"""
            SELECT 
                id.name,
                id.asset_id,
                id.yahoo_ticker,
                id.yahoo_fx_ticker,
                info.instrument
            FROM {self.schema}.asset_ids AS id
            LEFT JOIN {self.schema}.asset_owner AS own 
                ON id.asset_id = own.asset_id
            LEFT JOIN {self.schema}.asset_info AS info 
                ON id.asset_id = info.asset_id
            WHERE owner_id = {owner_id}
        """
        return self.fetch_data(query)

    def get_portfolio_transactions(self, owner_id: int) -> Optional[pl.DataFrame]:
        """Get SQL query for fetching transaction data for a specific owner."""
        query = f"""
            SELECT 
                event_type,
                asset_id,
                owner_id,
                name,
                date,
                quantity,
                price_fx,
                price_eur,
                amount
            FROM {self.schema}.asset_transactions
            WHERE owner_id = {owner_id}
            ORDER BY date ASC
        """
        return self.fetch_data(query)

    def get_active_assets(self, owner_id: int) -> Optional[pl.DataFrame]:
        """Get assets with quantity > 0 for a specific owner."""
        query = f"""
            WITH current_positions AS (
                SELECT 
                    asset_id,
                    SUM(quantity) as total_quantity
                FROM {self.schema}.asset_transactions
                WHERE owner_id = {owner_id}
                GROUP BY asset_id
                HAVING SUM(quantity) > 0
            )
            SELECT 
                id.name,
                id.asset_id,
                id.yahoo_ticker,
                cp.total_quantity,
                id.yahoo_fx_ticker
            FROM current_positions cp
            JOIN {self.schema}.asset_ids id ON cp.asset_id = id.asset_id
            WHERE id.yahoo_ticker IS NOT NULL 
            AND id.yahoo_ticker != ''
            ORDER BY id.asset_id;  -- Add ordering for easier comparison
        """
        
        result = self.fetch_data(query)
        
        assert isinstance(result, pl.DataFrame), f"Expected Polars DataFrame but got {type(result)}"
        print(f"Found {len(result)} active assets")
        
        return result

    def close(self):
        """Close database connection and cursor."""
        if self.cur:
            self.cur.close()
            self.cur = None
        if self.conn:
            self.conn.close()
            self.conn = None

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()