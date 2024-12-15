# src/db_connector.py

import psycopg2
import pandas as pd
from typing import Optional, Dict, Any
import os
from dotenv import load_dotenv

class PostgresConnector:
    def __init__(self, **kwargs):
        """
        Initialize PostgreSQL database connection parameters.
        Loads from environment variables by default, but allows override through kwargs
        """
        load_dotenv()  # Load environment variables from .env file
        
        # Default configuration from environment variables
        self.config = {
            "dbname": os.getenv("DB_NAME", "am_db_test"),
            "user": os.getenv("DB_USER", "docker_app"),
            "password": os.getenv("DB_PASSWORD"),
            "host": os.getenv("DB_HOST", "localhost"),
            "port": os.getenv("DB_PORT", "5432")
        }
        
        # Override with any provided kwargs
        self.config.update(kwargs)
        
        self.schema = "asset_management_test"
        self.conn = None
        self.cur = None

    def connect(self) -> bool:
        """
        Establish connection to the PostgreSQL database.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            self.conn = psycopg2.connect(**self.config)
            self.cur = self.conn.cursor()
            
            # Set the schema for this connection
            self.cur.execute(f"SET search_path TO {self.schema}")
            
            return True
        except Exception as e:
            print(f"Error connecting to database: {e}")
            return False

    def fetch_data(self, query: str, params: Optional[Dict[str, Any]] = None) -> Optional[pd.DataFrame]:
        """
        Execute a SELECT query and return results as a pandas DataFrame.
        
        Args:
            query (str): SQL query to execute
            params (dict, optional): Parameters for the SQL query
            
        Returns:
            Optional[pd.DataFrame]: Query results as DataFrame or None if query fails
        """
        try:
            if not self.conn or not self.cur:
                if not self.connect():
                    return None
            
            self.cur.execute(query, params)
            columns = [desc[0] for desc in self.cur.description]
            data = self.cur.fetchall()
            
            return pd.DataFrame(data, columns=columns)
            
        except Exception as e:
            print(f"Error executing query: {e}")
            return None

    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> bool:
        """
        Execute a query without returning results (INSERT, UPDATE, DELETE).
        
        Args:
            query (str): SQL query to execute
            params (dict, optional): Parameters for the SQL query
            
        Returns:
            bool: True if query executed successfully, False otherwise
        """
        try:
            if not self.conn or not self.cur:
                if not self.connect():
                    return False
            
            self.cur.execute(query, params)
            self.conn.commit()
            return True
            
        except Exception as e:
            print(f"Error executing query: {e}")
            if self.conn:
                self.conn.rollback()
            return False

    def get_asset_info(self, asset_id: Optional[int] = None) -> Optional[pd.DataFrame]:
        """
        Fetch asset information from asset_info table.
        
        Args:
            asset_id (int, optional): Specific asset ID to fetch
            
        Returns:
            Optional[pd.DataFrame]: Asset information
        """
        query = "SELECT * FROM asset_management_test.asset_info"
        if asset_id is not None:
            query += f" WHERE asset_id = {asset_id}"
        return self.fetch_data(query)

    def get_asset_transactions(self, 
                             asset_id: Optional[int] = None,
                             start_date: Optional[str] = None,
                             end_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Fetch asset transactions with optional filters.
        
        Args:
            asset_id (int, optional): Specific asset ID to fetch
            start_date (str, optional): Start date for transaction filter (YYYY-MM-DD)
            end_date (str, optional): End date for transaction filter (YYYY-MM-DD)
            
        Returns:
            Optional[pd.DataFrame]: Transaction data
        """
        query = "SELECT * FROM asset_management_test.asset_transactions"
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

    def get_asset_owners(self, asset_id: Optional[int] = None) -> Optional[pd.DataFrame]:
        """
        Fetch asset ownership information.
        
        Args:
            asset_id (int, optional): Specific asset ID to fetch
            
        Returns:
            Optional[pd.DataFrame]: Asset ownership data
        """
        query = "SELECT * FROM asset_management_test.asset_owner"
        if asset_id is not None:
            query += f" WHERE asset_id = {asset_id}"
        return self.fetch_data(query)

    def get_asset_ids(self, asset_id: Optional[int] = None) -> Optional[pd.DataFrame]:
        """
        Fetch asset identification information.
        
        Args:
            asset_id (int, optional): Specific asset ID to fetch
            
        Returns:
            Optional[pd.DataFrame]: Asset identification data
        """
        query = "SELECT * FROM asset_management_test.asset_ids"
        if asset_id is not None:
            query += f" WHERE asset_id = {asset_id}"
        return self.fetch_data(query)

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