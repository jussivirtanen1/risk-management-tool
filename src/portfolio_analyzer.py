"""
Portfolio Analysis Module

Analyzes portfolio composition over time using transaction data and market prices.
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, date
from pathlib import Path
import os
from typing import List, Tuple, Optional
from odf.text import P
from odf.opendocument import OpenDocumentSpreadsheet
from odf.table import Table, TableRow, TableCell
from src.db_connector import PostgresConnector

class PortfolioAnalyzer:
    def __init__(self, owner_id: int, start_date: str = "2020-01-01"):
        """
        Initialize Portfolio Analyzer.
        
        Args:
            owner_id: ID of the portfolio owner
            start_date: Start date for analysis
        """
        self.owner_id = owner_id
        # Convert start_date to pandas Timestamp immediately
        try:
            self.start_date = pd.to_datetime(start_date)
            print(f"[INIT] Start date set to: {self.start_date}")
        except Exception as e:
            print(f"[INIT ERROR] Invalid start_date format: {start_date} | Error: {e}")
            raise e
        self.assets_df = None
        self.transactions_df = None
        self.price_data = None
        self.name_ticker_map = {}  # Mapping from asset name to Yahoo ticker

    @staticmethod
    def get_analysis_path() -> str:
        """Get the path to analysis directory."""
        if os.path.exists('/.dockerenv'):
            return '/app/analysis'
        else:
            return str(Path.home() / "Desktop" / "analysis")

    def get_asset_query(self) -> str:
        """Get SQL query for fetching asset information."""
        return f"""
            SELECT 
                id.name,
                id.asset_id,
                id.yahoo_ticker,
                id.yahoo_fx_ticker,
                info.instrument
            FROM asset_management_test.asset_ids AS id
            LEFT JOIN asset_management_test.asset_owner AS own 
                ON id.asset_id = own.asset_id
            LEFT JOIN asset_management_test.asset_info AS info 
                ON id.asset_id = info.asset_id
            WHERE owner_id = {self.owner_id}
        """

    def get_transactions_query(self) -> str:
        """Get SQL query for fetching transaction data."""
        return f"""
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
            FROM asset_management_test.asset_transactions
            WHERE owner_id = {self.owner_id}
        """

    def fetch_portfolio_data(self) -> None:
        """Fetch asset and transaction data from database."""
        print("[FETCH_PORTFOLIO_DATA] Fetching portfolio data from database...")
        with PostgresConnector() as db:
            self.assets_df = db.fetch_data(self.get_asset_query())
            self.transactions_df = db.fetch_data(self.get_transactions_query())
        
        print(f"[FETCH_PORTFOLIO_DATA] Assets DataFrame shape: {self.assets_df.shape}")
        print(f"[FETCH_PORTFOLIO_DATA] Transactions DataFrame shape: {self.transactions_df.shape}")
        print(f"[FETCH_PORTFOLIO_DATA] Assets DataFrame columns: {self.assets_df.columns.tolist()}")
        print(f"[FETCH_PORTFOLIO_DATA] Transactions DataFrame columns: {self.transactions_df.columns.tolist()}")
        print(f"[FETCH_PORTFOLIO_DATA] Assets DataFrame head:\n{self.assets_df.head()}")
        print(f"[FETCH_PORTFOLIO_DATA] Transactions DataFrame head:\n{self.transactions_df.head()}")

    def fetch_market_data(self) -> None:
        """Fetch market data for the portfolio assets."""
        print("[FETCH_MARKET_DATA] Fetching market data from Yahoo Finance...")
        tickers = self.assets_df['yahoo_ticker'].dropna().unique().tolist()
        self.name_ticker_map = dict(zip(self.assets_df['name'], self.assets_df['yahoo_ticker']))
        
        if not tickers:
            raise ValueError("No tickers found in assets data.")

        # Fetch historical market data
        self.price_data = yf.download(
            tickers, 
            start=self.start_date.strftime("%Y-%m-%d"), 
            progress=False
        )['Adj Close']
        
        if isinstance(self.price_data, pd.Series):
            self.price_data = self.price_data.to_frame()
        
        # Rename columns to asset names for consistency
        self.price_data.rename(columns=self.name_ticker_map, inplace=True)
        print(f"[FETCH_MARKET_DATA] Market data fetched for tickers: {tickers}")

    def calculate_monthly_positions(self) -> Tuple[List[pd.Timestamp], pd.DataFrame]:
        """
        Calculate monthly positions considering transaction timing.
        
        Returns:
            Tuple of (monthly_dates, positions_df)
        """
        print("[CALCULATE_MONTHLY_POSITIONS] Calculating monthly positions...")
        if self.transactions_df is None or self.price_data is None:
            raise ValueError("Must fetch both transaction and market data first")

        # Merge transactions with asset data to get asset names
        merged_df = self.transactions_df.merge(
            self.assets_df[['asset_id', 'name', 'yahoo_ticker']],
            on='asset_id',
            how='left',
            suffixes=('_trans', '_asset')
        )
        print("[CALCULATE_MONTHLY_POSITIONS] Merged DataFrame columns:")
        print(merged_df.columns.tolist())

        # Ensure 'name' column exists
        if 'name' not in merged_df.columns:
            print("[CALCULATE_MONTHLY_POSITIONS ERROR] 'name' column is missing after merge.")
            raise KeyError("'name' column is missing in merged DataFrame.")

        # Sort transactions by date
        merged_df.sort_values(by='date', inplace=True)
        print("[CALCULATE_MONTHLY_POSITIONS] Transactions sorted by date.")

        # Initialize a DataFrame to hold positions
        positions_dict = {}
        current_positions = {}

        # Iterate over each transaction
        for _, transaction in merged_df.iterrows():
            asset = transaction['name']
            quantity = transaction['quantity']
            date = transaction['date']
            
            # Update current positions
            current_positions[asset] = current_positions.get(asset, 0) + quantity
            
            # Record positions at the transaction date
            positions_dict[date] = current_positions.copy()
            print(f"[CALCULATE_MONTHLY_POSITIONS] Updated positions on {date}: {current_positions}")

        # Convert positions_dict to DataFrame
        positions_df = pd.DataFrame(positions_dict).T
        positions_df.fillna(0, inplace=True)
        positions_df = positions_df.sort_index()
        print(f"[CALCULATE_MONTHLY_POSITIONS] Positions DataFrame shape: {positions_df.shape}")
        print(f"[CALCULATE_MONTHLY_POSITIONS] Positions DataFrame head:\n{positions_df.head()}")

        # Resample to monthly frequency by taking the last known position each month
        positions_df = positions_df.resample('M').last().fillna(method='ffill').fillna(0)
        monthly_dates = positions_df.index.tolist()
        print(f"[CALCULATE_MONTHLY_POSITIONS] Resampled to monthly frequency.")

        return monthly_dates, positions_df

    def calculate_portfolio_proportions(self, positions: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the proportion of each asset in the portfolio over time.
        
        Args:
            positions: DataFrame of asset positions over time
            
        Returns:
            DataFrame of portfolio proportions
        """
        print("[CALCULATE_PORTFOLIO_PROPORTIONS] Calculating portfolio proportions...")
        total_positions = positions.sum(axis=1)
        proportions = positions.divide(total_positions, axis=0).fillna(0)
        print(f"[CALCULATE_PORTFOLIO_PROPORTIONS] Proportions DataFrame shape: {proportions.shape}")
        print(f"[CALCULATE_PORTFOLIO_PROPORTIONS] Proportions DataFrame head:\n{proportions.head()}")
        return proportions

    def export_to_ods(self, df: pd.DataFrame) -> str:
        """
        Export DataFrame to .ods file.
        
        Args:
            df: DataFrame to export
            
        Returns:
            Path to saved file
        """
        print("[EXPORT_TO_ODS] Exporting DataFrame to .ods file...")
        
        # Initialize the ODS document and add a table
        doc = OpenDocumentSpreadsheet()
        table = Table(name="Portfolio Proportions")
        doc.spreadsheet.addElement(table)
        
        # Add header row
        header_row = TableRow()
        
        # Date header cell
        date_header = TableCell(valuetype="string")
        date_header.addElement(P(text="Date"))
        header_row.addElement(date_header)
        
        # Asset headers
        for col in df.columns:
            header_cell = TableCell(valuetype="string")
            header_cell.addElement(P(text=str(col)))
            header_row.addElement(header_cell)
        
        table.addElement(header_row)
        print("[EXPORT_TO_ODS] Header row added.")
        
        # Add data rows
        for idx, row in df.iterrows():
            tr = TableRow()
            
            # Add Date Cell
            date_cell = TableCell(valuetype="string")
            if isinstance(idx, pd.Timestamp):
                date_str = idx.strftime("%Y-%m-%d")
            else:
                date_str = str(idx)
            date_cell.addElement(P(text=date_str))
            tr.addElement(date_cell)
            
            # Add Portfolio Proportions Cells
            for value in row:
                if pd.isna(value):
                    numeric_value = 0.00
                    print(f"[EXPORT_TO_ODS WARNING] Found NaN value for date {date_str}. Replacing with 0.00.")
                else:
                    # Round to two decimal places
                    numeric_value = round(float(value), 2)
                    print(f"[EXPORT_TO_ODS INFO] Writing value {numeric_value} for date {date_str}.")
                
                # Create a cell with numerical value
                cell = TableCell(valuetype="float", value=str(numeric_value))
                
                # Note: No need to add Number or P elements for numerical cells
                tr.addElement(cell)
            
            table.addElement(tr)
        
        print("[EXPORT_TO_ODS] Data rows added.")
        
        # Define filename and path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"portfolio_proportions_{timestamp}.ods"
        output_path = self.get_analysis_path()
        os.makedirs(output_path, exist_ok=True)
        full_path = os.path.join(output_path, filename)
        
        # Save the document
        try:
            doc.save(full_path)
            print(f"[EXPORT_TO_ODS] File saved to: {full_path}")
            return full_path
        except Exception as e:
            print(f"[EXPORT_TO_ODS ERROR] Failed to export to .ods: {e}")
            raise e

    def analyze(self) -> Optional[pd.DataFrame]:
        """
        Run the complete portfolio analysis.
        
        Returns:
            DataFrame with portfolio proportions or None if analysis fails
        """
        print("[ANALYZE] Starting portfolio analysis...")
        try:
            # Fetch all required data
            self.fetch_portfolio_data()
            self.fetch_market_data()
            
            # Calculate positions and proportions
            monthly_dates, positions = self.calculate_monthly_positions()
            proportions = self.calculate_portfolio_proportions(positions)
            
            # Inspect proportions before exporting
            print(f"[ANALYZE] Proportions DataFrame shape: {proportions.shape}")
            print(f"[ANALYZE] Proportions DataFrame head:\n{proportions.head()}")

            # Export results
            if not proportions.empty:
                output_path = self.export_to_ods(proportions)
                print(f"[ANALYZE] Portfolio analysis exported to: {output_path}")
                return proportions
            else:
                print("[ANALYZE] No valid portfolio data to analyze")
                return None
                
        except Exception as e:
            print(f"[ANALYZE ERROR] Error during portfolio analysis: {e}")
            return None

def main(owner_id: int = 10, start_date: str = "2020-01-01") -> Optional[pd.DataFrame]:
    """
    Main function to run portfolio analysis.
    
    Args:
        owner_id: ID of the portfolio owner
        start_date: Start date for analysis
        
    Returns:
        DataFrame with portfolio proportions or None if analysis fails
    """
    analyzer = PortfolioAnalyzer(owner_id, start_date)
    return analyzer.analyze()

if __name__ == "__main__":
    main(owner_id=10, start_date="2020-01-01")