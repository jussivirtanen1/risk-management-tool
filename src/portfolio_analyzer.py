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
    def __init__(self, owner_id: int, start_date: str = "2023-01-01"):
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

    def fetch_portfolio_data(self) -> None:
        """Fetch asset and transaction data from database."""
        print("[FETCH_PORTFOLIO_DATA] Fetching portfolio data from database...")
        with PostgresConnector() as db:
            self.assets_df = db.get_portfolio_assets(self.owner_id)
            # Add date filter to transactions query
            self.transactions_df = db.get_portfolio_transactions(self.owner_id)
            if self.transactions_df is not None:
                # Convert date column to datetime
                self.transactions_df['date'] = pd.to_datetime(self.transactions_df['date'])
                # Filter transactions based on start_date
                self.transactions_df = self.transactions_df[self.transactions_df['date'] >= self.start_date]
                if self.transactions_df.empty:
                    print(f"[FETCH_PORTFOLIO_DATA] No transactions found after {self.start_date}")
        
        if self.assets_df is None or self.transactions_df is None:
            raise ValueError("Failed to fetch portfolio data from database")
        
        print(f"[FETCH_PORTFOLIO_DATA] Assets DataFrame shape: {self.assets_df.shape}")
        print(f"[FETCH_PORTFOLIO_DATA] Transactions DataFrame shape: {self.transactions_df.shape}")
        print(f"[FETCH_PORTFOLIO_DATA] Transaction date range: {self.transactions_df['date'].min()} to {self.transactions_df['date'].max()}")
        print(f"[FETCH_PORTFOLIO_DATA] Assets DataFrame head:\n{self.assets_df.head()}")
        print(f"[FETCH_PORTFOLIO_DATA] Transactions DataFrame head:\n{self.transactions_df.head()}")

        # Create a mapping from asset name to Yahoo ticker
        self.name_ticker_map = pd.Series(
            self.assets_df.yahoo_ticker.values,
            index=self.assets_df.name
        ).to_dict()
        print(f"[FETCH_PORTFOLIO_DATA] Asset Name to Ticker Mapping:\n{self.name_ticker_map}")

    def fetch_market_data(self) -> None:
        """Fetch market data from Yahoo Finance."""
        print("[FETCH_MARKET_DATA] Fetching market data from Yahoo Finance...")
        if self.assets_df is None:
            raise ValueError("Must fetch portfolio data before market data")
        
        # Extract Yahoo tickers from the name-ticker mapping
        tickers = list(self.name_ticker_map.values())
        # Remove any NaN or None tickers
        tickers = [ticker for ticker in tickers if isinstance(ticker, str) and ticker.strip() != '']
        print(f"[FETCH_MARKET_DATA] Tickers to fetch: {tickers}")
        
        if not tickers:
            raise ValueError("No valid tickers found to fetch market data")
            
        try:
            # Add some buffer days to ensure we have enough data
            start_date = self.start_date - pd.Timedelta(days=10)
            self.price_data = yf.download(tickers, start=start_date)['Close']
            if self.price_data.empty:
                raise ValueError("No price data returned from Yahoo Finance")
                
            self.price_data = self.price_data.ffill()  # Forward fill missing values
            print(f"[FETCH_MARKET_DATA] Price data fetched with shape: {self.price_data.shape}")
            print(f"[FETCH_MARKET_DATA] Price data date range: {self.price_data.index.min()} to {self.price_data.index.max()}")
            print(f"[FETCH_MARKET_DATA] Price data head:\n{self.price_data.head()}")
            
            # Rename price_data columns from tickers to asset names
            ticker_to_name = {v: k for k, v in self.name_ticker_map.items()}
            self.price_data.rename(columns=ticker_to_name, inplace=True)
            print(f"[FETCH_MARKET_DATA] Price data columns renamed to asset names:\n{self.price_data.columns.tolist()}")
        except Exception as e:
            print(f"[FETCH_MARKET_DATA ERROR] Failed to fetch price data: {e}")
            raise e

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

        if merged_df['name_asset'].isnull().any():
            print("[CALCULATE_MONTHLY_POSITIONS WARNING] Some transactions have missing asset names.")

        # Ensure all dates are in pandas datetime format
        merged_df['date'] = pd.to_datetime(merged_df['date'])
        self.price_data.index = pd.to_datetime(self.price_data.index)
        print("[CALCULATE_MONTHLY_POSITIONS] Converted transaction and price data dates to datetime.")

        # Group by month end ('ME') frequency
        monthly_last_dates = self.price_data.groupby(pd.Grouper(freq='ME')).last()
        valid_dates = monthly_last_dates[~monthly_last_dates.isnull().any(axis=1)].index
        print(f"[CALCULATE_MONTHLY_POSITIONS] Number of valid month-end dates: {len(valid_dates)}")

        # Initialize a dictionary to hold date: positions
        positions_dict = {}
        
        for monthly_date in valid_dates:
            # Keep monthly_date as pandas Timestamp
            valid_transactions = merged_df[
                merged_df['date'] <= monthly_date
            ]
            
            if not valid_transactions.empty:
                # Aggregate quantities by asset name (from assets_df)
                positions = valid_transactions.groupby('name_asset')['quantity'].sum()
                
                # Filter out positions with zero quantity
                positions = positions[positions != 0]
                
                if not positions.empty:
                    positions_dict[monthly_date] = positions
                    print(f"[CALCULATE_MONTHLY_POSITIONS] Added positions for date: {monthly_date.date()} | Positions: {positions.to_dict()}")
                else:
                    print(f"[CALCULATE_MONTHLY_POSITIONS] No non-zero positions for date: {monthly_date.date()}")
            else:
                print(f"[CALCULATE_MONTHLY_POSITIONS] No transactions up to date: {monthly_date.date()}")

        # Debugging: Print lengths
        print(f"[CALCULATE_MONTHLY_POSITIONS] Total valid_dates: {len(valid_dates)}")
        print(f"[CALCULATE_MONTHLY_POSITIONS] Total positions_dict entries: {len(positions_dict)}")
        
        # Ensure that positions_dict is not empty
        if not positions_dict:
            raise ValueError("No valid portfolio positions found for the given dates.")

        # Create DataFrame from the dictionary
        try:
            positions_df = pd.DataFrame.from_dict(positions_dict, orient='index').fillna(0)
            positions_df.index.name = 'Date'
            print(f"[CALCULATE_MONTHLY_POSITIONS] Positions DataFrame shape: {positions_df.shape}")
            print(f"[CALCULATE_MONTHLY_POSITIONS] Positions DataFrame head:\n{positions_df.head()}")
        except Exception as e:
            print(f"[CALCULATE_MONTHLY_POSITIONS ERROR] Failed to create positions DataFrame: {e}")
            raise e

        return list(positions_df.index), positions_df

    def calculate_portfolio_proportions(self, positions: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate portfolio proportions over time.
        
        Args:
            positions: DataFrame with asset positions
            
        Returns:
            DataFrame with portfolio proportions
        """
        print("[CALCULATE_PORTFOLIO_PROPORTIONS] Calculating portfolio proportions...")
        
        # Ensure that the price_data contains all asset names present in positions
        missing_assets = set(positions.columns) - set(self.price_data.columns)
        if missing_assets:
            print(f"[CALCULATE_PORTFOLIO_PROPORTIONS WARNING] Missing assets in price_data: {missing_assets}")
            # Optionally, fetch missing asset prices or handle accordingly
            # For simplicity, we'll add them with zero prices
            for asset in missing_assets:
                self.price_data[asset] = 0.0
            print(f"[CALCULATE_PORTFOLIO_PROPORTIONS INFO] Added missing assets with zero prices: {missing_assets}")
        
        # Reindex price_data to include all dates in positions.index
        # This ensures that we have price data aligned with each position date
        aligned_prices = self.price_data.reindex(index=positions.index, method='ffill').fillna(0)
        
        # Reindex columns to match positions.columns
        aligned_prices = aligned_prices.reindex(columns=positions.columns, fill_value=0)
        
        print(f"[CALCULATE_PORTFOLIO_PROPORTIONS] Aligned_prices shape: {aligned_prices.shape}")
        print(f"[CALCULATE_PORTFOLIO_PROPORTIONS] Aligned_prices head:\n{aligned_prices.head()}")
        
        # Debug: Verify alignment
        if not aligned_prices.index.equals(positions.index):
            print("[CALCULATE_PORTFOLIO_PROPORTIONS ERROR] Date indices of positions and aligned_prices do not match.")
            raise ValueError("Date indices alignment mismatch between positions and price_data.")
        
        if not set(aligned_prices.columns) == set(positions.columns):
            print("[CALCULATE_PORTFOLIO_PROPORTIONS ERROR] Column names of positions and aligned_prices do not match.")
            raise ValueError("Column names alignment mismatch between positions and price_data.")
        
        # Calculate portfolio values: Multiply positions by their respective prices
        portfolio_values = positions * aligned_prices
        print(f"[CALCULATE_PORTFOLIO_PROPORTIONS] Portfolio values shape: {portfolio_values.shape}")
        print(f"[CALCULATE_PORTFOLIO_PROPORTIONS] Portfolio values head:\n{portfolio_values.head()}")
        
        # Calculate total portfolio value per date
        total_values = portfolio_values.sum(axis=1).replace(0, pd.NA)
        print(f"[CALCULATE_PORTFOLIO_PROPORTIONS] Total portfolio values head:\n{total_values.head()}")
        
        # Calculate proportions
        proportions = portfolio_values.div(total_values, axis=0).multiply(100).fillna(0)
        proportions = proportions.round(2)
        
        print(f"[CALCULATE_PORTFOLIO_PROPORTIONS] Portfolio proportions calculated with shape: {proportions.shape}")
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
                    # print(f"[EXPORT_TO_ODS INFO] Writing value {numeric_value} for date {date_str}.")
                
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

def main(owner_id: int = 10, start_date: str = "2023-01-01") -> Optional[pd.DataFrame]:
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