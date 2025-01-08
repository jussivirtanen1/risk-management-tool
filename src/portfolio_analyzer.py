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

# Set pandas display options for better debugging
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.max_rows', None)     # Show all rows
pd.set_option('display.width', None)        # Don't wrap wide DataFrames
pd.set_option('display.max_colwidth', None) # Don't truncate cell contents

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
            # print(f"[INIT] Start date set to: {self.start_date}")
        except Exception as e:
            # print(f"[INIT ERROR] Invalid start_date format: {start_date} | Error: {e}")
            raise ValueError(f"Invalid start_date format: {start_date}")

        self.assets_df = None
        self.transactions_df = None
        self.price_data = None
        self.name_ticker_map = {}  # Mapping from asset name to Yahoo ticker

    @staticmethod
    def get_analysis_path(owner_id: int) -> str:
        """Get the path to analysis directory for specific owner."""
        base_path = '/app/analysis' if os.path.exists('/.dockerenv') else str(Path.home() / "Desktop" / "analysis")
        owner_path = os.path.join(base_path, f"owner_{owner_id}")
        os.makedirs(owner_path, exist_ok=True)
        return owner_path

    def fetch_portfolio_data(self) -> None:
        """Fetch asset and transaction data from database."""
        print("\n[FETCH_PORTFOLIO_DATA] Fetching portfolio data from database...")
        with PostgresConnector() as db:
            self.assets_df = db.get_active_assets(self.owner_id)
            self.transactions_df = db.get_portfolio_transactions(self.owner_id)
            
            print("\n[FETCH_PORTFOLIO_DATA] Assets DataFrame:")
            print(f"Columns: {self.assets_df.columns.tolist()}")
            print(f"Shape: {self.assets_df.shape}")
            print("Data:")
            print(self.assets_df)
            
            print("\n[FETCH_PORTFOLIO_DATA] Transactions DataFrame:")
            print(f"Columns: {self.transactions_df.columns.tolist()}")
            print(f"Shape: {self.transactions_df.shape}")
            print("Data:")
            print(self.transactions_df)
            
            if self.transactions_df is not None:
                self.transactions_df['date'] = pd.to_datetime(self.transactions_df['date'])
                self.transactions_df = self.transactions_df[self.transactions_df['date'] >= self.start_date]
                print("\n[FETCH_PORTFOLIO_DATA] Filtered Transactions:")
                print(f"Date range: {self.transactions_df['date'].min()} to {self.transactions_df['date'].max()}")
                print(f"Number of transactions: {len(self.transactions_df)}")
        
        if self.assets_df is None or self.transactions_df is None:
            raise ValueError("Failed to fetch portfolio data from database")

        # Create a mapping from asset name to Yahoo ticker
        self.name_ticker_map = pd.Series(
            self.assets_df.yahoo_ticker.values,
            index=self.assets_df.name
        ).to_dict()
        print(f"\n[FETCH_PORTFOLIO_DATA] Asset Name to Ticker Mapping:")
        for name, ticker in self.name_ticker_map.items():
            print(f"  {name}: {ticker}")

    def fetch_market_data(self) -> None:
        """Fetch market data from Yahoo Finance."""
        print("[FETCH_MARKET_DATA] Starting market data fetch...")
        if self.assets_df is None:
            raise ValueError("Must fetch portfolio data before market data")
        
        # Extract Yahoo tickers from the name-ticker mapping
        tickers = list(self.name_ticker_map.values())
        # Remove any NaN or None tickers
        tickers = [ticker for ticker in tickers if isinstance(ticker, str) and ticker.strip() != '']
        print(f"[FETCH_MARKET_DATA] Attempting to fetch data for tickers: {tickers}")
        
        if not tickers:
            raise ValueError("No valid tickers found to fetch market data")
            
        try:
            # Add some buffer days to ensure we have enough data
            start_date = self.start_date - pd.Timedelta(days=10)
            print(f"[FETCH_MARKET_DATA] Fetching from date: {start_date}")
            self.price_data = yf.download(tickers, start=start_date)['Close']
            print(f"[FETCH_MARKET_DATA] Raw price data shape: {self.price_data.shape}")
            print(f"[FETCH_MARKET_DATA] Raw price data columns: {self.price_data.columns.tolist()}")
            
            # Check for columns that are all NaN
            nan_columns = self.price_data.columns[self.price_data.isna().all()]
            if not nan_columns.empty:
                print(f"[FETCH_MARKET_DATA] Found columns with all NaN values: {nan_columns.tolist()}")
                print("[FETCH_MARKET_DATA] Removing these columns from analysis")
                self.price_data = self.price_data.drop(columns=nan_columns)
                
                # Also remove these from name_ticker_map
                ticker_to_name = {v: k for k, v in self.name_ticker_map.items()}
                for ticker in nan_columns:
                    if ticker in ticker_to_name:
                        asset_name = ticker_to_name[ticker]
                        del self.name_ticker_map[asset_name]
                        print(f"[FETCH_MARKET_DATA] Removed {asset_name} ({ticker}) from analysis due to missing data")

            print(f"[FETCH_MARKET_DATA] After removing NaN columns - shape: {self.price_data.shape}")
            print(f"[FETCH_MARKET_DATA] Remaining columns: {self.price_data.columns.tolist()}")
            print(f"[FETCH_MARKET_DATA] Raw price data index: {self.price_data.index.min()} to {self.price_data.index.max()}")
            
            if self.price_data.empty:
                raise ValueError("No valid price data remained after removing NaN columns")
                
            self.price_data = self.price_data.ffill()  # Forward fill missing values
            
            # Rename price_data columns from tickers to asset names
            ticker_to_name = {v: k for k, v in self.name_ticker_map.items()}
            print(f"[FETCH_MARKET_DATA] Ticker to name mapping after cleanup: {ticker_to_name}")
            self.price_data.rename(columns=ticker_to_name, inplace=True)
            print(f"[FETCH_MARKET_DATA] Final price data columns: {self.price_data.columns.tolist()}")
            
        except Exception as e:
            print(f"[FETCH_MARKET_DATA ERROR] Failed to fetch price data: {str(e)}")
            print(f"[FETCH_MARKET_DATA ERROR] Full error details: {repr(e)}")
            raise ValueError(f"Failed to fetch price data: {e}")

    def calculate_monthly_positions(self) -> Tuple[List[pd.Timestamp], pd.DataFrame]:
        """Calculate monthly positions considering transaction timing."""
        print("\n[CALCULATE_MONTHLY_POSITIONS] Starting monthly positions calculation...")
        if self.transactions_df is None or self.price_data is None:
            raise ValueError("Must fetch both transaction and market data first")

        # Merge transactions with asset data
        merged_df = self.transactions_df.merge(
            self.assets_df[['asset_id', 'name', 'yahoo_ticker']],
            on='asset_id',
            how='left',
            suffixes=('_trans', '_asset')
        )
        
        print("\n[CALCULATE_MONTHLY_POSITIONS] Transaction types in data:")
        print(merged_df['event_type'].value_counts())
        
        # Group by month end ('ME') frequency
        monthly_last_dates = self.price_data.groupby(pd.Grouper(freq='ME')).last()
        valid_dates = monthly_last_dates[~monthly_last_dates.isnull().any(axis=1)].index
        
        # Initialize positions dictionary
        positions_dict = {}
        
        for monthly_date in valid_dates:
            print(f"\n[CALCULATE_MONTHLY_POSITIONS] Processing date: {monthly_date}")
            valid_transactions = merged_df[merged_df['date'] <= monthly_date]
            print(f"Number of valid transactions: {len(valid_transactions)}")
            
            if not valid_transactions.empty:
                # Calculate net positions by summing quantities
                positions = valid_transactions.groupby('name_asset')['quantity'].sum()
                print(f"Raw positions before validation:\n{positions}")
                
                # Validate positions
                if (positions < 0).any():
                    print("\n[CALCULATE_MONTHLY_POSITIONS] WARNING: Found negative positions!")
                    print("Negative positions:")
                    print(positions[positions < 0])
                    
                    # Option 1: Set negative positions to 0
                    positions[positions < 0] = 0
                    print("\nPositions after setting negatives to 0:")
                    print(positions)
                
                # Filter out zero positions
                positions = positions[positions != 0]
                print(f"\nFinal positions for {monthly_date}:\n{positions}")
                
                if not positions.empty:
                    positions_dict[monthly_date] = positions
                else:
                    print("No non-zero positions found for this date")
            else:
                print("No valid transactions found for this date")

        print(f"\n[CALCULATE_MONTHLY_POSITIONS] Final positions dictionary size: {len(positions_dict)}")
        
        if not positions_dict:
            raise ValueError("No valid portfolio positions found for the given dates.")

        try:
            positions_df = pd.DataFrame.from_dict(positions_dict, orient='index').fillna(0)
            positions_df.index.name = 'Date'
            
            # Validate final positions DataFrame
            if (positions_df < 0).any().any():
                print("\n[CALCULATE_MONTHLY_POSITIONS] WARNING: Negative values in final positions DataFrame!")
                print("Negative positions:")
                print(positions_df[positions_df < 0].dropna(how='all'))
                # Set any remaining negatives to 0
                positions_df[positions_df < 0] = 0
            
            print(f"\n[CALCULATE_MONTHLY_POSITIONS] Final positions DataFrame:")
            print(f"Shape: {positions_df.shape}")
            print(f"Columns: {positions_df.columns.tolist()}")
            print(f"Sample data:\n{positions_df}")
            
            return list(positions_df.index), positions_df
            
        except Exception as e:
            print(f"[CALCULATE_MONTHLY_POSITIONS ERROR] Failed to create positions DataFrame: {str(e)}")
            print(f"[CALCULATE_MONTHLY_POSITIONS ERROR] Full error details: {repr(e)}")
            raise ValueError(f"Failed to create positions DataFrame: {e}")

    def calculate_portfolio_proportions(self, positions: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate portfolio proportions over time.
        
        Args:
            positions: DataFrame with asset positions
            
        Returns:
            DataFrame with portfolio proportions
        """
        # print("[CALCULATE_PORTFOLIO_PROPORTIONS] Calculating portfolio proportions...")
        
        # Ensure that the price_data contains all asset names present in positions
        missing_assets = set(positions.columns) - set(self.price_data.columns)
        if missing_assets:
            # print(f"[CALCULATE_PORTFOLIO_PROPORTIONS WARNING] Missing assets in price_data: {missing_assets}")
            # Optionally, fetch missing asset prices or handle accordingly
            # For simplicity, we'll add them with zero prices
            for asset in missing_assets:
                self.price_data[asset] = 0.0
            # print(f"[CALCULATE_PORTFOLIO_PROPORTIONS INFO] Added missing assets with zero prices: {missing_assets}")
        
        # Reindex price_data to include all dates in positions.index
        # This ensures that we have price data aligned with each position date
        aligned_prices = self.price_data.reindex(index=positions.index, method='ffill').fillna(0)
        
        # Reindex columns to match positions.columns
        aligned_prices = aligned_prices.reindex(columns=positions.columns, fill_value=0)
        
        # print(f"[CALCULATE_PORTFOLIO_PROPORTIONS] Aligned_prices shape: {aligned_prices.shape}")
        # print(f"[CALCULATE_PORTFOLIO_PROPORTIONS] Aligned_prices head:\n{aligned_prices.head()}")
        
        # Debug: Verify alignment
        if not aligned_prices.index.equals(positions.index):
            # print("[CALCULATE_PORTFOLIO_PROPORTIONS ERROR] Date indices of positions and aligned_prices do not match.")
            raise ValueError("Date indices alignment mismatch between positions and price_data.")
        
        if not set(aligned_prices.columns) == set(positions.columns):
            # print("[CALCULATE_PORTFOLIO_PROPORTIONS ERROR] Column names of positions and aligned_prices do not match.")
            raise ValueError("Column names alignment mismatch between positions and price_data.")
        
        # Calculate portfolio values: Multiply positions by their respective prices
        portfolio_values = positions * aligned_prices
        # print(f"[CALCULATE_PORTFOLIO_PROPORTIONS] Portfolio values shape: {portfolio_values.shape}")
        # print(f"[CALCULATE_PORTFOLIO_PROPORTIONS] Portfolio values head:\n{portfolio_values.head()}")
        
        # Calculate total portfolio value per date
        total_values = portfolio_values.sum(axis=1).replace(0, pd.NA)
        # print(f"[CALCULATE_PORTFOLIO_PROPORTIONS] Total portfolio values head:\n{total_values.head()}")
        
        # Calculate proportions
        proportions = portfolio_values.div(total_values, axis=0).multiply(100).fillna(0)
        proportions = proportions.round(2)
        
        # print(f"[CALCULATE_PORTFOLIO_PROPORTIONS] Portfolio proportions calculated with shape: {proportions.shape}")
        # print(f"[CALCULATE_PORTFOLIO_PROPORTIONS] Proportions DataFrame head:\n{proportions.head()}")

        return proportions

    def export_to_ods(self, df: pd.DataFrame) -> str:
        """
        Export DataFrame to .ods file.
        
        Args:
            df: DataFrame to export
            
        Returns:
            Path to saved file
        """
        # print("[EXPORT_TO_ODS] Exporting DataFrame to .ods file...")
        
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
        # print("[EXPORT_TO_ODS] Header row added.")
        
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
                    # print(f"[EXPORT_TO_ODS WARNING] Found NaN value for date {date_str}. Replacing with 0.00.")
                else:
                    # Round to two decimal places
                    numeric_value = round(float(value), 2)
                    # print(f"[EXPORT_TO_ODS INFO] Writing value {numeric_value} for date {date_str}.")
                
                # Create a cell with numerical value
                cell = TableCell(valuetype="float", value=str(numeric_value))
                
                # Note: No need to add Number or P elements for numerical cells
                tr.addElement(cell)
            
            table.addElement(tr)
        
        # print("[EXPORT_TO_ODS] Data rows added.")
        
        # Define filename and path with owner_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"portfolio_proportions_{timestamp}.ods"
        output_path = self.get_analysis_path(self.owner_id)  # Pass owner_id
        full_path = os.path.join(output_path, filename)
        
        # Save the document
        try:
            doc.save(full_path)
            # print(f"[EXPORT_TO_ODS] File saved to: {full_path}")
            return full_path
        except Exception as e:
            # print(f"[EXPORT_TO_ODS ERROR] Failed to export to .ods: {e}")
            raise ValueError(f"Failed to export to .ods: {e}")

    def analyze(self) -> Optional[pd.DataFrame]:
        """
        Run the complete portfolio analysis.
        
        Returns:
            DataFrame with portfolio proportions or None if analysis fails
        """
        try:
            # Fetch all required data
            self.fetch_portfolio_data()
            
            # If we have no assets or transactions, return None early
            if self.assets_df.empty or self.transactions_df.empty:
                return None
            
            try:
                self.fetch_market_data()
            except ValueError as e:
                if "No valid tickers found" in str(e):
                    return None
                raise e

            # Calculate positions and proportions
            monthly_dates, positions = self.calculate_monthly_positions()
            proportions = self.calculate_portfolio_proportions(positions)

            # Export results
            if not proportions.empty:
                self.export_to_ods(proportions)
                return proportions
            return None
                
        except Exception as e:
            raise ValueError(f"Error during portfolio analysis: {e}")

def main(start_date: str = "2023-01-01") -> dict:
    """
    Main function to run portfolio analysis for multiple owners.
    
    Args:
        start_date: Start date for analysis
        
    Returns:
        dict: Dictionary of owner_id: analysis_results pairs
    """
    owner_ids = [10, 20, 30]
    results = {}
    
    for owner_id in owner_ids:
        print(f"\nAnalyzing portfolio for owner {owner_id}")
        try:
            analyzer = PortfolioAnalyzer(owner_id, start_date)
            results[owner_id] = analyzer.analyze()
        except Exception as e:
            print(f"Error analyzing portfolio for owner {owner_id}: {e}")
            results[owner_id] = None
    
    return results

if __name__ == "__main__":
    main()