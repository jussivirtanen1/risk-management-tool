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
from src.data_fetcher import StockDataFetcher

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
        self.db = PostgresConnector()
        self.data_fetcher = StockDataFetcher(self.db)

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
        """Fetch market data from Yahoo Finance with currency conversion."""
        print("[FETCH_MARKET_DATA] Starting market data fetch...")
        if self.assets_df is None:
            raise ValueError("Must fetch portfolio data before market data")
            
        # Use StockDataFetcher instead of direct yfinance calls
        prices_df = self.data_fetcher.fetch_monthly_prices(
            self.owner_id, 
            self.start_date.strftime('%Y-%m-%d')
        )
        
        if prices_df.empty:
            raise ValueError("No valid price data fetched")
            
        # Convert the price data to the format we need
        self.price_data = prices_df.pivot(
            index='date',
            columns='asset_id',
            values='price'
        )
        
        # Rename columns to asset names
        asset_id_to_name = self.assets_df.set_index('asset_id')['name'].to_dict()
        self.price_data.columns = [asset_id_to_name.get(col, col) for col in self.price_data.columns]

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
        
        # Sort transactions by date to ensure proper accumulation
        merged_df = merged_df.sort_values('date')
        print("\n[CALCULATE_MONTHLY_POSITIONS] Transactions after sorting:")
        print(merged_df[['date', 'name_asset', 'quantity', 'event_type']])
        
        # Group by month end ('ME') frequency
        monthly_last_dates = self.price_data.groupby(pd.Grouper(freq='ME')).last()
        valid_dates = monthly_last_dates[~monthly_last_dates.isnull().any(axis=1)].index
        
        # Initialize positions dictionary and accumulation tracking
        positions_dict = {}
        running_positions = {}  # Track running total for each asset
        
        for monthly_date in valid_dates:
            print(f"\n[CALCULATE_MONTHLY_POSITIONS] Processing date: {monthly_date}")
            valid_transactions = merged_df[merged_df['date'] <= monthly_date]
            print(f"Number of valid transactions: {len(valid_transactions)}")
            
            if not valid_transactions.empty:
                # Calculate running positions by processing transactions chronologically
                for _, transaction in valid_transactions.iterrows():
                    asset_name = transaction['name_asset']
                    quantity = transaction['quantity']
                    
                    # Initialize position if not exists
                    if asset_name not in running_positions:
                        running_positions[asset_name] = 0
                    
                    # Update running position
                    running_positions[asset_name] += quantity
                    
                # Create positions Series from running positions
                positions = pd.Series(running_positions)
                
                print(f"\n[CALCULATE_MONTHLY_POSITIONS] Running positions as of {monthly_date}:")
                for asset, qty in running_positions.items():
                    print(f"  {asset}: {qty}")
                
                # Validate positions
                if (positions < 0).any():
                    print("\n[CALCULATE_MONTHLY_POSITIONS] WARNING: Found negative positions!")
                    print("Negative positions:")
                    print(positions[positions < 0])
                    
                    # Set negative positions to 0
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
            
            print("\n[CALCULATE_MONTHLY_POSITIONS] Complete positions DataFrame:")
            print(positions_df)
            
            return list(positions_df.index), positions_df
            
        except Exception as e:
            print(f"[CALCULATE_MONTHLY_POSITIONS ERROR] Failed to create positions DataFrame: {str(e)}")
            print(f"[CALCULATE_MONTHLY_POSITIONS ERROR] Full error details: {repr(e)}")
            raise ValueError(f"Failed to create positions DataFrame: {e}")

    def calculate_portfolio_proportions(self, positions: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate portfolio proportions over time.
        """
        print("\n[CALCULATE_PORTFOLIO_PROPORTIONS] Starting proportion calculations...")
        print("\nInput positions:")
        print(positions)
        
        # Get the latest EUR prices from transactions for assets missing in price_data
        missing_assets = set(positions.columns) - set(self.price_data.columns)
        if missing_assets:
            print(f"\n[CALCULATE_PORTFOLIO_PROPORTIONS] Assets missing Yahoo data: {missing_assets}")
            print("Using EUR prices from transactions for these assets...")
            
            for asset in missing_assets:
                try:
                    # Get asset_id for the missing asset
                    asset_matches = self.assets_df[self.assets_df['name'] == asset]
                    if asset_matches.empty:
                        print(f"WARNING: Asset {asset} not found in assets_df")
                        continue
                        
                    asset_id = asset_matches['asset_id'].iloc[0]
                    print(f"Found asset_id {asset_id} for {asset}")
                    
                    # Get all transactions for this asset with EUR prices
                    asset_transactions = self.transactions_df[
                        (self.transactions_df['asset_id'] == asset_id) &
                        (self.transactions_df['price_eur'].notna())
                    ]
                    
                    if asset_transactions.empty:
                        print(f"WARNING: No transactions found for asset {asset} (id: {asset_id})")
                        continue
                        
                    asset_transactions = asset_transactions.sort_values('date', ascending=False)
                    
                    # Get the latest EUR price
                    latest_price = asset_transactions.iloc[0]['price_eur']
                    latest_price_date = asset_transactions.iloc[0]['date']
                    print(f"Using price {latest_price} EUR for {asset} (from {latest_price_date})")
                    
                    # Create a Series with this price for all dates in our date range
                    if self.price_data.empty:
                        # If price_data is empty, create it with monthly dates
                        date_range = pd.date_range(
                            start=self.start_date,
                            end=pd.Timestamp.now(),
                            freq='ME'
                        )
                        self.price_data = pd.DataFrame(index=date_range)
                    
                    # Fill the entire date range with this price
                    self.price_data[asset] = latest_price
                    print(f"Added constant price {latest_price} EUR for {asset} across all dates")
                    
                except Exception as e:
                    print(f"ERROR processing asset {asset}: {str(e)}")
                    print(f"Debug info:")
                    print(f"Asset matches in assets_df:")
                    print(self.assets_df[self.assets_df['name'] == asset])
                    print(f"Transactions for asset:")
                    print(asset_transactions if 'asset_transactions' in locals() else "No transactions queried yet")
                    continue
        
        print("\n[CALCULATE_PORTFOLIO_PROPORTIONS] Price data after adding missing assets:")
        print(self.price_data)
        
        # Align prices with positions
        aligned_prices = self.price_data.reindex(index=positions.index, method='ffill')
        aligned_prices = aligned_prices.reindex(columns=positions.columns, fill_value=0)
        
        print("\n[CALCULATE_PORTFOLIO_PROPORTIONS] Aligned prices:")
        print(aligned_prices)
        
        # Verify no zero prices
        zero_prices = aligned_prices.columns[aligned_prices.eq(0).any()]
        if not zero_prices.empty:
            print(f"\nWARNING: Found zero prices for assets: {list(zero_prices)}")
            print("This might affect portfolio proportions!")
        
        # Calculate portfolio values
        portfolio_values = positions * aligned_prices
        print("\n[CALCULATE_PORTFOLIO_PROPORTIONS] Portfolio values:")
        print(portfolio_values)
        
        # Calculate total portfolio value per date
        total_values = portfolio_values.sum(axis=1).replace(0, pd.NA)
        print("\n[CALCULATE_PORTFOLIO_PROPORTIONS] Total portfolio values:")
        print(total_values)
        
        # Calculate proportions
        proportions = portfolio_values.div(total_values, axis=0).multiply(100)
        proportions = proportions.round(2).fillna(0)
        
        # Verify proportions sum to approximately 100%
        sum_proportions = proportions.sum(axis=1)
        print("\n[CALCULATE_PORTFOLIO_PROPORTIONS] Sum of proportions per date:")
        print(sum_proportions)
        
        if not all(sum_proportions.between(99, 101)):
            print("\nWARNING: Some dates have proportions not summing to 100%!")
        
        print("\n[CALCULATE_PORTFOLIO_PROPORTIONS] Final proportions:")
        print(proportions)
        
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