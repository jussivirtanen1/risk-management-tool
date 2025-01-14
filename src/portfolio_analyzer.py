"""
Portfolio Analysis Module

Analyzes portfolio composition over time using transaction data and market prices.
"""

import pandas as pd
import polars as pl
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

class PortfolioAnalyzer:
    def __init__(self, owner_id: int, start_date: str = "2023-01-01"):
        """
        Initialize Portfolio Analyzer.
        
        Args:
            owner_id: ID of the portfolio owner
            start_date: Start date for analysis
        """
        self.owner_id = owner_id
        self.start_date = start_date  # Keep the original string
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
            
            if self.transactions_df is not None:
                # Convert date strings to Polars Date type
                self.transactions_df = self.transactions_df.with_columns(
                    pl.col('date').str.strptime(pl.Date, format='%Y-%m-%d')
                )
                # Filter transactions after start date
                start_date_pl = pl.lit(self.start_date).str.strptime(pl.Date, format='%Y-%m-%d')
                self.transactions_df = self.transactions_df.filter(
                    pl.col('date') >= start_date_pl
                )
                print("\n[FETCH_PORTFOLIO_DATA] Filtered Transactions:")
                print(f"Date range: {self.transactions_df['date'].min()} to {self.transactions_df['date'].max()}")
                print(f"Number of transactions: {len(self.transactions_df)}")
        
        if self.assets_df is None or self.transactions_df is None:
            raise ValueError("Failed to fetch portfolio data from database")

        # Create a mapping from asset name to Yahoo ticker
        self.name_ticker_map = self.assets_df.select('name', 'yahoo_ticker').to_dict()

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
        
        if prices_df.is_empty():
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

    def calculate_monthly_positions(self) -> Tuple[List[pl.Datetime], pl.DataFrame]:
        """Calculate monthly positions considering transaction timing."""
        if self.transactions_df is None or self.price_data is None:
            raise ValueError("Must fetch both transaction and market data first")

        # Merge transactions with asset data
        merged_df = self.transactions_df.join(
            self.assets_df.select(['asset_id', 'name', 'yahoo_ticker']),
            on='asset_id',
            how='left'
        )
        # print("merged_df columns", merged_df.columns)
        # print("merged_df", merged_df.select('date', 'name', 'quantity', 'price_eur', 'yahoo_ticker'))
        # Sort transactions by date
        merged_df = merged_df.sort('date')
        # print("merged_df sorted", merged_df.select('date', 'name', 'quantity', 'price_eur', 'yahoo_ticker'))
        # print("merged_df pivot", merged_df.pivot("name", index="date", aggregate_function="sum", values="quantity"))
        # merged_df_pivoted = merged_df.pivot("name", index="date", aggregate_function="sum", values="quantity")
        # merged_df_pivoted.columns.remove('date')
        merged_df_cumsum = merged_df.with_columns(
            cumulative_quantity=pl.col("quantity")
            .cum_sum()
            .over("name", order_by="date")
            ).select('date', 'name', 'cumulative_quantity', 'quantity', 'price_eur', 'yahoo_ticker')\
             .pivot("name", index="date", values="cumulative_quantity")
        
        print("merged_df cumsum", merged_df_cumsum)
        # Group by month end and get last dates
        # print("\n[CALCULATE_MONTHLY_POSITIONS] Price data:" )
        # print(self.price_data)
        # print("self.price_data", self.price_data.columns)
        # print("self.price_data", self.price_data)
        # monthly_last_dates = self.price_data
        # print("monthly_last_dates", monthly_last_dates)
        
        # valid_dates = monthly_last_dates['date'].to_list()
        # print('valid_dates', valid_dates)
            
        return merged_df_cumsum
        # return valid_dates, positions_df
            
        except Exception as e:
            print(f"[CALCULATE_MONTHLY_POSITIONS ERROR] Failed to create positions DataFrame: {str(e)}")
            print(f"[CALCULATE_MONTHLY_POSITIONS ERROR] Full error details: {repr(e)}")
            raise ValueError(f"Failed to create positions DataFrame: {e}")

    def calculate_portfolio_proportions(self, positions: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate portfolio proportions over time.
        """
        # Get the latest EUR prices from transactions for assets missing in price_data
        missing_assets = set(positions.columns) - set(self.price_data.columns)
        if missing_assets:
            print(f"\n[CALCULATE_PORTFOLIO_PROPORTIONS] Assets missing Yahoo data: {missing_assets}")
            print("Using EUR prices from transactions for these assets...")
            
            for asset in missing_assets:
                try:
                    # Get asset_id for the missing asset
                    asset_matches = self.assets_df.filter(pl.col('name') == asset)
                    if asset_matches.is_empty():
                        print(f"WARNING: Asset {asset} not found in assets_df")
                        continue
                        
                    asset_id = asset_matches['asset_id'].iloc[0]
                    print(f"Found asset_id {asset_id} for {asset}")
                    
                    # Get all transactions for this asset with EUR prices
                    asset_transactions = self.transactions_df.filter(
                        (pl.col('asset_id') == asset_id) &
                        (pl.col('price_eur').is_not_null())
                    )
                    
                    if asset_transactions.is_empty():
                        print(f"WARNING: No transactions found for asset {asset} (id: {asset_id})")
                        continue
                        
                    # Get the latest EUR price
                    latest_price = asset_transactions.sort('date', descending=True)['price_eur'][0]
                    latest_price_date = asset_transactions.sort('date', descending=True)['date'][0]
                    print(f"Using price {latest_price} EUR for {asset} (from {latest_price_date})")
                    
                    # Create a Series with this price for all dates in our date range
                    if self.price_data.is_empty():
                        # If price_data is empty, create it with monthly dates
                        date_range = pl.date_range(
                            start=pl.lit(self.start_date).str.strptime(pl.Date, format='%Y-%m-%d'),
                            end=pl.now().date(),
                            interval="1mo",
                            closed="left"
                        ).alias("date")
                        
                        self.price_data = pl.DataFrame({
                            "date": date_range
                        })
                        
                        # Assert price data is properly initialized
                        assert not self.price_data.is_empty(), "Price data should not be empty after initialization"
                        assert 'date' in self.price_data.columns, "Price data should have a date column"
                        assert len(self.price_data) > 0, "Price data should have rows"
                        assert self.price_data['date'].dtype == pl.Date, "Date column should be of type Date"
                    
                    # Fill the entire date range with this price
                    self.price_data = self.price_data.with_columns(
                        pl.lit(latest_price).alias(asset)
                    )
                    
                    # Assert the new asset column was added correctly
                    assert asset in self.price_data.columns, f"Asset {asset} should be added to price data"
                    assert self.price_data[asset].mean() == latest_price, f"Asset {asset} should have price {latest_price}"
                    
                    print(f"Added constant price {latest_price} EUR for {asset} across all dates")
                    
                except Exception as e:
                    print(f"ERROR processing asset {asset}: {str(e)}")
                    print(f"Debug info:")
                    print(f"Asset matches in assets_df:")
                    # print(self.assets_df.filter(pl.col('name') == asset))
                    print(f"Transactions for asset:")
                    print(asset_transactions if 'asset_transactions' in locals() else "No transactions queried yet")
                    continue
        
        # Align prices with positions
        
        aligned_prices = self.price_data
        # aligned_prices = self.price_data.reindex(index=positions.index, method='ffill')
        # aligned_prices = aligned_prices.reindex(columns=positions.columns, fill_value=0)
        
        # print("\n[CALCULATE_PORTFOLIO_PROPORTIONS] Aligned prices:")
        # print(aligned_prices)
        
        # Verify no zero prices

        zero_prices = (aligned_prices
                        .filter(
                            pl.any_horizontal(pl.col('*').is_null())
                        )
                    )

        if not zero_prices.is_empty():
            print(f"\nWARNING: Found zero prices for assets: {list(zero_prices)}")
            print("This might affect portfolio proportions!")
        # random_var = pd.DataFrame()
        # assert isinstance(random_var, pl.DataFrame), f"positions {positions}"
        # assert isinstance(random_var, pl.DataFrame), f"aligned_prices {aligned_prices}"
        
        # Calculate portfolio values
        portfolio_values = positions * aligned_prices
        
        # Calculate total portfolio value per date
        total_values = portfolio_values.sum(axis=1).replace(0, pl.Null)
        
        # Calculate proportions
        proportions = portfolio_values.div(total_values, axis=0).multiply(100)
        proportions = proportions.round(2).fillna(0)
        
        # Verify proportions sum to approximately 100%
        sum_proportions = proportions.sum(axis=1)

        
        if not all(sum_proportions.between(99, 101)):
            print("\nWARNING: Some dates have proportions not summing to 100%!")
        
        
        return proportions

    def export_to_ods(self, df: pl.DataFrame) -> str:
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
            if isinstance(idx, pl.Datetime):
                date_str = idx.strftime("%Y-%m-%d")
            else:
                date_str = str(idx)
            date_cell.addElement(P(text=date_str))
            tr.addElement(date_cell)
            
            # Add Portfolio Proportions Cells
            for value in row:
                if pl.is_null(value):
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

    def analyze(self) -> Optional[pl.DataFrame]:
        """
        Run the complete portfolio analysis.
        
        Returns:
            DataFrame with portfolio proportions or None if analysis fails
        """
        try:
            # Fetch all required data
            self.fetch_portfolio_data()
            
            # If we have no assets or transactions, return None early
            if self.assets_df.is_empty() or self.transactions_df.is_empty():
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
            if not proportions.is_empty():
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