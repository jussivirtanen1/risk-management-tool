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
            ORDER BY date ASC
        """

    def fetch_portfolio_data(self) -> None:
        """Fetch asset and transaction data from database."""
        print("[FETCH_PORTFOLIO_DATA] Fetching portfolio data from database...")
        with PostgresConnector() as db:
            self.assets_df = db.fetch_data(self.get_asset_query())
            self.transactions_df = db.fetch_data(self.get_transactions_query())
        print(f"[FETCH_PORTFOLIO_DATA] Assets DataFrame shape: {self.assets_df.shape}")
        print(f"[FETCH_PORTFOLIO_DATA] Transactions DataFrame shape: {self.transactions_df.shape}")

    def fetch_market_data(self) -> None:
        """Fetch market data from Yahoo Finance."""
        print("[FETCH_MARKET_DATA] Fetching market data from Yahoo Finance...")
        if self.assets_df is None:
            raise ValueError("Must fetch portfolio data before market data")
            
        tickers = self.assets_df['yahoo_ticker'].dropna().unique().tolist()
        print(f"[FETCH_MARKET_DATA] Tickers to fetch: {tickers}")
        try:
            self.price_data = yf.download(tickers, start=self.start_date)['Close']
            self.price_data = self.price_data.fillna(method='ffill')
            print(f"[FETCH_MARKET_DATA] Price data fetched with shape: {self.price_data.shape}")
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

        # Ensure all dates are in pandas datetime format
        self.transactions_df['date'] = pd.to_datetime(self.transactions_df['date'])
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
            valid_transactions = self.transactions_df[
                self.transactions_df['date'] <= monthly_date
            ]
            
            if not valid_transactions.empty:
                # Aggregate quantities by asset name
                positions = valid_transactions.groupby('name')['quantity'].sum()
                
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
        # Initialize aligned_prices with the same columns as positions
        print(f"[CALCULATE_PORTFOLIO_PROPORTIONS] Positions DataFrame before alignment:\n{positions}")
        print(f"[CALCULATE_PORTFOLIO_PROPORTIONS] Creating aligned_prices DataFrame...")
        aligned_prices = pd.DataFrame(index=positions.index, columns=positions.columns).fillna(0)
        print(f"[CALCULATE_PORTFOLIO_PROPORTIONS] Aligned_prices DataFrame after creation:\n{aligned_prices}")

        print(f"[CALCULATE_PORTFOLIO_PROPORTIONS] Aligned_prices initialized with shape: {aligned_prices.shape}")

        for date_idx in positions.index:
            if date_idx in self.price_data.index:
                price_date = date_idx
            else:
                mask = self.price_data.index <= date_idx
                if not mask.any():
                    print(f"[CALCULATE_PORTFOLIO_PROPORTIONS] No price data available before or on {date_idx.date()}. Skipping.")
                    continue
                price_date = self.price_data.index[mask][-1]
            
            try:
                # Reindex to match positions.columns and fill missing with 0
                price_series = self.price_data.loc[price_date].reindex(positions.columns).fillna(0)
                aligned_prices.loc[date_idx] = price_series
                print(f"[CALCULATE_PORTFOLIO_PROPORTIONS] Aligned price for {date_idx.date()} set to price date {price_date.date()}")
            except Exception as e:
                print(f"[CALCULATE_PORTFOLIO_PROPORTIONS ERROR] Failed to align price for {date_idx.date()}: {e}")
                raise e

        print(f"[CALCULATE_PORTFOLIO_PROPORTIONS] Aligned_prices shape after alignment: {aligned_prices.shape}")

        # Ensure columns match
        if set(aligned_prices.columns) != set(positions.columns):
            print("[CALCULATE_PORTFOLIO_PROPORTIONS WARNING] Mismatch between aligned_prices columns and positions columns. Aligning them.")
            aligned_prices = aligned_prices.reindex(columns=positions.columns, fill_value=0)
            print(f"[CALCULATE_PORTFOLIO_PROPORTIONS] Aligned_prices shape after reindexing: {aligned_prices.shape}")

        portfolio_values = positions * aligned_prices
        print(f"[CALCULATE_PORTFOLIO_PROPORTIONS] Portfolio values calculated with shape: {portfolio_values.shape}")

        # Prevent division by zero by replacing zeros with NaN and then filling with zero
        total_values = portfolio_values.sum(axis=1).replace(0, pd.NA)
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
        from odf.opendocument import OpenDocumentSpreadsheet
        from odf.table import Table, TableRow, TableCell
        from odf.text import P
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"portfolio_proportions_{timestamp}.ods"
        
        output_path = self.get_analysis_path()
        os.makedirs(output_path, exist_ok=True)
        full_path = os.path.join(output_path, filename)
        
        try:
            doc = OpenDocumentSpreadsheet()
            table = Table(name="Portfolio Proportions")
            
            # Add header row
            header_row = TableRow()
            header_cell = TableCell(valuetype="string")
            header_cell.addElement(P(text="Date"))
            header_row.addElement(header_cell)
            for col in df.columns:
                cell = TableCell(valuetype="string")
                cell.addElement(P(text=str(col)))
                header_row.addElement(cell)
            table.addElement(header_row)
            print("[EXPORT_TO_ODS] Header row added.")
            
            # Add data rows
            for idx, row in df.iterrows():
                tr = TableRow()
                date_cell = TableCell(valuetype="string")
                # Ensure date is in a string format (e.g., "YYYY-MM-DD")
                date_str = idx.strftime("%Y-%m-%d") if isinstance(idx, pd.Timestamp) else str(idx)
                date_cell.addElement(P(text=date_str))
                tr.addElement(date_cell)
                
                for value in row:
                    cell = TableCell(valuetype="float")
                    cell.addElement(P(text=str(value)))
                    tr.addElement(cell)
                table.addElement(tr)
            print("[EXPORT_TO_ODS] Data rows added.")
            
            doc.spreadsheet.addElement(table)
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