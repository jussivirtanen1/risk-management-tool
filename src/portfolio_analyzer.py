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
        self.start_date = start_date
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
        with PostgresConnector() as db:
            self.assets_df = db.fetch_data(self.get_asset_query())
            self.transactions_df = db.fetch_data(self.get_transactions_query())

    def fetch_market_data(self) -> None:
        """Fetch market data from Yahoo Finance."""
        if self.assets_df is None:
            raise ValueError("Must fetch portfolio data before market data")
            
        tickers = self.assets_df['yahoo_ticker'].unique().tolist()
        self.price_data = yf.download(tickers, start=self.start_date)['Close']
        self.price_data = self.price_data.fillna(method='ffill')


    def calculate_monthly_positions(self) -> Tuple[List[date], pd.DataFrame]:
        """
        Calculate monthly positions considering transaction timing.
        
        Returns:
            Tuple of (monthly_dates, positions_df)
        """
        if self.transactions_df is None or self.price_data is None:
            raise ValueError("Must fetch both transaction and market data first")

        price_data_dates = pd.to_datetime(self.price_data.index)
        self.transactions_df['date'] = pd.to_datetime(self.transactions_df['date'])
        
        monthly_last_dates = self.price_data.groupby(pd.Grouper(freq='ME')).last()
        valid_dates = monthly_last_dates[~monthly_last_dates.isnull().any(axis=1)].index
        
        positions_list = []
        final_dates = []
        
        for monthly_date in valid_dates:
            # Keep monthly_date as pandas Timestamp instead of converting to date
            valid_transactions = self.transactions_df[
                self.transactions_df['date'] <= monthly_date
            ]
            
            if not valid_transactions.empty:
                positions = valid_transactions.groupby('name')['quantity'].sum()
                
                if (positions != 0).any():
                    positions.name = monthly_date.date()  # Convert to date only when storing
                    positions_list.append(positions)
                    final_dates.append(monthly_date.date())
        
        return final_dates, pd.DataFrame(positions_list, index=final_dates)

    def calculate_portfolio_proportions(self, 
                                     positions: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate portfolio proportions over time.
        
        Args:
            positions: DataFrame with asset positions
            
        Returns:
            DataFrame with portfolio proportions
        """
        aligned_prices = pd.DataFrame(index=positions.index)
        
        for date_idx in positions.index:
            if date_idx in self.price_data.index:
                price_date = date_idx
            else:
                mask = self.price_data.index <= date_idx
                if not mask.any():
                    continue
                price_date = self.price_data.index[mask][-1]
            
            aligned_prices.loc[date_idx] = self.price_data.loc[price_date]
        
        portfolio_values = positions * aligned_prices
        proportions = portfolio_values.div(
            portfolio_values.sum(axis=1), axis=0
        ) * 100
        
        return proportions.round(2)

    def export_to_ods(self, df: pd.DataFrame) -> str:
        """
        Export DataFrame to .ods file.
        
        Args:
            df: DataFrame to export
            
        Returns:
            Path to saved file
        """
        from odf.opendocument import OpenDocumentSpreadsheet
        from odf.table import Table, TableRow, TableCell
        from odf.text import P
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"portfolio_proportions_{timestamp}.ods"
        
        output_path = self.get_analysis_path()
        os.makedirs(output_path, exist_ok=True)
        full_path = os.path.join(output_path, filename)
        
        doc = OpenDocumentSpreadsheet()
        table = Table(name="Portfolio Proportions")
        
        # Add header row
        header_row = TableRow()
        header_row.addElement(TableCell(valuetype="string", value="Date"))
        for col in df.columns:
            cell = TableCell(valuetype="string", value=str(col))
            cell.addElement(P(text=str(col)))
            header_row.addElement(cell)
        table.addElement(header_row)
        
        # Add data rows
        for idx, row in df.iterrows():
            tr = TableRow()
            date_cell = TableCell(valuetype="string", value=str(idx))
            date_cell.addElement(P(text=str(idx)))
            tr.addElement(date_cell)
            
            for value in row:
                cell = TableCell(valuetype="float", value=str(value))
                cell.addElement(P(text=str(value)))
                tr.addElement(cell)
            table.addElement(tr)
        
        doc.spreadsheet.addElement(table)
        doc.save(full_path)
        
        return full_path

    def analyze(self) -> Optional[pd.DataFrame]:
        """
        Run the complete portfolio analysis.
        
        Returns:
            DataFrame with portfolio proportions or None if analysis fails
        """
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
                print(f"Portfolio analysis exported to: {output_path}")
                return proportions
            else:
                print("No valid portfolio data to analyze")
                return None
                
        except Exception as e:
            print(f"Error during portfolio analysis: {e}")
            return None

def main(owner_id: int = 1, start_date: str = "2020-01-01") -> Optional[pd.DataFrame]:
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