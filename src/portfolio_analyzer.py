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
pl.Config.set_tbl_rows(1000)

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
        # self.missing_assets = []
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
        print("Fetching portfolio data from database...")
        with PostgresConnector() as db:
            self.assets_df = db.get_active_assets(self.owner_id)
            self.transactions_df = db.get_portfolio_transactions(self.owner_id)
            
            if self.transactions_df is not None:
                # Convert date strings to Polars Date type
                self.transactions_df = self.transactions_df
                # Filter transactions after start date
                print(f"Date range for database transactions: {self.transactions_df['date'].min()} to {self.transactions_df['date'].max()}")
                print(f"Number of transactions: {len(self.transactions_df)}")
        
        if self.assets_df is None or self.transactions_df is None:
            raise ValueError("Failed to fetch portfolio data from database")

        # Create a mapping from asset name to Yahoo ticker
        self.name_ticker_map = self.assets_df.select('name', 'yahoo_ticker').to_dict()

        # print(f"Asset Name to Ticker Mapping:")
        # for name, ticker in self.name_ticker_map.items():
        #     print(f"  {name}: {ticker}")

    def fetch_market_data(self) -> None:
        """Fetch market data from Yahoo Finance with currency conversion."""
        print("Starting market data fetch...")
        if self.assets_df is None:
            raise ValueError("Must fetch portfolio data before market data")
            
        # Use StockDataFetcher instead of direct yfinance calls
        self.price_data = self.data_fetcher.fetch_prices_from_yahoo(
            self.owner_id, 
            self.start_date
        )
        
        # Rename columns to asset names
        # self.price_data.columns = [asset_id_to_name.get(col, col) for col in self.price_data.columns]

    def calculate_monthly_positions(self) -> Tuple[List[pl.Datetime], pl.DataFrame]:
        """Calculate monthly positions considering transaction timing."""
        if self.transactions_df is None:
            raise ValueError("Must fetch both transaction and market data first")
        # print("self.transactions_df", self.transactions_df.select('date', 'name', 'quantity', 'price_eur'))
        # Merge transactions with asset data
        merged_df = self.transactions_df.join(
            self.assets_df.select(['asset_id', 'name', 'yahoo_ticker']),
            on='asset_id',
            how='inner'
        )
        # print("merged_df", merged_df)
        # print("merged_df columns", merged_df.columns)
        # print("merged_df", merged_df.select('date', 'name', 'quantity', 'price_eur', 'yahoo_ticker'))
        # print("merged_df before cumsum", merged_df.select('date', 'name', 'quantity', 'price_eur', 'yahoo_ticker'))
        # print("merged_df sorted", merged_df.select('date', 'name', 'quantity', 'price_eur', 'yahoo_ticker'))
        # print("merged_df pivot", merged_df.pivot("name", index="date", aggregate_function="sum", values="quantity"))
        # merged_df_pivoted = merged_df.pivot("name", index="date", aggregate_function="sum", values="quantity")
        # merged_df_pivoted.columns.remove('date')
        merged_df_cumsum = merged_df.with_columns(
            cumulative_quantity=pl.col("quantity")
            .cum_sum()
            .over("name", order_by="date")
            ).select('date', 'name', 'cumulative_quantity', 'quantity', 'price_eur', 'yahoo_ticker')\
             .pivot("yahoo_ticker", index="date", values="cumulative_quantity", aggregate_function="max")\
             .fill_null(strategy="forward")


        # # Debugging cum_sum()
        # merged_df_cumsum_agg_sum = merged_df.with_columns(
        #     cumulative_quantity=pl.col("quantity")
        #     .cum_sum()
        #     .over("name", order_by="date")
        #     ).select('date', 'name', 'cumulative_quantity', 'quantity', 'price_eur', 'yahoo_ticker')\
        #      .pivot("yahoo_ticker", index="date", values="cumulative_quantity", aggregate_function="max")\
        #      .fill_null(strategy="forward")
        
        # merged_df_cumsum_no_pivot = merged_df.with_columns(
        #     cumulative_quantity=pl.col("quantity")
        #     .cum_sum()
        #     .over("name", order_by="date")
        #     ).select('date', 'name', 'cumulative_quantity', 'quantity', 'price_eur', 'yahoo_ticker')

        # print("merged_df_cumsum_no_pivot from calculate_monthly_positions", merged_df_cumsum_no_pivot.filter((pl.col('name') == 'Mandatum') | (pl.col('name') == 'Sampo Oyj')))
        # print("merged_df_cumsum_agg_sum from calculate_monthly_positions", merged_df_cumsum_agg_sum.select('date', 'MANTA.HE', 'SAMPO.HE'))

            
        return merged_df_cumsum
        # return valid_dates, positions_df

    def calculate_portfolio_proportions(self, merged_df_cumsum: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate portfolio proportions over time.
        """
        # Get the latest EUR prices from transactions for assets missing in price_data

        missing_assets = set(merged_df_cumsum.columns) - set(self.price_data.columns)
        if missing_assets is None:
            print("No missing assets")
        else:
            print(f"Assets missing Yahoo data: {missing_assets}")
            merged_df_price_eur = self.transactions_df.join(
                self.assets_df.select(['asset_id', 'name', 'yahoo_ticker']),
                on='asset_id',
                how='inner'
            )
            # print("merged_df_price_eur right after join of transactions and assets_df", merged_df_price_eur.select())
            merged_df_price_eur = merged_df_price_eur.select('date', 'name', 'quantity', 'price_eur', 'yahoo_ticker')\
                .pivot("yahoo_ticker", index="date", values="price_eur", aggregate_function="first")\
                .select(pl.col('date'), pl.col(missing_assets))\
                .fill_null(strategy="forward")
            merged_df_price_eur = self.price_data.select('date')\
                                                .join(merged_df_price_eur, on='date', how='left')\
                                                .fill_null(strategy="forward")
            self.price_data = self.price_data.join(merged_df_price_eur, on='date', how='left')

        cumulative_quantities = self.price_data.select('date')\
                                            .join(merged_df_cumsum, on='date', how='left')\
                                            .fill_null(strategy="forward")\
                                            .sort('date')\
                                            .fill_null(0)\
                                            .select(pl.col('date'), pl.col('*').exclude('date').round(2))

        # Multiply price data with positions
        asset_values = self.price_data.join(
            cumulative_quantities,
            on='date',
            how='left'
        )\
        .select([
            pl.col('date'),
            *[pl.col(f"{col}").mul(pl.col(f"{col}_right")).alias(col)
            for col in self.price_data.columns if col != 'date']
        ])
        # print("multiplied portfolio_values", asset_values)
        rename_dict = dict(self.assets_df.select('yahoo_ticker', 'name').iter_rows())
        asset_values_w_sum = asset_values.select(pl.col('*'), pl.sum_horizontal(pl.all().exclude('date')).alias('total'))\
                                         .select(pl.col('date'), pl.col('*').exclude('date').round(2))
        # print("asset_values_w_sum", asset_values_w_sum)
        asset_proportions = asset_values_w_sum.select(pl.col('date'), (pl.col('*').exclude('date') / pl.col('total')) * 100)  
        # print("asset_proportions", asset_proportions)
        asset_values_w_sum_to_export = asset_values_w_sum.rename(rename_dict)
        asset_proportions = asset_proportions.select(pl.col('date'), pl.col('*').exclude('date').round(2))\
                                             .rename(rename_dict)
        
        cumulative_quantities = cumulative_quantities.rename(rename_dict)

        return asset_proportions, asset_values_w_sum_to_export, cumulative_quantities

    def export_to_csv(self, df: pl.DataFrame, table_type: str) -> str:
        """
        Export DataFrame to CSV file.
        
        Args:
            df: DataFrame to export
            
        Returns:
            str: Path to saved CSV file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"Portfolio_{table_type}_{timestamp}.csv"
        output_path = self.get_analysis_path(self.owner_id)  # Pass owner_id
        full_path = os.path.join(output_path, filename)
        
        df.write_csv(full_path)
        print(f"Exported portfolio data to {full_path}")
        
        return output_path

    # def export_to_ods(self, df: pl.DataFrame) -> str:
    #     """
    #     Export DataFrame to .ods file.
        
    #     Args:
    #         df: DataFrame to export
            
    #     Returns:
    #         Path to saved file
    #     """
    #     # print("[EXPORT_TO_ODS] Exporting DataFrame to .ods file...")
        
    #     # Initialize the ODS document and add a table
    #     doc = OpenDocumentSpreadsheet()
    #     table = Table(name="Portfolio Proportions")
    #     doc.spreadsheet.addElement(table)
        
    #     # Add header row
    #     header_row = TableRow()
        
    #     # Date header cell
    #     date_header = TableCell(valuetype="string")
    #     date_header.addElement(P(text="Date"))
    #     header_row.addElement(date_header)
        
    #     # Asset headers
    #     for col in df.columns:
    #         header_cell = TableCell(valuetype="string")
    #         header_cell.addElement(P(text=str(col)))
    #         header_row.addElement(header_cell)
        
    #     table.addElement(header_row)
    #     # print("[EXPORT_TO_ODS] Header row added.")
        
    #     # Add data rows

    #     for row in df.iter_rows(named=True):
    #         print("row", row)
    #         date = row['date']
    #         tr = TableRow()
            
    #         # # Add Date Cell
    #         # date_cell = TableCell(valuetype="string")
    #         # if isinstance(date, pl.Datetime):
    #         #     date_str = date.strftime("%Y-%m-%d")
    #         # else:
    #         #     date_str = str(date)
    #         # date_cell.addElement(P(text=date_str))
    #         # tr.addElement(date_cell)
            
    #         # Add Portfolio Proportions Cells
    #         for value in row.values():
    #             print("value", type(value))
    #             # Add Date Cell
    #             if type(value) is datetime.date:
    #                 date_cell = TableCell(valuetype="string")
    #                 date_str = value.strftime("%Y-%m-%d")
    #                 date_str = str(value)
    #                 date_cell.addElement(P(text=date_str))
    #                 tr.addElement(date_cell)
    #             else:
    #                 # Round to two decimal places
    #                 numeric_value = round(float(value), 2)
    #                 # print(f"[EXPORT_TO_ODS INFO] Writing value {numeric_value} for date {date_str}.")
                
    #             # Create a cell with numerical value
    #             cell = TableCell(valuetype="float", value=str(numeric_value))
                
    #             # Note: No need to add Number or P elements for numerical cells
    #             tr.addElement(cell)
            
    #         table.addElement(tr)
        
    #     # print("[EXPORT_TO_ODS] Data rows added.")
        
    #     # Define filename and path with owner_id
    #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #     filename = f"portfolio_proportions_{timestamp}.ods"
    #     output_path = self.get_analysis_path(self.owner_id)  # Pass owner_id
    #     full_path = os.path.join(output_path, filename)
        
    #     # Save the document
    #     try:
    #         doc.save(full_path)
    #         # print(f"[EXPORT_TO_ODS] File saved to: {full_path}")
    #         return full_path
    #     except Exception as e:
    #         # print(f"[EXPORT_TO_ODS ERROR] Failed to export to .ods: {e}")
    #         raise ValueError(f"Failed to export to .ods: {e}")

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
                print(self.fetch_market_data(), "self.fetch_market_data()")
            except ValueError as e:
                if "No valid tickers found" in str(e):
                    return None
                raise e

            # Calculate positions and proportions
            merged_df_cumsum = self.calculate_monthly_positions()
            proportions, portfolio_values, cumulative_quantities = self.calculate_portfolio_proportions(merged_df_cumsum)

            # Export results
            if not proportions.is_empty():
                self.export_to_csv(proportions, "Proportions")
                self.export_to_csv(portfolio_values, "Portfolio Values")
                self.export_to_csv(cumulative_quantities, "Cumulative_Quantities")
                return proportions, portfolio_values
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