import os

import psycopg
from dotenv import load_dotenv
import polars as pl

# Load environment variables from .env file
load_dotenv()
# Get connection string
conn_string = os.getenv("DATABASE_URL")


try:
    with psycopg.connect(conn_string) as conn:
        print("Connection established")
        with conn.cursor() as cur:
            # Fetch all rows from the books table
            cur.execute("SELECT * FROM asset_management_test.asset_transactions;")
            data = cur.fetchall()
            columns = [desc[0] for desc in cur.description]
            df = pl.DataFrame(data, schema=columns, orient="row")
            with pl.Config(tbl_cols=-1):
                print(df)

except Exception as e:
    print("Connection failed.")
    print(e)