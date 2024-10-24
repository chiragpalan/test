import yfinance as yf
import pandas as pd
import sqlite3
from datetime import datetime

# Database path (in the same repository)
DATABASE_PATH = "stock_data.db"

# Connect to the SQLite database
conn = sqlite3.connect(DATABASE_PATH)

# Table names
TABLE_RELIANCE = "reliance_data"
TABLE_TCS = "tcs_data"
TABLE_NIFTY = "nifty_data"
TABLE_ASIAN = "asian_data"

def fetch_and_store_data(ticker, table_name):
    # Get data for the last 3 months
    data = yf.download(ticker, period="5y")
    data.reset_index(inplace=True)  # Convert index to 'Date' column

    # Append data to the respective table
    data.to_sql(table_name, conn, if_exists="append", index=False)
    print(f"Data for {ticker} stored successfully in {table_name}.")

if __name__ == "__main__":
    fetch_and_store_data("RELIANCE.NS", TABLE_RELIANCE)
    fetch_and_store_data("TCS.NS", TABLE_TCS)
    fetch_and_store_data('^NSEI', TABLE_NIFTY)
    fetch_and_store_data("ASIANPAINT.NS", TABLE_ASIAN)
    


    # Close the database connection
    conn.close()
