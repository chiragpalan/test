import yfinance as yf
import pandas as pd
import sqlite3
import os

# Ensure the database path is in the current directory of the GitHub runner
DATABASE_PATH = os.path.join(os.getcwd(), "stock_data.db")

# Connect to the SQLite database
conn = sqlite3.connect(DATABASE_PATH)

# Table names
TABLE_RELIANCE = "reliance_data"
TABLE_TCS = "tcs_data"
TABLE_NIFTY = "nifty_data"
TABLE_ASIAN = "asian_data"

def fetch_and_store_data(ticker, table_name):
    # Check the most recent date in the table
    query = f"SELECT MAX(Date) FROM {table_name}"
    latest_date = pd.read_sql(query, conn).iloc[0, 0]

    # Fetch data from the last saved date or the last 5 years if the table is empty
    start_date = (pd.to_datetime(latest_date) + pd.Timedelta(days=1)).strftime('%Y-%m-%d') if latest_date else '2019-01-01'
    data = yf.download(ticker, start=start_date)

    if not data.empty:
        data.reset_index(inplace=True)  # Convert index to 'Date' column
        data.to_sql(table_name, conn, if_exists="append", index=False)
        print(f"Data for {ticker} stored successfully in {table_name}.")
    else:
        print(f"No new data for {ticker}.")

if __name__ == "__main__":
    fetch_and_store_data("RELIANCE.NS", TABLE_RELIANCE)
    fetch_and_store_data("TCS.NS", TABLE_TCS)
    fetch_and_store_data('^NSEI', TABLE_NIFTY)
    fetch_and_store_data("ASIANPAINT.NS", TABLE_ASIAN)

    # Close the database connection
    conn.close()
