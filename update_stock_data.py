import yfinance as yf
import pandas as pd
import sqlite3
from datetime import datetime, timedelta

# Database path (update to your desired folder)
DATABASE_PATH = "update_project_plan/stock_data.db"

# Connect to the SQLite database
conn = sqlite3.connect(DATABASE_PATH)

# Table names
TABLE_RELIANCE = "reliance_data"
TABLE_TCS = "tcs_data"
TABLE_NIFTY = "nifty_data"
TABLE_ASIAN = "asian_data"

def initialize_table(table_name):
    """Create the table if it doesn't exist."""
    create_table_query = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        Date TEXT PRIMARY KEY,
        Open REAL,
        High REAL,
        Low REAL,
        Close REAL,
        Adj_Close REAL,
        Volume INTEGER
    );
    """
    conn.execute(create_table_query)
    conn.commit()

def fetch_and_store_data(ticker, table_name):
    """Fetch new data and store it in the database."""
    # Ensure the table exists
    initialize_table(table_name)

    # Query the latest date in the database table
    query = f"SELECT MAX(Date) FROM {table_name}"
    last_date = pd.read_sql(query, conn).iloc[0, 0]

    # If there's no data, download 5 years of historical data
    if last_date is None:
        start_date = "2019-01-01"
    else:
        # Start from the next day after the last recorded date
        start_date = (datetime.strptime(last_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")

    # Download the data from Yahoo Finance
    data = yf.download(ticker, start=start_date)
    if data.empty:
        print(f"No new data available for {ticker}.")
        return

    # Reset the index to have 'Date' as a column
    data.reset_index(inplace=True)

    # Rename 'Adj Close' to 'Adj_Close' to match the schema
    data = data.rename(columns={"Adj Close": "Adj_Close"})
    print(data.columns)

    # Append the new data to the database
    data.to_sql(table_name, conn, if_exists="append", index=False)
    print(f"New data for {ticker} stored successfully in {table_name}.")

if __name__ == "__main__":
    fetch_and_store_data("RELIANCE.NS", TABLE_RELIANCE)
    fetch_and_store_data("TCS.NS", TABLE_TCS)
    fetch_and_store_data("^NSEI", TABLE_NIFTY)
    fetch_and_store_data("ASIANPAINT.NS", TABLE_ASIAN)

    # Close the database connection
    conn.close()
