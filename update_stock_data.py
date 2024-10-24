import os
import yfinance as yf
import pandas as pd
import sqlite3

# Ensure the folder exists
# DATA_FOLDER = os.path.join(os.getcwd(), "update_project_plan")
# os.makedirs(DATA_FOLDER, exist_ok=True)

# Database path
DATABASE_PATH = "stock_data_v1.db"

# Connect to the SQLite database
conn = sqlite3.connect(DATABASE_PATH)

# Table names
TABLE_RELIANCE = "reliance_data"
TABLE_TCS = "tcs_data"
TABLE_NIFTY = "nifty_data"
TABLE_ASIAN = "asian_data"

def initialize_table(table_name):
    # Drop and recreate the table to avoid schema conflicts
    conn.execute(f"DROP TABLE IF EXISTS {table_name}")

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
    initialize_table(table_name)  # Ensure the table schema is correct

    query = f"SELECT MAX(Date) FROM {table_name}"
    result = pd.read_sql(query, conn).iloc[0, 0]

    start_date = (pd.to_datetime(result) + pd.Timedelta(days=1)).strftime('%Y-%m-%d') if result else '2019-01-01'
    data = yf.download(ticker, start=start_date)

    if not data.empty:
        data.reset_index(inplace=True)
        data = data.rename(columns={'Adj Close': 'Adj_Close'})
        data = data.dropna(subset=['Date'])

        print(f"Inserting data for {ticker} into {table_name}.")
        data.to_sql(table_name, conn, if_exists="append", index=False)
    else:
        print(f"No new data available for {ticker}.")

if __name__ == "__main__":
    fetch_and_store_data("RELIANCE.NS", TABLE_RELIANCE)
    fetch_and_store_data("TCS.NS", TABLE_TCS)
    fetch_and_store_data("^NSEI", TABLE_NIFTY)
    fetch_and_store_data("ASIANPAINT.NS", TABLE_ASIAN)
    conn.close()
