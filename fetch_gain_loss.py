from nsetools import Nse
import sqlite3
from datetime import datetime

# Initialize NSE object
nse = Nse()

# Fetch data
top_gainers = nse.get_top_gainers()[:20]
top_losers = nse.get_top_losers()[:20]

# Get today's date in 'YYYYMMDD' format
today = datetime.now().strftime('%Y%m%d')

# Database connection
conn = sqlite3.connect('gain_loos.db')
cursor = conn.cursor()

# Create a table named with today's date followed by '_top_gain_loss'
table_name = f"{today}_top_gain_loss"
cursor.execute(f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        symbol TEXT,
        ltp REAL,
        net_price REAL,
        traded_qty INTEGER,
        change REAL
    )
""")

# Helper function to insert data into the table
def insert_data(data):
    for stock in data:
        cursor.execute(f"""
            INSERT INTO {table_name} (symbol, ltp, net_price, traded_qty, change)
            VALUES (?, ?, ?, ?, ?)
        """, (stock['symbol'], stock['ltp'], stock['netPrice'], stock['tradedQuantity'], stock['change']))

# Insert gainers and losers into the table
insert_data(top_gainers)
insert_data(top_losers)

# Commit changes and close the connection
conn.commit()
conn.close()

print(f"Data saved to table {table_name} in gain_loos.db")
