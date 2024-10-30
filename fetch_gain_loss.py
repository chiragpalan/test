
import requests
from bs4 import BeautifulSoup
import sqlite3
from datetime import datetime

# URLs for top gainers and losers
gainers_url = "https://www.nseindia.com/live_market/dynaContent/live_analysis/gainers/niftyGainers1.json"
losers_url = "https://www.nseindia.com/live_market/dynaContent/live_analysis/losers/niftyLosers1.json"

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
}

def fetch_data(url):
    response = requests.get(url, headers=headers)
    return response.json()

# Fetch top gainers and losers
top_gainers = fetch_data(gainers_url)[:20]
top_losers = fetch_data(losers_url)[:20]

# Today's date
today = datetime.now().strftime('%Y%m%d')

# SQLite connection
conn = sqlite3.connect('gain_loos.db')
cursor = conn.cursor()

# Create table
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

def insert_data(data):
    for stock in data:
        cursor.execute(f"""
            INSERT INTO {table_name} (symbol, ltp, net_price, traded_qty, change)
            VALUES (?, ?, ?, ?, ?)
        """, (
            stock['symbol'], stock['ltp'], stock['netPrice'], 
            stock['tradedQuantity'], stock['change']
        ))

insert_data(top_gainers)
insert_data(top_losers)

conn.commit()
conn.close()
print(f"Data saved to {table_name}")
